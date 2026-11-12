"""Tests for the final evaluation and SHAP code. numpy parts always run; PyTorch
parts are skipped when torch is not installed.

Run:  python tests/test_explain.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.dashboard.prepare import drivers, value_text
from src.explain.shap_values import cox_shap, family_table, tf_tables
from src.final.evaluate import boot_values, seed_ensemble
from src.models.baselines import Preprocessor
from src.models.coxph import CoxPH
from src.models.metrics import c_index


def _df(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"min_7d": rng.gamma(4, 20, n), "prior_any_injuries": rng.poisson(1.5, n).astype(float),
                      "travel_km": rng.exponential(800, n), "age": rng.normal(27, 4, n)})
    d.loc[rng.random(n) < 0.05, "age"] = np.nan
    lin = 0.004 * d.min_7d + 0.4 * d.prior_any_injuries
    t_ev = rng.exponential(1 / (0.01 * np.exp(lin - lin.mean())))
    d["time"], d["event"] = np.floor(np.minimum(t_ev, 30)), (t_ev <= 30).astype(int)
    return d


def test_cox_shap_adds_up_to_log_hazard():
    d = _df()
    feats = ["min_7d", "prior_any_injuries", "travel_km", "age"]
    fit, score = d.iloc[:1500], d.iloc[1500:]
    pre = Preprocessor().fit(fit, feats)
    m = CoxPH(penalizer=1.0).fit(pre.transform(fit)[0], fit.time, fit.event)
    phi, base, eta, names = cox_shap({"pre": pre, "coef": m.coef_}, fit, score)
    assert np.allclose(phi.sum(1) + base, m.predict_log_hazard(pre.transform(score)[0]))
    assert np.allclose(eta, m.predict_log_hazard(pre.transform(score)[0]))
    assert phi.shape == (len(score), len(names))


def test_family_attributions_sum_to_total():
    rng = np.random.default_rng(0)
    static_names = ["min_7d", "prior_any_injuries", "travel_km", "age", "days_rest_missing"]
    seq_names = ["MIN", "back_to_back", "travel_km", "days_before"]
    ps, pq = rng.normal(size=(100, 5)), rng.normal(size=(100, 5, 4))
    feat, fam = tf_tables(ps, pq, static_names, seq_names, 5)
    total = ps.sum(1) + pq.sum((1, 2))
    assert np.allclose(sum(fam.values()), total)
    assert {"workload", "workload_seq", "history", "travel", "bio", "schedule"} <= set(fam)
    t = family_table(fam)
    assert abs(t.share.sum() - 1) < 1e-12 and len(feat) == 9


def test_seed_ensemble_and_bootstrap():
    rng = np.random.default_rng(0)
    etas = [rng.normal(size=500) * s + 3 * s for s in (1, 2, 5)]
    risk, prob = seed_ensemble(etas, [np.full(500, p) for p in (0.1, 0.2, 0.3)])
    assert np.allclose(risk, np.mean([(e - e.mean()) / e.std() for e in etas], 0)) and np.allclose(prob, 0.2)
    pid = rng.integers(0, 50, 500)
    v = boot_values(lambda i: len(np.unique(pid[i])) / 50, pid, 50)
    assert len(v) == 50 and (v <= 1).all() and v.mean() < 0.75        # resampling repeats players
    d = _df(800)
    r = d.prior_any_injuries.to_numpy()
    vals = boot_values(lambda i: c_index(d.time.to_numpy()[i], d.event.to_numpy()[i], r[i]), np.arange(800), 30)
    assert abs(np.median(vals) - c_index(d.time, d.event, r)) < 0.03


def test_dashboard_driver_table():
    assert value_text("back_to_back", 1.0) == "yes" and value_text("is_home", 0.0) == "no"
    assert value_text("ewma_min_short", 31.26) == "31.3 min/game" and value_text("age_missing", 1) == ""
    assert value_text("prior_any_injuries", 3.0) == "3" and value_text("age", np.nan) == "unknown"
    rng = np.random.default_rng(0)
    shap = {"tf_static_names": np.array(["ewma_min_short", "prior_any_injuries", "back_to_back"]),
            "tf_seq_names": np.array(["MIN", "travel_km", "days_before"]),
            "tf_static": rng.normal(size=(4, 3)), "tf_seq": rng.normal(size=(4, 5, 3))}
    score = pd.DataFrame({"ewma_min_short": [30.0, 12.0, 25.0, 33.0], "prior_any_injuries": [0.0, 2.0, 5.0, 1.0],
                          "back_to_back": [1.0, 0.0, 0.0, 1.0]})
    prev = {"MIN": np.array([31.0, np.nan, 20.0, 35.5]), "travel_km": np.array([900.0, 0, 1500, 300]),
            "_n": np.array([5, 0, 5, 5])}
    d = drivers(shap, score, prev, 5, n=4)
    assert len(d) == 16 and d.groupby("row")["rank"].apply(list).eq([[1, 2, 3, 4]] * 4).all()
    phi = np.c_[shap["tf_static"], shap["tf_seq"].sum(1)]
    for i in range(4):                        # ranked by |SHAP|, values match
        assert np.allclose(d[d.row == i].shap.abs().to_numpy(), np.sort(np.abs(phi[i]))[::-1][:4])
    m = d[d.label == "Last 5 games: minutes"]
    assert set(m.value) <= {"31.0 min/game", "", "20.0 min/game", "35.5 min/game"}
    assert (m[m.row == 1].value == "").all()            # player without previous games


def _torch_or_skip():
    try:
        import torch
        return torch
    except ImportError:
        print("  (torch not installed - skipped)")
        return None


def _inputs(torch, n=40, n_seq=5, n_step=6, n_static=7, seed=0):
    g = torch.Generator().manual_seed(seed)
    pad = torch.zeros(n, n_seq, dtype=torch.bool)
    pad[0] = True                    # no previous games
    pad[1, :3] = True                # 2 previous games
    seq = torch.randn(n, n_seq, n_step, generator=g) * (~pad).unsqueeze(-1)
    dt = torch.rand(n, n_seq, generator=g) * (~pad)
    return {"static": torch.randn(n, n_static, generator=g), "seq": seq, "pad": pad, "dt": dt}


def test_expected_gradients_exact_for_linear_model():
    torch = _torch_or_skip()
    if torch is None:
        return
    from src.explain.shap_values import expected_gradients

    class Lin(torch.nn.Module):
        def __init__(self):
            super().__init__()
            g = torch.Generator().manual_seed(1)
            self.ws, self.wq, self.wd = (torch.randn(7, generator=g), torch.randn(6, generator=g),
                                         torch.randn(1, generator=g))

        def forward(self, static, seq, pad, dt):
            keep = (~pad).float()
            return static @ self.ws + ((seq @ self.wq) * keep).sum(1) + (dt * keep).sum(1) * self.wd

    x = _inputs(torch)
    bg = {k: v[5:6] for k, v in _inputs(torch, seed=1).items()}       # one background row
    phi = expected_gradients(Lin(), {k: x[k] for k in ("static", "seq", "dt")}, bg, x["pad"], 3,
                             torch.Generator().manual_seed(0))
    m = Lin()
    assert torch.allclose(phi["static"], (x["static"] - bg["static"]) * m.ws, atol=1e-5)
    assert torch.allclose(phi["seq"], (x["seq"] - bg["seq"]) * m.wq * (~x["pad"]).unsqueeze(-1), atol=1e-5)
    assert phi["seq"][0].abs().max() == 0 and phi["dt"][1, :3].abs().max() == 0       # padding gets nothing


def test_expected_gradients_add_up_for_transformer():
    torch = _torch_or_skip()
    if torch is None:
        return
    from src.explain.shap_values import expected_gradients
    from src.models.neural import build_model
    torch.manual_seed(0)
    model = build_model("transformer", 7, 6, 5).eval()
    x = _inputs(torch, n=30)
    bg = {k: v[3:4] for k, v in _inputs(torch, seed=1).items()}
    phi = expected_gradients(model, {k: x[k] for k in ("static", "seq", "dt")}, bg, x["pad"], 1000,
                             torch.Generator().manual_seed(0))
    total = phi["static"].sum(1) + phi["seq"].sum((1, 2)) + phi["dt"].sum(1)
    with torch.no_grad():
        f_x = model(**x)
        f_bg = model(static=bg["static"].expand(30, -1), seq=bg["seq"].expand(30, -1, -1), pad=x["pad"],
                     dt=bg["dt"].expand(30, -1))
    err = (total - (f_x - f_bg)).abs()
    assert (err <= 0.10 * (f_x - f_bg).abs() + 0.02).all() and err.mean() < 0.02, err.max()
    assert phi["seq"][0].abs().max() == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
