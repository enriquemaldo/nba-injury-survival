"""Tests for the neural-model code. numpy parts always run; PyTorch parts are
skipped when torch is not installed.

Run:  python tests/test_neural.py
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.coxph import CoxPH
from src.models.neural import breslow_cumhaz, splits, static_columns


def _data(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    t_ev = rng.exponential(1 / (0.02 * np.exp(X @ np.array([0.6, -0.4, 0.0]))))
    time = np.floor(np.minimum(t_ev, 30.0))
    event = (t_ev <= 30).astype(int)
    return X, time, event


def _breslow_ll(eta, time, event):
    ll = 0.0
    for t in np.unique(time[event == 1]):
        d = (time == t) & (event == 1)
        ll += eta[d].sum() - d.sum() * np.log(np.exp(eta[time >= t]).sum())
    return ll


def test_breslow_cumhaz_matches_cox_baseline():
    X, time, event = _data()
    m = CoxPH().fit(X, time, event)
    for h in (0, 7, 14, 30):
        assert abs(breslow_cumhaz(X @ m.coef_, time, event, h) - m.cumulative_baseline_hazard(h)) < 1e-10


def test_splits_hold_out_whole_players():
    rng = np.random.default_rng(1)
    n = 5000
    data = {"season": rng.choice(["2018-19", "2022-23", "2023-24"], n, p=[.7, .2, .1]),
            "player_id": rng.integers(0, 300, n)}
    data["split"] = np.where(data["season"] == "2022-23", "val", np.where(data["season"] == "2023-24", "test", "train"))
    fit, inner, val = splits(data, seed=0)
    assert not set(data["player_id"][fit]) & set(data["player_id"][inner])      # no player in both
    assert set(data["split"][val]) == {"val"} and "test" not in set(data["split"][fit])
    assert 0.05 < len(inner) / (len(fit) + len(inner)) < 0.2


def test_static_columns_drop_family():
    data = {"static_family": np.array(["workload", "history", "workload", "bio"])}
    assert static_columns(data, ["workload"]).tolist() == [1, 3]
    assert static_columns(data, []).tolist() == [0, 1, 2, 3]


def _torch_or_skip():
    try:
        import torch
        return torch
    except ImportError:
        print("  (torch not installed - skipped)")
        return None


def test_cox_loss_matches_breslow_likelihood():
    torch = _torch_or_skip()
    if torch is None:
        return
    from src.models.neural import cox_loss
    X, time, event = _data(n=800)
    eta = X @ np.array([0.5, -0.3, 0.1])
    loss = cox_loss(torch.tensor(eta), torch.tensor(time), torch.tensor(event, dtype=torch.float64)).item()
    assert abs(loss - (-_breslow_ll(eta, time, event) / event.sum())) < 1e-8
    # Its minimizer is the Cox MLE: the gradient at the fitted coefficients is ~0.
    m = CoxPH().fit(X, time, event)
    beta = torch.tensor(m.coef_, requires_grad=True)
    cox_loss(torch.tensor(X) @ beta, torch.tensor(time), torch.tensor(event, dtype=torch.float64)).backward()
    assert beta.grad.abs().max().item() < 1e-6


def test_models_forward_with_padding():
    torch = _torch_or_skip()
    if torch is None:
        return
    from src.models.neural import Batcher, build_model
    rng = np.random.default_rng(0)
    n, n_games, n_seq = 50, 200, 5
    idx = rng.integers(0, n_games, (n, 20)).astype(np.int32)
    idx[0] = -1                     # no previous games at all
    idx[1, :17] = -1                # only 3 previous games
    data = {"static_X": rng.normal(size=(n, 7)).astype(np.float32), "time": rng.integers(0, 30, n).astype(np.float32),
            "event": rng.integers(0, 2, n).astype(np.int8), "game_X": rng.normal(size=(n_games, 13)).astype(np.float32),
            "seq_idx": idx, "seq_dt": rng.random((n, 20)).astype(np.float32)}
    b = Batcher(data, np.arange(7), n_seq, "cpu")
    x = b(np.arange(n))
    assert x["seq"].shape == (n, n_seq, 13) and x["pad"][0].all() and x["pad"][1].sum() == 2
    assert torch.equal(x["seq"][2, -1], torch.from_numpy(data["game_X"][idx[2, -1]]))
    for kind, ns in (("transformer", n_seq), ("deepsurv", 0)):
        m = build_model(kind, 7, 13, ns).eval()
        out = m(**(x if ns else {"static": x["static"]}))
        assert out.shape == (n,) and torch.isfinite(out).all()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
