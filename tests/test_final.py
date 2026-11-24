"""Tests for the Phase 5 additions: temporal attribution, event metrics, sensitivity helpers.

Run:  python tests/test_final.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.explain.temporal import temporal_matrix
from src.final.event_metrics import alert_metrics, evaluate as event_evaluate
from src.final.sensitivity import ACUTE, CHRONIC, RATIOS, compare, cox_fit_predict
from src.features.workload import WORKLOAD_FEATURES
from src.models.metrics import c_index


def test_temporal_matrix_orders_games_most_recent_first():
    phi = np.zeros((10, 5, 2))
    phi[:, 4, 0] = 1.0             # newest game (last position), first input
    phi[:, 0, 1] = -2.0            # oldest game, second input
    t = temporal_matrix(phi, ["MIN", "gap_days"])
    assert len(t) == 10 and set(t.games_ago) == {1, 2, 3, 4, 5}
    assert t.query("input == 'MIN' and games_ago == 1").mean_abs.item() == 1.0
    assert t.query("input == 'gap_days' and games_ago == 5")["mean"].item() == -2.0
    assert t.query("input == 'MIN' and games_ago == 5").mean_abs.item() == 0.0


def test_alert_metrics_by_hand():
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0, 1])
    risk = np.array([.9, .8, .7, .1, .1, .1, .1, .1, .1, .05])
    p, r, f2 = alert_metrics(y, risk, 0.2)          # flags the top 2: one injury out of three
    assert np.isclose(p, 0.5) and np.isclose(r, 1 / 3)
    assert np.isclose(f2, 5 * p * r / (4 * p + r))
    p, r, _ = alert_metrics(y, risk, 0.3)            # top 3: two injuries
    assert np.isclose(p, 2 / 3) and np.isclose(r, 2 / 3)


def _landmarks(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame({"PLAYER_ID": rng.integers(0, 200, n), "min_7d": rng.gamma(4, 20, n),
                      "prior_any_injuries": rng.poisson(1.5, n).astype(float), "acwr_min": rng.normal(1, .2, n)})
    lin = 0.004 * d.min_7d + 0.45 * d.prior_any_injuries
    t = rng.exponential(1 / (0.01 * np.exp(lin - lin.mean())))
    d["time"], d["event"] = np.floor(np.minimum(t, 30)), (t <= 30).astype(int)
    return d


def test_event_metrics_on_synthetic_predictions():
    d = _landmarks()
    d["cox_risk"] = 0.45 * d.prior_any_injuries + 0.004 * d.min_7d
    d["tf5_risk"] = d.cox_risk + np.random.default_rng(1).normal(0, .3, len(d))
    tab, known, y = event_evaluate(d)
    assert set(tab.key) == {"cox", "tf5"}
    prev = y.mean()
    assert (tab.pr_auc > prev).all()                        # informative scores beat no-skill
    assert (tab.recall_top20 >= tab.recall_top10).all() and (tab.recall_top10 >= tab.recall_top5).all()
    assert len(known) <= len(d)


def test_sensitivity_helpers():
    d = _landmarks()
    tr, te = d.iloc[:2000], d.iloc[2000:].reset_index(drop=True)
    feats = ["min_7d", "prior_any_injuries", "acwr_min"]
    ref = cox_fit_predict(tr, te, feats)
    same = compare(te, ref, ref)
    assert same["delta"] == 0 and same["ci_lo"] == 0 and same["ci_hi"] == 0
    worse = compare(te, ref, cox_fit_predict(tr, te, ["acwr_min"]))
    assert worse["delta"] < 0 and np.isclose(worse["c_index"], c_index(te.time, te.event,
                                                                       cox_fit_predict(tr, te, ["acwr_min"])))
    groups = ACUTE + CHRONIC + RATIOS
    assert len(groups) == len(set(groups)) and set(groups) <= set(WORKLOAD_FEATURES)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
