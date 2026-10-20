"""Tests for the Cox model, survival metrics, outcome construction and leakage.

Run:  python tests/test_models.py   (or python -m pytest tests/)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.coxph import CoxPH
from src.models.metrics import (auc_at, c_index, c_index_bruteforce, horizon_label, kaplan_meier,
                                logrank)


def _sim(n=4000, beta=(0.7, -0.5, 0.0), seed=1, ties=False, strata_shift=False):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, len(beta)))
    s = rng.integers(0, 2, n) if strata_shift else np.zeros(n, int)
    rate = 0.05 * np.exp(X @ np.array(beta)) * np.where(s == 1, 3.0, 1.0)
    t_ev = rng.exponential(1 / rate)
    t_c = rng.uniform(0, 40, n)
    time = np.minimum(t_ev, t_c)
    event = (t_ev <= t_c).astype(int)
    if ties:
        time = np.floor(time)
    return X, time, event, s


def _breslow_ll(X, time, event, beta):
    eta = X @ beta
    ll = 0.0
    for t in np.unique(time[event == 1]):
        d = (time == t) & (event == 1)
        ll += eta[d].sum() - d.sum() * np.log(np.exp(eta[time >= t]).sum())
    return ll


def test_c_index_matches_bruteforce():
    rng = np.random.default_rng(0)
    for _ in range(5):
        n = 150
        time = rng.integers(0, 10, n).astype(float)
        event = rng.random(n) < 0.4
        risk = rng.integers(0, 6, n).astype(float)        # many ties on purpose
        assert abs(c_index(time, event, risk) - c_index_bruteforce(time, event, risk)) < 1e-12


def test_c_index_within_player():
    from src.models.metrics import c_index_within
    rng = np.random.default_rng(1)
    n = 120
    g = rng.integers(0, 6, n)
    time = rng.integers(0, 10, n).astype(float)
    event = rng.random(n) < 0.4
    risk = rng.normal(size=n)
    num = den = 0.0
    for i in range(n):
        for j in range(n):
            if g[i] == g[j] and event[i] and (time[i] < time[j] or (time[i] == time[j] and not event[j])):
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    assert abs(c_index_within(time, event, risk, g) - num / den) < 1e-12
    # A per-player constant carries no within-player information.
    assert abs(c_index_within(time, event, g.astype(float), g) - 0.5) < 1e-12


def test_cox_loglik_matches_direct_formula():
    X, time, event, _ = _sim(n=300, ties=True, seed=3)
    m = CoxPH().fit(X, time, event)
    assert abs(m.loglik_ - _breslow_ll(X, time, event, m.coef_)) < 1e-8
    # The fitted beta maximizes the likelihood: nudging it lowers the log-likelihood.
    for d in np.eye(3) * 1e-3:
        assert _breslow_ll(X, time, event, m.coef_ + d) < m.loglik_


def test_cox_recovers_true_effects():
    X, time, event, _ = _sim(n=6000, seed=5)
    m = CoxPH().fit(X, time, event, cluster=np.arange(len(X)))
    assert m.converged_
    true = np.array([0.7, -0.5, 0.0])
    assert np.all(np.abs(m.coef_ - true) < 4 * m.se_), (m.coef_, m.se_)
    # Without clustering structure, robust and model-based SEs agree closely.
    assert np.all(np.abs(m.robust_se_ / m.se_ - 1) < 0.15)


def test_cox_strata_and_score_residuals():
    X, time, event, s = _sim(n=6000, seed=7, strata_shift=True)
    m = CoxPH().fit(X, time, event, strata=s)
    assert np.all(np.abs(m.coef_ - np.array([0.7, -0.5, 0.0])) < 4 * m.se_)
    X, time, event, s = _sim(n=6000, seed=7, strata_shift=True, ties=True)
    m = CoxPH().fit(X, time, event, strata=s, cluster=np.arange(len(X)) // 3)
    # Score residuals sum to the (unpenalized) gradient, which is ~0 at the optimum.
    m._X, m._t, m._e = X, time.astype(float), event.astype(float)
    assert np.allclose(m.score_residuals().sum(axis=0), 0, atol=1e-6)
    # Stratum with 3x hazard gets a higher baseline.
    assert m.cumulative_baseline_hazard(10, 1) > 2 * m.cumulative_baseline_hazard(10, 0)


def test_cox_light_ties_unbiased():
    """Integer-day times with a small event fraction per day (like the real data,
    ~0.1% of each risk set per day) give the same answer as continuous times."""
    rng = np.random.default_rng(13)
    n = 20000
    X = rng.normal(size=(n, 2))
    t_ev = rng.exponential(1 / (0.002 * np.exp(X @ np.array([0.5, -0.3]))))
    time = np.minimum(t_ev, 30.0)
    event = (t_ev <= 30).astype(int)
    cont = CoxPH().fit(X, time, event).coef_
    tied = CoxPH().fit(X, np.floor(time), event).coef_
    assert np.allclose(cont, tied, atol=0.01), (cont, tied)


def test_cox_risk_null_model_matches_kaplan_meier():
    rng = np.random.default_rng(2)
    time = rng.integers(0, 31, 3000).astype(float)
    event = (rng.random(3000) < 0.2).astype(int)
    X = rng.normal(size=(3000, 1)) * 0                       # no information
    m = CoxPH(penalizer=1.0).fit(X + 0.0, time, event)
    km = kaplan_meier(time, event)
    s14 = km.loc[km.time <= 14, "survival"].iloc[-1]
    assert abs((1 - m.predict_risk(X[:1], 14)[0]) - s14) < 0.01


def test_logrank_sanity():
    X, time, event, s = _sim(n=3000, seed=9, strata_shift=True)
    assert logrank(time, event, s)[2] < 1e-6                 # different hazards
    rng = np.random.default_rng(9)
    assert logrank(time, event, rng.integers(0, 2, len(time)))[2] > 0.001   # random split


def test_horizon_label_and_auc():
    time = np.array([3, 20, 10, 14, 30])
    event = np.array([1, 1, 0, 0, 0])
    known, y = horizon_label(time, event, 14)
    assert known.tolist() == [True, True, False, True, True] and y.tolist() == [1, 0, 0, 0, 0]
    assert auc_at(time, event, np.array([0.9, 0.1, 0.5, 0.2, 0.3]), 14) == 1.0


def test_cox_matches_lifelines_if_installed():
    try:
        from lifelines import CoxPHFitter
    except ImportError:
        print("  (lifelines not installed - skipped)")
        return
    X, time, event, _ = _sim(n=800, seed=11)                 # continuous times: Breslow == Efron
    df = pd.DataFrame(X, columns=["a", "b", "c"]).assign(T=time, E=event)
    lf = CoxPHFitter().fit(df, "T", "E")
    m = CoxPH().fit(X, time, event)
    assert np.allclose(m.coef_, lf.params_.to_numpy(), atol=1e-4)
    assert np.allclose(m.se_, lf.standard_errors_.to_numpy(), rtol=1e-3)


def test_preprocessor_skips_rare_missing_indicators():
    from src.models.baselines import Preprocessor
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": rng.normal(size=1000), "b": rng.normal(size=1000), "c": rng.normal(size=1000)})
    df.loc[:4, "a"] = np.nan                     # 0.5% missing -> imputed only
    df.loc[:199, "b"] = np.nan                   # 20% missing -> imputed + indicator
    pre = Preprocessor().fit(df, ["a", "b", "c"])
    X, names = pre.transform(df)
    assert names == ["a", "b", "c", "b_missing"] and not np.isnan(X).any()


# ---------------------------------------------------------- dataset logic --
def test_outcomes_rules():
    from src.survival.dataset import outcomes
    d = pd.Timestamp
    lm = pd.DataFrame({"PLAYER_ID": [1, 1, 1, 2, 3],
                       "GAME_DATE": [d("2022-01-01"), d("2022-01-10"), d("2022-03-01"), d("2022-01-01"),
                                     d("2022-04-05")],
                       "season_end": [d("2022-04-10")] * 5})
    ep = pd.DataFrame({"PLAYER_ID": [1, 1, 2], "injury_date": [d("2022-01-12"), d("2022-01-12"), d("2022-01-05")]})
    target = pd.Series([False, True, False])                 # same-day target + other: target wins
    o = outcomes(lm, ep, target, label_end=d("2025-01-12"), horizon=30)
    assert o.event.tolist() == [1, 1, 0, 0, 0]
    assert o.time.tolist() == [11, 2, 30, 4, 6]              # last: season end (Apr 10) + 1 day
    assert o.reason.tolist() == ["target_event", "target_event", "horizon", "other_injury", "season_end"]


def test_injury_date_uses_last_game_within_lag():
    from src.survival.dataset import injury_dates
    ep = pd.DataFrame({"onset_date": ["2022-11-05", "2022-11-05", "2022-11-05"],
                       "last_game_before": ["2022-11-03", "2022-10-01", None]})
    assert injury_dates(ep).dt.strftime("%m-%d").tolist() == ["11-03", "11-05", "11-05"]


def test_workload_features_have_no_lookahead():
    from src.features.workload import WORKLOAD_FEATURES, workload_features
    rng = np.random.default_rng(4)
    dates = pd.date_range("2021-10-20", periods=60, freq="2D")
    gl = pd.DataFrame({"PLAYER_ID": 7, "GAME_ID": range(60), "GAME_DATE": dates, "SEASON": "2021-22",
                       "TEAM_ABBREVIATION": "BOS", "MIN": rng.integers(10, 40, 60),
                       "FGA": rng.integers(0, 20, 60), "FTA": rng.integers(0, 8, 60),
                       "TOV": rng.integers(0, 5, 60)})
    tg = pd.DataFrame({"TEAM": "BOS", "SEASON": "2021-22", "GAME_DATE": dates})
    base = workload_features(gl, tg).set_index("GAME_ID")
    k = 40
    changed = gl.copy()
    changed.loc[changed.GAME_ID >= k, ["MIN", "FGA", "FTA", "TOV"]] = 0   # game k and everything after
    alt = workload_features(changed, tg).set_index("GAME_ID")
    a = base.loc[:k, WORKLOAD_FEATURES].fillna(-1)
    b = alt.loc[:k, WORKLOAD_FEATURES].fillna(-1)
    assert (a == b).all().all(), "features at game k must not depend on game k or later"
    assert (base.loc[k + 1:, "min_7d"] != alt.loc[k + 1:, "min_7d"]).any()   # sanity: later games do


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
