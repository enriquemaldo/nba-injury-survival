"""Survival and fixed-horizon metrics (numpy/scipy/scikit-learn only).

c_index_within    Same, but only pairs within the same player (ranks timing, not who).
c_index           Harrell's concordance. A pair (i, j) is comparable when i has the
                  event and T_i < T_j, or T_i == T_j and j is censored (the event is
                  taken to precede censoring at the same time). Risk ties count 1/2.
auc_at / brier_at Fixed-horizon AUC and Brier score at h days on landmarks whose
                  status at h is known (event by h, or followed for >= h days).
calibration_table Predicted vs observed risk by decile of predicted risk.
kaplan_meier      Survival curve.  logrank  k-group log-rank test.
cluster_bootstrap Percentile CIs resampling players (landmarks of the same player
                  are not independent, proposal section 4A).
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score


def _c_counts(time, event, risk):
    """(concordant + 0.5*tied, comparable pairs) for Harrell's C."""
    conc = ties = total = 0.0
    for t in np.unique(time[event]):
        ev = risk[event & (time == t)]
        pool = np.sort(risk[(time > t) | ((time == t) & ~event)])
        if not len(pool):
            continue
        lo = np.searchsorted(pool, ev, side="left")       # pool values < r_i
        hi = np.searchsorted(pool, ev, side="right")      # pool values <= r_i
        conc += lo.sum()
        ties += (hi - lo).sum()
        total += len(ev) * len(pool)
    return conc + 0.5 * ties, total


def c_index(time, event, risk):
    time, event, risk = np.asarray(time, float), np.asarray(event, bool), np.asarray(risk, float)
    num, den = _c_counts(time, event, risk)
    return num / den if den else np.nan


def c_index_within(time, event, risk, group):
    """C-index using only pairs from the SAME player: does the model rank a player's
    own higher-risk games ahead of his lower-risk games? (timing, not who)."""
    time, event, risk = np.asarray(time, float), np.asarray(event, bool), np.asarray(risk, float)
    group = np.asarray(group)
    num = den = 0.0
    order = np.argsort(group, kind="stable")
    bounds = np.flatnonzero(np.diff(group[order])) + 1
    for idx in np.split(order, bounds):
        if event[idx].any():
            n, d = _c_counts(time[idx], event[idx], risk[idx])
            num, den = num + n, den + d
    return num / den if den else np.nan


def c_index_bruteforce(time, event, risk):
    num = den = 0.0
    n = len(time)
    for i in range(n):
        if not event[i]:
            continue
        for j in range(n):
            if time[i] < time[j] or (time[i] == time[j] and not event[j]):
                den += 1
                num += 1.0 if risk[i] > risk[j] else 0.5 if risk[i] == risk[j] else 0.0
    return num / den


def horizon_label(time, event, h):
    """(known, y): status at day h is known, and whether the event happened by h."""
    time, event = np.asarray(time), np.asarray(event, bool)
    y = event & (time <= h)
    return y | (time >= h), y.astype(int)


def auc_at(time, event, prob, h):
    known, y = horizon_label(time, event, h)
    return roc_auc_score(y[known], np.asarray(prob)[known])


def brier_at(time, event, prob, h):
    known, y = horizon_label(time, event, h)
    p = np.asarray(prob)[known]
    return float(np.mean((p - y[known]) ** 2))


def calibration_table(time, event, prob, h, bins=10):
    known, y = horizon_label(time, event, h)
    d = pd.DataFrame({"p": np.asarray(prob)[known], "y": y[known]})
    d["bin"] = pd.qcut(d.p.rank(method="first"), bins, labels=False) + 1
    return (d.groupby("bin").agg(n=("y", "size"), mean_predicted=("p", "mean"), observed=("y", "mean"))
            .reset_index())


def kaplan_meier(time, event):
    time, event = np.asarray(time, float), np.asarray(event, bool)
    ut = np.unique(time)
    at_risk = np.array([(time >= t).sum() for t in ut])
    d = np.array([(event & (time == t)).sum() for t in ut])
    s = np.cumprod(1 - d / at_risk)
    return pd.DataFrame({"time": ut, "at_risk": at_risk, "events": d, "survival": s})


def logrank(time, event, group):
    """k-group log-rank test. Returns (chi2, df, p)."""
    time, event, group = np.asarray(time, float), np.asarray(event, bool), np.asarray(group)
    gs = np.unique(group)
    k = len(gs)
    O_E = np.zeros(k)
    V = np.zeros((k, k))
    for t in np.unique(time[event]):
        r = np.array([((time >= t) & (group == g)).sum() for g in gs], float)
        dg = np.array([(event & (time == t) & (group == g)).sum() for g in gs], float)
        n, d = r.sum(), dg.sum()
        if n <= 1:
            continue
        O_E += dg - d * r / n
        c = d * (n - d) / (n - 1) / n
        V += c * (np.diag(r) - np.outer(r, r) / n)
    chi2 = float(O_E[:-1] @ np.linalg.solve(V[:-1, :-1], O_E[:-1]))
    return chi2, k - 1, float(stats.chi2.sf(chi2, k - 1))


def cluster_bootstrap(fn, cluster, n_boot=200, seed=0):
    """Percentile 95% CI of fn(idx) over bootstrap samples of clusters (players)."""
    rng = np.random.default_rng(seed)
    cluster = np.asarray(cluster)
    uniq, inv = np.unique(cluster, return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        idx = np.concatenate([rows_of[i] for i in pick])
        vals.append(fn(idx))
    vals = np.asarray(vals, float)
    vals = vals[~np.isnan(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
