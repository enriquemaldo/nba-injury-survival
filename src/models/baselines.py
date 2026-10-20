"""Baselines 1-3 from the proposal: logistic regression, XGBoost, Cox PH.

Trained on the training seasons, evaluated on the VALIDATION season (2022-23).
The test season (2023-24) stays sealed until model selection is finished
(proposal section 6); pass --test only at the very end of the project.

    Logistic / XGBoost: classify "target injury within 14 days" (CLASSIF_DAYS),
        trained on landmarks whose 14-day status is known.
    Cox PH: time-to-event with censoring, all training landmarks, clustered SEs
        by player. Adjusted for the early reporting era with an indicator
        (main), and refit stratified by season (coefficient check).

Metrics (validation): C-index (primary), AUC and Brier score at 14 days,
calibration, Kaplan-Meier by predicted-risk tertile, 95% CIs by bootstrapping
players. Sensitivity runs (config.SENSITIVITY_RUNS) repeat everything with a
different set of training seasons.

All preprocessing (imputation medians, clipping bounds, scaling) is learned on
the training rows only. Hyperparameters are fixed in advance (no tuning).

Usage:  python -m src.models.baselines            (validation only)
        python -m src.models.baselines --test     (final evaluation - only at the end!)
Writes: data/processed/models/  (report, figures, validation predictions)
"""
import argparse
import time as _time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.config import CLASSIF_DAYS, PROCESSED, SENSITIVITY_RUNS
from src.models.coxph import CoxPH
from src.features.workload import HISTORY_FEATURES, WORKLOAD_FEATURES
from src.survival.dataset import SCHEDULE_FEATURES
from src.models.metrics import (auc_at, brier_at, c_index, calibration_table, cluster_bootstrap,
                                horizon_label, kaplan_meier, logrank)
from src.survival.dataset import LANDMARKS, model_features

OUT = PROCESSED / "models"
OUT.mkdir(parents=True, exist_ok=True)
H = CLASSIF_DAYS
COLORS = {"Logistic": "#2a78d6", "GBM": "#eb6834", "Cox": "#1baf7a", "Cox (season-stratified)": "#4a3aa7"}
# Simple reference models: how much does the full feature set add beyond these?
BENCHMARKS = {"Cox: injury history only": HISTORY_FEATURES,
              "Cox: minutes only (EWMA)": ["ewma_min_long", "ewma_min_short"],
              "Cox: workload + schedule/travel": WORKLOAD_FEATURES + SCHEDULE_FEATURES}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})


# ------------------------------------------------------------ preprocessing --
class Preprocessor:
    """Median imputation + missing indicators + clipping + standardization, fit on training rows."""

    def fit(self, df, features):
        X = df[features].astype(float)
        self.features = features
        self.missing_cols = [c for c in features if X[c].isna().any()]
        self.median = X.median()
        self.lo, self.hi = X.quantile(0.005), X.quantile(0.995)
        Z = self._raw(df)
        self.mean, self.std = Z.mean(), Z.std()
        self.keep = self.std[self.std > 0].index.tolist()          # drop constant columns
        return self

    def _raw(self, df):
        X = df[self.features].astype(float)
        ind = X[self.missing_cols].isna().astype(float).add_suffix("_missing")
        X = X.clip(self.lo, self.hi, axis=1).fillna(self.median)
        return pd.concat([X, ind], axis=1)

    def transform(self, df):
        Z = self._raw(df)[self.keep]
        return ((Z - self.mean[self.keep]) / self.std[self.keep]).to_numpy(), self.keep


def make_gbm():
    try:
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=5, eval_metric="logloss",
                             n_jobs=-1, random_state=0), "XGBoost"
    except Exception:  # xgboost missing (or libomp missing on macOS)
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=300,
                                              l2_regularization=1.0, random_state=0), \
            "scikit-learn HistGradientBoosting (xgboost not installed)"


# ------------------------------------------------------------------- run ----
def fit_models(train, feats, stratified=False):
    pre = Preprocessor().fit(train, feats)
    Xtr, names = pre.transform(train)
    ev = train.evaluable14.to_numpy() == 1
    y = train.y14.to_numpy()

    models = {}
    t0 = _time.time()
    lr = LogisticRegression(C=1.0, max_iter=3000).fit(Xtr[ev], y[ev])
    models["Logistic"] = ("clf", lr, pre)
    gbm, gbm_name = make_gbm()
    gbm.fit(train.loc[ev, feats].astype(float), y[ev])
    models["GBM"] = ("gbm", gbm, None)
    cox = CoxPH(penalizer=1.0).fit(Xtr, train.time, train.event, cluster=train.PLAYER_ID)
    models["Cox"] = ("cox", cox, pre)
    if stratified:
        for bname, bfeats in BENCHMARKS.items():
            pre_b = Preprocessor().fit(train, bfeats)
            models[bname] = ("cox", CoxPH(penalizer=1.0).fit(pre_b.transform(train)[0], train.time,
                                                             train.event), pre_b)
        pre_s = Preprocessor().fit(train, [f for f in feats if f != "early_era"])
        Xs, _ = pre_s.transform(train)
        cox_s = CoxPH(penalizer=1.0).fit(Xs, train.time, train.event, strata=train.SEASON,
                                         cluster=train.PLAYER_ID)
        models["Cox (season-stratified)"] = ("cox_strat", cox_s, pre_s)
    print(f"  trained {len(models)} models in {_time.time() - t0:.0f}s (GBM: {gbm_name})")
    return models, names, gbm_name


def predict(models, df, feats):
    """Returns {model: (risk score for ranking, 14-day probability or None)}."""
    out = {}
    for name, (kind, m, pre) in models.items():
        if kind == "clf":
            p = m.predict_proba(pre.transform(df)[0])[:, 1]
            out[name] = (p, p)
        elif kind == "gbm":
            p = m.predict_proba(df[feats].astype(float))[:, 1]
            out[name] = (p, p)
        elif kind == "cox":
            X = pre.transform(df)[0]
            out[name] = (m.predict_log_hazard(X), m.predict_risk(X, H))
        else:  # stratified Cox: ranking only (an unseen season has no baseline)
            out[name] = (m.predict_log_hazard(pre.transform(df)[0]), None)
    return out


def evaluate(df, preds, boot=False):
    rows = []
    t, e, pid = df.time.to_numpy(), df.event.to_numpy(), df.PLAYER_ID.to_numpy()
    known, y = horizon_label(t, e, H)
    for name, (risk, prob) in preds.items():
        r = {"model": name, "c_index": c_index(t, e, risk),
             "auc_14d": auc_at(t, e, risk, H)}
        if prob is not None:
            r["brier_14d"] = brier_at(t, e, prob, H)
            r["mean_predicted_14d"] = float(np.mean(prob[known]))
        r["observed_14d"] = float(y[known].mean())
        if boot:
            r["c_index_95ci"] = cluster_bootstrap(lambda i: c_index(t[i], e[i], risk[i]), pid)
            r["auc_14d_95ci"] = cluster_bootstrap(lambda i: auc_at(t[i], e[i], risk[i], H), pid)
        rows.append(r)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ figures --
def fig_calibration(df, preds, path):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    t, e = df.time.to_numpy(), df.event.to_numpy()
    top = 0
    for name, (_, prob) in preds.items():
        if prob is None or name not in COLORS:
            continue
        c = calibration_table(t, e, prob, H)
        ax.plot(c.mean_predicted, c.observed, marker="o", ms=5, lw=2, color=COLORS[name], label=name)
        top = max(top, c.mean_predicted.max(), c.observed.max())
    ax.plot([0, top * 1.05], [0, top * 1.05], ls="--", lw=1, color=INK2, label="Perfect calibration")
    ax.set_xlabel(f"Mean predicted {H}-day risk (decile)")
    ax.set_ylabel(f"Observed {H}-day injury rate")
    ax.set_title(f"Calibration on validation season ({H}-day risk)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_km(df, risk, path):
    groups = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["Low", "Medium", "High"]).to_numpy()
    fig, ax = plt.subplots(figsize=(6, 4))
    cols = {"Low": "#2a78d6", "Medium": "#eda100", "High": "#e34948"}
    for g in ["Low", "Medium", "High"]:
        m = groups == g
        km = kaplan_meier(df.time.to_numpy()[m], df.event.to_numpy()[m])
        ax.step(km.time, 1 - km.survival, where="post", lw=2, color=cols[g], label=f"{g} predicted risk")
    chi2, dof, p = logrank(df.time, df.event, groups)
    ax.set_xlabel("Days after the game")
    ax.set_ylabel("Cumulative injury incidence")
    ax.set_title("Validation season: injuries by Cox risk tertile", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return chi2, dof, p


def fig_forest(hr, path, n=15):
    z = (hr.coef / hr.se).abs()
    d = hr.reindex(z.sort_values(ascending=False).index).head(n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.33 * len(d) + 1.2))
    y = np.arange(len(d))
    ax.errorbar(d.hr, y, xerr=[d.hr - d.hr_lo, d.hr_hi - d.hr], fmt="o", color="#1baf7a",
                ecolor=INK2, elinewidth=1, capsize=2, ms=5)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, d.feature)
    ax.set_xlabel("Hazard ratio per 1 SD (95% CI, clustered by player)")
    ax.set_title(f"Cox model: {n} strongest associations (by |z|)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------- report --
def _fmt(v):
    if isinstance(v, tuple):
        return f"{v[0]:.3f}–{v[1]:.3f}"
    if isinstance(v, float):
        return "" if np.isnan(v) else f"{v:.3f}"
    return str(v)


def _md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join(_fmt(v) for v in r) + " |")
    return "\n".join(lines)


def tie_fraction(df):
    """Largest share of a risk set injured on the same day (justifies Breslow ties)."""
    t, e = df.time.to_numpy(), df.event.to_numpy()
    return max(((t == u) & (e == 1)).sum() / (t >= u).sum() for u in np.unique(t[e == 1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="also evaluate the sealed test season (final only)")
    args = ap.parse_args()

    lm = pd.read_csv(LANDMARKS, parse_dates=["GAME_DATE"])
    feats = model_features(lm)
    val = lm[lm.split == "val"].reset_index(drop=True)
    results, md_extra = [], []

    for run, seasons in SENSITIVITY_RUNS.items():
        train = lm[lm.SEASON.isin(seasons)].reset_index(drop=True)
        f = [c for c in feats if not (c == "early_era" and train.early_era.nunique() < 2)]
        print(f"[{run}] train {len(train):,} landmarks, {int(train.event.sum()):,} event landmarks")
        models, names, gbm_name = fit_models(train, f, stratified=(run == "main"))
        preds = predict(models, val, f)
        res = evaluate(val, preds, boot=(run == "main"))
        res.insert(0, "run", run)
        res.insert(1, "train_seasons", f"{seasons[0]}..{seasons[-1]}" + (" (excl. 2019-21)" if run == "excl_2019_21" else ""))
        results.append(res)

        if run == "main":
            main_models, main_preds, main_names, main_train, main_feats = models, preds, names, train, f
            pd.DataFrame({"PLAYER_ID": val.PLAYER_ID, "GAME_ID": val.GAME_ID, "time": val.time,
                          "event": val.event, **{f"{k}_risk": v[0] for k, v in preds.items()}}
                         ).to_csv(OUT / "val_predictions.csv", index=False)

    res = pd.concat(results, ignore_index=True)
    res.to_csv(OUT / "baseline_results.csv", index=False)

    # Main-run details: Cox hazard ratios, calibration, KM
    cox = main_models["Cox"][1]
    hr = pd.DataFrame(cox.summary(main_names))
    cox_s = main_models["Cox (season-stratified)"][1]
    names_s = main_models["Cox (season-stratified)"][2].keep
    hr_s = pd.DataFrame(cox_s.summary(names_s)).set_index("feature")
    hr["hr_stratified"] = hr.feature.map(hr_s.hr)
    hr.to_csv(OUT / "cox_hazard_ratios.csv", index=False)
    fig_calibration(val, main_preds, OUT / "fig_calibration.png")
    chi2, dof, p = fig_km(val, main_preds["Cox"][0], OUT / "fig_km_risk_tertiles.png")
    fig_forest(hr.set_index("feature", drop=False), OUT / "fig_cox_forest.png")
    cal = calibration_table(val.time, val.event, main_preds["Cox"][1], H)

    m = res[res.run == "main"]
    # Diagnostic: rank validation landmarks by each player's season-average Cox risk.
    # (Uses the whole season, so it is NOT a valid predictor - only a measure of how
    # much of the C-index comes from identifying which players get hurt.)
    avg = pd.Series(main_preds["Cox"][0]).groupby(val.PLAYER_ID.to_numpy()).transform("mean").to_numpy()
    c_avg = c_index(val.time, val.event, avg)
    get = lambda n: float(m.loc[m.model == n, "c_index"].iloc[0])
    top = hr.reindex((hr.coef / hr.se).abs().sort_values(ascending=False).index).head(15)
    md = ["# Baseline models (validation season 2022-23)\n",
          f"Training: {main_train.SEASON.min()}–{main_train.SEASON.max()} ({len(main_train):,} landmarks, "
          f"{int(main_train.event.sum()):,} event landmarks). Validation: {len(val):,} landmarks, "
          f"{int(val.event.sum()):,} event landmarks. **The test season (2023-24) was not used.**\n",
          f"Features: {len(main_feats)} ({len(main_names)} model columns after missing indicators). "
          f"Gradient boosting: {gbm_name}. Hyperparameters fixed in advance (no tuning).\n",
          "## Main results\n",
          "C-index is the primary metric (0.5 = random ranking). AUC and Brier are for injury within "
          f"{H} days, on landmarks whose {H}-day outcome is known. 95% CIs bootstrap players (200 resamples). "
          "The `Cox:` rows are simple reference models using one feature family each.\n",
          _md(m.drop(columns=["run", "train_seasons"])), "",
          "## What drives the ranking\n",
          f"- Injury history alone reaches C = {get('Cox: injury history only'):.3f}; the full Cox model "
          f"{get('Cox'):.3f}. Workload + schedule/travel without history: "
          f"{get('Cox: workload + schedule/travel'):.3f}; minutes alone: {get('Cox: minutes only (EWMA)'):.3f}.",
          f"- Diagnostic: ranking each player's games by his season-average Cox risk gives C = {c_avg:.3f}, "
          "higher than the game-by-game model. The C-index is therefore driven mainly by identifying "
          "*which* players get hurt (injury history, playing time), not *when*.",
          "- Whether recent workload predicts the *timing* of an injury within a player needs a "
          "within-player design (e.g., a model stratified by player), planned with the sequence model.", "",
          "## Sensitivity analyses (same validation season)\n",
          "`from_2018_19` drops the two seasons with heavy reason-less injured-list placements; "
          "`excl_2019_21` drops the bubble and shortened seasons (proposal section 6).\n",
          _md(res.loc[~res.model.str.startswith("Cox:"),
                  ["run", "train_seasons", "model", "c_index", "auc_14d", "brier_14d"]]), "",
          "## Cox model: strongest associations (hazard ratio per 1 SD, ranked by |z|)\n",
          "Robust 95% CIs clustered by player. Workload features are strongly correlated with each other, "
          "so individual hazard ratios among them should not be read as separate effects. "
          "`hr_stratified` = same feature in the season-stratified Cox model.\n",
          _md(top[["feature", "hr", "hr_lo", "hr_hi", "p", "hr_stratified"]]
              .assign(p=lambda d: d.p.map(lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"))), "",
          f"![Cox effects](fig_cox_forest.png)\n",
          "## Calibration (Cox, 14-day risk, deciles)\n", _md(cal), "",
          "![Calibration](fig_calibration.png)\n",
          "## Risk groups\n",
          f"Kaplan–Meier by tertile of Cox predicted risk (log-rank χ² = {chi2:.1f}, df {dof}). The log-rank "
          "test treats every landmark as independent, but a player's landmarks overlap, so its p-value "
          "overstates certainty and is not reported; the player-bootstrap CIs above are the reliable "
          "measure of uncertainty.\n",
          "![Kaplan-Meier](fig_km_risk_tertiles.png)\n",
          "## Notes\n",
          f"- Ties: at most {tie_fraction(main_train):.2%} of a risk set is injured on the same day, so "
          "Breslow's tie method is accurate here.",
          f"- Cox converged in {cox.n_iter_} Newton iterations.",
          "- Logistic/GBM are trained only on landmarks with a known 14-day outcome, so late-season "
          "landmarks (cut off by season end) are underrepresented in their training data.",
          "- Hazard ratios describe associations in this observational data, not causes."]

    if args.test:
        test = lm[lm.split == "test"].reset_index(drop=True)
        tres = evaluate(test, predict(main_models, test, main_feats), boot=True)
        tres.to_csv(OUT / "test_results.csv", index=False)
        md += ["", "## FINAL TEST SEASON (2023-24)\n", _md(tres)]
        print("TEST SEASON EVALUATED - do not change models after this.")

    (OUT / "baselines_report.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved to {OUT}")


if __name__ == "__main__":
    main()
