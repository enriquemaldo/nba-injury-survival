"""Final, one-time evaluation on the sealed test season (proposal section 6).

Fixed in advance (Update 3, before any test data were used):
    PRIMARY comparison: Cox vs. Transformer + DeepSurv (5 games), C-index on the
    2023-24 test season, difference with a paired 95% CI (bootstrap over players).

How the final models are built (also fixed in advance):
    - every model is refit on the training + validation seasons (2016-17 .. 2022-23)
      with exactly the settings used in Updates 2-3 (no tuning);
    - neural models: 3 seeds each, early stopping on 10% of those players held out,
      seed ensemble = average standardized log-risk (ranking) and average 14-day
      probability (calibration), as in Update 3.

Secondary (descriptive, not used to pick anything): logistic regression, XGBoost,
DeepSurv and the 20-game Transformer; AUC and Brier score at 14 days, calibration;
and the supplementary 2024-25 season (games up to the end of the injury data,
2025-01-12), a second season further from the training data.

Run ONCE, at the end:     python -m src.final.evaluate
It refuses to run again once results exist (use --force only to recover from a crash
before results were looked at).
Writes: data/processed/final/ (report, predictions, figures, final models)
"""
import argparse
import json
import pickle
import time as _time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import CLASSIF_DAYS, PROCESSED, TRAIN_SEASONS, VAL_SEASONS
from src.models.baselines import _md, fit_models, predict
from src.models.metrics import auc_at, brier_at, c_index, calibration_table, horizon_label, kaplan_meier
from src.survival.dataset import LANDMARKS, model_features

OUT = PROCESSED / "final"
MODEL_DIR = OUT / "models"
LOCK = OUT / "test_results.csv"
H = CLASSIF_DAYS
FIT_SEASONS = TRAIN_SEASONS + VAL_SEASONS
SETS = {"test": "Test season 2023-24", "supplementary": "Supplementary 2024-25 (games to 2025-01-12)"}
NEURAL = {"deepsurv": "deepsurv", "tf20": "tf_full", "tf5": "tf_len5"}   # key -> run in src.models.neural
LABELS = {"cox": "Cox", "logistic": "Logistic regression", "xgboost": "XGBoost", "deepsurv": "DeepSurv",
          "tf20": "Transformer + DeepSurv (20 games)", "tf5": "Transformer + DeepSurv (5 games)"}
PRIMARY = ("cox", "tf5")
SEEDS = 3
N_BOOT, N_BOOT_PRIMARY = 200, 1000
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
COL = {"cox": "#8a8984", "logistic": "#b5b4ae", "xgboost": "#b5b4ae", "deepsurv": "#7fb0e8",
       "tf20": "#4f93df", "tf5": "#1f5fae"}
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})


# ------------------------------------------------------------------ helpers --
def boot_values(fn, cluster, n_boot, seed=0):
    """fn(idx) over bootstrap resamples of clusters (players); same scheme as metrics.cluster_bootstrap."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(np.asarray(cluster), return_inverse=True)
    rows_of = np.split(np.argsort(inv, kind="stable"), np.cumsum(np.bincount(inv))[:-1])
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[i] for i in rng.integers(0, len(uniq), len(uniq))])
        vals.append(fn(idx))
    vals = np.asarray(vals, float)
    return vals[~np.isnan(vals)]


def ci(v):
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def seed_ensemble(etas, probs):
    """Ranking: average of each seed's standardized log-risk. Probability: average 14-day risk."""
    z = [(e - e.mean()) / e.std() for e in etas]
    return np.mean(z, axis=0), np.mean(probs, axis=0)


# ------------------------------------------------------------------ models ---
def baseline_predictions(fit, score, feats):
    models, names, gbm_name = fit_models(fit, feats)
    p = predict(models, score, feats)
    out = {"cox": p["Cox"], "logistic": p["Logistic"], "xgboost": p["GBM"]}
    _, cox, pre = models["Cox"]
    with open(MODEL_DIR / "cox.pkl", "wb") as f:   # small: coefficients + preprocessing only
        pickle.dump({"coef": cox.coef_, "names": names, "pre": pre, "features": feats,
                     "H0_14": float(cox.cumulative_baseline_hazard(H)), "fit_seasons": FIT_SEASONS}, f)
    return out, gbm_name


def neural_predictions(n_rows, device):
    """Trains the final neural models in a separate process (PyTorch and XGBoost must not share
    one process on macOS) and loads their predictions for the held-out rows."""
    import subprocess
    import sys
    pred = OUT / "neural_predictions.npz"
    pred.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-m", "src.final.train_neural", "--device", device])
    if r.returncode != 0 or not pred.exists():
        raise SystemExit("neural training failed (see the error above); nothing was evaluated")
    z = np.load(pred)
    if len(z["rows"]) != n_rows:
        raise SystemExit("neural predictions do not match the held-out rows")
    out, per_seed = {}, {}
    for key in NEURAL:
        etas = [z[f"{key}_eta_s{s}"] for s in range(SEEDS)]
        probs = [z[f"{key}_prob_s{s}"] for s in range(SEEDS)]
        out[key] = seed_ensemble(etas, probs)
        per_seed[key] = etas
    return out, per_seed, pd.read_csv(OUT / "neural_training.csv")


# ------------------------------------------------------------------ metrics --
def evaluate_set(df, preds, models):
    t, e, pid = df.time.to_numpy(), df.event.to_numpy(), df.PLAYER_ID.to_numpy()
    known, y = horizon_label(t, e, H)
    cox = preds["cox"][0]
    rows = []
    for k in models:
        risk, prob = preds[k]
        prim = k == PRIMARY[1]
        n_boot = N_BOOT_PRIMARY if prim else N_BOOT
        r = {"model": LABELS[k], "key": k, "c_index": c_index(t, e, risk),
             "c_ci": ci(boot_values(lambda i: c_index(t[i], e[i], risk[i]), pid, n_boot)),
             "auc_14d": auc_at(t, e, risk, H),
             "auc_ci": ci(boot_values(lambda i: auc_at(t[i], e[i], risk[i], H), pid, n_boot)),
             "brier_14d": brier_at(t, e, prob, H), "mean_pred_14d": float(prob[known].mean()),
             "observed_14d": float(y[known].mean())}
        if k != "cox":
            d = boot_values(lambda i: c_index(t[i], e[i], risk[i]) - c_index(t[i], e[i], cox[i]), pid, n_boot)
            r["dc"] = r["c_index"] - c_index(t, e, cox)
            r["dc_ci"] = ci(d)
            r["dc_share_le0"] = float((d <= 0).mean())
            if prim:
                da = boot_values(lambda i: auc_at(t[i], e[i], risk[i], H) - auc_at(t[i], e[i], cox[i], H), pid, n_boot)
                r["dauc"] = r["auc_14d"] - auc_at(t, e, cox, H)
                r["dauc_ci"] = ci(da)
        rows.append(r)
    return rows


def table(rows):
    return pd.DataFrame([{
        "model": r["model"], "c_index": r["c_index"], "c_95ci": f"{r['c_ci'][0]:.3f}–{r['c_ci'][1]:.3f}",
        "c_minus_cox": r.get("dc", 0.0),
        "paired_95ci": f"{r['dc_ci'][0]:+.3f} to {r['dc_ci'][1]:+.3f}" if "dc_ci" in r else "",
        "auc_14d": r["auc_14d"], "auc_95ci": f"{r['auc_ci'][0]:.3f}–{r['auc_ci'][1]:.3f}",
        "brier_14d": r["brier_14d"], "mean_pred_14d": r["mean_pred_14d"], "observed_14d": r["observed_14d"]}
        for r in rows])


# ------------------------------------------------------------------ figures --
def fig_cindex(results, path):
    keys = [r["key"] for r in results["test"]][::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.62 * len(keys) + 1.5))
    for s, (off, mk, fill) in zip(SETS, ((0.13, "o", True), (-0.13, "D", False))):
        by = {r["key"]: r for r in results.get(s, [])}
        for y, k in enumerate(keys):
            if k not in by:
                continue
            r, c = by[k], COL[k]
            ax.plot(r["c_ci"], [y + off] * 2, color=c, lw=2 if fill else 1.2, solid_capstyle="round")
            ax.plot(r["c_index"], y + off, mk, ms=7 if fill else 5.5, color=c,
                    mfc=c if fill else "white", mec=c if not fill else "white", mew=1.4)
            if fill:
                ax.text(r["c_ci"][1] + 0.004, y + off, f"{r['c_index']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(keys)), [LABELS[k] for k in keys])
    ax.plot([], [], "o", color=INK2, label=SETS["test"])
    ax.plot([], [], "D", color=INK2, mfc="white", label=SETS["supplementary"])
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.set_xlabel("C-index (95% CI, bootstrap over players); models trained on 2016-17 to 2022-23")
    ax.set_title("Final evaluation on held-out seasons", loc="left", color=INK, fontsize=11, pad=24)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_calibration(df, preds, path):
    t, e = df.time.to_numpy(), df.event.to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    top = 0
    for k in PRIMARY:
        c = calibration_table(t, e, preds[k][1], H)
        ax.plot(c.mean_predicted, c.observed, marker="o", ms=5, lw=2, color=COL[k], label=LABELS[k])
        top = max(top, c.mean_predicted.max(), c.observed.max())
    ax.plot([0, top * 1.05], [0, top * 1.05], ls="--", lw=1, color=INK2, label="Perfect calibration")
    ax.set_xlabel(f"Mean predicted {H}-day risk (decile)")
    ax.set_ylabel(f"Observed {H}-day injury rate")
    ax.set_title(f"Calibration, test season 2023-24 ({H}-day risk)", loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def fig_km(df, risk, path):
    groups = pd.qcut(pd.Series(risk).rank(method="first"), 3, labels=["Low", "Medium", "High"]).to_numpy()
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    cols = {"Low": "#7fb0e8", "Medium": "#4f93df", "High": "#1f5fae"}
    for g in ["Low", "Medium", "High"]:
        m = groups == g
        km = kaplan_meier(df.time.to_numpy()[m], df.event.to_numpy()[m])
        ax.step(km.time, 1 - km.survival, where="post", lw=2, color=cols[g], label=f"{g} predicted risk")
    ax.set_xlabel("Days after the game")
    ax.set_ylabel("Cumulative injury incidence")
    ax.set_title("Test season 2023-24: injuries by predicted-risk tertile\n(Transformer + DeepSurv, 5 games)",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ report ---
def primary_sentence(r_cox, r_tf):
    lo, hi = r_tf["dc_ci"]
    verdict = ("The difference is above zero: the advantage seen on the validation season holds on the test season."
               if lo > 0 else
               "The interval includes zero: the test season does not confirm a reliable advantage over Cox."
               if hi > 0 else
               "The interval is below zero: Cox does better than the Transformer on the test season.")
    return (f"**Primary result (fixed in advance).** On the 2023-24 test season, Transformer + DeepSurv (5 games) "
            f"reaches C-index {r_tf['c_index']:.3f} ({r_tf['c_ci'][0]:.3f}–{r_tf['c_ci'][1]:.3f}) vs. Cox "
            f"{r_cox['c_index']:.3f} ({r_cox['c_ci'][0]:.3f}–{r_cox['c_ci'][1]:.3f}); difference "
            f"{r_tf['dc']:+.3f}, paired 95% CI {lo:+.3f} to {hi:+.3f} ({N_BOOT_PRIMARY} player resamples). "
            + verdict)


def write_report(results, sets, gbm_name, seeds_info):
    md = ["# Final evaluation on held-out seasons\n",
          "All models were refit on 2016-17 to 2022-23 (training + validation seasons) with the settings "
          "fixed in Updates 2-3, then scored once. The primary comparison was fixed in Update 3, before the "
          "test season was used.\n"]
    if "test" in results:
        by = {r["key"]: r for r in results["test"]}
        md += [primary_sentence(by["cox"], by["tf5"]), ""]
    for s, rows in results.items():
        df = sets[s]
        md += [f"## {SETS[s]}\n",
               f"{len(df):,} prediction points (player-games), {df.PLAYER_ID.nunique():,} players, "
               f"{int(df.event.sum()):,} with a target injury within the follow-up.\n",
               _md(table(rows)), ""]
    md += ["![C-index](fig_test_cindex.png)\n", "![Calibration](fig_test_calibration.png)\n",
           "![Risk tertiles](fig_test_km.png)\n",
           "## Notes\n",
           "- `c_minus_cox` and its paired 95% CI score both models on the same bootstrap resamples of players.",
           "- Neural rows are 3-seed ensembles. `mean_pred_14d` vs `observed_14d` compares the average "
           "predicted 14-day risk with the observed rate (landmarks whose 14-day status is known).",
           f"- Gradient boosting implementation: {gbm_name}.",
           "- The supplementary season ends at the end of the injury data (2025-01-12), so it covers the first "
           "half of 2024-25 only.",
           "", "## Neural training (per seed)\n", _md(seeds_info)]
    (OUT / "test_report.md").write_text("\n".join(md), encoding="utf-8")
    return md


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--force", action="store_true", help="re-run after a crash (results not yet looked at)")
    args = ap.parse_args()
    if LOCK.exists() and not args.force:
        raise SystemExit(f"The test season was already evaluated ({LOCK}). It is evaluated once; "
                         "results must not be used to change the models.")
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    t0 = _time.time()

    from src.models.neural import load
    data = load()
    lm = pd.read_csv(LANDMARKS, parse_dates=["GAME_DATE"])
    if not (np.array_equal(data["game_id"], lm.GAME_ID.to_numpy()) and
            np.array_equal(data["player_id"], lm.PLAYER_ID.to_numpy())):
        raise SystemExit("sequences.npz does not match landmarks - run python -m src.sequence.build")
    feats = model_features(lm)
    fit = lm[lm.SEASON.isin(FIT_SEASONS)].reset_index(drop=True)
    mask = lm.split.isin(list(SETS)).to_numpy()
    score = lm[mask].reset_index(drop=True)
    print(f"fit: {len(fit):,} landmarks ({FIT_SEASONS[0]} to {FIT_SEASONS[-1]}); "
          f"scoring {len(score):,} held-out landmarks")

    print("neural models (3 seeds each, separate process) ...", flush=True)
    npred, per_seed, seeds_info = neural_predictions(int(mask.sum()), args.device)
    print("baselines (Cox, logistic, XGBoost) ...", flush=True)
    preds, gbm_name = baseline_predictions(fit, score, feats)
    LABELS["xgboost"] = "XGBoost" if gbm_name.startswith("XGBoost") else "Gradient boosting"
    preds.update(npred)
    models = ["cox", "logistic", "xgboost", "deepsurv", "tf20", "tf5"]

    # ---- predictions file (used by SHAP and the dashboard) -------------------
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM", "GAME_ID", "GAME_DATE", "SEASON", "split", "time", "event"]
    P = score[keep].copy()
    for k in models:
        P[f"{k}_risk"], P[f"{k}_prob14"] = preds[k][0], preds[k][1]
    for s, eta in enumerate(per_seed["tf5"]):
        P[f"tf5_eta_s{s}"] = eta
    P["tf5_eta"] = np.mean(per_seed["tf5"], axis=0)
    P.to_csv(OUT / "test_predictions.csv.gz", index=False)

    # ---- metrics -------------------------------------------------------------
    sets, results = {}, {}
    for s in SETS:
        m = (score.split == s).to_numpy()
        if not m.any():
            continue
        sets[s] = score[m].reset_index(drop=True)
        sub = {k: (np.asarray(v[0])[m], np.asarray(v[1])[m]) for k, v in preds.items()}
        print(f"evaluating {SETS[s]} ...")
        results[s] = evaluate_set(sets[s], sub, models)
        if s == "test":
            fig_calibration(sets[s], sub, OUT / "fig_test_calibration.png")
            fig_km(sets[s], sub["tf5"][0], OUT / "fig_test_km.png")
    fig_cindex(results, OUT / "fig_test_cindex.png")
    all_rows = pd.concat([table(r).assign(set=s) for s, r in results.items()], ignore_index=True)
    md = write_report(results, sets, gbm_name, seeds_info)
    all_rows.to_csv(LOCK, index=False)          # written last: marks the test season as evaluated
    json.dump({"evaluated_at": datetime.now().isoformat(timespec="seconds"), "fit_seasons": FIT_SEASONS,
               "primary": PRIMARY, "seeds": SEEDS, "n_boot_primary": N_BOOT_PRIMARY,
               "minutes": round((_time.time() - t0) / 60, 1)}, open(OUT / "run_info.json", "w"), indent=2)
    print("\n".join(md))
    print(f"\nsaved {OUT}  ({(_time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
