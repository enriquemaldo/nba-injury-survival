"""Sensitivity analyses of the final pipeline (proposal section 9).

Descriptive only: every variant changes ONE thing, is refit the same way as the
final models, and is scored on the 2023-24 test season. Nothing here replaces or
retunes the final models; the primary result in test_report.md stands as reported.

    Seasons        refit without the atypical 2019-20 and 2020-21 seasons (Cox and Transformer)
    Workload       alternative workload definitions for Cox: acute windows only (<= 7 days),
                   chronic windows only (14-28 days), no acute:chronic ratio features
    Architecture   smaller and larger Transformer (5 games)
    Sequence       20 instead of 5 previous games (already in the final evaluation)

Each variant is compared with its own primary model (Cox or Transformer + DeepSurv, 5 games)
on the same bootstrap resamples of players (paired 95% CI).

Usage:  python -m src.final.sensitivity              (about 20-30 min on a laptop CPU)
        python -m src.final.sensitivity --cox-only   (a minute; skips the Transformer variants)
Writes: data/processed/final/sensitivity.md, sensitivity.csv, fig_sensitivity.png
"""
import argparse
import subprocess
import sys
import time as _time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.final.evaluate import FIT_SEASONS, H, OUT, SEEDS, seed_ensemble
from src.models.baselines import Preprocessor, _md
from src.models.coxph import CoxPH
from src.models.metrics import auc_at, c_index, cluster_bootstrap
from src.survival.dataset import LANDMARKS, model_features

ATYPICAL = ["2019-20", "2020-21"]
ACUTE = ["prev_game_min", "min_7d", "poss_7d", "games_3d", "games_7d", "ewma_min_short", "consecutive_games"]
CHRONIC = ["min_14d", "min_28d", "poss_28d", "usage_per_min_28d", "games_14d", "b2b_apps_14d", "ewma_min_long",
           "missed_team_games_14d"]
RATIOS = ["acwr_min", "ewma_ratio_min"]
COX_VARIANTS = {  # name: (group, label, fit seasons, features dropped)
    "cox_excl_2019_21": ("Seasons", "Without 2019-20 and 2020-21", [s for s in FIT_SEASONS if s not in ATYPICAL], []),
    "cox_acute_only": ("Workload definition", "Acute windows only (up to 7 days)", FIT_SEASONS, CHRONIC + RATIOS),
    "cox_chronic_only": ("Workload definition", "Chronic windows only (14-28 days)", FIT_SEASONS, ACUTE + RATIOS),
    "cox_no_ratios": ("Workload definition", "No acute:chronic ratios", FIT_SEASONS, RATIOS),
}
TF_VARIANTS = {
    "tf5_excl_2019_21": ("Seasons", "Without 2019-20 and 2020-21"),
    "tf5_small": ("Architecture", "Smaller Transformer (d=16, 1 layer)"),
    "tf5_large": ("Architecture", "Larger Transformer (d=64, 3 layers)"),
}
INK, INK2 = "#0b0b0b", "#52514e"


def cox_fit_predict(train, score, feats):
    """Same specification as the final Cox model (penalizer 1.0, standardized features)."""
    pre = Preprocessor().fit(train, feats)
    m = CoxPH(penalizer=1.0).fit(pre.transform(train)[0], train.time, train.event)
    return m.predict_log_hazard(pre.transform(score)[0])


def compare(test, ref, alt):
    t, e, pid = test.time.to_numpy(), test.event.to_numpy(), test.PLAYER_ID.to_numpy()
    lo, hi = cluster_bootstrap(lambda i: c_index(t[i], e[i], alt[i]) - c_index(t[i], e[i], ref[i]), pid)
    return {"c_index": c_index(t, e, alt), "delta": c_index(t, e, alt) - c_index(t, e, ref),
            "ci_lo": lo, "ci_hi": hi, "auc_14d": auc_at(t, e, alt, H)}


def run_neural(device, seeds):
    pred = OUT / "sensitivity_neural.npz"
    pred.unlink(missing_ok=True)
    r = subprocess.run([sys.executable, "-m", "src.final.sensitivity_neural", "--device", device,
                        "--seeds", str(seeds)])
    if r.returncode != 0 or not pred.exists():
        raise SystemExit("Transformer variants failed (see the error above)")
    return np.load(pred)


def fig(tab, path):
    d = tab[tab.group != "Primary"].iloc[::-1]
    fig_, ax = plt.subplots(figsize=(8.6, 0.55 * len(d) + 1.4))
    y = np.arange(len(d))
    for yi, (_, r) in zip(y, d.iterrows()):
        col = "#1f5fae" if r.model.startswith("Transformer") else "#8a8984"
        ax.plot([r.ci_lo, r.ci_hi], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(r.delta, yi, "o", color=col, ms=7)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [f"{r.model}: {r.variant}" for _, r in d.iterrows()], fontsize=8.5)
    ax.set_xlabel("C-index change vs. the primary model, test season 2023-24 (paired 95% CI)")
    ax.set_title("Sensitivity analyses (one change at a time)", loc="left", color=INK, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#e6e5e1")
    fig_.tight_layout()
    fig_.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cox-only", action="store_true")
    ap.add_argument("--seeds", type=int, default=SEEDS)
    args = ap.parse_args()
    t0 = _time.time()
    lm = pd.read_csv(LANDMARKS)
    P = pd.read_csv(OUT / "test_predictions.csv.gz")
    feats = model_features(lm)
    held = lm[lm.split.isin(["test", "supplementary"])].reset_index(drop=True)
    if not np.array_equal(held.GAME_ID.to_numpy(), P.GAME_ID.to_numpy()):
        raise SystemExit("final predictions do not match the landmarks - rerun src.final.evaluate first")
    is_test = (held.split == "test").to_numpy()
    test = held[is_test].reset_index(drop=True)
    rows = []

    # ---- Cox -----------------------------------------------------------------
    print("Cox variants ...", flush=True)
    fit = lm[lm.SEASON.isin(FIT_SEASONS)]
    cox_ref = cox_fit_predict(fit, held, feats)[is_test]
    agree = np.corrcoef(cox_ref, P.cox_risk.to_numpy()[is_test])[0, 1]
    if agree < 0.999:
        print(f"  note: refit Cox differs slightly from the final one (correlation {agree:.4f})")
    rows.append({"group": "Primary", "model": "Cox", "variant": "Final model", "c_index": c_index(
        test.time, test.event, cox_ref), "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
        "auc_14d": auc_at(test.time, test.event, cox_ref, H)})
    for name, (group, label, seasons, drop) in COX_VARIANTS.items():
        f = [x for x in feats if x not in drop]
        alt = cox_fit_predict(lm[lm.SEASON.isin(seasons)], held, f)[is_test]
        rows.append({"group": group, "model": "Cox", "variant": label, **compare(test, cox_ref, alt)})
        print(f"  {label}: {rows[-1]['delta']:+.4f}", flush=True)

    # ---- Transformer -----------------------------------------------------------
    tf_ref = P.tf5_risk.to_numpy()[is_test]
    rows.append({"group": "Primary", "model": "Transformer + DeepSurv", "variant": "Final model (5 games)",
                 "c_index": c_index(test.time, test.event, tf_ref), "delta": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                 "auc_14d": auc_at(test.time, test.event, tf_ref, H)})
    rows.append({"group": "Sequence length", "model": "Transformer + DeepSurv", "variant": "20 previous games",
                 **compare(test, tf_ref, P.tf20_risk.to_numpy()[is_test])})
    if not args.cox_only:
        print(f"Transformer variants ({args.seeds} seeds each, separate process) ...", flush=True)
        z = run_neural(args.device, args.seeds)
        if len(z["rows"]) != len(held):
            raise SystemExit("Transformer variant predictions do not match the held-out rows")
        for name, (group, label) in TF_VARIANTS.items():
            etas = [z[f"{name}_eta_s{s}"] for s in range(args.seeds)]
            probs = [z[f"{name}_prob_s{s}"] for s in range(args.seeds)]
            risk, _ = seed_ensemble(etas, probs)
            rows.append({"group": group, "model": "Transformer + DeepSurv", "variant": label,
                         **compare(test, tf_ref, risk[is_test])})

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / "sensitivity.csv", index=False)
    fig(tab, OUT / "fig_sensitivity.png")
    show = tab.assign(paired_95ci=[f"{a:+.3f} to {b:+.3f}" if g != "Primary" else "" for a, b, g in
                                   zip(tab.ci_lo, tab.ci_hi, tab.group)])
    md = ["# Sensitivity analyses (test season 2023-24)\n",
          "Each row changes one thing and is refit the same way as the final models; `delta` is the change in "
          "C-index against that model's primary version, with a paired 95% CI over 200 player resamples. These "
          "are descriptive: none of them was used to choose or change the final models.\n",
          _md(show[["group", "model", "variant", "c_index", "delta", "paired_95ci", "auc_14d"]].round(3)), "",
          "![Sensitivity](fig_sensitivity.png)\n"]
    (OUT / "sensitivity.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"saved {OUT}  ({(_time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
