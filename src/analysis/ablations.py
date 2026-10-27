"""Feature-family ablations (proposal section 5): how much does each family add?

The Cox model is refitted on the training seasons with one feature family
removed at a time, and evaluated on the validation season. The drop in
C-index is reported with a PAIRED 95% CI: both models are scored on the same
bootstrap resamples of players, which is the right way to compare two models
on one dataset (the uncertainty of each model's own C-index is mostly shared).

Usage:  python -m src.analysis.ablations
Writes: data/processed/analysis/ablations.md (+ csv, figure)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import TRAIN_SEASONS
from src.features.workload import HISTORY_FEATURES, WORKLOAD_FEATURES
from src.models.baselines import Preprocessor, _md
from src.models.coxph import CoxPH
from src.models.metrics import c_index, cluster_bootstrap
from src.survival.dataset import BIO_FEATURES, LANDMARKS, model_features
from src.analysis.within_player import OUT

FAMILIES = {
    "Workload (minutes, usage, EWMA, ACWR, games played)": WORKLOAD_FEATURES,
    "Schedule (rest, back-to-back, home, team games)": ["is_home", "is_neutral", "days_rest", "back_to_back",
                                                        "team_games_last_7d"],
    "Travel (distance, time zones)": ["travel_km", "abs_tz_shift_h", "tz_shift_h", "km_last_7d", "km_last_14d"],
    "Injury history": HISTORY_FEATURES,
    "Player bio (age, height, weight, experience)": BIO_FEATURES,
}


def cox_risk(train, val, feats):
    pre = Preprocessor().fit(train, feats)
    m = CoxPH(penalizer=1.0).fit(pre.transform(train)[0], train.time, train.event)
    return m.predict_log_hazard(pre.transform(val)[0])


def paired(val, risk_full, risk_alt, n_boot=200):
    t, e, pid = val.time.to_numpy(), val.event.to_numpy(), val.PLAYER_ID.to_numpy()
    d = c_index(t, e, risk_full) - c_index(t, e, risk_alt)
    lo, hi = cluster_bootstrap(lambda i: c_index(t[i], e[i], risk_full[i]) - c_index(t[i], e[i], risk_alt[i]), pid,
                               n_boot=n_boot)
    return d, lo, hi


def fig(tab, path):
    d = tab.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.55 * len(d) + 1.3))
    y = np.arange(len(d))
    col = ["#2a78d6" if lo > 0 else "#8a8984" for lo in d.ci_lo]
    for yi, v, lo, hi, c in zip(y, d.c_drop, d.ci_lo, d.ci_hi, col):
        ax.plot([lo, hi], [yi, yi], color=c, lw=2, solid_capstyle="round")
        ax.plot(v, yi, "o", color=c, ms=8, mec="white", mew=1.5)
    ax.axvline(0, color="#52514e", lw=1, ls="--")
    ax.set_yticks(y, d.family)
    ax.set_xlabel("Drop in validation C-index when the family is removed (paired 95% CI)")
    ax.set_title("Feature-family ablations (Cox)", loc="left", fontsize=11)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    lm = pd.read_csv(LANDMARKS)
    feats = model_features(lm)
    train = lm[lm.SEASON.isin(TRAIN_SEASONS)].reset_index(drop=True)
    val = lm[lm.split == "val"].reset_index(drop=True)
    full = cox_risk(train, val, feats)
    c_full = c_index(val.time, val.event, full)
    rows = []
    for fam, cols in FAMILIES.items():
        present = [c for c in cols if c in feats]
        if not present:
            print(f"skip {fam}: not in features")
            continue
        alt = cox_risk(train, val, [f for f in feats if f not in present])
        d, lo, hi = paired(val, full, alt)
        rows.append({"family": fam, "n_features": len(present), "c_without": c_index(val.time, val.event, alt),
                     "c_drop": d, "ci_lo": lo, "ci_hi": hi})
        print(f"  {fam}: drop {d:+.4f} ({lo:+.4f}, {hi:+.4f})")
    tab = pd.DataFrame(rows).sort_values("c_drop", ascending=False).reset_index(drop=True)
    tab.to_csv(OUT / "ablations.csv", index=False)
    fig(tab, OUT / "fig_ablations.png")
    md = ["# Feature-family ablations (Cox, validation season 2022-23)\n",
          f"Full Cox model C-index: **{c_full:.3f}** ({len(feats)} features). Each row refits the model "
          "without one family. `c_drop` > 0 means the family helps; the paired 95% CI comes from scoring "
          "both models on the same 200 bootstrap resamples of players. A CI that excludes 0 is a "
          "reliable contribution.\n", _md(tab), "", "![Ablations](fig_ablations.png)\n"]
    (OUT / "ablations.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
