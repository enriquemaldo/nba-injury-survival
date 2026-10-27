"""Update 3: compare every model on the validation season and build the evidence pack.

Inputs (run these first):
    python -m src.models.baselines          -> data/processed/models/
    python -m src.models.neural             -> data/processed/neural/
    python -m src.analysis.within_player    -> data/processed/analysis/
    python -m src.analysis.ablations        -> data/processed/analysis/

Neural models are reported two ways: mean +/- sd of the C-index over seeds, and
a seed ensemble (average of each seed's standardized log-risk). Differences
from the Cox baseline use PAIRED player-bootstrap 95% CIs.

Usage:  python -m src.models.compare
Writes: data/processed/update3/ (report, figures, evidence_summary.pdf)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.config import PROCESSED
from src.models.baselines import _md
from src.models.metrics import auc_at, c_index, cluster_bootstrap
from src.survival.dataset import LANDMARKS

OUT = PROCESSED / "update3"
OUT.mkdir(parents=True, exist_ok=True)
BASE = PROCESSED / "models"
NEURAL = PROCESSED / "neural"
ANALYSIS = PROCESSED / "analysis"
H = 14
LABELS = {"deepsurv": "DeepSurv", "tf_full": "Transformer + DeepSurv (20 games + all features)",
          "tf_seq": "Transformer + DeepSurv (20 games, no workload aggregates)",
          "tf_len5": "Transformer + DeepSurv (5 games)", "tf_len10": "Transformer + DeepSurv (10 games)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})


def load():
    lm = pd.read_csv(LANDMARKS, usecols=["PLAYER_ID", "GAME_ID", "split", "time", "event"])
    val = lm[lm.split == "val"].reset_index(drop=True)
    base = pd.read_csv(BASE / "val_predictions.csv")
    neu = pd.read_csv(NEURAL / "val_predictions.csv")
    df = val.merge(base.drop(columns=["time", "event"]), on=["PLAYER_ID", "GAME_ID"], how="left") \
            .merge(neu, on=["PLAYER_ID", "GAME_ID"], how="left")
    if df.Cox_risk.isna().any():
        raise SystemExit("baseline predictions missing - run python -m src.models.baselines")
    return df


def ensemble(df, run):
    cols = [c for c in df.columns if c.startswith(f"{run}_s") and c.removeprefix(f"{run}_s").isdigit()]
    if not cols:
        return None, []
    z = [(df[c] - df[c].mean()) / df[c].std() for c in cols]
    return np.mean(z, axis=0), cols


def compare(df):
    t, e, pid = df.time.to_numpy(), df.event.to_numpy(), df.PLAYER_ID.to_numpy()
    cox = df.Cox_risk.to_numpy()
    rows = []
    models = [("Cox (baseline)", "Cox_risk"), ("Logistic regression (baseline)", "Logistic_risk"),
              ("XGBoost (baseline)", "GBM_risk")]
    info = pd.read_json(BASE / "run_info.json", typ="series") if (BASE / "run_info.json").exists() else None
    if info is not None and not str(info.gbm).startswith("XGBoost"):
        models[2] = ("Gradient boosting (baseline)", "GBM_risk")
    for label, col in models:
        r = df[col].to_numpy()
        rows.append({"model": label, "c_index": c_index(t, e, r), "c_sd_seeds": np.nan, "auc_14d": auc_at(t, e, r, H),
                     "ci": cluster_bootstrap(lambda i: c_index(t[i], e[i], r[i]), pid), "seeds": 1, "risk": r})
    for run, label in LABELS.items():
        ens, cols = ensemble(df, run)
        if ens is None:
            continue
        per_seed = [c_index(t, e, df[c].to_numpy()) for c in cols]
        rows.append({"model": label, "c_index": c_index(t, e, ens), "c_sd_seeds": np.std(per_seed, ddof=1) if len(cols) > 1 else np.nan,
                     "c_mean_seeds": np.mean(per_seed), "auc_14d": auc_at(t, e, ens, H),
                     "ci": cluster_bootstrap(lambda i: c_index(t[i], e[i], ens[i]), pid), "seeds": len(cols), "risk": ens})
    for r in rows:
        if r["model"].startswith("Cox (baseline)"):
            r["delta_vs_cox"], r["delta_ci"] = 0.0, (0.0, 0.0)
            continue
        rk = r["risk"]
        r["delta_vs_cox"] = r["c_index"] - c_index(t, e, cox)
        r["delta_ci"] = cluster_bootstrap(lambda i: c_index(t[i], e[i], rk[i]) - c_index(t[i], e[i], cox[i]), pid)
    return rows


def neural_ablations(df):
    t, e, pid = df.time.to_numpy(), df.event.to_numpy(), df.PLAYER_ID.to_numpy()
    full, _ = ensemble(df, "deepsurv")
    out = []
    if full is None:
        return pd.DataFrame()
    for fam in ["workload", "schedule", "travel", "history", "bio"]:
        alt, cols = ensemble(df, f"deepsurv_no_{fam}")
        if alt is None:
            continue
        lo, hi = cluster_bootstrap(lambda i: c_index(t[i], e[i], full[i]) - c_index(t[i], e[i], alt[i]), pid)
        out.append({"family removed": fam, "c_without": c_index(t, e, alt),
                    "c_drop": c_index(t, e, full) - c_index(t, e, alt), "ci_lo": lo, "ci_hi": hi})
    return pd.DataFrame(out)


def fig_models(rows, path):
    d = list(reversed(rows))
    fig, ax = plt.subplots(figsize=(7.6, 0.5 * len(d) + 1.4))
    for y, r in enumerate(d):
        col = "#8a8984" if "baseline" in r["model"] else "#2a78d6"
        lo, hi = r["ci"]
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["c_index"], y, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(hi + 0.005, y, f"{r['c_index']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(d)), [r["model"] for r in d])
    ax.set_xlabel("C-index, validation season 2022-23 (95% CI, bootstrap over players)")
    ax.set_title("Baselines (grey) vs. neural survival models (blue)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _f(v):
    return "" if isinstance(v, float) and np.isnan(v) else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)


def summary_page(rows, abl):
    cox = next(r for r in rows if r["model"].startswith("Cox (baseline)"))
    neu = [r for r in rows if "baseline" not in r["model"]]
    best = max(neu, key=lambda r: r["c_index"]) if neu else None
    wp = pd.read_csv(ANALYSIS / "within_player_hazard_ratios.csv").set_index("feature")
    m = wp.loc["ewma_min_short"]
    lines = [("NBA non-contact injury survival — Update 3: neural survival models", 15, INK), ("", 6, INK),
             ("Validation season 2022-23; test season 2023-24 still sealed.", 10, INK2), ("", 6, INK),
             ("C-index (95% CI)", 12, INK),
             (f"    Cox baseline: {cox['c_index']:.3f} ({cox['ci'][0]:.3f}–{cox['ci'][1]:.3f})", 11, INK)]
    for r in neu:
        lines.append((f"    {r['model']}: {r['c_index']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f});  vs Cox "
                      f"{r['delta_vs_cox']:+.3f} ({r['delta_ci'][0]:+.3f} to {r['delta_ci'][1]:+.3f})", 9, INK))
    lines += [("", 6, INK),
              (f"Within player-season, heavier recent minutes precede injury: HR {m.hr_within:.2f} per SD "
               f"({m.lo_within:.2f}–{m.hi_within:.2f}).", 10, INK2),
              ("Workload spikes (ACWR), back-to-backs, schedule density and travel: no association.", 10, INK2)]
    if len(abl):
        top = abl.sort_values("c_drop", ascending=False).iloc[0]
        lines.append((f"Ablation: removing {top['family removed']} costs the most "
                      f"(C −{top.c_drop:.3f}).", 10, INK2))
    fig = plt.figure(figsize=(8.5, 0.32 * len(lines) + 1.0))
    y = 0.94
    step = 0.86 / max(len(lines), 1)
    for text, size, color in lines:
        fig.text(0.04, y, text, fontsize=size, color=color, family="DejaVu Sans")
        y -= step if text else step * 0.5
    return fig


def image_page(path):
    img = mpimg.imread(path)
    h, w = img.shape[:2]
    fig = plt.figure(figsize=(8.5, 8.5 * h / w))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(img)
    ax.axis("off")
    return fig


def main():
    df = load()
    rows = compare(df)
    abl = neural_ablations(df)
    fig_models(rows, OUT / "fig_model_comparison.png")
    tab = pd.DataFrame([{"model": r["model"], "seeds": r["seeds"], "c_index": r["c_index"],
                         "c_95ci": f"{r['ci'][0]:.3f}–{r['ci'][1]:.3f}",
                         "c_sd_over_seeds": r["c_sd_seeds"], "auc_14d": r["auc_14d"],
                         "c_minus_cox": r["delta_vs_cox"],
                         "paired_95ci": f"{r['delta_ci'][0]:+.3f} to {r['delta_ci'][1]:+.3f}"} for r in rows])
    tab.to_csv(OUT / "model_comparison.csv", index=False)
    nres = pd.read_csv(NEURAL / "neural_results.csv") if (NEURAL / "neural_results.csv").exists() else None
    md = ["# Update 3: neural survival models vs. baselines (validation season 2022-23)\n",
          "Neural rows use the seed ensemble (average standardized log-risk over seeds); `c_sd_over_seeds` "
          "is the spread of single-seed C-indices. `c_minus_cox` and its paired 95% CI compare each model "
          "with the Cox baseline on the same bootstrap resamples of players: an interval containing 0 "
          "means no reliable difference. The test season (2023-24) was not used.\n",
          "\n".join(["| " + " | ".join(tab.columns) + " |", "|" + "---|" * len(tab.columns)] +
                    ["| " + " | ".join(_f(v) for v in r) + " |" for r in tab.itertuples(index=False)]), "",
          "![Model comparison](fig_model_comparison.png)\n"]
    if len(abl):
        md += ["## DeepSurv feature-family ablations (seed ensembles)\n", _md(abl), ""]
    if nres is not None:
        md += ["## Training details (per seed)\n",
               _md(nres[["run", "seed", "epochs", "inner_c", "c_index", "auc_14d", "brier_14d"]]), ""]
    md += ["## See also\n", "- `within_player.md`: does recent workload predict *when* a player is injured?",
           "- `ablations.md`: Cox feature-family ablations."]
    (OUT / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    with PdfPages(OUT / "evidence_summary.pdf") as pdf:
        pdf.savefig(summary_page(rows, abl))
        for f in (OUT / "fig_model_comparison.png", ANALYSIS / "fig_within_player.png", ANALYSIS / "fig_ablations.png"):
            if f.exists():
                pdf.savefig(image_page(f))
    plt.close("all")
    print("\n".join(md))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
