"""Evidence pack for Update 2 (baseline models), like src/evidence.py for Update 1.

Usage:  python -m src.models.evidence      (after python -m src.models.baselines)
Writes data/processed/models/:
    fig_cindex.png            C-index with 95% CI for every model (validation season)
    evidence_summary.pdf      key numbers + all Update 2 figures in one file
"""
import ast

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.models.baselines import OUT

INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
FULL, REF = "#2a78d6", "#8a8984"          # full models vs. single-family reference models
plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "axes.axisbelow": True, "figure.dpi": 150})


def _ci(v):
    return ast.literal_eval(v) if isinstance(v, str) else (float("nan"), float("nan"))


def load():
    res = pd.read_csv(OUT / "baseline_results.csv")
    info = pd.read_json(OUT / "run_info.json", typ="series")
    gbm_label = "XGBoost" if str(info.gbm).startswith("XGBoost") else "Gradient boosting"
    res["label"] = res.model.replace({"GBM": gbm_label, "Logistic": "Logistic regression",
                                      "Cox": "Cox (full)"})
    return res, info, gbm_label


def fig_cindex(main, path):
    order = ["Cox (full)", "Logistic regression", main.label[main.model == "GBM"].iat[0],
             "Cox: injury history only", "Cox: workload + schedule/travel", "Cox: minutes only (EWMA)"]
    d = main.set_index("label").loc[order].iloc[::-1]
    lo = d.c_index_95ci.map(lambda v: _ci(v)[0])
    hi = d.c_index_95ci.map(lambda v: _ci(v)[1])
    colors = [REF if m.startswith("Cox:") else FULL for m in d.index]
    fig, ax = plt.subplots(figsize=(7, 3.6))
    y = range(len(d))
    for yi, c, l, h, col in zip(y, d.c_index, lo, hi, colors):
        ax.plot([l, h], [yi, yi], color=col, lw=2, solid_capstyle="round")
        ax.plot(c, yi, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(h + 0.006, yi, f"{c:.3f}", va="center", color=INK, fontsize=9)
    ax.axvline(0.5, color=INK2, lw=1, ls="--")
    ax.text(0.503, -0.45, "random ranking", color=INK2, fontsize=8, va="center")
    ax.set_ylim(-0.7, len(d) - 0.5)
    lab = lambda m: (m[5].upper() + m[6:] + " (reference)") if m.startswith("Cox: ") else m
    ax.set_yticks(list(y), [lab(m) for m in d.index])
    ax.set_xlim(0.48, 0.86)
    ax.set_xlabel("C-index on validation season 2022-23 (95% CI, bootstrap over players)")
    ax.set_title("Full models vs. single-feature-family reference models", loc="left", color=INK,
                 fontsize=11, pad=10)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def summary_page(res, info, gbm_label):
    main = res[res.run == "main"].set_index("model")
    sens = res[(res.run != "main") & res.model.isin(["Cox", "Logistic", "GBM"])]
    delta = (sens.set_index(["run", "model"]).c_index
             - main.c_index.reindex(sens.model).to_numpy()).abs().max()
    c = lambda m: f"{main.c_index[m]:.3f} ({_ci(main.c_index_95ci[m])[0]:.3f}–{_ci(main.c_index_95ci[m])[1]:.3f})"
    lines = [
        ("NBA non-contact injury survival — Update 2: baseline models", 15, INK),
        ("", 6, INK),
        (f"Survival dataset: {info.landmarks:,} player-game landmarks, {info.players:,} players; "
         f"follow-up up to 30 days; {info.features} features", 10, INK2),
        (f"Training {info.train_seasons}: {info.train_landmarks:,} landmarks "
         f"({info.train_event_landmarks:,} ending in a target injury)", 10, INK2),
        (f"Validation 2022-23: {info.val_landmarks:,} landmarks ({info.val_event_landmarks:,} ending in a "
         f"target injury)", 10, INK2),
        ("Test season 2023-24 not used (sealed until final model selection).", 10, INK2),
        ("", 6, INK),
        ("C-index on validation season (95% CI)", 12, INK),
        (f"    Cox (full):              {c('Cox')}", 11, INK),
        (f"    Logistic regression:  {c('Logistic')}", 11, INK),
        (f"    {gbm_label}:{' ' * max(1, 22 - len(gbm_label))}{c('GBM')}", 11, INK),
        ("", 6, INK),
        (f"Injury history alone: {main.c_index['Cox: injury history only']:.3f};  workload + schedule/travel "
         f"alone: {main.c_index['Cox: workload + schedule/travel']:.3f};  minutes alone: "
         f"{main.c_index['Cox: minutes only (EWMA)']:.3f}", 10, INK2),
        ("-> The models mainly identify which players get hurt, not when.", 10, INK2),
        (f"Sensitivity analyses (train from 2018-19; exclude 2019-21) change C-index by at most "
         f"{delta:.3f}.", 10, INK2),
    ]
    fig = plt.figure(figsize=(8.5, 4.9))
    y = 0.93
    for text, size, color in lines:
        fig.text(0.05, y, text, fontsize=size, color=color, family="DejaVu Sans")
        y -= 0.064 if text else 0.028
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
    res, info, gbm_label = load()
    fig_cindex(res[res.run == "main"], OUT / "fig_cindex.png")
    with PdfPages(OUT / "evidence_summary.pdf") as pdf:
        pdf.savefig(summary_page(res, info, gbm_label))
        for f in ("fig_cindex.png", "fig_calibration.png", "fig_km_risk_tertiles.png", "fig_cox_forest.png"):
            if (OUT / f).exists():
                pdf.savefig(image_page(OUT / f))
    plt.close("all")
    print(f"saved {OUT / 'fig_cindex.png'} and {OUT / 'evidence_summary.pdf'}")


if __name__ == "__main__":
    main()
