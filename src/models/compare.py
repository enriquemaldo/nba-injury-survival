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

from matplotlib.ticker import MultipleLocator

from src.analysis.within_player import FEATURES as WP_FEATURES
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
    fig, ax = plt.subplots(figsize=(9.0, 0.5 * len(d) + 1.4))
    for y, r in enumerate(d):
        col = "#8a8984" if "baseline" in r["model"] else "#2a78d6"
        lo, hi = r["ci"]
        ax.plot([lo, hi], [y, y], color=col, lw=2, solid_capstyle="round")
        ax.plot(r["c_index"], y, "o", color=col, ms=8, mec="white", mew=1.5)
        ax.text(hi + 0.004, y, f"{r['c_index']:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(range(len(d)), [r["model"] for r in d])
    ax.xaxis.set_major_locator(MultipleLocator(0.05))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.grid(axis="x", which="minor", color=GRID, lw=0.5)
    ax.set_xlabel("C-index, validation season 2022-23 (95% CI, bootstrap over players)")
    ax.set_title("Baselines (grey) vs. neural survival models (blue)", loc="left", color=INK, fontsize=11)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


WP_SERIES = (("pooled", "#8a8984", "Pooled (between + within players)"),
             ("within", "#2a78d6", "Within player-season"),
             ("within_noramp", "#e07b39", "Within, without return-to-play periods"))


def fig_within(wp, path):
    d = wp[wp.feature.isin(WP_FEATURES)].iloc[::-1]
    series = [s for s in WP_SERIES if f"hr_{s[0]}" in d.columns]
    fig, ax = plt.subplots(figsize=(9.0, 0.55 * len(d) + 1.6))
    y = np.arange(len(d))
    offs = np.linspace(0.22, -0.22, len(series))
    for off, (pre, col, lab) in zip(offs, series):
        hr, lo, hi = d[f"hr_{pre}"], d[f"lo_{pre}"], d[f"hi_{pre}"]
        ax.errorbar(hr, y + off, xerr=[hr - lo, hi - hr], fmt="o", color=col, ecolor=col,
                    elinewidth=1.5, capsize=2, ms=4.5, label=lab)
    ax.axvline(1, color=INK2, lw=1, ls="--")
    ax.set_yticks(y, [WP_FEATURES[f] for f in d.feature])
    ax.set_xlabel("Hazard ratio per 1 SD (95% CI, clustered by player); training seasons")
    ax.set_title("When are players injured? Within-player vs. pooled associations", loc="left",
                 color=INK, fontsize=11, pad=28)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=3, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


FAMILY_SHORT = {"workload": "Workload", "schedule": "Schedule", "travel": "Travel",
                "history": "Injury history", "bio": "Player bio"}


def ablation_table(abl):
    """Cox (from src.analysis.ablations) and DeepSurv ablations in one table."""
    parts = []
    f = ANALYSIS / "ablations.csv"
    if f.exists():
        cox = pd.read_csv(f)
        cox["family"] = cox.family.str.split(" (", regex=False).str[0]
        parts.append(cox.assign(model="Cox")[["family", "model", "c_drop", "ci_lo", "ci_hi"]])
    if len(abl):
        ds = abl.rename(columns={"family removed": "family"})
        ds["family"] = ds.family.map(FAMILY_SHORT)
        parts.append(ds.assign(model="DeepSurv")[["family", "model", "c_drop", "ci_lo", "ci_hi"]])
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def fig_ablations(tab, path):
    order = ["Injury history", "Workload", "Schedule", "Travel", "Player bio"]
    fams = [f for f in order if f in set(tab.family)][::-1]
    fig, ax = plt.subplots(figsize=(9.0, 0.6 * len(fams) + 1.6))
    for off, (model, col) in zip((0.14, -0.14), (("Cox", "#8a8984"), ("DeepSurv", "#2a78d6"))):
        sub = tab[tab.model == model].set_index("family")
        ys = [i + off for i, f in enumerate(fams) if f in sub.index]
        if not ys:
            continue
        r = sub.loc[[f for f in fams if f in sub.index]]
        ax.errorbar(r.c_drop, ys, xerr=[r.c_drop - r.ci_lo, r.ci_hi - r.c_drop], fmt="o", color=col,
                    ecolor=col, elinewidth=1.8, capsize=2, ms=6, label=model)
    ax.axvline(0, color=INK2, lw=1, ls="--")
    ax.set_yticks(range(len(fams)), fams)
    ax.set_xlabel("Drop in validation C-index when the family is removed (paired 95% CI)")
    ax.set_title("Feature-family ablations: which features matter?", loc="left", color=INK, fontsize=11, pad=24)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def _f(v):
    return "" if isinstance(v, float) and np.isnan(v) else f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v)


def summary_page(rows, abl_tab, wp):
    cox = next(r for r in rows if r["model"].startswith("Cox (baseline)"))
    neu = [r for r in rows if "baseline" not in r["model"]]
    tf = [r for r in neu if r["model"].startswith("Transformer")]
    ds = next((r for r in neu if r["model"] == "DeepSurv"), None)
    best = max(neu, key=lambda r: r["c_index"]) if neu else None
    wp = wp.set_index("feature")
    m = wp.loc["ewma_min_short"]
    L = [("NBA non-contact injury survival — Update 3: neural survival models", 14, INK, "bold"),
         ("Validation season 2022-23. The test season (2023-24) has not been used.", 9.5, INK2, None), ("", 0, INK, None),
         ("1. Model comparison (C-index, 95% CI; difference vs Cox with paired 95% CI)", 11, INK, "bold"),
         (f"Cox baseline: {cox['c_index']:.3f} ({cox['ci'][0]:.3f}–{cox['ci'][1]:.3f})", 9.5, INK, None)]
    for r in neu:
        L.append((f"{r['model']}: {r['c_index']:.3f} ({r['ci'][0]:.3f}–{r['ci'][1]:.3f});  vs Cox "
                  f"{r['delta_vs_cox']:+.3f} ({r['delta_ci'][0]:+.3f} to {r['delta_ci'][1]:+.3f})", 9, INK, None))
    if tf and ds is not None:
        lo_d, hi_d = min(r["delta_vs_cox"] for r in tf), max(r["delta_vs_cox"] for r in tf)
        n_sig = sum(r["delta_ci"][0] > 0 for r in tf)
        above = all(r["delta_vs_cox"] > 0 for r in tf)
        L += [("", 0, INK, None),
              (f"• Transformer versions vs Cox: {lo_d:+.3f} to {hi_d:+.3f}"
               + (" (all above Cox)" if above else "") + f"; {n_sig} of {len(tf)} with a paired CI above 0.",
               9.5, INK2, None),
              (f"• DeepSurv on the same summary features vs Cox: {ds['delta_vs_cox']:+.3f}"
               + (" — the gain comes from the game-by-game sequence." if hi_d > ds["delta_vs_cox"] + 0.005 else "."),
               9.5, INK2, None),
              (f"• Best on validation: {best['model']}, chosen among {len(neu)} neural models on this season;",
               9.5, INK2, None),
              ("   the sealed test season decides whether its advantage over Cox is real.", 9.5, INK2, None)]
    L += [("", 0, INK, None), ("2. Timing: what precedes injury within the same player and season?", 11, INK, "bold"),
          (f"• Heavier recent minutes: HR {m.hr_within:.2f} per SD ({m.lo_within:.2f}–{m.hi_within:.2f})"
           + (f"; without return-to-play periods {m.hr_within_noramp:.2f} ({m.lo_within_noramp:.2f}–"
              f"{m.hi_within_noramp:.2f})." if "hr_within_noramp" in wp.columns else "."), 9.5, INK2, None),
          ("• Workload spikes (ACWR), back-to-backs, games in 7 days, travel and time zones: no association.",
           9.5, INK2, None)]
    if "hr_within_noramp" in wp.columns:
        mg = wp.loc["missed_team_games_14d"]
        L.append((f"• Missed games looked protective (HR {mg.hr_within:.2f}) but not after removing return-to-play "
                  f"periods (HR {mg.hr_within_noramp:.2f}): an artifact.", 9.5, INK2, None))
    if len(abl_tab):
        L += [("", 0, INK, None), ("3. Feature-family ablations", 11, INK, "bold")]
        h = abl_tab[abl_tab.family == "Injury history"].set_index("model")
        rest = abl_tab[abl_tab.family != "Injury history"]
        L.append(("• Injury history is the only family that reliably helps. Removing it costs:", 9.5, INK2, None))
        for mod, r in h.iterrows():
            L.append((f"     {mod}: C −{r.c_drop:.3f} (paired 95% CI {r.ci_lo:.3f}–{r.ci_hi:.3f})", 9.5, INK2, None))
        if len(rest):
            L.append((f"• Removing workload, schedule, travel or bio changes the C-index by at most "
                      f"{rest.c_drop.abs().max():.3f}.", 9.5, INK2, None))
    fig = plt.figure(figsize=(8.5, 11))
    y = 0.95
    for text, size, color, weight in L:
        if not text:
            y -= 0.012
            continue
        fig.text(0.06, y, text, fontsize=size, color=color, weight=weight or "normal", family="DejaVu Sans")
        y -= 0.034 if weight == "bold" and size >= 11 else 0.026
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
    wp = pd.read_csv(ANALYSIS / "within_player_hazard_ratios.csv")
    fig_within(wp, OUT / "fig_timing.png")
    abl_tab = ablation_table(abl)
    if len(abl_tab):
        fig_ablations(abl_tab, OUT / "fig_ablations_cox_deepsurv.png")
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
    md += ["![Within-player timing](fig_timing.png)\n", "![Ablations: Cox and DeepSurv](fig_ablations_cox_deepsurv.png)\n"]
    md += ["## See also\n", "- `within_player.md`: does recent workload predict *when* a player is injured?",
           "- `ablations.md`: Cox feature-family ablations."]
    (OUT / "model_comparison.md").write_text("\n".join(md), encoding="utf-8")
    with PdfPages(OUT / "evidence_summary.pdf") as pdf:
        pdf.savefig(summary_page(rows, abl_tab, wp))
        for f in (OUT / "fig_model_comparison.png", OUT / "fig_timing.png", OUT / "fig_ablations_cox_deepsurv.png"):
            if f.exists():
                pdf.savefig(image_page(f))
    plt.close("all")
    print("\n".join(md))
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
