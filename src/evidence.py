"""Evidence pack for progress updates.

Usage:
    python -m src.evidence                 # figures + PDF summary + audit sheet
    python -m src.evidence --score         # score the audit sheet after you fill it in

Outputs (data/processed/evidence/):
    fig1_events_per_season.png    proxy time-loss events per season, by split
    fig2_reporting_change.png     proxy events vs. reason-less IL placements
    fig3_events_by_site.png       proxy events by body site
    evidence_summary.pdf          key numbers + all three figures, one file
    label_audit.csv               100 random injury rows to check by hand
    label_audit_results.md        (after --score) agreement rate

Audit instructions: open label_audit.csv in Excel/Numbers. For each row, read
the note and put 1 in `correct` if the assigned tier is right, 0 if not. If
wrong, write the right tier in `correct_tier` (strict / proxy / broad /
other_injury / uncertain / non_injury). Save as CSV (same name), then run
--score.
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

from src.config import INTERIM, PROCESSED, SEASONS, split_of

OUT = PROCESSED / "evidence"
OUT.mkdir(parents=True, exist_ok=True)
AUDIT = OUT / "label_audit.csv"

# Validated categorical palette (blue, orange, aqua) + neutral for partial data.
SPLIT_COLORS = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a", "supplementary": "#b5b4ae"}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test",
                "supplementary": "Supplementary (partial labels)"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "figure.dpi": 150,
})


def _load():
    ep = pd.read_csv(INTERIM / "injury_episodes.csv", parse_dates=["onset_date"])
    ep["time_loss"] = ep.games_missed.fillna(0) >= 1
    rows = pd.read_csv(INTERIM / "injury_rows_labeled.csv", parse_dates=["Date"])
    per = pd.DataFrame({"season": SEASONS})
    per["split"] = per.season.map(split_of)
    per["proxy_tl"] = per.season.map((ep.is_proxy & ep.time_loss).groupby(ep.season).sum()).fillna(0).astype(int)
    bare = ep[(ep.tier == "uncertain")
              & ep.notes.fillna("").str.fullmatch(r"(?i)placed on IL(\s*\(P\))?")]
    per["bare_il"] = per.season.map(bare.groupby("season").size()).fillna(0).astype(int)
    per = per[per.season.isin(ep.season.unique())]  # drop seasons with no labels
    return ep, rows, per


def fig_events(per):
    fig, ax = plt.subplots(figsize=(8, 4))
    x = range(len(per))
    bars = ax.bar(x, per.proxy_tl, color=[SPLIT_COLORS[s] for s in per.split],
                  width=0.7, edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, per.proxy_tl):
        ax.text(b.get_x() + b.get_width() / 2, v + per.proxy_tl.max() * 0.015, str(v),
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_xticks(list(x), [s[2:] for s in per.season])
    ax.set_xlabel("Season")
    ax.set_ylabel("Proxy-tier injury events (time-loss)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=SPLIT_COLORS[s]) for s in SPLIT_COLORS if s in set(per.split)]
    ax.legend(handles, [SPLIT_LABELS[s] for s in SPLIT_COLORS if s in set(per.split)],
              frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=4, fontsize=9,
              handlelength=1, columnspacing=1.2)
    ax.set_title("Injury events per season by data split", loc="left", color=INK, fontsize=12, pad=26)
    ax.set_ylim(0, per.proxy_tl.max() * 1.12)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_reporting(per):
    per = per[per.split != "supplementary"]  # partial seasons would show a fake drop
    fig, ax = plt.subplots(figsize=(9, 4))
    x = list(range(len(per)))
    for col, color, label in [("proxy_tl", "#2a78d6", "Proxy injury events"),
                              ("bare_il", "#eb6834", "IL placements with no stated reason")]:
        ax.plot(x, per[col], color=color, linewidth=2, marker="o", markersize=5, label=label)
        top = max(per.proxy_tl.max(), per.bare_il.max())
        lift = 9 if per[col].iloc[-1] < 0.05 * top else 0  # keep end label off the x-axis
        ax.annotate(label, (x[-1], per[col].iloc[-1]), xytext=(6, lift), textcoords="offset points",
                    va="center", color=INK2, fontsize=9)
    ax.set_xticks(x, [s[2:] for s in per.season])  # "16-17" keeps ticks readable
    ax.set_xlabel("Season")
    ax.set_ylabel("Episodes per season")
    ax.set_title("Reporting change: early seasons record injuries without a reason",
                 loc="left", color=INK, fontsize=12, pad=26)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    ax.set_xlim(-0.4, len(x) + 2.6)
    ax.set_ylim(0, max(per.proxy_tl.max(), per.bare_il.max()) * 1.08)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def fig_sites(ep):
    s = ep[ep.is_proxy].site.value_counts().sort_values()
    names = {"groin_adductor": "Groin / adductor", "hip_flexor": "Hip flexor", "achilles": "Achilles",
             "hamstring": "Hamstring", "calf": "Calf", "quad": "Quadriceps"}
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh([names.get(i, i) for i in s.index], s.values, color="#2a78d6", height=0.65,
                   edgecolor="white", linewidth=1.5)
    for b, v in zip(bars, s.values):
        ax.text(v + 3, b.get_y() + b.get_height() / 2, str(v), va="center", color=INK, fontsize=9)
    ax.set_xlabel("Proxy-tier injury episodes (all seasons)")
    ax.set_title("Injury events by body site", loc="left", color=INK, fontsize=12)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, s.max() * 1.15)
    fig.tight_layout()
    return fig


def summary_page(ep, rows, per):
    src = pd.read_json(INTERIM / "label_source.json", typ="series")
    inj = rows[(rows.tier != "non_injury") & (rows.get("match_method", "") != "not_player")]
    tr, va, te = (int(per[per.split == k].proxy_tl.sum()) for k in ("train", "val", "test"))
    lines = [
        ("NBA non-contact injury survival — data feasibility summary", 15, INK),
        ("", 8, INK),
        (f"Label source: Pro Sports Transactions injured-list data (Kaggle 2016-2025), "
         f"{src.start} to {src.end}", 10, INK2),
        (f"Injury episodes built: {len(ep):,}    Player-name match rate: {inj.PLAYER_ID.notna().mean():.1%}", 10, INK2),
        ("", 8, INK),
        ("Proxy-tier time-loss events (primary target)", 12, INK),
        (f"    Train (2016-17 to 2021-22): {tr}", 11, INK),
        (f"    Validation (2022-23): {va}", 11, INK),
        (f"    Test (2023-24): {te}", 11, INK),
        ("", 8, INK),
        (f"About {tr / 6:.0f} events per training season (6 seasons)", 10, INK2),
    ]
    fig = plt.figure(figsize=(8.5, 4))
    y = 0.92
    for text, size, color in lines:
        fig.text(0.06, y, text, fontsize=size, color=color, family="DejaVu Sans")
        y -= 0.075 if text else 0.04
    return fig


def make_audit(rows, n=100, seed=0):
    pool = rows.copy()
    # Stratify so every tier is represented (target tiers oversampled a little).
    weights = pool.tier.map({"strict": 3, "proxy": 3, "broad": 1.5, "other_injury": 1,
                             "uncertain": 1, "non_injury": 1}).fillna(1)
    s = pool if len(pool) <= n else pool.sample(n=n, weights=weights, random_state=seed)
    out = pd.DataFrame({
        "row_id": s.index, "date": s.Date.dt.date, "player": s.Relinquished, "note": s.Notes,
        "assigned_tier": s.tier, "assigned_site": s.site,
        "correct": "", "correct_tier": "", "comment": "",
    })
    out.to_csv(AUDIT, index=False)
    return out


def score():
    a = pd.read_csv(AUDIT)
    a = a[a.correct.astype(str).str.strip().isin(["0", "1", "0.0", "1.0"])]
    if a.empty:
        raise SystemExit("No rows marked yet: fill the `correct` column with 1/0 first.")
    a["correct"] = a.correct.astype(float).astype(int)
    by = a.groupby("assigned_tier").correct.agg(["count", "mean"]).rename(
        columns={"count": "checked", "mean": "agreement"})
    by["agreement"] = by.agreement.map(lambda v: f"{v:.0%}")
    wrong = a[a.correct == 0][["note", "assigned_tier", "correct_tier"]]
    md = ["# Label audit results\n",
          f"Rows checked: **{len(a)}**. Agreement: **{a.correct.sum()}/{len(a)} = {a.correct.mean():.0%}**\n",
          "## By assigned tier\n",
          "\n".join(["| tier | checked | agreement |", "|---|---|---|"] +
                    [f"| {t} | {r.checked} | {r.agreement} |" for t, r in by.iterrows()]), "",
          f"## Misclassified rows ({len(wrong)})\n",
          "\n".join(["| note | assigned | correct |", "|---|---|---|"] +
                    [f"| {r.note} | {r.assigned_tier} | {r.correct_tier} |" for r in wrong.itertuples()])]
    p = OUT / "label_audit_results.md"
    p.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))
    print(f"\nsaved {p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score", action="store_true")
    args = ap.parse_args()
    if args.score:
        return score()

    ep, rows, per = _load()
    figs = {"fig1_events_per_season.png": fig_events(per),
            "fig2_reporting_change.png": fig_reporting(per),
            "fig3_events_by_site.png": fig_sites(ep)}
    for name, f in figs.items():
        f.savefig(OUT / name, bbox_inches="tight")
    with PdfPages(OUT / "evidence_summary.pdf") as pdf:
        pdf.savefig(summary_page(ep, rows, per))
        for f in figs.values():
            pdf.savefig(f, bbox_inches="tight")
    plt.close("all")

    if AUDIT.exists():
        print(f"{AUDIT.name} already exists - not overwriting your audit (delete it to resample)")
    else:
        make_audit(rows)
        print(f"wrote {AUDIT.name}: 100 rows to check by hand")
    print(f"evidence pack -> {OUT}")


if __name__ == "__main__":
    main()
