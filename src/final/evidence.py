"""Update 4 evidence pack: final test-season results and SHAP explanations.

Run after src.final.evaluate and src.explain.shap_values:
    python -m src.final.evidence
Writes data/processed/update4/ (evidence_summary.pdf + copies of the reports and figures)
"""
import json
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from src.config import PROCESSED
from src.explain.labels import FAMILY_LABELS
from src.explain.shap_values import OUT as EXPLAIN
from src.final.evaluate import N_BOOT_PRIMARY, OUT as FINAL
from src.models.compare import image_page

OUT = PROCESSED / "update4"
INK, INK2 = "#0b0b0b", "#52514e"
TF5, COX = "Transformer + DeepSurv (5 games)", "Cox"


def summary_lines(res, fam, chk, top):
    t = res[res.set == "test"].set_index("model")
    cox, tf = t.loc[COX], t.loc[TF5]
    lo, hi = (float(v) for v in tf.paired_95ci.replace(" to ", " ").split())
    verdict = ("The interval is above zero: the Transformer's advantage holds on the test season." if lo > 0 else
               "The interval includes zero: the test season does not confirm an advantage over Cox." if hi > 0 else
               "The interval is below zero: Cox does better on the test season.")
    L = [("NBA injury survival — Update 4: final evaluation and SHAP", 14, INK, "bold"),
         ("Models refit on 2016-17 to 2022-23 and scored once on the sealed 2023-24 test season.", 9.5, INK2, None),
         ("", 0, INK, None),
         ("1. Primary comparison (fixed in Update 3)", 11, INK, "bold"),
         (f"{TF5}: C-index {tf.c_index:.3f} ({tf.c_95ci})", 9.5, INK, None),
         (f"Cox: C-index {cox.c_index:.3f} ({cox.c_95ci})", 9.5, INK, None),
         (f"Difference {tf.c_minus_cox:+.3f}, paired 95% CI {tf.paired_95ci} ({N_BOOT_PRIMARY} player resamples).",
          9.5, INK, None),
         (verdict, 9.5, INK2, None), ("", 0, INK, None),
         ("2. Other results on the test season", 11, INK, "bold")]
    for m, r in t.iterrows():
        if m not in (COX, TF5):
            L.append((f"{m}: C-index {r.c_index:.3f} ({r.c_95ci}); vs Cox {r.c_minus_cox:+.3f} ({r.paired_95ci})",
                      9, INK2, None))
    L.append((f"14-day risk, Transformer: mean predicted {100 * tf.mean_pred_14d:.1f}% vs observed "
              f"{100 * tf.observed_14d:.1f}%; Cox: {100 * cox.mean_pred_14d:.1f}% vs {100 * cox.observed_14d:.1f}%.",
              9, INK2, None))
    s = res[res.set == "supplementary"].set_index("model")
    if len(s):
        L.append((f"Supplementary 2024-25 (to Jan 12, 2025): Transformer {s.loc[TF5].c_index:.3f}, "
                  f"Cox {s.loc[COX].c_index:.3f} (difference {s.loc[TF5].c_minus_cox:+.3f}, "
                  f"{s.loc[TF5].paired_95ci}).", 9, INK2, None))
    f = fam[fam.model == "transformer"].sort_values("share", ascending=False)
    fc = fam[fam.model == "cox"].set_index("family").share
    L += [("", 0, INK, None), ("3. SHAP: what drives the Transformer's predictions", 11, INK, "bold")]
    for _, r in f.head(4).iterrows():
        c = f"; Cox {100 * fc[r.family]:.0f}%" if r.family in fc else ""
        L.append((f"{FAMILY_LABELS.get(r.family, r.family)}: {100 * r.share:.0f}% of attribution{c}", 9.5, INK2, None))
    L.append((f"Top single inputs: {', '.join(top.input.head(3))}.", 9.5, INK2, None))
    L.append((f"Additivity check: contributions match the model output with mean gap {chk['mean_abs_gap']:.3f} "
              f"(output SD {chk['sd_of_output']:.2f}).", 9.5, INK2, None))
    return L


def summary_page(lines):
    fig = plt.figure(figsize=(8.5, 11))
    y = 0.95
    for text, size, color, weight in lines:
        if not text:
            y -= 0.012
            continue
        fig.text(0.06, y, text, fontsize=size, color=color, weight=weight or "normal", family="DejaVu Sans")
        y -= 0.034 if weight == "bold" and size >= 11 else 0.026
    return fig


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = pd.read_csv(FINAL / "test_results.csv")
    fam = pd.read_csv(EXPLAIN / "shap_families.csv")
    top = pd.read_csv(EXPLAIN / "shap_top_inputs.csv")
    chk = json.loads((EXPLAIN / "shap_check.json").read_text())
    figs = [FINAL / "fig_test_cindex.png", FINAL / "fig_test_calibration.png", FINAL / "fig_test_km.png",
            EXPLAIN / "fig_shap_families.png", EXPLAIN / "fig_shap_top.png", EXPLAIN / "fig_shap_positions.png",
            EXPLAIN / "fig_shap_dependence.png"]
    with PdfPages(OUT / "evidence_summary.pdf") as pdf:
        pdf.savefig(summary_page(summary_lines(res, fam, chk, top)))
        for f in figs:
            if f.exists():
                pdf.savefig(image_page(f))
    plt.close("all")
    for f in figs + [FINAL / "test_report.md", FINAL / "test_results.csv", EXPLAIN / "shap_report.md",
                     EXPLAIN / "shap_families.csv", EXPLAIN / "shap_top_inputs.csv"]:
        if f.exists():
            shutil.copy(f, OUT / f.name)
    print(f"saved {OUT / 'evidence_summary.pdf'} and the reports/figures it uses")


if __name__ == "__main__":
    main()
