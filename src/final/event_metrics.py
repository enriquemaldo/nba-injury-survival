"""Event-prediction metrics at 14 days (proposal section 8B), from the final, locked
test-season predictions. No model is refit and nothing here can change the models.

    PR-AUC (average precision), threshold-free, with a 95% CI (bootstrap over players)
    precision, recall and F2 when the model flags a fixed share of player-games
    (the top 5%, 10% or 20% by predicted risk)

Fixed alert rates are used instead of a tuned probability threshold: tuning a
threshold would need data the final models were not trained on, and every season
before the test season was used for training. Only player-games whose 14-day
status is known are used (follow-up of at least 14 days, or an injury before then).

Usage:  python -m src.final.event_metrics
Writes: data/processed/final/event_metrics.md, event_metrics.csv, fig_pr_curves.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve

from src.final.evaluate import H, LABELS, OUT, SETS
from src.models.baselines import _md
from src.models.metrics import cluster_bootstrap, horizon_label

MODELS = ["cox", "logistic", "xgboost", "deepsurv", "tf20", "tf5"]
ALERT_RATES = (0.05, 0.10, 0.20)
INK, INK2 = "#0b0b0b", "#52514e"


def alert_metrics(y, risk, rate):
    """Flag the top `rate` share by risk (ties broken by order); precision, recall, F2."""
    n_flag = max(1, int(round(rate * len(y))))
    flag = np.zeros(len(y), bool)
    flag[np.argsort(-risk, kind="stable")[:n_flag]] = True
    tp = int((flag & (y == 1)).sum())
    precision = tp / n_flag
    recall = tp / max(int(y.sum()), 1)
    f2 = 5 * precision * recall / (4 * precision + recall) if precision + recall > 0 else 0.0
    return precision, recall, f2


def evaluate(df):
    known, y_all = horizon_label(df.time.to_numpy(), df.event.to_numpy(), H)
    d = df[known].reset_index(drop=True)
    y = y_all[known]
    pid = d.PLAYER_ID.to_numpy()
    rows = []
    for k in MODELS:
        col = f"{k}_risk"
        if col not in d:
            continue
        risk = d[col].to_numpy(float)
        ap = average_precision_score(y, risk)
        lo, hi = cluster_bootstrap(lambda i: average_precision_score(y[i], risk[i]) if y[i].any() else np.nan, pid)
        r = {"model": LABELS.get(k, k), "key": k, "pr_auc": ap, "pr_auc_ci": f"{lo:.3f}–{hi:.3f}",
             "prevalence": float(y.mean())}
        for rate in ALERT_RATES:
            p, rc, f2 = alert_metrics(y, risk, rate)
            tag = f"top{int(rate * 100)}"
            r[f"precision_{tag}"], r[f"recall_{tag}"], r[f"f2_{tag}"] = p, rc, f2
        rows.append(r)
    return pd.DataFrame(rows), d, y


def fig_pr(d, y, path):
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    for k, col, lw in (("cox", "#8a8984", 2), ("tf5", "#1f5fae", 2.4)):
        if f"{k}_risk" not in d:
            continue
        p, r, _ = precision_recall_curve(y, d[f"{k}_risk"].to_numpy(float))
        ax.plot(r, p, color=col, lw=lw, label=f"{LABELS[k]} (PR-AUC {average_precision_score(y, d[f'{k}_risk']):.3f})")
    ax.axhline(y.mean(), color=INK2, ls="--", lw=1, label=f"No-skill (prevalence {100 * y.mean():.1f}%)")
    ax.set_xlabel("Recall (share of injuries flagged)")
    ax.set_ylabel("Precision (share of flags that were injuries)")
    ax.set_title(f"Precision-recall, {H}-day injury, test season 2023-24", loc="left", color=INK, fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, None)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#e6e5e1")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    P = pd.read_csv(OUT / "test_predictions.csv.gz")
    out, md = [], ["# Event-prediction metrics at 14 days\n",
                   "Computed from the locked final predictions (no refitting). Only player-games whose 14-day "
                   "status is known are used. Precision, recall and F2 are reported when the model flags a fixed "
                   "share of player-games (top 5%, 10%, 20% by predicted risk); no threshold was tuned. "
                   "PR-AUC 95% CIs come from 200 bootstrap resamples of players. Compare PR-AUC with the "
                   "prevalence, which is what a model with no skill would score.\n"]
    for s, title in SETS.items():
        df = P[P.split == s]
        if df.empty:
            continue
        tab, d, y = evaluate(df)
        tab.insert(0, "set", s)
        out.append(tab)
        show = tab[["model", "pr_auc", "pr_auc_ci", "prevalence"] +
                   [c for c in tab.columns if c.startswith(("precision_", "recall_", "f2_"))]]
        md += [f"## {title}\n", f"{len(d):,} player-games with known 14-day status, {int(y.sum()):,} injured "
               f"within 14 days (prevalence {100 * y.mean():.2f}%).\n", _md(show.round(3)), ""]
        if s == "test":
            fig_pr(d, y, OUT / "fig_pr_curves.png")
    pd.concat(out).to_csv(OUT / "event_metrics.csv", index=False)
    md += ["![Precision-recall curves](fig_pr_curves.png)\n"]
    (OUT / "event_metrics.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
