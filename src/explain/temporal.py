"""Temporal feature attribution (proposal sections 4G and 12): which of the previous
games, and which statistic in each game, drive the Transformer's predictions.

Uses the SHAP values already computed by src.explain.shap_values (no model is run):
tf_seq holds one contribution per prediction x previous game x per-game input.

Usage:  python -m src.explain.temporal
Writes: data/processed/explain/temporal_attribution.csv, fig_shap_temporal.png
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from src.explain.labels import SEQ_LABELS
from src.explain.shap_values import OUT

INK, INK2 = "#0b0b0b", "#52514e"


def temporal_matrix(tf_seq, seq_names):
    """Mean |SHAP| and mean SHAP per (input, game), games ordered most recent first."""
    a = tf_seq[:, ::-1, :]                                    # position 0 = most recent game
    n_games = a.shape[1]
    rows = []
    for j, nm in enumerate(seq_names):
        for k in range(n_games):
            v = a[:, k, j]
            rows.append({"input": nm, "label": SEQ_LABELS.get(nm, nm), "games_ago": k + 1,
                         "mean_abs": float(np.abs(v).mean()), "mean": float(v.mean())})
    return pd.DataFrame(rows)


def fig_temporal(tab, path):
    order = (tab.groupby("label").mean_abs.sum().sort_values(ascending=False)).index.tolist()
    n_games = tab.games_ago.max()
    mabs = tab.pivot(index="label", columns="games_ago", values="mean_abs").loc[order]
    mean = tab.pivot(index="label", columns="games_ago", values="mean").loc[order]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 0.42 * len(order) + 1.6), sharey=True)
    cols = [f"{k} game{'s' if k > 1 else ''} ago" for k in range(1, n_games + 1)]
    im0 = axes[0].imshow(mabs.to_numpy(), aspect="auto", cmap="Blues", vmin=0)
    axes[0].set_title("Size of contribution (mean |SHAP|)", loc="left", fontsize=10.5, color=INK)
    lim = float(np.abs(mean.to_numpy()).max()) or 1e-9
    im1 = axes[1].imshow(mean.to_numpy(), aspect="auto", cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim))
    axes[1].set_title("Direction (mean SHAP; red raises risk)", loc="left", fontsize=10.5, color=INK)
    for ax, im in ((axes[0], im0), (axes[1], im1)):
        ax.set_xticks(range(n_games), cols, rotation=0, fontsize=8.5)
        ax.set_yticks(range(len(order)), order, fontsize=9)
        ax.tick_params(length=0, colors=INK2)
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.suptitle("Temporal feature attribution: previous games x per-game inputs (Transformer + DeepSurv)",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    z = np.load(OUT / "shap_values.npz")
    tab = temporal_matrix(z["tf_seq"], list(z["tf_seq_names"]))
    tab.to_csv(OUT / "temporal_attribution.csv", index=False)
    fig_temporal(tab, OUT / "fig_shap_temporal.png")
    by_game = tab.groupby("games_ago").mean_abs.sum()
    by_input = tab.groupby("label").mean_abs.sum().sort_values(ascending=False)
    print("attribution by game (mean |SHAP| summed over inputs):")
    print(by_game.round(4).to_string())
    print("\nattribution by per-game input (summed over games):")
    print(by_input.round(4).to_string())
    print(f"\nsaved {OUT / 'temporal_attribution.csv'} and fig_shap_temporal.png")


if __name__ == "__main__":
    main()
