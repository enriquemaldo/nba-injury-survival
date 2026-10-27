"""Within-player timing analysis: does recent workload predict WHEN a player gets hurt?

The baseline models rank injury risk mostly by WHO is injury-prone (injury
history, playing time). To isolate timing, this fits a Cox model stratified by
player-season: every player-season gets its own baseline hazard, so a
coefficient can only reflect how risk changes from game to game *within the
same player and season*. Stable traits (history, age, role, team) cancel out.
Player-seasons without a target injury contribute nothing to a stratified Cox
likelihood (as in a conditional / case-crossover design); they are dropped
only for speed. (Cutting each player-season at its injury date would select on
the outcome and is deliberately NOT done.)

Sensitivity: a player returning from injury has just missed games and is
ramping his minutes back up. To check that these return-to-play periods do not
drive the results, the model is refitted without each player's first
RAMP_GAMES games after any injury - a rule that depends only on the past. The same features are also fitted
without player strata ("pooled") to show how much of each association comes
from differences between players.

The feature set is fixed in advance - one variable per concept, to avoid
collinearity and data dredging. `season_games_prior` adjusts for when in the
season a game falls (fatigue and reporting both change over a season).

Training seasons only (2016-17..2021-22). Robust 95% CIs clustered by player.

Usage:  python -m src.analysis.within_player
Writes: data/processed/analysis/within_player.md (+ csv, figure)
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROCESSED, TRAIN_SEASONS
from src.models.baselines import Preprocessor, _md
from src.models.coxph import CoxPH
from src.survival.dataset import LANDMARKS

OUT = PROCESSED / "analysis"
RAMP_GAMES = 10
OUT.mkdir(parents=True, exist_ok=True)

FEATURES = {
    "ewma_min_short": "Recent minutes per game (EWMA, ~7 games)",
    "ewma_ratio_min": "Workload spike (short / long EWMA minutes)",
    "games_7d": "Games played in the last 7 days",
    "back_to_back": "Tonight is a back-to-back",
    "days_since_prev_game": "Days since the player's previous game",
    "missed_team_games_14d": "Team games missed in the last 14 days",
    "km_last_7d": "Team travel in the last 7 days (km)",
    "abs_tz_shift_h": "Time-zone change for tonight's game (h)",
    "is_home": "Home game",
    "is_playoffs": "Playoff / play-in game",
    "season_games_prior": "Games already played this season (control)",
}


def fit(train, strata=None):
    feats = list(FEATURES)
    pre = Preprocessor().fit(train, feats)
    X, names = pre.transform(train)
    m = CoxPH(penalizer=1.0).fit(X, train.time, train.event, strata=strata, cluster=train.PLAYER_ID)
    return pd.DataFrame(m.summary(names)), m


def fig(tab, path):
    d = tab[tab.feature.isin(FEATURES)].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.4, 0.45 * len(d) + 1.4))
    y = np.arange(len(d))
    for off, pre, col, lab in ((0.15, "pooled", "#8a8984", "Pooled (between + within players)"),
                               (-0.15, "within", "#2a78d6", "Within player-season")):
        ax.errorbar(d[f"hr_{pre}"], y + off, xerr=[d[f"hr_{pre}"] - d[f"lo_{pre}"], d[f"hi_{pre}"] - d[f"hr_{pre}"]],
                    fmt="o", color=col, ecolor=col, elinewidth=1.5, capsize=2, ms=5, label=lab)
    ax.axvline(1, color="#52514e", lw=1, ls="--")
    ax.set_yticks(y, [FEATURES[f] for f in d.feature])
    ax.set_xlabel("Hazard ratio per 1 SD (95% CI, clustered by player)")
    ax.set_title("Workload and schedule: within-player vs. pooled associations", loc="left", fontsize=11, pad=24)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=8)
    ax.grid(axis="y", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


def main():
    lm = pd.read_csv(LANDMARKS)
    train = lm[lm.SEASON.isin(TRAIN_SEASONS)].reset_index(drop=True)
    train["GAME_DATE"] = pd.to_datetime(train.GAME_DATE)
    stratum = train.PLAYER_ID.astype(str) + "_" + train.SEASON
    has_event = (train.groupby(stratum).event.transform("max") == 1).to_numpy()

    # Games since returning from the most recent injury (past information only).
    train = train.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    stratum = train.PLAYER_ID.astype(str) + "_" + train.SEASON
    has_event = (train.groupby(stratum).event.transform("max") == 1).to_numpy()
    last_inj = train.GAME_DATE - pd.to_timedelta(train.days_since_last_injury, unit="D")
    games_since = np.full(len(train), np.inf)
    for _, idx in train.groupby("PLAYER_ID").indices.items():
        d = train.GAME_DATE.to_numpy()[idx]
        li = last_inj.to_numpy()[idx]
        ok = ~pd.isna(li)
        # player games strictly after the injury date and before this game
        games_since[idx[ok]] = (np.arange(len(idx))[ok] - np.searchsorted(d, li[ok], side="right"))
    ramp = games_since < RAMP_GAMES

    within_rows = train[has_event].reset_index(drop=True)
    within_strata = stratum[has_event].to_numpy()
    keep = has_event & ~ramp
    noramp_rows = train[keep].reset_index(drop=True)
    noramp_strata = stratum[keep].to_numpy()

    pooled, _ = fit(train)
    within, _ = fit(within_rows, strata=within_strata)
    within_noramp, _ = fit(noramp_rows, strata=noramp_strata)
    tab = pooled[["feature"]].copy()
    for name, t in (("pooled", pooled), ("within", within), ("within_noramp", within_noramp)):
        t = t.set_index("feature").reindex(pooled.feature).reset_index()
        tab[f"hr_{name}"], tab[f"lo_{name}"], tab[f"hi_{name}"], tab[f"p_{name}"] = t.hr, t.hr_lo, t.hr_hi, t.p
    tab.to_csv(OUT / "within_player_hazard_ratios.csv", index=False)
    fig(tab, OUT / "fig_within_player.png")

    show = tab[tab.feature.isin(FEATURES)].copy()
    show.insert(1, "description", show.feature.map(FEATURES))
    fmt = lambda v: "<0.001" if v < 0.001 else f"{v:.3f}"
    show["p_pooled"], show["p_within"] = show.p_pooled.map(fmt), show.p_within.map(fmt)
    sig = show[show.p_within.map(lambda v: v == "<0.001" or float(v) < 0.05)]
    md = ["# Within-player timing analysis (training seasons)\n",
          "Cox model stratified by player-season: each player-season has its own baseline hazard, so "
          "estimates reflect game-to-game changes *within* the same player and season. Stable traits "
          "(injury history, age, role) cancel out. `pooled` = same features without player strata.\n",
          f"- Player-seasons contributing (≥1 target injury): **{len(set(within_strata)):,}** "
          f"({len(within_rows):,} landmarks, {int(within_rows.event.sum()):,} event landmarks)",
          f"- Pooled model: {len(train):,} landmarks. Features fixed in advance; hazard ratios per 1 SD; "
          "robust 95% CIs clustered by player.", "",
          _md(show[["feature", "description", "hr_within", "lo_within", "hi_within", "p_within",
                    "hr_pooled", "lo_pooled", "hi_pooled", "p_pooled"]]), "",
          "![Within-player vs pooled](fig_within_player.png)\n",
          "## Associations within player-season with p < 0.05\n",
          ("\n".join(f"- {FEATURES[r.feature]}: HR {r.hr_within:.2f} ({r.lo_within:.2f}–{r.hi_within:.2f})"
                     for r in sig.itertuples()) if len(sig) else "- None."), "",
          f"## Sensitivity: without return-to-play periods\n",
          f"Within-player model refitted without each player's first {RAMP_GAMES} games after any injury "
          f"({len(noramp_rows):,} landmarks, {int(noramp_rows.event.sum()):,} event landmarks). Similar "
          "estimates mean the return-from-injury ramp-up does not drive the main results.\n",
          _md(show.assign(**{c: tab.loc[show.index, c] for c in ("hr_within_noramp", "lo_within_noramp", "hi_within_noramp")})
              [["feature", "hr_within", "lo_within", "hi_within", "hr_within_noramp", "lo_within_noramp",
                "hi_within_noramp"]]), "",
          "## Notes\n",
          "- Hazard ratios are associations, not causal effects. Within-player estimates remove stable "
          "confounders but not time-varying ones (e.g., a minor knock that both reduces minutes and "
          "precedes a reported injury).",
          "- The lower within-player hazard in playoff games may partly reflect reporting: teams make fewer "
          "injured-list moves in the playoffs, and follow-up is cut short when a team is eliminated.",
          f"- {len(FEATURES)} features tested; with this many tests, an isolated p-value near 0.05 can "
          "occur by chance (no multiplicity correction applied)."]
    (OUT / "within_player.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
