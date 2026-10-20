"""Player workload, scheduling and injury-history features at each player-game.

Every feature for a player-game uses ONLY information from strictly earlier
games/dates (pre-game landmark). The game's own minutes/stats are never used:
if a player were injured during the game, his low minutes that night would
leak the outcome. Schedule/travel facts about the game itself (rest days,
trip distance, home/away) are known in advance and come from travel.py.

Windows like "last 7 days" mean the calendar days [date-7, date-1].
EWMA follows the proposal: EWMA_t = a*x_t + (1-a)*EWMA_{t-1}, computed over the
player's games in the season, then shifted one game so it ends at the previous game.
"""
import numpy as np
import pandas as pd

EWMA_SHORT = 0.25   # ~ last 7 games
EWMA_LONG = 0.07    # ~ last 28 games
ACWR_MIN_CHRONIC = 60  # minutes in last 28 days needed for a meaningful ACWR

WORKLOAD_FEATURES = [
    "prev_game_min", "days_since_prev_game",
    "min_7d", "min_14d", "min_28d", "poss_7d", "poss_28d", "usage_per_min_28d",
    "games_3d", "games_7d", "games_14d", "b2b_apps_14d",
    "ewma_min_short", "ewma_min_long", "acwr_min", "ewma_ratio_min",
    "consecutive_games", "missed_team_games_14d",
    "season_games_prior", "season_min_prior", "games_prior_total", "min_prior_total",
]
HISTORY_FEATURES = ["prior_target_injuries", "prior_lower_leg_injuries", "prior_any_injuries",
                    "days_since_last_injury"]


def _rolling_left(df, col, window):
    """Sum of `col` over [date-window, date) per player (excludes the game day)."""
    out = (df.set_index("GAME_DATE").groupby("PLAYER_ID")[col]
           .rolling(window, closed="left").sum())
    return out.fillna(0).to_numpy()  # empty window = 0; same order as df (sorted by player, date)


def workload_features(gl: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    """gl: game logs (one row per player-game). team_games: TEAM, SEASON, GAME_DATE per team-game.
    Returns PLAYER_ID, GAME_ID + WORKLOAD_FEATURES."""
    df = gl[["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON", "TEAM_ABBREVIATION",
             "MIN", "FGA", "FTA", "TOV"]].copy()
    df["GAME_DATE"] = pd.to_datetime(df.GAME_DATE)
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    df["MIN"] = df.MIN.clip(lower=0).astype(float)
    df["poss"] = df.FGA + 0.44 * df.FTA + df.TOV          # possessions used (usage load)
    df["one"] = 1.0

    g = df.groupby("PLAYER_ID")
    df["prev_game_min"] = g.MIN.shift()
    df["days_since_prev_game"] = (df.GAME_DATE - g.GAME_DATE.shift()).dt.days
    df["is_b2b_app"] = (df.days_since_prev_game == 1).astype(float)

    for w in ("7D", "14D", "28D"):
        df[f"min_{w[:-1]}d"] = _rolling_left(df, "MIN", w)
    for w in ("7D", "28D"):
        df[f"poss_{w[:-1]}d"] = _rolling_left(df, "poss", w)
    for w in ("3D", "7D", "14D"):
        df[f"games_{w[:-1]}d"] = _rolling_left(df, "one", w)
    df["b2b_apps_14d"] = _rolling_left(df, "is_b2b_app", "14D")
    df["usage_per_min_28d"] = np.where(df.min_28d > 0, df.poss_28d / df.min_28d.where(df.min_28d > 0), np.nan)

    gs = df.groupby(["PLAYER_ID", "SEASON"])
    for name, a in (("ewma_min_short", EWMA_SHORT), ("ewma_min_long", EWMA_LONG)):
        ew = gs.MIN.transform(lambda s: s.ewm(alpha=a, adjust=False).mean())
        df[name] = ew.groupby([df.PLAYER_ID, df.SEASON]).shift()   # end at previous game
    chronic = df.min_28d / 4
    df["acwr_min"] = np.where(df.min_28d >= ACWR_MIN_CHRONIC, df.min_7d / chronic.where(chronic > 0), np.nan)
    df["ewma_ratio_min"] = df.ewma_min_short / df.ewma_min_long.where(df.ewma_min_long > 0)

    df["season_games_prior"] = gs.cumcount()
    df["season_min_prior"] = gs.MIN.cumsum() - df.MIN
    df["games_prior_total"] = g.cumcount()
    df["min_prior_total"] = g.MIN.cumsum() - df.MIN

    # Consecutive team games played (up to the previous game), and team games
    # missed in the last 14 days. Uses the team's schedule.
    tg = team_games[["TEAM", "SEASON", "GAME_DATE"]].drop_duplicates().copy()
    tg["GAME_DATE"] = pd.to_datetime(tg.GAME_DATE)
    tg = tg.sort_values(["TEAM", "GAME_DATE"])
    tg["team_idx"] = tg.groupby(["TEAM", "SEASON"]).cumcount()
    tg["one"] = 1.0
    tg["team_games_14d"] = (tg.set_index("GAME_DATE").groupby("TEAM")["one"]
                            .rolling("14D", closed="left").sum().fillna(0).to_numpy())
    df = df.merge(tg[["TEAM", "GAME_DATE", "team_idx", "team_games_14d"]],
                  left_on=["TEAM_ABBREVIATION", "GAME_DATE"], right_on=["TEAM", "GAME_DATE"], how="left")
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    same = ((df.PLAYER_ID == df.PLAYER_ID.shift()) & (df.SEASON == df.SEASON.shift())
            & (df.TEAM_ABBREVIATION == df.TEAM_ABBREVIATION.shift()))
    cont = same & (df.team_idx - df.team_idx.shift() == 1)
    run_id = (~cont).cumsum()
    df["consecutive_games"] = df.groupby(run_id).cumcount()   # games in the run before this one
    df["missed_team_games_14d"] = (df.team_games_14d - df.games_14d).clip(lower=0)

    return df[["PLAYER_ID", "GAME_ID", "GAME_DATE"] + WORKLOAD_FEATURES]


def injury_history(landmarks: pd.DataFrame, episodes: pd.DataFrame, target_mask: pd.Series) -> pd.DataFrame:
    """Counts of injury episodes with onset strictly before each landmark date.
    landmarks: PLAYER_ID, GAME_DATE (any order). Returns HISTORY_FEATURES aligned to landmarks.index."""
    date_col = "injury_date" if "injury_date" in episodes else "onset_date"
    ep = episodes[["PLAYER_ID", date_col, "is_broad"]].rename(columns={date_col: "onset_date"}).copy()
    ep["onset_date"] = pd.to_datetime(ep.onset_date)
    ep["is_target"] = target_mask.to_numpy()
    ep = ep.sort_values(["PLAYER_ID", "onset_date"])
    gp = ep.groupby("PLAYER_ID")
    ep["prior_any_injuries"] = gp.cumcount() + 1
    ep["prior_target_injuries"] = gp.is_target.cumsum()
    ep["prior_lower_leg_injuries"] = gp.is_broad.cumsum()
    ep = ep.rename(columns={"onset_date": "last_injury_date"})

    lm = landmarks[["PLAYER_ID", "GAME_DATE"]].copy()
    lm["GAME_DATE"] = pd.to_datetime(lm.GAME_DATE)
    lm["_row"] = np.arange(len(lm))
    lm = lm.sort_values("GAME_DATE")
    m = pd.merge_asof(lm, ep.sort_values("last_injury_date"), left_on="GAME_DATE",
                      right_on="last_injury_date", by="PLAYER_ID", allow_exact_matches=False)
    m = m.sort_values("_row")
    out = pd.DataFrame(index=landmarks.index)
    for c in ("prior_target_injuries", "prior_lower_leg_injuries", "prior_any_injuries"):
        out[c] = m[c].fillna(0).to_numpy().astype(float)
    out["days_since_last_injury"] = (m.GAME_DATE - m.last_injury_date).dt.days.to_numpy()
    return out
