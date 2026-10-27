"""Game-sequence inputs for the Transformer (proposal section 4F).

For every landmark (same rows and order as landmarks.csv.gz) this stores the
player's previous N_MAX games, all strictly before the landmark date:

    seq_idx  [landmarks, N_MAX]  row in game_X of each previous game, oldest -> newest,
                                 -1 = padding (player has fewer prior games)
    seq_dt   [landmarks, N_MAX]  log(1 + days between that game and the landmark)
    game_X   [games, F_game]     per-game step features (standardized):
                                 that game's minutes, possessions used, FGA, FTA,
                                 rebounds, fouls, plus its schedule/travel
                                 (team rest, back-to-back, trip km, time-zone shift,
                                 home, playoffs) and the gap since the player's
                                 previous game
    static_X [landmarks, F_static]  the tabular model features (standardized,
                                 missing indicators), with static_family naming the
                                 feature family of each column (for ablations)

Sequences may reach into the previous season (proposal: earlier games may be
used when they would genuinely have been available). All scaling statistics
come from training-season rows only.

Usage:  python -m src.sequence.build
Writes: data/processed/sequence/sequences.npz
"""
import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, TRAIN_SEASONS
from src.features.workload import HISTORY_FEATURES, WORKLOAD_FEATURES
from src.models.baselines import Preprocessor
from src.survival.dataset import BIO_FEATURES, CONTEXT_FEATURES, LANDMARKS, SCHEDULE_FEATURES, model_features

OUT_DIR = PROCESSED / "sequence"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT = OUT_DIR / "sequences.npz"
N_MAX = 20

GAME_STATS = ["MIN", "poss", "FGA", "FTA", "REB", "PF"]
GAME_SCHEDULE = ["days_rest", "back_to_back", "travel_km", "abs_tz_shift_h", "is_home", "is_playoffs",
                 "gap_days"]
LOG_COLS = ["travel_km", "days_rest", "gap_days"]          # heavy-tailed -> log1p


def family_of(col):
    base = col.removesuffix("_missing")
    for fam, cols in (("workload", WORKLOAD_FEATURES), ("history", HISTORY_FEATURES),
                      ("bio", BIO_FEATURES), ("context", CONTEXT_FEATURES)):
        if base in cols:
            return fam
    if base in ("travel_km", "abs_tz_shift_h", "tz_shift_h", "km_last_7d", "km_last_14d"):
        return "travel"
    if base in SCHEDULE_FEATURES:
        return "schedule"
    return "other"


def game_table():
    gl = pd.read_csv(RAW / "gamelogs_all.csv",
                     usecols=["PLAYER_ID", "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION",
                              "MIN", "FGA", "FTA", "TOV", "REB", "PF"])
    gl["GAME_DATE"] = pd.to_datetime(gl.GAME_DATE)
    tg = pd.read_csv(PROCESSED / "travel_team_games.csv", usecols=["GAME_ID", "TEAM", "days_rest", "back_to_back",
                                                                    "travel_km", "tz_shift_h", "is_home"])
    tg["abs_tz_shift_h"] = tg.tz_shift_h.abs()
    g = gl.merge(tg.drop(columns="tz_shift_h"), left_on=["GAME_ID", "TEAM_ABBREVIATION"],
                 right_on=["GAME_ID", "TEAM"], how="left")
    g = g.sort_values(["PLAYER_ID", "GAME_DATE"]).reset_index(drop=True)
    g["MIN"] = g.MIN.clip(lower=0).astype(float)
    g["poss"] = g.FGA + 0.44 * g.FTA + g.TOV
    g["is_playoffs"] = g.SEASON_TYPE.isin(["Playoffs", "PlayIn"]).astype(float)
    g["gap_days"] = (g.GAME_DATE - g.groupby("PLAYER_ID").GAME_DATE.shift()).dt.days.clip(upper=365)
    for c in ("back_to_back", "is_home"):
        g[c] = g[c].astype(float)
    return g


def build():
    lm = pd.read_csv(LANDMARKS)
    lm["GAME_DATE"] = pd.to_datetime(lm.GAME_DATE)
    g = game_table()

    # ---- per-game step features, scaled on training-season games -------------
    cols = GAME_STATS + GAME_SCHEDULE
    X = g[cols].astype(float).copy()
    for c in LOG_COLS:
        X[c] = np.log1p(X[c].clip(lower=0))
    tr = g.SEASON.isin(TRAIN_SEASONS)
    mu, sd = X[tr].mean(), X[tr].std().replace(0, 1)
    game_X = ((X - mu) / sd).fillna(0.0).to_numpy(np.float32)

    # ---- windows: previous N_MAX games of the same player ---------------------
    pos = g.groupby("PLAYER_ID").cumcount().to_numpy()          # position within player
    start = (np.arange(len(g)) - pos)                           # first row of that player
    row_of = pd.Series(np.arange(len(g)), index=pd.MultiIndex.from_frame(g[["PLAYER_ID", "GAME_ID"]]))
    lm_row = row_of.reindex(pd.MultiIndex.from_frame(lm[["PLAYER_ID", "GAME_ID"]])).to_numpy()
    if np.isnan(lm_row.astype(float)).any():
        raise SystemExit("some landmarks have no matching game-log row")
    lm_row = lm_row.astype(np.int64)
    offs = np.arange(-N_MAX, 0)                                 # previous games, oldest first
    idx = lm_row[:, None] + offs[None, :]
    valid = idx >= start[lm_row][:, None]
    seq_idx = np.where(valid, idx, -1).astype(np.int32)
    gdates = g.GAME_DATE.to_numpy()
    dt_days = (lm.GAME_DATE.to_numpy()[:, None] - gdates[np.clip(idx, 0, None)]) / np.timedelta64(1, "D")
    seq_dt = np.where(valid, np.log1p(np.clip(dt_days, 0, None)), 0).astype(np.float32)

    # ---- static (tabular) features --------------------------------------------
    feats = model_features(lm)
    pre = Preprocessor().fit(lm[lm.SEASON.isin(TRAIN_SEASONS)], feats)
    static_X, static_names = pre.transform(lm)
    static_family = np.array([family_of(c) for c in static_names])

    np.savez_compressed(
        OUT, game_X=game_X, game_feature_names=np.array(cols), seq_idx=seq_idx, seq_dt=seq_dt,
        static_X=static_X.astype(np.float32), static_names=np.array(static_names), static_family=static_family,
        time=lm.time.to_numpy(np.float32), event=lm.event.to_numpy(np.int8), y14=lm.y14.to_numpy(np.int8),
        evaluable14=lm.evaluable14.to_numpy(np.int8), split=lm.split.to_numpy().astype(str),
        season=lm.SEASON.to_numpy().astype(str), player_id=lm.PLAYER_ID.to_numpy(np.int64),
        game_id=lm.GAME_ID.to_numpy(np.int64), early_era=lm.early_era.to_numpy(np.int8))
    n_hist = (seq_idx >= 0).sum(1)
    print(f"{len(lm):,} landmarks x {N_MAX} previous games; {game_X.shape[1]} step features; "
          f"{static_X.shape[1]} static columns -> {OUT}")
    print(f"previous games available: median {np.median(n_hist):.0f}; "
          f"full {N_MAX}-game history for {np.mean(n_hist == N_MAX):.0%} of landmarks; "
          f"none for {np.mean(n_hist == 0):.1%}")
    print("static families:", pd.Series(static_family).value_counts().to_dict())
    return g, lm


if __name__ == "__main__":
    build()
