"""Data for the Streamlit dashboard (app.py), built from the final predictions and SHAP values.

Run after src.final.evaluate and src.explain.shap_values:
    python -m src.dashboard.prepare
Writes data/processed/dashboard/:
    games.csv.gz     one row per held-out player-game: 14-day risk (Transformer and Cox),
                     multiple of the season average, percentile, what happened next
    drivers.csv.gz   top SHAP drivers of each prediction, with readable labels and values
    gamelog.csv.gz   player game logs for the "previous games" table
    injuries.csv     target injuries in the held-out seasons (date and transaction note)
"""
import warnings

import numpy as np
import pandas as pd

from src.config import CLASSIF_DAYS, INTERIM, PROCESSED, RAW
from src.explain.labels import SEQ_LABELS, seq_family, static_family, static_label
from src.explain.shap_values import OUT as EXPLAIN
from src.final.evaluate import OUT as FINAL
from src.survival.dataset import LANDMARKS, injury_dates, target_masks

OUT = PROCESSED / "dashboard"
N_DRIVERS = 6
H = CLASSIF_DAYS

FMT = {  # how to show a feature's value next to its label
    "min_7d": "{:.0f} min", "min_14d": "{:.0f} min", "min_28d": "{:.0f} min", "prev_game_min": "{:.0f} min",
    "ewma_min_short": "{:.1f} min/game", "ewma_min_long": "{:.1f} min/game", "season_min_prior": "{:,.0f} min",
    "min_prior_total": "{:,.0f} min", "travel_km": "{:,.0f} km", "km_last_7d": "{:,.0f} km",
    "km_last_14d": "{:,.0f} km", "abs_tz_shift_h": "{:.0f} h", "tz_shift_h": "{:+.0f} h", "age": "{:.1f} yrs",
    "height_in": "{:.0f} in", "weight_lb": "{:.0f} lb", "days_since_last_injury": "{:.0f} days",
    "days_since_prev_game": "{:.0f} days", "days_rest": "{:.0f} days", "acwr_min": "{:.2f}",
    "ewma_ratio_min": "{:.2f}", "usage_per_min_28d": "{:.2f}",
}
YES_NO = {"is_home", "is_neutral", "back_to_back", "is_playoffs", "early_era"}
SEQ_RAW = {"MIN": ("MIN", "{:.1f} min/game"), "poss": ("poss", "{:.1f}/game"), "FGA": ("FGA", "{:.1f}/game"),
           "FTA": ("FTA", "{:.1f}/game"), "REB": ("REB", "{:.1f}/game"), "PF": ("PF", "{:.1f}/game"),
           "travel_km": ("travel_km", "{:,.0f} km total"), "back_to_back": ("back_to_back", "{:.0f} of them"),
           "gap_days": ("gap_days", "{:.1f} days apart")}


def value_text(name, v):
    if name.endswith("_missing"):
        return ""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "unknown"
    if name in YES_NO:
        return "yes" if v >= 0.5 else "no"
    f = FMT.get(name)
    if f:
        return f.format(v)
    return f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}"


def previous_games(lm_rows, n_seq):
    """Raw values of the n_seq previous games for each explained landmark (for driver values)."""
    from src.sequence.build import game_table
    from src.sequence.build import OUT as SEQ
    seq_idx = np.load(SEQ)["seq_idx"][lm_rows][:, -n_seq:]
    g = game_table()
    out = {}
    valid = seq_idx >= 0
    for key, (col, _) in SEQ_RAW.items():
        vals = g[col].to_numpy(float)[np.clip(seq_idx, 0, None)]
        vals = np.where(valid, vals, np.nan)
        with warnings.catch_warnings():                 # players with no previous games: NaN
            warnings.simplefilter("ignore", RuntimeWarning)
            out[key] = np.nansum(vals, 1) if key in ("travel_km", "back_to_back") else np.nanmean(vals, 1)
    out["_n"] = valid.sum(1)
    return out, g


def drivers(shap, score, prev, n_seq, n=N_DRIVERS):
    """Long table: the n largest |SHAP| inputs of each prediction (sequence inputs summed over games)."""
    st_names = list(shap["tf_static_names"])
    sq_names = list(shap["tf_seq_names"])
    phi = np.concatenate([shap["tf_static"], shap["tf_seq"].sum(1)], axis=1)
    names = st_names + [f"seq:{s}" for s in sq_names]
    labels = [static_label(s) for s in st_names] + \
             [f"Last {n_seq} games: {SEQ_LABELS.get(s, s)}" for s in sq_names]
    fams = [static_family(s) for s in st_names] + [seq_family(s) for s in sq_names]
    top = np.argsort(-np.abs(phi), axis=1)[:, :n]
    rows = []
    for i in range(len(phi)):
        for r, j in enumerate(top[i]):
            nm = names[j]
            if nm.startswith("seq:"):
                key = nm[4:]
                fmt = SEQ_RAW.get(key, (None, None))[1]
                v = prev[key][i] if key in prev else np.nan
                txt = "" if fmt is None or prev["_n"][i] == 0 or np.isnan(v) else fmt.format(v)
            else:
                base = nm.removesuffix("_missing")
                v = score[base].iat[i] if base in score else np.nan
                txt = value_text(nm, float(v) if pd.notna(v) else np.nan)
            rows.append((i, r + 1, labels[j], fams[j], float(phi[i, j]), txt))
    return pd.DataFrame(rows, columns=["row", "rank", "label", "family", "shap", "value"])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    P = pd.read_csv(FINAL / "test_predictions.csv.gz", parse_dates=["GAME_DATE"])
    shap = np.load(EXPLAIN / "shap_values.npz")
    lm = pd.read_csv(LANDMARKS, parse_dates=["GAME_DATE"])
    if len(shap["game_id"]) != len(P) or not np.array_equal(shap["game_id"], P.GAME_ID.to_numpy()):
        raise SystemExit("SHAP values do not cover the final predictions - rerun python -m src.explain.shap_values "
                         "without --max-rows")
    key = lm.set_index(["PLAYER_ID", "GAME_ID"]).index
    lm_rows = key.get_indexer(pd.MultiIndex.from_frame(P[["PLAYER_ID", "GAME_ID"]]))
    score = lm.iloc[lm_rows].reset_index(drop=True)
    n_seq = int(shap["n_seq"])

    # ---- games ------------------------------------------------------------------
    games = P[["PLAYER_ID", "PLAYER_NAME", "TEAM", "GAME_ID", "GAME_DATE", "SEASON", "split", "time", "event"]].copy()
    games["row"] = np.arange(len(games))
    games["risk"] = P.tf5_prob14
    games["cox_risk"] = P.cox_prob14
    games["season_avg"] = games.groupby("SEASON").risk.transform("mean")
    games["x_avg"] = games.risk / games.season_avg
    games["percentile"] = games.groupby("SEASON").risk.rank(pct=True) * 100
    games["injured_14d"] = ((games.event == 1) & (games.time <= H)).astype(int)
    games["injury_date"] = (games.GAME_DATE + pd.to_timedelta(games.time, unit="D")).where(games.event == 1)
    gl = pd.read_csv(RAW / "gamelogs_all.csv", usecols=["PLAYER_ID", "GAME_ID", "MATCHUP", "MIN", "PTS"])
    games = games.merge(gl[["PLAYER_ID", "GAME_ID", "MATCHUP"]], on=["PLAYER_ID", "GAME_ID"], how="left")
    games.to_csv(OUT / "games.csv.gz", index=False)

    # ---- drivers ----------------------------------------------------------------
    prev, g = previous_games(lm_rows, n_seq)
    drv = drivers(shap, score, prev, n_seq)
    drv.to_csv(OUT / "drivers.csv.gz", index=False)

    # ---- temporal attribution: previous games x per-game inputs, for each prediction ----
    from src.sequence.build import OUT as SEQ
    seq_idx = np.load(SEQ)["seq_idx"][lm_rows][:, -n_seq:]             # oldest -> newest
    np.savez_compressed(OUT / "seq_shap.npz", phi=shap["tf_seq"].astype(np.float16),
                        names=shap["tf_seq_names"])
    mu = gl.drop_duplicates(["PLAYER_ID", "GAME_ID"]).set_index(["PLAYER_ID", "GAME_ID"]).MATCHUP
    matchup = mu.reindex(pd.MultiIndex.from_frame(g[["PLAYER_ID", "GAME_ID"]])).to_numpy()   # g's row order
    r, k = np.nonzero(seq_idx >= 0)
    prev_games = pd.DataFrame({"row": r, "pos": k, "GAME_DATE": g.GAME_DATE.to_numpy()[seq_idx[r, k]],
                               "MATCHUP": matchup[seq_idx[r, k]]})
    prev_games.to_csv(OUT / "seq_games.csv.gz", index=False)

    # ---- game log for the "previous games" table -----------------------------------
    players = games.PLAYER_ID.unique()
    start = games.GAME_DATE.min() - pd.Timedelta(days=60)
    log = g[g.PLAYER_ID.isin(players) & (g.GAME_DATE >= start)]
    log = log.merge(gl[["PLAYER_ID", "GAME_ID", "MATCHUP", "PTS"]], on=["PLAYER_ID", "GAME_ID"], how="left")
    log = log[["PLAYER_ID", "GAME_DATE", "MATCHUP", "MIN", "PTS", "back_to_back", "travel_km", "abs_tz_shift_h"]]
    log.to_csv(OUT / "gamelog.csv.gz", index=False)

    # ---- injuries -----------------------------------------------------------------
    ep = pd.read_csv(INTERIM / "injury_episodes.csv")
    ep["injury_date"] = injury_dates(ep)
    tgt = ep[target_masks(ep)["proxy"].to_numpy()]
    inj = games.loc[games.event == 1, ["PLAYER_ID", "injury_date"]].drop_duplicates()
    inj = inj.merge(tgt[["PLAYER_ID", "injury_date", "site", "notes"]], on=["PLAYER_ID", "injury_date"], how="left")
    inj = inj.drop_duplicates(["PLAYER_ID", "injury_date"])
    inj.to_csv(OUT / "injuries.csv", index=False)
    print(f"dashboard data: {len(games):,} player-games, {games.PLAYER_ID.nunique():,} players, "
          f"{len(inj):,} target injuries -> {OUT}")
    print("start it with:  streamlit run app.py")


if __name__ == "__main__":
    main()
