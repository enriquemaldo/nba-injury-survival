"""Travel and schedule features for every team-game.

Reference tables (tracked in git, hand-verified):
    reference/arenas.csv         one row per team per arena period
                                 (first_season/last_season blank = open-ended)
    reference/neutral_sites.csv  games not played at the home team's arena:
                                 the 2020 bubble (a date range, any teams),
                                 international games and NBA Cup games in Las Vegas

Venue rule for each game:
    1. a matching neutral-site row (date in range, and the two teams match
       when the row names them) -> that venue
    2. otherwise -> the home team's arena for that season
       (home team comes from MATCHUP: "LAL vs. BOS" = LAL home, "LAL @ BOS" = BOS home)

Features (per team, games in date order, reset at each season's first game):
    travel_km         great-circle distance from the previous game's venue
                      (season opener: from the team's own arena)
    tz_shift_h        change in UTC offset vs. the previous venue (+ = moved east)
    days_rest         days since the team's previous game (NaN for the opener)
    back_to_back      days_rest == 1
    km_last_7d/14d    travel_km summed over games in the last 7 / 14 days (incl. this one)
    games_last_7d     team games in the last 7 days (incl. this one)

All features use only the schedule up to and including the game itself, which
is known in advance, so there is no outcome leakage. Player-games inherit
their team's features by joining on GAME_ID + TEAM_ABBREVIATION.

Usage:  python -m src.features.travel
Writes: data/processed/travel_team_games.csv, data/processed/travel_report.md
"""
from math import asin, cos, radians, sin, sqrt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import PROCESSED, RAW, ROOT

REF = ROOT / "reference"
OUT = PROCESSED / "travel_team_games.csv"
REPORT = PROCESSED / "travel_report.md"


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    a = sin((lat2 - lat1) / 2) ** 2 + cos(lat1) * cos(lat2) * sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * asin(sqrt(a))


def utc_offset_h(tz: str, date) -> float:
    """UTC offset in hours at local noon on `date` (handles DST and Arizona)."""
    noon = pd.Timestamp(date).to_pydatetime().replace(hour=12, tzinfo=ZoneInfo(tz))
    return noon.utcoffset().total_seconds() / 3600


def load_reference():
    arenas = pd.read_csv(REF / "arenas.csv", dtype={"first_season": str, "last_season": str})
    arenas["first_season"] = arenas.first_season.fillna("0000-00")
    arenas["last_season"] = arenas.last_season.fillna("9999-99")
    neutral = pd.read_csv(REF / "neutral_sites.csv", parse_dates=["start_date", "end_date"],
                          dtype={"team_a": str, "team_b": str})
    return arenas, neutral


def team_games(gamelogs: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game, with home team and opponent parsed from MATCHUP."""
    tg = (gamelogs[["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION", "MATCHUP"]]
          .drop_duplicates(["GAME_ID", "TEAM_ABBREVIATION"]).copy())
    tg["GAME_DATE"] = pd.to_datetime(tg.GAME_DATE)
    tg["is_home"] = tg.MATCHUP.str.contains(" vs. ", regex=False)
    tg["OPP"] = tg.MATCHUP.str.split().str[-1]
    tg["HOME_TEAM"] = np.where(tg.is_home, tg.TEAM_ABBREVIATION, tg.OPP)
    return tg.rename(columns={"TEAM_ABBREVIATION": "TEAM"})


def home_arena(arenas: pd.DataFrame, team: str, season: str) -> pd.Series | None:
    m = arenas[(arenas.team == team) & (arenas.first_season <= season) & (arenas.last_season >= season)]
    return m.iloc[0] if len(m) == 1 else None


def assign_venues(tg: pd.DataFrame, arenas: pd.DataFrame, neutral: pd.DataFrame):
    """Add venue columns; return (team-games, neutral-site match log)."""
    tg = tg.copy()
    tg["neutral_row"] = -1
    for i, r in neutral.iterrows():
        m = tg.GAME_DATE.between(r.start_date, r.end_date)
        if pd.notna(r.team_a):
            pair = {r.team_a, r.team_b}
            m &= tg.TEAM.isin(pair) & tg.OPP.isin(pair)
        tg.loc[m & (tg.neutral_row < 0), "neutral_row"] = i

    cols = ["venue", "city", "lat", "lon", "tz"]
    keyed = {}
    for (team, season), _ in tg.groupby(["HOME_TEAM", "SEASON"]):
        a = home_arena(arenas, team, season)
        if a is not None:
            keyed[(team, season)] = (a.arena, a.city, a.lat, a.lon, a.tz)
    home_vals = [keyed.get((h, s), (None,) * 5) for h, s in zip(tg.HOME_TEAM, tg.SEASON)]
    tg[cols] = pd.DataFrame(home_vals, index=tg.index, columns=cols)

    is_neu = tg.neutral_row >= 0
    nv = neutral.loc[tg.loc[is_neu, "neutral_row"], ["venue", "city", "lat", "lon", "tz"]]
    tg.loc[is_neu, cols] = nv.to_numpy()
    tg["is_neutral"] = is_neu

    log = []
    for i, r in neutral.iterrows():
        n = int((tg.neutral_row == i).sum())
        label = (f"{r.start_date:%Y-%m-%d}–{r.end_date:%Y-%m-%d} all games" if pd.isna(r.team_a)
                 else f"{r.start_date:%Y-%m-%d} {r.team_a}-{r.team_b}")
        expected = "≥1" if pd.isna(r.team_a) else "2"
        ok = n >= 1 if pd.isna(r.team_a) else n == 2
        if not ok and "not a regular-season game" in str(r.note):
            status = "not in game logs (expected: Cup final doesn't count as a regular-season game)"
        else:
            status = "OK" if ok else "NOT FOUND - check date/teams"
        log.append({"game": label, "venue": f"{r.venue}, {r.city}", "team_games_matched": n,
                    "expected": expected, "status": status})
    return tg.drop(columns="neutral_row"), pd.DataFrame(log)


def travel_features(tg: pd.DataFrame, arenas: pd.DataFrame) -> pd.DataFrame:
    tg = tg.sort_values(["TEAM", "GAME_DATE"]).reset_index(drop=True)
    tg["tz_offset_h"] = [utc_offset_h(z, d) for z, d in zip(tg.tz, tg.GAME_DATE)]

    prev_lat = tg.groupby(["TEAM", "SEASON"]).lat.shift()
    prev_lon = tg.groupby(["TEAM", "SEASON"]).lon.shift()
    prev_off = tg.groupby(["TEAM", "SEASON"]).tz_offset_h.shift()
    prev_date = tg.groupby(["TEAM", "SEASON"]).GAME_DATE.shift()

    # Season opener: the trip starts from the team's own arena.
    first = prev_lat.isna()
    for i in tg.index[first]:
        a = home_arena(arenas, tg.at[i, "TEAM"], tg.at[i, "SEASON"])
        if a is not None:
            prev_lat[i], prev_lon[i] = a.lat, a.lon
            prev_off[i] = utc_offset_h(a.tz, tg.at[i, "GAME_DATE"])

    tg["travel_km"] = [haversine_km(a, b, c, d) if pd.notna(a) else np.nan
                       for a, b, c, d in zip(prev_lat, prev_lon, tg.lat, tg.lon)]
    tg["tz_shift_h"] = tg.tz_offset_h - prev_off
    tg["days_rest"] = (tg.GAME_DATE - prev_date).dt.days
    tg["back_to_back"] = tg.days_rest == 1

    def rolling(g):
        s = g.set_index("GAME_DATE")
        return pd.DataFrame({
            "km_last_7d": s.travel_km.rolling("7D").sum().to_numpy(),
            "km_last_14d": s.travel_km.rolling("14D").sum().to_numpy(),
            "games_last_7d": s.travel_km.rolling("7D").count().to_numpy(),
        }, index=g.index)

    roll = pd.concat([rolling(g) for _, g in tg.groupby(["TEAM", "SEASON"])])
    tg = tg.join(roll)
    for c in ("travel_km", "km_last_7d", "km_last_14d"):
        tg[c] = tg[c].round(1)
    return tg


def _md(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r) + " |")
    return "\n".join(lines)


def report(tg: pd.DataFrame, log: pd.DataFrame) -> str:
    missing = tg[tg.venue.isna()]
    reg = tg[tg.SEASON_TYPE == "Regular Season"]
    per_season = (reg.groupby("SEASON")
                  .agg(team_games=("GAME_ID", "size"),
                       mean_km_per_game=("travel_km", "mean"),
                       back_to_backs=("back_to_back", "sum"),
                       neutral_team_games=("is_neutral", "sum"))
                  .round(1).reset_index())
    team_season = (reg.groupby(["TEAM", "SEASON"]).travel_km.sum().div(1000).round(1)
                   .rename("total_1000km").reset_index()
                   .sort_values(["total_1000km", "TEAM", "SEASON"], ascending=[False, True, True]))
    checks = [
        ("Raptors 2020-21 home games in Tampa",
         tg[(tg.HOME_TEAM == "TOR") & (tg.SEASON == "2020-21") & ~tg.is_neutral].city.unique().tolist()),
        ("Clippers 2024-25 home games",
         tg[(tg.HOME_TEAM == "LAC") & (tg.SEASON == "2024-25")].venue.unique().tolist()),
        ("Warriors home arena 2018-19 → 2019-20",
         [tg[(tg.HOME_TEAM == "GSW") & (tg.SEASON == s)].venue.mode().iat[0] for s in ("2018-19", "2019-20")]),
    ]
    md = ["# Travel features report\n",
          f"Team-games: **{len(tg):,}**. Venue assigned: **{tg.venue.notna().mean():.1%}**"
          f" ({len(missing)} missing).\n",
          "## Neutral-site games found in the game logs\n", _md(log), "",
          "## Spot checks\n"] + [f"- {k}: {v}" for k, v in checks] + [
          "", "## Regular-season travel by season\n", _md(per_season), "",
          "## Most travel (team-season, thousands of km)\n", _md(team_season.head(5)), "",
          "## Least travel\n", _md(team_season.tail(5)), ""]
    if len(missing):
        md += ["## Team-games with no venue (fix reference/arenas.csv)\n",
               _md(missing[["GAME_DATE", "SEASON", "TEAM", "MATCHUP"]].head(20)), ""]
    return "\n".join(md)


def main():
    arenas, neutral = load_reference()
    gl = pd.read_csv(RAW / "gamelogs_all.csv",
                     usecols=["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION", "MATCHUP"])
    tg, log = assign_venues(team_games(gl), arenas, neutral)
    tg = travel_features(tg, arenas)
    keep = ["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM", "OPP", "HOME_TEAM", "is_home",
            "is_neutral", "venue", "city", "lat", "lon", "tz", "travel_km", "tz_shift_h", "days_rest",
            "back_to_back", "km_last_7d", "km_last_14d", "games_last_7d"]
    tg[keep].to_csv(OUT, index=False)
    md = report(tg, log)
    REPORT.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nsaved {OUT.name} and {REPORT.name} in {PROCESSED}")


if __name__ == "__main__":
    main()
