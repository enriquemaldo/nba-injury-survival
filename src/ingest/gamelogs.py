"""Step 1: download every player-game (regular season, play-in, playoffs) from nba_api.

LeagueGameLog returns a whole season of player-games in one request, so the
full 10-season pull is ~20 requests. Raw CSVs are cached; re-running skips
seasons already downloaded.

Usage:  python -m src.ingest.gamelogs
"""
import time

import pandas as pd

from src.config import RAW, SEASONS

OUT = RAW / "gamelogs"
OUT.mkdir(exist_ok=True)


def fetch_season(season: str, season_type: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguegamelog

    for attempt in range(4):
        try:
            df = leaguegamelog.LeagueGameLog(
                season=season,
                season_type_all_star=season_type,
                player_or_team_abbreviation="P",
                timeout=60,
            ).get_data_frames()[0]
            df["SEASON"] = season
            df["SEASON_TYPE"] = season_type
            return df
        except Exception as e:  # stats.nba.com is flaky; back off and retry
            wait = 10 * (attempt + 1)
            print(f"  {season} {season_type}: {e!r}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"Failed to fetch {season} {season_type}")


# The play-in tournament (first held Aug 2020) is its own season type in the
# NBA stats API, separate from "Regular Season" and "Playoffs".
PLAYIN_FROM = "2019-20"


def main():
    for season in SEASONS:
        stypes = ["Regular Season", "Playoffs"] + (["PlayIn"] if season >= PLAYIN_FROM else [])
        for stype in stypes:
            path = OUT / f"{season}_{stype.replace(' ', '')}.csv"
            if path.exists():
                print(f"skip {path.name}")
                continue
            try:
                df = fetch_season(season, stype)
            except RuntimeError:
                if stype != "PlayIn":
                    raise
                print(f"  WARNING: could not fetch {season} play-in games; continuing without them")
                continue
            df.to_csv(path, index=False)
            print(f"saved {path.name}: {len(df):,} player-games")
            time.sleep(2)

    frames = [f for f in (pd.read_csv(p) for p in sorted(OUT.glob("*.csv"))) if len(f)]
    allg = pd.concat(frames, ignore_index=True)
    allg["GAME_DATE"] = pd.to_datetime(allg["GAME_DATE"])
    allg.to_csv(RAW / "gamelogs_all.csv", index=False)
    print(f"combined: {len(allg):,} rows, {allg.PLAYER_ID.nunique():,} players")

    # Player list (IDs + names) for name matching in Step 3.
    from nba_api.stats.static import players
    pd.DataFrame(players.get_players()).to_csv(RAW / "nba_players.csv", index=False)


if __name__ == "__main__":
    main()
