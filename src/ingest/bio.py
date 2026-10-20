"""Player bio by season: age, height, weight, draft year (nba_api).

One request per season (LeagueDashPlayerBioStats), cached like the game logs.
Writes data/raw/player_bio.csv with one row per PLAYER_ID x SEASON:
    PLAYER_ID, SEASON, age, height_in, weight_lb, draft_year

If the endpoint fails or a column is missing, the pipeline still runs; the
survival dataset simply leaves those player-level features out.

Usage:  python -m src.ingest.bio
"""
import time

import pandas as pd

from src.config import RAW, SEASONS

OUT_DIR = RAW / "bio"
OUT_DIR.mkdir(exist_ok=True)
OUT = RAW / "player_bio.csv"

COLMAP = {"PLAYER_ID": "PLAYER_ID", "AGE": "age", "PLAYER_HEIGHT_INCHES": "height_in",
          "PLAYER_WEIGHT": "weight_lb", "DRAFT_YEAR": "draft_year"}


def fetch(season: str) -> pd.DataFrame:
    from nba_api.stats.endpoints import leaguedashplayerbiostats

    for attempt in range(4):
        try:
            return leaguedashplayerbiostats.LeagueDashPlayerBioStats(
                season=season, season_type_all_star="Regular Season", timeout=60,
            ).get_data_frames()[0]
        except Exception as e:  # stats.nba.com is flaky
            wait = 10 * (attempt + 1)
            print(f"  {season}: {e!r}; retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"failed to fetch bio for {season}")


def main():
    for season in SEASONS:
        path = OUT_DIR / f"{season}.csv"
        if path.exists():
            print(f"skip {path.name}")
            continue
        try:
            df = fetch(season)
        except RuntimeError as e:
            print(f"  WARNING: {e}; continuing")
            continue
        df.to_csv(path, index=False)
        print(f"saved {path.name}: {len(df)} players")
        time.sleep(2)

    frames = []
    for p in sorted(OUT_DIR.glob("*.csv")):
        df = pd.read_csv(p)
        keep = {c: n for c, n in COLMAP.items() if c in df.columns}
        missing = sorted(set(COLMAP) - set(keep))
        if missing:
            print(f"  note: {p.name} has no {missing}")
        df = df[list(keep)].rename(columns=keep)
        df["SEASON"] = p.stem
        frames.append(df)
    if not frames:
        raise SystemExit("no bio files downloaded")
    bio = pd.concat(frames, ignore_index=True)
    for c in ("age", "height_in", "weight_lb", "draft_year"):
        if c in bio:
            bio[c] = pd.to_numeric(bio[c], errors="coerce")   # "Undrafted" -> NaN
    bio = bio.drop_duplicates(["PLAYER_ID", "SEASON"])
    bio.to_csv(OUT, index=False)
    print(f"{len(bio):,} player-seasons -> {OUT.name}")
    print(bio.describe().round(1).to_string())


if __name__ == "__main__":
    main()
