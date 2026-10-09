"""Steps 3-5 in one run: onset rows -> taxonomy -> player IDs -> episodes.

Usage:
    python -m src.labels.build

Label source: data/raw/mirror_injuries_clean.csv (the 2016-25 Kaggle dataset).
If a successful PST scrape (data/raw/pst_injuries.csv) ever exists, pass
--source pst to use it instead.

Outputs (data/interim/):
    injury_rows_labeled.csv   every onset row with tier, site, PLAYER_ID
    unmatched_names.csv       names to resolve by hand (-> manual_name_map.csv)
    injury_episodes.csv       one row per injury episode  <- main label table
"""
import argparse

import pandas as pd

from src.config import INTERIM, RAW, season_of
from src.labels.episodes import build_episodes
from src.labels.player_match import build_roster, match_names
from src.labels.taxonomy import label


SOURCE_FILES = {
    "mirror": RAW / "mirror_injuries_clean.csv",   # 2016-25 Kaggle dataset (default)
    "pst": RAW / "pst_injuries.csv",               # only if a PST scrape succeeds
}


def load_injuries(source: str, out_dir) -> pd.DataFrame:
    path = SOURCE_FILES[source]
    if not path.exists():
        raise SystemExit(f"{path} not found - run the matching ingest script first.")
    df = pd.read_csv(path, parse_dates=["Date"])
    print(f"loaded {len(df):,} rows from {path.name} "
          f"({df.Date.min():%Y-%m-%d} to {df.Date.max():%Y-%m-%d})")
    # Record which source/coverage produced these labels (read by feasibility.py).
    pd.Series({"source": source, "file": path.name,
               "start": f"{df.Date.min():%Y-%m-%d}", "end": f"{df.Date.max():%Y-%m-%d}"}
              ).to_json(out_dir / "label_source.json")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=list(SOURCE_FILES), default="mirror")
    args = ap.parse_args()

    out_dir = INTERIM
    df = load_injuries(args.source, out_dir)
    # Onsets are rows where a player is RELINQUISHED (goes out). Acquired rows
    # are returns - episodes infer return from game logs instead.
    df = df[df["Relinquished"].fillna("").str.strip() != ""].copy()
    df["season"] = df["Date"].map(season_of)
    df = df.dropna(subset=["season"])
    df = label(df)

    gl = pd.read_csv(RAW / "gamelogs_all.csv", parse_dates=["GAME_DATE"])
    roster = build_roster(gl)
    names = df.rename(columns={"Relinquished": "raw_name"})[["raw_name", "season"]]
    m = match_names(names, roster)
    df = df.merge(m, left_on=["Relinquished", "season"], right_on=["raw_name", "season"], how="left")
    df.to_csv(out_dir / "injury_rows_labeled.csv", index=False)

    # Coaches / team entries aren't players: drop them from the label data.
    df = df[df.match_method != "not_player"]
    injury_rows = df[df.tier != "non_injury"]
    rate = injury_rows.PLAYER_ID.notna().mean()
    print(f"name match rate (injury rows): {rate:.1%}")
    print(injury_rows.groupby(["match_method", "match_pool"], dropna=False).size().to_string())
    um = (injury_rows[injury_rows.PLAYER_ID.isna()]
          .groupby(["raw_name", "season"]).size().rename("n_rows").reset_index()
          .sort_values("n_rows", ascending=False))
    um.to_csv(out_dir / "unmatched_names.csv", index=False)
    print(f"{len(um)} unmatched name/season pairs -> unmatched_names.csv")

    # Audit files: check these by eye before trusting the labels.
    players = pd.read_csv(RAW / "nba_players.csv").rename(
        columns={"id": "PLAYER_ID", "full_name": "matched_name"})[["PLAYER_ID", "matched_name"]] \
        if (RAW / "nba_players.csv").exists() else None
    fz = m[m.match_method == "fuzzy"]
    if players is not None:
        fz = fz.merge(players, on="PLAYER_ID", how="left")
    fz.to_csv(out_dir / "fuzzy_matches.csv", index=False)
    (df[df.tier == "uncertain"].Notes.value_counts().rename("n_rows").rename_axis("note")
       .reset_index().to_csv(out_dir / "uncertain_notes.csv", index=False))
    (df[df.is_proxy].groupby(["site", "Notes"]).size().rename("n_rows").reset_index()
       .sort_values(["site", "n_rows"], ascending=[True, False])
       .to_csv(out_dir / "proxy_notes_by_site.csv", index=False))
    print("audit files -> fuzzy_matches.csv, uncertain_notes.csv, proxy_notes_by_site.csv")

    ep = build_episodes(df, gl)
    ep.to_csv(out_dir / "injury_episodes.csv", index=False)
    print(f"{len(ep):,} episodes -> injury_episodes.csv")
    print(ep.tier.value_counts().to_string())


if __name__ == "__main__":
    main()
