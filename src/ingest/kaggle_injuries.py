"""Step 2: load the injury-label dataset.

Source: Kaggle "2016-2025_nba_injury_data" (jacquesoberweis), a public export
of Pro Sports Transactions injured-list transactions, 2016-10-25 to 2025-01-12.
Columns: Date, Team, Acquired, Relinquished, Notes.

Download the CSV, save it as data/raw/kaggle_injuries_2016_2025.csv, then run:

    python -m src.ingest.kaggle_injuries

Writes data/raw/mirror_injuries_clean.csv (cleaned, deduplicated, restricted
to the proposal window).
"""
import pandas as pd

from src.config import RAW
from src.ingest.pst_scrape import COLUMNS, _clean_name

SRC = RAW / "kaggle_injuries_2016_2025.csv"
OUT = RAW / "mirror_injuries_clean.csv"


def load(path=SRC) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Normalise column names (files on Kaggle vary in case / index column).
    rename = {c: c2 for c in df.columns for c2 in COLUMNS if c.strip().lower() == c2.lower()}
    df = df.rename(columns=rename)
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise SystemExit(f"{path.name} is missing columns {missing}; found {list(df.columns)}")
    df = df[COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for c in ("Acquired", "Relinquished"):
        df[c] = df[c].fillna("").astype(str).map(_clean_name)
    df["Notes"] = df["Notes"].fillna("").astype(str)
    df = df.dropna(subset=["Date"])
    df = df[df["Date"] >= "2016-09-15"]  # proposal window only
    n0 = len(df)
    df = df.drop_duplicates(subset=COLUMNS)
    if len(df) < n0:
        print(f"dropped {n0 - len(df)} exact duplicate rows")
    df["source"] = "kaggle_2016_2025"
    return df


def main():
    if not SRC.exists():
        raise SystemExit(f"{SRC} not found - download the Kaggle CSV and save it under that name.")
    df = load()
    df.to_csv(OUT, index=False)
    print(f"{len(df):,} rows, {df.Date.min():%Y-%m-%d} to {df.Date.max():%Y-%m-%d} -> {OUT.name}")


if __name__ == "__main__":
    main()
