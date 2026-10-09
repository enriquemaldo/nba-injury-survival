"""Shared paths and constants for the Phase 1 pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
for _p in (RAW, INTERIM, PROCESSED):
    _p.mkdir(parents=True, exist_ok=True)

# Seasons in the proposal: 2016-17 through 2025-26.
SEASONS = [f"{y}-{str(y + 1)[-2:]}" for y in range(2016, 2026)]

# Chronological split (Option B, adopted Oct 2026): the primary label source
# ends 2025-01-11, so the split moves back one season to keep train/val/test
# on one consistent source. 2024-25 / 2025-26 = supplementary test with a
# different label source (official injury reports), reported separately.
TRAIN_SEASONS = SEASONS[:6]          # 2016-17 .. 2021-22
VAL_SEASONS = [SEASONS[6]]           # 2022-23
TEST_SEASONS = [SEASONS[7]]          # 2023-24
SUPPLEMENTARY_SEASONS = SEASONS[8:]  # 2024-25, 2025-26


def split_of(season: str) -> str:
    if season in TRAIN_SEASONS:
        return "train"
    if season in VAL_SEASONS:
        return "val"
    if season in TEST_SEASONS:
        return "test"
    return "supplementary"


# Approximate date window per season (regular season + playoffs, with margin
# for preseason injuries). 2019-20 ran to Oct 2020 (bubble); 2020-21 started
# Dec 2020 and ran to Jul 2021.
SEASON_WINDOWS = {
    "2016-17": ("2016-09-15", "2017-06-30"),
    "2017-18": ("2017-09-15", "2018-06-30"),
    "2018-19": ("2018-09-15", "2019-06-30"),
    "2019-20": ("2019-09-15", "2020-10-31"),
    "2020-21": ("2020-11-01", "2021-07-31"),
    "2021-22": ("2021-09-15", "2022-06-30"),
    "2022-23": ("2022-09-15", "2023-06-30"),
    "2023-24": ("2023-09-15", "2024-06-30"),
    "2024-25": ("2024-09-15", "2025-06-30"),
    "2025-26": ("2025-09-15", "2026-06-30"),
}


def season_of(date) -> str | None:
    """Map a date to its season label using SEASON_WINDOWS (None if offseason)."""
    import pandas as pd
    d = pd.Timestamp(date)
    for s, (a, b) in SEASON_WINDOWS.items():
        if pd.Timestamp(a) <= d <= pd.Timestamp(b):
            return s
    return None
