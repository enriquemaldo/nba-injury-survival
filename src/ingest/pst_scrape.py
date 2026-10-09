"""Step 2: scrape injury transactions from Pro Sports Transactions (PST).

Searches PST season by season with the "Movement to/from injured/inactive
list" and "Missed games due to injury" boxes checked, pages through the
results (25 rows per page), and caches every page's raw HTML so parsing can
be re-run without re-downloading.

Usage:
    python -m src.ingest.pst_scrape                 # all seasons
    python -m src.ingest.pst_scrape --season 2016-17
    python -m src.ingest.pst_scrape --parse-only    # re-parse cached HTML

VERIFY BEFORE A FULL RUN: open one search in your browser and confirm the
URL parameters in SEARCH_URL still match what the site produces. If the site
blocks scripted requests, save result pages from your browser into
data/raw/pst_html/<season>/ (any filenames ending in .html) and run with
--parse-only.

Be polite: keep DELAY at 2-3 s and don't run multiple copies in parallel.
"""
import argparse
import re
import time

import pandas as pd
import requests
from bs4 import BeautifulSoup

from src.config import RAW, SEASON_WINDOWS

HTML_DIR = RAW / "pst_html"
OUT = RAW / "pst_injuries.csv"
SEARCH_URL = (
    "https://www.prosportstransactions.com/basketball/Search/SearchResults.php"
    "?Player=&Team=&BeginDate={begin}&EndDate={end}"
    "&ILChkBx=yes&InjuriesChkBx=yes&Submit=Search&start={start}"
)
PAGE_SIZE = 25
DELAY = 2.5
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}
COLUMNS = ["Date", "Team", "Acquired", "Relinquished", "Notes"]


# ---------------------------------------------------------------- parsing --
def _clean_name(cell: str) -> str:
    """PST prefixes names with a bullet; some cells hold 'Name / Alt Name'."""
    s = cell.replace("•", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", s).strip()


def parse_page(html: str) -> pd.DataFrame:
    """Parse one results page into a DataFrame with COLUMNS."""
    soup = BeautifulSoup(html, "html.parser")
    target = None
    for table in soup.find_all("table"):
        first = table.find("tr")
        if first and "Relinquished" in first.get_text():
            target = table
            break
    if target is None:
        return pd.DataFrame(columns=COLUMNS)

    rows = []
    for tr in target.find_all("tr")[1:]:
        cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
        if len(cells) < 5:
            continue
        rows.append(dict(zip(COLUMNS, cells[:5])))
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("Acquired", "Relinquished"):
        df[c] = df[c].map(_clean_name)
    return df


# --------------------------------------------------------------- scraping --
def scrape_season(season: str, session: requests.Session) -> None:
    begin, end = SEASON_WINDOWS[season]
    folder = HTML_DIR / season
    folder.mkdir(parents=True, exist_ok=True)
    start = 0
    while True:
        path = folder / f"page_{start:06d}.html"
        if path.exists():
            html = path.read_text(encoding="utf-8")
        else:
            url = SEARCH_URL.format(begin=begin, end=end, start=start)
            resp = session.get(url, headers=HEADERS, timeout=60)
            if resp.status_code != 200:
                raise RuntimeError(
                    f"{season} start={start}: HTTP {resp.status_code}. If the site "
                    "is blocking scripts, see the module docstring (--parse-only)."
                )
            html = resp.text
            path.write_text(html, encoding="utf-8")
            time.sleep(DELAY)
        n = len(parse_page(html))
        print(f"  {season} start={start}: {n} rows")
        if n == 0:
            path.unlink(missing_ok=True)  # don't keep the empty sentinel page
            break
        start += PAGE_SIZE


def parse_all() -> pd.DataFrame:
    frames = []
    for season_dir in sorted(p for p in HTML_DIR.iterdir() if p.is_dir()):
        for f in sorted(season_dir.glob("*.html")):
            df = parse_page(f.read_text(encoding="utf-8"))
            df["season_folder"] = season_dir.name
            frames.append(df)
    if not frames:
        raise SystemExit("No cached HTML found in data/raw/pst_html/")
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    before = len(df)
    df = df.dropna(subset=["Date"]).drop_duplicates(subset=COLUMNS)
    print(f"parsed {before:,} rows -> {len(df):,} after dedup")
    df["source"] = "pst_scrape"
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", choices=list(SEASON_WINDOWS))
    ap.add_argument("--parse-only", action="store_true")
    args = ap.parse_args()

    if not args.parse_only:
        seasons = [args.season] if args.season else list(SEASON_WINDOWS)
        with requests.Session() as s:
            for season in seasons:
                print(f"scraping {season}")
                scrape_season(season, s)

    df = parse_all()
    df.to_csv(OUT, index=False)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
