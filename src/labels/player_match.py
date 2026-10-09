"""Step 3: match injury-transaction names to nba_api PLAYER_IDs.

Candidate pools, tried in order (stops at the first confident match):
  1. players in the game logs THAT season
  2. players in the game logs in ANY season - catches players who missed the
     whole season injured (e.g. Simmons 2016-17), who never log a game that year
  3. the full nba_api player list (data/raw/nba_players.csv)

Within each pool: known PST name variants (KNOWN_ALIASES) -> exact match on a
normalised name -> fuzzy match (score >= FUZZY_MIN and clearly ahead of the
runner-up). Manual overrides in data/interim/manual_name_map.csv
(raw_name, season, PLAYER_ID; blank season = all seasons) win over everything.

Coaches, executives and team names are dropped (NOT_PLAYERS).
Leftovers -> data/interim/unmatched_names.csv; fuzzy matches ->
data/interim/fuzzy_matches.csv so you can audit them.
"""
import re
import unicodedata

import pandas as pd

from src.config import INTERIM, RAW

FUZZY_MIN = 90
MANUAL = INTERIM / "manual_name_map.csv"

# PST spelling -> official NBA spelling(s). Extend as you find more.
KNOWN_ALIASES = {
    "james mcadoo": ["james michael mcadoo"],
    "wes matthews": ["wesley matthews"],
    "wes johnson": ["wesley johnson"],
    "ron artest": ["metta world peace", "metta sandiford artest"],
    "nene hilario": ["nene"],
    "jose barea": ["jj barea", "j j barea"],
    "nicolas claxton": ["nic claxton"],
    "moe harkless": ["maurice harkless"],
    "moe wagner": ["moritz wagner"],
    "enes kanter": ["enes freedom", "enes kanter"],
    "domas sabonis": ["domantas sabonis"],
    "alex ajinca": ["alexis ajinca"],
    "sheldon mac": ["sheldon mcclellan"],
    "billy ouattara": ["yakuba ouattara"],
    "mike young": ["michael young"],
    "kj mcdaniels": ["kj mcdaniels", "k j mcdaniels"],
    "cj wilcox": ["cj wilcox", "c j wilcox"],
}

# Coaches / executives / non-player entries that show up in PST.
NOT_PLAYERS = {
    "steve kerr", "tyronn lue", "steve clifford", "gregg popovich", "doc rivers",
    "rick carlisle", "dwane casey", "mike dantoni", "erik spoelstra",
}
TEAM_NAMES = {
    "76ers", "bucks", "bulls", "cavaliers", "celtics", "clippers", "grizzlies",
    "hawks", "heat", "hornets", "jazz", "kings", "knicks", "lakers", "magic",
    "mavericks", "nets", "nuggets", "pacers", "pelicans", "pistons", "raptors",
    "rockets", "spurs", "suns", "thunder", "timberwolves", "trail blazers",
    "blazers", "warriors", "wizards",
}

try:
    from rapidfuzz import fuzz

    def _score(a, b):
        return fuzz.token_sort_ratio(a, b)
except ImportError:  # fallback keeps the pipeline dependency-light
    from difflib import SequenceMatcher

    def _score(a, b):
        return 100 * SequenceMatcher(None, " ".join(sorted(a.split())), " ".join(sorted(b.split()))).ratio()


def normalize(name: str) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    s = re.sub(r"\b(jr|sr|ii|iii|iv)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _aliases(raw: str) -> list[str]:
    """PST name variants: 'Name / Alternate Name', '(a.k.a. X)', and
    parenthetical given names like 'Reggie Jackson (Shon)' or '(James) Mike Scott'."""
    raw = re.sub(r"\(a\.?k\.?a\.?\s*([^)]*)\)", r"/ \1", str(raw), flags=re.I)
    names = []
    for part in re.split(r"\s*/\s*", raw):
        if not part.strip():
            continue
        names.append(normalize(re.sub(r"\([^)]*\)", " ", part)))      # drop (...)
        names.append(normalize(part.replace("(", " ").replace(")", " ")))  # keep contents
    names = [n for n in names if n]
    out = []
    for n in names:
        out.extend(KNOWN_ALIASES.get(n, []))
        out.append(n)
    return list(dict.fromkeys(out))  # dedupe, keep order (known aliases first)


def is_not_player(raw: str) -> bool:
    n = normalize(raw)
    return n in NOT_PLAYERS or n in TEAM_NAMES


def build_roster(gamelogs: pd.DataFrame) -> pd.DataFrame:
    r = gamelogs[["SEASON", "PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    r["norm"] = r["PLAYER_NAME"].map(normalize)
    return r


def _static_players() -> pd.DataFrame:
    path = RAW / "nba_players.csv"
    if not path.exists():
        return pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "norm"])
    p = pd.read_csv(path).rename(columns={"id": "PLAYER_ID", "full_name": "PLAYER_NAME"})
    p = p[["PLAYER_ID", "PLAYER_NAME"]].drop_duplicates()
    p["norm"] = p["PLAYER_NAME"].map(normalize)
    return p


def _match_in(pool: pd.DataFrame, raw: str):
    """Return (PLAYER_ID, method) or (None, None) for one candidate pool."""
    if pool.empty:
        return None, None
    for alias in _aliases(raw):
        hit = pool[pool.norm == alias]
        if hit.PLAYER_ID.nunique() == 1:
            return int(hit.PLAYER_ID.iloc[0]), "exact"
    best = []
    for alias in _aliases(raw):
        sc = pool.norm.map(lambda x: _score(alias, x))
        top2 = sc.nlargest(2)
        runner = top2.iloc[1] if len(top2) > 1 else 0
        best.append((top2.iloc[0], runner, pool.loc[top2.index[0], "PLAYER_ID"]))
    top, runner, pid = max(best)
    if top >= FUZZY_MIN and top - runner >= 5:
        return int(pid), "fuzzy"
    return None, None


def match_names(names: pd.DataFrame, roster: pd.DataFrame) -> pd.DataFrame:
    """names: DataFrame with columns raw_name, season. Returns + PLAYER_ID, match_method, pool."""
    manual = (pd.read_csv(MANUAL) if MANUAL.exists()
              else pd.DataFrame(columns=["raw_name", "season", "PLAYER_ID"]))
    by_season = {s: g for s, g in roster.groupby("SEASON")}
    all_seasons = roster.drop_duplicates("PLAYER_ID")
    static = _static_players()

    out = []
    for raw, season in names[["raw_name", "season"]].drop_duplicates().itertuples(index=False):
        pid, method, pool_name = None, "unmatched", None
        if is_not_player(raw):
            method = "not_player"
        else:
            m = manual[(manual.raw_name == raw) & (manual.season.isna() | (manual.season == season))]
            if len(m):
                pid, method, pool_name = int(m.PLAYER_ID.iloc[0]), "manual", "manual"
            else:
                pools = [("season", by_season.get(season, roster.iloc[:0])),
                         ("any_season", all_seasons), ("nba_static", static)]
                for pname, pool in pools:
                    pid, how = _match_in(pool, raw)
                    if pid is not None:
                        method, pool_name = how, pname
                        break
        out.append({"raw_name": raw, "season": season, "PLAYER_ID": pid,
                    "match_method": method, "match_pool": pool_name})
    return pd.DataFrame(out)
