"""Offline smoke tests on synthetic data (no network).

Run:  python -m pytest tests/  (or python tests/test_pipeline.py)
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.ingest.pst_scrape import parse_page
from src.labels.episodes import build_episodes
from src.labels.player_match import build_roster, match_names, normalize
from src.labels.taxonomy import classify, label

PST_HTML = """<html><body><table class="datatable center">
<tr><td>Date</td><td>Team</td><td>Acquired</td><td>Relinquished</td><td>Notes</td></tr>
<tr><td>2016-11-02</td><td>Lakers</td><td></td><td>&bull; D'Angelo Russell</td>
<td>placed on IL with strained left hamstring</td></tr>
<tr><td>2016-11-20</td><td>Lakers</td><td>&bull; D'Angelo Russell</td><td></td>
<td>activated from IL</td></tr>
</table></body></html>"""


def test_parse_page():
    df = parse_page(PST_HTML)
    assert len(df) == 2
    assert df.Relinquished.iloc[0] == "D'Angelo Russell"
    assert df.Acquired.iloc[0] == ""


def test_taxonomy():
    cases = {
        "placed on IL with strained left hamstring": "strict",
        "sore right calf (DTD)": "proxy",
        "torn left Achilles tendon (out for season)": "strict",
        "sprained right ankle (DTD)": "broad",
        "strained hamstring after collision with teammate": "broad",
        "placed on IL with lower back tightness": "other_injury",
        "rest (DNP)": "non_injury",
        "illness (DNP)": "non_injury",
        "placed on IL (injury)": "uncertain",
        "groin strain": "strict",
        "bruised left quad (DTD)": "broad",
        "left thigh contusion": "broad",
        "placed on IL with fractured orbital bone": "other_injury",
        "placed on IL with bruised tailbone": "other_injury",
        "strained right rotator cuff (DTD)": "other_injury",
        "placed on IL with cervical strain": "other_injury",
        "sinus fracture (DTD)": "other_injury",
        "placed on IL with upper respiratory infection": "non_injury",
        "placed on IL with sinus surgery": "non_injury",
        "appendicitis (out for season)": "non_injury",
        "placed on IL with quarantine for leaving bubble": "non_injury",
        "placed on IL with sports hernia": "broad",
        "placed on IL": "uncertain",
        "general soreness (DTD)": "uncertain",
        "placed on IL with migraine headache": "non_injury",   # from label audit
        "placed on IL with head injury": "other_injury",
        "placed on IL with right thigh injury": "proxy",
    }
    for note, tier in cases.items():
        assert classify(note)["tier"] == tier, (note, classify(note))
    lab = label(pd.DataFrame({"Notes": list(cases)}))
    assert (lab.is_strict <= lab.is_proxy).all() and (lab.is_proxy <= lab.is_broad).all()


def _gamelogs():
    rows = []
    days = pd.date_range("2016-10-26", periods=40, freq="2D")
    for d in days:
        rows.append((1, "D'Angelo Russell", "LAL", d))
        rows.append((2, "Nene", "HOU", d))
        rows.append((3, "Brandon Ingram", "LAL", d))  # teammate keeps LAL's schedule
    gl = pd.DataFrame(rows, columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GAME_DATE"])
    gl["SEASON"] = "2016-17"
    # Russell misses Nov 3 - Nov 19 (injury window)
    out = (gl.PLAYER_ID == 1) & gl.GAME_DATE.between("2016-11-03", "2016-11-19")
    return gl[~out].reset_index(drop=True)


def test_matching():
    assert normalize("Kelly Oubre Jr.") == "kelly oubre"
    roster = build_roster(_gamelogs())
    names = pd.DataFrame({"raw_name": ["D'Angelo Russell", "Nene / Nene Hilario", "Dangelo Russel"],
                          "season": ["2016-17"] * 3})
    m = match_names(names, roster).set_index("raw_name")
    assert m.loc["D'Angelo Russell", "match_method"] == "exact"
    assert m.loc["Nene / Nene Hilario", "PLAYER_ID"] == 2
    assert m.loc["Dangelo Russel", "match_method"] == "fuzzy"


def test_matching_real_cases():
    gl = _gamelogs()
    extra = pd.DataFrame([(4, "Wesley Matthews", "DAL", pd.Timestamp("2016-11-01"), "2016-17"),
                          (5, "Ben Simmons", "PHI", pd.Timestamp("2017-11-01"), "2017-18")],
                         columns=gl.columns)
    roster = build_roster(pd.concat([gl, extra]))
    names = pd.DataFrame({"raw_name": ["Wes Matthews Jr.", "Ben Simmons", "Steve Kerr", "76ers"],
                          "season": ["2016-17"] * 4})
    m = match_names(names, roster).set_index("raw_name")
    assert m.loc["Wes Matthews Jr.", "PLAYER_ID"] == 4           # known alias
    assert m.loc["Ben Simmons", "PLAYER_ID"] == 5                # missed whole season
    assert m.loc["Ben Simmons", "match_pool"] == "any_season"
    assert m.loc["Steve Kerr", "match_method"] == "not_player"   # coach
    assert m.loc["76ers", "match_method"] == "not_player"        # team


def test_name_variants():
    from src.labels.player_match import _aliases
    assert "reggie jackson" in _aliases("Reggie Jackson (Shon)")
    assert "mike scott" in _aliases("(James) Mike Scott")
    assert "kay felder" in _aliases("Kahlil Felder / Kay Felder")
    assert "james michael mcadoo" in _aliases("James Michael McAdoo / James McAdoo (Michael)")


def test_episodes():
    gl = _gamelogs()
    inj = pd.DataFrame({
        "PLAYER_ID": [1, 1, 1],
        "Date": pd.to_datetime(["2016-11-02", "2016-11-06", "2016-12-10"]),
        "Notes": ["sore left hamstring (DTD)", "placed on IL with strained left hamstring",
                  "sprained right ankle (DTD)"],
        "season": "2016-17",
    })
    inj = label(inj)
    ep = build_episodes(inj, gl)
    assert len(ep) == 2, ep                       # first two rows merge
    first = ep.iloc[0]
    assert first.tier == "strict"                 # best tier of merged rows
    assert first.games_missed == 9                # Nov 3,5,...,19 = 9 team games
    assert ep.iloc[1].tier == "broad"


def _travel_tg(rows):
    from src.features.travel import team_games
    gl = pd.DataFrame(rows, columns=["GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE", "TEAM_ABBREVIATION", "MATCHUP"])
    return team_games(gl)


def test_travel_venues_and_features():
    from src.features.travel import assign_venues, haversine_km, load_reference, travel_features
    arenas, neutral = load_reference()
    R = "Regular Season"
    tg = _travel_tg([
        (1, "2019-10-24", "2019-20", R, "GSW", "GSW vs. LAC"),   # Chase Center (relocation)
        (1, "2019-10-24", "2019-20", R, "LAC", "LAC @ GSW"),
        (2, "2019-10-26", "2019-20", R, "LAC", "LAC vs. PHX"),   # back in LA, back-to-back? no (2 days)
        (2, "2019-10-26", "2019-20", R, "PHX", "PHX @ LAC"),
        (3, "2020-08-01", "2019-20", R, "LAC", "LAC vs. LAL"),   # bubble
        (3, "2020-08-01", "2019-20", R, "LAL", "LAL @ LAC"),
        (4, "2021-01-05", "2020-21", R, "TOR", "TOR vs. BOS"),   # Raptors in Tampa
        (4, "2021-01-05", "2020-21", R, "BOS", "BOS @ TOR"),
        (5, "2025-01-23", "2024-25", R, "SAS", "SAS vs. IND"),   # Paris
        (5, "2025-01-23", "2024-25", R, "IND", "IND @ SAS"),
    ])
    tg, log = assign_venues(tg, arenas, neutral)
    v = tg.set_index(["GAME_ID", "TEAM"])
    assert v.loc[(1, "LAC"), "venue"] == "Chase Center"
    assert v.loc[(3, "LAL"), "city"] == "Lake Buena Vista" and v.loc[(3, "LAL"), "is_neutral"]
    assert v.loc[(4, "BOS"), "city"] == "Tampa"
    assert v.loc[(5, "IND"), "city"] == "Paris"
    assert not v.loc[(2, "PHX"), "is_neutral"]

    f = travel_features(tg, arenas).set_index(["GAME_ID", "TEAM"])
    # LAC opener: Los Angeles -> San Francisco ~ 550 km; then back home
    assert 500 < f.loc[(1, "LAC"), "travel_km"] < 600
    assert 500 < f.loc[(2, "LAC"), "travel_km"] < 600 and f.loc[(2, "LAC"), "days_rest"] == 2
    assert f.loc[(1, "GSW"), "travel_km"] == 0 and pd.isna(f.loc[(1, "GSW"), "days_rest"])
    # Paris opener for IND (from Indianapolis): moves 6 h east in January
    assert f.loc[(5, "IND"), "tz_shift_h"] == 6
    assert abs(haversine_km(0, 0, 0, 1) - 111.2) < 0.1


def test_tz_offset_arizona():
    from src.features.travel import utc_offset_h
    # Phoenix stays at UTC-7 all year; Denver switches with DST.
    assert utc_offset_h("America/Phoenix", "2024-01-15") == utc_offset_h("America/Phoenix", "2024-04-15") == -7
    assert utc_offset_h("America/Denver", "2024-04-15") == -6


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
