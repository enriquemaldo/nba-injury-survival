"""Step 5: collapse injury transactions into injury EPISODES.

One injury typically produces several rows (DTD note, placed on IL, IL again
after a setback...). Rule used here: consecutive injury rows for the same
player belong to the SAME episode if the player played no game in between.
An episode ends at the player's next game appearance.

Per episode we record:
  onset_date        date of the first row
  tier / site       most specific tier among its rows (strict > proxy > ...)
  last_game_before  player's last game on or before onset
  return_game       player's first game after the episode's last row (NaT = none in data)
  games_missed      team games strictly between last_game_before and return_game
                    (or to the team's last game in the data if no return)
"""
import numpy as np
import pandas as pd

from src.labels.taxonomy import TIER_ORDER

RANK = {t: i for i, t in enumerate(TIER_ORDER)}


def build_episodes(inj: pd.DataFrame, gamelogs: pd.DataFrame) -> pd.DataFrame:
    """inj: matched onset rows with PLAYER_ID, Date, tier, site, Notes, season."""
    inj = inj[inj.tier != "non_injury"].dropna(subset=["PLAYER_ID"]).copy()
    inj["PLAYER_ID"] = inj["PLAYER_ID"].astype(int)
    inj = inj.sort_values(["PLAYER_ID", "Date"])

    gl = gamelogs[["PLAYER_ID", "GAME_DATE", "TEAM_ABBREVIATION"]].copy()
    gl["GAME_DATE"] = pd.to_datetime(gl["GAME_DATE"])
    player_games = {p: g.sort_values("GAME_DATE") for p, g in gl.groupby("PLAYER_ID")}
    team_games = {t: np.sort(g.GAME_DATE.unique()) for t, g in gl.groupby("TEAM_ABBREVIATION")}

    episodes = []
    for pid, rows in inj.groupby("PLAYER_ID"):
        games = player_games.get(pid)
        gdates = games.GAME_DATE.values if games is not None else np.array([], dtype="datetime64[ns]")
        current = []

        def flush():
            if not current:
                return
            ep = pd.DataFrame(current)
            best = ep.loc[ep.tier.map(RANK).idxmin()]
            onset, last_row = ep.Date.min(), ep.Date.max()
            before = gdates[gdates <= np.datetime64(onset)]
            after = gdates[gdates > np.datetime64(last_row)]
            lgb = pd.Timestamp(before[-1]) if len(before) else pd.NaT
            ret = pd.Timestamp(after[0]) if len(after) else pd.NaT
            team = None
            if games is not None and pd.notna(lgb):
                team = games.loc[games.GAME_DATE == lgb, "TEAM_ABBREVIATION"].iloc[0]
            missed = np.nan
            if team is not None:
                tg = team_games[team]
                hi = np.datetime64(ret) if pd.notna(ret) else tg[-1] + np.timedelta64(1, "D")
                missed = int(((tg > np.datetime64(lgb)) & (tg < hi)).sum())
            episodes.append({
                "PLAYER_ID": pid,
                "season": best.season,
                "onset_date": onset,
                "tier": best.tier,
                "site": best.site,
                "contact_language": bool(ep.contact_language.any()),
                "n_rows": len(ep),
                "notes": " | ".join(ep.Notes.astype(str).unique()),
                "team": team,
                "last_game_before": lgb,
                "return_game": ret,
                "games_missed": missed,
            })
            current.clear()

        for r in rows.itertuples(index=False):
            if current:
                prev = current[-1]["Date"]
                played_between = ((gdates > np.datetime64(prev)) & (gdates <= np.datetime64(r.Date))).any()
                if played_between:
                    flush()
            current.append(r._asdict())
        flush()

    ep = pd.DataFrame(episodes)
    rank = ep.tier.map(RANK)
    ep["is_strict"] = rank <= RANK["strict"]
    ep["is_proxy"] = rank <= RANK["proxy"]
    ep["is_broad"] = rank <= RANK["broad"]
    return ep.sort_values(["onset_date", "PLAYER_ID"]).reset_index(drop=True)
