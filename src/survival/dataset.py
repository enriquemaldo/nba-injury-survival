"""Build the landmark survival dataset: one row per player-game (pre-game).

Outcome for a landmark at game date g (proposal sections 4A and 4H):
    Follow-up starts at g and ends at the first of
      - onset of the next TARGET injury episode       -> event = 1
      - onset of any other injury episode              -> censored (not exposed while out)
      - g + HORIZON_DAYS                               -> censored (administrative horizon)
      - the day after the team's last game that season -> censored (season end)
      - the end of the injury-label data               -> censored (label coverage)
    time = days from g to that date (0 = injury reported the same day as the game).

INJURY DATE: the injured-list transaction often lags the injury (median 2 days,
94% within 14). An injury is therefore dated to the player's last game before
the placement when that game is within INJURY_DATE_MAX_LAG days, else to the
transaction date. The same date is used for target events and for censoring.

TARGET (primary) = proxy-tier, time-loss episode (non-contact lower-leg muscle /
tendon complaint, player missed >= 1 team game). Strict and broad variants are
also stored for sensitivity analyses. Uncertain episodes are never events; they
end follow-up as censoring, matching the proposal.

Fixed-horizon label for the classification baselines:
    y14 = 1 if a target injury occurs within CLASSIF_DAYS days of g
    evaluable14 = outcome at day 14 is known (event by day 14 or followed >= 14 days)

Usage:  python -m src.survival.dataset
Writes: data/processed/survival/landmarks.csv.gz, data/processed/survival/dataset_report.md
"""
import numpy as np
import pandas as pd

from src.config import (CLASSIF_DAYS, EARLY_ERA_SEASONS, HORIZON_DAYS, INTERIM, PROCESSED, RAW,
                        split_of)
from src.features.workload import (HISTORY_FEATURES, WORKLOAD_FEATURES, injury_history,
                                   workload_features)

OUT_DIR = PROCESSED / "survival"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LANDMARKS = OUT_DIR / "landmarks.csv.gz"
REPORT = OUT_DIR / "dataset_report.md"

SCHEDULE_FEATURES = ["is_home", "is_neutral", "days_rest", "back_to_back", "team_games_last_7d",
                     "travel_km", "abs_tz_shift_h", "tz_shift_h", "km_last_7d", "km_last_14d"]
BIO_FEATURES = ["age", "height_in", "weight_lb", "years_since_draft"]
CONTEXT_FEATURES = ["is_playoffs", "early_era"]


INJURY_DATE_MAX_LAG = 14


def injury_dates(ep: pd.DataFrame) -> pd.Series:
    """Best estimate of when each injury happened (see module docstring)."""
    onset = pd.to_datetime(ep.onset_date)
    last = pd.to_datetime(ep.last_game_before)
    use_last = last.notna() & ((onset - last).dt.days <= INJURY_DATE_MAX_LAG)
    return last.where(use_last, onset)


def target_masks(ep: pd.DataFrame) -> dict:
    time_loss = ep.games_missed.fillna(0) >= 1
    return {"proxy": ep.is_proxy & time_loss,     # primary
            "strict": ep.is_strict & time_loss,
            "broad": ep.is_broad & time_loss}


def outcomes(lm: pd.DataFrame, ep: pd.DataFrame, is_target: pd.Series, label_end: pd.Timestamp,
             horizon: int = HORIZON_DAYS) -> pd.DataFrame:
    """lm: PLAYER_ID, GAME_DATE, season_end. ep: PLAYER_ID, injury_date. Returns time, event, reason."""
    e = ep[["PLAYER_ID", "injury_date"]].rename(columns={"injury_date": "onset_date"}).copy()
    e["onset_date"] = pd.to_datetime(e.onset_date)
    e["is_target"] = is_target.to_numpy()
    # If two episodes start the same day, the target one wins.
    e = (e.sort_values(["PLAYER_ID", "onset_date", "is_target"], ascending=[True, True, False])
          .drop_duplicates(["PLAYER_ID", "onset_date"]))

    x = lm[["PLAYER_ID", "GAME_DATE", "season_end"]].copy()
    x["_row"] = np.arange(len(x))
    x = pd.merge_asof(x.sort_values("GAME_DATE"), e.sort_values("onset_date"), left_on="GAME_DATE",
                      right_on="onset_date", by="PLAYER_ID", direction="forward", allow_exact_matches=True)
    x = x.sort_values("_row")

    ends = pd.DataFrame({
        "horizon": x.GAME_DATE + pd.Timedelta(days=horizon),
        "season_end": x.season_end + pd.Timedelta(days=1),
        "label_end": pd.Series(label_end, index=x.index),
    })
    end = ends.min(axis=1)
    end_reason = ends.idxmin(axis=1)
    hit = x.onset_date.notna() & (x.onset_date <= end)
    event = hit & x.is_target.eq(True)        # NaN (no episode) -> False
    stop = np.where(hit, x.onset_date, end)
    reason = np.where(event, "target_event", np.where(hit, "other_injury", end_reason))
    days = (pd.Series(pd.to_datetime(stop), index=x.index) - x.GAME_DATE).dt.days
    out = pd.DataFrame({"time": days, "event": event.astype(int), "reason": reason}, index=x.index)
    out.index = lm.index
    return out


def build():
    gl = pd.read_csv(RAW / "gamelogs_all.csv")
    gl["GAME_DATE"] = pd.to_datetime(gl.GAME_DATE)
    tg = pd.read_csv(PROCESSED / "travel_team_games.csv", parse_dates=["GAME_DATE"])
    ep = pd.read_csv(INTERIM / "injury_episodes.csv", parse_dates=["onset_date"])
    ep["injury_date"] = injury_dates(ep)
    src = pd.read_json(INTERIM / "label_source.json", typ="series")
    label_end = pd.Timestamp(src.end)

    # ---- landmarks + features -------------------------------------------------
    lm = gl[["PLAYER_ID", "PLAYER_NAME", "GAME_ID", "GAME_DATE", "SEASON", "SEASON_TYPE",
             "TEAM_ABBREVIATION"]].rename(columns={"TEAM_ABBREVIATION": "TEAM"})
    wl = workload_features(gl, tg)
    lm = lm.merge(wl.drop(columns="GAME_DATE"), on=["PLAYER_ID", "GAME_ID"], how="left")

    trav = tg.rename(columns={"games_last_7d": "team_games_last_7d"})
    trav["abs_tz_shift_h"] = trav.tz_shift_h.abs()
    lm = lm.merge(trav[["GAME_ID", "TEAM"] + [c for c in SCHEDULE_FEATURES if c in trav]],
                  on=["GAME_ID", "TEAM"], how="left")
    n_no_travel = int(lm.travel_km.isna().sum())

    bio_path = RAW / "player_bio.csv"
    if bio_path.exists():
        bio = pd.read_csv(bio_path)
        if "draft_year" in bio:
            bio["years_since_draft"] = bio.SEASON.str[:4].astype(int) - bio.draft_year
        lm = lm.merge(bio[["PLAYER_ID", "SEASON"] + [c for c in BIO_FEATURES if c in bio]],
                      on=["PLAYER_ID", "SEASON"], how="left")
    for c in BIO_FEATURES:
        if c not in lm:
            lm[c] = np.nan

    masks = target_masks(ep)
    lm = lm.join(injury_history(lm, ep, masks["proxy"]))
    lm["is_playoffs"] = lm.SEASON_TYPE.isin(["Playoffs", "PlayIn"]).astype(int)
    lm["early_era"] = lm.SEASON.isin(EARLY_ERA_SEASONS).astype(int)
    for c in ("is_home", "is_neutral", "back_to_back"):
        lm[c] = lm[c].astype(float)

    # ---- outcomes -------------------------------------------------------------
    season_end = tg.groupby(["TEAM", "SEASON"]).GAME_DATE.max().rename("season_end").reset_index()
    lm = lm.merge(season_end, on=["TEAM", "SEASON"], how="left")
    for name, mask in masks.items():
        o = outcomes(lm, ep, mask, label_end)
        sfx = "" if name == "proxy" else f"_{name}"
        lm[f"time{sfx}"], lm[f"event{sfx}"] = o.time, o.event
        if name == "proxy":
            lm["censor_reason"] = o.reason
    lm["y14"] = ((lm.event == 1) & (lm.time <= CLASSIF_DAYS)).astype(int)
    lm["evaluable14"] = ((lm.y14 == 1) | (lm.time >= CLASSIF_DAYS)).astype(int)
    lm["split"] = lm.SEASON.map(split_of)

    # ---- exclusions -----------------------------------------------------------
    n_all = len(lm)
    excl = {"game after end of injury labels (no outcome)": lm.GAME_DATE > label_end,
            "no venue/travel row": lm.travel_km.isna()}
    keep = ~pd.concat(excl, axis=1).any(axis=1)
    excluded = {k: int(v.sum()) for k, v in excl.items()}
    lm = lm[keep].sort_values(["GAME_DATE", "PLAYER_ID"]).reset_index(drop=True)

    lm.to_csv(LANDMARKS, index=False)
    report(lm, ep, masks["proxy"], n_all, excluded, label_end, n_no_travel)
    print(f"{len(lm):,} landmarks -> {LANDMARKS}")
    return lm


def model_features(lm: pd.DataFrame) -> list:
    """Features available to the tabular models (bio only if it was downloaded)."""
    bio = [c for c in BIO_FEATURES if lm[c].notna().mean() > 0.5]
    return WORKLOAD_FEATURES + HISTORY_FEATURES + SCHEDULE_FEATURES + bio + CONTEXT_FEATURES


def _md(df):
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if pd.isna(v) else (f"{v:,.3f}".rstrip("0").rstrip(".")
                                        if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer))
                                        else str(v)) for v in r) + " |")
    return "\n".join(lines)


def report(lm, ep, is_target, n_all, excluded, label_end, n_no_travel):
    feats = model_features(lm)
    tgt = ep[is_target.to_numpy()].copy()
    tgt["onset_date"] = pd.to_datetime(tgt.injury_date)
    # Which target episodes are reached by at least one landmark?
    ev = lm[lm.event == 1].copy()
    ev["onset_date"] = ev.GAME_DATE + pd.to_timedelta(ev.time, unit="D")
    reached = ev[["PLAYER_ID", "onset_date"]].drop_duplicates()
    tgt_r = tgt.merge(reached.assign(reached=1), on=["PLAYER_ID", "onset_date"], how="left")
    tgt_r = tgt_r[tgt_r.onset_date <= label_end]

    by_season = (lm.groupby(["SEASON", "split"])
                 .agg(players=("PLAYER_ID", "nunique"), landmarks=("GAME_ID", "size"),
                      event_landmarks=("event", "sum"), y14_positive=("y14", "sum"),
                      evaluable14=("evaluable14", "sum"))
                 .reset_index())
    ev_season = tgt_r.groupby("season").agg(target_episodes=("PLAYER_ID", "size"),
                                            reached_by_landmark=("reached", "sum")).reset_index()
    by_season = by_season.merge(ev_season.rename(columns={"season": "SEASON"}), on="SEASON", how="left")
    by_season["reached_by_landmark"] = by_season.reached_by_landmark.fillna(0).astype(int)
    by_split = (lm.groupby("split").agg(landmarks=("GAME_ID", "size"), players=("PLAYER_ID", "nunique"),
                                        event_landmarks=("event", "sum"), y14_positive=("y14", "sum"))
                .reset_index())
    reasons = lm.censor_reason.value_counts().rename_axis("follow-up ended by").reset_index(name="landmarks")
    site = (tgt_r[tgt_r.reached == 1].site.value_counts().rename_axis("site")
            .reset_index(name="target_episodes"))
    fu = lm.time.describe(percentiles=[.25, .5, .75]).round(1).to_frame("follow-up days").T.reset_index(drop=True)
    tte = lm.loc[lm.event == 1, "time"].describe(percentiles=[.25, .5, .75]).round(1).to_frame(
        "days to event").T.reset_index(drop=True)
    miss = (lm[feats].isna().mean().mul(100).round(1).rename("% missing").rename_axis("feature")
            .reset_index().query("`% missing` > 0").sort_values("% missing", ascending=False))
    hist = lm.games_prior_total.describe(percentiles=[.1, .5, .9]).round(0).to_frame(
        "prior games available (since 2016-17)").T.reset_index(drop=True)
    uncertain = int((ep.tier == "uncertain").sum())

    md = ["# Survival dataset report (proposal section 7)\n",
          f"Prediction unit: player-game landmark (pre-game). Follow-up: up to {HORIZON_DAYS} days. "
          f"Primary event: proxy-tier, time-loss injury. Labels end {label_end:%Y-%m-%d}.\n",
          f"Injuries are dated to the player's last game before the injured-list placement when it is within "
          f"{INJURY_DATE_MAX_LAG} days ({(ep.injury_date != ep.onset_date).mean():.0%} of injury episodes), "
          f"otherwise to the placement date.\n",
          f"- Landmarks built: **{n_all:,}**; kept: **{len(lm):,}**; players: **{lm.PLAYER_ID.nunique():,}**; "
          f"seasons: **{lm.SEASON.nunique()}**",
          f"- Duplicate player-game rows: **{int(lm.duplicated(['PLAYER_ID', 'GAME_ID']).sum())}**",
          f"- Uncertain injury episodes (never events; end follow-up as censoring): **{uncertain:,}**",
          f"- Model features: **{len(feats)}**", ""]
    md += ["## Excluded landmarks\n", _md(pd.DataFrame(list(excluded.items()), columns=["reason", "landmarks"])), ""]
    md += ["## By split\n", _md(by_split), "",
           "## By season\n",
           "`event_landmarks` counts landmarks whose follow-up ends in a target injury (several landmarks "
           "before the same injury each count). `reached_by_landmark` = distinct target injuries that at "
           "least one landmark's follow-up reaches.\n", _md(by_season), "",
           "## How follow-up ended\n", _md(reasons), "",
           "## Follow-up and time to event (days)\n", _md(fu), "", _md(tte), "",
           "## Target injuries by site (reached by a landmark)\n", _md(site), "",
           "## History available per landmark\n", _md(hist), "",
           "## Missing values in model features\n",
           "Missing values are imputed with the training-set median plus a missing-indicator column "
           "(linear and Cox models); tree models handle them natively.\n", _md(miss), ""]
    REPORT.write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    build()
