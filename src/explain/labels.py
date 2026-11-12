"""Readable names for model inputs (SHAP report and dashboard)."""
from src.sequence.build import family_of

FEATURE_LABELS = {
    "prev_game_min": "Minutes in previous game",
    "days_since_prev_game": "Days since previous game",
    "min_7d": "Minutes, last 7 days", "min_14d": "Minutes, last 14 days", "min_28d": "Minutes, last 28 days",
    "poss_7d": "Possessions used, last 7 days", "poss_28d": "Possessions used, last 28 days",
    "usage_per_min_28d": "Usage per minute, last 28 days",
    "games_3d": "Games, last 3 days", "games_7d": "Games, last 7 days", "games_14d": "Games, last 14 days",
    "b2b_apps_14d": "Back-to-backs played, last 14 days",
    "ewma_min_short": "Recent minutes per game (~7 games)", "ewma_min_long": "Minutes per game (~28 games)",
    "acwr_min": "Acute:chronic minutes ratio", "ewma_ratio_min": "Workload spike (short/long minutes)",
    "consecutive_games": "Consecutive games played", "missed_team_games_14d": "Team games missed, last 14 days",
    "season_games_prior": "Games played this season", "season_min_prior": "Minutes played this season",
    "games_prior_total": "Career games (in data)", "min_prior_total": "Career minutes (in data)",
    "prior_target_injuries": "Prior lower-leg non-contact injuries",
    "prior_lower_leg_injuries": "Prior lower-leg injuries (any)",
    "prior_any_injuries": "Prior injuries (any)", "days_since_last_injury": "Days since last injury",
    "is_home": "Home game", "is_neutral": "Neutral-site game", "days_rest": "Team days of rest",
    "back_to_back": "Back-to-back tonight", "team_games_last_7d": "Team games, last 7 days",
    "travel_km": "Travel to this game (km)", "abs_tz_shift_h": "Time-zone change (h)",
    "tz_shift_h": "Time-zone shift, signed (h)", "km_last_7d": "Team travel, last 7 days (km)",
    "km_last_14d": "Team travel, last 14 days (km)",
    "age": "Age", "height_in": "Height (in)", "weight_lb": "Weight (lb)", "years_since_draft": "Years since draft",
    "is_playoffs": "Playoff / play-in game", "early_era": "Early reporting era (2016-18)",
}
SEQ_LABELS = {
    "MIN": "minutes", "poss": "possessions used", "FGA": "shot attempts", "FTA": "free-throw attempts",
    "REB": "rebounds", "PF": "fouls", "days_rest": "team rest days", "back_to_back": "back-to-backs",
    "travel_km": "travel", "abs_tz_shift_h": "time-zone changes", "is_home": "home/away",
    "is_playoffs": "playoff games", "gap_days": "days between games", "days_before": "timing of those games",
}
SEQ_FAMILY = {"MIN": "workload", "poss": "workload", "FGA": "workload", "FTA": "workload", "REB": "workload",
              "PF": "workload", "days_rest": "schedule", "back_to_back": "schedule", "gap_days": "schedule",
              "days_before": "schedule", "is_home": "schedule", "travel_km": "travel",
              "abs_tz_shift_h": "travel", "is_playoffs": "context"}
FAMILY_LABELS = {"history": "Injury history", "workload": "Workload (summary features)",
                 "workload_seq": "Workload (last games, sequence)", "schedule": "Schedule",
                 "travel": "Travel", "bio": "Player bio", "context": "Context (playoffs, era)", "other": "Other"}


def static_label(name):
    if name.endswith("_missing"):
        return FEATURE_LABELS.get(name.removesuffix("_missing"), name.removesuffix("_missing")) + " (missing)"
    return FEATURE_LABELS.get(name, name)


def seq_label(name, n_games):
    return f"Last {n_games} games: {SEQ_LABELS.get(name, name)}"


def static_family(name):
    return family_of(name)


def seq_family(name):
    """Sequence inputs: workload ones get their own group so the report can separate
    what the Transformer learns from raw games from the hand-made aggregates."""
    f = SEQ_FAMILY.get(name, "other")
    return "workload_seq" if f == "workload" else f
