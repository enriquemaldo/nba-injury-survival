# NBA Non-Contact Injury Survival Analysis

ECE 5904 · Enrique Maldonado · Virginia Tech

Pipeline for estimating NBA non-contact soft-tissue injury hazard from public
game, schedule and injury data (Transformer + DeepSurv, SHAP explanations).

**Status.** Phase 1 (injury labels, feasibility, travel features) and Phase 2
(landmark survival dataset, workload features, baseline models) are complete.
Models are evaluated on the validation season only; the test season is untouched.

```
src/
  config.py                  seasons, date windows, chronological split
  ingest/gamelogs.py         nba_api player-game logs, 2016-17 to 2025-26
  ingest/kaggle_injuries.py  load the injury-label dataset
  ingest/pst_scrape.py       Pro Sports Transactions scraper (site blocks scripts; not used)
  labels/taxonomy.py         injury tiers - every rule in one file
  labels/player_match.py     transaction names -> nba_api PLAYER_ID
  labels/episodes.py         transactions -> injury episodes
  labels/build.py            runs taxonomy + matching + episodes
  ingest/bio.py              player age, height, weight, draft year by season
  features/travel.py         venue per game + travel/schedule features
  features/workload.py       pre-game workload, scheduling and injury-history features
  survival/dataset.py        landmark survival dataset (time-to-event, censoring) + report
  models/coxph.py            Cox proportional-hazards model (strata, robust SEs)
  models/metrics.py          C-index, AUC/Brier at 14 days, calibration, Kaplan-Meier
  models/baselines.py        logistic regression, XGBoost, Cox + sensitivity runs
  feasibility.py             per-season event counts and summary report
  evidence.py                figures, PDF summary, label-audit sheet
tests/                       offline tests (labels, travel, Cox model, metrics, leakage)
reports/update1/             Update 1 reports, figures and label-audit results
reports/update2/             Update 2: survival dataset report and baseline results
reference/arenas.csv         season-specific home arenas (hand-verified)
reference/neutral_sites.csv  bubble, international and NBA Cup Las Vegas games
```

## Data sources

| Source | Provides | Coverage |
|---|---|---|
| [nba_api](https://github.com/swar/nba_api) | player-game logs, schedules, player IDs | 2016-17 to 2025-26 |
| `reference/` tables (this repo) | arena coordinates and time zones by team-season; neutral-site games | 2016-17 to 2025-26 |
| [2016-2025 NBA Injury Data (Kaggle)](https://www.kaggle.com/datasets/jacquesoberweis/2016-2025-nba-injury-data) | Pro Sports Transactions injured-list transactions (date, team, player, note) | 2016-10-25 to 2025-01-12 |

The injury dataset contains injured-list transactions only (no day-to-day
"missed game" notes), so the event of interest is an injury that led to an
injured-list placement. A Pro Sports Transactions scraper is included, but the
site blocks automated requests, so the published dataset is used instead.

## Chronological split

| Split | Seasons |
|---|---|
| Train | 2016-17 to 2021-22 |
| Validation | 2022-23 |
| Test | 2023-24 |
| Supplementary | 2024-25, 2025-26 (labels incomplete; to be labeled from official injury reports) |

The split was moved back one season from the proposal because the injury
labels end in January 2025; this keeps train, validation and test on one
consistent label source.

---

## Running the pipeline

### Setup (once)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python tests/test_pipeline.py          # every line should say "ok"
```

Always run scripts from the project root as modules (`python -m src....`).

### 1. Game logs (~10 min)

```bash
python -m src.ingest.gamelogs
```

Downloads regular-season, play-in (2020 onward) and playoff games. Writes `data/raw/gamelogs_all.csv` and `data/raw/nba_players.csv`.
Re-running skips seasons already downloaded; if stats.nba.com times out, run
it again.

### 2. Injury dataset

Download the CSV from the Kaggle page above, save it as
`data/raw/kaggle_injuries_2016_2025.csv`, then:

```bash
python -m src.ingest.kaggle_injuries
```

### 3. Labels, player IDs, episodes

```bash
python -m src.labels.build
```

Outputs in `data/interim/`:

| File | What it is |
|---|---|
| `injury_episodes.csv` | **one row per injury episode — the main label table** |
| `injury_rows_labeled.csv` | every "player out" transaction with tier, site, PLAYER_ID |
| `unmatched_names.csv` | names the matcher couldn't resolve |
| `fuzzy_matches.csv`, `uncertain_notes.csv`, `proxy_notes_by_site.csv` | audit files |

Unresolved names can be fixed by hand in `data/interim/manual_name_map.csv`
(`raw_name,season,PLAYER_ID`; blank season = all seasons).

### 4. Feasibility report

```bash
python -m src.feasibility
```

Writes `data/processed/feasibility_report.md`: events per season by tier,
split totals, events by body site, name-match rate and the split totals.

### 5. Travel and schedule features

```bash
python -m src.features.travel
```

Gives every team-game a venue: the home team's arena **for that season**
(Pistons 2017-18, Bucks 2018-19, Warriors 2019-20, Raptors in Tampa 2020-21,
Clippers at the Intuit Dome 2024-25), or a neutral site (2020 bubble,
Mexico City / London / Paris / Berlin games, NBA Cup knockout games in Las
Vegas). Then computes, per team in date order: `travel_km` (great-circle
distance from the previous venue; season opener from the home arena),
`tz_shift_h` (+ = moved east; handles DST and Arizona), `days_rest`,
`back_to_back`, `km_last_7d`, `km_last_14d` and `games_last_7d`.

Outputs `data/processed/travel_team_games.csv` (join to player-games on
`GAME_ID` + team) and `data/processed/travel_report.md`, which confirms that
every listed neutral-site game was found in the game logs and that every
team-game got a venue.

### 6. Evidence pack

```bash
python -m src.evidence             # figures + PDF summary + 100-row audit sheet
python -m src.evidence --score     # after filling in the audit sheet
```

Outputs in `data/processed/evidence/`: three figures, `evidence_summary.pdf`
and `label_audit.csv`. To audit, mark each row `1` (tier correct) or `0` in the
`correct` column (with the right tier in `correct_tier`), save as CSV, and run
`--score`.

### 7. Player bio (age, height, weight)

```bash
python -m src.ingest.bio
```

One request per season (`LeagueDashPlayerBioStats`), cached in `data/raw/bio/`.
If it fails, the next steps still run without these four features.

### 8. Landmark survival dataset

```bash
python -m src.survival.dataset
```

One row per player-game (**pre-game landmark**: features use only earlier
games). Follow-up lasts up to 30 days and ends at the first of: the next
target injury (event), any other injury (censored), the team's season end,
or the end of the injury labels. Injuries are dated to the player's last game
before the injured-list placement when it is within 14 days (the transaction
usually lags the injury by ~2 days). Also stores `y14` (target injury within
14 days) for the classification baselines, and strict/broad event variants.
Writes `data/processed/survival/landmarks.csv.gz` and `dataset_report.md`
(the descriptive analysis from proposal section 7).

Features (all leakage-safe; tested): minutes and possessions used over the
last 7/14/28 days, games in the last 3/7/14 days, back-to-back appearances,
EWMA minutes (short/long) and acute-to-chronic ratios, consecutive games,
team games missed, season-to-date and since-2016 load, prior injuries,
days since last injury, rest, travel, time-zone change, home/neutral,
playoffs, age, height, weight, years since draft, and an early-reporting-era flag.

### 9. Baseline models

```bash
python -m src.models.baselines
```

Logistic regression and XGBoost (14-day injury classification) and a Cox
model (time-to-event, standard errors clustered by player, adjusted for the
early reporting era, plus a season-stratified refit). Trained on 2016-17 to
2021-22, evaluated on 2022-23: C-index, AUC and Brier at 14 days,
calibration, Kaplan-Meier risk groups, player-bootstrap 95% CIs, and
reference models using one feature family each. Repeated for two
sensitivity analyses (train from 2018-19; exclude 2019-20 and 2020-21).
Outputs in `data/processed/models/`.

`--test` evaluates the sealed test season. Use it **once, at the end of the
project**, after every model choice is final.

If `xgboost` is not installed (on macOS it may also need `brew install libomp`),
scikit-learn's gradient-boosted trees are used instead and the report says so.

### 10. Update 2 evidence pack

```bash
python -m src.models.evidence
```

Writes `fig_cindex.png` (every model's C-index with 95% CI) and
`evidence_summary.pdf` (key numbers + all Update 2 figures) to
`data/processed/models/`.

---

## Label definitions

### Tiers (nested: strict ⊂ proxy ⊂ broad)

| Tier | Definition |
|---|---|
| strict | hamstring / calf / Achilles / groin-adductor / quadriceps / hip-flexor + strain, pull, tear or rupture; no contact language |
| proxy | same sites + any soft-tissue complaint (sore, tight, tendinitis, "injury"); no contact language — **primary target** |
| broad | any lower-extremity musculoskeletal injury, contact or not |
| other_injury | upper body, trunk, head |
| non_injury | illness, rest, health-and-safety protocols, personal reasons |
| uncertain | no stated reason (e.g. a bare "placed on IL") |

Contusions and bruises count as contact language, so a bruised quadriceps is
broad-tier, never strict or proxy.

### Episodes

Transactions for the same player belong to one episode until he plays again.
`games_missed` = team games between his last game before onset and his return.
The primary event is a **time-loss** proxy-tier episode (`games_missed ≥ 1`).

### Known issue: reporting change across seasons

Reason-less injured-list placements fall from ~340-420 per season in 2016-18
to almost none by 2023-24, while proxy events rise. Early seasons appear to
record many injuries without a reason, so the trend reflects record-keeping
rather than injury risk. Handling: an early-reporting-era indicator in every
model, a season-stratified Cox refit, and a sensitivity analysis trained on
2018-19 onward.

## Next steps

- DeepSurv (tabular) → Transformer + DeepSurv (game sequences)
- Within-player analysis: does recent workload predict *when* a player gets
  hurt? (The baselines' C-index is driven mostly by *which* players get hurt.)
- Feature-family ablations, SHAP attributions, Streamlit dashboard
- Supplementary labels for 2024-26 from official NBA injury reports
