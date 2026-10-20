# Survival dataset report (proposal section 7)

Prediction unit: player-game landmark (pre-game). Follow-up: up to 30 days. Primary event: proxy-tier, time-loss injury. Labels end 2025-01-12.

Injuries are dated to the player's last game before the injured-list placement when it is within 14 days (91% of injury episodes), otherwise to the placement date.

- Landmarks built: **273,667**; kept: **229,064**; players: **1,353**; seasons: **9**
- Duplicate player-game rows: **0**
- Uncertain injury episodes (never events; end follow-up as censoring): **1,265**
- Model features: **42**

## Excluded landmarks

| reason | landmarks |
|---|---|
| game after end of injury labels (no outcome) | 44,603 |
| no venue/travel row | 0 |

## By split

| split | landmarks | players | event_landmarks | y14_positive |
|---|---|---|---|---|
| supplementary | 12,335 | 519 | 494 | 311 |
| test | 28,206 | 572 | 1,159 | 715 |
| train | 160,783 | 1,085 | 5,425 | 3,296 |
| val | 27,740 | 541 | 1,472 | 882 |

## By season

`event_landmarks` counts landmarks whose follow-up ends in a target injury (several landmarks before the same injury each count). `reached_by_landmark` = distinct target injuries that at least one landmark's follow-up reaches.

| SEASON | split | players | landmarks | event_landmarks | y14_positive | evaluable14 | target_episodes | reached_by_landmark |
|---|---|---|---|---|---|---|---|---|
| 2016-17 | train | 486 | 27,876 | 610 | 371 | 23,563 | 65 | 65 |
| 2017-18 | train | 541 | 27,836 | 702 | 400 | 22,817 | 73 | 68 |
| 2018-19 | train | 530 | 27,862 | 832 | 496 | 23,059 | 95 | 89 |
| 2019-20 | train | 530 | 24,105 | 917 | 570 | 19,739 | 122 | 111 |
| 2020-21 | train | 540 | 25,054 | 1,259 | 783 | 20,585 | 152 | 143 |
| 2021-22 | train | 606 | 28,050 | 1,105 | 676 | 22,460 | 135 | 130 |
| 2022-23 | val | 541 | 27,740 | 1,472 | 882 | 22,687 | 158 | 154 |
| 2023-24 | test | 572 | 28,206 | 1,159 | 715 | 22,544 | 138 | 134 |
| 2024-25 | supplementary | 519 | 12,335 | 494 | 311 | 8,881 | 68 | 62 |

## How follow-up ended

| follow-up ended by | landmarks |
|---|---|
| horizon | 142,302 |
| other_injury | 45,021 |
| season_end | 29,239 |
| target_event | 8,550 |
| label_end | 3,952 |

## Follow-up and time to event (days)

| count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| 229,064 | 23.5 | 10 | 0 | 17 | 30 | 30 | 30 |

| count | mean | std | min | 25% | 50% | 75% | max |
|---|---|---|---|---|---|---|---|
| 8,550 | 12.2 | 9.1 | 0 | 4 | 11 | 20 | 30 |

## Target injuries by site (reached by a landmark)

| site | target_episodes |
|---|---|
| hamstring | 302 |
| groin_adductor | 222 |
| calf | 206 |
| achilles | 106 |
| quad | 89 |
| hip_flexor | 31 |

## History available per landmark

| count | mean | std | min | 10% | 50% | 90% | max |
|---|---|---|---|---|---|---|---|
| 229,064 | 183 | 151 | 0 | 19 | 145 | 412 | 707 |

## Missing values in model features

Missing values are imputed with the training-set median plus a missing-indicator column (linear and Cox models); tree models handle them natively.

| feature | % missing |
|---|---|
| days_since_last_injury | 23.4 |
| acwr_min | 16.3 |
| years_since_draft | 15.8 |
| usage_per_min_28d | 2.9 |
| ewma_min_short | 2.1 |
| ewma_min_long | 2.1 |
| ewma_ratio_min | 2.1 |
| days_rest | 1.3 |
| prev_game_min | 0.6 |
| days_since_prev_game | 0.6 |
