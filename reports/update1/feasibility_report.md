# Feasibility report (Update 1)

Label source: Kaggle 2016-2025 NBA injury data (Pro Sports Transactions injured-list transactions), `mirror_injuries_clean.csv`, 2016-10-25 to 2025-01-12.

Tiers are nested (strict ⊂ proxy ⊂ broad). `_tl` = time-loss episodes (player missed ≥1 team game). `bare_il` = uncertain episodes whose only note is a reason-less 'placed on IL' (likely roster management; excluded).

## Episodes per season

| season | split | label_coverage | players | player_games | strict | proxy | broad | strict_tl | proxy_tl | broad_tl | uncertain | bare_il | other_injury |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2016-17 | train | full | 486 | 27876 | 44 | 73 | 299 | 38 | 65 | 266 | 346 | 342 | 94 |
| 2017-18 | train | full | 541 | 27836 | 39 | 76 | 377 | 37 | 73 | 362 | 421 | 418 | 161 |
| 2018-19 | train | full | 530 | 27862 | 47 | 95 | 416 | 47 | 95 | 413 | 196 | 194 | 125 |
| 2019-20 | train | full | 530 | 24105 | 49 | 124 | 437 | 48 | 122 | 431 | 88 | 87 | 170 |
| 2020-21 | train | full | 540 | 25054 | 54 | 154 | 497 | 52 | 152 | 492 | 44 | 41 | 147 |
| 2021-22 | train | full | 606 | 28050 | 36 | 135 | 631 | 36 | 135 | 627 | 117 | 100 | 217 |
| 2022-23 | val | full | 541 | 27740 | 47 | 160 | 590 | 47 | 158 | 587 | 29 | 27 | 191 |
| 2023-24 | test | full | 572 | 28206 | 60 | 139 | 681 | 59 | 138 | 680 | 23 | 1 | 201 |
| 2024-25 | supplementary | partial (to 2025-01-12) | 569 | 28240 | 32 | 69 | 254 | 31 | 68 | 252 | 1 | 0 | 93 |
| 2025-26 | supplementary | none | 582 | 28698 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| TRAIN (6 seasons) |  |  | 1085 | 160783 | 269 | 657 | 2657 | 258 | 642 | 2591 | 1212 | 1182 | 914 |
| TOTAL |  |  | 1478 | 273667 | 408 | 1025 | 4182 | 395 | 1006 | 4110 | 1265 | 1210 | 1399 |

## Proxy-tier episodes by site

| site | proxy_episodes |
|---|---|
| hamstring | 333 |
| groin_adductor | 231 |
| calf | 222 |
| achilles | 113 |
| quad | 94 |
| hip_flexor | 32 |

## Player-name matching

Match rate on injury rows: **99.9%**

| match_method | rows |
|---|---|
| exact | 7415 |
| unmatched | 7 |
| fuzzy | 4 |

## Event summary

- Training-season events, proxy tier (time-loss): **642**
- Training-season events, strict tier (time-loss): **258**
- Training-season events, broad tier (time-loss): **2591**
- Validation (2022-23) proxy time-loss events: **158**
- Test (2023-24) proxy time-loss events: **138**
- Average proxy time-loss events per training season: **107**