# Feature Roadmap

Last reviewed: 2026-04-25.

This is the working roadmap for tennis feature engineering. It is meant to
keep future preprocessing/modeling passes organized before any new feature
shape is promoted into production.

## Current Feature Set

The active production schema is the 141-feature Tennis Abstract feature path.
Historical training comes from Jeff Sackmann match files, while live inference
uses Tennis Abstract profiles and match history plus current rankings data.

Current feature families:

- Ranking and points: ranks, rank difference, rank ratio, rank points, rank
  point difference, average rank/points.
- Player profile: age, height, hand, country buckets, peak-age flags.
- Tournament context: surface, level, round, draw size, season flags.
- Rolling form and activity: recent win rates, streaks, matches/sets in recent
  windows, rest/rust flags, rank movement, rank volatility.
- Surface/level/round history: surface experience and form, level win rates,
  round win rates, big-match win rates.
- Matchup history: head-to-head and handedness matchup features.

Feature-importance checks from the current XGBoost and Random Forest artifacts
show the model is still mostly driven by player-quality proxies:

- Ranking/points are the largest signal family.
- Form, level, surface, and activity/fatigue add useful but smaller signal.
- Country/hand/round/level one-hot categories are useful context, but not the
  main edge by themselves.
- H2H and seasonality are low weight, which is expected because they are sparse
  and often noisy.

The current training universe loaded by the experiment harness has 684,643
matches from 1990-01-01 through 2024-12-18. Of the 141 features, 131 are fully
populated after the standard rank filtering. The largest remaining missingness
is height (`Avg_Height` around 45%, each player height around 31%), then rank
points/age at much lower levels.

## Main Judgment

The next high-value work is probably not a new model library by itself. The
best current candidates are better player-state and match-performance features
from data we already have or can mirror in production from Tennis Abstract.

CatBoost and LightGBM remain worth screening, but their native categorical
advantage is limited with the current categorical set. We collapsed the one-hot
surface, level, round, hand, country, and handedness groups for the native-cat
screening; those are the categorical columns in the active 141-feature schema.
That did not make CatBoost clearly better because those categorical variables
are mostly low-cardinality context, while the strongest signal is still numeric
player quality and recent form.

CatBoost could become more interesting if we add higher-signal categoricals such
as player id, tournament id, entry type, seed bucket, or style buckets. That
needs strict chronological evaluation because high-cardinality player/tournament
categories can memorize historical strength and overstate performance.

## Highest-Value Feature Candidates

### 1. Score-Derived Form

This is the cleanest next feature family because `score` is present in the
historical Jeff Sackmann data and Tennis Abstract live histories.

Candidate features:

- Recent set win rate and set differential.
- Recent game win rate and game differential.
- Straight-set win/loss share.
- Deciding-set win rate.
- Tiebreak participation and tiebreak win rate.
- Retirement/walkover detection flags so unusual scorelines do not poison form.
- Surface-specific versions of the same features.

Why it could matter: a 6-4 6-4 loss to a top player is very different from a
6-1 6-2 loss, but the current model mostly sees both as a loss. These features
also give useful signal for players whose ranking lags current level.

Risk: score parsing must be shared between historical preprocessing and live TA
feature generation. Retirements, walkovers, unfinished matches, and match
tiebreaks need explicit tests.

### 2. Serve/Return Stat Form

Tennis Abstract raw match arrays already expose serve/return stat columns such
as aces, double faults, service points, first serves, first-serve points won,
second-serve points won, service games, and break points. The current live
scraper normalizes only metadata/profile/history fields and drops most of those
stats.

Historical coverage from `jeffsackmann_master_combined.csv`:

- 1990+: serve stat columns cover about 26% of all rows.
- 2010+: about 38%.
- 2020+: about 44%.
- ATP tour rows since 1990: about 87% stat coverage.
- Challenger/qualifier rows since 1990: about 57% stat coverage.
- Futures rows: effectively no serve-stat coverage.

Candidate features:

- Ace rate, double-fault rate, first-serve-in rate.
- First-serve points won, second-serve points won, service-points-won rate.
- Return-points-won rate inferred from opponent service stats.
- Break points saved/faced and break chances created/converted.
- Recent and surface-specific rolling windows with missingness flags.
- Serve/return strength differentials between players.

Why it could matter: serve and return quality are closer to tennis mechanics
than rank alone. They may also capture surface fit better than generic surface
win rate.

Risk: coverage is uneven by level and era. These should be added with explicit
availability flags and evaluated by level/source slices, not only aggregate log
loss.

### 3. Opponent-Adjusted Form And Elo-Style Ratings

Current form features treat wins mostly as wins. Beating a top-20 opponent and
beating a player ranked 400 should not update form equally.

Candidate features:

- Recent form weighted by opponent rank/points.
- Recent loss quality, e.g. close losses to highly ranked opponents.
- Chronological Elo or Glicko-style ratings from match history.
- Separate Elo ratings for overall, surface, and maybe level groups.
- Delta between current rank and model-internal rating.

Why it could matter: these are standard sports-modeling features and often
outperform raw rolling win percentage.

Risk: Elo implementation must be strictly chronological and deterministic.
Surface-specific ratings need shrinkage so grass and carpet samples do not get
too jumpy.

### 4. Rank/Points Shape Features

The model relies heavily on rank and points, so better transformations are
worth testing.

Candidate features:

- Log rank, log rank points, and log point ratio.
- Top-5, top-10, top-20, top-50, top-100 tier flags.
- Rank/points missingness flags instead of silent neutral defaults.
- Current rank minus historical/observed peak rank.
- Days or matches since peak rank, if computed chronologically.

Why it could matter: the difference between ranks 2 and 20 is not the same as
the difference between 102 and 120. Points and rank tiers encode that curve more
smoothly.

Risk: peak-rank features are easy live from TA profile, but historical versions
must use only information available before the match, not a future career peak.

### 5. Seed And Entry Context

Jeff Sackmann historical rows include seed and entry fields. Tennis Abstract raw
live match rows also include seed/entry fields, but the current scraper does not
persist them.

Candidate features:

- Player seeded flag, seed difference, seed bucket.
- Qualifier, lucky loser, wild card, protected ranking, direct acceptance flags.
- Qualifier-vs-main-draw-player interactions.

Why it could matter: entry type can capture hidden quality and fatigue. A
qualifier may be under-ranked but match-sharp, while a wild card can be weaker
than rank implies.

Risk: production availability must be verified for upcoming rows before adding
these to a live schema.

### 6. Calendar, Surface Transition, And Tournament Progression

The current model has rest windows and a binary surface-transition flag. It can
be made more tennis-specific.

Candidate features:

- Last surface and surface-switch direction, e.g. clay-to-grass or hard-to-clay.
- Consecutive-week tournament flag.
- Same-event matches already played this week.
- Recent minutes, games, or sets played when available.
- First match after long layoff, first match on new surface, first match at a
  new level.

Why it could matter: fatigue and adaptation are real in tennis, especially in
qualifying, late rounds, and quick surface changes.

Risk: current-tournament progression is more sensitive to date ordering and
post-start leakage. It needs extra safety checks for live inference.

## Weather And Elevation

Elevation is the more practical first external context feature. It is static by
tournament/city, easy to backfill once a tournament-location table exists, and
easy to use in production. Candidate features are tournament elevation in
meters, high-altitude flag, and elevation-by-surface interactions.

Weather is feasible but lower priority. Exact match-time weather is not
available from the current data, and the historical date heuristic can be off by
roughly a day for some rounds. If tested, use coarse and robust features:

- daily mean/high temperature rather than hourly weather
- 3-day rolling mean temperature as a robustness check
- humidity/wind only if the data source is reliable and backfillable
- indoor/roof flags where known
- missingness/source-quality flags

Professional approach: create the full historical backfill first, freeze a
versioned tournament-location/weather table, then evaluate offline with the same
chronological splits. Do not add weather to live inference before the same
feature can be reproduced historically and in production.

## Cutoffs, Recency Weighting, And Training Era

Training from 1990 onward is a reasonable default because modern tennis starts
around there for this data. It is also plausible that older matches should
count less because playing style, depth, surfaces, and scheduling have changed.

A hard cutoff and recency weighting answer related but different questions:

- Hard cutoff: "Ignore everything before this year."
- Recency weighting: "Keep old examples, but make recent examples matter more."

The professional workflow is to tune both with validation windows:

- hard cutoffs: 1990, 2000, 2005, 2010, 2015
- recency half-lives: 5, 8, 12, 16 years
- rolling train windows: last 8, 12, 16 years

Use the fixed split for quick screening, then blocked walk-forward windows for
sanity. Promotion should depend on log loss, Brier, ECE/calibration, and
market-edge slices, not just accuracy.

## Recommended Experiment Sequence

1. Create a side preprocessing path for feature-set candidates.
   It should write versioned local outputs rather than replacing
   `jeffsackmann_ml_ready_SURFACE_FIX.csv`.

2. Add shared score parsing utilities with unit tests.
   Use them from historical preprocessing and TA live feature code.

3. Screen score-derived form features.
   Train XGBoost first because it is the current strongest stable tree family.
   Also run CatBoost/LightGBM after the feature set exists.

4. Extend the TA scraper to preserve stat/seed/entry columns.
   Keep these as optional fields until production coverage is understood.

5. Screen serve/return stat features with missingness flags.
   Report aggregate metrics plus slices by source/level/stat availability.

6. Add opponent-adjusted form or Elo-style ratings.
   This is likely high value, but it is more implementation work than score
   parsing and needs careful leakage tests.

7. Run cutoff/recency grid on the best feature candidate.
   Compare hard cutoffs, recency half-lives, and rolling windows.

8. Consider elevation, then weather.
   Use a versioned tournament-location table before trying historical weather.

## Guardrails

- Every new model feature must have a historical implementation and a production
  TA implementation before it is considered live-ready.
- New feature shapes should be side experiments first. Do not silently replace
  the active 141-feature schema.
- Generated candidate datasets should stay local unless there is a deliberate
  reason to commit a small summary or manifest.
- Feature additions need chronological tests that prove current-match outcomes
  and future career information are not leaking into pre-match features.
- Model selection should use validation years only. Test-era metrics are for
  final reporting, not tuning.
- Keep feature descriptions close to the code and dashboard so confusing names
  are interpretable later.
