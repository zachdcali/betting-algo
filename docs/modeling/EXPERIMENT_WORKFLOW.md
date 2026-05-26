# Experiment Workflow

This is the project-specific workflow for side-model tuning and walk-forward evaluation.

Feature-engineering priorities and candidate feature-set guardrails live in
[Feature Roadmap](/Users/zachdodson/Documents/betting-algo/docs/modeling/FEATURE_ROADMAP.md).

## Goals

- test model tweaks without touching promoted production models
- keep candidate artifacts and summaries organized
- make future comparisons reproducible

## Where To Put Things

- committed experiment code:
  `src/models/professional_tennis/`
- local experiment outputs:
  `results/professional_tennis/experiments/`
- promoted production versions:
  `production/models/model_registry.json`
  plus `docs/production/MODEL_RELEASES.md`

## Current Harness

Run side experiments with:

```bash
tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py --mode all
```

Build the optional score/stat side feature set with:

```bash
PYTHONUNBUFFERED=1 tennis_env/bin/python \
  src/models/professional_tennis/build_feature_set.py \
  --feature-set performance_v1
```

This writes a local dataset under
`results/professional_tennis/feature_sets/performance_v1/<date>/` and checks row
alignment against the canonical 141-feature ML-ready CSV before saving. It does
not replace `data/JeffSackmann/jeffsackmann_ml_ready_SURFACE_FIX.csv`.

Modes:

- `fixed`
  one honest fixed split:
  train `< 2022-01-01`, val `2022`, test `>= 2023-01-01`
- `blocked`
  blocked walk-forward windows using train/val/test year blocks
- `all`
  both

By default the harness runs the NN/XGBoost side baselines. CatBoost and
LightGBM side experiments are available as optional booster families:

```bash
tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py \
  --mode fixed \
  --only-boosters \
  --booster-feature-mode both
```

Useful booster flags:

- `--include-boosters`
  add CatBoost/LightGBM to the normal NN/XGBoost run
- `--only-boosters`
  skip NN/XGBoost and run only the optional booster families
- `--booster-families catboost,lightgbm`
  choose one or both optional booster families
- `--booster-feature-mode one_hot|native_cat|both`
  compare the production 141-feature one-hot schema against a native categorical
  view that replaces surface/level/round/hand/country/handedness one-hot groups
  with categorical columns
- `--feature-set base_141|performance_v1`
  choose the canonical 141-feature set or the optional score/stat feature set
- `--dataset-path <csv>`
  train on a versioned side dataset instead of the canonical ML-ready CSV
- `--only-xgb`
  run only the standard XGBoost configs from the default NN/XGBoost block

Same-day output directories append `__run_HHMMSS` when a matching experiment
slug already has files, and batch summary directories include the selected
booster flags. This keeps reruns from silently overwriting earlier side-model
artifacts.

Run XGBoost recency-weighting experiments with:

```bash
tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py \
  --mode fixed \
  --only-recency-xgb \
  --recency-half-lives 3,5,8,12
```

The recency weights are exponential and mean-normalized. A half-life of `8`
means a training match eight years older than the most recent training match
gets half the raw weight before normalization. Validation and test rows remain
unweighted.

XGBoost side runs save `feature_importance.csv` next to `model.json`.

## 2026-04-25 Performance V1 Feature Screening

`performance_v1` was tested as a side feature set only. No production artifact
was promoted.

The feature set appends 57 score/stat features to the active 141-feature
schema, for 198 model features. It uses data already present in the local Jeff
Sackmann aggregate: score, minutes, aces, double faults, serve points,
first/second serve points won, service games, and break points. Tennis Abstract
also exposes these columns in live player match arrays, and the scraper now
preserves them for future parity work.

Local side dataset:

```text
results/professional_tennis/feature_sets/performance_v1/2026-04-25/jeffsackmann_ml_ready_performance_v1.csv
```

Fixed split result, using train `< 2022-01-01`, validation `2022`, test
`>= 2023-01-01`:

- Base XGBoost side baseline from 2026-04-24:
  test log loss `0.605837`, accuracy `0.665021`, AUC `0.730710`
- `performance_v1` XGBoost depth5:
  test log loss `0.601662`, accuracy `0.670548`, AUC `0.736136`
- `performance_v1` XGBoost depth6:
  test log loss `0.601464`, accuracy `0.670136`, AUC `0.736525`
- `performance_v1` LightGBM native categorical:
  test log loss `0.601656`, accuracy `0.670798`, AUC `0.736161`
- `performance_v1` CatBoost one-hot:
  test log loss `0.601993`, accuracy `0.670423`, AUC `0.735749`
- `performance_v1` XGBoost depth5 with 8-year recency weighting:
  test log loss `0.601044`, accuracy `0.671281`, AUC `0.737051`
- `performance_v1` XGBoost depth5 with 12-year recency weighting:
  test log loss `0.600838`, accuracy `0.671192`, AUC `0.737407`
- Best `performance_v1` NN screen:
  test log loss `0.603891`, accuracy `0.667382`, AUC `0.732995`

Blocked sanity check for `performance_v1` XGBoost depth5 with 12-year recency
weighting:

- Mean test log loss across four blocked windows: `0.601320`
- Prior 2026-04-24 base 141-feature recency XGBoost 12-year mean test log loss:
  `0.605251`

Feature-importance takeaway from the fixed 12-year recency run:
`Game_WinRate_Last10_Diff` was the fourth-highest gain feature after rank and
rank-points features. `Game_WinRate_90d_Diff`, `Stat_Matches_Last10_Diff`, and
`Service_Points_Won_Last10_Diff` also ranked highly. This suggests score/stat
features are carrying real tennis signal, not just adding noise.

Remaining risks:

- Forward live-shadow logging is available through the production orchestrator.
  It writes `production/logs/performance_v1_shadow_predictions.csv` when local
  side artifacts are present. The default forward set logs one-hot
  `performance_v1` candidates for XGBoost, CatBoost, LightGBM, and the best NN
  screen. These rows do not affect betting decisions, `prediction_log.csv`, or
  the production model registry.
- When `auto_settle.py` settles the corresponding operational prediction, it
  also copies `actual_winner`, score, and correctness fields onto matching
  forward shadow rows. This is scoring already-logged probabilities, not
  recomputing historical inference.
- Serve/stat coverage is uneven by source and era, especially for Futures.
- NN performance improved but still trails tree models on this fixed split.

To enable this specific side model locally, keep the side artifact under:

```text
results/professional_tennis/experiments/2026-04-25/xgboost/performance_v1__xgb_depth5_recency_hl_12y/
```

The XGBoost shadow predictor requires both `model.json` and
`feature_medians.json`. The CatBoost, LightGBM, and NN one-hot shadow candidates
reuse the same training-split medians for missing live features. Do not use live
rows or the test era to choose fill values.

For a controlled historical live-log check, backfill only exact feature-snapshot
rows into a separate side log:

```bash
tennis_env/bin/python production/shadow/backfill_performance_v1.py
```

The command reads `prediction_log.csv` with `write=False`, joins exact
`logs/features_*.csv` rows, recomputes only the new score/stat features from TA
history as of the logged match date, and writes
`production/logs/performance_v1_shadow_backfill.csv`. Treat rows with
`backfill_quality = snapshot_v2_performance_v1_backfill` as useful side
evidence, not as a production promotion by itself.

Initial 2026-04-26 exact-snapshot backfill:

- Cohort: `180` settled `snapshot_v2` rows with exact feature snapshots.
- `performance_v1` shadow XGBoost: accuracy `67.222%`, log loss `0.616584`,
  Brier `0.213315`, AUC `0.733135`.
- Current live XGBoost on the same rows: accuracy `66.667%`, log loss
  `0.631443`, Brier `0.219144`, AUC `0.713976`.
- Market on the same rows: accuracy `63.889%`, log loss `0.622425`, Brier
  `0.216457`, AUC `0.719122`.

This is encouraging, but it is still a small, rescraped backfill cohort. Keep
collecting forward shadow rows before promotion.

## 2026-04-24 Recency-Weighted XGBoost Screening

Recency weighting was tested as a side experiment only. No production artifact
was promoted.

Fixed split result, using the standard train `< 2022-01-01`, validation `2022`,
test `>= 2023-01-01` protocol:

- Unweighted XGBoost side baseline `xgb_depth5_balanced_regularized`: test log
  loss `0.605837`, accuracy `0.665021`, AUC `0.730710`
- 3-year half-life: test log loss `0.606398`, accuracy `0.665111`, AUC
  `0.730449`
- 5-year half-life: test log loss `0.605662`, accuracy `0.667132`, AUC
  `0.731388`
- 8-year half-life: test log loss `0.605131`, accuracy `0.667221`, AUC
  `0.731787`
- 12-year half-life: test log loss `0.605325`, accuracy `0.666291`, AUC
  `0.731531`

Blocked walk-forward sanity check:

- Existing unweighted XGBoost baseline mean test log loss across the four
  blocked windows: `0.605330`
- 8-year half-life mean test log loss: `0.605339`
- 12-year half-life mean test log loss: `0.605251`

Takeaway: recency weighting is more promising than switching libraries because
it gives a small fixed-split lift and is roughly stable in blocked windows
without changing the production feature schema. The effect is still small; treat
`8` to `12` years as the next tuning neighborhood, not as a promotion decision.

## 2026-04-24 Booster Screening

CatBoost and LightGBM were tested as side models only. No production artifact was
promoted.

Fixed split result, using the standard train `< 2022-01-01`, validation `2022`,
test `>= 2023-01-01` protocol:

- XGBoost side baseline `xgb_depth5_balanced_regularized`: test log loss
  `0.605837`, AUC `0.730710`, Brier `0.209543`
- CatBoost one-hot, longer 1400-iteration exploratory run: test log loss
  `0.605144`, AUC `0.731399`, Brier `0.209272`
- CatBoost one-hot, practical screening config: test log loss `0.605719`,
  AUC `0.730648`, Brier `0.209517`
- LightGBM native categorical screening config: test log loss `0.605621`,
  AUC `0.731021`, Brier `0.209448`
- CatBoost native categorical screening config: test log loss `0.606102`,
  AUC `0.730217`, Brier `0.209669`

Blocked walk-forward sanity check:

- Existing XGBoost baseline mean test log loss across the four blocked windows:
  `0.605330`
- LightGBM native categorical mean test log loss: `0.605731`
- CatBoost one-hot screening mean test log loss: `0.606122`

Takeaway: CatBoost/LightGBM are plausible candidates, but the lift is small and
not yet robust across blocked windows. The native categorical view is useful for
experimentation, but it is not clearly superior enough to justify a production
feature-schema change by itself.

## Rules

- do not overwrite promoted live artifacts while tuning
- save promising side models as local experiment outputs first
- only promote after explicit review of offline metrics and live-shadow evidence
- prefer log loss, Brier, and ECE over raw accuracy when choosing probability models

## When To Update Docs

Update this file and [AGENTS.md](/Users/zachdodson/Documents/betting-algo/AGENTS.md) whenever:

- the experiment harness command changes
- the experiment output folder convention changes
- the standard split / walk-forward policy changes
- promotion rules for side models change
