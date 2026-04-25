# Experiment Workflow

This is the project-specific workflow for side-model tuning and walk-forward evaluation.

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
