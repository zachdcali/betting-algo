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
