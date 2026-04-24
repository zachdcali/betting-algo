# Repository Structure

This is the current intended layout for the project.

## Source of Truth Areas

- `production/`
  Live tennis pipeline code. This is the operational path that fetches Bovada odds, builds TA features, runs inference, logs predictions, and settles results.
- `docs/production/`
  Authoritative production documentation and versioning rules.
- `dashboard/`
  Interactive operations dashboard code built on top of the production CSV logs and audit layer.
- `src/models/professional_tennis/`
  Historical preprocessing, dataset preparation, and model training scripts.
- `docs/modeling/`
  Experiment workflow, tuning conventions, and other modeling-process documentation.
- `production/tests/`
  Lightweight smoke tests for live-system behavior.

## Research / Experiment Areas

- `analysis_scripts/`
  Offline research, charting, bankroll studies, and backtests.
- `results/`
  Exported evaluation artifacts and model analysis outputs.
  Promoted release artifacts, candidates, and local experiments should follow the layout described in `results/professional_tennis/README.md`.
- `analysis_output/`, `out/`
  Derived experiment outputs.

## Data / Runtime Areas

- `data/`
  Local datasets, caches, and source material. Some tracked legacy files still exist here, but generated and sensitive data should stay out of Git going forward.
- `production/logs/`
  Runtime logs, feature snapshots, bet slips, and odds archives.
- `production/prediction_log.csv`
  Operational log snapshot for live prediction tracking.

## Cleanup Policy

To avoid deleting something important by accident:

- prefer moving ambiguous scripts into clearer folders over removing them
- prefer adding README files and ownership notes before large file moves
- treat `production/` and `src/models/professional_tennis/` as high-sensitivity areas
- treat older screenshots, debug images, and one-off analysis artifacts as cleanup candidates only after confirming they are not referenced anywhere

## Practical Rule

If a file affects live betting behavior, it should live under `production/` or be explicitly documented from `docs/production/`.

If a file is for operational visibility rather than decision logic, it should live under `dashboard/` or `docs/`.
