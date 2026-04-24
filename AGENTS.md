# AGENTS

Project instructions for future Codex/Claude-style maintenance sessions.

## Source Of Truth

- Live operations docs start at [docs/production/README.md](/Users/zachdodson/Documents/betting-algo/docs/production/README.md).
- Versioning rules live in [docs/production/VERSIONING.md](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md).
- Human-readable promoted model notes live in [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md).
- Side-model tuning workflow lives in [docs/modeling/EXPERIMENT_WORKFLOW.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/EXPERIMENT_WORKFLOW.md).
- The live orchestrator is [production/main.py](/Users/zachdodson/Documents/betting-algo/production/main.py).
- The active live features path is TA-based, not the older UTR path.

## Model Versioning Rules

- Keep versioning separate by family:
  `nn`, `xgboost`, and `random_forest`.
- Do not silently overwrite a promoted production artifact.
- It is acceptable to keep a retrained model as a registry-tracked candidate when it has not yet earned promotion.
- When promoting a new model version:
  1. create a stable versioned artifact copy
  2. archive the previous promoted artifact with a stable versioned filename
  3. update [production/models/model_registry.json](/Users/zachdodson/Documents/betting-algo/production/models/model_registry.json)
  4. update [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md)
- Use [production/models/validate_registry.py](/Users/zachdodson/Documents/betting-algo/production/models/validate_registry.py) after registry changes.
- A changed train/validation/test protocol warrants at least a minor version bump.
- Probability calibration for the NN should be tracked separately from the base NN artifact version.

## Training Rules

- Use chronological splits, never random splits.
- Current standard split for tree/NN retrains:
  train `< 2022-01-01`, validation `2022`, test `>= 2023-01-01`.
- Do not use the test era for early stopping, model selection, or threshold tuning.
- If an old archived artifact cannot be evaluated because the feature schema changed, handle that comparison gracefully and do not crash the training script.

## Logging And Lineage

- `prediction_log.csv` is the operational log.
- `prediction_snapshots.csv`, `odds_history.csv`, and `logs/features_*.csv` are the immutable lineage layer.
- Treat `logging_quality = snapshot_v2` rows as decision-grade.
- Treat `legacy_backfilled` rows as context, not exact lineage.
- Settlement should enrich existing predictions; it should not recompute historical inference.

## Audit And Dashboard

- Audit CSVs under `production/logs/audit/` are first-class operational data:
  `run_history.csv`, `skipped_live_matches.csv`, `settlement_audit.csv`.
- The dashboard should read production CSVs directly and should not invent a shadow dataset.

## Commit Hygiene

- Avoid committing mutable local churn unless it is the point of the change:
  `production/prediction_log.csv`, ranking refresh files, ad hoc screenshots, and other live/generated data.
- Prefer code, docs, and explicit migrations in commits.
- Side-model experiment outputs should stay local under `results/professional_tennis/experiments/` unless there is a deliberate reason to commit a tiny text summary.

## When To Update This File

Update `AGENTS.md` when a future session changes:

- model-version promotion rules
- experiment folder conventions
- experiment harness commands
- the standard chronological split or walk-forward policy
- logging/lineage expectations that future sessions need to know up front

## Safety Notes

- For delayed production runs, inference should only happen before the configured pre-start cutoff and before the current match appears in TA history as completed.
- For future cloud scheduling, hourly odds capture is good, but settlement should remain a separate concern.
