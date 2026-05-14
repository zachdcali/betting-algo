# AGENTS

Project instructions for future Codex/Claude-style maintenance sessions.

## Source Of Truth

- Live operations docs start at [docs/production/README.md](/Users/zachdodson/Documents/betting-algo/docs/production/README.md).
- Versioning rules live in [docs/production/VERSIONING.md](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md).
- Human-readable promoted model notes live in [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md).
- Side-model tuning workflow lives in [docs/modeling/EXPERIMENT_WORKFLOW.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/EXPERIMENT_WORKFLOW.md).
- Feature-engineering priorities and candidate feature-set guardrails live in [docs/modeling/FEATURE_ROADMAP.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/FEATURE_ROADMAP.md).
- The live orchestrator is [production/main.py](/Users/zachdodson/Documents/betting-algo/production/main.py).
- The active live features path is TA-based, not the older UTR path.
- CatBoost and LightGBM are supported only as side experiments for now; do not
  promote or wire them into live inference without explicit versioning and
  registry updates.

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
- Optional CatBoost/LightGBM screening uses the same chronological splits via:
  `tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py --mode fixed --only-boosters --booster-feature-mode both`.
  The `native_cat` feature mode is experimental and replaces one-hot
  surface/level/round/hand/country/handedness groups for side-model evaluation
  only.
- Optional XGBoost recency-weighting screening uses:
  `tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py --mode fixed --only-recency-xgb --recency-half-lives 3,5,8,12`.
  Validation/test rows stay unweighted; only training rows receive exponential,
  mean-normalized recency weights.
- Optional score/stat side features use:
  `tennis_env/bin/python src/models/professional_tennis/build_feature_set.py --feature-set performance_v1`.
  Train them with `run_side_experiments.py --feature-set performance_v1 --dataset-path <side_csv>`.
  Live pipeline runs may compute these fields and log side-model shadow
  predictions, but they must not affect betting decisions or promotion status
  without explicit registry/versioning work.

## Logging And Lineage

- `prediction_log.csv` is the operational log.
- `prediction_snapshots.csv`, `odds_history.csv`, and `logs/features_*.csv` are the immutable lineage layer.
- `python main.py --dry-run` should not start a betting session or write
  `logs/all_bets.csv`; use it for pipeline smoke checks when duplicate live
  bet logging would be risky.
- `BetTracker.log_bets()` should skip duplicate pending bets for the same match
  and bet side, so reruns do not double-log open recommendations.
- `logs/performance_v1_shadow_predictions.csv` is a side-model evaluation log,
  not an operational betting log.
- `logs/performance_v1_shadow_backfill.csv` is also side-model evidence only.
  Keep it separate from forward shadow logs and mark rows by backfill quality.
- Treat `logging_quality = snapshot_v2` rows as decision-grade.
- Treat `legacy_backfilled` rows as context, not exact lineage.
- Live predictions with noisy/defaulted features should still be logged with
  `features_complete=False` rather than skipped; clean accuracy excludes them,
  but settlement and bet reconciliation need the operational row.
- Settlement should enrich existing predictions; it should not recompute historical inference.
- `ta_match_unfinished` in settlement audit means TA still lists the matchup as
  upcoming/unfinished and no completed result has posted yet.

## Audit And Dashboard

- Audit CSVs under `production/logs/audit/` are first-class operational data:
  `run_history.csv`, `skipped_live_matches.csv`, `settlement_audit.csv`.
- The dashboard should read production CSVs directly and should not invent a shadow dataset.

## Commit Hygiene

- Avoid committing mutable local churn unless it is the point of the change:
  `production/prediction_log.csv`, ranking refresh files, ad hoc screenshots, and other live/generated data.
- Prefer code, docs, and explicit migrations in commits.
- Side-model experiment outputs should stay local under `results/professional_tennis/experiments/` unless there is a deliberate reason to commit a tiny text summary.
- Experiment output dirs are date/family/slug based and append
  `__run_HHMMSS` when a same-day slug already contains files; avoid relying on a
  generic same-day slug as a stable ledger.
- Candidate feature-set preprocessing should write versioned side outputs and
  should not silently replace the active 141-feature ML-ready dataset.
- Large local side datasets under `results/professional_tennis/feature_sets/`
  should remain uncommitted unless an explicit small manifest/summary is being
  promoted as documentation.

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
