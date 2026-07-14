# Model Releases

This file is the human-readable release ledger for promoted production models.

Registry truth still lives in [model_registry.json](/Users/zachdodson/Documents/betting-algo/production/models/model_registry.json), but every promoted model family version should also be explained here in plain English.

## Current Production Versions

- Neural Network: `v1.2.3` (weights from `v1.2.1`; pipeline-only releases
  `v1.2.2` and `v1.2.3`)
- XGBoost: `v1.2.2` (weights from `v1.2.0`; pipeline-only releases
  `v1.2.1` and `v1.2.2`)
- Random Forest: `v1.2.2` (weights from `v1.2.0`; pipeline-only releases
  `v1.2.1` and `v1.2.2`)

## Current Candidate Versions

- Neural Network: `v1.3.0`

## Release Notes

### 2026-07-14

#### Registry generation ordering

- The registry document now has an explicit monotonic generation and effective
  timestamp. Database promotion state follows that generation rather than file
  import order, so replaying an older registry cannot demote a newer release.
- No promoted model, candidate status, artifact path, or artifact byte changed
  in this registry-contract release.

### 2026-07-13

#### Registry contract metadata

- Promoted artifacts remain at their existing stable paths; no model bytes were
  renamed or replaced.
- The registry now identifies the ordered representation as
  `base_141@1.0.0` and pins its schema SHA-256.
- Historical training semantics are explicitly recorded as
  `sackmann_historical_legacy@1.0.0`; active live semantics are recorded as
  `ta_live_legacy@3.0.0`. Their mismatch is a retraining stop gate, not hidden
  behind a filename.
- `base_141_shared@1.0.0` is reserved for the future parity-tested feature
  kernel and is not active.
- New artifacts will use family/version directories such as
  `candidates/xgboost/v2.0.0/model.json`; registry metadata binds the feature
  contract, dataset, calibration, and hashes.

### 2026-07-11

#### Pipeline-only releases

- NN `v1.2.3`, XGBoost `v1.2.2`, and Random Forest `v1.2.2` identify the
  week-boundary and store-ingest feature regime introduced on July 11.
- Model weights are unchanged from NN `v1.2.1` and XGB/RF `v1.2.0`.
- These versions improve operational lineage; they are not claims of a newly
  trained or better-performing artifact.

### 2026-04-23

#### Neural Network candidate `v1.3.0`

- Retrained with the corrected chronological split:
  train `< 2022-01-01`, validation `2022`, test `>= 2023-01-01`
- Test metrics:
  accuracy `0.6622`, AUC `0.7262`, log loss `0.6098`, Brier `0.2112`, ECE `0.0142`
- Stored as a candidate, not the promoted live NN, because it was slightly worse than `v1.2.1` on offline metrics and the live NN still needs more calibration / extreme-prediction analysis before promotion
- Practical takeaway: the corrected split retrain exists and is organized, but we are not pretending it earned automatic promotion

#### XGBoost `v1.2.0`

- Promoted because the training protocol changed to an honest chronological split:
  train `< 2022-01-01`, validation `2022`, test `>= 2023-01-01`
- Validation winner: `balanced_depth6`
- Test metrics:
  accuracy `0.6649`, AUC `0.7302`, log loss `0.6064`
- Archived prior promoted artifact as `v1.1.0`
- Practical takeaway: the fixed split barely moved headline performance, which is reassuring because it means the earlier result was not wildly inflated

#### Random Forest `v1.2.0`

- Promoted because the training protocol changed to an honest chronological split:
  train `< 2022-01-01`, validation `2022`, test `>= 2023-01-01`
- Validation winner: `rf_depth15_currentish`
- Test metrics:
  accuracy `0.6603`, AUC `0.7251`, log loss `0.6105`
- Archived prior promoted artifact as `v1.1.0`
- Practical takeaway: same story as XGB, with small metric movement but cleaner evaluation discipline

## Promotion Checklist

When promoting a new model family version:

- create a versioned artifact copy
- archive the previous promoted artifact with a stable versioned filename
- update [model_registry.json](/Users/zachdodson/Documents/betting-algo/production/models/model_registry.json)
- update this file with plain-English release notes
- avoid committing mutable live CSV churn unless the release intentionally includes data migration
