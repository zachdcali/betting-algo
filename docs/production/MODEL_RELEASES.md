# Model Releases

This file is the human-readable release ledger for promoted production models.

Registry truth still lives in [model_registry.json](/Users/zachdodson/Documents/betting-algo/production/models/model_registry.json), but every promoted model family version should also be explained here in plain English.

## Current Production Versions

- Neural Network: `v1.2.1`
- XGBoost: `v1.2.0`
- Random Forest: `v1.2.0`

## Release Notes

### 2026-04-23

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
