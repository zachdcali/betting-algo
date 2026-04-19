# Versioning

This project now needs two kinds of versioning, not one.

## 1. Model Versions

Model versions should be tracked separately by family:

- Neural Network
- XGBoost
- Random Forest

That means:

- `model_registry.json` should keep independent current versions per family.
- The live prediction log should record all family versions that were present on a row, not just the primary NN version.
- A new model artifact should not silently replace an old one without a version bump.

Recommended bump rules:

- Patch bump: same dataset and feature set, bugfix to inference or packaging only
- Minor bump: retrain with a changed split, changed features, changed heuristic, changed calibration, or changed hyperparameters
- Major bump: materially different data source, feature philosophy, or modeling approach

Practical guidance for the next NN retrain:

- Because the historical NN holdout was used for early stopping, the next honest retrain should be treated as a new NN version, not a silent overwrite.
- `v1.3.0` is a reasonable next NN version if the feature set stays conceptually the same but the train/validation/test protocol changes.

Probability mode should be tracked separately from the base artifact version:

- `nn_model_version = v1.2.1`
  The trained NN weights/scaler family
- `nn_probability_source = raw`
  Direct sigmoid output from that model
- `nn_probability_source = calibrated`
  A post-hoc calibration layer applied to that same family

That separation matters because a calibrated probability layer should not silently masquerade as a brand-new artifact version.

## 2. Logging Schema Versions

Operational logging also needs its own versioning.

Current intent:

- `legacy_v1`
  Old rows that predate immutable lineage ids
- `prediction_log_v2`
  Rows created under the hardened logging path with `run_id`, `match_uid`, `feature_snapshot_id`, and immutable snapshot ids

These are different from model versions:

- A logging schema bump does not mean a new model.
- A new model does not automatically mean a new logging schema.

## 3. Lineage Quality

Not every row in history is equally trustworthy for retroactive rescoring.

Use these concepts consistently:

- `logging_quality = snapshot_v2`
  The row came from the new schema-backed logging path
- `logging_quality = legacy_backfilled`
  The row was upgraded after the fact and does not have original immutable feature lineage
- `rescore_quality = exact_feature_snapshot`
  The feature row is known exactly
- `rescore_quality = legacy_fallback_match`
  The system had to reconstruct the match by name/date heuristics

## 4. Pending / Stale Rows

Pending rows should also be categorized explicitly.

Recommended meanings:

- `pending`
  Current live row with a real model prediction
- `pending_legacy`
  Older model row without exact lineage ids
- `pending_no_model`
  Market-only or otherwise incomplete row that is still recent
- `stale_no_model`
  Old row with no model output that should not block normal production settlement
- `settled`
  Finalized row with winner and scoring columns filled

## 5. Promotion Rule

A model should only become the production `current_version` after:

- honest chronological evaluation
- calibration review
- basic live smoke testing
- explicit registry update

That keeps training experiments, archived artifacts, and live production behavior from drifting out of sync.
