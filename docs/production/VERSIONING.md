# Versioning

This project uses separate version domains. A model version, feature contract,
dataset snapshot, logging schema, and operational database schema must never be
inferred from one another.

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

Candidate versions are useful when a model has been retrained honestly but is not yet the promoted live artifact:

- keep `current_version` pointed at the live promoted version
- track a `candidate_version` separately when a retrain exists but should not silently replace production
- use candidate status for cases like "honest retrain completed, but offline or live review is not strong enough to promote"

The registry document is an independently ordered release stream. Its root
must carry:

- `registry_schema_version`: shape of the registry document;
- `registry_generation`: a positive, monotonically increasing integer; and
- `registry_effective_at`: an explicit timezone-aware timestamp.

Every registry content hash is stored as an immutable database generation.
Model status events are unique per release and generation, and "current" is
selected by `registry_generation`, never by import time. Increment the
generation whenever a committed registry change alters release status or
contract metadata. Reusing a generation number with different bytes is an
integrity error.

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
- `calibration_version = 1.0.0`
  The independently versioned calibrator contract and pinned artifact

That separation matters because a calibrated probability layer should not silently masquerade as a brand-new artifact version.

Live calibrated promotion is currently fail-closed. It stays blocked until
the authoritative CSV prediction log, immutable snapshots, normalized
importer, replay tooling, and evaluation ledger all persist and group by an
`nn_calibration_version` on every calibrated observation. A candidate
calibrator can be validated offline before that lineage upgrade, but it cannot
become a promoted status event. Each promoted calibration must also use a new
NN model-family version, even when the base weights are identical, while
retaining its separate calibration version.

## 2. Logging Schema Versions

Operational logging also needs its own versioning.

Current intent:

- `legacy_v1`
  Old rows that predate immutable lineage ids
- `prediction_log_v2`
  Rows created under the hardened logging path with `run_id`, `match_uid`, `feature_snapshot_id`, and immutable snapshot ids

New live `prediction_log_v2` rows also carry an additive player-identity
sub-contract:

- `player_identity_schema_version = live_player_name@1.0.0`
- oriented `p1_identity_key` and `p2_identity_key` values copied from the exact
  odds-ingestion inputs used to create `match_uid` and `feature_snapshot_id`
- raw `p1` and `p2` remain display/audit text and are not substituted for those
  keys during lineage validation or market joins

Older v2 rows may have blank identity-key columns. They are not rewritten. A
later exact-UID refresh may backfill the keys only after the supplied display
names, keys, UID, feature snapshot, and match metadata all pass the fail-closed
identity checks.

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

## 6. Feature Schema And Semantics

Do not use ambiguous names such as `base_141_v2`, `model_final_v2.pkl`, or
`latest_fixed.pth` as identifiers. Version identity lives in metadata and uses
Semantic Versioning.

Two feature versions are tracked independently:

- `feature_schema_id`
  Ordered names, types, and cardinality. The current representation is
  `base_141@1.0.0`, with 141 ordered fields and schema SHA-256
  `17a33325776292ad31e4bbaff81cb223355f026aa019b8b516a0281945930b4d`.
- `feature_semantics_id`
  Formula, as-of, source-priority, missingness, and orientation behavior.

The current historical and live implementations deliberately have different
semantic identities because parity is not proven:

- `sackmann_historical_legacy@1.0.0`
- `ta_live_legacy@3.0.0`

The future shared, parity-tested implementation is reserved as
`base_141_shared@1.0.0`. Its opt-in implementation and synthetic chronological
contract are documented in
[SHARED_FEATURE_SEMANTICS.md](../modeling/SHARED_FEATURE_SEMANTICS.md). Passing
that fixture is necessary but does not activate the candidate: source
provenance, immutable dataset/model releases, chronological evaluation, and an
explicit registry promotion are still required. If the shared implementation
keeps the same ordered fields, the schema can remain `base_141@1.0.0`; changing
formulas is represented by the semantics ID and normally warrants a major
model-family version because behavior changes materially.

Every new model-registry entry must record:

- `feature_schema_id` and `feature_schema_sha256`
- `feature_semantics_id`
- `training_dataset_id` and manifest SHA-256
- training protocol/version and code commit
- model, scaler, and calibrator hashes where applicable

## 7. Artifact Paths

Existing promoted filenames remain untouched for compatibility. New artifacts
use a stable directory hierarchy:

```text
releases/<family>/v<model-semver>/model.<format>
candidates/<family>/v<model-semver>/model.<format>
archive/<family>/v<model-semver>/model.<format>
```

For example:

```text
candidates/xgboost/v2.0.0/model.json
candidates/nn/v2.0.0/model.pth
candidates/nn/v2.0.0/scaler.pkl
```

The model registry—not a long filename—binds an artifact to its family,
version, feature contract, dataset, calibration, and hashes.

The extension remains the real serialization format (`.pth`, `.pkl`, `.json`,
or another reviewed format). Do not encode release identity in suffixes such as
`_v2`, `_final`, or `_fixed`; the family directory and SemVer registry entry are
the release identity.

## 8. Dataset Releases

Training reads an immutable Parquet release exported from a specific canonical
database watermark. Each release has a JSON manifest with its own SemVer,
content hashes, query hash, canonical snapshot/watermark, feature schema and
semantics IDs, cleanup-map version, row counts, chronological cutoffs, code
commit, and deterministic orientation rule.

Never overwrite a dataset release directory. A corrected canonical snapshot or
changed feature semantics creates a new release even when the row count happens
to be unchanged.

## 9. Operational Database Schema

Typed Postgres migrations have an independent version. The current normalized
operational contract is `1.1.0`; migration filenames use an ordered UTC
timestamp plus a descriptive slug. Contract `1.0.0` is the base schema and
`1.1.0` adds the integrity layer. This does not bump any model.

The legacy-to-Postgres normalizer is versioned separately as
`operational_csv_normalizer@1.0.0`. An import batch identity includes the
source-file hashes, operational schema version, normalizer version, and exact
normalized target manifest. A mapping change therefore creates a new batch
even when the source bytes are unchanged.

Logging remains `prediction_log_v2` during dual-write. A logging-schema bump
occurs only when the persisted record contract changes, not merely because the
same evidence is copied into Postgres.
