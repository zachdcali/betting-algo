# Operational Database

This document describes operational Postgres contract `1.1.0`, built by the
ordered migrations
[`20260714010000_operational_schema_v1.sql`](../../supabase/migrations/20260714010000_operational_schema_v1.sql)
and
[`20260714020000_operational_integrity_v1_1.sql`](../../supabase/migrations/20260714020000_operational_integrity_v1_1.sql).
It is a side-by-side migration target, not authorization to change the
production source of truth.

The current production contract remains:

- CSV is the application-facing operational write format.
- `dash_*` tables plus `dash_sync_manifest` are the durable recovery and public
  dashboard bridge.
- `players`, `matches`, `match_stats`, `match_conflicts`, and `ingest_runs` in
  the existing Supabase project are the separate canonical tennis-history
  store used by live features.
- The normalized `raw`, `ops`, and `ml` schemas remain private and additive
  until import parity, dual-write parity, recovery drills, and the production
  soak all pass.

No command in this document should be pointed at the production database. A
separate Supabase staging project or disposable PostgreSQL database is a hard
prerequisite.

For dated counts and the exact separation between branch-local implementation
and the current live Supabase project, see
[`CUTOVER_STATUS_2026-07-14.md`](CUTOVER_STATUS_2026-07-14.md).

## Schema map

### `raw`: immutable collection evidence

- `source_fetches` records every source attempt, including blocked, partial,
  empty, and failed attempts. A failure is evidence; it must never delete an
  older valid observation.
- `source_artifacts` records content hashes and storage URIs for preserved raw
  source material and legacy CSV import artifacts.

Raw payload bodies belong in private object storage. PostgreSQL stores their
location, checksum, capture time, source fetch, and parser/context metadata.

### `ops`: operational facts and state transitions

- `schema_versions` records applied operational schema contracts.
- `import_batches`, `import_batch_memberships`, `import_conflicts`, and
  `import_conflict_resolutions` preserve deterministic manifests, many-batch
  fact membership, contradictory legacy candidates, and explicit human
  remediation without first-wins data loss.
- `pipeline_runs` and `pipeline_run_stages` record run lifecycle and stage
  evidence.
- `odds_observations` records typed point-in-time markets.
- `match_metadata_observations` records append-first event, start-time, round,
  surface, and level evidence with field-level provenance.
- `paper_accounts`, `paper_sessions`, and `account_ledger` record paper-account
  configuration, sessions, and financial events.
- `bet_recommendations` and `bet_state_events` separate the recommendation from
  its changing state.
- `settlement_attempts` retain every conservative match attempt, while
  `settlement_events` retain accepted result facts.
- `skip_events` make pipeline rejection reasons durable and queryable.

Terminal lifecycle rows cannot be changed after terminal state. Settlement and
bet corrections are append-only, form one-child acyclic chains, and cannot
silently regress terminal state. Immutable retries must carry the identical
semantic record SHA-256; a reused idempotency key with different content aborts
the transaction.

The `match_anchor_key` is the operational grouping key for metadata that may be
corrected later. The legacy `match_uid` is retained for compatibility and
lineage; consumers must not assume it is immune to a round, surface, event, or
date correction.

### `ml`: feature and inference evidence

- `feature_schemas` registers the ordered representation. The current contract
  is `base_141@1.0.0`, independently versioned from feature formulas.
- `feature_snapshots` records the exact per-match vector, schema hash, vector
  hash, build status, completeness, and feature-semantics identity.
- `prediction_observations` stores one row per model family/version rather than
  packing NN, XGBoost, and Random Forest into one mutable record.
- `model_registry_generations`, `model_releases`, and
  `model_release_status_events` bind inference to immutable artifact contracts
  and monotonic registry generations. Current authority is resolved only from
  the globally highest generation, releases omitted by it are no longer
  current, and a generation may promote at most one release per family. An
  older registry imported later cannot become current. A database constraint
  independently proves the known family, Semantic Versioning, feature width,
  schema/semantics bindings, artifact hashes, scaler metadata, and optional
  calibration metadata before `contract_complete=true` can be stored. The
  calibrated NN promotion status remains blocked until per-prediction
  calibration-version lineage exists in the authoritative CSV path.

Feature schema, feature semantics, model family version, dataset release, and
operational schema are separate version domains. Formula parity work will use
`base_141_shared@1.0.0` only after the chronological golden fixtures pass. Do
not use informal identifiers such as `base_141_v2` or `final_fixed_v2.pkl`.

New model artifacts follow the versioned directory convention documented in
[`VERSIONING.md`](VERSIONING.md):

```text
releases/<family>/v<semver>/model.<format>
candidates/<family>/v<semver>/model.<format>
archive/<family>/v<semver>/model.<format>
```

The model registry binds those paths to model, scaler, and calibrator hashes,
feature schema and semantics, training dataset, protocol, and code commit.

### `api`: reviewed projections only

Contract `1.1.0` creates private projections for current metadata, pipeline
runs, source freshness, predictions, bet states, settlements, and paper-account
state. It grants none of them to a browser role.

The public dashboard must continue using its accepted `dash_*` generation until
a separate reviewed migration:

1. defines a generation-pinned dashboard contract;
2. materializes metrics through `production/evaluation`, not browser math;
3. excludes raw evidence and potentially sensitive error fields;
4. grants only `SELECT` on named `api` views to a dedicated read role; and
5. proves each published generation's counts and watermarks.

Direct browser access to `raw`, `ops`, or `ml` is never part of the design.

## Idempotency and provenance

Every repository-managed row has a stable `idempotency_key`. Facts and runtime
lifecycle rows also carry a semantic `record_sha256`; the import-batch control
row uses its deterministic `manifest_sha256` as the corresponding stable
identity. Exact retries are accepted, while a reused key with different
normalized content raises and rolls back. Lifecycle records may advance only
while their stored status is non-terminal. Once terminal, an exact-hash retry
is a no-op and contradictory content fails closed through both repository
guards and database triggers.

Legacy facts preserve:

- `import_batch_id`;
- `source_file`;
- physical `source_row_number`;
- canonical `source_row_json`; and
- `source_row_sha256`.

The import batch UUID is deterministically derived from the operational schema
version, normalizer contract, complete source-file hash manifest, and exact
normalized target manifest. Rebuilding unchanged input and mapping code
therefore produces the same batch and row keys; changing normalization creates
a new batch even if source bytes are unchanged.

Facts retain their first accepted source provenance and can participate in
later manifests through `import_batch_memberships`. Each membership also
preserves that batch's source row and expected target hash. Parity queries the
exact planned global key set plus the batch membership set, so overlapping
additive imports do not rewrite immutable facts or lose prior provenance.

Imported foreign keys are `DEFERRABLE INITIALLY DEFERRED`. This lets one
transaction insert dependency-related batches in a deterministic order while
still requiring all referenced rows to exist before commit.

## Safe commands

Run all commands from the repository root.

### 1. Build a read-only plan

This is the default. It reads local files and prints the deterministic manifest,
counts, and hashes; it does not connect to PostgreSQL or rewrite CSVs.

```bash
tennis_env/bin/python -m production.storage.import_csv --prod-dir production
```

To plan only the aggregate feature-vector file and omit immutable per-run
feature files:

```bash
tennis_env/bin/python -m production.storage.import_csv \
  --prod-dir production \
  --skip-run-feature-files
```

Save and review the JSON output outside any credential-bearing path. Check the
recognized files and per-table counts before considering an apply.

### 2. Create the schema on staging

Set a URL for a disposable local or separate staging database. Never reuse the
production variable name or value.

```bash
export OPERATIONAL_STAGING_DATABASE_URL='postgresql://...staging-only...'
psql "$OPERATIONAL_STAGING_DATABASE_URL" \
  -v ON_ERROR_STOP=1 \
  -f supabase/migrations/20260714010000_operational_schema_v1.sql
psql "$OPERATIONAL_STAGING_DATABASE_URL" \
  -v ON_ERROR_STOP=1 \
  -f supabase/migrations/20260714020000_operational_integrity_v1_1.sql
```

The migrations use additive objects and version registration. The integration
test applies each newly required migration twice at its own version boundary
and is safe to rerun against a database already at `1.1.0`.

The `1.1.0` preflight deliberately refuses an already populated `1.0.0`
staging schema. SQL cannot reconstruct the required semantic hashes and
validation decisions losslessly from an unverified partial import. Drop and
recreate that **staging-only** database, apply both migrations in order, and
reimport from preserved source evidence. Never use that instruction to erase a
production database.

### 3. Run the contract tests

Static and pure-Python verification does not need a database:

```bash
tennis_env/bin/python -m pytest -q \
  tests/test_operational_schema.py \
  tests/test_operational_repository.py \
  tests/test_operational_import.py \
  tests/test_live_records.py
```

For the real migration/insertion test, use only a local or separate staging
database. The test refuses a remote host unless explicitly overridden; that
override is not appropriate for production.

```bash
export TEST_DATABASE_URL="$OPERATIONAL_STAGING_DATABASE_URL"
tennis_env/bin/python -m pytest -q \
  tests/test_operational_postgres_integration.py
```

### 4. Apply one import to staging

The environment-variable-name option keeps the connection string out of the
process argument list and shell history. No environment variable is read
implicitly.

```bash
tennis_env/bin/python -m production.storage.import_csv \
  --prod-dir production \
  --apply \
  --database-url-env OPERATIONAL_STAGING_DATABASE_URL
```

`apply_plan()` performs all writes on one caller-owned connection. It compares
the exact planned global keys and semantic hashes, then verifies every expected
batch membership before the connection commits. A missing key, unexpected
planned-key result, contradictory reused key, or hash mismatch raises, and the
connection rolls the whole transaction back. A successful apply advances the
import batch from `planned` to `verified` and records per-table counts.

The direct `--database-url` option exists for controlled local use, but the
environment-name form is preferred for CI and staging.

## Cutover phases

### Phase 0: isolate and back up

- Provision a separate Supabase staging project or disposable PostgreSQL
  database. A second schema inside production is not staging.
- Capture an encrypted logical backup of the current canonical and `dash_*`
  state and verify restore access.
- Preserve exact hashes of every CSV being imported.
- Confirm production PITR/backup retention before any future production DDL.
- Freeze new paper exposure while the existing pending ledger is reconciled.

### Phase 1: staging import

- Apply schema contracts `1.0.0` and `1.1.0` in order, replaying each new
  migration at its own boundary to prove idempotent DDL.
- Generate a fresh read-only plan, review counts, then apply that exact plan.
- Require exact key/hash parity and a terminal `verified` import batch.
- Query representative feature, prediction, odds, settlement, and account rows
  back to their source file and row.
- Restore staging from backup and repeat the import as a recovery drill.

CSV remains authoritative throughout this phase.

### Phase 2: dual-write shadow

- Wire the pure live-record builders into the orchestrator without changing
  betting decisions.
- Commit one run's database observations in a caller-owned transaction.
- Keep the existing CSV writes and compare run IDs, match anchors, feature
  hashes, prediction probabilities, odds, skips, settlements, and account
  values after every run.
- Treat either durable-write failure as a failed/degraded run, but do not
  hydrate production from the new tables yet.
- Exercise retry, overlapping runner, blocked-source, cancellation, stale
  runner, and transaction-kill scenarios.

This phase must not silently turn a filesystem/database split-brain into an
accepted run. The reconciliation report is the gate.

### Phase 3: database-backed dashboard shadow

- Build generation-pinned `api` projections alongside the current dashboard.
- Compare every visible count, freshness timestamp, metric cohort, bankroll,
  pending exposure, available capital, settlement, and skip reason.
- Confirm no model metric is recomputed in JavaScript.
- Run desktop/mobile browser checks and alert-freshness drills.

The public dashboard stays on `dash_*` until its database-backed shadow is
exact and stable.

### Phase 4: operational read cutover

Only after the gates below pass may the orchestrator read normalized Postgres
state for dedupe, pending exposure, settlement, and prior observations. CSV,
Parquet, and SQLite then become generated exports from a database watermark.

Keep dual writes during a rollback window. A feature flag must restore the old
read path without data deletion.

### Phase 5: database write authority

Postgres becomes operational source of truth only after the seven-day clean
cloud soak, recovery drill, security review, and sign-off on the schema gaps
below. At that point:

- database commits define accepted operational facts;
- exports are reproducible and carry a watermark and manifest hash;
- Git no longer carries hourly mutable runtime state; and
- the `dash_*` bridge is retired in a separate reversible migration.

Canonical tennis-history cleanup and shared train/serve feature parity remain
independent retraining gates. Moving rows to Postgres does not make historically
incorrect data correct.

## Private-by-default security

The migration:

- revokes schema and object privileges from `PUBLIC`;
- revokes default table privileges in all four new schemas;
- enables RLS on every raw, operational, and model fact table; and
- grants no browser access to `api` views.

There are intentionally no permissive RLS policies. Direct administrative
connections can still bypass RLS, so production must not give the routine
pipeline a database-owner credential. Before cutover, create separate roles:

- a migration owner used only by controlled migrations;
- an ingest writer with only the inserts and lifecycle updates it requires;
- a settlement/account writer with narrower transactional permissions;
- a dashboard reader with `SELECT` only on approved `api` projections; and
- a backup/monitoring role with audited read access.

Explicitly inspect grants for `anon`, `authenticated`, `service_role`, and every
custom role in staging. RLS is defense in depth, not a substitute for least
privilege or safe views.

Never expose raw URLs, error payloads, source evidence, feature vectors, source
row JSON, account events, or unrestricted metadata JSON through a browser view.

## Integrity work completed in `1.1.0`

The integrity migration closes the database-contract gaps found during the
first schema review:

1. Terminal run, fetch, and stage state is protected by PostgreSQL, and
   append-only bet/settlement correction chains cannot silently regress.
2. Bets link the exact feature, prediction, and odds observations used for the
   decision; decision-grade inserts verify calculation and timing consistency.
3. Predictions link immutable model releases and monotonic registry-generation
   status. Complete feature, schema, semantics, probability-pair, and promoted
   release contracts are checked before `decision_eligible=true` is accepted.
4. Feature snapshots labeled complete must match the registered ordered schema,
   contain finite numeric values, include no defaults, and satisfy every
   one-hot cardinality group.
5. Odds evidence has an explicit inference-grade state and must be complete,
   two-sided, finite, and pre-start before a decision-grade bet can use it.
6. Semantic record hashes, conflict quarantine, append-only batch memberships,
   exact-target parity, identical retry proof, and changed-manifest proof make
   legacy imports fail closed without first-wins guessing.

## Remaining authority-cutover gates

These are still open and prevent a production source-of-truth cutover:

1. **Account journal semantics are provisional.** The imported account ledger
   is evidence, but it is not yet a transaction-balanced reserve/release/win/
   loss/void/deposit/withdrawal journal. Define those posting rules and prove
   equity, pending exposure, and available capital against the existing paper
   ledger before enabling database-backed allocation.
2. **Conflicting legacy identities require review.** Quarantined prediction,
   feature, and settlement candidates remain unaccepted until explicit mappings
   or resolutions are reviewed. Database normalization does not make ambiguous
   source history correct.
3. **Runtime dual-write is not integrated into `main.py`.** The sink and pure
   live-record builders are opt-in, atomic, schema-gated components, but the
   orchestrator still writes the accepted CSV plus `dash_*` path. Wiring it
   requires a separate staging URL, run-by-run parity report, and failure-mode
   soak; enabling it directly against live Supabase would skip the safety gate.
4. **Dashboard generations and evaluation snapshots remain on the accepted
   bridge.** Contract `1.1.0` intentionally does not replace
   `dash_sync_manifest` or `dash_model_metrics`. A normalized API generation
   must match every visible count, watermark, cohort, and account value before
   browser reads change.
5. **Cloud operational controls remain unproved.** Least-privilege roles,
   backup/PITR, timed restore, blocked-source and transaction-kill drills,
   alerting, and the seven-day staging soak need a real separate Supabase
   staging project.
6. **Retraining gates are independent.** Canonical 2025/2026 duplicate and
   player-identity cleanup plus field-exact shared train/serve feature parity
   must pass before new data is used to train or promote models.

These are explicit stop gates, not reasons to discard the migration. The schema
provides the typed staging target and durable provenance needed to solve them
without changing today's live pipeline.

## Production approval checklist

Production deployment requires all of the following evidence:

- separate staging database used for migration, import, and destructive-failure
  simulations;
- current production logical backup plus a timed restore drill;
- migration applied twice and real Postgres integration test passed;
- read-only plan reviewed and exact import parity passed;
- changed-manifest second import passed without losing provenance;
- terminal-state, correction-chain, feature-validation, and evidence-link gaps
  closed;
- least-privilege roles and explicit Supabase grant audit passed;
- dual-write parity clean across every table and calculation;
- dashboard shadow exact and browser/security reviewed;
- blocked-source and settlement recovery drills passed;
- pending paper ledger reconciled and available capital non-negative; and
- seven consecutive days of terminal cloud runs with freshness alerts and no
  lineage, settlement, identity, or account regression.

Until then, the normalized database is a staging/shadow system. It must not be
used to justify retraining, model promotion, live exposure, deletion of source
CSVs, or retirement of the accepted `dash_*` recovery bridge.
