# Production Readiness Audit — 2026-07-13

## Executive decision

**Verdict: NO-GO.** The project is **not ready to retrain or promote models on
the 2025/2026 extension, and it should not add new paper exposure**, until the
stop gates in this report are closed with recorded evidence.

This is not a judgment that the project is unsalvageable. The system has a
useful research foundation, substantial immutable lineage, and a much stronger
local implementation after this audit. The no-go is about the difference
between a preventive code control and corrected production truth:

- The audit branch adds durable-state recovery, transactional dashboard
  generations, fail-closed lineage, safer time/odds/settlement handling,
  exposure-aware staking, artifact checks, and a redesigned operations UI.
- Those controls do **not** repair the existing canonical match and player rows,
  reconcile the current pending paper-bet backlog, prove train/serve feature
  parity, recover parked runner history, or demonstrate a successful cloud
  soak.

Accordingly:

| Decision | Status | Release condition |
| --- | --- | --- |
| Deploy preventive pipeline/dashboard changes to a controlled staging environment | **GO** | Full automated suite, staging migration, recovery drill, and browser QA pass |
| Resume unattended paper recommendations | **NO-GO** | Pending exposure reconciled, available capital positive, durable-state soak passes, and source failures degrade safely |
| Build exploratory feature code or read-only analyses | **GO WITH GUARDRAILS** | Work stays side-only and does not consume the contaminated extension as training truth |
| Retrain on the 2025/2026 extension | **NO-GO** | Canonical cleanup, explicit identity map, date provenance audit, and field-exact train/serve parity all pass |
| Promote or use a newly trained artifact | **NO-GO** | Retraining gates plus registry validation, calibration review, unbiased ledger evaluation, and promotion procedure pass |

## Scope and evidence boundary

The audit covered the hourly workflow, orchestration, odds capture, ATP/ITF/TA
source behavior, round/surface/start-time gates, canonical Supabase store,
operational CSV lineage, settlement, staking/accounting, model artifact loading,
evaluation ledger, local Streamlit dashboard, and public static dashboard.

Evidence came from repository and workflow inspection, automated/local-log
analysis, the public dashboard, GitHub run/branch state, and **read-only** queries
against the canonical and dashboard databases. **No live database writes, live
cleanup, production pipeline run, real bet, or model promotion were performed
during this audit.** All cleanup SQL below is a reviewed execution plan, not a
record of work already applied.

The audit snapshot is internally consistent as of 2026-07-13 Pacific time
(some cloud identifiers are dated 2026-07-14 UTC). Mutable counts will naturally
advance; the anomalies and stop gates must be re-measured before sign-off.

## Measured state

### Canonical store and data-quality findings

| Measure | Observed |
| --- | ---: |
| Canonical `players` rows | 68,750 |
| Canonical `matches` rows | 972,670 |
| Exact duplicate-name groups | 690 groups / 1,556 player rows |
| Sequence-issued player IDs requiring identity review | 2,078 |
| Shifted ATP result duplicates | 595 pairs: 357 shifted +7 days, 238 shifted +14 days |
| Same-result pairs with conflicting round labels | 213 |
| Pending-slate rows touched by shifted duplicates | 56 of 67 |

The 690 name groups are **candidates**, not 690 automatic merges. Legitimate
homonyms must remain separate. Likewise, sequence-issued identities are a review
population, not proof that all 2,078 are wrong. The critical defect is that the
old ITF creation path could interpret an ambiguous lookup as a missing person
and create another player on a later run.

The 595 shifted pairs are a specific, reproducible failure mode: the same ATP
result was stored again on a later week's inferred event date. They contaminate
recency, activity, streak, surface-form, rank movement, and days-since features.
The 213 round-label pairs are separate and cannot be bulk-deleted safely without
checking official draw/schedule provenance.

### Operational lineage and paper account

| Measure | Observed |
| --- | ---: |
| Operational prediction rows | 2,602 |
| Valid settled predictions (`actual_winner` 1 or 2) | 2,057 |
| Void/cancellation outcome (`actual_winner = -1`) | 1 |
| Rows labeled feature-complete | 1,302 |
| Immutable prediction snapshots | 1,409 |
| Immutable odds observations | 1,409 |
| Local shadow-model observations | 13,010 |
| Local audited pipeline/settlement runs | 71 |
| Paper bets | 153 |
| Settled paper bets | 60 |
| Pending paper bets | 93 |
| Pending stake | $1,878.03 |
| Realized net P&L | -$427.08 |
| Account equity from a $1,000 starting balance | $572.92 |
| Available capital after pending exposure | $0.00 |

Every pending paper bet is dated before 2026-07-14. The pending stake is more
than three times current equity. This is not just a dashboard presentation bug:
the previous allocator could normalize capped Kelly weights to spend the full
18% block and apply that budget repeatedly across date groups, while a new
session reset its bankroll view and did not reserve earlier pending exposure.
Until the 93 pending rows are settled, voided, or explicitly reconciled, the
correct new-exposure capacity is zero.

The backlog is not one homogeneous queue: 76 pending rows / $1,564.79 predate
July 11; 51 rows fall into 22 duplicate match/side/date identities; 26 have a
valid settled prediction outcome available for review; and 63 pending
`match_uid` values are absent from the current prediction log. Do not run a
blind bulk tracker sync. First decide which duplicate recommendations represent
one paper decision versus distinct intended exposure, then reconcile exact
outcomes and investigate orphan IDs.

`features_complete=True` is only a historical label for the 1,302 rows; it is
not retroactive proof of clean upstream history or train/serve equivalence.
Decision-grade GOLD evaluation now additionally requires a verifiable persisted
feature snapshot.

### Live operations observation

- The public mirror's latest visible attempt, `run_20260714T003215Z`, remained
  `running` with zero stage counts while the separately accepted prediction
  state contained 71 rows. A dashboard that calls those the same thing is
  misleading, even if both records are technically present.
- There were 72 parked `runner-logs-*` branches from unresolved git push races.
  A parked branch preserved recoverability but previously allowed the workflow
  to look green without advancing `main`.
- Cloud runner access remains structurally unreliable for Tennis Abstract and
  ITF because those sources block or degrade datacenter traffic. A residential
  collector has worked where the cloud runner has not.

## Severity-ranked findings

Status vocabulary in this table matters:

- **Implemented locally** means a preventive control exists in the audit branch.
- **Remediation outstanding** means production data or operations are still
  wrong even if the recurrence path has been blocked.
- **Unproven** means the implementation still needs deployment/recovery/soak
  evidence.

| Severity | Area | Finding and consequence | Current status |
| --- | --- | --- | --- |
| P0 | Training data | 595 shifted-result pairs and 213 round-label conflicts pollute temporal features; 56/67 pending-slate rows were touched | Runtime quarantine implemented locally; canonical remediation outstanding |
| P0 | Feature semantics | Historical preprocessing and live serving differ in H2H orientation, `Sets_14d`, form trend, rank movement/volatility, first-surface behavior, and recent-H2H smoothing | Retraining stop gate; shared implementation and golden parity outstanding |
| P0 | Capital safety | 93 pending bets reserve $1,878.03 against $572.92 equity; previous sizing could overspend its intended budget | Run-wide/pending-exposure gate implemented locally; ledger reconciliation outstanding |
| P0 | Player identity | Ambiguous names could be treated as missing and create duplicate ITF identities; 690 duplicate-name groups and 2,078 sequence IDs require review | Ambiguous auto-creation blocked locally; explicit production merge/keep map outstanding |
| P0 | Durable operations | Git push races and stale CSV bases could regress settlement/slate state while a cloud run appeared successful | Hydrate + transactional manifest publication + false-green failure implemented locally; staging recovery drill and live soak outstanding |
| P1 | Source reliability | TA/ITF cloud blocking and empty event discovery can collapse round resolution or settlement coverage | Circuit breakers/cache protections improved; residential collection architecture outstanding |
| P1 | Time and market truth | Book display time, UTC conversion, missing start times, malformed odds, and fabricated market `0.5` could permit wrong inference/evaluation | Eastern-time conversion, UTC lineage, missing-start gate, and two-sided odds validation implemented locally |
| P1 | Lineage/evaluation | Missing/corrupt feature files, void coercion, repeated hourly shadow rows, and stale correctness columns could inflate or change metrics | Content fingerprints, fail-closed GOLD verification, valid-winner filter, authoritative joins, and one-opening shadow scoring implemented locally |
| P1 | Run observability | A stale `running` attempt with zero counts was visually conflated with an accepted slate | Latest-attempt/accepted-state split and stage diagnostics implemented in the redesigned static dashboard; live deployment outstanding |
| P1 | Artifact integrity | A cache/release mismatch could load artifacts that did not match registry intent | Registry SHA-256, deserialization, and schema checks implemented locally; workflow validation required on deploy |
| P2 | Git operations | 72 parked runner branches make recovery and incident review difficult | Do not delete until every branch is proven subsumed by an accepted durable generation |
| P2 | Derived state | `production/logs/betting.db` created binary churn despite being rebuildable from CSV | Excluded from cloud commits; continue treating it as a derived read model |

## Model and evaluation assessment

The regenerated authoritative ledger strengthens the no-go rather than relaxing
it. On the current `gold_intersection` cohort (`n=624`), the market is the best
probabilistic forecaster:

| Model | Log loss | Brier | Calibration slope | Flat ROI | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| De-vigged market | 0.6188 | 0.2143 | 1.0372 | n/a | Best current probability baseline |
| XGBoost | 0.6317 | 0.2202 | 1.0432 | +1.6% | Closest promoted model; calibration is materially better than NN |
| Random forest | 0.6442 | 0.2261 | 1.1834 | -3.1% | Weaker quality and realized counterfactual return |
| Neural net | 0.6903 | 0.2348 | 0.4648 | -3.7% | Severely over-confident; current paper staking is NN-driven |

These results do not authorize switching the live betting family during this
audit. They do establish the right order of work:

1. Fix orientation and train/serve parity before diagnosing the NN from a
   training replay; the existing misaligned replay cannot distinguish model
   calibration from serving-distribution error.
2. Once parity is proven, fit calibration on validation-era predictions only,
   version the calibrator separately, and evaluate it on untouched chronological
   test and forward GOLD data.
3. Rank candidates by log loss, Brier, ECE, and calibration slope before
   accuracy or ROI. Kelly sizing amplifies probability error, especially at the
   extremes.
4. Treat shadow-model results (`n=104` in the current GOLD report) as research
   leads, not promotion proof. Repeated hourly observations no longer inflate
   `n`, but the independent-match sample remains small.
5. Predeclare promotion criteria and comparison cohorts before training. Do not
   choose a model, threshold, or calibration method on the 2023+ test era.

The ledger itself must be regenerated again after canonical cleanup and feature
rebuild. Current numbers are an honest description of logged evidence, not a
certificate that the upstream feature inputs were correct.

## Current and target architecture

### Current transitional path after the audit changes

```text
GitHub schedule (:17 / :47; one active run)
  -> restore + validate versioned model artifacts
  -> hydrate additive operational CSV state from latest accepted Supabase manifest
  -> checkpoint conservative settlement
  -> scrape Bovada odds and source metadata/results
  -> resolve canonical identities + event/round/surface/start time
  -> build and fingerprint exact feature snapshot
  -> run promoted models + side-model shadows
  -> qualify offered-price edge + allocate against available capital
  -> append operational and immutable lineage CSVs
  -> publish every dash_* table in one locked Postgres transaction
  -> commit CSV/audit evidence to git; a failed push race is a failed workflow
```

Supabase currently serves two different roles that must stay conceptually
separate:

1. `players`, `matches`, stats, conflicts, and ingest records are the canonical
   tennis store used by live features.
2. `dash_*` tables plus `dash_sync_manifest` form an additive durable recovery
   bridge and a generation-pinned read projection for the public dashboard.

CSV remains the application's immediate write format during this transition.
The local Streamlit dashboard reads those production CSVs. The public static
dashboard reads exactly one accepted manifest generation. Git is now secondary
recovery/provenance, not permission to claim that a failed durable publication
was successful.

### Recommended industry-grade end state

Move operational truth to normalized, append-first Postgres tables with stable
idempotency keys:

- `pipeline_runs` and `pipeline_run_stages`
- `source_fetches` and source artifacts/checksums
- `odds_observations`
- `feature_snapshots`
- `prediction_observations`
- `bet_recommendations` and `bet_state_events`
- `settlement_events`
- canonical `players`, `player_aliases`, `matches`, and provenance/conflict rows

Build `dash_*` as views or transactionally materialized projections from those
tables. Generate CSV exports and SQLite from the database, not the reverse.
This removes merge policy from git, gives every mutation an idempotency key and
provenance record, and makes partial-run rollback a database transaction rather
than a file-reconciliation exercise.

The blocked-source problem should be isolated from the compute pipeline. Run a
small authenticated residential collector for TA/ITF, write signed/idempotent
source observations to Postgres, and let the cloud orchestrator consume those
observations. An empty scrape should be an explicit failed source observation,
never evidence that a previously known event, round, or result disappeared.

Operationally, add explicit service-level objectives and alerts rather than
relying on someone opening the dashboard:

- accepted prediction generation age and scheduled-run success rate;
- source fetch age/coverage by ATP, ITF, TA, and Bovada;
- eligible/blocked/expired slate transitions and abrupt coverage changes;
- oldest pending settlement, settlement success rate, and ambiguous-match count;
- feature-build success, exact-snapshot coverage, and schema/hash mismatch;
- account equity, pending exposure, available capital, and any attempted stake
  rejection;
- canonical ingest lag, quarantine growth, identity ambiguity, and conflict
  backlog.

Keep the dashboard publishable credential restricted by read-only RLS/views;
only the runner/maintenance role should mutate operational tables. Add migration
versioning, tested restore/PITR, a periodic encrypted logical export, dependency
lock/vulnerability review, and retention policies for raw source artifacts,
lineage, and parked recovery branches. These controls are inexpensive compared
with debugging silent historical corruption later.

## Preventive controls implemented in the audit branch

These controls reduce recurrence risk. They are **not** substitutes for the
remaining cleanup and acceptance gates.

### Persistence, workflow, and recovery

- Hydration plans and validates all durable files before replacing any local
  state, with rollback on an incomplete generation.
- State reconciliation is field-aware: the best complete inference row,
  terminal settlement fields, and freshest `latest_*` observation fields are
  merged rather than trusting one whole stale row.
- Dashboard publication plans all tables first, acquires an advisory lock, and
  replaces them in one transaction with a manifest and exact row counts.
- Settlement and other durable progress are checkpointed before later scraping
  or inference stages can fail.
- Required durable publication failure is fatal in cloud mode.
- A parked log branch is recoverable but no longer reports a green workflow.
- The workflow validates model artifacts against the registry and stages
  durable handedness state while excluding derived SQLite churn.

### Scraping, timing, and bettability

- Bovada output must have valid page/header coverage and two finite decimal
  prices greater than 1.0; malformed/degraded pages cannot invent even money.
- Bovada display times are interpreted in `America/New_York`, then stored as
  immutable UTC start timestamps.
- A missing or unparseable start time blocks inference instead of silently
  passing the pre-start safety check.
- Missing tournament/round/surface and post-start/completed evidence remain
  explicit skip reasons.
- Prefetch failures, ITF/TA source failures, canonical ingest, reconciliation,
  and auto-settlement are surfaced as distinct run stages. A partial run remains
  partial.
- Future or inferred wrong-week history and the observed +7/+14-day exact-result
  pattern are quarantined from feature reads.
- Ambiguous existing player identities block ITF auto-creation and are logged
  for review.

### Logging, evaluation, and artifacts

- Skipped/error feature builds persist their status and cannot become complete
  just because an empty default list was emitted.
- Feature snapshots carry schema/vector SHA-256 fingerprints; promoted-model
  inference and GOLD verification require 141 finite values plus valid one-hot
  binary/cardinality groups, and fail closed on unreadable or mismatched data.
- Only winner values 1 or 2 are settled ground truth; void/cancelled outcomes
  are excluded or refunded.
- Evaluation joins model probabilities to the authoritative winner map rather
  than trusting stale `*_correct` columns.
- Shadow evaluation uses one deterministic opening observation per
  `(match_uid, model_version)` and the matching operational opening snapshot;
  hourly repeats remain lineage but do not increase sample size.
- Edge qualification compares a model probability with the offered price's raw
  break-even probability. The de-vigged market remains a fair-market baseline,
  not the price required to make money.
- Promoted artifacts are hashed and validated for deserialization and expected
  schema before inference.

### Paper account and sizing

- Account equity is starting capital plus realized settled P&L; recording a new
  pending bet reserves exposure but does not create or destroy equity.
- Available capital is equity minus all pending stake across prior sessions.
- Kelly sizing is capped per bet, across the whole run, and by available
  capital; separate date groups cannot each spend a fresh run budget.
- Zero available capital records a blocked-exposure run state and produces no
  new paper bets.

## Dashboard assessment and redesign

The prior dashboard mixed three different truths: the latest attempted run, the
last usable prediction slate, and the model-evaluation cohort. Counts could be
correct in isolation and still read like data loss. It also recomputed some
analysis in the browser, allowed a stale tab to run stale code, and made
incomplete rows look like broken eligible bets.

The redesigned static dashboard is organized around explicit questions:

1. **Overview — is the pipeline working?** Latest attempt status, stage funnel,
   source freshness, block reasons, account equity/exposure, and accepted state.
2. **Current slate — what is actionable?** Eligible, blocked/incomplete, and
   started/expired rows are separate; each blocked row states why.
3. **Performance — how good are the models?** Only server-generated ledger
   metrics are displayed, with named cohort definition and sample size.
4. **Bets/results — what actually happened?** Pending, settled, void, cancelled,
   realized P&L, and exposure remain distinct.
5. **System — what did each stage do?** Latest attempts, post-run stage results,
   errors, accepted manifest, and data inventory remain auditable.

The browser reads a single manifest-pinned generation, requires a referentially
valid feature snapshot before labeling a row eligible, and checks a version
sentinel so a long-open tab reloads new application code. The local Streamlit
surface remains the deeper CSV-backed forensic tool.

The dashboard should still be treated as a projection, not an independent
source of truth. Its production acceptance requires desktop/mobile browser QA
against real data and a deliberate deployment after the database migration.

## Mandatory stop gates

### A. Canonical match cleanup

- Back up and checksum the affected `matches`, `match_stats`, and conflict rows.
- Create a reviewed keep/quarantine map for all 595 shifted pairs.
- Review all 213 round-label conflicts against an explicit official result,
  draw, or schedule source; an inferred Monday is not sufficient provenance.
- Apply the map transactionally, rerun the candidate queries, and require zero
  unreviewed shifted/round-conflict rows in the 2025/2026 training extension.
- Rebuild every affected temporal feature and dataset manifest from the clean
  canonical snapshot.

### B. Player identity cleanup

- Classify every exact-name group as `merge` or `keep_separate`; name equality
  alone is not merge authority.
- Review all 2,078 sequence-issued IDs against source IDs, birthdate, country,
  hand, event history, and aliases.
- Apply an explicit duplicate-to-canonical mapping, resolve would-be unique
  match collisions first, then update all foreign keys and aliases in one
  transaction.
- Reset/verify the sequence and require identity-resolution tests to prove that
  an ambiguous lookup never creates a new person.

### C. Field-exact train/serve parity

- Move the disputed formulas into shared pure code used by both historical
  preprocessing and live serving.
- Create chronological golden fixtures with a hard as-of cutoff for H2H
  orientation, `Sets_14d`, activity windows, form trend, rank movement,
  volatility, first-surface behavior, and recent-H2H smoothing.
- Require exact field/schema parity across all 141 promoted features; numeric
  tolerance must be documented only where floating-point order warrants it.
- Prove the training reproduction harness has correct player orientation and
  reproduces the promoted artifact's expected evaluation before investigating
  or calibrating NN extremes.
- Complete an independent temporal-leakage review. No test-era row may affect
  early stopping, selection, thresholding, or calibration.

### D. Operational correctness and capital

- Reconcile the 93 pending bets from source results. Preserve void/cancelled as
  explicit states; do not force them into player-two losses.
- Require pending exposure <= equity and available capital > 0 before a new
  recommendation can reserve stake.
- Recover and compare every one of the 72 parked branches with accepted durable
  state before deleting any branch.
- Prove atomic publication, stale-run hydration, settlement monotonicity, and
  kill/restart recovery in staging.
- Complete at least seven consecutive days of scheduled-run soak with no false
  green, no accepted-count regression, no partial generation exposed, no
  duplicate pending bet, and no negative available capital.

### E. Source and dashboard acceptance

- An empty/blocked source fetch must retain earlier durable event metadata and
  produce an alert, not zero a good slate.
- TA/ITF collection must have a reliable execution surface, preferably the
  residential collector split described above, with cache age and provenance.
- Dashboard latest-attempt, accepted-state, source-freshness, and capital values
  must reconcile exactly to their backing generation.
- Automated contract tests plus browser checks at desktop and mobile widths
  must pass without console/network errors.

## Controlled database remediation plan

The following SQL is a **template for a reviewed maintenance window**. It was
not executed by this audit. Run it first on a fresh database clone. Take a
logical backup/PITR checkpoint and record pre/post counts and checksums. The
application should be read-only during the final transaction.

### 1. Recreate the shifted-result candidate set

```sql
-- Read-only discovery used by this audit.
WITH suspect_pairs AS (
  SELECT DISTINCT ON (late.match_id)
         early.match_id AS keep_match_id,
         late.match_id AS quarantine_match_id,
         early.match_date AS keep_date,
         late.match_date AS quarantine_date,
         late.event,
         late.round,
         late.score
  FROM matches early
  JOIN matches late
    ON late.match_id <> early.match_id
   AND late.winner_id = early.winner_id
   AND late.loser_id = early.loser_id
   AND late.round IS NOT DISTINCT FROM early.round
   AND late.score = early.score
   AND late.match_date > early.match_date
   AND late.match_date - early.match_date BETWEEN 1 AND 14
  WHERE early.source = 'atp_results'
    AND late.source = 'atp_results'
    AND early.score IS NOT NULL
    AND early.score <> ''
  ORDER BY late.match_id, early.match_date ASC
)
SELECT *, quarantine_date - keep_date AS shift_days
FROM suspect_pairs
ORDER BY quarantine_date, event, round;
```

Materialize that result in a dated audit schema with `approved`, `reviewer`,
`evidence_url`, and `reviewed_at` fields. The expected audit result is 595
reviewed pairs split 357/238; any difference pauses the procedure.

### 2. Recreate round-label conflicts

```sql
SELECT
  array_agg(m.match_id ORDER BY m.match_id) AS match_ids,
  m.match_date,
  lower(trim(split_part(m.event, ',', 1))) AS event_key,
  m.winner_id,
  m.loser_id,
  m.score,
  array_agg(DISTINCT m.round ORDER BY m.round) AS rounds,
  array_agg(DISTINCT m.source ORDER BY m.source) AS sources
FROM matches m
WHERE m.source LIKE 'atp%'
GROUP BY m.match_date, event_key, m.winner_id, m.loser_id, m.score
HAVING count(*) > 1 AND count(DISTINCT m.round) > 1;
```

Do not auto-delete these 213 pairs. Attach official round/date evidence and a
review decision to each one. If one row has richer `match_stats`, conflicts, or
better provenance, merge those child facts before quarantining the duplicate.

### 3. Create an auditable soft quarantine and backups

```sql
BEGIN;

CREATE SCHEMA IF NOT EXISTS audit_20260713;

CREATE TABLE audit_20260713.match_quarantine (
  match_id           bigint PRIMARY KEY REFERENCES matches(match_id),
  canonical_match_id bigint REFERENCES matches(match_id),
  reason             text NOT NULL,
  evidence           text NOT NULL,
  reviewed_by        text NOT NULL,
  reviewed_at        timestamptz NOT NULL,
  CHECK (match_id IS DISTINCT FROM canonical_match_id)
);

CREATE TABLE audit_20260713.matches_backup AS
SELECT m.* FROM matches m WHERE false;
CREATE TABLE audit_20260713.match_stats_backup AS
SELECT s.* FROM match_stats s WHERE false;
CREATE TABLE audit_20260713.match_conflicts_backup AS
SELECT c.* FROM match_conflicts c WHERE false;

-- Insert only reviewer-approved rows into backup tables and match_quarantine.
-- Assert the inserted row counts before COMMIT.

ROLLBACK; -- change to COMMIT only after clone validation and human approval
```

Production feature/history queries should exclude an approved quarantine with
`NOT EXISTS (SELECT 1 FROM audit_20260713.match_quarantine q WHERE
q.match_id = m.match_id)`. Keep the soft quarantine through at least one clean
dataset rebuild and parity pass. Physical deletion can happen later under a
separate retention policy.

### 4. Build an explicit player identity map

```sql
-- Discovery only: these are candidates, not automatic merges.
SELECT lower(name) AS exact_name, count(*) AS row_count,
       array_agg(player_id ORDER BY player_id) AS player_ids
FROM players
GROUP BY lower(name)
HAVING count(*) > 1
ORDER BY row_count DESC, exact_name;

SELECT count(*) AS sequence_issued_players
FROM players
WHERE player_id >= 9000000;

CREATE TABLE audit_20260713.player_identity_review (
  duplicate_player_id bigint PRIMARY KEY REFERENCES players(player_id),
  canonical_player_id bigint REFERENCES players(player_id),
  decision            text NOT NULL CHECK (decision IN ('merge', 'keep_separate')),
  evidence            text NOT NULL,
  reviewed_by         text NOT NULL,
  reviewed_at         timestamptz NOT NULL,
  CHECK (duplicate_player_id IS DISTINCT FROM canonical_player_id),
  CHECK ((decision = 'merge') = (canonical_player_id IS NOT NULL))
);
```

Before applying any `merge`, preview unique-key collisions after rewriting IDs:

```sql
WITH merge_map AS (
  SELECT duplicate_player_id, canonical_player_id
  FROM audit_20260713.player_identity_review
  WHERE decision = 'merge'
), rewritten AS (
  SELECT m.match_id, m.match_date, m.event, m.round,
         coalesce(w.canonical_player_id, m.winner_id) AS winner_id,
         coalesce(l.canonical_player_id, m.loser_id) AS loser_id
  FROM matches m
  LEFT JOIN merge_map w ON w.duplicate_player_id = m.winner_id
  LEFT JOIN merge_map l ON l.duplicate_player_id = m.loser_id
)
SELECT match_date, event, round, winner_id, loser_id,
       array_agg(match_id ORDER BY match_id) AS colliding_match_ids
FROM rewritten
GROUP BY match_date, event, round, winner_id, loser_id
HAVING count(*) > 1;
```

Every collision needs a reviewed canonical match decision first. After backing
up all affected players, aliases, matches, stats, and conflicts, the merge
transaction may update `matches.winner_id`, `matches.loser_id`, and
`player_aliases.player_id`, then delete only reviewed duplicate player rows.
Finally verify and advance the sequence without reusing an existing ID:

```sql
SELECT setval(
  'players_new_id_seq',
  (SELECT greatest(9000000::bigint, coalesce(max(player_id), 8999999) + 1)
   FROM players),
  false
);
```

### 5. Required post-transaction assertions

- All foreign keys validate; there are no orphan `match_stats`, aliases, or
  conflicts.
- The approved shifted candidate query returns zero unquarantined rows.
- Every one of the 213 round-label cases has a recorded decision.
- Every one of the 690 exact-name groups has a `merge` or `keep_separate`
  decision, and all 2,078 sequence-issued IDs have been reviewed.
- The latest canonical match coverage by tour/date/source meets a predeclared
  completeness contract.
- A clean 2025/2026 feature rebuild produces a versioned dataset manifest with
  source-table snapshot IDs, row counts, schema hash, and content hash.

## Verification and acceptance plan

### Local and CI verification

Run from the repository root:

```bash
tennis_env/bin/python -m pytest -q tests
(cd production && ../tennis_env/bin/python -m pytest -q)
node --test tests/dashboard_logic.test.cjs tests/dashboard_contract.test.cjs
node --check docs/dashboard.js
tennis_env/bin/python production/models/validate_registry.py
git diff --check
```

Regenerate the model ledger from the shared evaluation package and review every
cohort label/count change rather than accepting a changed Markdown file blindly:

```bash
cd production
../tennis_env/bin/python -m evaluation.ledger \
  --prod-dir . \
  --experiments-root ../results/professional_tennis/experiments \
  --out-dir ../results/professional_tennis/ledger/2026-07-13 \
  --report ../docs/modeling/MODEL_LEDGER.md
```

Before and after any smoke test, hash mutable operational CSVs. `main.py
--dry-run` must not create a betting session or mutate `logs/all_bets.csv`.

### Staging recovery drill

On a disposable Supabase clone:

1. Publish accepted generation A and record every table count/hash.
2. Start stale runner B, advance a settlement in runner C, then publish B.
3. Confirm terminal settlement and freshest `latest_*` fields survive.
4. Kill a publish between table loads; confirm no partial generation is visible.
5. Corrupt/remove one hydrated file; confirm the entire hydrate is rejected and
   local state rolls back.
6. Force a git push race; confirm the workflow fails after preserving the
   recoverable branch.
7. Cancel a running job; confirm its run is terminal/cancelled rather than
   permanently `running`.

### Source-degradation drill

- Replay valid Bovada, bot-wall, empty, one-sided, and malformed-price fixtures.
- Return an empty ATP calendar after a previously resolved event; the durable
  round/surface must remain and the fetch must be marked failed/stale.
- Exercise TA 403/429 and ITF Incapsula responses; verify bounded backoff,
  circuit-breaker behavior, provenance, and no slate collapse.
- Test start times over Eastern daylight-saving transitions and verify the same
  `match_start_at_utc` in odds, prediction, feature, skip, and dashboard rows.

### Feature and model drill

- Run field-exact historical/live parity fixtures at several chronological
  cutoffs, including week boundaries and same-event rematches.
- Verify no post-cutoff match, rank, odds, or result contributes to a feature.
- Reproduce the promoted artifacts on their declared dataset orientation.
- Validate registry hashes, deserialization, feature count/order, scaler, and NN
  calibration version independently.
- Evaluate candidates only on named chronological tiers and intersections;
  record log loss, Brier, ECE, calibration slope, accuracy, and both flat and
  Kelly counterfactual ROI with sample sizes.

### Live soak acceptance

For at least seven consecutive days after controlled deployment:

- every scheduled run reaches a terminal status;
- latest attempt and accepted state remain distinct and internally consistent;
- every accepted manifest count equals the visible table generation;
- settlement and immutable lineage never regress;
- no completed/past-start match enters inference;
- no malformed or missing market is represented as 50/50;
- pending exposure never exceeds equity and available capital never goes below
  zero;
- no source failure erases durable event metadata;
- dashboard freshness alerts fire on missed SLA, and desktop/mobile browser
  checks show no console or network errors.

## Go/no-go checklist

| Gate | Required evidence | Current |
| --- | --- | --- |
| Shifted canonical duplicates | 595 reviewed, approved quarantine/removal applied, zero unreviewed candidates | **NO-GO** |
| Round-label conflicts | 213 official-source decisions | **NO-GO** |
| Player identity | 690 groups + 2,078 sequence IDs explicitly classified; FK/collision checks clean | **NO-GO** |
| Train/serve parity | Shared pure code + field-exact 141-feature chronological fixtures | **NO-GO** |
| Temporal leakage | Independent as-of review and tests | **NO-GO** |
| Paper account | 93 pending reconciled; exposure <= equity; available > 0 | **NO-GO** |
| Durable recovery | Atomic/stale/kill/restart drills pass | **UNPROVEN** |
| Source reliability | Empty/blocked-source tests pass; reliable TA/ITF collection path | **UNPROVEN** |
| Model artifacts | Registry hash/deserialization/schema validation in CI and cloud | **IMPLEMENTED LOCALLY** |
| Evaluation | Regenerated ledger on verified lineage; unbiased named cohorts | **BLOCKED BY DATA/PARITY** |
| Dashboard | Contract tests + real-data desktop/mobile QA + generation reconciliation | **IMPLEMENTED LOCALLY; DEPLOYMENT PENDING** |
| Operations | Seven-day clean soak and alert review | **NOT STARTED** |

## Recommended execution order

1. Merge only after the local/CI suite and browser QA pass; deploy database
   schema/projection changes to staging first.
2. Freeze new paper exposure and reconcile the 93 pending rows.
3. Back up the canonical database, create reviewed quarantine/identity maps, and
   execute cleanup on a clone before production.
4. Build shared train/serve feature code and the chronological parity/leakage
   harness.
5. Split blocked-source collection onto a residential worker and feed source
   observations into durable Postgres state.
6. Complete recovery/source drills, deploy, and run the seven-day soak.
7. Rebuild a versioned 2025/2026 candidate dataset with content/source hashes.
8. Only then begin retraining, calibration work, and feature experiments as
   registry-tracked candidates. Promotion remains a separate evidence-based
   decision.

That order protects the core dependency chain: trustworthy source facts lead to
clean canonical history; clean history plus train/serve parity leads to valid
features; valid features plus immutable odds/results lead to meaningful model
evaluation; only then does a polished dashboard represent reliable results
rather than merely polished uncertainty.
