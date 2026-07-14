# Operational Cutover Status — 2026-07-14

This is the dated truth surface for the CSV-to-Postgres migration. It keeps
repository implementation, the current live Supabase project, and future
production authority separate.

## Executive status

- The existing Supabase project remains live and unchanged by the normalized
  database work.
- It currently owns canonical tennis history in `players`, `matches`,
  `match_stats`, `match_conflicts`, and `ingest_runs`.
- Its `dash_*` tables remain the accepted cloud-run recovery/dashboard bridge.
- CSV remains the application-facing operational write contract.
- Postgres contract `1.1.0`, importer normalizer `1.0.0`, replay tooling, and
  guarded runtime components are repository-ready and tested, but not wired
  into the live orchestrator.
- No `raw`, `ops`, `ml`, or `api` migration has been applied to live Supabase.
- Retraining and model promotion remain blocked by canonical-history cleanup
  and shared train/serve feature parity, independent of database progress.
- Calibrated NN promotion is additionally blocked until immutable per-row
  `nn_calibration_version` lineage is present in logging, replay, import, and
  ledger paths.

## Live Supabase observation

Read-only catalog/API inspection on 2026-07-14 found:

| Surface | Current observation |
| --- | ---: |
| Canonical players | 68,831 |
| Canonical matches | 973,022 |
| Canonical match stats | 260,035 |
| Canonical match conflicts | 0 |
| Canonical ingest runs | 1,255 |
| `dash_predictions` | 2,786 |
| `dash_snapshots` | 1,827 |
| `dash_odds_history` | 1,820 |
| `dash_shadow` | 16,230 |
| `dash_runs` | 76 |
| `dash_bets` | 190 |
| `dash_features` | 2,204 |
| `dash_settlement_audit` | 5,731 |
| `dash_skipped_live_matches` | 200 |
| `dash_bankroll` | 158 |
| `dash_sessions` | 65 |
| `dash_model_metrics` | 36 |

This observation was taken after the 10:08 UTC publication. The accepted
generation is `sync_20260714T100757Z_df73bcd7`. Every one of its 12 table counts matches the
rows carrying that exact `sync_id`; there is no mixed-generation dashboard
state. Both `latest_attempt_run_id` and `accepted_prediction_run_id` are
`run_20260714T094649Z`, whose lifecycle row is terminal `partial`, not stale
`running`.

That run fetched 105/105 odds rows, produced 76/105 predictions, persisted 29
skipped/error rows, ingested 23 canonical event-result rows, and published the
terminal manifest successfully. The 29 non-inference rows are explicit: 14
failed the round one-hot contract, three TA profiles failed, three had entered
the five-minute pre-start buffer, and nine had already started. It placed
no new paper bets because the global pending-exposure gate correctly reported
zero available capital. The public dashboard therefore reports a current but
degraded signal rather than hiding the accepted data or claiming the portfolio
can allocate.

## Local normalized-import plan

The current read-only local-machine plan is batch
`fa5f4014-25a3-5306-bd4c-6aba97f49b74` and contains 46,268 rows across all 20
targets, including its import batch control row. Two immediate builds were
byte-identical and emitted no warnings:

| Target | Rows |
| --- | ---: |
| `raw.source_fetches` | 966 |
| `raw.source_artifacts` | 160 |
| `ops.import_batches` | 1 |
| `ops.import_conflicts` | 4,179 |
| `ops.pipeline_runs` | 76 |
| `ops.skip_events` | 200 |
| `ops.odds_observations` | 1,820 |
| `ops.settlement_attempts` | 5,731 |
| `ops.settlement_events` | 2,078 |
| `ops.bet_recommendations` | 190 |
| `ops.bet_state_events` | 190 |
| `ops.account_ledger` | 158 |
| `ops.paper_accounts` | 1 |
| `ops.paper_sessions` | 65 |
| `ml.feature_schemas` | 1 |
| `ml.feature_snapshots` | 4,998 |
| `ml.prediction_observations` | 25,417 |
| `ml.model_registry_generations` | 1 |
| `ml.model_releases` | 18 |
| `ml.model_release_status_events` | 18 |

The 4,179 quarantined candidates are not accepted operational facts: 3,438 are
feature candidates across 1,718 reused snapshot keys and 741 are prediction
candidates across 336 reused external keys. They require explicit reviewed
resolution; the importer never selects a first row merely because it appeared
first.

The plan above predates the feature-evidence recovery commit and must be
regenerated before staging. The clean `f5669c7` proof contained 74
repository-tracked `logs/features_*.csv` files. In particular, all 17 immutable
files selected by the 56 cross-format reconciliation repairs are tracked and
clean-clone reproducible; see `FEATURE_LINEAGE_AUTHORITY.md`. Regenerate the
count after integration, and freeze any future private lineage file with the
exact import source manifest before apply.

A prior 38,675-row snapshot was applied to fresh disposable PostgreSQL 16 as
batch `8120d931-0953-570a-88c4-f4aff29ba4d1`. All target rows and 38,674 fact
memberships matched by key and semantic SHA-256, and an identical retry was a
parity-preserving no-op. That remains valid mechanism evidence, but it is not a
claim that the newer 46,268-row plan has been staged. The current plan must be
applied and replayed in a fresh staging database before cutover. The PostgreSQL
16 integration suite passes against disposable databases and proves changed
manifests preserve prior memberships while deduplicating unchanged facts.

Terminal lifecycle retries now compare the stored semantic hash before they
can report success: an exact retry is a no-op and contradictory content aborts
the transaction. Model promotion is anchored to the globally latest registry
generation, so a release omitted by a newer generation cannot remain current;
Postgres also permits at most one promoted release per family and generation.

The current paper state reports starting capital $1,000; realized P&L
-$389.5770624607107; equity $610.4229375392893; pending stake
$2,028.0347736603512; available capital $0; and 129 pending bets. This is
read-only plan/accounting evidence, not approval to make the provisional
account journal authoritative.

## Historical evidence recovered

The replay manifest classifies 2,755 historical matches:

| Replay tier | Matches | Meaning |
| --- | ---: | --- |
| `GOLD_REPLAY` | 510 | exact, complete, pre-start vector and outcome/odds evidence |
| `EXACT_INCOMPLETE` | 1,256 | exact ID exists but one or more GOLD gates fail |
| `LEGACY_MATCHED` | 327 | one unambiguous same-orientation legacy vector; context only |
| `NO_VECTOR` | 662 | no safe vector or ambiguous vector evidence |

Promoted artifacts can be replayed read-only on compatible stored vectors. GOLD
same-schema results are documented as regression evidence, not as proof that
old feature formulas were semantically correct and not as a new untouched test
set for tuning.

| Promoted family | GOLD log loss | GOLD Brier | GOLD AUC | GOLD accuracy |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.625392 | 0.218126 | 0.703521 | 0.645098 |
| Random Forest | 0.640447 | 0.224613 | 0.683375 | 0.629412 |
| Neural Network | 0.698384 | 0.236339 | 0.663507 | 0.625490 |

The GOLD decline from 566 to 510 was not ordinary sample drift and no settled
outcome changed. Durable hydration exposed duplicate snapshot IDs whose
per-run CSV and aggregate JSON copies produce different bit-exact SHA-256
values. All 56 downgraded vectors are element-wise equal within `1e-12` and
their maximum absolute difference is `7.105e-15`. The branch now implements a
reviewed cross-format resolver: round-trip parsing, immutable per-run
precedence, identity agreement, and a `1e-12` tolerance used only to reconcile
the derived copy. The immutable bit-exact v1 hash remains the sole prediction
referential hash. A clean `f5669c7` proof moves 55 to 111 GOLD rows—exactly 56
restored—and all 17 selected authority files are tracked. The 510-row table
above remains dated pre-repair evidence and must be regenerated after merge;
normalized staging apply remains blocked until that fresh proof passes.

## Paper-account backlog

Read-only reconciliation currently finds 129 pending recommendations reserving
$2,028.0347736603512:

| Classification | Rows | Stake |
| --- | ---: | ---: |
| Exact authoritative winner available | 27 | $529.295509 |
| Orphan UID absent from prediction log | 63 | $1,258.739265 |
| Unresolved or ambiguous | 39 | $240.000000 |

Fifty-one rows across 22 deterministic match/side/date identities are labeled
duplicate exposure. This is a review label, not authorization to delete or
settle rows. The 27 exact rows have review-only candidate P&L; no automatic
mutation path was added.

An independent review of the separate plan/apply implementation found it is
not yet safe to merge or production-apply: recovery is not enforced for every
shared-lock reader/writer, replay does not revalidate authoritative prediction
outcomes, and ledger/session plus multi-session terminal-state gates remain
incomplete. The read-only classifier above remains valid; no backlog mutation
was performed.

## Next safe production sequence

1. Preserve the completed checkpoint: the audit writer/dashboard is deployed,
   a terminal manifest-pinned generation is live, and all 12 counts reconcile.
2. Merge the normalized contract/tooling while leaving its runtime sink off and
   applying no `raw`, `ops`, `ml`, or `api` DDL to live Supabase.
3. Provision a separate Supabase staging project and provide its URL only as
   `OPERATIONAL_STAGING_DATABASE_URL` / `OPERATIONAL_DATABASE_URL`.
4. Apply Postgres contracts `1.0.0` then `1.1.0`, import the exact reviewed
   manifest, repeat it, apply a changed second manifest, and time a restore.
5. Deploy and verify the implemented cross-format feature-lineage resolver,
   then regenerate both fresh-clone and machine-private replay/import evidence.
   Keep staging apply blocked until that refreshed proof reconciles exactly.
6. Finish and independently review deterministic pending-bet plan/apply tooling;
   do not production-apply it until crash recovery, replay, shared locking, and
   ledger/session integrity gates are proven.
7. Wire the normalized sink in `shadow` mode only after exact logger timestamps,
   source IDs/provenance, bet/settlement records, and account journal semantics
   are available. Keep the accepted CSV plus `dash_*` writer required.
8. Prove run-by-run parity, least-privilege roles, recovery drills, and seven
   clean staging days before any operational read or write authority cutover.
9. Separately remediate canonical 2025/2026 identity/date conflicts and move
   training/live feature formulas into shared pure code with chronological
   golden-fixture parity before retraining.

The desired final state is one Supabase platform with canonical history,
private normalized operational schemas, reviewed API projections, and
reproducible CSV/Parquet exports. A separate staging project is a rollout tool,
not a competing permanent production database.
