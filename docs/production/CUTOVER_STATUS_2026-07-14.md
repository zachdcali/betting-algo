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

## Live Supabase observation

Read-only catalog/API inspection on 2026-07-14 found:

| Surface | Current observation |
| --- | ---: |
| Canonical players | 68,770 |
| Canonical matches | 972,759 |
| Canonical match stats | 260,035 |
| `dash_predictions` | 2,730 |
| `dash_snapshots` | 1,627 |
| `dash_odds_history` | 1,624 |
| `dash_shadow` | 14,920 |
| `dash_runs` | 74 |
| `dash_bets` | 190 |
| `dash_features` | 1,890 |
| `dash_settlement_audit` | 5,727 |
| `dash_skipped_live_matches` | 125 |
| `dash_bankroll` | 158 |
| `dash_sessions` | 65 |
| `dash_model_metrics` | 36 |

The accepted generation is
`sync_20260714T083207Z_d9d77850`. Every one of its 12 table counts matches the
rows carrying that exact `sync_id`; there is no mixed-generation dashboard
state. Both `latest_attempt_run_id` and `accepted_prediction_run_id` are
`run_20260714T080338Z`, whose lifecycle row is terminal `partial`, not stale
`running`.

That run fetched 104/104 odds rows, produced 84/104 predictions, persisted 20
skipped/error rows, and published the terminal manifest successfully. It placed
no new paper bets because the global pending-exposure gate correctly reported
zero available capital. The public dashboard therefore reports a current but
degraded signal rather than hiding the accepted data or claiming the portfolio
can allocate.

## Local normalized-import plan

The current read-only real-data plan is batch
`1a211ee3-68b7-566a-a0b9-20756fd8e9b0` and contains 42,269 rows including its
import batch control row:

| Target | Rows |
| --- | ---: |
| Feature snapshots | 6,413 |
| Prediction observations | 23,750 |
| Odds observations | 1,624 |
| Settlement attempts | 5,727 |
| Settlement events | 2,078 |
| Bet recommendations | 190 |
| Bet state events | 190 |
| Account ledger evidence | 158 |
| Pipeline runs | 74 |
| Skip events | 131 |
| Model releases | 18 |
| Model release status events | 18 |
| Model registry generations | 1 |
| Quarantined conflicts | 808 |

The 808 quarantined candidates are not accepted operational facts: 600 are
contradictory reused external prediction IDs and 208 are contradictory feature
snapshot candidates.
They require explicit reviewed resolution; the importer never selects a first
row merely because it appeared first.

A prior 38,675-row snapshot was applied to fresh disposable PostgreSQL 16 as
batch `8120d931-0953-570a-88c4-f4aff29ba4d1`. All target rows and 38,674 fact
memberships matched by key and semantic SHA-256, and an identical retry was a
parity-preserving no-op. That remains valid mechanism evidence, but it is not a
claim that the newer 42,269-row plan has been staged. The current plan must be
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

The replay manifest classifies 2,710 historical matches:

| Replay tier | Matches | Meaning |
| --- | ---: | --- |
| `GOLD_REPLAY` | 566 | exact, complete, pre-start vector and outcome/odds evidence |
| `EXACT_INCOMPLETE` | 1,155 | exact ID exists but one or more GOLD gates fail |
| `LEGACY_MATCHED` | 327 | one unambiguous same-orientation legacy vector; context only |
| `NO_VECTOR` | 662 | no safe vector or ambiguous vector evidence |

Promoted artifacts can be replayed read-only on compatible stored vectors. GOLD
same-schema results are documented as regression evidence, not as proof that
old feature formulas were semantically correct and not as a new untouched test
set for tuning.

| Promoted family | GOLD log loss | GOLD Brier | GOLD AUC | GOLD accuracy |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.622756 | 0.216884 | 0.708408 | 0.644876 |
| Random Forest | 0.638720 | 0.223689 | 0.689923 | 0.636042 |
| Neural Network | 0.688160 | 0.233085 | 0.672237 | 0.632509 |

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

## Next safe production sequence

1. Preserve the completed checkpoint: the audit writer/dashboard is deployed,
   a terminal manifest-pinned generation is live, and all 12 counts reconcile.
2. Merge the normalized contract/tooling while leaving its runtime sink off and
   applying no `raw`, `ops`, `ml`, or `api` DDL to live Supabase.
3. Provision a separate Supabase staging project and provide its URL only as
   `OPERATIONAL_STAGING_DATABASE_URL` / `OPERATIONAL_DATABASE_URL`.
4. Apply Postgres contracts `1.0.0` then `1.1.0`, import the exact reviewed
   manifest, repeat it, apply a changed second manifest, and time a restore.
5. Finish and independently review deterministic pending-bet plan/apply tooling;
   do not production-apply it until crash recovery, replay, shared locking, and
   ledger/session integrity gates are proven.
6. Wire the normalized sink in `shadow` mode only after exact logger timestamps,
   source IDs/provenance, bet/settlement records, and account journal semantics
   are available. Keep the accepted CSV plus `dash_*` writer required.
7. Prove run-by-run parity, least-privilege roles, recovery drills, and seven
   clean staging days before any operational read or write authority cutover.
8. Separately remediate canonical 2025/2026 identity/date conflicts and move
   training/live feature formulas into shared pure code with chronological
   golden-fixture parity before retraining.

The desired final state is one Supabase platform with canonical history,
private normalized operational schemas, reviewed API projections, and
reproducible CSV/Parquet exports. A separate staging project is a rollout tool,
not a competing permanent production database.
