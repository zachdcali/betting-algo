# Operational Cutover Status — 2026-07-14

This is the dated truth surface for the CSV-to-Postgres migration. It keeps
branch-local implementation, the current live Supabase project, and future
production authority separate.

## Executive status

- The existing Supabase project remains live and unchanged by the normalized
  database work.
- It currently owns canonical tennis history in `players`, `matches`,
  `match_stats`, `match_conflicts`, and `ingest_runs`.
- Its `dash_*` tables remain the accepted cloud-run recovery/dashboard bridge.
- CSV remains the application-facing operational write contract.
- Postgres contract `1.1.0`, importer normalizer `1.0.0`, replay tooling, and
  guarded runtime components are branch-local and locally tested only.
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
| `dash_predictions` | 2,675 |
| `dash_snapshots` | 1,487 |
| `dash_shadow` | 13,740 |
| `dash_runs` | 72 |
| `dash_bets` | 188 |
| `dash_sync_manifest` | absent |

The latest GitHub Actions job completed successfully, generated predictions,
and reported a dashboard sync. However, live `dash_runs` retained that run as
`running`, and the public API has no `dash_sync_manifest`. This is a code/runtime
version mismatch:

- production `main` still has the old publish-before-terminal-flush ordering;
- the audit branch fixes that order and adds transactional manifest publication;
- the new dashboard is correctly withholding unpinned rows until that writer is
  deployed and produces its first accepted generation.

Do not describe the current blank dashboard as "no data." It is intentionally
reporting that the available data has no accepted generation contract.

## Local normalized-import plan

The current read-only real-data plan contains 38,675 rows including its import
batch control row:

| Target | Rows |
| --- | ---: |
| Feature snapshots | 6,185 |
| Prediction observations | 21,354 |
| Odds observations | 1,409 |
| Settlement attempts | 5,588 |
| Settlement events | 2,053 |
| Bet recommendations | 153 |
| Bet state events | 153 |
| Account ledger evidence | 158 |
| Model releases | 18 |
| Model release status events | 18 |
| Model registry generations | 1 |
| Quarantined conflicts | 438 |

The 438 quarantined candidates are not accepted operational facts: 432 are
contradictory reused external prediction IDs, four are ambiguous settlement
identity candidates, and two are contradictory feature snapshot candidates.
They require explicit reviewed resolution; the importer never selects a first
row merely because it appeared first.

The exact plan was applied to fresh disposable PostgreSQL 16 as batch
`8120d931-0953-570a-88c4-f4aff29ba4d1`. All 38,675 target rows and 38,674 fact
memberships matched by key and semantic SHA-256. An identical full retry kept
the same counts and passed both target and membership parity. The PostgreSQL
integration suite also passed twice consecutively and proves a second changed
manifest preserves the first membership ledger while globally deduplicating
unchanged facts.

Terminal lifecycle retries now compare the stored semantic hash before they
can report success: an exact retry is a no-op and contradictory content aborts
the transaction. Model promotion is anchored to the globally latest registry
generation, so a release omitted by a newer generation cannot remain current;
Postgres also permits at most one promoted release per family and generation.

The imported paper projection reproduces the current evidence exactly:
starting capital $1,000; realized P&L -$427.0770624607107915; equity
$572.9229375392892085; pending stake $1,878.034773660351415; available capital
$0; 93 pending bets. This is parity evidence, not approval to make the
provisional account journal authoritative.

## Historical evidence recovered

The replay manifest classifies 2,600 historical matches:

| Replay tier | Matches | Meaning |
| --- | ---: | --- |
| `GOLD_REPLAY` | 551 | exact, complete, pre-start vector and outcome/odds evidence |
| `EXACT_INCOMPLETE` | 1,060 | exact ID exists but one or more GOLD gates fail |
| `LEGACY_MATCHED` | 327 | one unambiguous same-orientation legacy vector; context only |
| `NO_VECTOR` | 662 | no safe vector or ambiguous vector evidence |

Promoted artifacts can be replayed read-only on compatible stored vectors. GOLD
same-schema results are documented as regression evidence, not as proof that
old feature formulas were semantically correct and not as a new untouched test
set for tuning.

| Promoted family | GOLD log loss | GOLD Brier | GOLD AUC | GOLD accuracy |
| --- | ---: | ---: | ---: | ---: |
| XGBoost | 0.624153 | 0.217367 | 0.707619 | 0.646098 |
| Random Forest | 0.640081 | 0.224212 | 0.689132 | 0.637024 |
| Neural Network | 0.690992 | 0.233829 | 0.671002 | 0.633394 |

## Paper-account backlog

Read-only reconciliation currently finds 93 pending recommendations reserving
$1,878.034773660351415:

| Classification | Rows | Stake |
| --- | ---: | ---: |
| Exact authoritative winner available | 26 | $520.716261098644975 |
| Orphan UID absent from prediction log | 63 | $1,258.739264670652775 |
| Unresolved or ambiguous | 4 | $98.579247891053665 |

Fifty-one rows across 22 deterministic match/side/date identities are labeled
duplicate exposure. This is a review label, not authorization to delete or
settle rows. The 26 exact rows have review-only candidate P&L; no automatic
mutation path was added.

## Next safe production sequence

1. Merge and deploy the audit writer/dashboard changes; require one new
   manifest-pinned `dash_*` generation and confirm the stale `running` symptom
   is gone.
2. Provision a separate Supabase staging project and provide its URL only as
   `OPERATIONAL_STAGING_DATABASE_URL` / `OPERATIONAL_DATABASE_URL`.
3. Apply Postgres contracts `1.0.0` then `1.1.0`, import the exact reviewed
   manifest, repeat it, apply a changed second manifest, and time a restore.
4. Wire the normalized sink in `shadow` mode only after exact logger timestamps,
   source IDs/provenance, bet/settlement records, and account journal semantics
   are available. Keep the accepted CSV plus `dash_*` writer required.
5. Prove run-by-run parity, least-privilege roles, recovery drills, and seven
   clean staging days before any operational read or write authority cutover.
6. Separately remediate canonical 2025/2026 identity/date conflicts and move
   training/live feature formulas into shared pure code with chronological
   golden-fixture parity before retraining.

The desired final state is one Supabase platform with canonical history,
private normalized operational schemas, reviewed API projections, and
reproducible CSV/Parquet exports. A separate staging project is a rollout tool,
not a competing permanent production database.
