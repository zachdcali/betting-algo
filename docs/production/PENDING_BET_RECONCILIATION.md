# Pending Paper-Bet Reconciliation

The pending reconciliation command is read-only by default. It compares pending
rows in `logs/all_bets.csv` with authoritative outcomes already present in
`prediction_log.csv`. It never scrapes results and it never uses a fuzzy,
name-only, pair-only, or date-only result fallback. The evidence-report schema
is `1.1.0`.

Run the summary from `production/`:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation --prod-dir .
```

The default terminal JSON is compact: it includes counts, stake totals, input
paths, and integrity checks, but omits row-level bet IDs, duplicate identity
hashes, and input SHA-256 values. Use `--verbose` when those details are needed
at the terminal:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . --verbose
```

Explicit JSON/CSV review exports remain fully detailed regardless of the
terminal verbosity setting.

The command writes nothing by default. A review artifact is created only when
its full output path is supplied:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . \
  --output-csv /secure/review/pending-bets.csv \
  --output-json /secure/review/pending-bets.json
```

## Classification contract

Each pending row has exactly one outcome classification:

- `exact_authoritative_winner_available`: the row's exact `match_uid` exists in
  the prediction log and resolves to one valid winner identity.
- `orphan_match_uid_absent_from_prediction_log`: the UID is blank or absent
  from the prediction log. No player/date inference is used to upgrade it.
- `unresolved_or_ambiguous`: the UID exists, but has no valid winner, conflicting
  winners, a void/cancellation marker, or observations spanning more than one
  normalized player-pair or match date. Reused UIDs fail closed even when their
  rows happen to name the same winner.

`duplicate_pending_match_side_date_identity` is an independent review label.
Duplicate exposure and result availability are orthogonal: the same row can be
both a duplicate and an exact-result row, or both a duplicate and an orphan.
The duplicate key is a deterministic hash of normalized, order-insensitive
match players, bet side, and match date. Event labels are excluded because the
book's suffix can change between scrapes.

Only winner values `1` and `2` are exact results. `-1`, `void`, and
`cancelled` remain `void_or_cancelled`; they are never treated as a player-two
win. The `bet_result_if_applied` and `profit_if_applied` fields are review math,
not settlement instructions.

Stake totals include only finite, positive stakes. Missing, nonnumeric,
nonfinite, zero, and negative values increment `pending.invalid_stake_rows` and
cannot produce proposed profit. For exact-result rows, missing, nonfinite, or
decimal odds at or below `1.0` are reported in
`invalid_exact_outcome_odds`; they are never silently treated as valid prices.
JSON output rejects NaN and Infinity rather than emitting non-standard JSON.

## Manual deterministic settlement plan

An explicit manual apply path exists for the subset that passes every safety
gate. It is not wired into pipeline startup, hourly runs, hydration, or
automatic settlement. The evidence report, settlement plan, and apply-audit
schemas are each `1.1.0` and remain independently versioned.

Result evidence has two modes:

- `exact-uid` is the default. It accepts only a valid result attached to the
  bet's exact `match_uid` in `prediction_log.csv`. This is exact settlement and
  exact model attribution, so the row is `metric_eligible=true`.
- `exact-pair-date` is an explicit plan-time recovery mode. If the original UID
  is absent, it may use exactly one valid prediction-log result for the same
  normalized player pair and operational match date. It settles the recorded
  paper exposure, but does not assert that the rotated UID is the same model
  prediction: `attribution_quality=unattributed_rotated_match_uid` and
  `metric_eligible=false`. Multiple semantic result UIDs fail closed.

The recovery mode may also consume a private official-result manifest. Manifest
schema `pending_result_recovery_evidence@1.0.0` binds each result to one unique
`bet_id`, the exact bet row and hash, its pair/date/side/stake/odds, one target
prediction row and orientation, an ATP external match ID and HTTPS URL, and a
retained raw artifact with byte count, observation time, and SHA-256. The
manifest is also bound to the complete bet and prediction file hashes. The
official winner must be the sole winner-marked player and the operational loser
must be the sole nonwinner in the exact raw match segment. The match's ATP day,
tournament ID, observation time, and operational-date shift are also bounded,
and `model_metric_write_authorized` must be false. Winner names must normalize
exactly unless a specific transliteration pair is explicitly code-reviewed; no
general first-name, surname, or fuzzy fallback is allowed. A valid record can
settle the paper account with `attribution_quality=uid_unlinked` and
`metric_eligible=false`; it never rewrites `prediction_log.csv` or creates a
model result.

Settlement truth and model attribution are deliberately separate contracts. A
known match winner can close a specific, hash-bound paper exposure without
claiming that a broken or rotated internal UID is valid model lineage.

A row is plan-eligible only when all applicable gates are true:

- `bet_id` is nonblank and unique across the entire bet log;
- the row is still `pending`, with blank outcome, profit, bankroll-after,
  settlement timestamp, settlement-note, and settlement-quality fields;
- its selected evidence mode yields exactly one non-conflicting result under
  the evidence rules above;
- no void/cancellation marker, conflicting winner, incomplete identity, reused
  semantic UID, or official/internal evidence conflict exists;
- the bet's normalized pair, date, selected player, and `bet_on_player1`
  orientation exactly match the bound result identity;
- stake is finite and positive, and decimal odds are finite and greater than
  `1.0`;
- the session ID is nonblank. When it maps to exactly one session, the bet must
  not predate that session and exactly one canonical `Session started` bankroll
  event must have the same timestamp and initial balance and occur before the
  bet;
- global paper-account arithmetic is valid: statuses are supported, all odds,
  stakes, and session-summary inputs are finite, and every settled/void row has
  compatible outcome, timestamp, bankroll, and exact win/loss/refund math.

Normalization only removes representational differences such as case, accents,
and punctuation. It is not fuzzy identity matching. A wrong or orphan UID stays
rejected in the default mode; recovery requires the explicit exact pair/date
contract or a complete bet-bound official manifest.

Recorded duplicate exposures are not collapsed. If two unique `bet_id` rows
represent exposure to the same pair/date/side, each stake is a separate ledger
fact and each eligible row settles independently. The plan and apply audit keep
the group size, position, and member IDs so the duplication remains visible.

Unavailable session lineage does not block otherwise valid global paper-account
recovery. When a nonblank session ID is missing or nonunique, the plan records
that quality explicitly, appends an accounting event with a blank accounting
session ID, and does not fabricate or alter a session row. Exact session
lineage continues to update its existing session summary.

Build a plan from `production/`. The plan output remains an operator-chosen
private review artifact. All four mutable recovery targets are canonical and
independently derivable from `logs/`; the apply audit must use
`.private/pending_reconciliation_apply_audit.csv`, separate from
`logs/audit/settlement_audit.csv`:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . \
  --plan-output /secure/review/pending-plan.json \
  --apply-audit ./.private/pending_reconciliation_apply_audit.csv \
  --lock-file ./logs/.operational_csv.lock \
  --transaction-dir ./logs/.pending_reconciliation_transaction \
  --starting-capital 1000
```

For reviewed rotated-UID and official-result recovery, opt in at plan time and
pass the private manifest when one is used:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . \
  --plan-output /secure/review/pending-recovery-plan.json \
  --result-evidence-mode exact-pair-date \
  --result-evidence-manifest /secure/evidence/official-results.json \
  --apply-audit ./.private/pending_reconciliation_apply_audit.csv \
  --lock-file ./logs/.operational_csv.lock \
  --transaction-dir ./logs/.pending_reconciliation_transaction \
  --starting-capital 1000
```

The plan is deterministic for identical input bytes and paths. It contains:

- SHA-256 for the bet, prediction, bankroll, session, and pre-existing apply
  audit inputs;
- a canonical SHA-256 plan digest;
- source-row hashes for every eligible bet, prediction observation, and
  available session, plus the official manifest and raw-artifact hashes when
  used;
- deterministic candidate ordering, expected result math, bankroll sequence,
  rejected rows, and machine-readable rejection reasons;
- settlement, attribution, metric-eligibility, result-evidence, exposure-group,
  and session-lineage quality for every candidate;
- the exact intended post-apply bet and bankroll rows, and applicable session
  rows, plus their hashes, so the mutation is reviewed rather than derived only
  after approval.

The settled bet row durably records `settlement_quality`,
`attribution_quality`, `metric_eligible`, `result_evidence_kind`, and
`result_evidence_sha256`. `BetTracker` preserves these additive columns on
future writes.

Rebuilding byte-identical content at the same plan path is a no-op. Different
content never overwrites an existing plan; use a new review filename.

Review the file, retain it immutably, and copy the exact `plan_digest` printed by
the command. The lock and recovery directory are canonical and non-overridable:
`BetTracker`, durable-state hydration, and this apply path all participate in
the same lock. A maintenance window remains the safest operator practice. Apply
requires both the reviewed file and that digest:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . \
  --apply-plan /secure/review/pending-plan.json \
  --expected-plan-digest <64-lowercase-hex-digest> \
  --apply-audit ./.private/pending_reconciliation_apply_audit.csv \
  --lock-file ./logs/.operational_csv.lock \
  --transaction-dir ./logs/.pending_reconciliation_transaction
```

If the reviewed plan used an official-result manifest, apply must receive the
same manifest path as well:

```bash
../tennis_env/bin/python -m operations.pending_reconciliation \
  --prod-dir . \
  --apply-plan /secure/review/pending-recovery-plan.json \
  --expected-plan-digest <64-lowercase-hex-digest> \
  --result-evidence-manifest /secure/evidence/official-results.json \
  --apply-audit ./.private/pending_reconciliation_apply_audit.csv \
  --lock-file ./logs/.operational_csv.lock \
  --transaction-dir ./logs/.pending_reconciliation_transaction
```

Apply rechecks the plan digest, every bound path, every input hash, the full
deterministically rebuilt plan, and every row hash while holding the exclusive
lock. It then updates `all_bets.csv`, `bankroll_history.csv`,
`betting_sessions.csv`, and the dedicated apply-audit CSV through a durable,
fsynced transaction journal. A normal failure restores every exact original;
after process death or power loss, recovery runs as part of the next first-level
canonical lock acquisition, before that holder enters its critical section. A
prepared transaction restores the complete old file set; a committed journal
must verify the complete new file set before cleanup. An unreadable, corrupt,
or wrong-scope journal fails closed and yields to no reader or writer. The
private audit directory is created with owner-only permissions and fsynced in
the production parent before journaling begins. Each
audit event has a deterministic ID bound to the plan digest and source hashes;
every event in one apply shares a single timezone-aware UTC timestamp.

Replaying a completely applied plan is a verified no-op. Replay rehashes the
complete settled bet, bankroll event, session row, and audit payload, then
recomputes current account and session invariants. A partial audit, missing
bankroll event, changed post-state field, duplicated audit event, or any other
conflict fails closed. A stale plan also fails: it cannot be used after any
bound input changes, even if the candidate row itself appears unchanged.

## Safe operating sequence

1. Save the JSON/CSV outputs in a private review location.
2. Review every recorded duplicate exposure as a separate stake; never collapse
   distinct `bet_id` rows during settlement.
3. Review exact UID outcomes and the proposed win/loss math row by row.
4. For recovery, verify each exact pair/date result and each official manifest
   record against its retained raw artifact. Keep recovered rotated/unlinked
   rows metric-ineligible. Preserve void/cancellation semantics.
5. Build and review a deterministic plan. Do not edit the plan after recording
   its digest.
6. Back up the operational CSV set, then run the explicit digest-gated apply.
7. Re-run the default read-only report and account/dashboard reconciliation.
8. Keep automatic startup wiring disabled until this manual path has produced
   reviewed production evidence.
