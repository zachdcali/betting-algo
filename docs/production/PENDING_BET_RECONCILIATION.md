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

## Phase-one deterministic settlement plan

An explicit phase-one apply path exists for the narrow subset that passes every
safety gate. It is not wired into pipeline startup, hourly runs, hydration, or
automatic settlement. Plan schema `1.0.0` and apply-audit schema `1.0.0` are
versioned separately from the evidence report.

A row is plan-eligible only when all of these are true:

- `bet_id` is nonblank and unique across the entire bet log;
- the row is still `pending`, with blank outcome, profit, bankroll-after,
  settlement timestamp, and settlement-note fields;
- its nonblank exact `match_uid` resolves inside `prediction_log.csv`;
- every observation for that UID has one exact normalized player pair and date,
  every nonblank winner is `1` or `2`, and all valid winner observations resolve
  to one normalized winner identity;
- no void/cancellation marker, conflicting winner, incomplete identity, or
  reused semantic UID exists;
- the bet's normalized pair, date, selected player, and `bet_on_player1`
  orientation exactly match the authoritative prediction identity;
- the normalized pair/date/side identity occurs exactly once across all bet
  rows, including pending and settled history;
- stake is finite and positive, and decimal odds are finite and greater than
  `1.0`;
- the nonblank session ID maps to exactly one session, the bet does not predate
  that session, and exactly one canonical `Session started` bankroll event has
  the same timestamp and initial balance and occurs before the bet;
- global paper-account arithmetic is valid: statuses are supported, all odds,
  stakes, and session-summary inputs are finite, and every settled/void row has
  compatible outcome, timestamp, bankroll, and exact win/loss/refund math.

Normalization only removes representational differences such as case, accents,
and punctuation. It is not fuzzy identity matching. A wrong or orphan UID stays
rejected even if player names and dates look similar.

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

The plan is deterministic for identical input bytes and paths. It contains:

- SHA-256 for the bet, prediction, bankroll, session, and pre-existing apply
  audit inputs;
- a canonical SHA-256 plan digest;
- source-row hashes for every eligible bet, prediction observation, and session;
- deterministic candidate ordering, expected result math, bankroll sequence,
  rejected rows, and machine-readable rejection reasons;
- the exact intended post-apply bet, bankroll, and session rows plus their
  hashes, so session repair is reviewed rather than derived only after approval.

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
2. Resolve duplicate identities and orphan UID aliases through the separate
   [Pending Identity Remediation](PENDING_IDENTITY_REMEDIATION.md) contract.
   Candidate name/date or feature joins are not authority, and registry apply
   does not itself alter the bet log.
3. Review exact UID outcomes and the proposed win/loss math row by row.
4. Investigate orphan and ambiguous UIDs against source evidence. Preserve
   void/cancellation semantics.
5. Build and review a deterministic plan. Do not edit the plan after recording
   its digest.
6. Back up the operational CSV set, then run the explicit digest-gated apply.
7. Re-run the default read-only report and account/dashboard reconciliation.
8. Keep automatic startup wiring disabled until this manual phase has produced
   reviewed production evidence.
