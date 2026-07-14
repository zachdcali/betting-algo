# Pending Paper-Bet Reconciliation

The pending reconciliation command is a strictly read-only evidence tool. It
compares pending rows in `logs/all_bets.csv` with authoritative outcomes already
present in `prediction_log.csv`. It never scrapes, settles, voids, deduplicates,
or changes account equity. The current report schema is `1.1.0`.

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

## Safe operating sequence

1. Save the JSON/CSV outputs in a private review location.
2. Resolve duplicate identities first: decide whether repeated rows represent
   one paper decision or intentionally separate exposure.
3. Review exact UID outcomes and the proposed win/loss math row by row.
4. Investigate orphan and ambiguous UIDs against source evidence. Preserve
   void/cancellation semantics.
5. Make any approved corrections through a separately reviewed settlement or
   migration path. This auditor intentionally provides no apply mode.
