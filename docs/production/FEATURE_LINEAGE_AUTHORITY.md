# Feature Lineage Authority And CSV Round Trips

The ordered `base_141@1.0.0` vector hash remains the existing bit-exact SHA-256
contract. This document defines how duplicate serializations of that vector are
resolved; it does not redefine the vector or weaken its hash.

## Authority order

For a nonblank `feature_snapshot_id`:

1. `production/logs/features_*.csv` is immutable per-run authority.
2. `production/logs/feature_vectors.csv` is a derived application-facing copy.
3. Supabase `dash_features` is a derived durable projection.

All feature CSV readers use pandas `float_precision="round_trip"`. Duplicate
rows must agree on `run_id`, `match_uid`, and the ordered schema SHA-256. Two
immutable occurrences must retain the same exact v1 vector hash. A derived copy
may have a different exact hash only when every ordered element is equal within
`rtol=1e-12` and `atol=1e-12`; the immutable vector and hash remain canonical.
The alternate exact hash is tolerated duplicate-audit provenance only. It is
not accepted as a prediction's referential hash, substituted into
the immutable vector, or used to change `feature_contract.vector_sha256`.

Material vector divergence, identity disagreement, conflicting immutable
sources, contradictory partial status-`ok` payloads, and invalid durable rows
attempting to act as repair authorities fail closed. Skip/error rows remain
non-decision-grade. Their historical aggregate may omit the ordered vector, but
any fields retained by both copies must still agree within the same tolerance
and neither copy may claim an exact decision-grade hash.

## Live match identity and snapshot attachment

Forward live identity canonicalizes player names and removes only Bovada's
trailing numeric bucket count from the event key. Thus `Challenger - Lincoln
(8)` and `Challenger - Lincoln (3)` produce the same event component, while
date, round, surface, canonical event, and player orientation remain explicit.
Display tournament text may enrich independently when the canonical event key
is unchanged.

For snapshot-backed rows, prediction refresh is exact-`match_uid` only. The
feature snapshot ID is recomputed from `match_uid`, `run_id`, and oriented
players before prediction evidence is written. A mismatch aborts before a
prediction, shadow, odds, or bet can use that pointer.

A nearby pre-contract row with an otherwise identical metadata contract is
retired as `superseded_identity`, and the canonical replacement records the old
UID as an explicit alias. The same explicit supersession is allowed when an
unsettled, incomplete blank-round row enriches to a complete known round. A
date, nonblank-round, surface, canonical-event, or orientation shift is instead
persisted as `identity_conflict` with `features_complete=False`; an impossible
collision on an already-used exact UID raises and writes nothing.

An identity tombstone on any row blocks the whole match UID in settlement,
ground truth, replay, dashboard GOLD, shadow evaluation, and staking until an
explicit resolution. Dashboard metric publication uses the same match, run,
orientation, deterministic-ID, and vector-hash checks as the evaluation ledger;
it may not reconstruct GOLD from a hash-only join.

## Dashboard repair and clean-clone precedence

Dashboard publication reconstructs every locally available exact-ID row from
the immutable per-run source, validates any prior durable copy, and publishes
the repaired projection in the same all-or-nothing manifest transaction as the
other dashboard tables. Hydration uses the same resolver before writing
`feature_vectors.csv`, so a repaired vector/hash survives another CSV round
trip.

If a clean clone has no immutable row for an ID, an already accepted durable
`dash_features` row outranks a stale local aggregate after identity and
element-wise compatibility checks. This is a recovery fallback, not a way to
promote derived-only evidence to GOLD. An explicitly incomplete status-`ok`
row with neither an exact hash nor contradictory metadata may be preserved as non-GOLD
operational evidence; an invalid row that claims completeness/exactness cannot
become repair authority, and a material local/durable mismatch aborts.

Safe rollout procedure:

1. Run the read-only replay manifest and focused lineage/dashboard tests.
2. Deploy the code before attempting any normalized Postgres cutover.
3. Let the normal cloud hydration read one accepted Supabase manifest.
4. Let the normal transactional dashboard publication rebuild and replace
   `dash_features`; do not hand-edit the table or bulk-commit unrelated logs.
5. Verify the new manifest count and regenerate replay/ledger evidence from the
   hydrated files.

No normalized `raw`/`ops`/`ml` migration is part of this repair.

## Fresh-clone durability proof (2026-07-14)

Against clean commit `f5669c7`, the old verifier produced 55 `GOLD_REPLAY`
rows. The shared authority resolver produces 111, restoring exactly the 56
rows lost only to cross-format floating serialization. All 56 select immutable
authorities from 17 repository-tracked files; `git ls-files --error-unmatch`
passed for every file:

```text
production/logs/features_20260708_035812.csv
production/logs/features_20260709_005113.csv
production/logs/features_20260709_052134.csv
production/logs/features_20260709_203932.csv
production/logs/features_20260709_221840.csv
production/logs/features_20260709_234449.csv
production/logs/features_20260710_000958.csv
production/logs/features_20260710_002233.csv
production/logs/features_20260710_015303.csv
production/logs/features_20260710_043144.csv
production/logs/features_20260710_045243.csv
production/logs/features_20260710_082815.csv
production/logs/features_20260710_202330.csv
production/logs/features_20260711_013935.csv
production/logs/features_20260711_030418.csv
production/logs/features_20260711_045734.csv
production/logs/features_20260711_155708.csv
```

This proof is intentionally separate from older machine-local replay totals in
the dated cutover report. Those totals must be regenerated after merge; the
fresh-clone result proves that these 56 repairs do not depend on ignored local
feature files.
