# Eligibility provenance and cutover

This is the staging and rollout contract for canonical player identity,
aliases, typed profile fields (especially height and hand), and prospective
round evidence. It is scaffolding only. Required mode remains disabled, and no
migration or data write from this work has been applied to live Supabase.

## Why this is the next pipeline gate

The accepted dashboard generation audited on 2026-07-14 contained 83 feature
snapshots: 19 complete and 64 incomplete. Incomplete-row markers overlapped:
height 36, identity conflict 30, round 21, structural round validation 21, and
rank volatility 2. Those counts cannot be added to claim recoverable sample
size. A replay result remains unknown until exact source inputs are pinned,
reprocessed, and accepted.

The read-only diagnostic must use the exact accepted dashboard export:

```bash
/Users/zachdodson/Documents/betting-algo/tennis_env/bin/python \
  -m production.operations.eligibility_coverage \
  --run-id run_20260714T210027Z \
  --sync-id sync_20260714T212732Z_33131875 \
  --snapshots '<pinned-export>/prediction_snapshots.csv' \
  --skips '<pinned-export>/skipped_live_matches.csv'
```

The tool fails when that run is absent and leaves `after_replay_rows` null. It
does not query Supabase, infer manifest acceptance, or estimate replay yield.

## Authority and table boundaries

Operational contract `1.2.0` keeps evidence, decisions, and compatibility
surfaces separate:

1. `raw.source_fetches` records collection attempts. Private bodies belong in
   object storage and are referenced by `raw.source_artifacts` URI and SHA-256.
2. `ops.player_entities` is a generation-bound canonical identity target.
3. Identity, alias, and typed profile observations are append-only and include
   source URI/hash, observation time, confidence, initial review state, raw
   artifact linkage, and optional evidence expiry.
4. Candidate round evidence lives only in
   `ops.eligibility_match_round_observations`. Its raw artifact must belong to
   the supplied fetch, and that fetch must belong to the supplied pipeline run.
   Round evidence has a finite expiry.
5. Explicit review events are append-only. Source observations can start only
   `unreviewed` or `quarantined`; a source cannot declare itself accepted.

The narrow compatibility import exception is typed as
`public_compatibility_projection` with a
`compatibility://public.players-player_aliases/...` URI. It does not claim a
raw artifact and still requires an explicit review event before use.

`public.players`, `public.player_aliases`, and
`api.current_match_metadata` remain the production compatibility surfaces.
Contract 1.2 neither alters the legacy metadata view nor inserts candidate
rounds into its underlying table. A candidate QF can appear in
`api.candidate_eligibility_match_metadata` while the byte-for-byte legacy row
continues to report R32. Promotion requires a later explicit cutover migration.

## Sealed generation lifecycle

A generation with no status is a mutable draft. Its immutable generation row
pins the expected projection seal and positive row count. The seal covers the
ordered `(table, idempotency_key, record_sha256)` projection for:

- player entities;
- identity, alias, and profile observations;
- dedicated eligibility round observations; and
- review events.

Both Python and PostgreSQL use the same versioned UTF-8 byte-length-prefixed
serialization and SHA-256. The database recomputes it at both lifecycle gates:

```text
draft -> candidate (sealed and frozen) -> accepted -> retired
                                   \----> rejected
```

Candidate sealing requires all of the following:

- the actual seal and row count equal the generation and candidate pins;
- every identity, alias, profile, and round observation has an explicit review
  no earlier than the observation and no later than candidate effective time;
- accepted active evidence contains no source-key, identity, alias, unified
  name, profile, or round conflict; and
- at least one active accepted identity binding exists.

Once candidate exists, all projection tables are frozen. An exact same-key,
same-hash retry remains a no-op; any new or changed fact is rejected. A
correction therefore requires a new generation. Acceptance recomputes the same
seal, repeats readiness at acceptance time, and must carry the exact candidate
seal/count. This prevents evidence that expires between candidate and accepted
from becoming operational authority.

Statuses are monotonic. `candidate -> rejected` and `accepted -> retired` are
terminal; an explicit retirement falls back to the highest older generation
whose current state remains accepted. Future-effective status and review rows
do not activate early. Seal/content writers are deliberately restricted to
PostgreSQL `READ COMMITTED`; stronger snapshot isolation fails closed.

Every reader pins all of:

- operational schema `1.2.0`;
- eligibility contract `1.0.0`;
- generation SHA-256;
- projection seal SHA-256; and
- positive projection row count.

Missing, stale, expired, ambiguous, conflicting, or mismatched evidence is
unavailable. It is never guessed or defaulted into authority.

Raw eligibility tables are a trusted-normalizer boundary, not a public write
API. SQL constraints require trimmed semantic identifiers, canonical uppercase
round codes, and lowercase URI schemes. The Python builders additionally
derive `observed_name_norm` and `alias_norm`; routine ingestion must use those
builders through a least-privilege writer role. Direct owner/admin inserts can
forge normalized-name columns and are reserved for disposable migration tests
and reviewed recovery. No browser role receives write access.

## One derived local profile bundle

Legacy mode continues to read and write `data/atp_heights.json` and
`data/atp_hands.json` exactly as before. Required mode does not consume those
files. It reads one derived bundle:

Before live feature construction, legacy mode now plans one bounded current-
slate ATP profile batch using only matches that pass the same pre-start clock
guard as inference. Players are deduplicated by canonical store ID;
lookups that can complete a matchup are prioritized, Challenger precedes ITF,
and unobserved/expired evidence precedes a fresh source-bound negative. One
browser page is reused for up to 32 official-profile attempts by default
(`ATP_PROFILE_RUN_HYDRATION_LIMIT`). A positive can enter this run-level lane
only when the official URL, rendered full name, canonical player ID, body
SHA-256, observed value, and physical range all bind. Every attempt persists
its exact status; unresolved players remain default-marked and ineligible.
Canonical display-key uniqueness and the strict cache allowlist cover every
slate player, including valid-height rows that need only handedness; a shared
display key across two canonical IDs fails the run before fallback.
The run-owned feature-store connection uses autocommit for reads, explicit root
transactions for profile write-through, and closes in the pipeline `finally`
path so a prior `SELECT` cannot turn a durable update into an uncommitted
savepoint.

- `eligibility_profiles_bundle.json` contains normalized name bindings,
  canonical player IDs, and accepted height/hand values;
- `eligibility_cache_manifest.json` pins its generation, projection seal, row
  count, schema/contract versions, SHA-256, counts, export time, and expiry.

The exporter checks name conflicts before reading profiles. A normalized name
that maps to multiple canonical IDs aborts the whole export. Publication uses
an exclusive lock, unique temporary files, file and directory `fsync`, atomic
replacement, and writes the manifest acceptance marker last. The bundle
expires at the earliest of its evidence expiry and a hard 15-minute TTL. Local
hashes assume a trusted filesystem; the short TTL bounds retirement staleness,
not hostile local modification.

Staging export:

```bash
export ELIGIBILITY_PROVENANCE_GENERATION_SHA256='<accepted-generation-sha256>'
export ELIGIBILITY_PROJECTION_SEAL_SHA256='<accepted-projection-seal-sha256>'

/Users/zachdodson/Documents/betting-algo/tennis_env/bin/python \
  -m production.operations.export_eligibility_cache \
  --database-url-env OPERATIONAL_STAGING_DATABASE_URL \
  --generation-sha256 "$ELIGIBILITY_PROVENANCE_GENERATION_SHA256" \
  --projection-seal-sha256 "$ELIGIBILITY_PROJECTION_SEAL_SHA256" \
  --output-dir data
```

The exporter sets its database transaction read-only. Required scraper mode
rejects caller-supplied cache dictionaries or forged bundle objects, partial
files, wrong pins, expired evidence, and any manifest/payload mismatch. It does
not scrape or write through after a miss. Required feature calculation resolves
both players from the bundle on every call, even when the legacy compatibility
store has populated height/hand values. It replaces those values with accepted
evidence, clears unavailable fields, and blocks inference when a bundle
canonical player ID disagrees with the store identity. The supported manual
scraper entry point is:

```bash
python -m production.scraping.atp_height_scraper 'Player Name'
```

Required mode remains off:

```bash
unset ELIGIBILITY_PROVENANCE_MODE  # current production behavior

# Do not set this until every cutover gate below is explicitly accepted.
# export ELIGIBILITY_PROVENANCE_MODE=required
```

Only `legacy` and `required` are valid values. Any other nonblank value fails
before cache access or compatibility-table mutation.

## Staging and deployment gates

Use a fresh disposable staging database. This finalized migration includes a
preflight that rejects a database claiming `1.2.0` with the incompatible
rejected-draft shape. `CREATE TABLE IF NOT EXISTS` is not a forward migration.
If any non-disposable environment ever applied that draft, stop and write a
reviewed forward migration; do not edit history or run this file over it.

Do not enable required mode or target production until all of these pass:

1. apply `1.0.0`, `1.1.0`, and final `1.2.0` twice at their version boundaries
   on PostgreSQL 16;
2. preserve raw bodies privately and prove artifact -> fetch -> run linkage;
3. prove Python/PostgreSQL seal parity, exact key/hash parity, deferred trigger
   ordering, late-write races, immutable retries, rollback, and recovery;
4. prove every legacy `api.current_match_metadata` row/column is JSON-equivalent
   before and after candidate round acceptance;
5. review every observation and conflict pile; quarantine unresolved evidence;
6. export and validate the one-bundle cache, including expiry and retirement;
7. define and test the least-privilege writer/reader roles, function `EXECUTE`
   grants, and RLS policies. Current staging tests use an owner/admin connection
   and are not the final production role model;
8. run shadow comparisons without changing bet decisions and obtain explicit
   cutover acceptance; and
9. only then add a forward cutover migration and enable required mode.

This contract does not claim that historical incomplete rows became eligible,
that replay has occurred, or that the normalized store is live authority.
