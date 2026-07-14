# Pending Identity Remediation

Pending identity remediation is a separate review layer between the read-only
pending-bet report and settlement. It exists to record two decisions that the
operational logs cannot safely infer:

1. whether an orphan or semantically ambiguous match UID is an alias of one
   exact authoritative UID;
2. whether repeated recommendations represent one accidentally duplicated
   paper decision or deliberately separate exposure.

The contract is fail closed. It does not update `logs/all_bets.csv`, alter
account exposure, infer a winner, or settle a bet. Applying an approved plan
only appends entries to an immutable decision registry. Operational
materialization remains a later, separately reviewed step.

The report digest is an integrity checksum, not a trust root. Both plan and
apply reconstruct the complete report from exactly four named, distinct source
files and require exact object equality before a decision is considered.

Current contract versions are:

- review report `1.0.0`;
- reviewer decision document `1.0.0`;
- apply plan `1.0.0`;
- immutable decision registry `1.0.0`.

## Candidate versus authority

The report may show a unique prediction UID for the same normalized player
pair and date. It may also show a target attached to the same run and feature
snapshot ID. These are investigation leads only.

They are **not** remap authority. Historical snapshot ownership has had known
conflicts, names are not durable IDs, dates can shift, and feature-vector
contents cannot reconstruct match identity. The command therefore reports
`automatic_approvals: 0` even when every candidate join is unique.

A target UID is shown as a candidate only when every prediction-log row for
that UID has a complete identity and the complete identity set is exactly one
player-pair/date matching the case. Reused or incomplete target UIDs are
reported separately and cannot be selected. For selectable targets, the report
also publishes exact prediction source-row hashes. Original UIDs expose bound
bet, prediction-snapshot, and odds-history rows where available.

The same rule applies to duplicate labels. A deterministic match/side/date key
can prove that two rows describe the same nominal recommendation. It cannot
prove whether the operator intended one exposure or two.

## Build the review report

Run from `production/` and write artifacts to a private review directory:

```bash
../tennis_env/bin/python -m operations.pending_identity_remediation report \
  --prod-dir . \
  --output /secure/review/pending-identity-review.json \
  --decision-template /secure/review/pending-identity-decisions.json
```

The report is deterministic for identical input bytes and paths. It binds the
exact SHA-256 and byte size of:

- `logs/all_bets.csv`;
- `prediction_log.csv`;
- `prediction_snapshots.csv`;
- `odds_history.csv`.

Blank or duplicate `bet_id` values fail report generation before cases are
built; rows may never collapse through a dictionary lookup. Each case also has
a stable `subject_key`. Remap subjects are keyed by unique bet ID. Duplicate
subjects are keyed by the normalized pending identity, so adding or removing a
group member changes the case ID but cannot evade a prior subject decision.

It emits one `match_uid_remap` case per orphan UID and one
`duplicate_intent` case per deterministic duplicate group. Rows merely waiting
for a result are counted but are not misclassified as identity cases.

Review and template files never overwrite different existing content. Use a
new path when the source generation changes.

## Required evidence

Every approved decision requires both a locally retained raw source artifact
and a typed evidence envelope. Each evidence declaration must include:

- a unique `evidence_id`;
- one supported kind: `bookmaker_event_record`, `identity_capture_record`,
  `official_match_record`, `raw_source_artifact`, or
  `operator_intent_record`;
- a source URI using `https`, `http`, `s3`, `gs`, or `file`;
- a timezone-aware `observed_at_utc`;
- an artifact path relative to the decision document;
- the exact lowercase SHA-256 of that artifact;
- a typed-envelope path and exact envelope SHA-256;
- a nonblank source system and external record ID;
- a specific claim explaining what the source proves.

Envelope schema `1.0.0` binds the raw artifact hash, source URI, source system,
external record ID, observation time, evidence kind, and claim. Its decision
binding is exact:

- remaps bind the stable subject, case, bet and row hash, original UID, target
  UID, pair, date, and any superseded decision;
- duplicate decisions bind the stable subject, case, pending identity, every
  member bet and row hash, complete resolution partition, disposition, and any
  superseded decision.

The decision document also requires a reviewer, timezone-aware review time,
and a nontrivial reason for every decision. The tool reads and hashes both the
raw artifact and envelope and compares the envelope object to the exact
decision subject. Arbitrary hashed bytes, free-text claims, a URI, or a typed
checksum without this binding are not enough.

The envelope cannot invent the approval facts. The retained raw artifact is
parsed too:

- a remap requires a `bookmaker_event_record` or `identity_capture_record` JSON
  document with schema/type, source identity, exact normalized pair/date, and
  exactly two UID bindings. The original and target binding must each name an
  operational source row and reproduce an exact row SHA-256 from the regenerated
  report. The target binding must use the prediction log;
- a duplicate disposition requires an `operator_intent_record` JSON document
  with the stable subject, pending identity, every member bet ID and row hash,
  complete resolution, and an intent ID per member. `duplicate_of` requires one
  shared intent ID; `retain_distinct_exposure` requires distinct intent IDs.

An official-result document or generic opaque artifact may be supplemental,
but neither is sufficient by itself. Machine validation proves that the
retained bytes are structurally and semantically bound to the reviewed case.
It does **not** prove that the external source or reviewer is truthful; source
authenticity and reviewer authorization remain explicit operational trust and
access-control responsibilities.

A remap raw identity-capture artifact first binds both operational rows:

```json
{
  "identity_capture_schema_version": "1.0.0",
  "record_type": "bookmaker_event_identity_capture",
  "source_system": "bookmaker",
  "external_record_id": "event-123",
  "source_uri": "https://source.example/event/123",
  "observed_at_utc": "2026-07-09T00:55:00+00:00",
  "match_pair": ["<normalized-player-1>", "<normalized-player-2>"],
  "match_date": "2026-07-10",
  "uid_bindings": [
    {
      "role": "original",
      "match_uid": "<old-uid>",
      "operational_source": "bets",
      "source_row": 42,
      "source_row_sha256": "<exact-report-row-sha256>"
    },
    {
      "role": "target",
      "match_uid": "<canonical-uid>",
      "operational_source": "predictions",
      "source_row": 314,
      "source_row_sha256": "<exact-report-row-sha256>"
    }
  ]
}
```

Its separate envelope then binds those source bytes to the exact decision;
every placeholder must equal the bound report case and decision exactly:

```json
{
  "evidence_envelope_schema_version": "1.0.0",
  "evidence_id": "book-event-123",
  "evidence_kind": "bookmaker_event_record",
  "source": {
    "source_uri": "https://source.example/event/123",
    "source_system": "bookmaker",
    "external_record_id": "event-123",
    "observed_at_utc": "2026-07-09T00:55:00+00:00",
    "artifact_sha256": "<raw-source-sha256>"
  },
  "claim": "Stable source event ID binds both observed UIDs.",
  "assertion": "same_external_match_identity",
  "binding": {
    "case_type": "match_uid_remap",
    "subject_key": "<subject-key>",
    "case_id": "<case-id>",
    "bet_id": "<bet-id>",
    "bet_row_sha256": "<bet-row-sha256>",
    "original_match_uid": "<old-uid>",
    "target_match_uid": "<canonical-uid>",
    "match_pair": ["<normalized-player-1>", "<normalized-player-2>"],
    "match_date": "2026-07-10",
    "supersedes_decision_id": ""
  }
}
```

Acceptable evidence must carry an identity or intent fact that is independent
of the candidate join. Examples include a retained bookmaker record with a
stable external event ID, an official match record with an explicit source ID,
or an immutable operator/run artifact showing that a write was retried rather
than intentionally placed twice.

The following are not sufficient by themselves:

- normalized player names;
- match date, tournament, or round similarity;
- feature values, vector hashes, or snapshot similarity;
- model probabilities or odds similarity;
- a reviewer assertion without a retained source artifact.

## Decision types

`match_uid_remap` cases accept only `remap_match_uid`. The target must already
be one of the exact pair/date candidates in the bound report, and independent
source evidence must still establish the alias.

`duplicate_intent` cases accept either:

- `duplicate_of`, with one canonical bet ID and every other group member
  listed exactly once as a duplicate; or
- `retain_distinct_exposure`, preserving every group member as a separately
  intended decision.

`defer` records that the case remains unsupported. Deferred cases are shown in
the plan but are not added to the immutable registry. This allows later source
research without replacing a prior decision.

Registry decisions are append-only by stable subject. A first decision must
have a blank `supersedes_decision_id`. A later changed case or disposition must
explicitly name the currently active decision ID, and its new evidence envelope
must bind the full current subject. Old decisions remain in the chain but cannot
be replayed after supersession. Missing, forked, or stale supersession links are
hard conflicts.

## Build and review an apply plan

After editing the decision document and retaining the evidence files beside it:

```bash
../tennis_env/bin/python -m operations.pending_identity_remediation plan \
  --report /secure/review/pending-identity-review.json \
  --decisions /secure/review/pending-identity-decisions.json \
  --registry ./.private/pending-identity-decision-registry.json \
  --output /secure/review/pending-identity-plan.json
```

Planning requires exactly the four expected source descriptors, rejects
duplicate paths, regenerates the report from source bytes, and compares the
complete regenerated object to the reviewed report. It then rechecks every
evidence artifact/envelope hash and binding, target UID identity contract, case
membership, duplicate partition, stable-subject chain, and existing registry
entry. One case or stable subject cannot have two decisions in one document.
An active entry can be replayed only when it is completely identical.

The plan contains the exact post-registry document and its hash. It explicitly
states:

```json
{
  "canonical_bet_log_mutation": false,
  "settlement_mutation": false
}
```

Keep this distinction visible during review. Registry approval is not account
or settlement approval.

## Apply to the immutable registry

Copy the exact plan digest from the plan command, then apply:

```bash
../tennis_env/bin/python -m operations.pending_identity_remediation apply \
  --plan /secure/review/pending-identity-plan.json \
  --expected-plan-digest <64-lowercase-hex-digest>
```

Apply obtains an exclusive registry lock and rechecks:

- the expected and embedded plan digests;
- report, decision, registry, and operational-source hashes;
- an independent exact report regeneration from all four source files;
- every evidence artifact;
- every typed evidence envelope and external-record binding;
- a complete deterministic rebuild of the reviewed plan.

The registry update uses an fsynced temporary file and atomic rename. Registry
generation equals the number of immutable entries. Reapplying an identical
decision through a freshly built plan is a verified no-op. A stale source,
changed evidence file, changed decision document, or conflicting prior entry
fails before any write.

## Database boundary

No live Supabase change is part of this contract. CSV remains the operational
source of truth, and no reviewed decisions exist yet to migrate. During the
normalized staging rollout, registry entries can map losslessly to reviewed
conflict-resolution rows with the same case, decision, evidence, and record
hashes. That mapping should be proven in staging before any database-backed
identity registry becomes authoritative.

Do not publish raw evidence bodies or reviewer-private material in public
`dash_*` tables. Store evidence in private object storage and retain only URI,
checksum, capture time, and reviewed claims in the normalized operational
schema.

## Safe operating sequence

1. Generate a fresh report and template from one accepted source generation.
2. Preserve source artifacts outside the public dashboard and credential paths.
3. Review one case at a time; defer anything without independent authority.
4. Build and retain the exact digest-bound plan.
5. Apply only to the immutable registry.
6. Independently review registry-to-operational materialization before changing
   a bet UID, pending exposure, session, bankroll, or settlement.
7. Re-run pending reconciliation after any separately approved materialization.
