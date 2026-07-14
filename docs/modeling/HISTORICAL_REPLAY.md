# Historical Replay Manifest

The historical replay manifest inventories what can be evaluated again from
captured point-in-time feature vectors. It is an evidence and cohort-selection
tool, not a rescoring job. It never loads a model, overwrites a prediction, or
reconstructs a missing result.

## Why this exists

Historical rows have materially different evidence quality. Some retain an
immutable feature snapshot, genuine two-sided odds, and a conflict-free result.
Others can only be matched to an older unnamed vector by player orientation and
date. Combining those rows under one label would overstate the effective sample
size and make later model comparisons difficult to reproduce.

The manifest emits exactly one deterministic row per `match_uid` and assigns one
of four tiers:

- `GOLD_REPLAY`: the operational opening snapshot has a verified immutable ID,
  `snapshot_v2` / `exact_feature_snapshot` logging provenance, a structurally
  valid finite ordered 141-feature vector, a compatible schema hash, and the
  same ordered players in the prediction and feature evidence. Both the
  operational row and selected feature row must mark the vector complete. The
  prediction evidence and the vector itself must be timestamped strictly before
  match start. Genuine two-sided decimal odds and one conflict-free
  `actual_winner in {1, 2}` are also required.
- `EXACT_INCOMPLETE`: the operational row claims an exact snapshot ID, but one
  or more GOLD requirements fails. The reason codes say whether lineage,
  completeness, timing, outcome, odds, or schema evidence is missing.
- `LEGACY_MATCHED`: there is no immutable opening snapshot ID, but exact player
  orientation plus exact match date resolves to one structurally valid vector
  hash. Multiple copies are acceptable only when their content hash is
  identical. This is contextual evidence, not exact lineage.
- `NO_VECTOR`: no structurally valid vector can be matched, or multiple distinct
  legacy vectors fit the same match. The tool never resolves ambiguity by
  selecting the nearest timestamp or the most complete-looking retry.

## Opening selection and alternatives

`prediction_log.csv` is the operational opening-selection authority. Its
`feature_snapshot_id` remains selected even if a later feature build exists and
would move the row into a better tier. Later IDs and their source-file/row
locations are preserved in `alternative_snapshot_ids` and
`alternative_snapshot_provenance`, but they cannot upgrade the cohort.

Legacy matching is deliberately narrower: same ordered players, exact match
date, and exactly one validated vector hash. Reversed player order is not
accepted because it changes feature orientation. A different legacy vector is
ambiguity, not an invitation to cherry-pick.

Exact-ID evidence also cross-checks ordered player identity. Historical
`match_uid` values are not required to be byte-identical because the UID formula
changed when round and surface identity inputs were corrected; player
orientation is the stable fail-closed check across those known UID migrations.

The shared structural checks in `evaluation.cohorts.verify_feature_frame` are
used for both exact and legacy evidence. Valid vectors must contain the ordered
141-feature schema, finite numeric values, and valid binary/cardinality groups.
The manifest records both the schema SHA-256 and vector SHA-256, along with the
source CSV path and physical CSV row number.

The representation is identified centrally as `base_141@1.0.0`; the replay
manifest contract is SemVer `1.0.0`. Those values come from
`production/versioning.py`, rather than an informal filename suffix. Feature
schema and feature semantics are separate contracts: the ordered columns may
remain `base_141@1.0.0` while a formula correction receives a new semantics
identifier.

## Odds and results

Only decimal prices greater than 1.0 on both players count as genuine odds
evidence. A historical `market_p1_prob = 0.5` and `market_p2_prob = 0.5` with no
real decimal prices is explicitly rejected as
`FABRICATED_MARKET_0_5_WITHOUT_DECIMAL_ODDS`. Genuine observed even money with
`2.0 / 2.0` decimal prices remains valid.

Results are not recomputed. A replay row receives an authoritative result only
when every terminal observation for its `match_uid` agrees on player 1 or player
2. Voids, cancellations, missing outcomes, and conflicting outcomes cannot enter
`GOLD_REPLAY`.

## Running it

Read-only summary:

```bash
cd production
../tennis_env/bin/python -m evaluation.replay_manifest --prod-dir .
```

Versioned export:

```bash
cd production
../tennis_env/bin/python -m evaluation.replay_manifest \
  --prod-dir . \
  --out-dir ../results/professional_tennis/replay_manifests
```

Without `--out-dir`, nothing is written. With it, the command creates a new
`historical_replay_<UTC timestamp>/` directory. Existing exports are never
overwritten. The directory contains:

- `historical_replay.csv`: one row per match with tier, evidence, hashes,
  provenance, alternative snapshots, and reason codes.
- `manifest.json`: schema identity, counts by tier and reason, hashes and row
  counts for every input file, and the output CSV hash.

## What same-schema replay can prove

A `GOLD_REPLAY` vector can be fed to an artifact that accepts the identical
ordered schema. This can reproduce or compare model probabilities on the exact
captured inputs, then score them against the preserved result and prediction-time
prices. It is useful for artifact regression tests, same-schema candidate
screening, and reproducible counterfactual metrics.

It cannot prove that the original features were semantically correct. A vector
may faithfully preserve a wrong event date, contaminated identity, or an old
train/serve formula. It also cannot create a newly engineered feature or repair
an old feature value because the raw point-in-time source history may no longer
exist. Those changes require rebuilding from a cleaned, frozen canonical history
snapshot. A formula-only correction must receive a new feature-semantics
identity; the ordered schema version changes only when the representation
itself changes.

For that reason:

- Keep `GOLD_REPLAY` untouched as same-schema replay or forward-holdout evidence.
- Never tune calibration or choose a model on the same cohort later reported as
  an untouched test.
- Keep `LEGACY_MATCHED` separate from exact lineage in every metric table.
- Treat reconstructed vectors as a new, explicitly labeled dataset rather than
  silently replacing captured opening vectors.
- Version model artifacts by model family according to the production registry.
  Feature schema identity is the ordered schema hash; it should not be hidden in
  an informal filename suffix such as `_v2`.

## Replaying the promoted artifacts

`evaluation.replay_models` is the read-only next step after the evidence
manifest. It resolves the currently promoted NN, XGBoost, and Random Forest
entries from `model_registry.json`, verifies their pinned artifact hashes and
`base_141@1.0.0` contract, reloads each selected source vector, and verifies the
source-file, schema, and vector hashes again before inference.

```bash
cd production
../tennis_env/bin/python -m evaluation.replay_models --prod-dir .
```

The default command writes nothing. To retain a result deliberately:

```bash
cd production
../tennis_env/bin/python -m evaluation.replay_models \
  --prod-dir . \
  --out-dir ../results/professional_tennis/model_replays
```

That creates a non-overwriting `model_replay_<UTC timestamp>/` directory with
long-form probabilities, a separate model-by-tier metric table, and a JSON
manifest containing input, registry, model, scaler, and output hashes.

The tool preserves `GOLD_REPLAY`, `EXACT_INCOMPLETE`, and `LEGACY_MATCHED` as
separate metric cohorts. Only conflict-free preserved outcomes are scored;
missing or conflicting results remain unscored. Metrics reuse
`evaluation.metrics` and include `n`, accuracy, AUC, log loss, Brier, ECE, and
calibration slope/intercept. It performs no training, calibration, threshold
tuning, settlement, or historical outcome reconstruction, and it never writes
to operational logs.

These are same-representation results, not proof that every historical vector
used the current feature semantics. The output records the promoted artifact's
training and live semantics contracts, while older captured rows may predate
those contracts. Keep that limitation explicit when comparing replay metrics
with a newly rebuilt canonical dataset.
