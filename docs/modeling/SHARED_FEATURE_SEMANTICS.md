# Shared base-141 feature-semantics candidate

Status: **reserved candidate; activation is false**

- Ordered schema: `base_141@1.0.0`
- Candidate semantics: `base_141_shared@1.0.0`
- Historical default (unchanged): `sackmann_historical_legacy@1.0.0`
- Live default (unchanged): `ta_live_legacy@3.0.0`

The candidate kernel lives in
`production/features/base_141_shared.py`. It is deterministic, performs no
I/O, and mutates no caller-owned history. Historical preprocessing and TA live
serving have explicit opt-in adapters. Neither adapter opts in by default, and
the model registry continues to mark the candidate `reserved_candidate` with
`parity_verified=false`.

Passing the synthetic golden fixture proves the implementation contract. It
does **not** authorize current promoted artifacts to consume these changed
formulas. Activation requires a new immutable dataset release, chronological
evaluation, appropriately versioned model artifacts, the provenance gates
below, and an explicit registry promotion.

## Formula contract

Both adapters normalize evidence and the prediction reference to a calendar
day before comparison (timezone-aware inputs are converted to UTC first).
Sackmann's date-only midnight and a live
kickoff clock on that same day are therefore equivalent. All historical
evidence is evaluated strictly before the prediction day, and rolling windows
are half-open: `[as_of_day - window, as_of_day)`. A match on the prediction day
is never evidence for another match that day.

| Field family | `base_141_shared@1.0.0` behavior |
| --- | --- |
| First history / rust | No prior tournament emits the compatibility value `Days_Since_Last=60`, but `Rust_Flag=0`: missing history is unknown and is not observed inactivity. Both adapters use this exact pair. |
| Activity | Match and surface counts use the same half-open as-of windows in both adapters. Career surface experience counts all prior known matches; it is not capped to an arbitrary number of days. |
| `Sets_14d` | Sum parseable set-score tokens inside the 14-day window. Walkovers/defaults contribute zero. A genuinely absent or unparseable score uses the explicit historical estimate of `2.5` sets for that row. |
| Form trend | Exponentially weighted win share across every valid result in the 30-day window with weight `exp(-age_days / 15)`. The constant is a 15-day exponential time constant, not a mathematical 15-day half-life. Zero observations returns `0.5`; one or two observations are not discarded. |
| Rank change | `rank_at_or_before_cutoff - rank_as_of_ref`; positive means ranking improved. The cutoff anchor must be no more than half the requested lookback older than the cutoff. |
| Rank volatility | Population standard deviation (`ddof=0`) over ranks in the 90-day half-open window, requiring at least three valid observations; otherwise `0.0`. |
| First surface | No prior surface is unknown and neutral. A transition is flagged only when at least one known last surface differs from the current surface. |
| Career H2H | Counts and win rate are always from Player 1's perspective. Career win rate uses neutral-prior Laplace smoothing with `alpha=3`. |
| Recent H2H | Use the newest three P1-perspective meetings. Advantage is `Laplace(P1 wins, n, alpha=3) - 0.5`, including when only one meeting exists. |

Historical rows with the same inferred calendar day read one immutable
pre-day state. Their results are applied as one deterministic batch only after
every row for that day has been calculated. This prevents intra-day leakage
through activity, career surface/level/round counters, form/streak, left-handed
opponent records, and H2H. Tied source order is resolved deterministically so
permuting equal-day input rows cannot change the next day's vector.

The candidate's profile missingness contract is also shared and finite:

| Input | Shared candidate default |
| --- | --- |
| missing/nonpositive rank | `999` |
| missing rank points with missing/unranked rank | `0` |
| missing rank points for an otherwise ranked player | `500` |
| missing/invalid height | `180 cm` |
| missing/invalid age | `25 years` |
| missing/unknown hand | `U` |
| missing/unrecognized country | `Other` one-hot |

Historical preprocessing applies those values before calculating rank, height,
age, and points differences/averages/ratios. Both adapters then apply the same
finite fallback to any remaining nonnumeric/nonfinite ordered value and require
all 141 values plus the structural one-hot groups to validate before the
candidate output is accepted.

The remaining representation fields retain their existing definitions in the
candidate adapters. The golden test also checks their full ordered output so a
seemingly local temporal change cannot silently reorder or corrupt another
field.

## Opt-in adapters and guardrails

Historical candidate calculations opt in with:

```python
calculate_temporal_features(
    frame,
    feature_semantics_id="base_141_shared@1.0.0",
)
```

Full candidate preprocessing additionally requires an explicit side-output
path and fails closed if that path aliases the active legacy
`jeffsackmann_ml_ready_SURFACE_FIX.csv` by resolved/case-folded path, symlink,
hard link, or same inode. It emits the 141 base fields in `schema_141.json`
order and does not currently combine the `performance_v1` side-feature set.

Live candidate calculations opt in with:

```python
TAFeatureCalculator(feature_semantics_id="base_141_shared@1.0.0")
```

The no-argument live constructor remains on `ta_live_legacy@3.0.0`.

## Chronological golden proof

`tests/fixtures/base_141_shared_v1.json` is a source ledger with alternating
player orientation, score and rank histories, three direct H2Hs, multiple
surfaces, explicit profile values, and a hard as-of timestamp. The test adapts
that one ledger independently through:

1. the historical chronological state machine; and
2. the live TA-shaped player-perspective history builder.

`tests/test_base_141_shared_parity.py` requires:

- field-exact agreement for all 141 ordered fields (absolute float tolerance
  `1e-12` only);
- the pinned full-vector SHA-256
  `a304021ffbb1c52486df9433f7d4e213df258488f24eafabb88791d8c41b5a69`;
- finite numeric values and valid one-hot cardinalities;
- a real full-preprocessing row with missing rank/points/height/age/hand/country
  that remains finite and is field-exact against the live profile-only adapter;
- neutral first-history rust behavior in both adapters;
- equal-day snapshot isolation and input-permutation invariance;
- date-only historical midnight parity with a clocked live kickoff;
- stable-ID H2H selection, exact-full-name fallback, and surname-collision and
  ambiguity refusal fixtures;
- hard-link and case-folded active-output alias refusal;
- pinned hand-checkable values for every disputed family;
- positive rank-change sign for improvement;
- neutral no-history surface behavior; and
- an identical target vector after a future match with extreme ranks and score
  is appended to both inputs.

## Remaining source-provenance gates

The formulas above are unified. These input-provenance problems still block
activation and cannot be solved honestly by formula code alone:

1. **TA-fallback H2H canonical identity.** Store-backed history now retains and
   selects by `opp_id`. TA player history still exposes display names rather
   than a stable canonical opponent ID, so fallback accepts only an exact
   normalized full name and never surname-only evidence. Conflicting available
   IDs fail closed. This prevents Francisco/Juan Manuel Cerundolo conflation,
   but a renamed or abbreviated TA opponent can still undercount H2H; canonical
   IDs for fallback history remain an activation gate.
2. **Rank-as-of lineage.** Historical rows carry a rank recorded for that
   match. Live serving passes the current rankings snapshot, but the feature
   input does not yet bind that value to an immutable snapshot effective time.
   A future/backdated replay could therefore supply a rank unavailable at the
   requested as-of. Persist and validate rankings-snapshot ID/effective time
   before activation.
3. **Score completeness.** The set formula distinguishes observed scores from
   the `2.5` fallback, but source coverage/quality must be measured and the
   fallback flag persisted. Otherwise two equal numeric vectors can hide
   different evidence quality.
4. **Match-date provenance.** The golden ledger uses explicit dates and an
   offset-neutral Monday/R32 context. Qualifiers, non-Monday starts, and
   two-week tournaments still depend on upstream round-date inference. Add
   official-date chronological fixtures for those event shapes; inferred
   Mondays may not be promoted into canonical history.

No model was retrained or promoted as part of this candidate work.
