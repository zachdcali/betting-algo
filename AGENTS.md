# AGENTS

Project instructions for future Codex/Claude-style maintenance sessions.

## Source Of Truth

- Live operations docs start at [docs/production/README.md](/Users/zachdodson/Documents/betting-algo/docs/production/README.md).
- Versioning rules live in [docs/production/VERSIONING.md](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md).
- Human-readable promoted model notes live in [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md).
- Side-model tuning workflow lives in [docs/modeling/EXPERIMENT_WORKFLOW.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/EXPERIMENT_WORKFLOW.md).
- Feature-engineering priorities and candidate feature-set guardrails live in [docs/modeling/FEATURE_ROADMAP.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/FEATURE_ROADMAP.md).
- The source of truth for **measured model performance** (live and offline, all families) is the model evaluation ledger: [docs/modeling/MODEL_LEDGER.md](/Users/zachdodson/Documents/betting-algo/docs/modeling/MODEL_LEDGER.md), regenerated from the `production/evaluation/` package. See the Model Evaluation Ledger section below.
- The live orchestrator is [production/main.py](/Users/zachdodson/Documents/betting-algo/production/main.py).
- The active live features path is TA-based, not the older UTR path.
- CatBoost and LightGBM are supported only as side experiments for now; do not
  promote or wire them into live inference without explicit versioning and
  registry updates.

## Model Versioning Rules

- Keep versioning separate by family:
  `nn`, `xgboost`, and `random_forest`.
- Do not silently overwrite a promoted production artifact.
- It is acceptable to keep a retrained model as a registry-tracked candidate when it has not yet earned promotion.
- When promoting a new model version:
  1. create a stable versioned artifact copy
  2. archive the previous promoted artifact with a stable versioned filename
  3. update [production/models/model_registry.json](/Users/zachdodson/Documents/betting-algo/production/models/model_registry.json)
  4. update [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md)
- Use [production/models/validate_registry.py](/Users/zachdodson/Documents/betting-algo/production/models/validate_registry.py) after registry changes.
- Every committed registry change must increment the positive
  `registry_generation` and set a timezone-aware `registry_effective_at`.
  Database release status is ordered by that explicit generation, never by
  import time; reusing a generation number with different registry bytes is an
  integrity failure.
- Current promoted artifact entries must pin model/scaler SHA-256 values and
  pass checksum, deserialization, and feature-count validation before cloud
  inference.
- A changed train/validation/test protocol warrants at least a minor version bump.
- Probability calibration for the NN should be tracked separately from the base NN artifact version.
- Keep live `probability_mode=calibrated` promotion blocked until immutable
  prediction and snapshot lineage persists `nn_calibration_version` and the
  importer, replay, and evaluation ledger carry that version. Candidate
  calibrators may still be checksum- and schema-validated offline.
- Keep the ordered feature schema and feature semantics as separate versioned
  contracts. The current representation is `base_141@1.0.0`; its historical
  and live semantics remain explicitly different until shared chronological
  parity passes. Do not hide feature behavior behind names such as `_v2` or
  `final_fixed`.
- Existing promoted artifact paths stay stable. New artifacts use
  `releases|candidates|archive/<family>/v<semver>/model.<format>`, with registry
  metadata binding the feature schema/semantics, training dataset, protocol,
  calibration, and content hashes.

## Training Rules

- Use chronological splits, never random splits.
- Current standard split for tree/NN retrains:
  train `< 2022-01-01`, validation `2022`, test `>= 2023-01-01`.
- Do not use the test era for early stopping, model selection, or threshold tuning.
- If an old archived artifact cannot be evaluated because the feature schema changed, handle that comparison gracefully and do not crash the training script.
- Optional CatBoost/LightGBM screening uses the same chronological splits via:
  `tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py --mode fixed --only-boosters --booster-feature-mode both`.
  The `native_cat` feature mode is experimental and replaces one-hot
  surface/level/round/hand/country/handedness groups for side-model evaluation
  only.
- Optional XGBoost recency-weighting screening uses:
  `tennis_env/bin/python src/models/professional_tennis/run_side_experiments.py --mode fixed --only-recency-xgb --recency-half-lives 3,5,8,12`.
  Validation/test rows stay unweighted; only training rows receive exponential,
  mean-normalized recency weights.
- Optional score/stat side features use:
  `tennis_env/bin/python src/models/professional_tennis/build_feature_set.py --feature-set performance_v1`.
  Train them with `run_side_experiments.py --feature-set performance_v1 --dataset-path <side_csv>`.
  Live pipeline runs may compute these fields and log side-model shadow
  predictions for the configured `performance_v1` XGBoost, CatBoost,
  LightGBM, and NN side candidates, but they must not affect betting decisions
  or promotion status without explicit registry/versioning work.

## Logging And Lineage

- `prediction_log.csv` is the operational log.
- `prediction_snapshots.csv`, `odds_history.csv`, and `logs/features_*.csv` are the immutable lineage layer.
- Supabase `dash_*` tables are an additive durable recovery bridge while CSV
  remains the application-facing write format. Every cloud run hydrates from
  the latest accepted generation before settlement/dedupe, then publishes all
  dashboard tables in one transaction with a `dash_sync_manifest` generation.
  A cloud run is not successful if durable publication fails. Do not treat a
  git push, SQLite read model, or an individual `dash_*` table as an independent
  source of truth.
- `logs/betting.db` is derived from the CSV logs and must not be committed as
  hourly binary churn.
- The normalized Postgres operational schema is versioned independently from
  models, features, and logging. During migration, typed `raw`/`ops`/`ml`
  tables are additive and CSV remains authoritative until an import batch has
  exact key/hash parity, staging recovery drills pass, and the cutover is
  explicitly accepted. Never big-bang replace the CSV path or run a migration
  directly against production without backup/PITR and a staging proof.
- Import batches are keyed by source hashes, operational schema, normalizer
  version, and normalized target manifest. Facts can belong to multiple batches
  through append-only memberships. A reused idempotency key with a different
  semantic record hash is a hard conflict, never a first-wins retry.
- After cutover, Postgres is the operational source of truth; CSV/Parquet and
  SQLite are one-way exports. Raw source bodies belong in private object
  storage with URI/checksum provenance, not public dashboard tables.
- `python main.py --dry-run` should not start a betting session or write
  `logs/all_bets.csv`; use it for pipeline smoke checks when duplicate live
  bet logging would be risky.
- `BetTracker.log_bets()` should skip duplicate pending bets for the same match
  and bet side, so reruns do not double-log open recommendations.
- `logs/performance_v1_shadow_predictions.csv` is a side-model evaluation log,
  not an operational betting log. It can contain multiple `performance_v1`
  model families/versions and settlement scoring columns populated after the
  corresponding operational prediction settles.
- `logs/performance_v1_shadow_backfill.csv` is also side-model evidence only.
  Keep it separate from forward shadow logs and mark rows by backfill quality.
- Treat `logging_quality = snapshot_v2` rows as decision-grade.
- Treat `legacy_backfilled` rows as context, not exact lineage.
- Live predictions with noisy/defaulted features should still be logged with
  `features_complete=False` rather than skipped; clean accuracy excludes them,
  but settlement and bet reconciliation need the operational row.
- A skipped/error feature build must persist `build_status` and can never be
  labeled `features_complete=True` merely because no default list was emitted.
- Exact feature lineage stores the ordered 141-feature schema SHA-256 and vector
  SHA-256. Inference and GOLD verification require a structurally valid, finite
  persisted vector with valid binary/cardinality one-hot groups; a corrupt file,
  mismatched hash, or ambiguous duplicate ID fails closed.
- Immutable per-run `logs/features_*.csv` rows outrank derived
  `logs/feature_vectors.csv` and `dash_features` copies for the same snapshot
  ID. Parse feature CSV floats with round-trip precision; require run, match,
  and schema identity; tolerate derived serialization differences only when
  every ordered element is within `1e-12`. The immutable v1 SHA remains the
  sole prediction referential hash. Material or conflicting immutable evidence
  fails closed; an accepted durable row may preserve compatible explicitly
  incomplete evidence in a derived-only clean clone, but an invalid row cannot
  claim exactness or be promoted to GOLD. See
  `docs/production/FEATURE_LINEAGE_AUTHORITY.md`.
- Bovada display times are interpreted in `America/New_York`; immutable
  prediction/odds lineage also stores `match_start_at_utc`.
- Missing odds evidence is null, never a fabricated 50/50 market probability.
  A genuine observed even-money market remains a valid 0.5.
- Only `actual_winner in {1, 2}` is settled model ground truth. Voids and
  cancellations are excluded/refunded rather than coerced into a player-two win.
- Paper-account equity is configured starting capital plus realized settled
  P&L. Pending stake reserves capital across every session. New allocation is
  capped at 5% of equity per bet, 18% across the whole run, and the remaining
  available capital; zero availability is a hard no-new-exposure gate.
- Settlement should enrich existing predictions; it should not recompute historical inference.
- Settlement uses a conservative TA identity score across opponent, date
  window, tournament, surface, and round. Ambiguous/low-confidence matches
  should remain pending rather than guessed.
- Standalone settlement is intentionally paced for Tennis Abstract:
  `auto_settle.py` defaults to an 18-hour post-start grace period, 75 eligible
  candidates per run, an 8-second request delay, and early stop/cooldown on TA
  429s. It also skips rows attempted by real settlement runs within the last
  18 hours so repeated catch-up passes move beyond stubborn old rows. Use CLI
  flags only when you deliberately want a deeper backlog pass.
- `ta_match_unfinished` in settlement audit means TA still lists the matchup as
  upcoming/unfinished and no completed result has posted yet.

## Audit And Dashboard

- Audit CSVs under `production/logs/audit/` are first-class operational data:
  `run_history.csv`, `skipped_live_matches.csv`, `settlement_audit.csv`.
- The public static dashboard reads one manifest-pinned Supabase generation;
  the local Streamlit dashboard reads production CSVs for deeper inspection.
  Neither dashboard may invent model metrics or a shadow dataset. Browser model
  metrics come from `dash_model_metrics`, materialized through the evaluation
  ledger's metric code.

## Model Evaluation Ledger

- The source of truth for how every model actually performs (live + offline) is
  the ledger, regenerated by:
  `cd production && ../tennis_env/bin/python -m evaluation.ledger --prod-dir . --experiments-root ../results/professional_tennis/experiments --out-dir ../results/professional_tennis/ledger/<YYYY-MM-DD> --report ../docs/modeling/MODEL_LEDGER.md`
- Implemented in `production/evaluation/` as single-responsibility modules:
  `metrics` (pure scoring), `cohorts` (load + ground truth + tiers), `roi`
  (counterfactual staking), `offline` (experiment ingest), `ledger` (assemble +
  report + CLI). `cohorts.py` is the only module that knows the on-disk log
  format, so a future SQLite migration swaps its loaders and leaves the rest intact.
- Metric math lives once in `evaluation/metrics.py` and is reused by
  `dashboard/data.py`. Do not duplicate metric formulas.
- Correctness is derived by joining each model's probability to the authoritative
  `match_uid -> actual_winner` map from `prediction_log.csv`, never from
  pre-computed `*_correct` columns (which are stale or absent for several models).
- Cohort tiers are always reported with their label and n, never silently mixed:
  - GOLD = settled & `snapshot_v2` & `exact_feature_snapshot` &
    features_complete & feature snapshot ID verified against persisted lineage
    (decision-grade; headline).
  - COMPLETE = settled & features_complete (~65% `legacy_backfilled`; context only).
  - `*_intersection` = restricted to match_uids where all of nn/xgb/rf/market
    predicted; the cross-model "which model wins" verdict uses this, to avoid
    coverage bias (nn has broader settled coverage than xgb/rf).
- ROI is a counterfactual backtest at the odds logged at prediction time. Bet
  qualification matches live execution: model probability must clear the
  offered price's raw break-even probability by 0.02; de-vigged market
  probability remains the fair-market model baseline. Report both flat-stake
  and live-Kelly (multiplier 0.18, edge threshold 0.02, cap 0.05) modes.
  ROI break-even = beating the vig.
- Rank models by probabilistic quality first (log loss / Brier / ECE /
  calibration slope), accuracy last — consistent with the calibration + Kelly strategy.
- `performance_v1` shadow variants (10 as of 2026-06-29, defined in
  `DEFAULT_SHADOW_MODEL_SPECS`) are logged every run and scored in the ledger
  keyed by `model_version` (so same-family variants stay distinct). Evaluation
  uses one deterministic opening observation per `(match_uid, model_version)`
  joined to the operational opening feature snapshot; hourly repeats do not
  increase n. Each variant is scored on its own settled coverage; newly-added
  variants only accumulate live, settleable data as future slates settle.
  `native_cat` variants are not yet trackable (native-categorical live inference
  is unwired).

## Retraining Readiness Stop Gates

- Do not retrain or promote from the 2025/2026 extension until the canonical
  shifted-result duplicates and ambiguous player identities have been reviewed
  and remediated with backups and explicit mappings.
- Historical preprocessing and live serving currently have proven semantic
  differences in H2H orientation, set/activity windows, form trend, rank
  movement/volatility, first-surface defaults, and recent-H2H smoothing. Move
  those formulas into shared pure code and require field-exact chronological
  golden-fixture parity before new artifacts are considered valid.
- Never promote an inferred Monday/event date into canonical history. Rows
  without verified date provenance stay in quarantine until an official or
  otherwise explicit source confirms them.

## Commit Hygiene

- Avoid committing mutable local churn unless it is the point of the change:
  `production/prediction_log.csv`, ranking refresh files, ad hoc screenshots, and other live/generated data.
- Prefer code, docs, and explicit migrations in commits.
- Side-model experiment outputs should stay local under `results/professional_tennis/experiments/` unless there is a deliberate reason to commit a tiny text summary.
- Experiment output dirs are date/family/slug based and append
  `__run_HHMMSS` when a same-day slug already contains files; avoid relying on a
  generic same-day slug as a stable ledger.
- Candidate feature-set preprocessing should write versioned side outputs and
  should not silently replace the active 141-feature ML-ready dataset.
- Large local side datasets under `results/professional_tennis/feature_sets/`
  should remain uncommitted unless an explicit small manifest/summary is being
  promoted as documentation.

## When To Update This File

Update `AGENTS.md` when a future session changes:

- model-version promotion rules
- experiment folder conventions
- experiment harness commands
- the standard chronological split or walk-forward policy
- logging/lineage expectations that future sessions need to know up front
- the model evaluation ledger's cohort tiers, metrics, ROI methodology, or `production/evaluation/` structure

## Safety Notes

- For delayed production runs, inference should only happen before the configured pre-start cutoff and before the current match appears in TA history as completed.
- For future cloud scheduling, hourly odds capture is good, but settlement should remain a separate concern.
