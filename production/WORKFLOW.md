# Live Tennis Betting Workflow

This is the current production workflow, not the older UTR-era version.

Longer-form documentation for production lineage and version semantics lives in [docs/production/README.md](/Users/zachdodson/Documents/betting-algo/docs/production/README.md) and [docs/production/VERSIONING.md](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md).

## End-to-End Flow

```text
[Bovada Odds]
    ->
[Tournament Resolution]
    ->
[TA Feature Extraction]
    ->
[NN + XGB + RF Inference]
    ->
[Kelly Stakes + Bet Slips]
    ->
[Prediction / Odds Logging]
    ->
[Auto Settlement + Bet Sync]
```

## Daily Run

```bash
cd production
python main.py
```

For scheduled cloud prediction runs where you do not want old settlement work to block the odds/prediction path:

```bash
cd production
python main.py --skip-auto-settle
```

What that does:

- refreshes ATP rankings
- fetches current Bovada odds
- resolves surface / level / draw / round
- builds 141 live features from Tennis Abstract
- skips matches that have already reached the configured pre-start cutoff, or that already appear in TA history as completed, to keep the current match from leaking into features
- scores the slate with NN, XGBoost, and Random Forest
- logs configured `performance_v1` side-model shadow probabilities when the
  granular score/stat features are available; these do not affect bet selection
- writes per-run feature logs and bet slips
- logs immutable prediction and odds snapshots
- updates the deduped live prediction log
- auto-settles any older completed predictions before the new run starts

## Key Files

- `main.py`
  Main orchestrator
- `features/ta_feature_calculator.py`
  Current live feature builder
- `features/round_offsets.py`
  Shared round/date heuristic used by training and production
- `models/inference.py`
  Registry-driven artifact loading and live inference
- `prediction_logger.py`
  Deduped opening log plus append-only prediction/odds history
- `auto_settle.py`
  Settles `prediction_log.csv` from Tennis Abstract, syncs tracked bets, and
  scores matching rows in the `performance_v1` shadow log
- `sync_bet_tracker.py`
  Backfills tracked bet settlement from already-settled prediction rows
- `analyze_predictions.py`
  Accuracy, calibration, and edge reporting

## Logging Layout

- `prediction_log.csv`
  One operational row per live match call. Opening prices and original model call are preserved.
- `prediction_snapshots.csv`
  Append-only prediction snapshots. Use this for experiment lineage.
- `odds_history.csv`
  Append-only odds snapshots. Use this for line-movement analysis.
- `logs/audit/run_history.csv`
  Per-run pipeline and settlement summaries.
- `logs/audit/skipped_live_matches.csv`
  Explicit skipped-match ledger with reason codes.
- `logs/audit/settlement_audit.csv`
  Per-attempt settlement audit with reason codes.
- `logs/features_*.csv`
  Per-run feature snapshots keyed by stable match identifiers.
- `logs/performance_v1_shadow_predictions.csv`
  Optional forward side-model log for the `performance_v1` score/stat
  experiment. It can include the configured XGBoost, CatBoost, LightGBM, and NN
  side candidates. It is not used for staking or production settlement.
- `logs/performance_v1_shadow_backfill.csv`
  Optional controlled backfill for exact feature-snapshot rows. It should not be
  mixed with forward live-shadow evidence.
- `docs/production/VERSIONING.md`
  Current rules for model-family versions, logging schema versions, and lineage quality.
- `logs/all_bets.csv`
  Logged bet recommendations.
- `logs/bankroll_history.csv`
  Bankroll changes through tracked bet settlement.
- `logs/betting_sessions.csv`
  Session-level rollups.

## Settlement

For predictions:

```bash
python auto_settle.py
```

Defaults are intentionally gentle on Tennis Abstract and conservative about
identity matching: 18-hour post-start grace period, 75 eligible rows per run,
8 seconds between TA requests, and early stop/cooldown on repeated 429s. The
settler scores opponent/date/tournament/surface/round evidence; ambiguous or
low-confidence matches remain pending for a later pass.

For tracked bets that predate the new auto-sync path:

```bash
python sync_bet_tracker.py
```

For manual inspection / manual settlement:

```bash
python settle_bets.py --show-pending
python settle_bets.py --interactive
```

## Important Distinction

The repo still contains older helper paths such as `features/extract_features.py` and older UTR/cloud scripts. Those are not the active live production path.

The active path now uses two inference guardrails: Bovada scheduled start times and TA match-state checks. If a match is already started, inside the configured pre-start buffer, or appears to have already completed in TA history, the orchestrator skips feature generation for that row and records the skip reason in the per-run features log.

For dashboards and operational review, prefer the audit logs first:

- `run_history.csv` for run-level health and counts
- `skipped_live_matches.csv` for understanding why matches never reached prediction logging
- `settlement_audit.csv` for understanding why pending rows did or did not settle
