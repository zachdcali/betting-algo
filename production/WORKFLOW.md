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
- skips matches that have already reached the configured pre-start cutoff, to keep TA history from leaking the current match into features
- scores the slate with NN, XGBoost, and Random Forest
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
  Settles `prediction_log.csv` from Tennis Abstract and now syncs tracked bets too
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
- `logs/features_*.csv`
  Per-run feature snapshots keyed by stable match identifiers.
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

The active path now also uses Bovada scheduled start times as an inference guardrail: if a match is already started or inside the configured pre-start buffer, the orchestrator skips feature generation for that row and records the skip reason in the per-run features log.
