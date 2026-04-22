# Production Pipeline

Internal notes for the current live tennis pipeline.

Longer-form production docs now live under [docs/production/README.md](/Users/zachdodson/Documents/betting-algo/docs/production/README.md).

## Current Architecture

The active production path is:

1. Scrape Bovada odds with [fetch_bovada.py](/Users/zachdodson/Documents/betting-algo/production/odds/fetch_bovada.py)
2. Resolve tournament metadata with `tournaments/resolve_tournament.py`
3. Build live features from Tennis Abstract with `features/ta_feature_calculator.py`
4. Run NN, XGBoost, and Random Forest inference with `models/inference.py`
5. Calculate Kelly stakes with `utils/stake_calculator.py`
6. Log predictions and odds history with `prediction_logger.py`
7. Auto-settle results from Tennis Abstract with `auto_settle.py`

## Logging Model

There are now two logging layers:

- `prediction_log.csv`
  This is the operational, deduped match log. It preserves the opening snapshot for upcoming matches and tracks settlement.
- `prediction_snapshots.csv`
  This is append-only. Every logged prediction snapshot is preserved with immutable IDs.
- `odds_history.csv`
  This is append-only. Every logged odds snapshot is preserved with immutable IDs.

Supporting run artifacts:

- `logs/features_*.csv`
  Per-run feature snapshots with stable `match_uid` and `feature_snapshot_id`
- `logs/odds/bovada_tennis_*.csv`
  Raw odds scrapes
- `logs/all_bets.csv`
  Bet recommendations actually logged by the pipeline
- `logs/bankroll_history.csv`
  Bankroll changes over time
- `logs/betting_sessions.csv`
  Session summaries

## Model Artifacts

The model registry lives in `models/model_registry.json`.

`models/inference.py` now loads the active NN/XGB/RF artifacts from that registry instead of hardcoding only the filenames.

## Useful Commands

```bash
cd production
python main.py
python main.py --skip-auto-settle
python auto_settle.py
python analyze_predictions.py
python sync_bet_tracker.py
python tests/test_system.py
```

## Important Notes

- The live production path is Tennis Abstract based, not the older UTR-based extractor.
- `prediction_log.csv` is for the original live call on a match; use `prediction_snapshots.csv` and `logs/features_*.csv` for full historical lineage.
- `sync_bet_tracker.py` is useful for backfilling tracked bets from already-settled prediction rows.
- For future hourly cloud runs, `python main.py --skip-auto-settle` is the safer default. Settlement can run on its own cadence.
- `main.py` now skips feature generation for matches that are already at or inside a small pre-start buffer from the Bovada scheduled time, so a late run does not accidentally score a match after TA has turned it into history.
