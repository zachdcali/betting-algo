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
8. Publish a manifest-versioned durable state generation with `dashboard_sync.py`

## Logging Model

There are now two logging layers:

- `prediction_log.csv`
  This is the operational, deduped match log. It preserves the opening snapshot for upcoming matches and tracks settlement.
  Rows with noisy/defaulted live features are still logged with
  `features_complete=False`; clean accuracy reporting excludes them, but the
  operational row remains available for settlement and bet reconciliation.
- `prediction_snapshots.csv`
  This is append-only. Every logged prediction snapshot is preserved with immutable IDs.
- `odds_history.csv`
  This is append-only. Every logged odds snapshot is preserved with immutable IDs.
- `logs/audit/run_history.csv`
  One row per pipeline or standalone settlement run with stage counts and outcome summaries.
- `logs/audit/skipped_live_matches.csv`
  Append-only audit log for matches skipped from live prediction lineage, including reason codes.
- `logs/audit/settlement_audit.csv`
  Append-only audit log for every settlement attempt and why it did or did not settle.
- `.private/pending_reconciliation_apply_audit.csv`
  The canonical private file for manual digest-gated backlog settlement events.
  It is independently derivable for crash recovery, never mixed into
  source-evidence settlement audit, and is not written by the normal pipeline.

Supporting run artifacts:

- `logs/features_*.csv`
  Per-run feature snapshots with stable `match_uid` and `feature_snapshot_id`.
  These are immutable authority over `feature_vectors.csv`/`dash_features`;
  see `docs/production/FEATURE_LINEAGE_AUTHORITY.md` for the round-trip and
  duplicate-reconciliation contract.
- `logs/performance_v1_shadow_predictions.csv`
  Forward side-model predictions for the score/stat `performance_v1`
  experiment. The live pipeline logs configured one-hot XGBoost, CatBoost,
  LightGBM, and NN side candidates here, then fills settlement/correctness
  columns after the operational prediction settles. Useful for evaluation, not
  live betting decisions.
- `logs/performance_v1_shadow_backfill.csv`
  Controlled side-model backfill over exact feature snapshots; useful as
  experiment evidence only
- `logs/odds/bovada_tennis_*.csv`
  Raw odds scrapes
- `logs/all_bets.csv`
  Bet recommendations actually logged by the pipeline
- `logs/bankroll_history.csv`
  Bankroll changes over time
- `logs/betting_sessions.csv`
  Session summaries

Durability and presentation:

- Supabase `dash_*` tables are an additive recovery bridge for the ephemeral
  runner. `dashboard_sync.py --hydrate` runs before the pipeline; terminal
  state is published transactionally afterward. `dash_sync_manifest` identifies
  the accepted generation and exact row counts.
- `logs/betting.db` is a derived SQLite read model, not hot persistence and not
  an hourly commit artifact.
- Paper sizing uses persistent account equity (starting capital plus settled
  P&L), reserves every pending stake across sessions, and caps new exposure at
  5% per bet, 18% per run, and remaining available capital.
- `docs/index.html` is the public dashboard. It pins every request to one
  manifest generation. `dashboard/app.py` remains the local forensic dashboard.

## Model Artifacts

The model registry lives in `models/model_registry.json`.

`models/inference.py` now loads the active NN/XGB/RF artifacts from that registry instead of hardcoding only the filenames.

Promoted model-family release notes live in [docs/production/MODEL_RELEASES.md](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md).

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

Dashboard:

```bash
cd /Users/zachdodson/Documents/betting-algo
tennis_env/bin/streamlit run dashboard/app.py
```

## Important Notes

- The live production path is Tennis Abstract based, not the older UTR-based extractor.
- `prediction_log.csv` is for the original live call on a match; use `prediction_snapshots.csv` and `logs/features_*.csv` for full historical lineage.
- `sync_bet_tracker.py` is useful for backfilling tracked bets from already-settled prediction rows.
- For future hourly cloud runs, `python main.py --skip-auto-settle` is the safer default. Settlement can run on its own cadence.
- `BetTracker.log_bets()` skips duplicate pending bets for the same match and
  bet side, so rerunning a slate should not double-log open recommendations.
- `python main.py --dry-run` does not start a betting session or write
  `logs/all_bets.csv`, although it still exercises odds/features/predictions.
- `python -m operations.pending_reconciliation` is read-only by default. The
  optional phase-one writer requires an explicit plan output, dedicated apply
  audit at its canonical private path, canonical shared lock/recovery paths,
  reviewed plan file, and exact plan digest; it is manual only.
- `main.py` now skips feature generation both when a match is already at/inside a small pre-start buffer and when the matchup already appears to have completed in Tennis Abstract history, so a late run does not accidentally score a post-start match as if it were still upcoming.
- `auto_settle.py` now defaults to a safe backlog pace: 18-hour settlement
  grace period, 75 eligible rows per run, 8 seconds between TA requests, and
  early stop/cooldown on repeated TA 429s. It skips rows attempted by real
  settlement runs in the last 18 hours so catch-up reruns move past stubborn
  old misses. The matcher uses opponent plus date/tournament/surface/round
  evidence and leaves ambiguous or low-confidence rows pending.
- The audit CSVs under `logs/audit/` are the easiest foundation for future dashboards because they explain run outcomes, skipped matches, and settlement reasons directly instead of forcing you to reconstruct them from `prediction_log.csv`.
- Settlement uses `ta_match_unfinished` when Tennis Abstract still lists the
  matchup as upcoming/unfinished, instead of grouping that state into
  `opponent_not_found`.
- The dashboard under `dashboard/` reads `prediction_log.csv` for settled performance, `prediction_snapshots.csv` and `odds_history.csv` for live lineage, and the audit CSVs when they exist.
- `performance_v1` shadow logging is allowed as experiment evidence, but it stays outside the production registry and outside the operational betting log until explicitly promoted.
- Retraining/promotion is currently gated on reviewed canonical-history and
  player-identity cleanup plus exact train/serve feature parity. See the
  production readiness audit before building 2025/2026 artifacts.
