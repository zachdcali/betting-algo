# Dashboard

This folder contains the live operations dashboard for the tennis betting pipeline.

## Two dashboard surfaces

The public operations dashboard is `docs/index.html`. It is a static,
GitHub-Pages-compatible client that reads one accepted Supabase
`dash_sync_manifest` generation. Every table request is pinned to that
generation so the page cannot mix two publications.

The Streamlit app remains the local forensic surface. It keeps analysis in
Python, reads production CSV logs directly, and supports deeper drill-down while
the write path is still migrating.

The dashboard is built around the current production truth sources:

- `production/prediction_log.csv` for settled results and operational lineage
- `production/prediction_snapshots.csv` for append-only live prediction snapshots
- `production/odds_history.csv` for line movement
- `production/logs/audit/*.csv` for run, skip, and settlement audit data when available
- `production/logs/all_bets.csv` and `production/logs/betting_sessions.csv` for bet tracking

## Run locally

```bash
tennis_env/bin/streamlit run dashboard/app.py
```

Public static dashboard:

```bash
python3 -m http.server 8765
# open http://127.0.0.1:8765/docs/index.html
```

## Current scope

The first version includes:

- Overview metrics and cumulative model-vs-market charts
- apples-to-apples evaluation metrics for NN, XGB, RF, and market including AUC, ROC, Brier, log loss, and ECE
- insight cards for largest live model-market disagreement, recent model-vs-market performance, feature lineage coverage, and latest pipeline run status
- a prediction log browser with tournament/player filtering
- per-match drilldown showing model versions, player-vs-market probabilities, snapshot lineage, and the exact feature vector when `feature_snapshot_id` exists
- readable feature snapshot tables with player/context grouping and plain-English descriptions for the main engineered features
- Live slate view for the latest pending snapshots, including P1/P2 model-vs-market edge and lift columns
- Match explorer with snapshot and odds timelines
- Bets and bankroll view
- Ops and audit view, with graceful fallback when audit CSVs do not exist yet

## Important interpretation rule

Use the `Decision-grade only` filter when you want the cleanest cohort for
evaluating current operational performance. GOLD requires settled
`snapshot_v2`, `exact_feature_snapshot`, complete features, and a feature
snapshot ID verified against persisted lineage. Browser metrics are published
from `production/evaluation/`; the static client does not recalculate them.
