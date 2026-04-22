# Dashboard

This folder contains the live operations dashboard for the tennis betting pipeline.

## Why Streamlit

We are using Streamlit for the first in-repo dashboard because it keeps the entire stack in Python, reads the existing CSV logs directly, and is easy to iterate on locally while the logging layer is still evolving.

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

## Current scope

The first version includes:

- Overview metrics and cumulative model-vs-market charts
- Live slate view for the latest pending snapshots
- Match explorer with snapshot and odds timelines
- Bets and bankroll view
- Ops and audit view, with graceful fallback when audit CSVs do not exist yet

## Important interpretation rule

Use the `Decision-grade only` filter when you want the cleanest cohort for evaluating current operational performance. That filter keeps only `snapshot_v2` rows with `exact_feature_snapshot` lineage.
