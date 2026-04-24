# Production Docs

This folder is the current source of truth for the live tennis pipeline.

Start here:

- [Operator Workflow](/Users/zachdodson/Documents/betting-algo/production/WORKFLOW.md)
  Day-to-day commands and run order
- [Production README](/Users/zachdodson/Documents/betting-algo/production/README.md)
  Quick architecture and logging map
- [Versioning](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md)
  How model-family versions, logging schema versions, and lineage quality should work
- [Model Releases](/Users/zachdodson/Documents/betting-algo/docs/production/MODEL_RELEASES.md)
  Plain-English release notes for promoted NN/XGB/RF versions
- [Dashboard README](/Users/zachdodson/Documents/betting-algo/dashboard/README.md)
  How to run the live operations dashboard and which CSVs it reads

Current production principles:

- The live pipeline is Tennis Abstract based, not the legacy UTR path.
- Prediction generation and auto-settlement are separate concerns.
- `prediction_log.csv` is the operational view, while `prediction_snapshots.csv`, `odds_history.csv`, and `logs/features_*.csv` are the lineage layer.
- Old rows without immutable snapshot ids are legacy history and should be treated differently from new schema-backed rows.
- Bovada scheduled start times and TA match-state checks are part of the safety layer: the orchestrator skips feature generation once a match is at/past the configured pre-start cutoff or appears to have already completed in TA history, so delayed runs do not drift into post-start inference.
- `logs/audit/run_history.csv`, `logs/audit/skipped_live_matches.csv`, and `logs/audit/settlement_audit.csv` are the audit layer for dashboards and ops debugging.
- `dashboard/app.py` is the first-class visibility layer for operators. It should consume the production CSVs, not invent a separate shadow dataset.
