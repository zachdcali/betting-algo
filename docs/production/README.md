# Production Docs

This folder is the current source of truth for the live tennis pipeline.

Start here:

- [Operator Workflow](/Users/zachdodson/Documents/betting-algo/production/WORKFLOW.md)
  Day-to-day commands and run order
- [Production README](/Users/zachdodson/Documents/betting-algo/production/README.md)
  Quick architecture and logging map
- [Versioning](/Users/zachdodson/Documents/betting-algo/docs/production/VERSIONING.md)
  How model-family versions, logging schema versions, and lineage quality should work

Current production principles:

- The live pipeline is Tennis Abstract based, not the legacy UTR path.
- Prediction generation and auto-settlement are separate concerns.
- `prediction_log.csv` is the operational view, while `prediction_snapshots.csv`, `odds_history.csv`, and `logs/features_*.csv` are the lineage layer.
- Old rows without immutable snapshot ids are legacy history and should be treated differently from new schema-backed rows.
- Bovada scheduled start times are part of the safety layer: the orchestrator skips feature generation once a match is at or past the configured pre-start cutoff so delayed runs do not drift into post-start TA history.
