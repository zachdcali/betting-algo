# Analysis Scripts

This folder is the research and offline-analysis side of the repo, not the live production pipeline.

Use it this way:

- `analysis_tools/`
  Reusable Python analysis utilities and validation scripts.
- `backtests/tools/`
  Backtest and chart-generation scripts that produce artifacts in `backtests/`.
- `backtests/`
  Generated plots and point-in-time backtest outputs.
- `betting_logs/`, `pure_bet_logs/`, `variance_analysis/`, `comprehensive_kelly_optimization/`
  Offline experiment inputs and outputs.
- `verified/`
  Verified comparison datasets and exports.

Conventions going forward:

- New executable analysis code should live in a `tools/` subfolder, not beside generated images.
- Generated charts and CSV outputs should stay out of source-code folders when practical.
- Nothing in this folder should be treated as the source of truth for live production behavior unless it is explicitly referenced from `docs/production/`.
