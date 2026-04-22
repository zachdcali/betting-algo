# Backtests

This folder mixes two different things:

- generated outputs such as charts and summary artifacts
- scripts that produce those outputs

To keep that separation clearer:

- source scripts now belong in `backtests/tools/`
- generated charts can remain in `backtests/`

If you add a new backtest utility, prefer:

```text
analysis_scripts/backtests/tools/<script>.py
```

and have it write results into:

```text
analysis_scripts/backtests/
```
