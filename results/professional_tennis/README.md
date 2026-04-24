# Professional Tennis Artifacts

This folder mixes two different things:

- mutable training outputs such as `metrics.csv`, `predictions.csv`, and plots
- stable release artifacts that should be referenced by the production registry

To keep that distinction clear, model-family folders should use this layout:

## Recommended Layout

- `releases/`
  Promoted versioned artifacts referenced by `production/models/model_registry.json`
- `candidates/`
  Trained but not yet promoted artifacts
- `experiments/`
  Local side-model tuning runs and walk-forward outputs
- `backups/`
  One-off safety backups made during migrations or retrains
- top-level files
  Mutable "current run" outputs from training scripts and compatibility files used by older scripts

## Important Rule

Do not treat top-level generic artifact filenames like `*_SURFACE_FIX.*` as the version ledger.

The version ledger is:

- `production/models/model_registry.json`
- `docs/production/MODEL_RELEASES.md`

Top-level generic filenames may mirror the current promoted release for compatibility, but promoted versions should also exist as stable files under `releases/`.
