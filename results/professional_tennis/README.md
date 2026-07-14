# Professional Tennis Artifacts

This folder mixes two different things:

- mutable training outputs such as `metrics.csv`, `predictions.csv`, and plots
- stable release artifacts that should be referenced by the production registry

To keep that distinction clear, model-family folders should use this layout:

## Versioned Layout

- `releases/<family>/v<semver>/model.<format>`
  New promoted artifacts referenced by `production/models/model_registry.json`
- `candidates/<family>/v<semver>/model.<format>`
  New trained artifacts that have not earned promotion
- `archive/<family>/v<semver>/model.<format>`
  Stable prior promoted artifacts
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

Existing registered artifact paths remain unchanged so production compatibility
does not depend on a bulk rename. New paths use the model family and Semantic
Version directory; filenames describe only the serialization format, for
example:

```text
candidates/nn/v2.0.0/model.pth
candidates/nn/v2.0.0/scaler.pkl
candidates/xgboost/v2.0.0/model.json
```

`.pth`, `.pkl`, and `.json` say how an artifact is serialized. They are not its
version. The registry binds the directory to feature schema, feature semantics,
training dataset, protocol, calibration, and content hashes. Do not introduce
new names such as `model_v2_final_fixed.pth`.
