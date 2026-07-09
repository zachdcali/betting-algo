"""Per-build feature validation against the model's own training distribution.

Every feature vector is checked, on every build, against the exact per-feature
mean/std the NN's scaler learned at training time (models/feature_training_stats.json,
extracted from scaler_SURFACE_FIX.pkl in EXACT_141_FEATURES order):

- presence: all 141 features must exist (catches renames/drops in refactors)
- one-hot sanity: Surface_/Round_/Level_ flags must be exactly 0 or 1
- distribution: |z| > Z_LOUD against training mean/std flags a likely unit,
  scale, or source bug (a height in meters, rank points x100, a sign flip)

Flags never block a bet by themselves — a legit new extreme (a record streak)
is in-distribution tail, not corruption — but every flag is printed loudly and
appended to logs/audit/feature_range_flags.csv so drift is visible the run it
starts, not weeks later in a ledger anomaly.
"""
from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone

BASE = os.path.dirname(__file__)
STATS_PATH = os.path.join(BASE, "models", "feature_training_stats.json")
FLAGS_PATH = os.path.join(BASE, "logs", "audit", "feature_range_flags.csv")
Z_LOUD = 8.0
_ONE_HOT_PREFIXES = ("Surface_", "Round_", "Level_")

_stats: dict | None = None


def _load() -> dict:
    global _stats
    if _stats is None:
        with open(STATS_PATH) as fh:
            _stats = json.load(fh)
    return _stats


def validate_features(features: dict, p1: str, p2: str, run_id: str) -> list[str]:
    """Return human-readable flags; log and print any found."""
    stats = _load()
    flags: list[str] = []
    for name, st in stats.items():
        raw = features.get(name)
        try:
            v = float(raw)
        except (TypeError, ValueError):
            flags.append(f"{name}=MISSING")
            continue
        if name.startswith(_ONE_HOT_PREFIXES) and v not in (0.0, 1.0):
            flags.append(f"{name}={v!r} (one-hot violated)")
            continue
        std = st.get("std") or 0.0
        if std > 0:
            z = abs(v - st["mean"]) / std
            if z > Z_LOUD:
                flags.append(f"{name}={v:.4g} (z={z:.1f})")
    if flags:
        print(f"      🔬 OOD features for {p1} vs {p2}: {', '.join(flags[:4])}"
              + (f" (+{len(flags)-4} more)" if len(flags) > 4 else ""))
        os.makedirs(os.path.dirname(FLAGS_PATH), exist_ok=True)
        new = not os.path.exists(FLAGS_PATH)
        with open(FLAGS_PATH, "a", newline="") as fh:
            w = csv.writer(fh)
            if new:
                w.writerow(["logged_at", "run_id", "p1", "p2", "flag"])
            now = datetime.now(timezone.utc).isoformat(timespec="seconds")
            for f in flags:
                w.writerow([now, run_id, p1, p2, f])
    return flags
