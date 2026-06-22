"""Ingest offline experiment metrics into ledger rows.

Each leaf experiment directory under ``results/professional_tennis/experiments``
holds a ``summary.json`` with this shape (verified 2026-06-21)::

    {"family": "xgboost", "feature_set": "performance_v1", "feature_mode": "one_hot",
     "n_features": 198, "split_label": "fixed_2022_val_2023plus_test",
     "config": {"slug": "xgb_depth5_recency_hl_12y", ...},
     "test_accuracy": ..., "test_auc": ..., "test_brier": ..., "test_ece": ...,
     "test_log_loss": ...}

Missing metrics become NaN; experiments are never silently dropped.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

_COLUMNS = ["source", "run_date", "experiment", "family", "feature_set",
            "feature_mode", "split", "n_features",
            "accuracy", "auc", "log_loss", "brier", "ece", "path"]


def _first(d: dict, *keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return np.nan


def discover_experiment_metrics(experiments_root: str) -> pd.DataFrame:
    rows = []
    if not os.path.isdir(experiments_root):
        return pd.DataFrame(columns=_COLUMNS)

    for dirpath, _dirnames, filenames in os.walk(experiments_root):
        if "summary.json" not in filenames:
            continue
        path = os.path.join(dirpath, "summary.json")
        try:
            with open(path) as fh:
                d = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(d, dict):
            continue

        config = d.get("config") or {}
        slug = config.get("slug") if isinstance(config, dict) else None
        experiment = slug or os.path.basename(dirpath)

        # run_date = the first path segment under experiments_root that looks like a date dir
        rel = os.path.relpath(dirpath, experiments_root)
        run_date = rel.split(os.sep)[0] if rel not in (".", "") else np.nan

        rows.append({
            "source": "offline",
            "run_date": run_date,
            "experiment": experiment,
            "family": _first(d, "family"),
            "feature_set": _first(d, "feature_set"),
            "feature_mode": _first(d, "feature_mode"),
            "split": _first(d, "split_label", "split"),
            "n_features": _first(d, "n_features"),
            "accuracy": _first(d, "test_accuracy", "accuracy"),
            "auc": _first(d, "test_auc", "auc"),
            "log_loss": _first(d, "test_log_loss", "log_loss", "logloss"),
            "brier": _first(d, "test_brier", "brier"),
            "ece": _first(d, "test_ece", "ece"),
            "path": path,
        })

    return pd.DataFrame(rows, columns=_COLUMNS)
