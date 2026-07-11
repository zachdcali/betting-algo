"""Persist each match's full feature vector for the dashboard's detail view.

One row per (p1, p2, match_date) — the same join key the dashboard uses.
Write-once, with a single upgrade: an incomplete vector is replaced by the
first complete one (mirroring the prediction log's first-complete-wins rule),
then frozen. Keeps the file at one row per match, not one per run.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

PATH = os.path.join(os.path.dirname(__file__), "logs", "feature_vectors.csv")
COLS = ["p1", "p2", "match_date", "logged_at", "run_id", "features_complete", "features_json"]


def save_feature_vector(p1: str, p2: str, match_date, run_id: str,
                        features: dict, features_complete: bool) -> None:
    payload = {k: v for k, v in features.items() if not str(k).startswith("_")}
    payload["_defaulted_features"] = features.get("_defaulted_features", "") or ""
    for dbg in ("_build_ref", "_hist_tail_p1", "_hist_tail_p2", "_regime"):
        if features.get(dbg): payload[dbg] = features[dbg]
    row = {
        "p1": str(p1).strip(), "p2": str(p2).strip(),
        "match_date": str(match_date)[:10],
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id,
        "features_complete": bool(features_complete),
        "features_json": json.dumps(payload, default=str),
    }
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
    else:
        df = pd.DataFrame(columns=COLS)
    mask = (df["p1"] == row["p1"]) & (df["p2"] == row["p2"]) & (df["match_date"] == row["match_date"])
    if mask.any():
        already_complete = str(df.loc[mask, "features_complete"].iloc[0]) in ("True", "true", "1")
        if already_complete:
            # frozen at first complete — EXCEPT across a model-regime bump
            # (feature-semantics fix): a complete new build under a different
            # regime replaces the stale vector, mirroring the prediction log
            try:
                stored_regime = json.loads(df.loc[mask, "features_json"].iloc[0]).get("_regime", "")
            except Exception:
                stored_regime = ""
            new_regime = str(payload.get("_regime", ""))
            if not (bool(features_complete) and new_regime and stored_regime != new_regime):
                return
        # incomplete vectors TRACK the latest build (a round can resolve while a
        # height stays missing — the panel must show current truth, not the first
        # snapshot); they freeze only once a complete build arrives
        df = df[~mask]
    df = pd.concat([df, pd.DataFrame([{c: str(row[c]) for c in COLS}])], ignore_index=True)
    df.to_csv(PATH, index=False)
