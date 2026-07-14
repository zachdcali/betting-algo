"""Persist immutable feature snapshots for dashboard and audit drill-down.

New rows are keyed by ``feature_snapshot_id`` so the dashboard can display the
exact vector used by a prediction. Legacy rows without an immutable id retain
the old one-row-per-match upgrade behaviour.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd

from feature_contract import (
    feature_fingerprint as contract_feature_fingerprint,
    normalize_feature_vector,
)

PATH = os.path.join(os.path.dirname(__file__), "logs", "feature_vectors.csv")
COLS = ["p1", "p2", "match_date", "logged_at", "run_id", "match_uid",
        "feature_snapshot_id", "build_status", "features_complete", "p1_hand", "p2_hand",
        "feature_schema_sha256", "feature_vector_sha256", "feature_count",
        "features_json"]


def feature_validation_issues(features: dict) -> list[str]:
    """Hard schema/domain checks required before promoted-model inference."""
    from models.inference import EXACT_141_FEATURES
    _, issues = normalize_feature_vector(features, EXACT_141_FEATURES)
    return list(issues)


def feature_fingerprint(features: dict) -> tuple[str, str, int]:
    """Return deterministic schema/content fingerprints for the live 141-vector.

    A lineage ID says which run produced a row; this hash says what the model
    actually received. Missing or non-finite values intentionally produce an
    empty content hash so structural failures cannot be presented as exact.
    """
    from models.inference import EXACT_141_FEATURES
    return contract_feature_fingerprint(features, EXACT_141_FEATURES)


def save_feature_vector(p1: str, p2: str, match_date, run_id: str,
                        features: dict, features_complete: bool,
                        match_uid: str = "", feature_snapshot_id: str = "",
                        build_status: str = "ok") -> None:
    payload = {k: v for k, v in features.items() if not str(k).startswith("_")}
    payload["_defaulted_features"] = features.get("_defaulted_features", "") or ""
    for dbg in ("_build_ref", "_hist_tail_p1", "_hist_tail_p2", "_regime"):
        if features.get(dbg): payload[dbg] = features[dbg]
    schema_hash, vector_hash, feature_count = feature_fingerprint(features)
    # surface handedness as first-class columns (dashboard slate badges unknown
    # hand — a missingness proxy the models lean on; see FEATURE_AUDIT.md)
    def _hand(pref):
        if float(features.get(f"{pref}_Hand_U", 0) or 0) == 1: return "U"
        if float(features.get(f"{pref}_Hand_L", 0) or 0) == 1: return "L"
        if float(features.get(f"{pref}_Hand_R", 0) or 0) == 1: return "R"
        return ""
    row = {
        "p1": str(p1).strip(), "p2": str(p2).strip(),
        "match_date": str(match_date)[:10],
        "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "run_id": run_id, "match_uid": match_uid or "",
        "feature_snapshot_id": feature_snapshot_id or "",
        "build_status": build_status or "unknown",
        # A guard/error row can legitimately have no default labels because no
        # feature vector was built. Never let that absence masquerade as a
        # complete, decision-grade snapshot.
        "features_complete": bool(
            features_complete and build_status == "ok" and vector_hash
        ),
        "p1_hand": _hand("P1"), "p2_hand": _hand("P2"),
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": vector_hash,
        "feature_count": feature_count,
        "features_json": json.dumps(payload, default=str),
    }
    if os.path.exists(PATH):
        df = pd.read_csv(PATH, dtype=str)
    else:
        df = pd.DataFrame(columns=COLS)
    for column in COLS:
        if column not in df.columns:
            df[column] = ""

    if feature_snapshot_id:
        # Immutable lineage: a retry of the same run is idempotent, while a
        # later run receives a new id and remains independently inspectable.
        if (df["feature_snapshot_id"].fillna("") == feature_snapshot_id).any():
            return
        df = pd.concat(
            [df[COLS], pd.DataFrame([{c: str(row[c]) for c in COLS}])],
            ignore_index=True,
        )
        df.to_csv(PATH, index=False)
        return

    # Compatibility path for legacy callers without immutable lineage ids.
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
    df = pd.concat([df[COLS], pd.DataFrame([{c: str(row[c]) for c in COLS}])], ignore_index=True)
    df.to_csv(PATH, index=False)
