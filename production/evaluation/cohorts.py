"""Load logs, derive authoritative ground truth, assemble a long scored frame.

This is the only module that knows the on-disk storage format. A future SQLite
migration swaps the ``load_*`` helpers and leaves the rest of the package intact.
"""
from __future__ import annotations

import os
from glob import glob
import json

import numpy as np
import pandas as pd

# Model name -> the prediction_log column holding its P(player1 wins).
MODEL_PROB_COLS = {
    "nn": "model_p1_prob",
    "xgb": "xgb_p1_prob",
    "rf": "rf_p1_prob",
    "market": "market_p1_prob",
}
SHADOW_FAMILIES = ["xgboost", "catboost", "lightgbm", "nn"]
SCORED_COLUMNS = [
    "match_uid", "model", "family", "p1_prob",
    "p1_odds_decimal", "p2_odds_decimal", "y1",
    "is_gold", "is_complete", "prediction_time",
]


def _coerce_probability(value) -> float | None:
    """Return a finite probability in [0, 1], otherwise no evidence."""
    try:
        probability = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        return None
    return probability


def _coerce_decimal_odds(value) -> float:
    """Return valid finite decimal odds, otherwise NaN so ROI skips the row."""
    try:
        odds = float(value)
    except (TypeError, ValueError, OverflowError):
        return float("nan")
    return odds if np.isfinite(odds) and odds > 1.0 else float("nan")


def verify_feature_frame(frame: pd.DataFrame) -> tuple[dict[str, str], set[str]]:
    """Validate exact-vector evidence and return id->content hash plus bad IDs."""
    from feature_vector_log import feature_fingerprint
    from models.inference import EXACT_141_FEATURES

    hashes: dict[str, set[str]] = {}
    invalid: set[str] = set()
    if frame.empty or "feature_snapshot_id" not in frame.columns:
        return {}, invalid

    aggregate_format = "features_json" in frame.columns
    for _, row in frame.iterrows():
        snapshot_id = str(row.get("feature_snapshot_id", "") or "").strip()
        if not snapshot_id:
            continue
        try:
            if aggregate_format:
                status = str(row.get("build_status", "") or "").strip().lower()
                complete = str(row.get("features_complete", "")).strip().lower()
                if status != "ok" or complete not in {"true", "1", "1.0", "t", "yes"}:
                    invalid.add(snapshot_id)
                    continue
                payload = json.loads(str(row.get("features_json", "") or ""))
                if not isinstance(payload, dict):
                    raise ValueError("features_json is not an object")
            else:
                status = str(row.get("status", "") or "").strip().lower()
                if status != "ok":
                    invalid.add(snapshot_id)
                    continue
                if any(name not in frame.columns for name in EXACT_141_FEATURES):
                    invalid.add(snapshot_id)
                    continue
                payload = {name: row.get(name) for name in EXACT_141_FEATURES}

            schema_hash, vector_hash, feature_count = feature_fingerprint(payload)
            if not vector_hash or feature_count != len(EXACT_141_FEATURES):
                invalid.add(snapshot_id)
                continue
            stored_schema = str(row.get("feature_schema_sha256", "") or "").strip()
            stored_vector = str(row.get("feature_vector_sha256", "") or "").strip()
            if ((stored_schema and stored_schema != schema_hash)
                    or (stored_vector and stored_vector != vector_hash)):
                invalid.add(snapshot_id)
                continue
            hashes.setdefault(snapshot_id, set()).add(vector_hash)
        except (TypeError, ValueError, json.JSONDecodeError):
            invalid.add(snapshot_id)

    for snapshot_id, values in hashes.items():
        if len(values) != 1:
            invalid.add(snapshot_id)
    return {
        snapshot_id: next(iter(values))
        for snapshot_id, values in hashes.items()
        if snapshot_id not in invalid and len(values) == 1
    }, invalid


def load_prediction_log(prod_dir: str) -> pd.DataFrame:
    from feature_lineage import (
        expected_feature_hash_matches,
        load_production_feature_lineage,
        read_feature_csv,
    )
    from models.inference import EXACT_141_FEATURES

    frame = pd.read_csv(os.path.join(prod_dir, "prediction_log.csv"), low_memory=False)
    scanned_paths: list[str] = []
    legacy_paths: list[str] = []
    paths = [
        *glob(os.path.join(prod_dir, "logs", "features_*.csv")),
        os.path.join(prod_dir, "logs", "feature_vectors.csv"),
    ]
    for path in paths:
        if not os.path.exists(path):
            continue
        try:
            header = read_feature_csv(path, nrows=0)
        except Exception as exc:
            raise RuntimeError(f"feature snapshot verification could not read {path}: {exc}") from exc
        if "feature_snapshot_id" not in header.columns:
            legacy_paths.append(path)
            continue
        scanned_paths.append(path)
    try:
        lineage = load_production_feature_lineage(prod_dir, EXACT_141_FEATURES)
    except Exception as exc:
        raise RuntimeError(
            f"feature snapshot authority reconciliation failed: {exc}"
        ) from exc
    invalid_ids = set(lineage.invalid_ids)
    evidence = {
        snapshot_id: occurrence.verified_vector_sha256
        for snapshot_id, occurrence in lineage.canonical_by_id.items()
        if snapshot_id not in invalid_ids and occurrence.verified_vector_sha256
    }
    referential_hashes = {
        snapshot_id: lineage.referential_vector_hashes(snapshot_id)
        for snapshot_id in evidence
    }
    ids = frame.get("feature_snapshot_id", pd.Series("", index=frame.index)).fillna("").astype(str)
    # Referential verification is fail-closed. If no compatible lineage file
    # exists, no row can claim GOLD merely because it carries an ID-shaped
    # string. Publication also fails loudly on partially unreadable inputs.
    expected_hash = frame.get(
        "feature_vector_sha256", pd.Series("", index=frame.index)
    ).fillna("").astype(str).str.strip()
    matched_hash = pd.Series(
        [evidence.get(snapshot_id, "") for snapshot_id in ids], index=frame.index
    )
    expected_matches = pd.Series([
        expected_feature_hash_matches(
            expected, referential_hashes.get(snapshot_id, frozenset())
        )
        for snapshot_id, expected in zip(ids, expected_hash)
    ], index=frame.index)
    frame["feature_snapshot_verified"] = (
        ids.ne("") & matched_hash.ne("")
        & expected_matches
    )
    frame.attrs["feature_snapshot_verification"] = {
        "scanned_paths": scanned_paths,
        "legacy_paths_without_ids": legacy_paths,
        "verified_id_count": len(evidence),
        "invalid_id_count": len(invalid_ids),
    }
    return frame


def load_shadow_log(prod_dir: str) -> pd.DataFrame | None:
    path = os.path.join(prod_dir, "logs", "performance_v1_shadow_predictions.csv")
    return pd.read_csv(path, low_memory=False) if os.path.exists(path) else None


def build_ground_truth(pred_log: pd.DataFrame) -> pd.Series:
    """Authoritative match_uid -> y1 (1 if player1 won). Conflict-free, deduped."""
    winner = pd.to_numeric(pred_log["actual_winner"], errors="coerce")
    # Only explicit player 1/2 outcomes are ground truth. Historical void
    # sentinels such as -1 must never become an implicit player-two win.
    s = pred_log[winner.isin([1, 2])].copy()
    s["y1"] = (pd.to_numeric(s["actual_winner"], errors="coerce") == 1).astype(int)
    nunique = s.groupby("match_uid")["y1"].nunique()
    consistent = nunique[nunique == 1].index
    # keep="last" = the most recently logged row, matching the operational
    # "latest prediction wins" semantics and the DB's INSERT OR REPLACE.
    s = s[s["match_uid"].isin(consistent)].drop_duplicates("match_uid", keep="last")
    return s.set_index("match_uid")["y1"]


def _coerce_bool(series: pd.Series) -> pd.Series:
    """Robustly coerce a truthy column to bool regardless of source representation.

    CSVs give Python bools; a SQLite round-trip can yield "1"/"0" or "True"/"False"
    strings or 1/0 ints. Normalize them all so cohort tiers behave identically
    whether the data came from a CSV or the DB.
    """
    truthy = {"true", "1", "1.0", "t", "yes"}
    return series.map(lambda v: str(v).strip().lower() in truthy).astype(bool)


def _tier_flags(pred_log: pd.DataFrame) -> pd.DataFrame:
    f = pred_log.copy()
    if "features_complete" in f.columns:
        f["is_complete"] = _coerce_bool(f["features_complete"])
    else:
        f["is_complete"] = False
    lq = f["logging_quality"] if "logging_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    rq = f["rescore_quality"] if "rescore_quality" in f.columns else pd.Series(index=f.index, dtype=object)
    verified = (
        _coerce_bool(f["feature_snapshot_verified"])
        if "feature_snapshot_verified" in f.columns
        else pd.Series(False, index=f.index, dtype=bool)
    )
    f["is_gold"] = (
        f["is_complete"] & verified
        & (lq == "snapshot_v2") & (rq == "exact_feature_snapshot")
    )
    return f[["match_uid", "is_complete", "is_gold"]].drop_duplicates("match_uid")


def build_scored_frame(pred_log: pd.DataFrame, shadow_log: pd.DataFrame | None) -> pd.DataFrame:
    """Long format: one row per (match_uid, model), settled rows with a non-null prob."""
    gt = build_ground_truth(pred_log)
    tiers = _tier_flags(pred_log).set_index("match_uid")
    settled = (
        pred_log[pred_log["match_uid"].isin(gt.index)]
        .drop_duplicates("match_uid", keep="last")
        .set_index("match_uid")
    )
    operational_feature_ids = (
        settled.get("feature_snapshot_id", pd.Series("", index=settled.index))
        .fillna("").astype(str)
    )
    rows = []
    for model, col in MODEL_PROB_COLS.items():
        if col not in settled.columns:
            continue
        for uid, r in settled.iterrows():
            probability = _coerce_probability(r[col])
            if probability is None:
                continue
            rows.append({
                "match_uid": uid, "model": model, "family": model,
                "p1_prob": probability,
                "p1_odds_decimal": _coerce_decimal_odds(r.get("p1_odds_decimal")),
                "p2_odds_decimal": _coerce_decimal_odds(r.get("p2_odds_decimal")),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]),
                "is_complete": bool(tiers.loc[uid, "is_complete"]),
                "prediction_time": r.get("logged_at", r.get("odds_scraped_at")),
            })

    if shadow_log is not None and "model_family" in shadow_log.columns:
        sh = shadow_log[shadow_log["match_uid"].isin(gt.index)].copy()
        version = sh.get("model_version", sh["model_family"]).fillna("").astype(str).str.strip()
        sh["_model_label"] = version.where(version.ne(""), sh["model_family"].astype(str))
        if "feature_snapshot_id" in sh.columns:
            opening = operational_feature_ids.rename("_operational_feature_snapshot_id")
            sh = sh.merge(opening, left_on="match_uid", right_index=True, how="left")
            opening_id = sh["_operational_feature_snapshot_id"].fillna("").astype(str)
            shadow_id = sh["feature_snapshot_id"].fillna("").astype(str)
            sh = sh[opening_id.eq("") | shadow_id.eq(opening_id)].copy()
        if "logged_at" in sh.columns:
            sh["_logged_sort"] = pd.to_datetime(
                sh["logged_at"], errors="coerce", utc=True, format="mixed"
            )
            sh = sh.sort_values(["_logged_sort"], kind="stable", na_position="last")
        # One deterministic opening observation per match/model variant. Hourly
        # repeats are correlated evidence, not independent samples.
        sh = sh.drop_duplicates(["match_uid", "_model_label"], keep="first")
        for _, r in sh.iterrows():
            probability = _coerce_probability(r.get("shadow_p1_prob"))
            if probability is None:
                continue
            uid = r["match_uid"]
            in_tiers = uid in tiers.index
            # Key by model_version so multiple variants of the same family (e.g.
            # several XGB recency/depth configs) stay distinct in the ledger.
            # Fall back to family when version is absent (older rows / fixtures).
            label = str(r["_model_label"])
            rows.append({
                "match_uid": uid,
                "model": f"shadow_{label}",
                "family": r["model_family"],
                "p1_prob": probability,
                "p1_odds_decimal": _coerce_decimal_odds(r.get("p1_odds_decimal")),
                "p2_odds_decimal": _coerce_decimal_odds(r.get("p2_odds_decimal")),
                "y1": int(gt.loc[uid]),
                "is_gold": bool(tiers.loc[uid, "is_gold"]) if in_tiers else False,
                "is_complete": bool(tiers.loc[uid, "is_complete"]) if in_tiers else False,
                "prediction_time": settled.loc[uid].get(
                    "logged_at", settled.loc[uid].get("odds_scraped_at")
                ),
            })

    frame = pd.DataFrame(rows, columns=SCORED_COLUMNS)
    # Preserve a typed empty result so a fully invalid hydrated cohort produces
    # an empty ledger instead of failing during boolean tier selection.
    frame["is_gold"] = frame["is_gold"].astype(bool)
    frame["is_complete"] = frame["is_complete"].astype(bool)
    return frame


def intersection_uids(scored: pd.DataFrame, models: list[str], tier_col: str) -> set:
    """match_uids that appear for *every* listed model within the given tier."""
    sub = scored[scored[tier_col] & scored["model"].isin(models)]
    counts = sub.groupby("match_uid")["model"].nunique()
    needed = len(set(models))
    return set(counts[counts == needed].index)
