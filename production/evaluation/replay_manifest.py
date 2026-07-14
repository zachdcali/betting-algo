"""Build a read-only manifest of historically replayable prediction evidence.

The manifest never runs a model and never mutates prediction or feature logs.  It
answers a narrower question: which exact, point-in-time feature vectors can be
fed to an artifact that expects the same ordered 141-feature schema?

The operational ``prediction_log.csv`` row is the selection authority.  Its
opening ``feature_snapshot_id`` is selected even when a later snapshot would
produce a more convenient cohort.  Alternative snapshots are retained as
provenance only; they are never used to upgrade a match's replay tier.

CLI examples::

    cd production
    ../tennis_env/bin/python -m evaluation.replay_manifest --prod-dir .

    ../tennis_env/bin/python -m evaluation.replay_manifest \
        --prod-dir . \
        --out-dir ../results/professional_tennis/replay_manifests

Without ``--out-dir`` the command only prints a summary.  With ``--out-dir`` it
creates a new timestamp-versioned child directory containing a CSV and a JSON
manifest with input/output hashes.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from pathlib import Path
import unicodedata
from typing import Any, Iterable, Sequence

import pandas as pd

from evaluation import cohorts
from feature_vector_log import feature_fingerprint
from feature_lineage import load_production_feature_lineage, read_feature_csv
from models.inference import EXACT_141_FEATURES
from versioning import (
    DATASET_MANIFEST_VERSION,
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    ordered_schema_sha256,
)


MANIFEST_SCHEMA_VERSION = DATASET_MANIFEST_VERSION
ARTIFACT_SCHEMA_ID = FEATURE_SCHEMA_ID
ARTIFACT_SCHEMA_SHA256 = FEATURE_SCHEMA_SHA256
if ordered_schema_sha256(tuple(EXACT_141_FEATURES)) != ARTIFACT_SCHEMA_SHA256:
    raise RuntimeError("base_141 feature order does not match its versioned schema hash")
TIERS = ("GOLD_REPLAY", "EXACT_INCOMPLETE", "LEGACY_MATCHED", "NO_VECTOR")
OUTPUT_CSV = "historical_replay.csv"
OUTPUT_JSON = "manifest.json"


OUTPUT_COLUMNS = [
    "match_uid",
    "replay_tier",
    "match_date",
    "tournament",
    "p1",
    "p2",
    "selected_snapshot_kind",
    "snapshot_selection_rule",
    "feature_snapshot_id",
    "snapshot_verified",
    "features_complete",
    "feature_schema_sha256",
    "feature_vector_sha256",
    "feature_count",
    "artifact_schema_id",
    "artifact_schema_sha256",
    "artifact_schema_compatible",
    "prediction_observed_at",
    "feature_observed_at",
    "match_start_at_utc",
    "is_opening_snapshot",
    "is_prestart_snapshot",
    "feature_source_file",
    "feature_source_row",
    "prediction_source_file",
    "prediction_source_row",
    "prediction_source_rows",
    "alternative_snapshot_count",
    "alternative_snapshot_ids",
    "alternative_snapshot_provenance",
    "actual_winner",
    "outcome_status",
    "outcome_source_file",
    "outcome_source_rows",
    "p1_odds_decimal",
    "p2_odds_decimal",
    "market_p1_prob",
    "market_p2_prob",
    "odds_evidence_status",
    "odds_source_file",
    "odds_source_row",
    "reason_codes",
]


@dataclass
class ReplayManifest:
    """In-memory replay rows plus the immutable inputs used to derive them."""

    frame: pd.DataFrame
    source_files: list[dict[str, Any]]


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _truthy(value: Any) -> bool:
    return _text(value).lower() in {"true", "1", "1.0", "t", "yes"}


def _normal_name(value: Any) -> str:
    normalized = unicodedata.normalize("NFKD", _text(value).lower())
    return "".join(character for character in normalized if character.isalnum())


def _date_text(value: Any) -> str:
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def _numeric(value: Any) -> float | None:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number):
        return None
    result = float(number)
    return result if math.isfinite(result) else None


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_metadata(path: Path, prod_dir: Path, rows: int) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(prod_dir.resolve()).as_posix(),
        "sha256": _sha256_file(path),
        "rows": int(rows),
    }


def _read_csv(path: Path, prod_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        is_feature_file = (
            path.parent.name == "logs"
            and (path.name == "feature_vectors.csv" or path.name.startswith("features_"))
        )
        frame = (
            read_feature_csv(path)
            if is_feature_file
            else pd.read_csv(path, low_memory=False, keep_default_na=False)
        )
    except Exception as exc:  # fail closed on partially readable lineage
        relative = path.resolve().relative_to(prod_dir.resolve()).as_posix()
        raise RuntimeError(f"historical replay could not read {relative}: {exc}") from exc
    return frame, _source_metadata(path, prod_dir, len(frame))


def _parse_utc(value: Any, *, assume_timezone: str = "UTC") -> pd.Timestamp | None:
    raw = _text(value)
    if not raw:
        return None
    timestamp = pd.to_datetime(raw, errors="coerce")
    if pd.isna(timestamp):
        return None
    try:
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(
                assume_timezone, ambiguous="NaT", nonexistent="NaT"
            )
        timestamp = timestamp.tz_convert("UTC")
    except (TypeError, ValueError):
        return None
    return None if pd.isna(timestamp) else timestamp


def _iso_utc(timestamp: pd.Timestamp | None) -> str:
    if timestamp is None or pd.isna(timestamp):
        return ""
    return timestamp.isoformat().replace("+00:00", "Z")


def _prediction_observed_at(row: pd.Series) -> pd.Timestamp | None:
    # Odds collection is the point-in-time market observation.  The prediction
    # write timestamp is a conservative fallback for older rows.
    return _parse_utc(row.get("odds_scraped_at")) or _parse_utc(row.get("logged_at"))


def _match_start_at_utc(row: pd.Series) -> pd.Timestamp | None:
    for column in ("match_start_at_utc", "latest_match_start_at_utc"):
        timestamp = _parse_utc(row.get(column))
        if timestamp is not None:
            return timestamp
    # Historical Bovada display values are Eastern time, per the operational
    # logging contract.  Never infer a match start from match_date alone.
    for column in ("match_start_time", "latest_match_start_time"):
        timestamp = _parse_utc(row.get(column), assume_timezone="America/New_York")
        if timestamp is not None:
            return timestamp
    return None


def _feature_payload(row: pd.Series, aggregate_format: bool) -> dict[str, Any]:
    if aggregate_format:
        try:
            payload = json.loads(_text(row.get("features_json")))
        except json.JSONDecodeError:
            return {}
        return payload if isinstance(payload, dict) else {}
    return {name: row.get(name) for name in EXACT_141_FEATURES}


def _feature_complete(row: pd.Series, aggregate_format: bool) -> bool:
    if "features_complete" in row.index and _text(row.get("features_complete")):
        return _truthy(row.get("features_complete"))
    if "_has_defaulted_features" in row.index:
        return not _truthy(row.get("_has_defaulted_features"))
    if "meta_defaulted_features" in row.index:
        return not bool(_text(row.get("meta_defaulted_features")))
    # Legacy tabular logs predate an explicit completeness flag.  Structural
    # validity is not equivalent to completeness, so remain conservative.
    return False if not aggregate_format else _truthy(row.get("features_complete"))


def _first_present(row: pd.Series, columns: Iterable[str]) -> str:
    for column in columns:
        value = _text(row.get(column))
        if value:
            return value
    return ""


def _feature_identity_compatible(
    candidate: dict[str, Any] | None, prediction: pd.Series
) -> bool:
    """Require one exact match UID and the same player orientation."""
    if candidate is None:
        return False
    candidate_p1 = _normal_name(candidate.get("p1"))
    candidate_p2 = _normal_name(candidate.get("p2"))
    prediction_p1 = _normal_name(prediction.get("p1"))
    prediction_p2 = _normal_name(prediction.get("p2"))
    candidate_match_uid = _text(candidate.get("match_uid"))
    prediction_match_uid = _text(prediction.get("match_uid"))
    return bool(
        candidate_match_uid
        and prediction_match_uid
        and candidate_match_uid == prediction_match_uid
        and candidate_p1
        and candidate_p2
        and prediction_p1
        and prediction_p2
        and candidate_p1 == prediction_p1
        and candidate_p2 == prediction_p2
    )


def _feature_candidate(
    row: pd.Series,
    *,
    source_file: str,
    source_row: int,
    aggregate_format: bool,
    verified_ids: dict[str, str],
    invalid_ids: set[str],
    synthetic_id: str,
) -> dict[str, Any]:
    payload = _feature_payload(row, aggregate_format)
    schema_hash, vector_hash, feature_count = feature_fingerprint(payload)
    snapshot_id = _text(row.get("feature_snapshot_id"))
    verification_id = snapshot_id or synthetic_id
    structurally_verified = (
        bool(vector_hash)
        and verification_id in verified_ids
        and verification_id not in invalid_ids
        and verified_ids.get(verification_id) == vector_hash
    )
    timestamp = _first_present(row, ("timestamp", "logged_at", "run_started_at"))
    return {
        "feature_snapshot_id": snapshot_id,
        "match_uid": _text(row.get("match_uid")),
        "p1": _first_present(row, ("player1_raw", "p1")),
        "p2": _first_present(row, ("player2_raw", "p2")),
        "match_date": _date_text(
            _first_present(row, ("meta_match_date", "match_date"))
        ),
        "feature_observed_at": _iso_utc(_parse_utc(timestamp)),
        "source_file": source_file,
        "source_row": int(source_row),
        "feature_schema_sha256": schema_hash,
        "feature_vector_sha256": vector_hash,
        "feature_count": int(feature_count),
        "features_complete": _feature_complete(row, aggregate_format),
        "structurally_verified": bool(structurally_verified),
        "synthetic_verification_id": verification_id if not snapshot_id else "",
        "is_authoritative": False,
    }


def _prepare_verification_frame(frame: pd.DataFrame, source_file: str) -> pd.DataFrame:
    """Give legacy rows deterministic temporary IDs, then use cohort validation."""
    prepared = frame.copy()
    ids = prepared.get(
        "feature_snapshot_id", pd.Series("", index=prepared.index, dtype=object)
    ).map(_text)
    prepared["feature_snapshot_id"] = [
        snapshot_id or f"legacy::{source_file}::{index + 2}"
        for index, snapshot_id in zip(prepared.index, ids)
    ]
    if "features_json" in prepared.columns:
        if "build_status" not in prepared.columns:
            # Aggregate legacy rows predate build_status.  This only allows the
            # shared structural validator to inspect their payload; it does not
            # grant exact-lineage or GOLD status.
            prepared["build_status"] = "ok"
        if "features_complete" not in prepared.columns:
            prepared["features_complete"] = False
    elif "status" not in prepared.columns:
        # Old tabular files contain feature columns but no explicit status.
        prepared["status"] = "ok"
    return prepared


def _load_feature_evidence(
    prod_dir: Path,
) -> tuple[
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
    dict[tuple[str, str, str], list[dict[str, Any]]],
    list[dict[str, Any]],
]:
    paths = sorted((prod_dir / "logs").glob("features_*.csv"))
    aggregate = prod_dir / "logs" / "feature_vectors.csv"
    if aggregate.exists():
        paths.append(aggregate)

    all_candidates: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    lineage = load_production_feature_lineage(prod_dir, EXACT_141_FEATURES)
    for path in paths:
        frame, source = _read_csv(path, prod_dir)
        source_files.append(source)
        source_file = source["path"]
        prepared = _prepare_verification_frame(frame, source_file)
        try:
            verified_ids, invalid_ids = cohorts.verify_feature_frame(prepared)
        except Exception as exc:
            raise RuntimeError(
                f"historical replay feature verification failed for {source_file}: {exc}"
            ) from exc
        aggregate_format = "features_json" in frame.columns
        for position, (_, row) in enumerate(frame.iterrows(), start=2):
            synthetic_id = f"legacy::{source_file}::{position}"
            candidate = _feature_candidate(
                row,
                source_file=source_file,
                source_row=position,
                aggregate_format=aggregate_format,
                verified_ids=verified_ids,
                invalid_ids=invalid_ids,
                synthetic_id=synthetic_id,
            )
            snapshot_id = candidate["feature_snapshot_id"]
            if snapshot_id:
                authority = lineage.canonical_by_id.get(snapshot_id)
                is_authoritative = bool(
                    authority is not None
                    and authority.location == (source_file, position)
                )
                candidate["is_authoritative"] = is_authoritative
                candidate["structurally_verified"] = bool(
                    is_authoritative
                    and authority is not None
                    and authority.structurally_verified
                )
                if is_authoritative and authority is not None:
                    candidate["feature_schema_sha256"] = authority.schema_sha256
                    candidate["feature_vector_sha256"] = (
                        authority.verified_vector_sha256
                    )
                    candidate["feature_count"] = authority.feature_count
            all_candidates.append(candidate)

    by_id: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_match_uid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    legacy_by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in all_candidates:
        snapshot_id = candidate["feature_snapshot_id"]
        if snapshot_id:
            by_id[snapshot_id].append(candidate)
            if candidate["match_uid"]:
                by_match_uid[candidate["match_uid"]].append(candidate)
        else:
            key = (
                _normal_name(candidate["p1"]),
                _normal_name(candidate["p2"]),
                candidate["match_date"],
            )
            if all(key):
                legacy_by_key[key].append(candidate)

    sort_key = lambda item: (
        not item.get("is_authoritative", False),
        item["source_file"],
        item["source_row"],
    )
    for mapping in (by_id, by_match_uid, legacy_by_key):
        for candidates in mapping.values():
            candidates.sort(key=sort_key)
    return dict(by_id), dict(by_match_uid), dict(legacy_by_key), source_files


def _primary_prediction(group: pd.DataFrame) -> pd.Series:
    ordered = group.copy()
    ordered["_observed_sort"] = ordered.apply(
        lambda row: _prediction_observed_at(row), axis=1
    )
    ordered["_missing_observed"] = ordered["_observed_sort"].isna()
    ordered = ordered.sort_values(
        ["_missing_observed", "_observed_sort", "_source_row"],
        kind="stable",
        na_position="last",
    )
    return ordered.iloc[0]


def _outcome_evidence(group: pd.DataFrame) -> tuple[str, int | str, list[int]]:
    winners = pd.to_numeric(group.get("actual_winner"), errors="coerce")
    terminal = winners[winners.isin([1, 2])]
    rows = group.loc[terminal.index, "_source_row"].astype(int).tolist()
    values = sorted({int(value) for value in terminal.tolist()})
    if len(values) == 1:
        return "authoritative_conflict_free", values[0], rows
    if len(values) > 1:
        return "conflicting_terminal_outcomes", "", rows
    return "missing_terminal_outcome", "", []


def _odds_evidence(row: pd.Series) -> tuple[str, float | str, float | str]:
    p1_odds = _numeric(row.get("p1_odds_decimal"))
    p2_odds = _numeric(row.get("p2_odds_decimal"))
    genuine = p1_odds is not None and p2_odds is not None and p1_odds > 1 and p2_odds > 1
    if genuine:
        return "genuine_two_sided_decimal", p1_odds, p2_odds
    p1_market = _numeric(row.get("market_p1_prob"))
    p2_market = _numeric(row.get("market_p2_prob"))
    if p1_market == 0.5 and p2_market == 0.5:
        return "rejected_0_5_without_decimal_odds", p1_odds or "", p2_odds or ""
    return "missing_two_sided_decimal", p1_odds or "", p2_odds or ""


def _candidate_provenance(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "feature_snapshot_id": candidate["feature_snapshot_id"],
        "feature_vector_sha256": candidate["feature_vector_sha256"],
        "source_file": candidate["source_file"],
        "source_row": candidate["source_row"],
        "structurally_verified": candidate["structurally_verified"],
    }


def _alternative_evidence(
    *,
    match_uid: str,
    primary_snapshot_id: str,
    primary_candidate: dict[str, Any] | None,
    exact_by_match_uid: dict[str, list[dict[str, Any]]],
    legacy_candidates: list[dict[str, Any]],
    snapshot_frame: pd.DataFrame | None,
    primary: pd.Series,
) -> tuple[list[str], list[dict[str, Any]]]:
    candidates = list(exact_by_match_uid.get(match_uid, []))
    candidates.extend(legacy_candidates)
    seen = {
        (candidate["source_file"], int(candidate["source_row"]))
        for candidate in candidates
    }
    if primary_candidate is not None:
        selected_key = (
            primary_candidate["source_file"],
            int(primary_candidate["source_row"]),
        )
    else:
        selected_key = None

    provenance = [
        _candidate_provenance(candidate)
        for candidate in candidates
        if (candidate["source_file"], int(candidate["source_row"])) != selected_key
    ]

    ids = {
        candidate["feature_snapshot_id"]
        for candidate in candidates
        if candidate["feature_snapshot_id"]
        and candidate["feature_snapshot_id"] != primary_snapshot_id
    }
    latest_id = _text(primary.get("latest_feature_snapshot_id"))
    if latest_id and latest_id != primary_snapshot_id:
        ids.add(latest_id)
    if snapshot_frame is not None and not snapshot_frame.empty:
        subset = snapshot_frame[snapshot_frame.get("match_uid", "").map(_text) == match_uid]
        for _, row in subset.iterrows():
            snapshot_id = _text(row.get("feature_snapshot_id"))
            if snapshot_id and snapshot_id != primary_snapshot_id:
                ids.add(snapshot_id)
            entry = {
                "feature_snapshot_id": snapshot_id,
                "prediction_uid": _text(row.get("prediction_uid")),
                "source_file": "prediction_snapshots.csv",
                "source_row": int(row["_source_row"]),
                "snapshot_role": _text(row.get("snapshot_role")),
            }
            key = (entry["source_file"], entry["source_row"])
            if key not in seen:
                provenance.append(entry)
                seen.add(key)
    provenance.sort(
        key=lambda item: (
            _text(item.get("source_file")),
            int(item.get("source_row", 0)),
            _text(item.get("feature_snapshot_id")),
        )
    )
    return sorted(ids), provenance


def _reason_codes(
    *,
    tier: str,
    exact_id_claimed: bool,
    snapshot_verified: bool,
    features_complete: bool,
    schema_compatible: bool,
    outcome_status: str,
    odds_status: str,
    prediction_observed: pd.Timestamp | None,
    match_start: pd.Timestamp | None,
    prediction_is_prestart: bool,
    feature_observed: pd.Timestamp | None,
    feature_is_prestart: bool,
    feature_identity_compatible: bool,
    logging_quality_valid: bool,
    rescore_quality_valid: bool,
    legacy_candidates: list[dict[str, Any]],
    legacy_hash_count: int,
    identity_record_status: str,
) -> list[str]:
    if tier == "GOLD_REPLAY":
        return ["READY_FOR_SAME_SCHEMA_REPLAY"]
    reasons: list[str] = []
    if identity_record_status == "identity_conflict":
        reasons.append("MATCH_IDENTITY_CONFLICT")
    elif identity_record_status == "superseded_identity":
        reasons.append("SUPERSEDED_MATCH_IDENTITY")
    if tier == "LEGACY_MATCHED":
        reasons.append("LEGACY_NO_IMMUTABLE_LINEAGE")
    elif tier == "NO_VECTOR":
        reasons.append("NO_REPLAYABLE_VECTOR")
        if legacy_candidates and legacy_hash_count > 1:
            reasons.append("AMBIGUOUS_LEGACY_VECTOR")
        elif legacy_candidates:
            reasons.append("LEGACY_VECTOR_STRUCTURALLY_INVALID")
        elif not exact_id_claimed:
            reasons.append("NO_EXACT_LINEAGE_ID")
    if exact_id_claimed and not snapshot_verified:
        reasons.append("EXACT_LINEAGE_UNVERIFIED")
    if exact_id_claimed and not feature_identity_compatible:
        reasons.append("FEATURE_IDENTITY_UNVERIFIED")
    if exact_id_claimed and not logging_quality_valid:
        reasons.append("LOGGING_NOT_SNAPSHOT_V2")
    if exact_id_claimed and not rescore_quality_valid:
        reasons.append("RESCORE_NOT_EXACT_FEATURE_SNAPSHOT")
    if not features_complete:
        reasons.append("FEATURES_INCOMPLETE")
    if not schema_compatible:
        reasons.append("SCHEMA_NOT_ARTIFACT_COMPATIBLE")
    if outcome_status == "conflicting_terminal_outcomes":
        reasons.append("CONFLICTING_AUTHORITATIVE_OUTCOME")
    elif outcome_status != "authoritative_conflict_free":
        reasons.append("MISSING_AUTHORITATIVE_OUTCOME")
    if odds_status == "rejected_0_5_without_decimal_odds":
        reasons.append("FABRICATED_MARKET_0_5_WITHOUT_DECIMAL_ODDS")
    elif odds_status != "genuine_two_sided_decimal":
        reasons.append("MISSING_TWO_SIDED_DECIMAL_ODDS")
    if prediction_observed is None:
        reasons.append("MISSING_SNAPSHOT_OBSERVED_AT")
    if match_start is None:
        reasons.append("MISSING_MATCH_START_AT_UTC")
    elif prediction_observed is not None and not prediction_is_prestart:
        reasons.append("SNAPSHOT_NOT_PRESTART")
    if feature_observed is None:
        reasons.append("MISSING_FEATURE_OBSERVED_AT")
    elif match_start is not None and not feature_is_prestart:
        reasons.append("FEATURE_VECTOR_NOT_PRESTART")
    # Preserve semantic order while eliminating duplicates.
    return list(dict.fromkeys(reasons))


def build_replay_manifest(prod_dir: str | Path) -> ReplayManifest:
    """Derive one deterministic replay classification per ``match_uid``.

    All reads are fail-closed.  No predictions are rescored and no input file is
    written.  Use :func:`write_replay_manifest` explicitly to publish a
    versioned export.
    """
    production = Path(prod_dir).resolve()
    prediction_path = production / "prediction_log.csv"
    raw_predictions, prediction_source = _read_csv(prediction_path, production)
    # Reuse the ledger's exact-vector foreign-key and structural verification.
    verified_predictions = cohorts.load_prediction_log(str(production))
    if len(raw_predictions) != len(verified_predictions):
        raise RuntimeError("prediction verification changed row cardinality")
    predictions = verified_predictions.copy()
    predictions["_source_row"] = range(2, len(predictions) + 2)

    exact_by_id, exact_by_match_uid, legacy_by_key, feature_sources = (
        _load_feature_evidence(production)
    )
    sources = [prediction_source, *feature_sources]

    snapshot_frame: pd.DataFrame | None = None
    snapshot_path = production / "prediction_snapshots.csv"
    if snapshot_path.exists():
        snapshot_frame, snapshot_source = _read_csv(snapshot_path, production)
        snapshot_frame["_source_row"] = range(2, len(snapshot_frame) + 2)
        sources.append(snapshot_source)

    rows: list[dict[str, Any]] = []
    groups = predictions.groupby("match_uid", sort=True, dropna=False)
    for raw_uid, group in groups:
        match_uid = _text(raw_uid)
        if not match_uid:
            # A blank ID cannot satisfy the one-row-per-match contract.
            raise RuntimeError("prediction_log.csv contains a blank match_uid")
        primary = _primary_prediction(group)
        group_identity_statuses = {
            _text(value).lower() for value in group.get(
                "record_status", pd.Series("", index=group.index)
            )
        }
        identity_record_status = (
            "identity_conflict"
            if "identity_conflict" in group_identity_statuses
            else "superseded_identity"
            if "superseded_identity" in group_identity_statuses
            else _text(primary.get("record_status")).lower()
        )
        identity_terminal = identity_record_status in {
            "identity_conflict", "superseded_identity",
        }
        primary_snapshot_id = _text(primary.get("feature_snapshot_id"))
        exact_candidates = exact_by_id.get(primary_snapshot_id, []) if primary_snapshot_id else []
        selected: dict[str, Any] | None = exact_candidates[0] if exact_candidates else None
        prediction_verified = _truthy(primary.get("feature_snapshot_verified"))
        feature_identity_compatible = _feature_identity_compatible(selected, primary)
        snapshot_verified = bool(
            primary_snapshot_id
            and selected is not None
            and selected["structurally_verified"]
            and prediction_verified
            and feature_identity_compatible
        )

        legacy_key = (
            _normal_name(primary.get("p1")),
            _normal_name(primary.get("p2")),
            _date_text(primary.get("match_date")),
        )
        legacy_candidates = legacy_by_key.get(legacy_key, [])
        valid_legacy = [
            candidate for candidate in legacy_candidates if candidate["structurally_verified"]
        ]
        legacy_hashes = {
            candidate["feature_vector_sha256"] for candidate in valid_legacy
        }
        legacy_selected = False
        if not primary_snapshot_id and len(legacy_hashes) == 1:
            selected = next(
                candidate
                for candidate in valid_legacy
                if candidate["feature_vector_sha256"] in legacy_hashes
            )
            legacy_selected = True
            feature_identity_compatible = _feature_identity_compatible(selected, primary)

        # The operational row and the immutable feature evidence must agree.
        # A stale optimistic flag on either side cannot promote the vector.
        features_complete = bool(
            not identity_terminal
            and _truthy(primary.get("features_complete"))
            and selected is not None
            and selected["features_complete"]
        )
        schema_compatible = bool(
            selected
            and selected["structurally_verified"]
            and selected["feature_schema_sha256"] == ARTIFACT_SCHEMA_SHA256
            and selected["feature_count"] == len(EXACT_141_FEATURES)
        )
        prediction_observed = _prediction_observed_at(primary)
        feature_observed = (
            _parse_utc(selected.get("feature_observed_at")) if selected else None
        )
        match_start = _match_start_at_utc(primary)
        prediction_is_prestart = bool(
            prediction_observed is not None
            and match_start is not None
            and prediction_observed < match_start
        )
        feature_is_prestart = bool(
            feature_observed is not None
            and match_start is not None
            and feature_observed < match_start
        )
        is_prestart = prediction_is_prestart and feature_is_prestart
        logging_quality_valid = _text(primary.get("logging_quality")) == "snapshot_v2"
        rescore_quality_valid = (
            _text(primary.get("rescore_quality")) == "exact_feature_snapshot"
        )
        outcome_status, actual_winner, outcome_rows = _outcome_evidence(group)
        odds_status, p1_odds, p2_odds = _odds_evidence(primary)

        gold = all(
            [
                primary_snapshot_id,
                snapshot_verified,
                features_complete,
                schema_compatible,
                outcome_status == "authoritative_conflict_free",
                odds_status == "genuine_two_sided_decimal",
                is_prestart,
                logging_quality_valid,
                rescore_quality_valid,
            ]
        )
        if identity_terminal:
            tier = "NO_VECTOR"
            selected_kind = "identity_terminal_excluded"
            selection_rule = "identity_contract_fail_closed"
        elif gold:
            tier = "GOLD_REPLAY"
            selected_kind = "exact_opening"
            selection_rule = "prediction_log_operational_opening_id"
        elif primary_snapshot_id:
            tier = "EXACT_INCOMPLETE"
            selected_kind = "exact_opening" if selected else "exact_id_unresolved"
            selection_rule = "prediction_log_operational_opening_id"
        elif legacy_selected:
            tier = "LEGACY_MATCHED"
            selected_kind = "legacy_unambiguous_vector"
            selection_rule = "same_oriented_players_exact_date_single_vector_hash"
        else:
            tier = "NO_VECTOR"
            selected_kind = "none"
            selection_rule = "no_unambiguous_opening_vector"

        alternative_ids, alternative_provenance = _alternative_evidence(
            match_uid=match_uid,
            primary_snapshot_id=primary_snapshot_id,
            primary_candidate=selected,
            exact_by_match_uid=exact_by_match_uid,
            legacy_candidates=legacy_candidates,
            snapshot_frame=snapshot_frame,
            primary=primary,
        )
        reasons = _reason_codes(
            tier=tier,
            exact_id_claimed=bool(primary_snapshot_id),
            snapshot_verified=snapshot_verified,
            features_complete=features_complete,
            schema_compatible=schema_compatible,
            outcome_status=outcome_status,
            odds_status=odds_status,
            prediction_observed=prediction_observed,
            match_start=match_start,
            prediction_is_prestart=prediction_is_prestart,
            feature_observed=feature_observed,
            feature_is_prestart=feature_is_prestart,
            feature_identity_compatible=feature_identity_compatible,
            logging_quality_valid=logging_quality_valid,
            rescore_quality_valid=rescore_quality_valid,
            legacy_candidates=legacy_candidates,
            legacy_hash_count=len(legacy_hashes),
            identity_record_status=identity_record_status,
        )
        market_p1 = _numeric(primary.get("market_p1_prob"))
        market_p2 = _numeric(primary.get("market_p2_prob"))
        source_rows = sorted(group["_source_row"].astype(int).tolist())
        feature_observed_text = _iso_utc(feature_observed)
        rows.append(
            {
                "match_uid": match_uid,
                "replay_tier": tier,
                "match_date": _date_text(primary.get("match_date")),
                "tournament": _text(primary.get("tournament")),
                "p1": _text(primary.get("p1")),
                "p2": _text(primary.get("p2")),
                "selected_snapshot_kind": selected_kind,
                "snapshot_selection_rule": selection_rule,
                "feature_snapshot_id": primary_snapshot_id,
                "snapshot_verified": snapshot_verified,
                "features_complete": features_complete,
                "feature_schema_sha256": selected.get("feature_schema_sha256", "") if selected else "",
                "feature_vector_sha256": selected.get("feature_vector_sha256", "") if selected else "",
                "feature_count": selected.get("feature_count", "") if selected else "",
                "artifact_schema_id": ARTIFACT_SCHEMA_ID,
                "artifact_schema_sha256": ARTIFACT_SCHEMA_SHA256,
                "artifact_schema_compatible": schema_compatible,
                "prediction_observed_at": _iso_utc(prediction_observed),
                "feature_observed_at": feature_observed_text,
                "match_start_at_utc": _iso_utc(match_start),
                "is_opening_snapshot": bool(primary_snapshot_id or legacy_selected),
                "is_prestart_snapshot": is_prestart,
                "feature_source_file": selected.get("source_file", "") if selected else "",
                "feature_source_row": selected.get("source_row", "") if selected else "",
                "prediction_source_file": "prediction_log.csv",
                "prediction_source_row": int(primary["_source_row"]),
                "prediction_source_rows": _json(source_rows),
                "alternative_snapshot_count": len(alternative_provenance),
                "alternative_snapshot_ids": _json(alternative_ids),
                "alternative_snapshot_provenance": _json(alternative_provenance),
                "actual_winner": actual_winner,
                "outcome_status": outcome_status,
                "outcome_source_file": "prediction_log.csv",
                "outcome_source_rows": _json(outcome_rows),
                "p1_odds_decimal": p1_odds,
                "p2_odds_decimal": p2_odds,
                "market_p1_prob": market_p1 if market_p1 is not None else "",
                "market_p2_prob": market_p2 if market_p2 is not None else "",
                "odds_evidence_status": odds_status,
                "odds_source_file": "prediction_log.csv",
                "odds_source_row": int(primary["_source_row"]),
                "reason_codes": ";".join(reasons),
            }
        )

    result = pd.DataFrame(rows, columns=OUTPUT_COLUMNS).sort_values(
        "match_uid", kind="stable"
    ).reset_index(drop=True)
    if result["match_uid"].duplicated().any():
        raise RuntimeError("historical replay manifest is not one row per match_uid")
    sources.sort(key=lambda item: item["path"])
    return ReplayManifest(frame=result, source_files=sources)


def _reason_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for value in frame.get("reason_codes", pd.Series(dtype=object)):
        counts.update(reason for reason in _text(value).split(";") if reason)
    return dict(sorted(counts.items()))


def manifest_summary(manifest: ReplayManifest) -> dict[str, Any]:
    tier_counts = manifest.frame["replay_tier"].value_counts().to_dict()
    return {
        "manifest_type": "historical_replay",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "rows": int(len(manifest.frame)),
        "unique_match_uids": int(manifest.frame["match_uid"].nunique()),
        "tier_counts": {tier: int(tier_counts.get(tier, 0)) for tier in TIERS},
        "reason_counts": _reason_counts(manifest.frame),
        "artifact_schema": {
            "id": ARTIFACT_SCHEMA_ID,
            "feature_count": len(EXACT_141_FEATURES),
            "sha256": ARTIFACT_SCHEMA_SHA256,
        },
    }


def write_replay_manifest(
    manifest: ReplayManifest,
    out_dir: str | Path,
    *,
    generated_at: datetime | None = None,
) -> Path:
    """Write CSV + JSON into a new timestamp-versioned child directory."""
    timestamp = generated_at or datetime.now(timezone.utc)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    timestamp = timestamp.astimezone(timezone.utc)
    version = timestamp.strftime("historical_replay_%Y%m%dT%H%M%SZ")
    destination = Path(out_dir).resolve() / version
    # exist_ok=False protects prior evidence from silent replacement.
    destination.mkdir(parents=True, exist_ok=False)

    csv_path = destination / OUTPUT_CSV
    manifest.frame.to_csv(csv_path, index=False)
    metadata = manifest_summary(manifest)
    metadata.update(
        {
            "generated_at": timestamp.isoformat().replace("+00:00", "Z"),
            "selection_contract": (
                "operational opening snapshot; alternatives retained but never selected"
            ),
            "source_files": manifest.source_files,
            "artifacts": {
                OUTPUT_CSV: {
                    "sha256": _sha256_file(csv_path),
                    "rows": int(len(manifest.frame)),
                }
            },
        }
    )
    json_path = destination / OUTPUT_JSON
    json_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prod-dir",
        default=str(Path(__file__).resolve().parents[1]),
        help="Production directory containing prediction_log.csv and logs/",
    )
    parser.add_argument(
        "--out-dir",
        help=(
            "Optional parent directory. When omitted, no files are written. "
            "When supplied, a new timestamp-versioned child is created."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    manifest = build_replay_manifest(args.prod_dir)
    summary = manifest_summary(manifest)
    if args.out_dir:
        destination = write_replay_manifest(manifest, args.out_dir)
        summary["written_to"] = str(destination)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through main()
    raise SystemExit(main())
