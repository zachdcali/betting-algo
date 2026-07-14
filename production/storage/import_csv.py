"""Plan and optionally apply the additive legacy-CSV Postgres import.

Planning is the default and is entirely read-only.  ``--apply`` requires an
explicit database URL; it inserts one deterministic import batch and all facts
inside a single caller-owned transaction, then proves key/hash parity before
commit.  Source CSVs are never rewritten.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Iterator, Mapping
from uuid import NAMESPACE_URL, uuid5

from .parity import compare_memberships, compare_plan
from .records import (
    ImportPlan, RecordBatch, canonical_json, content_sha256, deterministic_key,
    record_sha256,
)
from .repository import OperationalRepository

try:
    from feature_contract import normalize_feature_vector, vector_sha256
except ImportError:  # pragma: no cover - package-style execution
    from production.feature_contract import (  # type: ignore
        normalize_feature_vector, vector_sha256,
    )

try:
    from versioning import (
        FEATURE_SCHEMA_ID, FEATURE_SCHEMA_NAME, FEATURE_SCHEMA_SHA256,
        FEATURE_SCHEMA_VERSION, LIVE_SEMANTICS_ID, LOGGING_SCHEMA_VERSION,
        OPERATIONAL_NORMALIZER_VERSION, OPERATIONAL_SCHEMA_VERSION,
    )
except ImportError:  # pragma: no cover - supports package-style execution
    from production.versioning import (  # type: ignore
        FEATURE_SCHEMA_ID, FEATURE_SCHEMA_NAME, FEATURE_SCHEMA_SHA256,
        FEATURE_SCHEMA_VERSION, LIVE_SEMANTICS_ID, LOGGING_SCHEMA_VERSION,
        OPERATIONAL_NORMALIZER_VERSION, OPERATIONAL_SCHEMA_VERSION,
    )


BASE_SOURCES = (
    "models/model_registry.json",
    "logs/audit/run_history.csv",
    "logs/audit/skipped_live_matches.csv",
    "logs/audit/settlement_audit.csv",
    "odds_history.csv",
    "logs/feature_vectors.csv",
    "prediction_snapshots.csv",
    "prediction_log.csv",
    "logs/performance_v1_shadow_predictions.csv",
    "logs/performance_v1_shadow_backfill.csv",
    "logs/all_bets.csv",
    "logs/bankroll_history.csv",
    "logs/betting_sessions.csv",
)

MODEL_REGISTRY_SOURCE = "models/model_registry.json"
MODEL_FAMILIES = ("nn", "xgboost", "random_forest")


@dataclass(frozen=True)
class SourceRow:
    relative_path: str
    row_number: int
    row: Mapping[str, str]
    row_json: str
    row_sha256: str


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_paths(production_dir: Path, *, include_run_feature_files: bool) -> list[Path]:
    paths = [production_dir / relative for relative in BASE_SOURCES]
    if include_run_feature_files:
        paths.extend(sorted((production_dir / "logs").glob("features_*.csv")))
    priority = {relative: index for index, relative in enumerate(BASE_SOURCES)}
    existing = {path.resolve() for path in paths if path.is_file()}
    return sorted(
        existing,
        key=lambda path: (
            priority.get(_relative(path, production_dir), len(priority)),
            _relative(path, production_dir),
        ),
    )


def _relative(path: Path, production_dir: Path) -> str:
    return path.resolve().relative_to(production_dir.resolve()).as_posix()


def _read_rows(path: Path, production_dir: Path) -> Iterator[SourceRow]:
    relative = _relative(path, production_dir)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return
        for physical_row, source in enumerate(reader, start=2):
            # DictReader can expose a None key for surplus cells. Preserve it
            # under an explicit marker rather than silently dropping evidence.
            row = {
                ("__surplus_cells__" if key is None else str(key)): (
                    "" if value is None else (
                        canonical_json(value) if isinstance(value, list) else str(value)
                    )
                )
                for key, value in source.items()
            }
            row_json = canonical_json(row)
            yield SourceRow(
                relative_path=relative,
                row_number=physical_row,
                row=row,
                row_json=row_json,
                row_sha256=sha256(row_json.encode("utf-8")).hexdigest(),
            )


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _decimal(value: Any) -> Decimal | None:
    text = _text(value)
    if text is None:
        return None
    try:
        result = Decimal(text)
    except InvalidOperation:
        return None
    return result if result.is_finite() else None


def _integer(value: Any) -> int | None:
    result = _decimal(value)
    if result is None or result != result.to_integral_value():
        return None
    return int(result)


def _boolean(value: Any) -> bool | None:
    text = (_text(value) or "").lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def _date(value: Any) -> date | None:
    text = _text(value)
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _timestamp(value: Any) -> datetime | None:
    text = _text(value)
    if not text:
        return None
    normalized = text.replace("Z", "+00:00")
    try:
        result = datetime.fromisoformat(normalized)
    except ValueError:
        for pattern in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p"):
            try:
                result = datetime.strptime(text, pattern)
                break
            except ValueError:
                continue
        else:
            return None
    # Operational timestamps without an offset came from cloud/local process
    # clocks whose historical timezone was not persisted. Treating them as UTC
    # is an explicit legacy-import assumption; the original string remains in
    # source_row_json for correction and no match-start conversion is inferred.
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _json(value: Any, default: Any) -> str:
    text = _text(value)
    if text is None:
        return canonical_json(default)
    try:
        return canonical_json(json.loads(text))
    except (json.JSONDecodeError, TypeError):
        return canonical_json(default)


def _provenance(source: SourceRow, import_batch_id: str) -> dict[str, Any]:
    return {
        "import_batch_id": import_batch_id,
        "source_file": source.relative_path,
        "source_row_number": source.row_number,
        "source_row_sha256": source.row_sha256,
        "source_row_json": source.row_json,
    }


def _id(prefix: str, value: Any, *fallback: Any) -> str:
    return _text(value) or deterministic_key(prefix, *fallback)


def _metrics(row: Mapping[str, str], excluded: set[str]) -> str:
    return canonical_json({
        key: value for key, value in row.items()
        if key not in excluded and _text(value) is not None
    })


def _normalized_model_version(value: Any) -> str | None:
    """Normalize registry/log spelling without inventing a new version.

    The human-facing registry historically uses ``v1.2.3`` while operational
    logs use ``1.2.3``.  The database stores the SemVer value and the registry
    entry preserves the original spelling for auditability.
    """
    version = _text(value)
    if version and re.match(r"^[vV](?=\d)", version):
        return version[1:]
    return version


def _model_release_status(
    *, bucket: str, version: str, current_version: str | None,
    entry: Mapping[str, Any],
) -> str:
    if bucket == "candidates":
        return "candidate"
    if version == current_version:
        return "promoted"
    if entry.get("archived") is True:
        return "archived"
    return "superseded"


def _model_contract_complete(
    registry_schema_version: str | None, entry: Mapping[str, Any]
) -> bool:
    required = (
        registry_schema_version,
        _text(entry.get("feature_schema_id")),
        _text(entry.get("feature_schema_sha256")),
        _text(entry.get("training_feature_semantics_id")),
        _text(entry.get("live_feature_semantics_id")),
        _text(entry.get("training_dataset_id")),
        _text(entry.get("model_sha256")),
    )
    if not all(required):
        return False
    if _text(entry.get("scaler_file")) and not _text(entry.get("scaler_sha256")):
        return False
    probability_mode = (_text(entry.get("probability_mode")) or "raw").lower()
    calibrator_sha256 = _text(
        entry.get("calibrator_sha256") or entry.get("calibrated_model_sha256")
    )
    if probability_mode == "calibrated" and not calibrator_sha256:
        return False
    return True


def _model_registry_records(
    production_dir: Path, batch_id: str,
) -> tuple[
    dict[str, Any], list[dict[str, Any]], list[dict[str, Any]],
    dict[tuple[str, str], str]
]:
    """Materialize every registry release and a family/version FK index."""
    path = production_dir / MODEL_REGISTRY_SOURCE
    if not path.is_file():
        return {}, [], [], {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("model_registry.json must contain a JSON object")
    registry_schema_version = _text(payload.get("registry_schema_version"))
    registry_generation_sha256 = _file_sha256(path)
    registry_generation = payload.get("registry_generation")
    if (
        isinstance(registry_generation, bool)
        or not isinstance(registry_generation, int)
        or registry_generation < 1
    ):
        raise ValueError("model registry requires a positive registry_generation")
    registry_effective_text = _text(payload.get("registry_effective_at"))
    try:
        registry_effective_at = datetime.fromisoformat(
            str(registry_effective_text or "").replace("Z", "+00:00")
        )
        if (
            registry_effective_at.tzinfo is None
            or registry_effective_at.utcoffset() is None
        ):
            raise ValueError("timezone required")
        registry_effective_at = registry_effective_at.astimezone(timezone.utc)
    except ValueError:
        raise ValueError(
            "model registry requires timezone-aware registry_effective_at"
        ) from None
    registry_source_json = canonical_json({
        "registry_generation": registry_generation,
        "registry_effective_at": registry_effective_at,
        "registry_schema_version": registry_schema_version,
        "registry_generation_sha256": registry_generation_sha256,
    })
    registry_source = SourceRow(
        MODEL_REGISTRY_SOURCE,
        0,
        {},
        registry_source_json,
        sha256(registry_source_json.encode("utf-8")).hexdigest(),
    )
    generation_record = {
        "idempotency_key": (
            f"model_registry_generation:{registry_generation_sha256}"
        ),
        "registry_generation_sha256": registry_generation_sha256,
        "generation_sequence": registry_generation,
        "registry_schema_version": registry_schema_version,
        "effective_at": registry_effective_at,
        **_provenance(registry_source, batch_id),
    }
    releases: list[dict[str, Any]] = []
    status_events: list[dict[str, Any]] = []
    release_keys: dict[tuple[str, str], str] = {}
    for family in MODEL_FAMILIES:
        family_statuses: list[dict[str, Any]] = []
        raw_block = payload if family == "nn" else payload.get(family, {})
        if not isinstance(raw_block, Mapping):
            raise ValueError(f"model registry family {family!r} must be an object")
        current_version = _normalized_model_version(raw_block.get("current_version"))
        for bucket in ("models", "candidates"):
            raw_entries = raw_block.get(bucket, {})
            if not isinstance(raw_entries, Mapping):
                raise ValueError(
                    f"model registry {family}.{bucket} must be an object"
                )
            for registry_version, raw_entry in raw_entries.items():
                if not isinstance(raw_entry, Mapping):
                    raise ValueError(
                        f"model registry entry {family}:{registry_version} must be an object"
                    )
                version = _normalized_model_version(registry_version)
                if not version:
                    raise ValueError(f"model registry entry for {family} has no version")
                identity = (family, version)
                if identity in release_keys:
                    raise ValueError(
                        f"duplicate model registry release {family}:{version}"
                    )
                release_key = f"model_release:{family}:{version}"
                entry = dict(raw_entry)
                source_payload = {
                    "bucket": bucket,
                    "family": family,
                    "registry_entry": entry,
                    "registry_schema_version": registry_schema_version,
                    "registry_version": str(registry_version),
                }
                source_json = canonical_json(source_payload)
                source = SourceRow(
                    MODEL_REGISTRY_SOURCE,
                    0,
                    {},
                    source_json,
                    sha256(source_json.encode("utf-8")).hexdigest(),
                )
                calibrator_sha256 = _text(
                    entry.get("calibrator_sha256")
                    or entry.get("calibrated_model_sha256")
                )
                releases.append({
                    "idempotency_key": release_key,
                    "model_family": family,
                    "model_version": version,
                    "release_status": "registered",
                    "registry_schema_version": registry_schema_version,
                    "feature_schema_identifier": _text(entry.get("feature_schema_id")),
                    "feature_schema_sha256": _text(entry.get("feature_schema_sha256")),
                    "training_feature_semantics_id": _text(
                        entry.get("training_feature_semantics_id")
                    ),
                    "live_feature_semantics_id": _text(
                        entry.get("live_feature_semantics_id")
                    ),
                    "training_dataset_id": _text(entry.get("training_dataset_id")),
                    "model_sha256": _text(entry.get("model_sha256")),
                    "scaler_sha256": _text(entry.get("scaler_sha256")),
                    "calibrator_sha256": calibrator_sha256,
                    "registry_entry": canonical_json(entry),
                    "contract_complete": _model_contract_complete(
                        registry_schema_version, entry
                    ),
                    **_provenance(source, batch_id),
                })
                status_event = {
                    "idempotency_key": (
                        f"model_release_status:{registry_generation_sha256}:"
                        f"{family}:{version}"
                    ),
                    "model_release_key": release_key,
                    "model_family": family,
                    "registry_generation_sha256": registry_generation_sha256,
                    "release_status": _model_release_status(
                        bucket=bucket,
                        version=version,
                        current_version=current_version,
                        entry=entry,
                    ),
                    **_provenance(source, batch_id),
                }
                status_events.append(status_event)
                family_statuses.append(status_event)
                release_keys[identity] = release_key
        promoted = [
            event for event in family_statuses
            if event["release_status"] == "promoted"
        ]
        if len(promoted) > 1:
            raise ValueError(
                f"model registry generation promotes multiple {family} releases"
            )
        if current_version is not None and len(promoted) != 1:
            raise ValueError(
                f"model registry {family}.current_version {current_version!r} "
                "must identify exactly one models release"
            )
    return generation_record, releases, status_events, release_keys


def _bind_prediction_release(
    record: Mapping[str, Any],
    release_keys: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    """Bind operational observations; shadow variants stay explicitly null."""
    bound = dict(record)
    bound["model_release_key"] = None
    if (_text(bound.get("model_role")) or "").lower() == "shadow":
        return bound
    family = (_text(bound.get("model_family")) or "").lower()
    version = _normalized_model_version(bound.get("model_version"))
    if family and version:
        bound["model_version"] = version
        bound["model_release_key"] = release_keys.get((family, version))
    return bound


def _pipeline_run(source: SourceRow, batch_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    run_key = _id("run", row.get("run_id"), source.row_sha256)
    excluded = {
        "run_id", "run_kind", "status", "started_at", "completed_at", "error_message",
    }
    yield "ops.pipeline_runs", {
        "idempotency_key": f"pipeline_run:{run_key}",
        "external_run_id": run_key,
        "run_kind": _text(row.get("run_kind")) or "legacy_unknown",
        "status": (_text(row.get("status")) or "failed").lower(),
        "started_at": _timestamp(row.get("started_at")),
        "completed_at": _timestamp(row.get("completed_at")),
        "error_message": _text(row.get("error_message")),
        "metrics": _metrics(row, excluded),
        **_provenance(source, batch_id),
    }


def _odds(source: SourceRow, batch_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    observed_at = _timestamp(row.get("odds_scraped_at") or row.get("logged_at"))
    external_run_id = _text(row.get("run_id"))
    fetch_key = deterministic_key("bovada_fetch", external_run_id, observed_at)
    yield "raw.source_fetches", {
        "idempotency_key": fetch_key,
        "external_run_id": external_run_id,
        "source_name": "bovada",
        "fetch_kind": "odds",
        "status": "success",
        "started_at": observed_at,
        "completed_at": observed_at,
        "request": canonical_json({"legacy_import": True}),
        **_provenance(source, batch_id),
    }
    observation_key = _id(
        "odds", row.get("odds_snapshot_uid"), row.get("match_uid"), observed_at,
        row.get("p1_odds_decimal"), row.get("p2_odds_decimal"),
    )
    yield "ops.odds_observations", {
        "idempotency_key": f"odds_observation:{observation_key}",
        "external_observation_id": observation_key,
        "source_fetch_key": fetch_key,
        "external_run_id": external_run_id,
        "match_uid": _text(row.get("match_uid")),
        "observed_at": observed_at,
        "match_date": _date(row.get("match_date")),
        "tournament": _text(row.get("tournament")),
        "surface": _text(row.get("surface")),
        "level": _text(row.get("level")),
        "round": _text(row.get("round")),
        "player1": _text(row.get("p1")),
        "player2": _text(row.get("p2")),
        "bookmaker": "bovada",
        "market_type": "moneyline",
        "player1_decimal_odds": _decimal(row.get("p1_odds_decimal")),
        "player2_decimal_odds": _decimal(row.get("p2_odds_decimal")),
        "player1_american_odds": _integer(row.get("p1_odds_american")),
        "player2_american_odds": _integer(row.get("p2_odds_american")),
        "player1_market_probability": _decimal(row.get("market_p1_prob")),
        "player2_market_probability": _decimal(row.get("market_p2_prob")),
        "market_payload": _metrics(row, set()),
        **_provenance(source, batch_id),
    }


def _load_feature_names(production_dir: Path) -> tuple[str, ...]:
    path = production_dir / "features" / "schema_141.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    ordered = sorted(payload["features"], key=lambda feature: int(feature["index"]))
    names = tuple(str(feature["name"]) for feature in ordered)
    joined_hash = sha256("\x1f".join(names).encode("utf-8")).hexdigest()
    if joined_hash != FEATURE_SCHEMA_SHA256:
        raise ValueError(
            f"schema_141.json hash {joined_hash} does not match version contract "
            f"{FEATURE_SCHEMA_SHA256}"
        )
    return names


def _feature_payload(
    row: Mapping[str, str], feature_names: tuple[str, ...]
) -> tuple[dict[str, float], str | None, tuple[str, ...]]:
    raw_payload = _text(row.get("features_json"))
    if raw_payload:
        try:
            decoded = json.loads(raw_payload)
        except json.JSONDecodeError:
            decoded = {}
    else:
        decoded = row
    if not isinstance(decoded, Mapping):
        return {}, None, ("payload_not_object",)
    vector, issues = normalize_feature_vector(decoded, feature_names)
    if issues:
        return vector, None, issues
    return vector, vector_sha256(vector, feature_names), ()


def _string_list(value: Any) -> list[str]:
    text = _text(value)
    if text is None:
        return []
    try:
        decoded = json.loads(text)
    except json.JSONDecodeError:
        decoded = text
    if isinstance(decoded, list):
        return [str(item).strip() for item in decoded if str(item).strip()]
    if isinstance(decoded, str):
        return [item.strip() for item in decoded.split(",") if item.strip()]
    return [str(decoded)]


def _feature(
    source: SourceRow, batch_id: str, feature_names: tuple[str, ...]
) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    vector, computed_hash, structural_issues = _feature_payload(row, feature_names)
    schema_key = f"feature_schema:{FEATURE_SCHEMA_ID}:{FEATURE_SCHEMA_SHA256}"
    yield "ml.feature_schemas", {
        "idempotency_key": schema_key,
        "schema_name": FEATURE_SCHEMA_NAME,
        "schema_version": FEATURE_SCHEMA_VERSION,
        "schema_identifier": FEATURE_SCHEMA_ID,
        "schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_count": len(feature_names),
        "feature_names": canonical_json(list(feature_names)),
        "feature_contract": canonical_json({
            "semantics_id": LIVE_SEMANTICS_ID,
            "logging_schema_version": LOGGING_SCHEMA_VERSION,
        }),
        **_provenance(source, batch_id),
    }
    external_id = _id(
        "feature_snapshot", row.get("feature_snapshot_id"), row.get("match_uid"),
        row.get("run_id"), row.get("p1") or row.get("player1_raw"),
        row.get("p2") or row.get("player2_raw"), row.get("match_date"), source.row_sha256,
    )
    build_status = (_text(row.get("build_status") or row.get("status")) or "unknown").lower()
    explicit_complete = _boolean(row.get("features_complete"))
    defaulted_features = _string_list(
        row.get("defaulted_features") or row.get("meta_defaulted_features")
    )
    if explicit_complete is None:
        explicit_complete = not bool(defaulted_features)
    source_schema_hash = _text(row.get("feature_schema_sha256"))
    if source_schema_hash and source_schema_hash != FEATURE_SCHEMA_SHA256:
        structural_issues += ("source_schema_hash_mismatch",)
    source_feature_count = _integer(row.get("feature_count"))
    if source_feature_count is not None and source_feature_count != len(feature_names):
        structural_issues += ("source_feature_count_mismatch",)
    source_vector_hash = _text(row.get("feature_vector_sha256"))
    if source_vector_hash and computed_hash and source_vector_hash != computed_hash:
        structural_issues += ("source_vector_hash_mismatch",)
    if defaulted_features:
        structural_issues += ("declared_defaulted_features",)
    if structural_issues:
        computed_hash = None
        defaulted_features.extend(
            f"structural:{issue}" for issue in structural_issues
            if f"structural:{issue}" not in defaulted_features
        )
    lineage_quality = (
        "exact_feature_snapshot_id"
        if _text(row.get("feature_snapshot_id"))
        else "legacy_source_row_snapshot"
    )
    yield "ml.feature_snapshots", {
        "idempotency_key": f"feature_snapshot:{external_id}",
        "external_feature_snapshot_id": external_id,
        "external_run_id": _text(row.get("run_id")),
        "match_uid": _text(row.get("match_uid")),
        "feature_schema_identifier": FEATURE_SCHEMA_ID,
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "feature_semantics_identifier": LIVE_SEMANTICS_ID,
        "lineage_quality": lineage_quality,
        "captured_at": _timestamp(
            row.get("logged_at") or row.get("timestamp") or row.get("run_started_at")
        ),
        "build_status": build_status,
        "features_complete": bool(explicit_complete and computed_hash and build_status == "ok"),
        "feature_count": len(vector),
        "feature_vector_sha256": computed_hash,
        "feature_vector": canonical_json(vector),
        "defaulted_features": canonical_json(defaulted_features),
        **_provenance(source, batch_id),
    }


def _prediction(source: SourceRow, batch_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    base_id = _id(
        "prediction", row.get("prediction_uid"), row.get("match_uid"),
        row.get("feature_snapshot_id"), row.get("logged_at"), source.row_sha256,
    )
    families = (
        ("nn", "model_p1_prob", "model_p2_prob", "nn_model_version"),
        ("xgboost", "xgb_p1_prob", "xgb_p2_prob", "xgb_model_version"),
        ("random_forest", "rf_p1_prob", "rf_p2_prob", "rf_model_version"),
    )
    primary = (_text(row.get("primary_model_family")) or "nn").lower()
    for family, p1_field, p2_field, version_field in families:
        p1_probability = _decimal(row.get(p1_field))
        p2_probability = _decimal(row.get(p2_field))
        if p1_probability is None:
            continue
        if p2_probability is None:
            p2_probability = Decimal("1") - p1_probability
        version = _text(row.get(version_field))
        if family == "nn":
            version = version or _text(row.get("model_version"))
        observation_key = f"prediction:{base_id}:{family}:{version or 'unknown'}"
        yield "ml.prediction_observations", {
            "idempotency_key": observation_key,
            "external_prediction_id": base_id,
            "external_run_id": _text(row.get("run_id")),
            "match_uid": _text(row.get("match_uid")),
            "external_feature_snapshot_id": _text(row.get("feature_snapshot_id")),
            "predicted_at": _timestamp(row.get("logged_at")),
            "model_family": family,
            "model_version": version or "unknown",
            "model_role": "promoted" if family == primary else "companion",
            "player1_probability": p1_probability,
            "player2_probability": p2_probability,
            "logging_quality": _text(row.get("logging_quality")),
            **_provenance(source, batch_id),
        }


def _shadow_prediction(
    source: SourceRow, batch_id: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    """Map both forward and historical side-model evidence without promotion.

    Failed shadow attempts remain observations with null probabilities; their
    status/error metadata is operational evidence and must not disappear merely
    because a model did not return a score.
    """
    row = source.row
    source_kind = (
        "backfill" if source.relative_path.endswith("_backfill.csv") else "forward"
    )
    family = _text(row.get("model_family")) or "unknown"
    version = _text(row.get("model_version")) or "unknown"
    external_id = _id(
        "shadow_prediction",
        row.get("shadow_prediction_uid"),
        source_kind,
        row.get("match_uid"),
        row.get("feature_snapshot_id"),
        row.get("logged_at"),
        family,
        version,
        source.row_sha256,
    )
    p1_probability = _decimal(row.get("shadow_p1_prob"))
    p2_probability = _decimal(row.get("shadow_p2_prob"))
    if p2_probability is None and p1_probability is not None:
        p2_probability = Decimal("1") - p1_probability
    metadata = {
        "shadow_source": source_kind,
        "feature_set": _text(row.get("feature_set")),
        "feature_count": _integer(row.get("n_features")),
        "performance_features_available": _boolean(
            row.get("performance_features_available")
        ),
        "shadow_status": _text(row.get("shadow_status")) or "unknown",
        "shadow_error": _text(row.get("shadow_error")),
        "backfill_quality": _text(row.get("backfill_quality")),
        "backfill_source": _text(row.get("backfill_source")),
        "source_feature_file": _text(row.get("source_feature_file")),
        "prediction_logged_at": _text(row.get("prediction_logged_at")),
    }
    yield "ml.prediction_observations", {
        "idempotency_key": deterministic_key(
            "prediction_shadow", source_kind, external_id, family, version,
        ),
        "external_prediction_id": external_id,
        "external_run_id": _text(row.get("run_id")),
        "match_uid": _text(row.get("match_uid")),
        "external_feature_snapshot_id": _text(row.get("feature_snapshot_id")),
        "predicted_at": _timestamp(row.get("logged_at")),
        "model_family": family,
        "model_version": version,
        "model_role": "shadow",
        "player1_probability": p1_probability,
        "player2_probability": p2_probability,
        "logging_quality": (
            _text(row.get("backfill_quality")) or "shadow_forward"
        ),
        "metadata": canonical_json(metadata),
        **_provenance(source, batch_id),
    }


def _settlement_from_prediction(
    source: SourceRow, batch_id: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    winner = _integer(row.get("actual_winner"))
    match_uid = _text(row.get("match_uid"))
    if winner not in {1, 2} or not match_uid:
        return
    yield "ops.settlement_events", {
        "idempotency_key": f"settlement:{match_uid}",
        "match_uid": match_uid,
        "external_run_id": _text(row.get("latest_run_id") or row.get("run_id")),
        "settled_at": _timestamp(row.get("settled_at")),
        "result_status": "settled",
        "actual_winner": winner,
        "score": _text(row.get("score")),
        "evidence": canonical_json({"legacy_authoritative_prediction_log": True}),
        **_provenance(source, batch_id),
    }


def _settlement_attempt(
    source: SourceRow, batch_id: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    external_id = _id(
        "settlement_attempt", row.get("settlement_event_id"), row.get("run_id"),
        row.get("match_uid"), row.get("logged_at"), row.get("outcome_code"),
    )
    yield "ops.settlement_attempts", {
        "idempotency_key": f"settlement_attempt:{external_id}",
        "external_attempt_id": external_id,
        "external_run_id": _text(row.get("run_id")),
        "match_uid": _text(row.get("match_uid")),
        "attempted_at": _timestamp(row.get("logged_at")),
        "dry_run": _boolean(row.get("dry_run")) or False,
        "outcome_code": _text(row.get("outcome_code")) or "unknown",
        "outcome_detail": _text(row.get("outcome_detail")),
        "confidence_score": _decimal(row.get("settlement_score")),
        "evidence": _json(row.get("settlement_evidence"), {}),
        **_provenance(source, batch_id),
    }


def _skip(source: SourceRow, batch_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    external_id = _id(
        "skip", row.get("skip_event_id"), row.get("run_id"), row.get("match_uid"),
        row.get("stage"), row.get("skip_reason_code"), row.get("logged_at"),
    )
    yield "ops.skip_events", {
        "idempotency_key": f"skip:{external_id}",
        "external_skip_event_id": external_id,
        "external_run_id": _text(row.get("run_id")),
        "match_uid": _text(row.get("match_uid")),
        "external_feature_snapshot_id": _text(row.get("feature_snapshot_id")),
        "external_prediction_id": _text(row.get("prediction_uid")),
        "skipped_at": _timestamp(row.get("logged_at")),
        "stage_name": _text(row.get("stage")) or "unknown",
        "reason_code": _text(row.get("skip_reason_code")) or "unknown",
        "reason_detail": _text(row.get("skip_reason_detail")),
        "context": _metrics(row, {
            "skip_event_id", "run_id", "match_uid", "feature_snapshot_id",
            "prediction_uid", "logged_at", "stage", "skip_reason_code",
            "skip_reason_detail",
        }),
        **_provenance(source, batch_id),
    }


def _bet(source: SourceRow, batch_id: str) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    external_id = _id(
        "bet", row.get("bet_id"), row.get("match_uid"), row.get("bet_on"),
        row.get("match_date"), row.get("timestamp"),
    )
    account_key = "paper_account:default"
    yield "ops.paper_accounts", {
        "idempotency_key": account_key,
        "account_code": "default",
        "display_name": "Legacy paper account",
        "currency": "USD",
        "status": "active",
        "starting_capital": _decimal(row.get("bankroll_before")),
        **_provenance(source, batch_id),
    }
    yield "ops.bet_recommendations", {
        "idempotency_key": f"bet_recommendation:{external_id}",
        "external_bet_id": external_id,
        "account_code": "default",
        "external_session_id": _text(row.get("session_id")),
        "external_run_id": _text(row.get("run_id")),
        "match_uid": _text(row.get("match_uid")),
        "external_feature_snapshot_id": _text(row.get("feature_snapshot_id")),
        "recommended_at": _timestamp(row.get("timestamp")),
        "bet_side": _text(row.get("bet_on")) or "unknown",
        "bet_on_player1": _boolean(row.get("bet_on_player1")),
        "decimal_odds": _decimal(row.get("odds_decimal")),
        "stake": _decimal(row.get("stake")),
        "stake_fraction": _decimal(row.get("stake_fraction")),
        "model_probability": _decimal(row.get("model_prob")),
        "market_probability": _decimal(row.get("market_prob")),
        "edge": _decimal(row.get("edge")),
        "kelly_fraction": _decimal(row.get("kelly_fraction")),
        "model_version": _text(row.get("model_version")),
        **_provenance(source, batch_id),
    }
    status = (_text(row.get("status")) or "pending").lower()
    occurred_at = _timestamp(row.get("settled_timestamp") or row.get("timestamp"))
    yield "ops.bet_state_events", {
        "idempotency_key": f"bet_state:{external_id}:{status}",
        "external_bet_id": external_id,
        "occurred_at": occurred_at,
        "state": status,
        "outcome": _text(row.get("outcome")),
        "actual_profit": _decimal(row.get("actual_profit")),
        "balance_after": _decimal(row.get("bankroll_after")),
        "notes": _text(row.get("notes")),
        **_provenance(source, batch_id),
    }


def _account_ledger(
    source: SourceRow, batch_id: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    external_id = deterministic_key(
        "ledger", row.get("timestamp"), row.get("session_id"),
        row.get("change_reason"), row.get("change_amount"), source.row_sha256,
    )
    yield "ops.paper_accounts", {
        "idempotency_key": "paper_account:default",
        "account_code": "default",
        "display_name": "Legacy paper account",
        "currency": "USD",
        "status": "active",
        "starting_capital": _decimal(row.get("bankroll")),
        **_provenance(source, batch_id),
    }
    yield "ops.account_ledger", {
        "idempotency_key": external_id,
        "account_code": "default",
        "external_session_id": _text(row.get("session_id")),
        "occurred_at": _timestamp(row.get("timestamp")),
        "amount": _decimal(row.get("change_amount")),
        "balance_after": _decimal(row.get("bankroll")),
        "reason": _text(row.get("change_reason")) or "legacy_unknown",
        "metadata": _metrics(row, {
            "timestamp", "session_id", "change_amount", "bankroll", "change_reason",
        }),
        **_provenance(source, batch_id),
    }


def _paper_session(
    source: SourceRow, batch_id: str
) -> Iterable[tuple[str, dict[str, Any]]]:
    row = source.row
    external_id = _id(
        "paper_session", row.get("session_id"), row.get("start_time"),
        source.row_sha256,
    )
    yield "ops.paper_accounts", {
        "idempotency_key": "paper_account:default",
        "account_code": "default",
        "display_name": "Legacy paper account",
        "currency": "USD",
        "status": "active",
        "starting_capital": _decimal(row.get("initial_bankroll")),
        **_provenance(source, batch_id),
    }
    yield "ops.paper_sessions", {
        "idempotency_key": f"paper_session:{external_id}",
        "external_session_id": external_id,
        "account_code": "default",
        "started_at": _timestamp(row.get("start_time")),
        "completed_at": _timestamp(row.get("end_time")),
        "initial_balance": _decimal(row.get("initial_bankroll")),
        "final_balance": _decimal(row.get("final_bankroll")),
        "total_bets": _integer(row.get("total_bets_placed")),
        "total_staked": _decimal(row.get("total_staked")),
        "total_profit_loss": _decimal(row.get("total_profit_loss")),
        "win_rate": _decimal(row.get("win_rate")),
        "average_odds": _decimal(row.get("avg_odds")),
        "average_edge": _decimal(row.get("avg_edge")),
        "kelly_multiplier": _decimal(row.get("kelly_multiplier_used")),
        "notes": _text(row.get("notes")),
        **_provenance(source, batch_id),
    }


Mapper = Callable[[SourceRow, str], Iterable[tuple[str, dict[str, Any]]]]
MAPPERS: dict[str, Mapper] = {
    "logs/audit/run_history.csv": _pipeline_run,
    "logs/audit/skipped_live_matches.csv": _skip,
    "logs/audit/settlement_audit.csv": _settlement_attempt,
    "odds_history.csv": _odds,
    "prediction_snapshots.csv": _prediction,
    "logs/performance_v1_shadow_predictions.csv": _shadow_prediction,
    "logs/performance_v1_shadow_backfill.csv": _shadow_prediction,
    "logs/all_bets.csv": _bet,
    "logs/bankroll_history.csv": _account_ledger,
    "logs/betting_sessions.csv": _paper_session,
}


def _consolidate_paper_account(records: dict[str, list[dict[str, Any]]]) -> None:
    """Choose one opening-capital fact instead of first-winning mutable balances."""
    candidates = records.get("ops.paper_accounts", [])
    if not candidates:
        return

    sessions = sorted(
        records.get("ops.paper_sessions", []),
        key=lambda row: (
            row.get("started_at") is None,
            row.get("started_at") or datetime.max.replace(tzinfo=timezone.utc),
            str(row.get("external_session_id") or ""),
        ),
    )
    chosen: dict[str, Any] | None = None
    if sessions:
        opening = sessions[0]
        for candidate in candidates:
            if (
                candidate.get("source_file") == opening.get("source_file")
                and candidate.get("source_row_number")
                == opening.get("source_row_number")
            ):
                chosen = dict(candidate)
                chosen["starting_capital"] = opening.get("initial_balance")
                break
    if chosen is None:
        chosen = dict(candidates[0])
    records["ops.paper_accounts"] = [chosen]


def _quarantine_conflicting_records(
    records: dict[str, list[dict[str, Any]]],
    sources: dict[str, set[str]],
) -> None:
    """Move contradictory keys out of accepted facts without guessing a winner."""
    conflict_types = {
        "ml.feature_snapshots": "conflicting_feature_snapshot",
        "ml.prediction_observations": "conflicting_external_prediction",
        "ops.settlement_events": "ambiguous_settlement_identity",
    }
    quarantined: list[dict[str, Any]] = []
    for table in sorted(tuple(records)):
        if table in {"ops.import_batches", "ops.import_conflicts"}:
            continue
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records[table]:
            grouped.setdefault(str(record.get("idempotency_key") or ""), []).append(record)
        conflicting_keys = {
            key for key, candidates in grouped.items()
            if key and len({record_sha256(candidate) for candidate in candidates}) > 1
        }
        if not conflicting_keys:
            continue
        records[table] = [
            record for record in records[table]
            if str(record.get("idempotency_key") or "") not in conflicting_keys
        ]
        for key in sorted(conflicting_keys):
            for candidate in grouped[key]:
                candidate_hash = record_sha256(candidate)
                provenance = {
                    name: candidate.get(name)
                    for name in (
                        "import_batch_id", "source_file", "source_row_number",
                        "source_row_sha256", "source_row_json",
                    )
                }
                quarantined.append({
                    "idempotency_key": deterministic_key(
                        "import_conflict", table, key, candidate_hash,
                        candidate.get("source_file"),
                        candidate.get("source_row_number"),
                        candidate.get("source_row_sha256"),
                    ),
                    "target_table": table,
                    "target_idempotency_key": key,
                    "conflict_type": conflict_types.get(
                        table, "contradictory_idempotency_key"
                    ),
                    "candidate_record_sha256": candidate_hash,
                    "candidate_record": canonical_json({
                        name: value for name, value in candidate.items()
                        if name not in {
                            "import_batch_id", "source_file", "source_row_number",
                            "source_row_sha256", "source_row_json", "record_sha256",
                        }
                    }),
                    "review_status": "open",
                    **provenance,
                })
                if candidate.get("source_file"):
                    sources.setdefault("ops.import_conflicts", set()).add(
                        str(candidate["source_file"])
                    )
    if quarantined:
        records.setdefault("ops.import_conflicts", []).extend(quarantined)


def _entity_uuid(kind: str, idempotency_key: str) -> str:
    return str(uuid5(NAMESPACE_URL, f"betting-algo:{kind}:{idempotency_key}"))


def _assign_deterministic_ids_and_links(
    records: dict[str, list[dict[str, Any]]]
) -> None:
    """Populate only unambiguous physical links; unresolved legacy links stay null."""
    primary_keys = {
        "ops.pipeline_runs": ("run_id", "pipeline_run"),
        "ops.pipeline_run_stages": ("stage_id", "pipeline_stage"),
        "raw.source_fetches": ("source_fetch_id", "source_fetch"),
        "raw.source_artifacts": ("source_artifact_id", "source_artifact"),
        "ops.odds_observations": ("odds_observation_id", "odds_observation"),
        "ml.feature_schemas": ("feature_schema_id", "feature_schema"),
        "ml.feature_snapshots": ("feature_snapshot_id", "feature_snapshot"),
        "ml.model_releases": ("model_release_id", "model_release"),
        "ml.model_registry_generations": (
            "model_registry_generation_id", "model_registry_generation"
        ),
        "ml.model_release_status_events": (
            "model_release_status_event_id", "model_release_status_event"
        ),
        "ml.prediction_observations": (
            "prediction_observation_id", "prediction_observation"
        ),
        "ops.paper_accounts": ("paper_account_id", "paper_account"),
        "ops.paper_sessions": ("paper_session_id", "paper_session"),
        "ops.account_ledger": ("account_ledger_id", "account_ledger"),
        "ops.bet_recommendations": (
            "bet_recommendation_id", "bet_recommendation"
        ),
        "ops.bet_state_events": ("bet_state_event_id", "bet_state_event"),
        "ops.settlement_attempts": (
            "settlement_attempt_id", "settlement_attempt"
        ),
        "ops.settlement_events": ("settlement_event_id", "settlement_event"),
        "ops.skip_events": ("skip_event_id", "skip_event"),
        "ops.import_conflicts": ("import_conflict_id", "import_conflict"),
    }
    for table, (column, kind) in primary_keys.items():
        for record in records.get(table, []):
            record.setdefault(
                column, _entity_uuid(kind, str(record["idempotency_key"]))
            )

    def unique_index(table: str, external_field: str) -> dict[str, dict[str, Any]]:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for record in records.get(table, []):
            value = _text(record.get(external_field))
            if value:
                grouped.setdefault(value, []).append(record)
        return {
            value: candidates[0]
            for value, candidates in grouped.items()
            if len(candidates) == 1
        }

    runs = unique_index("ops.pipeline_runs", "external_run_id")
    features = unique_index(
        "ml.feature_snapshots", "external_feature_snapshot_id"
    )
    bets = unique_index("ops.bet_recommendations", "external_bet_id")
    fetches = {
        str(record["idempotency_key"]): record
        for record in records.get("raw.source_fetches", [])
    }

    for table in (
        "ops.pipeline_run_stages", "raw.source_fetches",
        "ops.odds_observations", "ml.feature_snapshots",
        "ml.prediction_observations", "ops.bet_recommendations",
        "ops.settlement_attempts", "ops.settlement_events", "ops.skip_events",
    ):
        for record in records.get(table, []):
            run = runs.get(_text(record.get("external_run_id")) or "")
            if run is not None:
                record["run_id"] = run["run_id"]

    for record in records.get("ops.odds_observations", []):
        fetch = fetches.get(_text(record.get("source_fetch_key")) or "")
        if fetch is not None:
            record["source_fetch_id"] = fetch["source_fetch_id"]

    for table in ("ml.prediction_observations", "ops.bet_recommendations"):
        for record in records.get(table, []):
            feature = features.get(
                _text(record.get("external_feature_snapshot_id")) or ""
            )
            if feature is not None and (
                not _text(record.get("match_uid"))
                or not _text(feature.get("match_uid"))
                or record.get("match_uid") == feature.get("match_uid")
            ):
                record["feature_snapshot_id"] = feature["feature_snapshot_id"]

    for record in records.get("ops.bet_state_events", []):
        bet = bets.get(_text(record.get("external_bet_id")) or "")
        if bet is not None:
            record["bet_recommendation_id"] = bet["bet_recommendation_id"]


def build_plan(
    production_dir: str | Path,
    *,
    include_run_feature_files: bool = True,
) -> ImportPlan:
    production_dir = Path(production_dir).resolve()
    paths = _source_paths(
        production_dir, include_run_feature_files=include_run_feature_files,
    )
    file_hashes = {_relative(path, production_dir): _file_sha256(path) for path in paths}
    # Provenance is excluded from semantic fact hashes, so records can be
    # normalized with a deterministic placeholder before the final batch UUID
    # is derived from the complete target manifest.
    batch_id = "00000000-0000-0000-0000-000000000000"
    records: dict[str, list[dict[str, Any]]] = {}
    sources: dict[str, set[str]] = {}
    warnings: list[str] = []
    feature_names = _load_feature_names(production_dir)
    (
        model_registry_generation,
        model_releases,
        model_release_statuses,
        model_release_keys,
    ) = _model_registry_records(production_dir, batch_id)
    if model_releases:
        records["ml.model_registry_generations"] = [model_registry_generation]
        sources["ml.model_registry_generations"] = {MODEL_REGISTRY_SOURCE}
        records["ml.model_releases"] = model_releases
        sources["ml.model_releases"] = {MODEL_REGISTRY_SOURCE}
        records["ml.model_release_status_events"] = model_release_statuses
        sources["ml.model_release_status_events"] = {MODEL_REGISTRY_SOURCE}

    for path in paths:
        relative = _relative(path, production_dir)
        mapper = MAPPERS.get(relative)
        is_feature = relative == "logs/feature_vectors.csv" or (
            include_run_feature_files and relative.startswith("logs/features_")
        )
        is_authoritative_prediction = relative == "prediction_log.csv"
        if mapper is None and not is_feature and not is_authoritative_prediction:
            continue
        for source in _read_rows(path, production_dir):
            mapped: list[tuple[str, dict[str, Any]]] = []
            if mapper is not None:
                mapped.extend(mapper(source, batch_id))
            if is_feature:
                mapped.extend(_feature(source, batch_id, feature_names))
            if is_authoritative_prediction:
                mapped.extend(_prediction(source, batch_id))
                mapped.extend(_settlement_from_prediction(source, batch_id))
            for table, record in mapped:
                if table == "ml.prediction_observations":
                    record = _bind_prediction_release(record, model_release_keys)
                records.setdefault(table, []).append(record)
                sources.setdefault(table, set()).add(relative)

    _consolidate_paper_account(records)
    _quarantine_conflicting_records(records, sources)

    # Every source file is itself an immutable import artifact. This is not a
    # claim that it is the original HTTP payload; artifact_kind makes the
    # distinction explicit and gives operators a verifiable import manifest.
    for relative, digest in file_hashes.items():
        row_json = canonical_json({"source_file": relative, "sha256": digest})
        source = SourceRow(relative, 0, {}, row_json, sha256(row_json.encode()).hexdigest())
        is_registry = relative == MODEL_REGISTRY_SOURCE
        artifact_namespace = "model_registry_artifact" if is_registry else "legacy_csv_artifact"
        artifact_kind = "model_registry_import_source" if is_registry else "legacy_csv_import_source"
        uri_scheme = "repository-file" if is_registry else "legacy-csv"
        record = {
            "idempotency_key": f"{artifact_namespace}:{digest}:{relative}",
            "artifact_kind": artifact_kind,
            "storage_uri": f"{uri_scheme}://{relative}",
            "content_sha256": digest,
            "captured_at": None,
            "metadata": canonical_json({
                "operational_schema_version": OPERATIONAL_SCHEMA_VERSION,
                "file_size": (production_dir / relative).stat().st_size,
            }),
            **_provenance(source, batch_id),
        }
        records.setdefault("raw.source_artifacts", []).append(record)
        sources.setdefault("raw.source_artifacts", set()).add(relative)

    _assign_deterministic_ids_and_links(records)

    provisional_batches = tuple(
        RecordBatch.from_records(table, table_records)
        for table, table_records in sorted(records.items())
    )
    target_manifest = [
        {
            "table": batch.table,
            "idempotency_key": str(record["idempotency_key"]),
            "record_sha256": str(record["record_sha256"]),
        }
        for batch in provisional_batches
        for record in batch.unique_records
    ]
    target_manifest.sort(
        key=lambda item: (item["table"], item["idempotency_key"])
    )
    manifest = {
        "operational_schema_version": OPERATIONAL_SCHEMA_VERSION,
        "normalizer_contract_version": OPERATIONAL_NORMALIZER_VERSION,
        "files": file_hashes,
        "target_manifest_sha256": content_sha256(target_manifest),
        "target_row_count": len(target_manifest),
    }
    batch_id = str(uuid5(NAMESPACE_URL, canonical_json(manifest)))
    for table_records in records.values():
        for record in table_records:
            if "import_batch_id" in record:
                record["import_batch_id"] = batch_id

    batch_record = {
        "idempotency_key": f"import_batch:{batch_id}",
        "batch_id": batch_id,
        "schema_version": OPERATIONAL_SCHEMA_VERSION,
        "manifest_sha256": content_sha256(manifest),
        "source_manifest": canonical_json(manifest),
        "status": "planned",
        "planned_at": datetime.now(timezone.utc),
    }
    records.setdefault("ops.import_batches", []).insert(0, batch_record)

    batches = tuple(
        RecordBatch.from_records(
            table, table_records, source_files=sorted(sources.get(table, set())),
        )
        for table, table_records in sorted(records.items())
    )
    if not paths:
        warnings.append("no recognized CSV source files were found")
    return ImportPlan(
        import_batch_id=batch_id,
        production_dir=production_dir,
        batches=batches,
        file_sha256=file_hashes,
        warnings=tuple(warnings),
    )


def apply_plan(connection: Any, plan: ImportPlan) -> tuple[dict[str, int], dict[str, Any]]:
    """Write a plan through a caller-owned transaction and prove parity."""
    repository = OperationalRepository(connection)
    counts = repository.write_batches(plan.batches)
    parity = compare_plan(repository, plan)
    if not parity.matches:
        raise RuntimeError("operational import parity failed: " + canonical_json(parity.as_dict()))
    membership_records: list[dict[str, Any]] = []
    for batch in plan.batches:
        if batch.table == "ops.import_batches":
            continue
        for record in batch.unique_records:
            target_key = str(record["idempotency_key"])
            membership_records.append({
                "idempotency_key": deterministic_key(
                    "import_membership",
                    plan.import_batch_id,
                    batch.table,
                    target_key,
                ),
                "import_batch_id": plan.import_batch_id,
                "target_table": batch.table,
                "target_idempotency_key": target_key,
                "source_file": record.get("source_file"),
                "source_row_number": record.get("source_row_number"),
                "source_row_sha256": record.get("source_row_sha256"),
                "source_row_json": record.get("source_row_json"),
                "target_record_sha256": record.get("record_sha256"),
            })
    membership_count = repository.write_batch(RecordBatch.from_records(
        "ops.import_batch_memberships", membership_records
    ))
    membership_parity = compare_memberships(repository, plan)
    if not membership_parity.matches:
        raise RuntimeError(
            "operational import membership parity failed: "
            + canonical_json(membership_parity.as_dict())
        )
    counts["ops.import_batch_memberships"] = membership_count
    control_record = next(
        dict(record)
        for batch in plan.batches
        if batch.table == "ops.import_batches"
        for record in batch.unique_records
    )
    control_record.update({
        "status": "verified",
        "completed_at": datetime.now(timezone.utc),
        "row_counts": canonical_json(counts),
    })
    # PostgreSQL validates NOT NULL columns on the proposed INSERT before it
    # takes the ON CONFLICT update path. Re-send the complete control row rather
    # than a partial status patch so the terminal transition is valid on both
    # first apply and an idempotent retry.
    repository.write_batch(RecordBatch.from_records(
        "ops.import_batches", [control_record]
    ))
    parity_payload = parity.as_dict()
    parity_payload["membership_matches"] = membership_parity.matches
    return counts, parity_payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plan or apply the side-by-side operational Postgres import.",
    )
    parser.add_argument("--prod-dir", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--skip-run-feature-files", action="store_true",
        help="Only use logs/feature_vectors.csv; skip immutable per-run feature CSVs.",
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply in one transaction. Without this flag the command is read-only.",
    )
    parser.add_argument(
        "--database-url",
        help="Explicit Postgres connection string for local/manual use.",
    )
    parser.add_argument(
        "--database-url-env",
        help=(
            "Name of the environment variable containing the Postgres URL. "
            "Required for secret-safe CI/cloud use; no variable is read implicitly."
        ),
    )
    return parser


def _resolve_database_url(args: argparse.Namespace) -> tuple[str | None, str | None]:
    direct = _text(args.database_url)
    env_name = _text(args.database_url_env)
    if direct and env_name:
        return None, "choose only one of --database-url or --database-url-env"
    if env_name:
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_name):
            return None, "--database-url-env must be a valid environment variable name"
        value = _text(os.environ.get(env_name))
        if not value:
            return None, f"environment variable {env_name} is not set"
        return value, None
    if direct:
        return direct, None
    return None, "--apply requires --database-url or --database-url-env"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_plan(
        args.prod_dir, include_run_feature_files=not args.skip_run_feature_files,
    )
    if not args.apply:
        print(canonical_json(plan.summary()))
        return 0
    database_url, url_error = _resolve_database_url(args)
    if url_error:
        print(url_error, file=sys.stderr)
        return 2

    import psycopg  # Imported only for an explicit mutation request.

    with psycopg.connect(database_url) as connection:
        counts, parity = apply_plan(connection, plan)
    print(canonical_json({
        "mode": "applied", "import_batch_id": plan.import_batch_id,
        "row_counts": counts, "parity": parity,
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
