#!/usr/bin/env python3
"""Helpers for loading and validating the production model registry."""

from __future__ import annotations

import json
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

try:
    from versioning import ordered_schema_sha256, validate_semver
except ModuleNotFoundError:  # pragma: no cover - package import path
    from production.versioning import ordered_schema_sha256, validate_semver

REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
REPO_ROOT = REGISTRY_PATH.parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "professional_tennis"
FEATURE_SCHEMA_PATH = REPO_ROOT / "production" / "features" / "schema_141.json"

FAMILY_DIRS = {
    "nn": "Neural_Network",
    "xgboost": "XGBoost",
    "random_forest": "Random_Forest",
}

PROBABILITY_MODES = frozenset({"raw", "calibrated"})


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_feature_count(artifact: object) -> Optional[int]:
    """Return the declared input width for a loaded sklearn-style artifact.

    CalibratedClassifierCV has changed where it exposes the fitted estimator
    across sklearn releases.  Walk only the small, known estimator/scaler
    graph so validation can prove the calibrated wrapper still consumes the
    registry's ordered feature vector.
    """
    pending = [artifact]
    seen: set[int] = set()
    while pending:
        candidate = pending.pop(0)
        if candidate is None or id(candidate) in seen:
            continue
        seen.add(id(candidate))

        feature_count = getattr(candidate, "n_features_in_", None)
        try:
            if feature_count is not None:
                return int(feature_count)
        except (TypeError, ValueError):
            pass

        for attribute in ("scaler", "estimator", "base_estimator"):
            nested = getattr(candidate, attribute, None)
            if nested is not None:
                pending.append(nested)
        for calibrated in getattr(candidate, "calibrated_classifiers_", ()) or ():
            pending.append(calibrated)
    return None


def ordered_feature_names(expected_sha256: str) -> tuple[str, ...]:
    """Load and hash-check the canonical ordered base_141 schema."""
    if not str(expected_sha256 or "").strip():
        raise ValueError("feature schema SHA-256 is required")
    with open(FEATURE_SCHEMA_PATH, encoding="utf-8") as fh:
        payload = json.load(fh)
    raw_features = payload.get("features")
    if not isinstance(raw_features, list):
        raise ValueError("base_141 schema features must be a list")
    ordered = sorted(raw_features, key=lambda item: int(item["index"]))
    indexes = [int(item["index"]) for item in ordered]
    if indexes != list(range(len(ordered))):
        raise ValueError("base_141 schema indexes must be contiguous")
    names = tuple(str(item["name"]) for item in ordered)
    if int(payload.get("total_count") or 0) != len(names):
        raise ValueError("base_141 schema count does not match its feature list")
    actual_sha256 = ordered_schema_sha256(names)
    if actual_sha256 != str(expected_sha256).lower():
        raise ValueError(
            "base_141 schema checksum mismatch "
            f"expected={str(expected_sha256).lower()} actual={actual_sha256}"
        )
    return names


def load_registry() -> Dict:
    if not REGISTRY_PATH.exists():
        return {}
    with open(REGISTRY_PATH) as f:
        return json.load(f)


def _family_block(registry: Dict, family: str) -> Dict:
    if family == "nn":
        return registry
    return registry.get(family, {})


def get_current_version(family: str, registry: Optional[Dict] = None) -> str:
    if registry is None:
        registry = load_registry()
    block = _family_block(registry, family)
    return block.get("current_version", "unknown")


def get_candidate_version(family: str, registry: Optional[Dict] = None) -> Optional[str]:
    if registry is None:
        registry = load_registry()
    block = _family_block(registry, family)
    return block.get("candidate_version")


def get_model_entry(
    family: str,
    version: Optional[str] = None,
    registry: Optional[Dict] = None,
    include_candidates: bool = False,
) -> Dict:
    if registry is None:
        registry = load_registry()
    block = _family_block(registry, family)
    version = version or block.get("current_version")
    entry = block.get("models", {}).get(version)
    if entry is None and include_candidates:
        entry = block.get("candidates", {}).get(version)
    return entry or {}


def resolve_artifact_path(
    family: str,
    artifact_key: str = "model_file",
    version: Optional[str] = None,
    registry: Optional[Dict] = None,
    include_candidates: bool = False,
) -> Optional[Path]:
    if registry is None:
        registry = load_registry()
    entry = get_model_entry(
        family,
        version=version,
        registry=registry,
        include_candidates=include_candidates,
    )
    rel_path = entry.get(artifact_key)
    if not rel_path:
        return None
    return RESULTS_ROOT / FAMILY_DIRS[family] / rel_path


def validate_registry(
    registry: Optional[Dict] = None,
    *,
    artifact_scope: str = "promoted",
) -> Dict[str, list[str]]:
    """Validate registry structure and the requested artifact inventory.

    Cloud inference packages only the promoted artifacts, so its fail-closed
    check uses ``artifact_scope="promoted"``.  Maintainers can request
    ``artifact_scope="all"`` when auditing the complete local archive and
    candidate inventory.
    """
    if artifact_scope not in {"promoted", "all"}:
        raise ValueError(f"unknown artifact scope: {artifact_scope!r}")
    if registry is None:
        registry = load_registry()
    issues: Dict[str, list[str]] = {"missing": [], "invalid": []}
    feature_contracts = registry.get("feature_contracts", {})
    feature_schemas = feature_contracts.get("schemas", {})
    feature_semantics = feature_contracts.get("semantics", {})

    generation = registry.get("registry_generation")
    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 1:
        issues["invalid"].append(
            "registry_generation must be a positive integer"
        )
    effective_at = registry.get("registry_effective_at")
    try:
        parsed_effective_at = datetime.fromisoformat(
            str(effective_at or "").replace("Z", "+00:00")
        )
        if parsed_effective_at.tzinfo is None or parsed_effective_at.utcoffset() is None:
            raise ValueError("timezone required")
    except ValueError:
        issues["invalid"].append(
            "registry_effective_at must be an ISO-8601 timestamp with timezone"
        )
    try:
        validate_semver(registry.get("registry_schema_version"))
    except ValueError:
        issues["invalid"].append(
            "registry_schema_version must be a semantic version"
        )

    for family in FAMILY_DIRS:
        block = _family_block(registry, family)
        current_version = block.get("current_version")
        if not str(current_version or "").strip():
            issues["invalid"].append(
                f"{family}: current_version is required for promoted inference"
            )
        if current_version:
            try:
                validate_semver(current_version)
            except ValueError:
                issues["invalid"].append(
                    f"{family}: current_version must be a semantic version"
                )
        if current_version and current_version not in block.get("models", {}):
            issues["invalid"].append(
                f"{family}: current_version={current_version} missing from models"
            )
        candidate_version = block.get("candidate_version")
        if candidate_version:
            try:
                validate_semver(candidate_version)
            except ValueError:
                issues["invalid"].append(
                    f"{family}: candidate_version must be a semantic version"
                )
        if candidate_version and candidate_version not in block.get("candidates", {}):
            issues["invalid"].append(
                f"{family}: candidate_version={candidate_version} missing from candidates"
            )

        for bucket_name in ("models", "candidates"):
            for version, entry in block.get(bucket_name, {}).items():
                try:
                    validate_semver(version)
                except ValueError:
                    issues["invalid"].append(
                        f"{family}:{version} release key must be a semantic version"
                    )
                probability_mode = str(
                    entry.get("probability_mode") or "raw"
                ).strip().lower()
                calibration_version = str(
                    entry.get("calibration_version") or ""
                ).strip()
                if calibration_version:
                    try:
                        validate_semver(calibration_version)
                    except ValueError:
                        issues["invalid"].append(
                            f"{family}:{version} calibration_version must be a "
                            "semantic version"
                        )
                if probability_mode not in PROBABILITY_MODES:
                    issues["invalid"].append(
                        f"{family}:{version} probability_mode must be one of "
                        f"{sorted(PROBABILITY_MODES)}"
                    )
                elif family != "nn" and probability_mode != "raw":
                    issues["invalid"].append(
                        f"{family}:{version} live inference supports only "
                        "probability_mode='raw'"
                    )
                should_require_artifact = (
                    artifact_scope == "all"
                    or (bucket_name == "models" and version == current_version)
                )
                if not should_require_artifact:
                    continue
                if (
                    family == "nn"
                    and bucket_name == "models"
                    and version == current_version
                    and probability_mode == "calibrated"
                ):
                    issues["invalid"].append(
                        f"{family}:{version} calibrated live promotion is blocked "
                        "until nn_calibration_version is persisted in prediction "
                        "and snapshot lineage"
                    )
                calibrated_fields_complete = True
                if family == "nn" and probability_mode == "calibrated":
                    for field in (
                        "calibrated_model_file",
                        "calibrated_model_sha256",
                        "calibration_version",
                    ):
                        if not str(entry.get(field) or "").strip():
                            calibrated_fields_complete = False
                            issues["invalid"].append(
                                f"{family}:{version} calibrated probability mode "
                                f"requires {field}"
                            )
                model_path = resolve_artifact_path(
                    family,
                    "model_file",
                    version=version,
                    registry=registry,
                    include_candidates=True,
                )
                if entry.get("artifact_available", True) and model_path and not model_path.exists():
                    issues["missing"].append(f"{family}:{version} missing model file {model_path}")

                scaler_file = entry.get("scaler_file")
                if scaler_file:
                    scaler_path = resolve_artifact_path(
                        family,
                        "scaler_file",
                        version=version,
                        registry=registry,
                        include_candidates=True,
                    )
                    if scaler_path and not scaler_path.exists():
                        issues["missing"].append(f"{family}:{version} missing scaler file {scaler_path}")

                if (
                    family == "nn"
                    and probability_mode == "calibrated"
                    and calibrated_fields_complete
                ):
                    calibrated_path = resolve_artifact_path(
                        family,
                        "calibrated_model_file",
                        version=version,
                        registry=registry,
                        include_candidates=(bucket_name == "candidates"),
                    )
                    if calibrated_path is None:
                        issues["missing"].append(
                            f"{family}:{version} calibrated artifact path is unresolved"
                        )
                    elif not calibrated_path.exists():
                        issues["missing"].append(
                            f"{family}:{version} missing calibrated artifact "
                            f"{calibrated_path}"
                        )
                    else:
                        expected_hash = str(
                            entry.get("calibrated_model_sha256") or ""
                        ).lower()
                        actual_hash = _sha256_file(calibrated_path)
                        if actual_hash != expected_hash:
                            issues["invalid"].append(
                                f"{family}:{version} calibrated artifact checksum "
                                f"mismatch expected={expected_hash} actual={actual_hash}"
                            )
                        try:
                            with open(calibrated_path, "rb") as fh:
                                calibrated_model = pickle.load(fh)
                            if not callable(
                                getattr(calibrated_model, "predict_proba", None)
                            ):
                                raise ValueError("predict_proba is missing")
                            calibrated_features = artifact_feature_count(
                                calibrated_model
                            )
                            expected_features = int(entry.get("features") or 0)
                            if calibrated_features != expected_features:
                                raise ValueError(
                                    "calibrated artifact features="
                                    f"{calibrated_features!r} expected="
                                    f"{expected_features}"
                                )
                        except Exception as exc:
                            issues["invalid"].append(
                                f"{family}:{version} calibrated artifact "
                                f"load/schema validation failed: {exc}"
                            )

        # Promoted artifacts are execution inputs, not merely paths. Pin their
        # bytes and prove they deserialize with the registry's feature count.
        if not current_version:
            continue
        entry = block.get("models", {}).get(current_version, {})
        if (
            "artifact_available" in entry
            and entry.get("artifact_available") is not True
        ):
            issues["invalid"].append(
                f"{family}:{current_version} promoted artifact_available must "
                "be absent or boolean true"
            )
        required_artifact_fields = ["model_file", "model_sha256"]
        if family == "nn":
            required_artifact_fields.extend(("scaler_file", "scaler_sha256"))
        for field in required_artifact_fields:
            if not str(entry.get(field) or "").strip():
                issues["invalid"].append(
                    f"{family}:{current_version} missing promoted {field}"
                )
        raw_expected_features = entry.get("features")
        if (
            isinstance(raw_expected_features, bool)
            or not isinstance(raw_expected_features, int)
            or raw_expected_features < 1
        ):
            issues["invalid"].append(
                f"{family}:{current_version} features must be a positive integer"
            )
            expected_features = 0
        else:
            expected_features = raw_expected_features
        required_contract_fields = (
            "feature_schema_id",
            "feature_schema_sha256",
            "training_feature_semantics_id",
            "live_feature_semantics_id",
            "training_dataset_id",
        )
        for field in required_contract_fields:
            if not str(entry.get(field) or "").strip():
                issues["invalid"].append(
                    f"{family}:{current_version} missing {field}"
                )
        feature_schema_id = str(entry.get("feature_schema_id") or "")
        declared_schema = feature_schemas.get(feature_schema_id, {})
        if feature_schema_id and not declared_schema:
            issues["invalid"].append(
                f"{family}:{current_version} unknown feature_schema_id={feature_schema_id}"
            )
        elif declared_schema:
            declared_hash = str(declared_schema.get("schema_sha256") or "").lower()
            entry_hash = str(entry.get("feature_schema_sha256") or "").lower()
            if declared_hash != entry_hash:
                issues["invalid"].append(
                    f"{family}:{current_version} feature schema checksum mismatch"
                )
            if int(declared_schema.get("feature_count") or 0) != expected_features:
                issues["invalid"].append(
                    f"{family}:{current_version} feature count does not match schema"
                )
        for field in ("training_feature_semantics_id", "live_feature_semantics_id"):
            semantics_id = str(entry.get(field) or "")
            if semantics_id and semantics_id not in feature_semantics:
                issues["invalid"].append(
                    f"{family}:{current_version} unknown {field}={semantics_id}"
                )
        model_path = resolve_artifact_path(
            family, "model_file", version=current_version, registry=registry
        )
        if model_path is None:
            issues["missing"].append(
                f"{family}:{current_version} promoted model path is unresolved"
            )
        elif not model_path.exists():
            issues["missing"].append(
                f"{family}:{current_version} missing promoted model file {model_path}"
            )
        else:
            expected_hash = str(entry.get("model_sha256") or "").lower()
            if expected_hash:
                actual_hash = _sha256_file(model_path)
                if actual_hash != expected_hash:
                    issues["invalid"].append(
                        f"{family}:{current_version} model checksum mismatch "
                        f"expected={expected_hash} actual={actual_hash}"
                    )
            try:
                if family == "nn":
                    import torch
                    state = torch.load(model_path, map_location="cpu", weights_only=True)
                    matrices = [value for value in state.values() if getattr(value, "ndim", 0) == 2]
                    if not matrices or int(matrices[0].shape[1]) != expected_features:
                        raise ValueError(
                            f"first layer input={matrices[0].shape[1] if matrices else 'missing'}"
                        )
                elif family == "xgboost":
                    import xgboost as xgb
                    model = xgb.XGBClassifier()
                    model.load_model(str(model_path))
                    if int(model.n_features_in_) != expected_features:
                        raise ValueError(f"model features={model.n_features_in_}")
                    expected_names = ordered_feature_names(
                        str(entry.get("feature_schema_sha256") or "")
                    )
                    artifact_names = tuple(
                        str(name) for name in model.feature_names_in_
                    )
                    if artifact_names != expected_names:
                        raise ValueError(
                            "model feature order does not match base_141"
                        )
                elif family == "random_forest":
                    with open(model_path, "rb") as fh:
                        model = pickle.load(fh)
                    if int(model.n_features_in_) != expected_features:
                        raise ValueError(f"model features={model.n_features_in_}")
                    expected_names = ordered_feature_names(
                        str(entry.get("feature_schema_sha256") or "")
                    )
                    artifact_names = tuple(
                        str(name) for name in model.feature_names_in_
                    )
                    if artifact_names != expected_names:
                        raise ValueError(
                            "model feature order does not match base_141"
                        )
            except Exception as exc:
                issues["invalid"].append(
                    f"{family}:{current_version} model load/schema validation failed: {exc}"
                )

        scaler_path = resolve_artifact_path(
            family, "scaler_file", version=current_version, registry=registry
        )
        if family == "nn" and scaler_path is None:
            issues["missing"].append(
                f"{family}:{current_version} promoted scaler path is unresolved"
            )
        elif family == "nn" and not scaler_path.exists():
            issues["missing"].append(
                f"{family}:{current_version} missing promoted scaler file {scaler_path}"
            )
        elif scaler_path and scaler_path.exists():
            expected_hash = str(entry.get("scaler_sha256") or "").lower()
            if expected_hash:
                actual_hash = _sha256_file(scaler_path)
                if actual_hash != expected_hash:
                    issues["invalid"].append(
                        f"{family}:{current_version} scaler checksum mismatch "
                        f"expected={expected_hash} actual={actual_hash}"
                    )
            try:
                with open(scaler_path, "rb") as fh:
                    scaler = pickle.load(fh)
                if int(scaler.n_features_in_) != expected_features:
                    raise ValueError(f"scaler features={scaler.n_features_in_}")
            except Exception as exc:
                issues["invalid"].append(
                    f"{family}:{current_version} scaler load/schema validation failed: {exc}"
                )

    return issues
