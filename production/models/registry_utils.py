#!/usr/bin/env python3
"""Helpers for loading and validating the production model registry."""

from __future__ import annotations

import json
import hashlib
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

REGISTRY_PATH = Path(__file__).with_name("model_registry.json")
REPO_ROOT = REGISTRY_PATH.parents[2]
RESULTS_ROOT = REPO_ROOT / "results" / "professional_tennis"

FAMILY_DIRS = {
    "nn": "Neural_Network",
    "xgboost": "XGBoost",
    "random_forest": "Random_Forest",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
    registry = registry or load_registry()
    block = _family_block(registry, family)
    return block.get("current_version", "unknown")


def get_candidate_version(family: str, registry: Optional[Dict] = None) -> Optional[str]:
    registry = registry or load_registry()
    block = _family_block(registry, family)
    return block.get("candidate_version")


def get_model_entry(
    family: str,
    version: Optional[str] = None,
    registry: Optional[Dict] = None,
    include_candidates: bool = False,
) -> Dict:
    registry = registry or load_registry()
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
    registry = registry or load_registry()
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


def validate_registry(registry: Optional[Dict] = None) -> Dict[str, list[str]]:
    registry = registry or load_registry()
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

    for family in FAMILY_DIRS:
        block = _family_block(registry, family)
        current_version = block.get("current_version")
        if current_version and current_version not in block.get("models", {}):
            issues["invalid"].append(
                f"{family}: current_version={current_version} missing from models"
            )
        candidate_version = block.get("candidate_version")
        if candidate_version and candidate_version not in block.get("candidates", {}):
            issues["invalid"].append(
                f"{family}: candidate_version={candidate_version} missing from candidates"
            )

        for bucket_name in ("models", "candidates"):
            for version, entry in block.get(bucket_name, {}).items():
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

        # Promoted artifacts are execution inputs, not merely paths. Pin their
        # bytes and prove they deserialize with the registry's feature count.
        if not current_version:
            continue
        entry = block.get("models", {}).get(current_version, {})
        expected_features = int(entry.get("features") or 0)
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
        if entry.get("artifact_available", True) and model_path and model_path.exists():
            expected_hash = str(entry.get("model_sha256") or "").lower()
            if not expected_hash:
                issues["invalid"].append(f"{family}:{current_version} missing model_sha256")
            else:
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
                elif family == "random_forest":
                    with open(model_path, "rb") as fh:
                        model = pickle.load(fh)
                    if int(model.n_features_in_) != expected_features:
                        raise ValueError(f"model features={model.n_features_in_}")
            except Exception as exc:
                issues["invalid"].append(
                    f"{family}:{current_version} model load/schema validation failed: {exc}"
                )

        scaler_path = resolve_artifact_path(
            family, "scaler_file", version=current_version, registry=registry
        )
        if scaler_path and scaler_path.exists():
            expected_hash = str(entry.get("scaler_sha256") or "").lower()
            if not expected_hash:
                issues["invalid"].append(f"{family}:{current_version} missing scaler_sha256")
            else:
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
