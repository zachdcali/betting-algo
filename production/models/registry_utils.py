#!/usr/bin/env python3
"""Helpers for loading and validating the production model registry."""

from __future__ import annotations

import json
import hashlib
import pickle
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
    registry = registry or load_registry()
    issues: Dict[str, list[str]] = {"missing": [], "invalid": []}

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
                should_require_artifact = (
                    artifact_scope == "all"
                    or (bucket_name == "models" and version == current_version)
                )
                if not should_require_artifact:
                    continue
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
