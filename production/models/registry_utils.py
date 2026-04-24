#!/usr/bin/env python3
"""Helpers for loading and validating the production model registry."""

from __future__ import annotations

import json
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

    return issues
