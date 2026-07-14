"""Shared structural contract for persisted model feature vectors.

This module is deliberately stdlib-only so live logging, historical replay,
database import, and cloud writers can validate the same ordered vector without
loading a model framework. Formula semantics are versioned separately.
"""

from __future__ import annotations

from hashlib import sha256
import json
import math
from typing import Any, Mapping, Sequence


def normalize_feature_vector(
    features: Mapping[str, Any], ordered_names: Sequence[str]
) -> tuple[dict[str, float], tuple[str, ...]]:
    """Return an ordered finite numeric vector and structural issues."""
    missing = [name for name in ordered_names if name not in features]
    if missing:
        return {}, (f"missing:{','.join(missing)}",)

    numeric: dict[str, float] = {}
    for name in ordered_names:
        try:
            value = float(features[name])
        except (TypeError, ValueError, OverflowError):
            return numeric, (f"nonnumeric:{name}",)
        if not math.isfinite(value):
            return numeric, (f"nonfinite:{name}",)
        numeric[name] = value

    groups = {
        "surface": [
            name for name in ordered_names
            if name.startswith("Surface_") and name != "Surface_Transition_Flag"
        ],
        "level": [name for name in ordered_names if name.startswith("Level_")],
        "round": [name for name in ordered_names if name.startswith("Round_")],
        "p1_hand": [name for name in ordered_names if name.startswith("P1_Hand_")],
        "p2_hand": [name for name in ordered_names if name.startswith("P2_Hand_")],
        "p1_country": [
            name for name in ordered_names if name.startswith("P1_Country_")
        ],
        "p2_country": [
            name for name in ordered_names if name.startswith("P2_Country_")
        ],
        "hand_matchup": [
            name for name in ordered_names
            if name.startswith("Handedness_Matchup_")
        ],
    }
    issues: list[str] = []
    for label, names in groups.items():
        values = [numeric[name] for name in names]
        if any(value not in {0.0, 1.0} for value in values):
            issues.append(f"one_hot_nonbinary:{label}")
            continue
        total = sum(values)
        if label == "hand_matchup":
            if total not in {0.0, 1.0}:
                issues.append(f"one_hot_cardinality:{label}:{total:g}")
        elif total != 1.0:
            issues.append(f"one_hot_cardinality:{label}:{total:g}")
    return numeric, tuple(issues)


def vector_sha256(vector: Mapping[str, float], ordered_names: Sequence[str]) -> str:
    payload = json.dumps(
        [[name, float(vector[name])] for name in ordered_names],
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def feature_fingerprint(
    features: Mapping[str, Any], ordered_names: Sequence[str]
) -> tuple[str, str, int]:
    """Return ordered schema hash, verified vector hash, and expected count."""
    names = tuple(ordered_names)
    schema_hash = sha256("\x1f".join(names).encode("utf-8")).hexdigest()
    vector, issues = normalize_feature_vector(features, names)
    return (
        schema_hash,
        "" if issues else vector_sha256(vector, names),
        len(names),
    )
