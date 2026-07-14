"""Stable identifiers for models, feature contracts, datasets, and storage.

Versions are deliberately kept in metadata instead of being embedded in
descriptive filenames such as ``*_v2_final``.  Existing promoted artifacts keep
their historical paths; new artifacts use family/version directories and bind
to an explicit feature schema plus feature-semantics contract in the registry.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re


SEMVER_RE = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)"
    r"(?:-([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?"
    r"(?:\+([0-9A-Za-z-]+(?:\.[0-9A-Za-z-]+)*))?$"
)
FAMILY_NAMES = {"nn", "xgboost", "random_forest"}
ARTIFACT_ROLES = {"releases", "candidates", "archive"}

OPERATIONAL_SCHEMA_VERSION = "1.1.0"
OPERATIONAL_NORMALIZER_VERSION = "1.0.0"
DATASET_MANIFEST_VERSION = "1.0.0"
LOGGING_SCHEMA_VERSION = "prediction_log_v2"

# The ordered 141-column representation is stable even though the historical
# and live formula implementations are not yet semantically equivalent.
FEATURE_SCHEMA_NAME = "base_141"
FEATURE_SCHEMA_VERSION = "1.0.0"
FEATURE_SCHEMA_SHA256 = "17a33325776292ad31e4bbaff81cb223355f026aa019b8b516a0281945930b4d"

# These IDs make the current mismatch explicit.  The shared contract is
# reserved for the parity-tested kernel and is not active production semantics.
HISTORICAL_SEMANTICS_ID = "sackmann_historical_legacy@1.0.0"
LIVE_SEMANTICS_ID = "ta_live_legacy@3.0.0"
SHARED_SEMANTICS_CANDIDATE_ID = "base_141_shared@1.0.0"


def validate_semver(version: str) -> str:
    """Return normalized SemVer without a leading ``v`` or raise ValueError."""
    value = str(version or "").strip()
    if value.startswith("v"):
        value = value[1:]
    if not SEMVER_RE.fullmatch(value):
        raise ValueError(f"invalid semantic version: {version!r}")
    return value


@dataclass(frozen=True)
class VersionedId:
    name: str
    version: str

    def __post_init__(self) -> None:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", self.name):
            raise ValueError(f"invalid versioned identifier name: {self.name!r}")
        object.__setattr__(self, "version", validate_semver(self.version))

    @property
    def id(self) -> str:
        return f"{self.name}@{self.version}"

    @classmethod
    def parse(cls, value: str) -> "VersionedId":
        name, separator, version = str(value or "").partition("@")
        if not separator or not name or not version:
            raise ValueError(f"expected name@semver, got {value!r}")
        return cls(name=name, version=version)


FEATURE_SCHEMA_ID = VersionedId(FEATURE_SCHEMA_NAME, FEATURE_SCHEMA_VERSION).id


def ordered_schema_sha256(feature_names: list[str] | tuple[str, ...]) -> str:
    """Hash ordered feature names using the production lineage encoding."""
    return sha256("\x1f".join(feature_names).encode("utf-8")).hexdigest()


def artifact_directory(role: str, family: str, model_version: str) -> Path:
    """Return the standard directory for a new model artifact release.

    The filename inside this directory is format-oriented (``model.json``,
    ``model.pkl``, ``model.pth``, ``scaler.pkl``); identity comes from registry
    metadata and the directory, not an accumulating filename suffix.
    """
    if role not in ARTIFACT_ROLES:
        raise ValueError(f"unknown artifact role: {role!r}")
    if family not in FAMILY_NAMES:
        raise ValueError(f"unknown model family: {family!r}")
    normalized = validate_semver(model_version)
    return Path(role) / family / f"v{normalized}"
