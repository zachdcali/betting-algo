from pathlib import Path
import sys
from datetime import datetime

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "production"))

from models.inference import EXACT_141_FEATURES  # noqa: E402
from models.registry_utils import get_current_version, get_model_entry, load_registry  # noqa: E402
from versioning import (  # noqa: E402
    FEATURE_SCHEMA_ID,
    FEATURE_SCHEMA_SHA256,
    SHARED_SEMANTICS_CANDIDATE_ID,
    VersionedId,
    artifact_directory,
    ordered_schema_sha256,
    validate_semver,
)


def test_live_feature_schema_identity_matches_ordered_141_contract():
    assert FEATURE_SCHEMA_ID == "base_141@1.0.0"
    assert len(EXACT_141_FEATURES) == 141
    assert ordered_schema_sha256(EXACT_141_FEATURES) == FEATURE_SCHEMA_SHA256


def test_shared_semantics_candidate_is_explicitly_versioned():
    parsed = VersionedId.parse(SHARED_SEMANTICS_CANDIDATE_ID)
    assert parsed.name == "base_141_shared"
    assert parsed.version == "1.0.0"


def test_new_artifacts_use_family_and_semver_directories():
    assert artifact_directory("candidates", "xgboost", "v2.0.0") == Path(
        "candidates/xgboost/v2.0.0"
    )


def test_all_promoted_families_bind_artifacts_to_declared_feature_contracts():
    registry = load_registry()
    schemas = registry["feature_contracts"]["schemas"]
    semantics = registry["feature_contracts"]["semantics"]
    assert registry["registry_schema_version"] == "2.0.0"
    assert isinstance(registry["registry_generation"], int)
    assert registry["registry_generation"] > 0
    effective_at = datetime.fromisoformat(
        registry["registry_effective_at"].replace("Z", "+00:00")
    )
    assert effective_at.tzinfo is not None
    for family in ("nn", "xgboost", "random_forest"):
        version = get_current_version(family, registry)
        entry = get_model_entry(family, version, registry)
        schema_id = entry["feature_schema_id"]
        assert schema_id in schemas
        assert entry["feature_schema_sha256"] == schemas[schema_id]["schema_sha256"]
        assert entry["training_feature_semantics_id"] in semantics
        assert entry["live_feature_semantics_id"] in semantics
        assert entry["training_dataset_id"]


@pytest.mark.parametrize("value", ["v2", "2", "2.0", "final", "2.0.0_v2"])
def test_ad_hoc_versions_are_rejected(value):
    with pytest.raises(ValueError):
        validate_semver(value)
