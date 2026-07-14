from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from models.registry_utils import validate_registry  # noqa: E402


def _registry_with_unshipped_archive() -> dict:
    return {
        "models": {
            "v0.9.0": {
                "model_file": "archive/missing-model.pth",
                "scaler_file": "archive/missing-scaler.pkl",
            },
        },
        "xgboost": {},
        "random_forest": {},
    }


def test_promoted_scope_does_not_require_unshipped_archive_files():
    issues = validate_registry(
        _registry_with_unshipped_archive(), artifact_scope="promoted"
    )

    assert issues["missing"] == []


def test_all_scope_reports_unshipped_archive_files():
    issues = validate_registry(
        _registry_with_unshipped_archive(), artifact_scope="all"
    )

    assert len(issues["missing"]) == 2
    assert all("nn:v0.9.0 missing" in issue for issue in issues["missing"])


def test_unknown_artifact_scope_fails_closed():
    with pytest.raises(ValueError, match="unknown artifact scope"):
        validate_registry({}, artifact_scope="current-ish")
