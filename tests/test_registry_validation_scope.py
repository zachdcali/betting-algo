from pathlib import Path
from copy import deepcopy
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from models.registry_utils import load_registry, validate_registry  # noqa: E402


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


@pytest.mark.parametrize("family", ("nn", "xgboost", "random_forest"))
def test_promoted_artifact_metadata_cannot_be_removed_or_marked_unavailable(
    family, monkeypatch
):
    registry = deepcopy(load_registry())
    monkeypatch.setattr(
        "models.registry_utils.resolve_artifact_path",
        lambda *args, **kwargs: None,
    )
    block = registry if family == "nn" else registry[family]
    version = block["current_version"]
    entry = block["models"][version]
    entry["artifact_available"] = False
    entry.pop("model_file", None)
    entry.pop("model_sha256", None)
    if family == "nn":
        entry.pop("scaler_file", None)
        entry.pop("scaler_sha256", None)

    issues = validate_registry(registry, artifact_scope="promoted")
    combined = "\n".join([*issues["invalid"], *issues["missing"]])

    assert "promoted artifact cannot be unavailable" in combined
    assert "missing promoted model_file" in combined
    assert "missing promoted model_sha256" in combined
    assert "promoted model path is unresolved" in combined
    if family == "nn":
        assert "missing promoted scaler_file" in combined
        assert "missing promoted scaler_sha256" in combined
        assert "promoted scaler path is unresolved" in combined
