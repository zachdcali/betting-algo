from pathlib import Path
from copy import deepcopy
from hashlib import sha256
import pickle
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from models import registry_utils  # noqa: E402
from models.nn_runtime import NNWrapper  # noqa: E402
from models.registry_utils import load_registry, validate_registry  # noqa: E402


class _CalibratedArtifact:
    def __init__(self, feature_count: int):
        self.n_features_in_ = feature_count

    def predict_proba(self, values):
        return values


class _TreeArtifact:
    def __init__(self, feature_names):
        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)

    def predict_proba(self, values):
        return values


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


def test_explicit_empty_registry_fails_closed_without_loading_disk_registry():
    issues = validate_registry({}, artifact_scope="promoted")
    combined = "\n".join(issues["invalid"])

    assert "registry_generation must be a positive integer" in combined
    assert "registry_schema_version must be a semantic version" in combined
    for family in ("nn", "xgboost", "random_forest"):
        assert (
            f"{family}: current_version is required for promoted inference"
            in combined
        )


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

    assert "promoted artifact_available must be absent or boolean true" in combined
    assert "missing promoted model_file" in combined
    assert "missing promoted model_sha256" in combined
    assert "promoted model path is unresolved" in combined
    if family == "nn":
        assert "missing promoted scaler_file" in combined
        assert "missing promoted scaler_sha256" in combined
        assert "promoted scaler path is unresolved" in combined


@pytest.mark.parametrize("value", (False, "false", 1))
def test_promoted_artifact_availability_requires_boolean_true(value):
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    registry["models"][version]["artifact_available"] = value

    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        "promoted artifact_available must be absent or boolean true" in issue
        for issue in issues["invalid"]
    )


@pytest.mark.parametrize("value", (True, 141.0, 0, -1))
def test_promoted_feature_count_requires_positive_integer(value):
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    registry["models"][version]["features"] = value

    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        "features must be a positive integer" in issue
        for issue in issues["invalid"]
    )


@pytest.mark.parametrize("family", ("nn", "xgboost", "random_forest"))
def test_probability_mode_rejects_unknown_values(family):
    registry = deepcopy(load_registry())
    block = registry if family == "nn" else registry[family]
    version = block["current_version"]
    block["models"][version]["probability_mode"] = "best_available"

    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        f"{family}:{version} probability_mode must be one of" in issue
        for issue in issues["invalid"]
    )


def test_calibrated_nn_requires_pinned_artifact_metadata():
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    entry = registry["models"][version]
    entry["probability_mode"] = "calibrated"
    entry.pop("calibrated_model_file", None)
    entry.pop("calibrated_model_sha256", None)

    issues = validate_registry(registry, artifact_scope="promoted")
    combined = "\n".join(issues["invalid"])

    assert "requires calibrated_model_file" in combined
    assert "requires calibrated_model_sha256" in combined
    assert "requires calibration_version" in combined


def test_promoted_scope_ignores_unshipped_calibrated_candidate_artifacts():
    registry = deepcopy(load_registry())
    version = registry["candidate_version"]
    entry = registry["candidates"][version]
    entry["probability_mode"] = "calibrated"
    entry["calibration_version"] = "v1.0.0"
    entry.pop("calibrated_model_file", None)
    entry.pop("calibrated_model_sha256", None)

    promoted_issues = validate_registry(registry, artifact_scope="promoted")
    all_issues = validate_registry(registry, artifact_scope="all")

    assert not any(
        f"nn:{version} calibrated probability mode requires" in issue
        for issue in promoted_issues["invalid"]
    )
    assert any(
        f"nn:{version} calibrated probability mode requires" in issue
        for issue in all_issues["invalid"]
    )


@pytest.mark.parametrize("family", ("xgboost", "random_forest"))
def test_non_nn_calibrated_probability_mode_is_rejected(family):
    registry = deepcopy(load_registry())
    block = registry[family]
    version = block["current_version"]
    block["models"][version]["probability_mode"] = "calibrated"

    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        f"{family}:{version} live inference supports only " in issue
        for issue in issues["invalid"]
    )


def _validate_with_calibrated_path(
    registry: dict, calibrated_path: Path, monkeypatch
) -> dict[str, list[str]]:
    original_resolver = registry_utils.resolve_artifact_path

    def _resolve(family, artifact_key="model_file", **kwargs):
        if family == "nn" and artifact_key == "calibrated_model_file":
            return calibrated_path
        return original_resolver(family, artifact_key, **kwargs)

    monkeypatch.setattr(registry_utils, "resolve_artifact_path", _resolve)
    return validate_registry(registry, artifact_scope="promoted")


def test_calibrated_nn_artifact_hash_load_and_feature_count_validate(
    tmp_path, monkeypatch
):
    calibrated_path = tmp_path / "calibrated.pkl"
    calibrated_path.write_bytes(pickle.dumps(_CalibratedArtifact(141)))
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    entry = registry["models"][version]
    entry["probability_mode"] = "calibrated"
    entry["calibrated_model_file"] = calibrated_path.name
    entry["calibrated_model_sha256"] = sha256(
        calibrated_path.read_bytes()
    ).hexdigest()
    entry["calibration_version"] = "v1.0.0"

    issues = _validate_with_calibrated_path(registry, calibrated_path, monkeypatch)

    assert not any("calibrated artifact" in issue for issue in issues["missing"])
    assert not any("calibrated artifact" in issue for issue in issues["invalid"])
    assert any(
        "calibrated live promotion is blocked" in issue
        for issue in issues["invalid"]
    )


def test_calibrated_nn_artifact_rejects_wrong_hash_and_feature_count(
    tmp_path, monkeypatch
):
    calibrated_path = tmp_path / "calibrated.pkl"
    calibrated_path.write_bytes(pickle.dumps(_CalibratedArtifact(140)))
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    entry = registry["models"][version]
    entry["probability_mode"] = "calibrated"
    entry["calibrated_model_file"] = calibrated_path.name
    entry["calibrated_model_sha256"] = "0" * 64
    entry["calibration_version"] = "v1.0.0"

    issues = _validate_with_calibrated_path(registry, calibrated_path, monkeypatch)
    combined = "\n".join(issues["invalid"])

    assert "calibrated artifact checksum mismatch" in combined
    assert "calibrated artifact features=140 expected=141" in combined


def test_calibrated_nn_artifact_rejects_corrupt_pickle(tmp_path, monkeypatch):
    calibrated_path = tmp_path / "calibrated.pkl"
    calibrated_path.write_bytes(b"not-a-pickle")
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    entry = registry["models"][version]
    entry["probability_mode"] = "calibrated"
    entry["calibrated_model_file"] = calibrated_path.name
    entry["calibrated_model_sha256"] = sha256(
        calibrated_path.read_bytes()
    ).hexdigest()
    entry["calibration_version"] = "v1.0.0"

    issues = _validate_with_calibrated_path(registry, calibrated_path, monkeypatch)

    assert any(
        "calibrated artifact load/schema validation failed" in issue
        for issue in issues["invalid"]
    )


@pytest.mark.parametrize(
    "mutator,expected",
    (
        (
            lambda registry: registry.__setitem__(
                "registry_schema_version", "final"
            ),
            "registry_schema_version must be a semantic version",
        ),
        (
            lambda registry: registry.__setitem__("current_version", "final"),
            "nn: current_version must be a semantic version",
        ),
        (
            lambda registry: registry.__setitem__("candidate_version", "next"),
            "nn: candidate_version must be a semantic version",
        ),
    ),
)
def test_registry_pointer_versions_require_semver(mutator, expected):
    registry = deepcopy(load_registry())
    mutator(registry)

    issues = validate_registry(registry, artifact_scope="promoted")

    assert expected in issues["invalid"]


def test_calibration_version_requires_semver_when_present():
    registry = deepcopy(load_registry())
    version = registry["current_version"]
    registry["models"][version]["calibration_version"] = "final"

    issues = validate_registry(registry, artifact_scope="promoted")

    assert (
        f"nn:{version} calibration_version must be a semantic version"
        in issues["invalid"]
    )


def test_registry_release_keys_require_semver():
    registry = deepcopy(load_registry())
    entry = registry["models"].pop(registry["current_version"])
    registry["current_version"] = "final"
    registry["models"]["final"] = entry

    issues = validate_registry(registry, artifact_scope="promoted")

    assert "nn:final release key must be a semantic version" in issues["invalid"]


@pytest.mark.parametrize("family", ("nn", "xgboost", "random_forest"))
def test_each_production_family_requires_a_current_version(family):
    registry = deepcopy(load_registry())
    block = registry if family == "nn" else registry[family]
    block.pop("current_version", None)

    issues = validate_registry(registry, artifact_scope="promoted")

    assert (
        f"{family}: current_version is required for promoted inference"
        in issues["invalid"]
    )


def test_validator_script_bootstrap_can_unpickle_models_module_artifact(tmp_path):
    script = ROOT / "production/models/validate_registry.py"
    artifact = tmp_path / "calibrated.pkl"
    artifact.write_bytes(
        pickle.dumps(NNWrapper(model=None, scaler=None, device="cpu"))
    )
    code = (
        "import pickle, runpy; "
        f"runpy.run_path({str(script)!r}, run_name='validator_bootstrap'); "
        f"pickle.load(open({str(artifact)!r}, 'rb'))"
    )

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_promoted_xgboost_rejects_swapped_feature_order(tmp_path, monkeypatch):
    registry = deepcopy(load_registry())
    version = registry["xgboost"]["current_version"]
    entry = registry["xgboost"]["models"][version]
    expected = list(registry_utils.ordered_feature_names(
        entry["feature_schema_sha256"]
    ))
    expected[0], expected[1] = expected[1], expected[0]
    artifact_path = tmp_path / "xgb.json"
    artifact_path.write_text("{}", encoding="utf-8")
    entry["model_sha256"] = sha256(artifact_path.read_bytes()).hexdigest()
    original_resolver = registry_utils.resolve_artifact_path

    def _resolve(family, artifact_key="model_file", **kwargs):
        if family == "xgboost" and artifact_key == "model_file":
            return artifact_path
        return original_resolver(family, artifact_key, **kwargs)

    class _WrongOrderXGBoost(_TreeArtifact):
        def __init__(self):
            super().__init__(expected)

        def load_model(self, path):
            return None

    import xgboost as xgb

    monkeypatch.setattr(xgb, "XGBClassifier", _WrongOrderXGBoost)
    monkeypatch.setattr(registry_utils, "resolve_artifact_path", _resolve)
    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        "xgboost:" in issue
        and "model feature order does not match base_141" in issue
        for issue in issues["invalid"]
    )


def test_promoted_random_forest_rejects_swapped_feature_order(
    tmp_path, monkeypatch
):
    registry = deepcopy(load_registry())
    version = registry["random_forest"]["current_version"]
    entry = registry["random_forest"]["models"][version]
    expected = list(registry_utils.ordered_feature_names(
        entry["feature_schema_sha256"]
    ))
    expected[0], expected[1] = expected[1], expected[0]
    artifact_path = tmp_path / "rf.pkl"
    artifact_path.write_bytes(pickle.dumps(_TreeArtifact(expected)))
    entry["model_sha256"] = sha256(artifact_path.read_bytes()).hexdigest()
    original_resolver = registry_utils.resolve_artifact_path

    def _resolve(family, artifact_key="model_file", **kwargs):
        if family == "random_forest" and artifact_key == "model_file":
            return artifact_path
        return original_resolver(family, artifact_key, **kwargs)

    monkeypatch.setattr(registry_utils, "resolve_artifact_path", _resolve)
    issues = validate_registry(registry, artifact_scope="promoted")

    assert any(
        "random_forest:" in issue
        and "model feature order does not match base_141" in issue
        for issue in issues["invalid"]
    )
