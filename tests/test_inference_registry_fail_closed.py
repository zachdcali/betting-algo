from pathlib import Path
from hashlib import sha256
import pickle
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from models import inference  # noqa: E402


class _WrongOrderTreeArtifact:
    def __init__(self, feature_names):
        self.feature_names_in_ = feature_names
        self.n_features_in_ = len(feature_names)


@pytest.mark.parametrize(
    "predictor_class,family,version",
    (
        (inference.TennisPredictor, "nn", inference.MODEL_VERSION),
        (inference.XGBoostPredictor, "xgboost", inference.XGB_MODEL_VERSION),
        (
            inference.RandomForestPredictor,
            "random_forest",
            inference.RF_MODEL_VERSION,
        ),
    ),
)
def test_live_predictors_do_not_fall_back_when_promoted_registry_entry_is_empty(
    predictor_class, family, version, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(inference, "_registry_model_entry", lambda *_: {})

    assert predictor_class(model_dir=str(tmp_path)).load_model() is False
    assert "registry entry is unavailable or incomplete" in capsys.readouterr().out


@pytest.mark.parametrize("family", ("nn", "xgboost", "random_forest"))
def test_live_registry_lookup_never_resolves_same_version_candidate(
    family, monkeypatch
):
    version = "v9.9.9"
    family_block = {
        "current_version": version,
        "models": {},
        "candidates": {version: {"model_file": "candidate/model.bin"}},
    }
    registry = {
        **(family_block if family == "nn" else {}),
        "xgboost": family_block if family == "xgboost" else {},
        "random_forest": family_block if family == "random_forest" else {},
    }
    monkeypatch.setattr(inference, "_REGISTRY", registry)

    assert inference._registry_model_entry(family, version) == {}


def test_live_registry_lookup_never_resolves_archived_model(monkeypatch):
    registry = {
        "current_version": "v2.0.0",
        "models": {
            "v1.0.0": {"model_file": "archive/model.pth"},
            "v2.0.0": {"model_file": "releases/model.pth"},
        },
        "xgboost": {},
        "random_forest": {},
    }
    monkeypatch.setattr(inference, "_REGISTRY", registry)

    assert inference._registry_model_entry("nn", "v1.0.0") == {}


def test_nn_inference_rejects_unknown_probability_mode(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.pth",
            "scaler_file": "scaler.pkl",
            "probability_mode": "automatic",
        },
    )

    assert inference.TennisPredictor(model_dir=str(tmp_path)).load_model() is False
    assert "unsupported probability_mode='automatic'" in capsys.readouterr().out


def test_nn_inference_has_no_generic_calibrated_filename_fallback(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "neural_network_calibrated_SURFACE_FIX.pkl").write_bytes(
        b"legacy-generic-artifact"
    )
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.pth",
            "scaler_file": "scaler.pkl",
            "probability_mode": "calibrated",
        },
    )

    assert inference.TennisPredictor(model_dir=str(tmp_path)).load_model() is False
    output = capsys.readouterr().out
    assert "calibrated registry entry is incomplete" in output
    assert "calibrated_model_file, calibrated_model_sha256" in output
    assert "SemVer calibration_version are required" in output


def test_nn_inference_blocks_calibrated_mode_until_lineage_is_versioned(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "model.pth").write_bytes(b"model")
    (tmp_path / "scaler.pkl").write_bytes(b"scaler")
    calibrated_path = tmp_path / "calibrated.pkl"
    calibrated_path.write_bytes(b"calibrated")
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.pth",
            "scaler_file": "scaler.pkl",
            "probability_mode": "calibrated",
            "calibrated_model_file": calibrated_path.name,
            "calibrated_model_sha256": sha256(b"different").hexdigest(),
            "calibration_version": "v1.0.0",
        },
    )

    assert inference.TennisPredictor(model_dir=str(tmp_path)).load_model() is False
    output = capsys.readouterr().out
    assert "Calibrated NN live inference is blocked" in output
    assert "nn_calibration_version" in output


def test_future_calibrated_enablement_still_rejects_checksum_mismatch(
    tmp_path, monkeypatch, capsys
):
    (tmp_path / "model.pth").write_bytes(b"model")
    (tmp_path / "scaler.pkl").write_bytes(b"scaler")
    calibrated_path = tmp_path / "calibrated.pkl"
    calibrated_path.write_bytes(b"calibrated")
    monkeypatch.setattr(inference, "CALIBRATED_LIVE_INFERENCE_ENABLED", True)
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.pth",
            "scaler_file": "scaler.pkl",
            "probability_mode": "calibrated",
            "calibrated_model_file": calibrated_path.name,
            "calibrated_model_sha256": sha256(b"different").hexdigest(),
            "calibration_version": "v1.0.0",
        },
    )

    assert inference.TennisPredictor(model_dir=str(tmp_path)).load_model() is False
    assert "calibrated NN artifact checksum mismatch" in capsys.readouterr().out


@pytest.mark.parametrize(
    "predictor_class,label",
    (
        (inference.XGBoostPredictor, "XGBoost"),
        (inference.RandomForestPredictor, "Random Forest"),
    ),
)
def test_tree_inference_rejects_non_raw_probability_mode(
    predictor_class, label, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.bin",
            "probability_mode": "calibrated",
        },
    )

    assert predictor_class(model_dir=str(tmp_path)).load_model() is False
    output = capsys.readouterr().out
    assert f"Promoted {label} live inference supports only" in output
    assert "probability_mode='raw'" in output


def test_xgboost_inference_rejects_swapped_feature_order(
    tmp_path, monkeypatch, capsys
):
    entry = inference._REGISTRY["xgboost"]["models"][
        inference.XGB_MODEL_VERSION
    ]
    expected = list(inference.ordered_feature_names(
        entry["feature_schema_sha256"]
    ))
    expected[0], expected[1] = expected[1], expected[0]

    class _WrongOrderXGBoost(_WrongOrderTreeArtifact):
        def __init__(self):
            super().__init__(expected)

        def load_model(self, path):
            return None

    import xgboost as xgb

    (tmp_path / "model.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(xgb, "XGBClassifier", _WrongOrderXGBoost)
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": "model.json",
            "feature_schema_sha256": entry["feature_schema_sha256"],
            "probability_mode": "raw",
        },
    )

    assert inference.XGBoostPredictor(model_dir=str(tmp_path)).load_model() is False
    assert "feature order does not match base_141" in capsys.readouterr().out


def test_random_forest_inference_rejects_swapped_feature_order(
    tmp_path, monkeypatch, capsys
):
    entry = inference._REGISTRY["random_forest"]["models"][
        inference.RF_MODEL_VERSION
    ]
    expected = list(inference.ordered_feature_names(
        entry["feature_schema_sha256"]
    ))
    expected[0], expected[1] = expected[1], expected[0]
    artifact_path = tmp_path / "model.pkl"
    artifact_path.write_bytes(pickle.dumps(_WrongOrderTreeArtifact(expected)))
    monkeypatch.setattr(
        inference,
        "_registry_model_entry",
        lambda *_: {
            "model_file": artifact_path.name,
            "feature_schema_sha256": entry["feature_schema_sha256"],
            "probability_mode": "raw",
        },
    )

    assert (
        inference.RandomForestPredictor(model_dir=str(tmp_path)).load_model()
        is False
    )
    assert "feature order does not match base_141" in capsys.readouterr().out
