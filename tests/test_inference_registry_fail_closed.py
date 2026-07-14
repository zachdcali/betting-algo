from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
PRODUCTION = ROOT / "production"
sys.path.insert(0, str(PRODUCTION))

from models import inference  # noqa: E402


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
