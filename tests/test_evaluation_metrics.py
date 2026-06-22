import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
from evaluation import metrics


def test_perfect_predictions():
    y = np.array([1, 0, 1, 0])
    p = np.array([0.99, 0.01, 0.99, 0.01])
    assert metrics.accuracy(y, p) == 1.0
    assert metrics.brier_score(y, p) < 0.001
    assert metrics.auc_score(y, p) == 1.0
    assert metrics.log_loss_score(y, p) < 0.02


def test_ece_zero_for_calibrated():
    # 100 preds at p=0.5 with exactly half positive -> perfectly calibrated bin
    y = np.array([1, 0] * 50)
    p = np.full(100, 0.5)
    assert metrics.ece(y, p, n_bins=10) < 1e-9


def test_calibration_slope_near_one_when_calibrated():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.05, 0.95, size=5000)
    y = (rng.uniform(size=5000) < p).astype(int)
    slope, intercept = metrics.calibration_slope_intercept(y, p)
    assert 0.8 < slope < 1.2
    assert abs(intercept) < 0.2


def test_compute_all_keys_and_n():
    y = np.array([1, 0, 1, 1, 0])
    p = np.array([0.6, 0.4, 0.7, 0.55, 0.3])
    out = metrics.compute_all(y, p)
    assert out["n"] == 5
    for k in ["accuracy", "log_loss", "brier", "auc", "ece", "cal_slope", "cal_intercept"]:
        assert k in out


def test_auc_nan_single_class():
    y = np.array([1, 1, 1])
    p = np.array([0.6, 0.7, 0.8])
    assert np.isnan(metrics.auc_score(y, p))
