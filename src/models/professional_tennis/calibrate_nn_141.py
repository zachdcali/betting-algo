#!/usr/bin/env python3
"""
Calibrate the current 141-feature live Neural Network model.

Uses:
- train period (< 2022) only for missing-value medians
- calibration period (2022) for Platt scaling
- test period (>= 2023) for honest evaluation

Outputs a production-loadable calibrated artifact at:
results/professional_tennis/Neural_Network/neural_network_calibrated_SURFACE_FIX.pkl
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


REPO_ROOT = Path(__file__).resolve().parents[3]
PROD_ROOT = REPO_ROOT / "production"
if str(PROD_ROOT) not in sys.path:
    sys.path.insert(0, str(PROD_ROOT))

from models.inference import EXACT_141_FEATURES  # noqa: E402
from models.nn_runtime import NNWrapper, TennisNet  # noqa: E402


RESULTS_DIR = REPO_ROOT / "results" / "professional_tennis" / "Neural_Network"
MODEL_PATH = RESULTS_DIR / "neural_network_model_SURFACE_FIX.pth"
SCALER_PATH = RESULTS_DIR / "scaler_SURFACE_FIX.pkl"
CALIBRATED_PATH = RESULTS_DIR / "neural_network_calibrated_SURFACE_FIX.pkl"
METRICS_PATH = RESULTS_DIR / "calibration_metrics_current_141.csv"
ML_READY_PATH = REPO_ROOT / "data" / "JeffSackmann" / "jeffsackmann_ml_ready_SURFACE_FIX.csv"


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Expected Calibration Error."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        mask = (y_prob >= edges[i]) & (y_prob < edges[i + 1])
        if mask.sum() == 0:
            continue
        ece += mask.sum() * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece / len(y_true))


def metrics_dict(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    preds = (y_prob > 0.5).astype(int)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10)
    return {
        "accuracy": float(accuracy_score(y_true, preds)),
        "auc": float(roc_auc_score(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "ece": float(np.mean(np.abs(prob_true - prob_pred))),
        "ece_weighted": float(ece_score(y_true, y_prob)),
        "extreme_90_rate": float(((y_prob > 0.9) | (y_prob < 0.1)).mean()),
        "extreme_99_rate": float(((y_prob > 0.99) | (y_prob < 0.01)).mean()),
    }


def print_metrics(label: str, metrics: dict[str, float]) -> None:
    print(f"\n{label}:")
    print(f"  Accuracy:   {metrics['accuracy']:.4f}")
    print(f"  AUC:        {metrics['auc']:.4f}")
    print(f"  Log Loss:   {metrics['log_loss']:.4f}")
    print(f"  Brier:      {metrics['brier']:.4f}")
    print(f"  ECE:        {metrics['ece']:.4f}")
    print(f"  ECE weight: {metrics['ece_weighted']:.4f}")
    print(f"  >90/<10:    {metrics['extreme_90_rate']:.2%}")
    print(f"  >99/<1:     {metrics['extreme_99_rate']:.2%}")


def main() -> None:
    print("=" * 72)
    print("CALIBRATE CURRENT NN-141 LIVE MODEL")
    print("=" * 72)

    if not MODEL_PATH.exists() or not SCALER_PATH.exists():
        raise FileNotFoundError("Current SURFACE_FIX NN model/scaler not found")
    if not ML_READY_PATH.exists():
        raise FileNotFoundError("ML-ready SURFACE_FIX dataset not found")

    print("\n1. Loading dataset...")
    df = pd.read_csv(ML_READY_PATH, low_memory=False)
    df["tourney_date"] = pd.to_datetime(df["tourney_date"])
    df = df[df["tourney_date"] >= "1990-01-01"].copy()
    df = df.dropna(subset=["Player1_Rank", "Player2_Rank"])

    feature_cols = [col for col in EXACT_141_FEATURES if col in df.columns]
    if len(feature_cols) != 141:
        missing = [col for col in EXACT_141_FEATURES if col not in df.columns]
        raise RuntimeError(f"Expected 141 features, found {len(feature_cols)}. Missing: {missing[:10]}")

    train_df = df[df["tourney_date"] < "2022-01-01"].copy()
    cal_df = df[(df["tourney_date"] >= "2022-01-01") & (df["tourney_date"] < "2023-01-01")].copy()
    test_df = df[df["tourney_date"] >= "2023-01-01"].copy()
    print(f"   Train rows: {len(train_df):,}")
    print(f"   Cal rows:   {len(cal_df):,}")
    print(f"   Test rows:  {len(test_df):,}")

    medians = train_df[feature_cols].median()
    X_cal = cal_df[feature_cols].fillna(medians)
    y_cal = cal_df["Player1_Wins"].astype(int).values
    X_test = test_df[feature_cols].fillna(medians)
    y_test = test_df["Player1_Wins"].astype(int).values

    print("\n2. Loading raw NN...")
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    model = TennisNet(scaler.n_features_in_)
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    wrapper = NNWrapper(model=model, scaler=scaler, device=torch.device("cpu"))

    print("\n3. Evaluating raw probabilities...")
    raw_test_prob = wrapper.predict_proba(X_test.values)[:, 1]
    raw_metrics = metrics_dict(y_test, raw_test_prob)
    print_metrics("RAW", raw_metrics)

    print("\n4. Fitting Platt calibration on 2022...")
    calibrated = CalibratedClassifierCV(estimator=wrapper, method="sigmoid", cv="prefit")
    calibrated.fit(X_cal.values, y_cal)

    print("\n5. Evaluating calibrated probabilities...")
    cal_test_prob = calibrated.predict_proba(X_test.values)[:, 1]
    cal_metrics = metrics_dict(y_test, cal_test_prob)
    print_metrics("CALIBRATED", cal_metrics)

    delta = {
        key: cal_metrics[key] - raw_metrics[key]
        for key in ["accuracy", "auc", "log_loss", "brier", "ece", "ece_weighted", "extreme_90_rate", "extreme_99_rate"]
    }
    print("\nDELTA (calibrated - raw):")
    for key, value in delta.items():
        suffix = "pp" if "rate" in key else ""
        print(f"  {key}: {value:+.6f}{suffix}")

    print("\n6. Saving calibrated artifact...")
    with open(CALIBRATED_PATH, "wb") as f:
        pickle.dump(calibrated, f)

    metrics_rows = []
    for family, metrics in [("raw", raw_metrics), ("calibrated", cal_metrics)]:
        for metric_name, metric_value in metrics.items():
            metrics_rows.append({"model": family, "metric": metric_name, "value": metric_value})
    pd.DataFrame(metrics_rows).to_csv(METRICS_PATH, index=False)

    print(f"   Saved calibrated model to: {CALIBRATED_PATH}")
    print(f"   Saved metrics to:          {METRICS_PATH}")


if __name__ == "__main__":
    main()
