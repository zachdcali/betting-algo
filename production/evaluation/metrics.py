"""Pure model-scoring functions. No I/O.

All probabilities are ``P(player1 wins)`` and all ground truth is
``y1 = 1 if player1 won else 0``. Every model and the market are scored from
player1's perspective so the same functions apply uniformly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

_EPS = 1e-15


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


def accuracy(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) >= 0.5).astype(float) == y_true))


def log_loss_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = _clip(p)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def brier_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean((np.asarray(p, dtype=float) - y_true) ** 2))


def auc_score(y_true, p) -> float:
    y_true = np.asarray(y_true, dtype=int)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, np.asarray(p, dtype=float)))


def roc_table(y_true, p) -> pd.DataFrame:
    """Return the exact ROC threshold sweep used by the scalar AUC.

    ``sklearn`` drops only collinear intermediate thresholds by default.  The
    resulting curve retains its endpoints and area while avoiding redundant
    public dashboard rows.  A one-class cohort has no defined ROC curve and
    therefore returns an empty, schema-stable frame.
    """
    y_true = np.asarray(y_true, dtype=int)
    columns = [
        "point_index", "threshold", "false_positive_rate", "true_positive_rate",
    ]
    if len(np.unique(y_true)) < 2:
        return pd.DataFrame(columns=columns)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y_true, np.asarray(p, dtype=float), drop_intermediate=True,
    )
    rows = []
    for point_index, (fpr, tpr, threshold) in enumerate(
        zip(false_positive_rate, true_positive_rate, thresholds)
    ):
        rows.append({
            "point_index": point_index,
            "threshold": float(threshold) if np.isfinite(threshold) else np.nan,
            "false_positive_rate": float(fpr),
            "true_positive_rate": float(tpr),
        })
    return pd.DataFrame(rows, columns=columns)


def ece(y_true, p, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins)."""
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    total = len(p)
    err = 0.0
    for b in range(n_bins):
        mask = idx == b
        if not mask.any():
            continue
        conf = p[mask].mean()
        acc = y_true[mask].mean()
        err += (mask.sum() / total) * abs(acc - conf)
    return float(err)


def calibration_slope_intercept(y_true, p) -> tuple[float, float]:
    """Logistic regression of outcome on logit(p).

    slope ~ 1 and intercept ~ 0 indicate good calibration. slope < 1 means the
    model is over-confident; slope > 1 means under-confident.
    """
    y_true = np.asarray(y_true, dtype=int)
    p = _clip(p)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    if len(np.unique(y_true)) < 2:
        return float("nan"), float("nan")
    lr = LogisticRegression(C=1e6, solver="lbfgs")
    lr.fit(logit, y_true)
    return float(lr.coef_[0][0]), float(lr.intercept_[0])


def reliability_table(y_true, p, n_bins: int = 10) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        rows.append({
            "bin_lo": float(edges[b]),
            "bin_hi": float(edges[b + 1]),
            "mean_pred": float(p[mask].mean()) if mask.any() else float("nan"),
            "frac_pos": float(y_true[mask].mean()) if mask.any() else float("nan"),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(rows)


def compute_all(y_true, p, n_bins: int = 10) -> dict:
    """All scalar metrics in one dict, keyed for the ledger."""
    slope, intercept = calibration_slope_intercept(y_true, p)
    return {
        "n": int(len(np.asarray(p))),
        "accuracy": accuracy(y_true, p),
        "log_loss": log_loss_score(y_true, p),
        "brier": brier_score(y_true, p),
        "auc": auc_score(y_true, p),
        "ece": ece(y_true, p, n_bins),
        "cal_slope": slope,
        "cal_intercept": intercept,
    }
