#!/usr/bin/env python3
"""Shared Neural Network runtime classes for training, calibration, and inference."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils import Bunch


class TennisNet(nn.Module):
    """Neural-network architecture used by the current 141-feature model."""

    def __init__(self, input_size: int):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.network(x)


class NNWrapper(BaseEstimator, ClassifierMixin):
    """
    sklearn-compatible wrapper for the PyTorch NN.

    This lives in a real module so calibrated pickle artifacts can be loaded by
    production without the old `__main__.NNWrapper` failure mode.
    """

    def __init__(self, model, scaler, device):
        self.model = model
        self.scaler = scaler
        self.device = device
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def __sklearn_tags__(self):
        return Bunch(
            estimator_type="classifier",
            requires_fit=False,
            input_tags=Bunch(sparse=False),
        )

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)

    def predict_proba(self, X):
        if self.model is None or self.scaler is None:
            raise ValueError("Model and scaler must be set")
        X_scaled = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(torch.FloatTensor(X_scaled).to(self.device)).squeeze().cpu().numpy()
        if np.ndim(outputs) == 0:
            outputs = np.array([[outputs]])
        elif np.ndim(outputs) == 1:
            outputs = outputs.reshape(-1, 1)
        return np.hstack((1 - outputs, outputs))
