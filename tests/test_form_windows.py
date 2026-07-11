"""Window-math net (audit phase 2): form/fatigue helpers vs hand-computed
values on tiny fixtures where the right answer is checkable by eye.

Pins: Laplace smoothing constants, EWM half-life and window edges
(>= cut, < ref), streak sign convention, set counting from scores,
days-since week logic (see also test_feature_symmetry.py).
"""
import math
import os
import sys
from datetime import datetime

import pandas as pd
import pytest

BASE = os.path.join(os.path.dirname(__file__), "..", "production")
for p in (BASE, os.path.join(BASE, "features"), os.path.join(BASE, "scraping")):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ["USE_STORE_HISTORY"] = "0"

from features.ta_feature_calculator import TAFeatureCalculator  # noqa: E402

REF = datetime(2026, 7, 11)


def _calc():
    return TAFeatureCalculator.__new__(TAFeatureCalculator)  # helpers only


def test_laplace_hand_values():
    c = TAFeatureCalculator
    assert c._laplace(0, 0) == pytest.approx(0.5)
    assert c._laplace(1, 1) == pytest.approx(2.5 / 4)      # 0.625
    assert c._laplace(10, 10) == pytest.approx(11.5 / 13)  # the 10-0 streak reads 0.8846? no:
    # (10 + 1.5) / (10 + 3) = 11.5/13 = 0.8846 — document the actual constant
    assert c._laplace(10, 10) == pytest.approx(0.884615, abs=1e-6)
    assert c._laplace(0, 3) == pytest.approx(1.5 / 6)      # 0.25


def test_streak_sign_and_length():
    c = TAFeatureCalculator
    df = pd.DataFrame({"result": list("WWWL")})   # most-recent first
    assert c._streak(df) == 3
    df = pd.DataFrame({"result": list("LLW")})
    assert c._streak(df) == -2
    assert c._streak(pd.DataFrame()) == 0


def test_sets_14d_counts_scores_inside_window_only():
    df = pd.DataFrame([
        {"date": pd.Timestamp(REF) - pd.Timedelta(days=2), "score": "6-4 6-4"},        # 2 sets
        {"date": pd.Timestamp(REF) - pd.Timedelta(days=13), "score": "6-4 3-6 7-6(4)"}, # 3 sets
        {"date": pd.Timestamp(REF) - pd.Timedelta(days=15), "score": "6-0 6-0"},        # outside
    ])
    assert _calc()._sets_14d(df, REF) == 5


def test_form_trend_ewm_hand_computed():
    # three matches at 5, 10, 20 days ago: W, L, W — half-life 15d weights e^{-d/15}
    rows = [(5, "W"), (10, "L"), (20, "W")]
    df = pd.DataFrame([
        {"date": pd.Timestamp(REF) - pd.Timedelta(days=d), "result": r} for d, r in rows
    ])
    w = {d: math.exp(-d / 15.0) for d, _ in rows}
    expected = (w[5] + w[20]) / (w[5] + w[10] + w[20])
    assert _calc()._form_trend_ewm(df, REF) == pytest.approx(expected, abs=1e-12)
    # fewer than 3 in-window matches -> neutral 0.5
    assert _calc()._form_trend_ewm(df.head(2), REF) == 0.5
