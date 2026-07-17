import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_DIR = REPO_ROOT / "production"
if str(PRODUCTION_DIR) not in sys.path:
    sys.path.insert(0, str(PRODUCTION_DIR))

import numpy as np
import pandas as pd
import pytest
from evaluation import roi


def test_devig_normalizes_to_one():
    p1, p2 = roi.devig_two_way(1.5, 2.5)
    assert abs((p1 + p2) - 1.0) < 1e-9
    assert p1 > p2  # shorter odds -> higher prob


def test_flat_roi_winning_edge():
    df = pd.DataFrame([
        dict(p1_prob=0.8, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1),
        dict(p1_prob=0.8, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1),
    ])
    out = roi.simulate(df, mode="flat")
    assert out["n_bets"] == 2
    assert out["roi"] > 0
    assert out["pnl"] > 0
    assert out["win_rate"] == 1.0


def test_no_bet_when_edge_below_threshold():
    df = pd.DataFrame([dict(p1_prob=0.5, p1_odds_decimal=2.0, p2_odds_decimal=2.0, y1=1)])
    out = roi.simulate(df, mode="flat", edge_threshold=0.02)
    assert out["n_bets"] == 0
    assert out["roi"] == 0.0


def test_flat_policy_never_qualifies_negative_ev_de_vig_edge():
    # De-vigged fair market is 50%, but 1.91 requires 52.36% to break even.
    # A 52% model is therefore not a live or replay bet.
    df = pd.DataFrame([dict(
        p1_prob=0.52, p1_odds_decimal=1.91, p2_odds_decimal=1.91, y1=1,
    )])
    out = roi.simulate(df, mode="flat", edge_threshold=0.02)
    assert out["n_candidates"] == 1
    assert out["n_bets"] == 0


def test_kelly_caps_stake():
    df = pd.DataFrame([dict(p1_prob=0.99, p1_odds_decimal=5.0, p2_odds_decimal=1.2, y1=1)])
    out = roi.simulate(df, mode="kelly", kelly_mult=1.0, cap=0.05, bankroll=1000.0)
    assert out["total_staked"] <= 0.05 * 1000.0 + 1e-9


def test_skips_rows_with_bad_odds():
    df = pd.DataFrame([
        dict(p1_prob=0.8, p1_odds_decimal=np.nan, p2_odds_decimal=2.0, y1=1),
        dict(p1_prob=0.8, p1_odds_decimal=1.0, p2_odds_decimal=2.0, y1=1),
    ])
    out = roi.simulate(df, mode="flat")
    assert out["n_candidates"] == 0
    assert out["n_bets"] == 0


def test_kelly_drawdown_orders_mixed_timestamp_formats_chronologically():
    rows = pd.DataFrame([
        {"prediction_time": "2026-07-13T12:00:00+00:00", "p1_prob": 0.8,
         "p1_odds_decimal": 2.0, "p2_odds_decimal": 2.0, "y1": 1},
        {"prediction_time": "2026-07-13 10:00:00", "p1_prob": 0.8,
         "p1_odds_decimal": 2.0, "p2_odds_decimal": 2.0, "y1": 0},
        {"prediction_time": "2026-07-13T11:00:00Z", "p1_prob": 0.8,
         "p1_odds_decimal": 2.0, "p2_odds_decimal": 2.0, "y1": 1},
    ])
    chronological = roi.simulate(rows, mode="kelly", kelly_mult=1.0, cap=0.05)
    explicit = rows.iloc[[1, 2, 0]].copy()
    explicit["prediction_time"] = None
    explicitly_sorted = roi.simulate(
        explicit, mode="kelly", kelly_mult=1.0, cap=0.05,
    )
    assert chronological["max_drawdown"] == explicitly_sorted["max_drawdown"]


def test_kalshi_flat_roi_uses_raw_ask_and_subtracts_winner_fee():
    rows = pd.DataFrame([{
        "p1_prob": 0.80,
        "kalshi_p1_ask": 0.60,
        "kalshi_p2_ask": 0.41,
        "kalshi_observation_at": "2026-07-17T10:00:00Z",
        "y1": 1,
    }])

    result = roi.simulate_kalshi(rows)

    expected_fee = 0.07 * (1.0 - 0.60)
    expected_profit = (1.0 / 0.60) - 1.0 - expected_fee
    assert result["n_candidates"] == 1
    assert result["n_bets"] == 1
    assert result["fees"] == pytest.approx(expected_fee)
    assert result["pnl"] == pytest.approx(expected_profit)
    assert result["roi"] == pytest.approx(expected_profit)
    assert result["logging_since"] == "2026-07-17T10:00:00+00:00"


def test_kalshi_asks_are_not_devigged_before_edge_selection():
    # The asks sum to 1.05. Normalizing them would manufacture a larger P1 edge;
    # the raw 0.62 hurdle leaves this 0.63 model below the two-point gate.
    rows = pd.DataFrame([{
        "p1_prob": 0.63,
        "kalshi_p1_ask": 0.62,
        "kalshi_p2_ask": 0.43,
        "kalshi_observation_at": "2026-07-17T10:00:00Z",
        "y1": 1,
    }])

    result = roi.simulate_kalshi(rows, edge_threshold=0.02)

    assert result["n_candidates"] == 1
    assert result["n_bets"] == 0
    assert np.isnan(result["roi"])


def test_kalshi_loss_is_one_flat_unit_without_fabricated_payout():
    rows = pd.DataFrame([{
        "p1_prob": 0.85,
        "kalshi_p1_ask": 0.55,
        "kalshi_p2_ask": 0.46,
        "kalshi_observation_at": "2026-07-17T10:00:00Z",
        "y1": 0,
    }])

    result = roi.simulate_kalshi(rows)

    assert result["n_bets"] == 1
    assert result["pnl"] == -1.0
    assert result["roi"] == -1.0
    assert result["fees"] == 0.0
