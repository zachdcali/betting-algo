"""Counterfactual staking simulation at logged odds.

For each settled match, a model picks the side with the larger edge versus the
offered price's break-even probability and bets if that edge clears
``edge_threshold``. This exactly matches the live selection policy; de-vigged
market probability remains a model-quality baseline, not the execution hurdle. Stakes are
either flat (1 unit) or fractional-Kelly on a fixed (non-compounding) notional
bankroll, so models are comparable on the same cohort. All probabilities are
``P(player1 wins)``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def devig_two_way(o1: float, o2: float) -> tuple[float, float]:
    """Fair (p1, p2) implied by two decimal odds, normalized to remove the vig."""
    q1, q2 = 1.0 / o1, 1.0 / o2
    s = q1 + q2
    return q1 / s, q2 / s


def _kelly_fraction(p: float, dec_odds: float) -> float:
    b = dec_odds - 1.0
    if b <= 0:
        return 0.0
    f = (b * p - (1 - p)) / b
    return max(0.0, f)


def simulate(df: pd.DataFrame, mode: str = "flat", edge_threshold: float = 0.02,
             kelly_mult: float = 0.18, cap: float = 0.05, bankroll: float = 1000.0) -> dict:
    n_candidates = 0
    staked = pnl = 0.0
    wins = n_bets = 0
    equity = peak = bankroll
    max_dd = 0.0

    ordered = df.copy()
    if "prediction_time" in ordered.columns:
        ordered["_prediction_time"] = pd.to_datetime(
            ordered["prediction_time"], errors="coerce", utc=True,
            format="mixed",
        )
        ordered = ordered.sort_values("_prediction_time", kind="stable", na_position="last")

    for _, r in ordered.iterrows():
        o1, o2 = r.get("p1_odds_decimal"), r.get("p2_odds_decimal")
        if pd.isna(o1) or pd.isna(o2) or float(o1) <= 1.0 or float(o2) <= 1.0:
            continue
        n_candidates += 1
        p1 = float(r["p1_prob"])
        break_even1, break_even2 = 1.0 / float(o1), 1.0 / float(o2)
        edge1, edge2 = p1 - break_even1, (1 - p1) - break_even2
        if edge1 >= edge2:
            side_p, side_odds, edge, won = p1, float(o1), edge1, (r["y1"] == 1)
        else:
            side_p, side_odds, edge, won = 1 - p1, float(o2), edge2, (r["y1"] == 0)
        if edge < edge_threshold:
            continue

        if mode == "flat":
            stake = 1.0
        else:
            stake = min(kelly_mult * _kelly_fraction(side_p, side_odds) * bankroll, cap * bankroll)
        if stake <= 0:
            continue

        n_bets += 1
        staked += stake
        profit = stake * (side_odds - 1.0) if won else -stake
        pnl += profit
        wins += int(bool(won))
        equity += profit
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "mode": mode,
        "n_candidates": n_candidates,
        "n_bets": n_bets,
        "win_rate": (wins / n_bets) if n_bets else 0.0,
        "total_staked": staked,
        "pnl": pnl,
        "roi": (pnl / staked) if staked else 0.0,
        "ending_bankroll": bankroll + pnl,
        "max_drawdown": max_dd,
    }


def simulate_kalshi(
    df: pd.DataFrame,
    *,
    edge_threshold: float = 0.02,
    fee_rate: float = 0.07,
) -> dict:
    """Flat-stake ROI using raw Kalshi asks and fee-adjusted winning payouts.

    ``kalshi_p1_ask`` and ``kalshi_p2_ask`` are already probabilities. They are
    deliberately not normalized or de-vigged. One unit of stake buys
    ``1 / ask`` contracts. On a win, the taker fee per contract is
    ``fee_rate * ask * (1 - ask)``, or ``fee_rate * (1 - ask)`` per unit
    staked. A loss remains the full one-unit stake, matching the requested
    conservative measurement contract.
    """
    n_candidates = n_bets = wins = 0
    staked = pnl = fees = 0.0
    logging_times: list[pd.Timestamp] = []

    ordered = df.copy()
    if "kalshi_observation_at" in ordered.columns:
        ordered["_kalshi_observation_at"] = pd.to_datetime(
            ordered["kalshi_observation_at"], errors="coerce", utc=True,
            format="mixed",
        )
        ordered = ordered.sort_values(
            "_kalshi_observation_at", kind="stable", na_position="last",
        )

    for _, row in ordered.iterrows():
        try:
            p1 = float(row["p1_prob"])
            ask1 = float(row.get("kalshi_p1_ask"))
            ask2 = float(row.get("kalshi_p2_ask"))
            y1 = int(row["y1"])
        except (TypeError, ValueError, OverflowError):
            continue
        if not (
            np.isfinite(p1) and 0.0 <= p1 <= 1.0
            and np.isfinite(ask1) and 0.0 < ask1 < 1.0
            and np.isfinite(ask2) and 0.0 < ask2 < 1.0
            and y1 in {0, 1}
        ):
            continue
        n_candidates += 1
        observation_at = pd.to_datetime(
            row.get("kalshi_observation_at"), errors="coerce", utc=True,
            format="mixed",
        )
        if not pd.isna(observation_at):
            logging_times.append(observation_at)

        edge1 = p1 - ask1
        edge2 = (1.0 - p1) - ask2
        if edge1 >= edge2:
            side_probability, price, edge, won = p1, ask1, edge1, y1 == 1
        else:
            side_probability, price, edge, won = 1.0 - p1, ask2, edge2, y1 == 0
        if not np.isfinite(side_probability) or edge < edge_threshold:
            continue

        stake = 1.0
        n_bets += 1
        staked += stake
        if won:
            fee = stake * fee_rate * (1.0 - price)
            profit = stake * ((1.0 / price) - 1.0) - fee
            wins += 1
            fees += fee
        else:
            profit = -stake
        pnl += profit

    logging_since = min(logging_times).isoformat() if logging_times else ""
    return {
        "mode": "flat_kalshi",
        "n_candidates": n_candidates,
        "n_bets": n_bets,
        "win_rate": (wins / n_bets) if n_bets else np.nan,
        "total_staked": staked,
        "pnl": pnl,
        "fees": fees,
        "roi": (pnl / staked) if staked else np.nan,
        "logging_since": logging_since,
    }
