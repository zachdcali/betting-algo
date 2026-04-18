#!/usr/bin/env python3
"""Backfill pending bet settlements from the settled prediction log."""

from pathlib import Path

import pandas as pd

from utils.bet_tracker import BetTracker


BASE_DIR = Path(__file__).parent
PREDICTION_LOG_PATH = BASE_DIR / "prediction_log.csv"
LOGS_DIR = BASE_DIR / "logs"


def main():
    if not PREDICTION_LOG_PATH.exists():
        print("No prediction_log.csv found.")
        return 0

    tracker = BetTracker(str(LOGS_DIR))
    df = pd.read_csv(PREDICTION_LOG_PATH)
    settled = df[df["actual_winner"].notna()].copy()
    if settled.empty:
        print("No settled prediction rows found.")
        return 0

    settled_count = 0
    for _, row in settled.iterrows():
        settled_count += tracker.settle_pending_bets_for_match(
            match_uid=row.get("match_uid"),
            p1=row.get("p1"),
            p2=row.get("p2"),
            actual_winner=int(row["actual_winner"]),
            notes=f"Backfilled from prediction log | score={row.get('score', '')}",
        )

    print(f"Backfilled {settled_count} tracked bet settlement(s)")
    return settled_count


if __name__ == "__main__":
    main()
