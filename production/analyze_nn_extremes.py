#!/usr/bin/env python3
"""
Inspect where the live NN is making extreme predictions relative to XGBoost and the market.

Examples:
    python analyze_nn_extremes.py
    python analyze_nn_extremes.py --version v1.2.1 --threshold 0.95 --decision-grade-only
    python analyze_nn_extremes.py --pending-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from prediction_logger import upgrade_prediction_log


LOG_PATH = Path(__file__).parent / "prediction_log.csv"
REGISTRY_PATH = Path(__file__).parent / "models" / "model_registry.json"


def load_current_nn_version() -> str:
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH) as f:
            registry = json.load(f)
        return registry.get("current_version", "unknown")
    return "unknown"


def pct(value):
    if pd.isna(value):
        return "n/a"
    return f"{value:.1%}"


def conf(prob_series: pd.Series) -> pd.Series:
    return np.maximum(prob_series, 1 - prob_series)


def print_table(df: pd.DataFrame, columns: list[str], limit: int = 15):
    if df.empty:
        print("  (none)")
        return
    display = df[columns].head(limit).copy()
    for col in ["model_p1_prob", "xgb_p1_prob", "rf_p1_prob", "market_p1_prob", "nn_conf", "xgb_conf", "market_conf", "nn_vs_xgb", "nn_vs_market"]:
        if col in display.columns:
            display[col] = display[col].map(pct)
    print(display.to_string(index=False))


def summarize_bucket(name: str, df: pd.DataFrame):
    print(f"\n{name}: {len(df)} match(es)")
    if df.empty:
        return
    print(f"  NN accuracy:     {pct(df['model_correct'].mean())} ({int(df['model_correct'].sum())}/{len(df)})")
    print(f"  XGB accuracy:    {pct(df['xgb_correct'].mean())} ({int(df['xgb_correct'].sum())}/{len(df)})")
    print(f"  Market accuracy: {pct(df['market_correct'].mean())} ({int(df['market_correct'].sum())}/{len(df)})")
    print(f"  Avg NN conf:     {pct(df['nn_conf'].mean())}")
    print(f"  Avg |NN-XGB|:    {pct(df['nn_vs_xgb'].mean())}")
    print(f"  Avg |NN-Mkt|:    {pct(df['nn_vs_market'].mean())}")


def main():
    parser = argparse.ArgumentParser(description="Analyze extreme live NN predictions versus XGB and market.")
    parser.add_argument("--version", default=load_current_nn_version(), help="NN model version to analyze")
    parser.add_argument("--threshold", type=float, default=0.95, help="Extreme NN confidence threshold")
    parser.add_argument("--decision-grade-only", action="store_true", help="Restrict to snapshot_v2 exact-feature rows")
    parser.add_argument("--pending-only", action="store_true", help="Show pending rows only")
    parser.add_argument("--top", type=int, default=15, help="Rows to show in each table")
    args = parser.parse_args()

    df = upgrade_prediction_log(LOG_PATH, stale_days=5, write=False)
    if df.empty:
        print("No prediction log rows found.")
        return

    df["features_complete"] = df["features_complete"].fillna(True).astype(bool)
    df["nn_model_version"] = df["nn_model_version"].fillna(df["model_version"])

    version_mask = df["nn_model_version"].astype(str) == args.version
    if args.pending_only:
        subset = df[
            version_mask
            & df["actual_winner"].isna()
            & df["model_p1_prob"].notna()
            & df["xgb_p1_prob"].notna()
            & df["market_p1_prob"].notna()
        ].copy()
    else:
        subset = df[
            version_mask
            & df["actual_winner"].notna()
            & df["features_complete"]
            & df["model_p1_prob"].notna()
            & df["xgb_p1_prob"].notna()
            & df["market_p1_prob"].notna()
            & df["model_correct"].notna()
            & df["xgb_correct"].notna()
            & df["market_correct"].notna()
            & (df["market_p1_prob"].round(4) != 0.5)
        ].copy()

    if args.decision_grade_only:
        subset = subset[
            (subset["logging_quality"] == "snapshot_v2")
            & (subset["rescore_quality"] == "exact_feature_snapshot")
        ].copy()

    if subset.empty:
        print("No rows matched the requested filters.")
        return

    subset["nn_conf"] = conf(subset["model_p1_prob"])
    subset["xgb_conf"] = conf(subset["xgb_p1_prob"])
    subset["market_conf"] = conf(subset["market_p1_prob"])
    subset["nn_vs_xgb"] = (subset["model_p1_prob"] - subset["xgb_p1_prob"]).abs()
    subset["nn_vs_market"] = (subset["model_p1_prob"] - subset["market_p1_prob"]).abs()
    subset["nn_extreme"] = subset["nn_conf"] >= args.threshold
    subset["xgb_not_extreme"] = subset["xgb_conf"] < args.threshold
    subset["market_not_extreme"] = subset["market_conf"] < args.threshold

    print("=" * 72)
    print("LIVE NN EXTREME PREDICTION ANALYSIS")
    print("=" * 72)
    print(f"Version:            {args.version}")
    print(f"Extreme threshold:  {args.threshold:.0%}")
    print(f"Decision-grade:     {'yes' if args.decision_grade_only else 'no'}")
    print(f"Mode:               {'pending only' if args.pending_only else 'settled only'}")
    print(f"Rows analyzed:      {len(subset)}")

    if not args.pending_only:
        for threshold in [0.90, 0.95, 0.99]:
            bucket = subset[subset["nn_conf"] >= threshold]
            print(f"  NN >= {threshold:.0%}: {len(bucket)} | accuracy {pct(bucket['model_correct'].mean()) if len(bucket) else 'n/a'}")

    extreme = subset[subset["nn_extreme"]].copy()
    summarize_bucket("Extreme NN subset", extreme if not args.pending_only else pd.DataFrame(columns=subset.columns))

    if not args.pending_only:
        extreme_wrong = extreme[extreme["model_correct"] == 0].sort_values(
            ["nn_conf", "nn_vs_market", "nn_vs_xgb"],
            ascending=[False, False, False],
        )
        print("\nMost confident NN misses:")
        print_table(
            extreme_wrong,
            [
                "match_date", "tournament", "surface", "level", "round", "p1", "p2",
                "model_p1_prob", "xgb_p1_prob", "market_p1_prob",
                "nn_conf", "nn_vs_xgb", "nn_vs_market",
                "model_correct", "xgb_correct", "market_correct",
                "logging_quality",
            ],
            limit=args.top,
        )

    biggest_gaps = subset.sort_values(
        ["nn_vs_market", "nn_vs_xgb", "nn_conf"],
        ascending=[False, False, False],
    )
    print("\nLargest NN disagreements vs market / XGB:")
    print_table(
        biggest_gaps,
        [
            "match_date", "tournament", "surface", "level", "round", "p1", "p2",
            "model_p1_prob", "xgb_p1_prob", "market_p1_prob",
            "nn_conf", "xgb_conf", "market_conf",
            "nn_vs_market", "nn_vs_xgb",
            "model_correct", "xgb_correct", "market_correct",
        ] if not args.pending_only else [
            "match_date", "tournament", "surface", "level", "round", "p1", "p2",
            "model_p1_prob", "xgb_p1_prob", "market_p1_prob",
            "nn_conf", "xgb_conf", "market_conf",
            "nn_vs_market", "nn_vs_xgb",
            "record_status", "logging_quality",
        ],
        limit=args.top,
    )

    if not args.pending_only:
        print("\nExtreme rows by surface:")
        surface_counts = extreme.groupby("surface").size().sort_values(ascending=False)
        if surface_counts.empty:
            print("  (none)")
        else:
            print(surface_counts.to_string())

        print("\nExtreme rows by round:")
        round_counts = extreme.groupby("round").size().sort_values(ascending=False)
        if round_counts.empty:
            print("  (none)")
        else:
            print(round_counts.to_string())
    else:
        pending_extremes = subset[subset["nn_extreme"]].sort_values(
            ["nn_conf", "nn_vs_market", "nn_vs_xgb"],
            ascending=[False, False, False],
        )
        print("\nCurrent pending NN extremes:")
        print_table(
            pending_extremes,
            [
                "match_date", "tournament", "surface", "level", "round", "p1", "p2",
                "model_p1_prob", "xgb_p1_prob", "rf_p1_prob", "market_p1_prob",
                "nn_conf", "nn_vs_market", "nn_vs_xgb",
                "record_status", "logging_quality",
            ],
            limit=args.top,
        )


if __name__ == "__main__":
    main()
