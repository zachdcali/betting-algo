#!/usr/bin/env python3
"""
Prediction log analysis — run anytime for a detailed accuracy & diagnostics report.

Usage:
    python analyze_predictions.py              # full report
    python analyze_predictions.py --version v1.2.1   # filter to one model version
    python analyze_predictions.py --pending    # show pending/unsettled summary
"""

import argparse
import sys
from pathlib import Path
from collections import Counter

import pandas as pd
import numpy as np

from prediction_logger import upgrade_prediction_log

LOG_PATH = Path(__file__).parent / "prediction_log.csv"


def load_clean(df: pd.DataFrame, version: str = None) -> pd.DataFrame:
    """Return settled predictions with complete features, excluding 50/50 market."""
    settled = df[df['actual_winner'].notna()].copy()
    clean = settled[
        settled['model_correct'].notna() &
        settled['features_complete'].fillna(True).astype(bool)
    ]
    if version:
        clean = clean[clean['model_version'] == version]
    return clean


def section(title: str, width: int = 70):
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")


def divider(width: int = 66):
    print(f"  {'─'*width}")


# ---------------------------------------------------------------------------
# 1. Accuracy
# ---------------------------------------------------------------------------
def report_accuracy(df: pd.DataFrame, version: str = None):
    clean = load_clean(df, version)
    if clean.empty:
        print("  No settled predictions to analyze.")
        return

    has_pick = clean['market_p1_prob'] != 0.50
    main = clean[has_pick]
    n_5050 = len(clean) - len(main)

    section("ACCURACY" + (f" — {version}" if version else ""))

    n = len(main)
    m_correct = int(main['model_correct'].sum())
    mk_correct = int(main['market_correct'].sum())
    ma = main['model_correct'].mean()
    mk = main['market_correct'].mean()
    print(f"  Settled (complete features, market has pick): {n}")
    print(f"  Excluded from market comparison: {n_5050} matches where market was 50/50 (no pick)")
    print(f"")
    print(f"  --- Model vs Market (apples-to-apples, excluding 50/50) ---")
    print(f"  Model:   {m_correct}/{n}  =  {ma:.1%}")
    print(f"  Market:  {mk_correct}/{n}  =  {mk:.1%}")
    print(f"  Edge:    {ma - mk:+.1%}")
    print(f"  Both correct: {int(((main['model_correct'] == 1) & (main['market_correct'] == 1)).sum())}")
    print(f"  Both wrong:   {int(((main['model_correct'] == 0) & (main['market_correct'] == 0)).sum())}")
    print(f"  Model right, market wrong: {int(((main['model_correct'] == 1) & (main['market_correct'] == 0)).sum())}")
    print(f"  Market right, model wrong: {int(((main['model_correct'] == 0) & (main['market_correct'] == 1)).sum())}")

    if 'xgb_correct' in main.columns and main['xgb_correct'].notna().any():
        xgb_main = main[main['xgb_correct'].notna()]
        print(f"")
        print(f"  --- Side-by-side model family comparison ---")
        print(f"  XGBoost: {int(xgb_main['xgb_correct'].sum())}/{len(xgb_main)}  =  {xgb_main['xgb_correct'].mean():.1%}")
    if 'rf_correct' in main.columns and main['rf_correct'].notna().any():
        rf_main = main[main['rf_correct'].notna()]
        if 'xgb_correct' not in main.columns or not main['xgb_correct'].notna().any():
            print(f"")
            print(f"  --- Side-by-side model family comparison ---")
        print(f"  Random Forest: {int(rf_main['rf_correct'].sum())}/{len(rf_main)}  =  {rf_main['rf_correct'].mean():.1%}")

    # Model accuracy including 50/50 matches (model always makes a pick)
    if n_5050 > 0:
        all_n = len(clean)
        all_m_correct = int(clean['model_correct'].sum())
        all_ma = clean['model_correct'].mean()
        fiftyfifty = clean[~has_pick]
        f_correct = int(fiftyfifty['model_correct'].sum())
        print(f"")
        print(f"  --- Model standalone (including 50/50 matches) ---")
        print(f"  Model:   {all_m_correct}/{all_n}  =  {all_ma:.1%}")
        print(f"  On 50/50 matches alone: {f_correct}/{n_5050}  =  {fiftyfifty['model_correct'].mean():.1%}")

    # By model version
    if not version:
        divider()
        print(f"  By model version:")
        print(f"    {'Version':<25s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
        print(f"    {'─'*25} {'─'*14} {'─'*14} {'─'*7}")
        for ver, grp in main.groupby('model_version'):
            gmc = int(grp['model_correct'].sum())
            gmkc = int(grp['market_correct'].sum())
            gn = len(grp)
            gma = grp['model_correct'].mean()
            gmk = grp['market_correct'].mean()
            print(f"    {ver:<25s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")

    # By surface
    if main['surface'].notna().any():
        divider()
        print(f"  By surface:")
        print(f"    {'Surface':<10s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
        print(f"    {'─'*10} {'─'*14} {'─'*14} {'─'*7}")
        for surf, grp in main.groupby('surface'):
            gmc = int(grp['model_correct'].sum())
            gmkc = int(grp['market_correct'].sum())
            gn = len(grp)
            gma = grp['model_correct'].mean()
            gmk = grp['market_correct'].mean()
            print(f"    {surf:<10s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")

    # By level
    if main['level'].notna().any():
        divider()
        print(f"  By level:")
        print(f"    {'Level':<10s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
        print(f"    {'─'*10} {'─'*14} {'─'*14} {'─'*7}")
        for lvl, grp in main.groupby('level'):
            if len(grp) < 3:
                continue
            gmc = int(grp['model_correct'].sum())
            gmkc = int(grp['market_correct'].sum())
            gn = len(grp)
            gma = grp['model_correct'].mean()
            gmk = grp['market_correct'].mean()
            print(f"    {str(lvl):<10s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")

    # By round
    if main['round'].notna().any():
        divider()
        print(f"  By round:")
        print(f"    {'Round':<10s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
        print(f"    {'─'*10} {'─'*14} {'─'*14} {'─'*7}")
        for rnd, grp in main.groupby('round'):
            if len(grp) < 3:
                continue
            gmc = int(grp['model_correct'].sum())
            gmkc = int(grp['market_correct'].sum())
            gn = len(grp)
            gma = grp['model_correct'].mean()
            gmk = grp['market_correct'].mean()
            print(f"    {str(rnd):<10s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")

    # By best rank (highest-ranked player in the match)
    if main['p1_rank'].notna().any() or main['p2_rank'].notna().any():
        divider()
        tmp = main.copy()
        tmp['best_rank'] = tmp[['p1_rank', 'p2_rank']].min(axis=1)
        has_rank = tmp[tmp['best_rank'].notna()]
        if len(has_rank) > 0:
            rank_bins = [0, 20, 50, 100, 200, 10000]
            rank_labels = ['Top 20', '21-50', '51-100', '101-200', '200+']
            has_rank = has_rank.copy()
            has_rank['rank_tier'] = pd.cut(has_rank['best_rank'], bins=rank_bins, labels=rank_labels, right=True)
            print(f"  By best rank in match:")
            print(f"    {'Rank Tier':<10s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
            print(f"    {'─'*10} {'─'*14} {'─'*14} {'─'*7}")
            for label in rank_labels:
                grp = has_rank[has_rank['rank_tier'] == label]
                if len(grp) < 3:
                    continue
                gmc = int(grp['model_correct'].sum())
                gmkc = int(grp['market_correct'].sum())
                gn = len(grp)
                gma = grp['model_correct'].mean()
                gmk = grp['market_correct'].mean()
                print(f"    {label:<10s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")

    # By rank gap (difference between the two players' ranks)
    if main['p1_rank'].notna().any() or main['p2_rank'].notna().any():
        divider()
        tmp = main.copy()
        tmp['rank_gap'] = (tmp['p1_rank'] - tmp['p2_rank']).abs()
        has_gap = tmp[tmp['rank_gap'].notna()]
        if len(has_gap) > 0:
            gap_bins = [0, 20, 50, 100, 200, 10000]
            gap_labels = ['0-20', '21-50', '51-100', '101-200', '200+']
            has_gap = has_gap.copy()
            has_gap['gap_tier'] = pd.cut(has_gap['rank_gap'], bins=gap_bins, labels=gap_labels, right=True)
            print(f"  By rank gap (|p1_rank - p2_rank|):")
            print(f"    {'Gap':<10s} {'Model':>14s} {'Market':>14s} {'Edge':>7s}")
            print(f"    {'─'*10} {'─'*14} {'─'*14} {'─'*7}")
            for label in gap_labels:
                grp = has_gap[has_gap['gap_tier'] == label]
                if len(grp) < 3:
                    continue
                gmc = int(grp['model_correct'].sum())
                gmkc = int(grp['market_correct'].sum())
                gn = len(grp)
                gma = grp['model_correct'].mean()
                gmk = grp['market_correct'].mean()
                print(f"    {label:<10s} {gmc:>3d}/{gn:<3d} = {gma:.1%} {gmkc:>3d}/{gn:<3d} = {gmk:.1%} {gma-gmk:>+6.1%}")


# ---------------------------------------------------------------------------
# 2. Calibration
# ---------------------------------------------------------------------------
def report_calibration(df: pd.DataFrame, version: str = None):
    clean = load_clean(df, version)
    if clean.empty:
        return

    section("CALIBRATION" + (f" — {version}" if version else ""))
    print(f"  Model predicted probability vs actual win rate")
    print(f"  (grouped into bins — closer to diagonal = better calibrated)")
    print()

    # Use model_p1_prob and actual_winner==1
    clean = clean.copy()
    clean['model_conf'] = clean['model_p1_prob'].clip(0, 1)
    clean['p1_won'] = (clean['actual_winner'] == 1).astype(int)

    bins = [0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    labels = ['0-20%', '20-30%', '30-40%', '40-50%', '50-60%', '60-70%', '70-80%', '80-100%']
    clean['bin'] = pd.cut(clean['model_conf'], bins=bins, labels=labels, right=False)

    print(f"  {'Predicted':<12s} {'Actual':>8s} {'Count':>7s} {'Diff':>7s}")
    print(f"  {'─'*12}  {'─'*8} {'─'*7} {'─'*7}")
    for label in labels:
        grp = clean[clean['bin'] == label]
        if len(grp) == 0:
            continue
        pred_avg = grp['model_conf'].mean()
        actual_avg = grp['p1_won'].mean()
        diff = actual_avg - pred_avg
        print(f"  {label:<12s} {actual_avg:>7.1%} {len(grp):>7d} {diff:>+6.1%}")


# ---------------------------------------------------------------------------
# 3. Edge analysis
# ---------------------------------------------------------------------------
def report_edge_analysis(df: pd.DataFrame, version: str = None):
    clean = load_clean(df, version)
    if clean.empty:
        return

    section("EDGE ANALYSIS" + (f" — {version}" if version else ""))

    clean = clean.copy()
    clean['abs_edge'] = (clean['model_p1_prob'] - clean['market_p1_prob']).abs()

    bins = [0, 0.05, 0.10, 0.15, 0.20, 1.0]
    labels = ['0-5%', '5-10%', '10-15%', '15-20%', '20%+']
    clean['edge_bin'] = pd.cut(clean['abs_edge'], bins=bins, labels=labels, right=False)

    print(f"  Does the model perform better on higher-edge bets?")
    print()
    print(f"  {'Edge Range':<12s} {'Model':>8s} {'Market':>8s} {'Edge':>7s} {'Count':>7s}")
    print(f"  {'─'*12}  {'─'*8} {'─'*8} {'─'*7} {'─'*7}")
    for label in labels:
        grp = clean[clean['edge_bin'] == label]
        if len(grp) == 0:
            continue
        ma = grp['model_correct'].mean()
        mk = grp['market_correct'].mean()
        print(f"  {label:<12s} {ma:>7.1%} {mk:>7.1%} {ma-mk:>+6.1%} {len(grp):>7d}")


# ---------------------------------------------------------------------------
# 4. Feature completeness
# ---------------------------------------------------------------------------
def report_features(df: pd.DataFrame):
    section("FEATURE COMPLETENESS")

    all_with_model = df[df['model_p1_prob'].notna()]
    complete = all_with_model[all_with_model['features_complete'].fillna(True).astype(bool)]
    incomplete = all_with_model[~all_with_model['features_complete'].fillna(True).astype(bool)]

    print(f"  Predictions with model output: {len(all_with_model)}")
    print(f"  Complete features:             {len(complete)}")
    print(f"  Incomplete (excluded):         {len(incomplete)}")

    # Also count matches that were skipped entirely (no model output)
    no_model = df[df['model_p1_prob'].isna()]
    print(f"  No model prediction (skipped): {len(no_model)}")

    if len(incomplete) > 0:
        divider()
        print(f"  Top defaulted features:")
        all_defaults = []
        for _, r in incomplete.iterrows():
            d = str(r.get('defaulted_features', ''))
            if d.strip():
                all_defaults.extend([x.strip() for x in d.split(',')])
        for feat, cnt in Counter(all_defaults).most_common(10):
            print(f"    {feat}: {cnt}x")


def report_logging_quality(df: pd.DataFrame):
    section("LOGGING QUALITY")

    if 'logging_quality' in df.columns:
        counts = df['logging_quality'].fillna('unknown').value_counts()
        print("  Logging lineage:")
        for label, count in counts.items():
            print(f"    {label}: {count}")

    if 'rescore_quality' in df.columns:
        counts = df['rescore_quality'].fillna('unknown').value_counts()
        print("  Rescore lineage:")
        for label, count in counts.items():
            print(f"    {label}: {count}")

    if 'record_status' in df.columns:
        counts = df['record_status'].fillna('unknown').value_counts()
        print("  Record status:")
        for label, count in counts.items():
            print(f"    {label}: {count}")


# ---------------------------------------------------------------------------
# 5. Duplicate check
# ---------------------------------------------------------------------------
def report_duplicates(df: pd.DataFrame):
    section("DUPLICATE CHECK")

    unsettled = df[df['actual_winner'].isna()].copy()
    if 'record_status' in unsettled.columns:
        unsettled = unsettled[~unsettled['record_status'].isin(['stale_no_model'])]

    def pair_key(row):
        names = sorted([str(row.p1).lower().strip(), str(row.p2).lower().strip()])
        return f"{names[0]} vs {names[1]}"

    unsettled['pair'] = unsettled.apply(pair_key, axis=1)
    dupes = unsettled.groupby('pair').filter(lambda g: len(g) > 1)

    if len(dupes) == 0:
        print(f"  No duplicates found in {len(unsettled)} unsettled predictions.")
        return

    n_extra = len(dupes) - dupes['pair'].nunique()
    print(f"  Found {n_extra} potential duplicate(s) in unsettled predictions:")
    for pair, grp in dupes.groupby('pair'):
        # Only flag if dates are within 3 days of each other
        dates = pd.to_datetime(grp['match_date'], errors='coerce').dt.date
        if dates.notna().all():
            spread = (dates.max() - dates.min()).days
            if spread > 5:
                continue  # likely different actual matches
        print(f"\n    {pair}:")
        for _, r in grp.iterrows():
            mdl = f"{r.model_p1_prob:.2f}" if pd.notna(r.model_p1_prob) else "nan"
            print(f"      date={r.match_date} | model={mdl} mkt={r.market_p1_prob:.2f} | logged={str(r.logged_at)[:16]} | {r.tournament}")


# ---------------------------------------------------------------------------
# 6. Pending summary
# ---------------------------------------------------------------------------
def report_pending(df: pd.DataFrame):
    section("PENDING / UNSETTLED")

    unsettled = df[df['actual_winner'].isna()]
    has_model = unsettled[unsettled['model_p1_prob'].notna()]
    no_model = unsettled[unsettled['model_p1_prob'].isna()]

    print(f"  Total unsettled:    {len(unsettled)}")
    print(f"  With model pred:    {len(has_model)}")
    print(f"  Without model pred: {len(no_model)}")

    if 'record_status' in unsettled.columns:
        divider()
        print("  Unsettled by status:")
        for label, count in unsettled['record_status'].fillna('unknown').value_counts().items():
            print(f"    {label}: {count}")

    if len(has_model) > 0:
        divider()
        print(f"  Upcoming with model predictions by date:")
        for date, grp in has_model.groupby('match_date'):
            print(f"    {date}: {len(grp)} matches")

    if len(no_model) > 0:
        divider()
        stale = no_model[no_model.get('record_status', pd.Series(index=no_model.index)).eq('stale_no_model')]
        pending_no_model = no_model[no_model.get('record_status', pd.Series(index=no_model.index)).ne('stale_no_model')]
        print(f"  No-model rows:")
        print(f"    Stale legacy rows: {len(stale)}")
        print(f"    Recent/pending no-model rows: {len(pending_no_model)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Prediction log analysis")
    parser.add_argument('--version', '-v', type=str, help='Filter to model version (e.g. v1.2.1)')
    parser.add_argument('--pending', action='store_true', help='Show pending/unsettled summary only')
    args = parser.parse_args()

    if not LOG_PATH.exists():
        print("No prediction_log.csv found.")
        sys.exit(1)

    df = upgrade_prediction_log(LOG_PATH, write=False)

    if args.pending:
        report_pending(df)
        return

    report_accuracy(df, args.version)
    report_calibration(df, args.version)
    report_edge_analysis(df, args.version)
    report_features(df)
    report_logging_quality(df)
    report_duplicates(df)
    report_pending(df)


if __name__ == '__main__':
    main()
