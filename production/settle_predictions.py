#!/usr/bin/env python3
"""
Settle logged predictions against actual results.
Run after matches complete to record outcomes and compute live accuracy.
Usage: python3 settle_predictions.py
"""
import pandas as pd
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_log.csv')

def load_log():
    if not os.path.exists(LOG_PATH):
        print("No prediction log found yet.")
        return None
    return pd.read_csv(LOG_PATH)

def show_pending(df):
    pending = df[df['actual_winner'].isna()]
    if pending.empty:
        print("No unsettled predictions.")
        return
    print(f"\n=== {len(pending)} UNSETTLED PREDICTIONS ===")
    for i, row in pending.iterrows():
        print(f"  [{i}] {row['p1']} vs {row['p2']} ({row['tournament']}, {row['match_date']})")
        print(f"      Model: {row['model_p1_prob']:.0%} P1 | Market: {row['market_p1_prob']:.0%} P1")

def settle(df, idx, winner):
    """winner: 1 for P1, 2 for P2"""
    row = df.loc[idx]
    df.at[idx, 'actual_winner'] = winner
    df.at[idx, 'settled_at'] = datetime.now().isoformat()
    model_correct = (winner == 1 and row['model_p1_prob'] > 0.5) or (winner == 2 and row['model_p1_prob'] < 0.5)
    market_correct = (winner == 1 and row['market_p1_prob'] > 0.5) or (winner == 2 and row['market_p1_prob'] < 0.5)
    df.at[idx, 'model_correct'] = int(model_correct)
    df.at[idx, 'market_correct'] = int(market_correct)
    df.to_csv(LOG_PATH, index=False)
    print(f"  Settled: {row['p1']} vs {row['p2']} → P{winner} wins | Model {'✓' if model_correct else '✗'} | Market {'✓' if market_correct else '✗'}")

def show_stats(df):
    settled = df[df['actual_winner'].notna()].copy()
    if settled.empty:
        print("No settled predictions yet.")
        return
    n = len(settled)
    model_acc = settled['model_correct'].mean()
    market_acc = settled['market_correct'].mean()
    print(f"\n=== LIVE ACCURACY ({n} settled matches) ===")
    print(f"  Model:  {model_acc:.1%}")
    print(f"  Market: {market_acc:.1%}")
    print(f"  Edge:   {model_acc - market_acc:+.1%}")

if __name__ == '__main__':
    df = load_log()
    if df is not None:
        show_pending(df)
        show_stats(df)
