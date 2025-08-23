#!/usr/bin/env python3
"""
Check for duplicate rows in pure bet logs.
"""

import pandas as pd
import os
from pathlib import Path

def check_duplicates():
    pure_logs_dir = Path("/Users/zachdodson/Documents/betting-algo/analysis_scripts/pure_bet_logs")
    
    if not pure_logs_dir.exists():
        print(f"❌ Directory not found: {pure_logs_dir}")
        return
    
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    
    print("🔍 Checking for duplicate rows in pure bet logs...")
    print("=" * 70)
    
    for model in models:
        csv_path = pure_logs_dir / f"{model}_pure_bets.csv"
        
        if not csv_path.exists():
            print(f"⚠️  {model:<20} - File not found: {csv_path}")
            continue
        
        # Load the CSV
        df = pd.read_csv(csv_path)
        total_rows = len(df)
        
        # Check for complete duplicates (all columns identical)
        complete_dupes = df.duplicated().sum()
        
        # Check for duplicates ignoring model column (since it's the same for all rows in one file)
        key_cols = [col for col in df.columns if col != 'model']
        key_dupes = df.duplicated(subset=key_cols).sum()
        
        # Check for duplicates on core betting info (date, players, bet_on_player)
        core_cols = ['date', 'player1', 'player2', 'bet_on_player']
        if all(col in df.columns for col in core_cols):
            core_dupes = df.duplicated(subset=core_cols).sum()
        else:
            core_dupes = "N/A"
        
        print(f"{model:<20} - Total rows: {total_rows:4,} | Complete dupes: {complete_dupes:3} | Key dupes: {key_dupes:3} | Core dupes: {core_dupes}")
        
        # If there are duplicates, show a few examples
        if complete_dupes > 0:
            print(f"  📋 First few complete duplicates:")
            duped_rows = df[df.duplicated(keep=False)].sort_values(['date', 'player1', 'player2'])
            print(duped_rows[['date', 'player1', 'player2', 'bet_on_player', 'odds', 'prob']].head(6).to_string(index=False))
            print()
    
    print("=" * 70)
    print("✅ Duplicate check complete!")

if __name__ == "__main__":
    check_duplicates()