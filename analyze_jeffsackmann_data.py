#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

def analyze_jeffsackmann_data():
    """Analyze data quality in the JeffSackmann ML dataset"""
    
    print("=" * 60)
    print("JEFFSACKMANN DATA QUALITY ANALYSIS")
    print("=" * 60)
    
    # Load the ML-ready dataset
    ml_path = "/app/data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv"
    df = pd.read_csv(ml_path, low_memory=False)
    
    print(f"\n1. OVERALL DATASET:")
    print(f"   Total matches: {len(df):,}")
    print(f"   Date range: {df['tourney_date'].min()} to {df['tourney_date'].max()}")
    
    # Check for handedness columns
    print(f"\n2. DATASET COLUMNS:")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Available columns: {list(df.columns)}")
    
    print(f"\n3. HANDEDNESS COLUMNS:")
    hand_cols = [col for col in df.columns if 'Hand' in col]
    print(f"   Hand columns found: {hand_cols}")
    
    # Check handedness data availability
    if hand_cols:
        print(f"   Handedness data (non-zero values):")
        for col in hand_cols:
            non_zero = (df[col] == 1).sum()
            print(f"     {col}: {non_zero:,} matches")
    
    # Filter to 1990+ and check data quality
    df_1990 = df[df['year'] >= 1990].copy()
    print(f"\n4. POST-1990 DATA QUALITY:")
    print(f"   Total 1990+ matches: {len(df_1990):,}")
    
    # Key data availability
    rank_avail = df_1990['Player1_Rank'].notna().sum()
    age_avail = df_1990['Player1_Age'].notna().sum()
    height_avail = df_1990['Player1_Height'].notna().sum()
    points_avail = df_1990['Player1_Rank_Points'].notna().sum()
    
    print(f"   ATP Ranking: {rank_avail:,} ({rank_avail/len(df_1990)*100:.1f}%)")
    print(f"   Age: {age_avail:,} ({age_avail/len(df_1990)*100:.1f}%)")
    print(f"   Height: {height_avail:,} ({height_avail/len(df_1990)*100:.1f}%)")
    print(f"   Rank Points: {points_avail:,} ({points_avail/len(df_1990)*100:.1f}%)")
    
    # Look at matches with good data quality
    good_data = df_1990.dropna(subset=['Player1_Rank', 'Player2_Rank', 'Player1_Age', 'Player2_Age'])
    print(f"\n5. HIGH-QUALITY SUBSET (with ranks + age):")
    print(f"   Matches: {len(good_data):,}")
    
    if len(good_data) > 0:
        height_pct = good_data['Player1_Height'].notna().mean() * 100
        points_pct = good_data['Player1_Rank_Points'].notna().mean() * 100
        
        print(f"   Height availability: {height_pct:.1f}%")
        print(f"   Points availability: {points_pct:.1f}%")
        
        # Check year distribution
        print(f"   Year range: {good_data['year'].min()} to {good_data['year'].max()}")
        print(f"   Matches by decade:")
        decade_counts = good_data.groupby(good_data['year'] // 10 * 10).size()
        for decade, count in decade_counts.items():
            print(f"     {decade}s: {count:,} matches")
    
    print(f"\n6. TOURNAMENT LEVEL ANALYSIS:")
    level_cols = [col for col in df.columns if 'Level_' in col]
    print(f"   Tournament level columns: {level_cols}")
    
    if level_cols:
        print(f"   Tournament level distribution (1990+):")
        for col in level_cols:
            matches = (df_1990[col] == 1).sum()
            if matches > 0:
                print(f"     {col}: {matches:,} matches")
    
    print(f"\n7. RECOMMENDATION:")
    if len(good_data) > 50000:
        print(f"   ✅ Good dataset: {len(good_data):,} matches with ranks + age")
        print(f"   Consider filtering to this subset for better feature quality")
    elif rank_avail > 500000:
        print(f"   ⚠️  Large dataset but sparse features: {rank_avail:,} matches with rankings")
        print(f"   Model will mainly rely on ranking features")
    else:
        print(f"   ❌ Data quality issues: Only {rank_avail:,} matches with rankings")

if __name__ == "__main__":
    analyze_jeffsackmann_data()