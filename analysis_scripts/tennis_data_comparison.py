#!/usr/bin/env python3
"""
Quick comparison of Tennis-Data.co.uk vs Jeff Sackmann data for 2023-2024
"""
import pandas as pd
import numpy as np

def calculate_betting_accuracy(data, win_col, lose_col):
    """Calculate betting odds accuracy - favorite (lower decimal odds) should win"""
    odds_data = data.dropna(subset=[win_col, lose_col]).copy()
    if len(odds_data) == 0:
        return None, 0
    
    odds_data[win_col] = pd.to_numeric(odds_data[win_col], errors='coerce')
    odds_data[lose_col] = pd.to_numeric(odds_data[lose_col], errors='coerce')
    odds_clean = odds_data.dropna(subset=[win_col, lose_col])
    
    if len(odds_clean) == 0:
        return None, 0
    
    # Favorite (lower decimal odds) should win
    # Since Winner actually won, we check if Winner had lower odds than Loser
    betting_correct = (odds_clean[win_col] < odds_clean[lose_col]).sum()
    betting_accuracy = betting_correct / len(odds_clean)
    
    return betting_accuracy, len(odds_clean)

def analyze_tennis_data():
    print("=" * 70)
    print("TENNIS-DATA.CO.UK COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    
    # Load Tennis-Data.co.uk data
    print("\n1. Loading Tennis-Data.co.uk dataset...")
    tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)
    
    # Convert date 
    tennis_data['Date'] = pd.to_datetime(tennis_data['Date'])
    tennis_data['year'] = tennis_data['Date'].dt.year
    
    print(f"   Total matches 2000-2025: {len(tennis_data):,}")
    print(f"   Years covered: {tennis_data['year'].min()}-{tennis_data['year'].max()}")
    
    # Filter to ATP main tour events
    atp_levels = ['Masters Cup', 'Masters 1000', 'Grand Slam', 'ATP250', 'ATP500', 'International Series', 'International']
    atp_data = tennis_data[tennis_data['Series'].isin(atp_levels)].copy()
    
    print(f"\n2. Year-by-Year Betting Odds Accuracy Analysis:")
    print("   (Average odds across bookmakers - lower decimal odds = favorite should win)")
    print("   " + "-" * 60)
    
    # Analyze each year
    for year in sorted(tennis_data['year'].unique()):
        if year < 2000:  # Skip incomplete early years
            continue
            
        year_data = atp_data[atp_data['year'] == year].copy()
        if len(year_data) == 0:
            continue
            
        # Clean ranking data for ATP baseline
        year_clean = year_data.dropna(subset=['WRank', 'LRank']).copy()
        year_clean['WRank'] = pd.to_numeric(year_clean['WRank'], errors='coerce')
        year_clean['LRank'] = pd.to_numeric(year_clean['LRank'], errors='coerce')
        year_clean = year_clean.dropna(subset=['WRank', 'LRank'])
        
        if len(year_clean) == 0:
            continue
            
        # ATP baseline
        atp_correct = (year_clean['WRank'] < year_clean['LRank']).sum()
        atp_accuracy = atp_correct / len(year_clean) if len(year_clean) > 0 else 0
        
        # Betting odds accuracy (using average odds across bookmakers)
        betting_acc, betting_count = calculate_betting_accuracy(year_clean, 'AvgW', 'AvgL')
        
        if betting_acc is not None and betting_count > 50:
            print(f"   {year}: ATP {atp_accuracy:.3f} ({atp_accuracy*100:.1f}%) | AvgOdds {betting_acc:.3f} ({betting_acc*100:.1f}%) | N: {len(year_clean):,} | Odds: {betting_count:,}")
        else:
            print(f"   {year}: ATP {atp_accuracy:.3f} ({atp_accuracy*100:.1f}%) | AvgOdds N/A | N: {len(year_clean):,}")
    
    # Focus on 2023-2024 for comparison
    print(f"\n3. Detailed 2023-2024 Analysis (Test Years):")
    test_data = atp_data[atp_data['year'].isin([2023, 2024])].copy()
    print(f"   Total ATP-level matches: {len(test_data):,}")
    
    # Tournament breakdown
    print(f"\n   Tournament breakdown:")
    series_counts = test_data['Series'].value_counts()
    for series, count in series_counts.items():
        print(f"     {series}: {count:,} matches")
    
    # Clean data
    test_clean = test_data.dropna(subset=['WRank', 'LRank']).copy()
    test_clean['WRank'] = pd.to_numeric(test_clean['WRank'], errors='coerce')
    test_clean['LRank'] = pd.to_numeric(test_clean['LRank'], errors='coerce')
    test_clean = test_clean.dropna(subset=['WRank', 'LRank'])
    
    # ATP ranking accuracy
    atp_correct = (test_clean['WRank'] < test_clean['LRank']).sum()
    atp_accuracy = atp_correct / len(test_clean)
    
    # Betting odds accuracy (using average odds)
    betting_acc, betting_count = calculate_betting_accuracy(test_clean, 'AvgW', 'AvgL')
    
    print(f"\n4. 2023-2024 Results Summary:")
    print(f"   Matches with rankings: {len(test_clean):,}")
    print(f"   ATP Ranking Baseline: {atp_accuracy:.4f} ({atp_accuracy*100:.2f}%)")
    if betting_acc is not None:
        print(f"   Average Odds Accuracy: {betting_acc:.4f} ({betting_acc*100:.2f}%) | N: {betting_count:,}")
    
    print(f"\n5. Comparison to Jeff Sackmann:")
    print(f"   Jeff Sackmann ATP-Level (2023+): 8,133 matches")
    print(f"   Tennis-Data ATP-Level (2023-2024): {len(test_clean):,} matches")
    print(f"   -> Tennis-Data missing ~{8133 - len(test_clean):,} matches (likely 2025 data)")
    print(f"   ")
    print(f"   ATP Baselines: Jeff Sackmann 62.7% vs Tennis-Data {atp_accuracy*100:.1f}%")
    if betting_acc is not None:
        print(f"   Jeff Sackmann XGBoost: 68.4% vs Tennis-Data AvgOdds: {betting_acc*100:.1f}%")

if __name__ == "__main__":
    analyze_tennis_data()