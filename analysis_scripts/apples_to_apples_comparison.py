#!/usr/bin/env python3
"""
Create apples-to-apples comparison by filtering to same tournament levels
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
    betting_correct = (odds_clean[win_col] < odds_clean[lose_col]).sum()
    betting_accuracy = betting_correct / len(odds_clean)
    
    return betting_accuracy, len(odds_clean)

print("==" * 50)
print("APPLES-TO-APPLES COMPARISON: HIGH-LEVEL TOURNAMENTS ONLY")
print("==" * 50)

# Load datasets
print("\n1. Loading datasets...")
jeff_data = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

# Fix date formats
jeff_data['tourney_date'] = pd.to_datetime(jeff_data['tourney_date'], format='%Y%m%d')
jeff_data['year'] = jeff_data['tourney_date'].dt.year
tennis_data['Date'] = pd.to_datetime(tennis_data['Date'])
tennis_data['year'] = tennis_data['Date'].dt.year

# Filter to 2023-2024 only (exclude 2025 for fair comparison)
print("\n2. Filtering to 2023-2024 test period...")
jeff_test = jeff_data[jeff_data['year'].isin([2023, 2024])].copy()
tennis_test = tennis_data[tennis_data['year'].isin([2023, 2024])].copy()

print(f"   Jeff Sackmann 2023-2024: {len(jeff_test):,} total matches")
print(f"   Tennis-Data 2023-2024: {len(tennis_test):,} total matches")

# Define high-level tournaments for comparison
print("\n3. Filtering to high-level tournaments...")

# Jeff Sackmann: G (Grand Slams), M (Masters), F (Finals)
# Note: ATP 500s are coded as 'A' but we need to identify them by prize money or other means
# For now, let's focus on clear categories: G, M, F
jeff_high_level = jeff_test[jeff_test['tourney_level'].isin(['G', 'M', 'F'])].copy()

# Tennis-Data: Grand Slam, Masters 1000, ATP500, Masters Cup
tennis_high_level = tennis_test[tennis_test['Series'].isin(['Grand Slam', 'Masters 1000', 'ATP500', 'Masters Cup'])].copy()

print(f"   Jeff Sackmann high-level (G+M+F): {len(jeff_high_level):,} matches")
print(f"   Tennis-Data high-level (GS+M1000+500+Cup): {len(tennis_high_level):,} matches")

# Break down by category
print("\n4. Tournament level breakdown:")
print("   Jeff Sackmann:")
jeff_breakdown = jeff_high_level['tourney_level'].value_counts()
level_names = {'G': 'Grand Slams', 'M': 'Masters 1000', 'F': 'Finals'}
for level, count in jeff_breakdown.items():
    print(f"     {level_names.get(level, level)}: {count:,} matches")

print("\n   Tennis-Data:")
tennis_breakdown = tennis_high_level['Series'].value_counts()
for level, count in tennis_breakdown.items():
    print(f"     {level}: {count:,} matches")

# Let's also check if we can identify ATP 500s in Jeff Sackmann data
print("\n5. Attempting to identify ATP 500s in Jeff Sackmann...")
# ATP 500s are level 'A' but are the higher-tier ATP events
jeff_atp_a = jeff_test[jeff_test['tourney_level'] == 'A'].copy()
print(f"   Jeff Sackmann 'A' level tournaments: {len(jeff_atp_a):,} matches")

# Get unique tournament names to see if we can identify 500s
jeff_atp_names = jeff_atp_a['tourney_name'].value_counts()
print(f"   Unique 'A' level tournaments: {len(jeff_atp_names)}")

# Tennis-Data ATP500 tournaments for reference
tennis_500_names = tennis_high_level[tennis_high_level['Series'] == 'ATP500']['Tournament'].value_counts()
print(f"   Tennis-Data ATP500 tournaments: {len(tennis_500_names)}")

print("\n   Tennis-Data ATP500 tournament names:")
for name, count in tennis_500_names.head(10).items():
    print(f"     {name}: {count} matches")

# Try to match some common ATP 500 tournaments
atp_500_keywords = ['Barcelona', 'Hamburg', 'Washington', 'Beijing', 'Tokyo', 'Basel', 'Vienna', 'Rotterdam', 'Dubai', 'Acapulco', 'Rio', 'Memphis', 'Marseille', 'Stockholm', 'St. Petersburg', 'Queen', 'Halle', 'Eastbourne']

jeff_likely_500s = jeff_atp_a[jeff_atp_a['tourney_name'].str.contains('|'.join(atp_500_keywords), case=False, na=False)]
print(f"\n   Jeff Sackmann likely ATP 500s (by name matching): {len(jeff_likely_500s):,} matches")

# Create revised high-level comparison including likely 500s
jeff_high_level_plus = pd.concat([jeff_high_level, jeff_likely_500s]).drop_duplicates()
print(f"   Jeff Sackmann high-level + likely 500s: {len(jeff_high_level_plus):,} matches")

# Now let's do the comparison on the most comparable subset
print("\n" + "==" * 50)
print("FINAL APPLES-TO-APPLES COMPARISON")
print("==" * 50)

# Use the most restrictive common set: Grand Slams + Masters only
jeff_gs_masters = jeff_test[jeff_test['tourney_level'].isin(['G', 'M'])].copy()
tennis_gs_masters = tennis_test[tennis_test['Series'].isin(['Grand Slam', 'Masters 1000'])].copy()

print(f"\nGrand Slams + Masters Only:")
print(f"   Jeff Sackmann: {len(jeff_gs_masters):,} matches")
print(f"   Tennis-Data: {len(tennis_gs_masters):,} matches")
print(f"   Difference: {abs(len(jeff_gs_masters) - len(tennis_gs_masters)):,} matches ({abs(len(jeff_gs_masters) - len(tennis_gs_masters))/max(len(jeff_gs_masters), len(tennis_gs_masters))*100:.1f}%)")

# Calculate accuracies for Grand Slams + Masters
print(f"\n1. Jeff Sackmann Grand Slams + Masters (2023-2024):")
# Clean ranking data
jeff_clean = jeff_gs_masters.dropna(subset=['winner_rank', 'loser_rank']).copy()
jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])

# ATP baseline (winner has better rank)
jeff_atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
jeff_atp_accuracy = jeff_atp_correct / len(jeff_clean)

print(f"   Matches with rankings: {len(jeff_clean):,}")
print(f"   ATP Ranking Baseline: {jeff_atp_accuracy:.4f} ({jeff_atp_accuracy*100:.2f}%)")

print(f"\n2. Tennis-Data Grand Slams + Masters (2023-2024):")
# Clean ranking data
tennis_clean = tennis_gs_masters.dropna(subset=['WRank', 'LRank']).copy()
tennis_clean['WRank'] = pd.to_numeric(tennis_clean['WRank'], errors='coerce')
tennis_clean['LRank'] = pd.to_numeric(tennis_clean['LRank'], errors='coerce')
tennis_clean = tennis_clean.dropna(subset=['WRank', 'LRank'])

# ATP baseline
tennis_atp_correct = (tennis_clean['WRank'] < tennis_clean['LRank']).sum()
tennis_atp_accuracy = tennis_atp_correct / len(tennis_clean)

# Betting odds accuracy
betting_acc, betting_count = calculate_betting_accuracy(tennis_clean, 'AvgW', 'AvgL')

print(f"   Matches with rankings: {len(tennis_clean):,}")
print(f"   ATP Ranking Baseline: {tennis_atp_accuracy:.4f} ({tennis_atp_accuracy*100:.2f}%)")
if betting_acc is not None:
    print(f"   Average Odds Accuracy: {betting_acc:.4f} ({betting_acc*100:.2f}%) | N: {betting_count:,}")

print(f"\n" + "==" * 50)
print("SUMMARY FOR MODEL COMPARISON")
print("==" * 50)
print(f"Dataset: Grand Slams + Masters 1000 tournaments only (2023-2024)")
print(f"Jeff Sackmann: {len(jeff_clean):,} matches")
print(f"Tennis-Data: {len(tennis_clean):,} matches")
print(f"Sample size difference: {abs(len(jeff_clean) - len(tennis_clean)):,} matches ({abs(len(jeff_clean) - len(tennis_clean))/max(len(jeff_clean), len(tennis_clean))*100:.1f}%)")
print(f"")
print(f"ATP Ranking Baselines:")
print(f"  Jeff Sackmann: {jeff_atp_accuracy*100:.2f}%")
print(f"  Tennis-Data: {tennis_atp_accuracy*100:.2f}%")
print(f"")
if betting_acc is not None:
    print(f"Performance Targets for Your Models:")
    print(f"  ATP Baseline: ~{jeff_atp_accuracy*100:.1f}%")
    print(f"  Professional Betting Odds: {betting_acc*100:.2f}%")
    print(f"  Your XGBoost Model: [Need to run on GS+Masters subset]")

# Save the filtered Jeff Sackmann data for model re-evaluation
print(f"\n3. Saving filtered dataset for model re-evaluation...")
jeff_clean.to_csv('data/JeffSackmann/jeffsackmann_gs_masters_2023_2024.csv', index=False)
print(f"   Saved: data/JeffSackmann/jeffsackmann_gs_masters_2023_2024.csv")
print(f"   This dataset can be used to re-run your models on the same tournament levels as Tennis-Data")