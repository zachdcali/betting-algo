#!/usr/bin/env python3
"""
Check data quality of Tennis-Data.co.uk odds
"""
import pandas as pd
import numpy as np

print("="*80)
print("CHECKING TENNIS-DATA.CO.UK ODDS DATA QUALITY")
print("="*80)

# Load tennis data
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)

print(f"Total matches: {len(tennis_matched):,}")

# Check for missing odds
print(f"\nMissing AvgW: {tennis_matched['AvgW'].isna().sum():,}")
print(f"Missing AvgL: {tennis_matched['AvgL'].isna().sum():,}")

# Convert to numeric
tennis_matched['AvgW'] = pd.to_numeric(tennis_matched['AvgW'], errors='coerce')
tennis_matched['AvgL'] = pd.to_numeric(tennis_matched['AvgL'], errors='coerce')

# Remove missing values
odds_data = tennis_matched.dropna(subset=['AvgW', 'AvgL']).copy()
print(f"Matches with both odds: {len(odds_data):,}")

# Check for identical odds
identical_odds = odds_data[odds_data['AvgW'] == odds_data['AvgL']]
print(f"\nMatches with IDENTICAL winner/loser odds: {len(identical_odds):,} ({len(identical_odds)/len(odds_data)*100:.1f}%)")

# Check odds distribution
print(f"\nAvgW (winner odds) statistics:")
print(f"  Min: {odds_data['AvgW'].min():.2f}")
print(f"  Max: {odds_data['AvgW'].max():.2f}")
print(f"  Mean: {odds_data['AvgW'].mean():.2f}")
print(f"  Median: {odds_data['AvgW'].median():.2f}")

print(f"\nAvgL (loser odds) statistics:")
print(f"  Min: {odds_data['AvgL'].min():.2f}")
print(f"  Max: {odds_data['AvgL'].max():.2f}")
print(f"  Mean: {odds_data['AvgL'].mean():.2f}")
print(f"  Median: {odds_data['AvgL'].median():.2f}")

# Check for suspicious odds
print(f"\nSuspicious odds patterns:")
print(f"  AvgW < 1.01: {(odds_data['AvgW'] < 1.01).sum():,}")
print(f"  AvgW > 50: {(odds_data['AvgW'] > 50).sum():,}")
print(f"  AvgL < 1.01: {(odds_data['AvgL'] < 1.01).sum():,}")
print(f"  AvgL > 50: {(odds_data['AvgL'] > 50).sum():,}")

# Show some examples of identical odds
if len(identical_odds) > 0:
    print(f"\nExamples of matches with identical odds:")
    for i, (_, row) in enumerate(identical_odds.head(5).iterrows()):
        print(f"  {row['Winner']} def. {row['Loser']} | Both at {row['AvgW']:.2f} odds")

# Check if winner always has lower odds (as it should)
winner_favored = odds_data[odds_data['AvgW'] < odds_data['AvgL']]
print(f"\nMatches where winner had lower odds: {len(winner_favored):,} ({len(winner_favored)/len(odds_data)*100:.1f}%)")
print(f"Matches where LOSER had lower odds: {len(odds_data) - len(winner_favored):,} ({(len(odds_data) - len(winner_favored))/len(odds_data)*100:.1f}%)")

# Check overround (vig)
odds_data['implied_prob_winner'] = 1 / odds_data['AvgW']
odds_data['implied_prob_loser'] = 1 / odds_data['AvgL']
odds_data['overround'] = odds_data['implied_prob_winner'] + odds_data['implied_prob_loser']

print(f"\nOverround (total implied probability) statistics:")
print(f"  Min: {odds_data['overround'].min():.3f}")
print(f"  Max: {odds_data['overround'].max():.3f}")
print(f"  Mean: {odds_data['overround'].mean():.3f}")
print(f"  Should be > 1.0 (bookmaker margin)")

# Check for overrounds that are exactly 1.0 (suspicious)
exact_100_percent = odds_data[abs(odds_data['overround'] - 1.0) < 0.001]
print(f"  Matches with exactly 100% overround: {len(exact_100_percent):,}")

print(f"\n" + "="*80)
print(f"DIAGNOSIS:")
print(f"="*80)
if len(identical_odds) > len(odds_data) * 0.1:
    print(f"❌ MAJOR DATA QUALITY ISSUE: {len(identical_odds)/len(odds_data)*100:.1f}% of matches have identical odds")
    print(f"   This is impossible in real betting markets")
if (odds_data['AvgW'] >= odds_data['AvgL']).sum() > len(odds_data) * 0.1:
    print(f"❌ LOGIC ERROR: Many winners have higher odds than losers")
if odds_data['overround'].mean() < 1.02:
    print(f"❌ UNREALISTIC: Average overround too low for real betting markets")

print(f"\nThis data quality issue explains the unrealistic betting returns!")