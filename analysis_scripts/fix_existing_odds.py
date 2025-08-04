#!/usr/bin/env python3
"""
Fix odds in existing matched datasets
"""
import pandas as pd
import numpy as np

print("="*80)
print("FIXING ODDS IN EXISTING MATCHED DATASETS")
print("="*80)

# Load existing matched datasets
print("\n1. Loading existing matched datasets...")
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)

print(f"   Jeff matches: {len(jeff_matched):,}")
print(f"   Tennis matches: {len(tennis_matched):,}")

# Check current odds quality
print("\n2. Checking current odds quality...")
tennis_matched['AvgW'] = pd.to_numeric(tennis_matched['AvgW'], errors='coerce')
tennis_matched['AvgL'] = pd.to_numeric(tennis_matched['AvgL'], errors='coerce')
odds_data = tennis_matched.dropna(subset=['AvgW', 'AvgL'])

identical_odds = odds_data[odds_data['AvgW'] == odds_data['AvgL']]
print(f"   Matches with identical odds: {len(identical_odds):,} ({len(identical_odds)/len(odds_data)*100:.1f}%)")

# Get the original Tennis-Data.co.uk file to fix from
print("\n3. Loading original Tennis-Data.co.uk file...")
tennis_orig = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)
tennis_orig['Date'] = pd.to_datetime(tennis_orig['Date'])
tennis_orig = tennis_orig[tennis_orig['Date'].dt.year.isin([2023, 2024])]

print(f"   Original Tennis-Data 2023-2024: {len(tennis_orig):,} matches")

# Fix the matched Tennis-Data by looking up correct odds
print("\n4. Fixing odds by matching back to original data...")

fixed_rows = []
matches_fixed = 0
matches_not_found = 0

for i, tennis_row in tennis_matched.iterrows():
    if i % 100 == 0:
        print(f"   Processing match {i:,}/{len(tennis_matched):,}")
    
    # Try to find this match in the original data
    winner = str(tennis_row['Winner']).strip()
    loser = str(tennis_row['Loser']).strip()
    match_date = pd.to_datetime(tennis_row['Date'])
    
    # Look for exact match in original data
    candidates = tennis_orig[
        (tennis_orig['Winner'].str.strip() == winner) &
        (tennis_orig['Loser'].str.strip() == loser) &
        (tennis_orig['Date'] == match_date)
    ]
    
    if len(candidates) == 1:
        # Found exact match - use original odds
        orig_match = candidates.iloc[0]
        tennis_row_fixed = tennis_row.copy()
        
        # Copy all odds columns from original
        for col in ['AvgW', 'AvgL', 'B365W', 'B365L', 'MaxW', 'MaxL', 'PSW', 'PSL']:
            if col in orig_match:
                tennis_row_fixed[col] = orig_match[col]
        
        fixed_rows.append(tennis_row_fixed)
        matches_fixed += 1
    else:
        # Couldn't find match - keep original (possibly corrupted) data
        fixed_rows.append(tennis_row)
        matches_not_found += 1

tennis_fixed = pd.DataFrame(fixed_rows)

print(f"\n5. Fix results:")
print(f"   Matches fixed: {matches_fixed:,}")
print(f"   Matches not found: {matches_not_found:,}")

# Check odds quality after fix
print("\n6. Checking odds quality after fix...")
tennis_fixed['AvgW'] = pd.to_numeric(tennis_fixed['AvgW'], errors='coerce')
tennis_fixed['AvgL'] = pd.to_numeric(tennis_fixed['AvgL'], errors='coerce')
odds_fixed = tennis_fixed.dropna(subset=['AvgW', 'AvgL'])

identical_fixed = odds_fixed[odds_fixed['AvgW'] == odds_fixed['AvgL']]
print(f"   Matches with identical odds after fix: {len(identical_fixed):,} ({len(identical_fixed)/len(odds_fixed)*100:.1f}%)")

# Show some examples
print(f"\n7. Examples of fixed odds:")
print("   Match | AvgW Before | AvgL Before | AvgW After | AvgL After | Fixed?")
print("-" * 70)
for i in range(min(10, len(tennis_matched))):
    before_w = tennis_matched.iloc[i]['AvgW']
    before_l = tennis_matched.iloc[i]['AvgL']
    after_w = tennis_fixed.iloc[i]['AvgW']
    after_l = tennis_fixed.iloc[i]['AvgL']
    
    fixed = "YES" if (before_w != after_w or before_l != after_l) else "NO"
    winner = str(tennis_matched.iloc[i]['Winner'])[:15]
    
    print(f"   {winner:15s} | {before_w:8.2f} | {before_l:8.2f} | {after_w:8.2f} | {after_l:8.2f} | {fixed}")

# Save fixed dataset
print(f"\n8. Saving fixed dataset...")
tennis_fixed.to_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', index=False)

print(f"   ✅ Saved fixed Tennis-Data dataset")
print(f"   Ready for accurate betting analysis!")

print(f"\n" + "="*80)
print(f"ODDS FIX COMPLETE")
print(f"="*80)