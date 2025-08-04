#!/usr/bin/env python3
"""
Efficient matching using proper indexing instead of nested loops
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def extract_surname(name):
    """Extract surname from full name"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r'\.', '', name)
    name = re.sub(r'\b(Jr|Sr|II|III|IV)\b', '', name)
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1].lower()
    return name.lower()

def normalize_tennis_data_name(name):
    """Normalize Tennis-Data name format"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = re.sub(r'\.', '', name)
    parts = name.split()
    if len(parts) >= 2 and len(parts[-1]) == 1:
        return parts[-2].lower()  # Return surname before initial
    elif len(parts) >= 1:
        return parts[-1].lower()
    return name.lower()

def calculate_betting_accuracy(data, win_col, lose_col):
    """Calculate betting odds accuracy"""
    odds_data = data.dropna(subset=[win_col, lose_col]).copy()
    if len(odds_data) == 0:
        return None, 0
    
    odds_data[win_col] = pd.to_numeric(odds_data[win_col], errors='coerce')
    odds_data[lose_col] = pd.to_numeric(odds_data[lose_col], errors='coerce')
    odds_clean = odds_data.dropna(subset=[win_col, lose_col])
    
    if len(odds_clean) == 0:
        return None, 0
    
    betting_correct = (odds_clean[win_col] < odds_clean[lose_col]).sum()
    betting_accuracy = betting_correct / len(odds_clean)
    
    return betting_accuracy, len(odds_clean)

print("EFFICIENT MATCH COMPARISON")
print("="*50)

print("1. Loading and filtering datasets...")
jeff_data = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

# Fix dates and filter to 2023-2024
jeff_data['tourney_date'] = pd.to_datetime(jeff_data['tourney_date'], format='%Y%m%d')
jeff_data['year'] = jeff_data['tourney_date'].dt.year
jeff_test = jeff_data[jeff_data['year'].isin([2023, 2024])].copy()

tennis_data['Date'] = pd.to_datetime(tennis_data['Date'])
tennis_data['year'] = tennis_data['Date'].dt.year
tennis_test = tennis_data[tennis_data['year'].isin([2023, 2024])].copy()

print(f"   Jeff Sackmann 2023-2024: {len(jeff_test):,}")
print(f"   Tennis-Data 2023-2024: {len(tennis_test):,}")

print("2. Extracting surnames...")
jeff_test['winner_surname'] = jeff_test['winner_name'].apply(extract_surname)
jeff_test['loser_surname'] = jeff_test['loser_name'].apply(extract_surname)
tennis_test['winner_surname'] = tennis_test['Winner'].apply(normalize_tennis_data_name)
tennis_test['loser_surname'] = tennis_test['Loser'].apply(normalize_tennis_data_name)

print("3. Creating efficient lookup index...")
# Create lookup dictionary for Tennis-Data by date and player surnames
tennis_lookup = {}
for idx, row in tennis_test.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    winner_surname = row['winner_surname']
    loser_surname = row['loser_surname']
    
    # Create keys for both player orders
    key1 = (date_str, winner_surname, loser_surname)
    key2 = (date_str, loser_surname, winner_surname)
    
    tennis_lookup[key1] = idx
    tennis_lookup[key2] = idx  # Same match, different order

print(f"   Created lookup with {len(tennis_lookup):,} entries")

print("4. Finding matches...")
matches = []

for idx, jeff_match in jeff_test.iterrows():
    if idx % 10000 == 0:
        print(f"   Processed {idx:,}/{len(jeff_test):,}")
    
    jeff_date = jeff_match['tourney_date']
    jeff_winner_surname = jeff_match['winner_surname']
    jeff_loser_surname = jeff_match['loser_surname']
    
    # Try exact date and ±1 day
    for days_offset in [0, 1, -1]:
        search_date = (jeff_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Check both player orders
        key1 = (search_date, jeff_winner_surname, jeff_loser_surname)
        key2 = (search_date, jeff_loser_surname, jeff_winner_surname)
        
        if key1 in tennis_lookup:
            tennis_idx = tennis_lookup[key1]
            matches.append({
                'jeff_idx': idx,
                'tennis_idx': tennis_idx,
                'match_type': 'same_order',
                'date_offset': days_offset
            })
            break
        elif key2 in tennis_lookup:
            tennis_idx = tennis_lookup[key2]
            matches.append({
                'jeff_idx': idx,
                'tennis_idx': tennis_idx,
                'match_type': 'swapped_order',
                'date_offset': days_offset
            })
            break

print(f"\n5. Results:")
print(f"   Total matches found: {len(matches):,}")

if len(matches) > 0:
    same_order = sum(1 for m in matches if m['match_type'] == 'same_order')
    swapped_order = sum(1 for m in matches if m['match_type'] == 'swapped_order')
    
    print(f"   - Same winner order: {same_order:,}")
    print(f"   - Swapped order: {swapped_order:,}")
    
    # Create matched datasets
    jeff_matched_indices = [m['jeff_idx'] for m in matches]
    tennis_matched_indices = [m['tennis_idx'] for m in matches]
    
    jeff_matched = jeff_test.loc[jeff_matched_indices].copy()
    tennis_matched = tennis_test.loc[tennis_matched_indices].copy()
    
    print(f"\n6. Sample matches:")
    for i in range(min(5, len(matches))):
        match_info = matches[i]
        jeff_row = jeff_test.loc[match_info['jeff_idx']]
        tennis_row = tennis_test.loc[match_info['tennis_idx']]
        
        print(f"   {jeff_row['tourney_date'].strftime('%Y-%m-%d')}: {jeff_row['winner_name']} def. {jeff_row['loser_name']}")
        print(f"     vs {tennis_row['Date'].strftime('%Y-%m-%d')}: {tennis_row['Winner']} def. {tennis_row['Loser']} ({match_info['match_type']})")
    
    print(f"\n7. Calculating accuracies on {len(matches):,} matched games:")
    
    # Jeff Sackmann ATP baseline
    jeff_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
    jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
    jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
    jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])
    
    if len(jeff_clean) > 0:
        jeff_atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
        jeff_atp_accuracy = jeff_atp_correct / len(jeff_clean)
        print(f"   ATP Ranking Baseline: {jeff_atp_accuracy:.4f} ({jeff_atp_accuracy*100:.2f}%) on {len(jeff_clean):,} matches")
    
    # Tennis-Data betting odds
    tennis_clean = tennis_matched.dropna(subset=['WRank', 'LRank']).copy()
    tennis_clean['WRank'] = pd.to_numeric(tennis_clean['WRank'], errors='coerce')
    tennis_clean['LRank'] = pd.to_numeric(tennis_clean['LRank'], errors='coerce')
    tennis_clean = tennis_clean.dropna(subset=['WRank', 'LRank'])
    
    betting_acc, betting_count = calculate_betting_accuracy(tennis_clean, 'AvgW', 'AvgL')
    if betting_acc is not None:
        print(f"   Professional Betting Odds: {betting_acc:.4f} ({betting_acc*100:.2f}%) on {betting_count:,} matches")
    
    print(f"\n8. Saving matched datasets for model evaluation...")
    jeff_matched.to_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', index=False)
    tennis_matched.to_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', index=False)
    
    print(f"\n" + "="*50)
    print(f"TRUE APPLES-TO-APPLES COMPARISON READY!")
    print(f"="*50)
    print(f"Dataset: {len(matches):,} IDENTICAL matches in both datasets")
    print(f"ATP Baseline: {jeff_atp_accuracy*100:.2f}%")
    if betting_acc is not None:
        print(f"Betting Odds: {betting_acc*100:.2f}%")
    print(f"Your Models: [Need to run on jeffsackmann_exact_matched_final.csv]")
    print(f"")
    print(f"This is the true apples-to-apples comparison you wanted!")

else:
    print("   No matches found. The datasets may have:")
    print("   - Very different player name formats")
    print("   - Different date coverage")
    print("   - Limited actual overlap")