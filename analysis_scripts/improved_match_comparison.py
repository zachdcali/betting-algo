#!/usr/bin/env python3
"""
Improved matching between Jeff Sackmann and Tennis-Data using fuzzy matching
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
from difflib import SequenceMatcher

def normalize_name_fuzzy(name):
    """More aggressive name normalization for fuzzy matching"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    
    # Remove common suffixes
    name = re.sub(r'\b(Jr|Sr|II|III|IV)\b', '', name)
    name = re.sub(r'\.', '', name)
    name = re.sub(r'-', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    
    # Extract last name and first initial for Tennis-Data format
    parts = name.split()
    if len(parts) >= 2:
        # For "Lastname F." format, return just lastname
        if len(parts[-1]) == 1 or (len(parts[-1]) == 2 and parts[-1].endswith('.')):
            return parts[-2].lower()  # Return surname before the initial
        else:
            return parts[-1].lower()  # Return last name
    
    return name.lower().strip()

def extract_surname(name):
    """Extract surname from full name"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    parts = name.split()
    if len(parts) >= 2:
        return parts[-1].lower()
    return name.lower()

def similarity(a, b):
    """Calculate similarity between two strings"""
    return SequenceMatcher(None, a, b).ratio()

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

print("="*80)
print("IMPROVED EXACT MATCH COMPARISON")
print("="*80)

# Load datasets
print("\n1. Loading datasets...")
jeff_data = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

# Fix date formats and filter to 2023-2024
print("\n2. Filtering to 2023-2024...")
jeff_data['tourney_date'] = pd.to_datetime(jeff_data['tourney_date'], format='%Y%m%d')
jeff_data['year'] = jeff_data['tourney_date'].dt.year
jeff_test = jeff_data[jeff_data['year'].isin([2023, 2024])].copy()

tennis_data['Date'] = pd.to_datetime(tennis_data['Date'])
tennis_data['year'] = tennis_data['Date'].dt.year
tennis_test = tennis_data[tennis_data['year'].isin([2023, 2024])].copy()

print(f"   Jeff Sackmann 2023-2024: {len(jeff_test):,} matches")
print(f"   Tennis-Data 2023-2024: {len(tennis_test):,} matches")

# Extract surnames for matching
print("\n3. Extracting surnames for matching...")
jeff_test['winner_surname'] = jeff_test['winner_name'].apply(extract_surname)
jeff_test['loser_surname'] = jeff_test['loser_name'].apply(extract_surname)

tennis_test['winner_surname'] = tennis_test['Winner'].apply(normalize_name_fuzzy)
tennis_test['loser_surname'] = tennis_test['Loser'].apply(normalize_name_fuzzy)

# Create date-based matching
print("\n4. Creating date-surname index for Tennis-Data...")
tennis_by_date_surnames = {}
for idx, row in tennis_test.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    key = (date_str, row['winner_surname'], row['loser_surname'])
    tennis_by_date_surnames[key] = idx

# Alternative key with swapped players
for idx, row in tennis_test.iterrows():
    date_str = row['Date'].strftime('%Y-%m-%d')
    key = (date_str, row['loser_surname'], row['winner_surname'])
    if key not in tennis_by_date_surnames:  # Don't overwrite exact matches
        tennis_by_date_surnames[f"swapped_{key}"] = idx

print(f"   Created {len(tennis_by_date_surnames):,} date-surname combinations")

# Find exact matches
print("\n5. Finding exact surname + date matches...")
exact_matches = []
fuzzy_matches = []

for idx, jeff_match in jeff_test.iterrows():
    if idx % 10000 == 0:
        print(f"   Processing {idx:,}/{len(jeff_test):,} ({idx/len(jeff_test)*100:.1f}%)")
    
    jeff_date = jeff_match['tourney_date']
    jeff_winner_surname = jeff_match['winner_surname']
    jeff_loser_surname = jeff_match['loser_surname']
    
    # Try exact date match
    for days_offset in [0, 1, -1]:  # Same day, +1 day, -1 day
        search_date = (jeff_date + timedelta(days=days_offset)).strftime('%Y-%m-%d')
        
        # Try normal order
        key = (search_date, jeff_winner_surname, jeff_loser_surname)
        if key in tennis_by_date_surnames:
            tennis_idx = tennis_by_date_surnames[key]
            exact_matches.append({
                'jeff_idx': idx,
                'tennis_idx': tennis_idx,
                'match_type': 'exact',
                'date_diff': days_offset
            })
            break
        
        # Try swapped order
        swapped_key = f"swapped_{(search_date, jeff_winner_surname, jeff_loser_surname)}"
        if swapped_key in tennis_by_date_surnames:
            tennis_idx = tennis_by_date_surnames[swapped_key]
            exact_matches.append({
                'jeff_idx': idx,
                'tennis_idx': tennis_idx,
                'match_type': 'swapped',
                'date_diff': days_offset
            })
            break

print(f"\n6. Match results:")
print(f"   Exact matches found: {len(exact_matches):,}")

if len(exact_matches) > 0:
    # Analyze match quality
    exact_count = sum(1 for m in exact_matches if m['match_type'] == 'exact')
    swapped_count = sum(1 for m in exact_matches if m['match_type'] == 'swapped')
    print(f"   - Same winner: {exact_count:,}")
    print(f"   - Swapped winner/loser: {swapped_count:,}")
    
    # Create matched datasets
    print(f"\n7. Creating matched datasets...")
    jeff_matched_indices = [m['jeff_idx'] for m in exact_matches]
    tennis_matched_indices = [m['tennis_idx'] for m in exact_matches]
    
    jeff_matched = jeff_test.loc[jeff_matched_indices].copy()
    tennis_matched = tennis_test.loc[tennis_matched_indices].copy()
    
    # Show sample matches
    print(f"\n   Sample matches:")
    for i in range(min(5, len(exact_matches))):
        match_info = exact_matches[i]
        jeff_idx = match_info['jeff_idx']
        tennis_idx = match_info['tennis_idx']
        
        jeff_row = jeff_test.loc[jeff_idx]
        tennis_row = tennis_test.loc[tennis_idx]
        
        print(f"     {jeff_row['tourney_date'].strftime('%Y-%m-%d')}: {jeff_row['winner_name']} def. {jeff_row['loser_name']}")
        print(f"       vs {tennis_row['Date'].strftime('%Y-%m-%d')}: {tennis_row['Winner']} def. {tennis_row['Loser']} ({match_info['match_type']})")
        print()
    
    # Calculate accuracies on matched dataset
    print(f"\n8. Calculating accuracies on matched dataset...")
    
    # Jeff Sackmann ATP baseline
    jeff_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
    jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
    jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
    jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])
    
    if len(jeff_clean) > 0:
        jeff_atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
        jeff_atp_accuracy = jeff_atp_correct / len(jeff_clean)
        print(f"   Jeff Sackmann ATP baseline: {jeff_atp_accuracy:.4f} ({jeff_atp_accuracy*100:.2f}%) on {len(jeff_clean):,} matches")
    
    # Tennis-Data baseline and betting odds
    tennis_clean = tennis_matched.dropna(subset=['WRank', 'LRank']).copy()
    tennis_clean['WRank'] = pd.to_numeric(tennis_clean['WRank'], errors='coerce')
    tennis_clean['LRank'] = pd.to_numeric(tennis_clean['LRank'], errors='coerce')
    tennis_clean = tennis_clean.dropna(subset=['WRank', 'LRank'])
    
    if len(tennis_clean) > 0:
        tennis_atp_correct = (tennis_clean['WRank'] < tennis_clean['LRank']).sum()
        tennis_atp_accuracy = tennis_atp_correct / len(tennis_clean)
        print(f"   Tennis-Data ATP baseline: {tennis_atp_accuracy:.4f} ({tennis_atp_accuracy*100:.2f}%) on {len(tennis_clean):,} matches")
        
        # Betting odds accuracy
        betting_acc, betting_count = calculate_betting_accuracy(tennis_clean, 'AvgW', 'AvgL')
        if betting_acc is not None:
            print(f"   Tennis-Data betting odds: {betting_acc:.4f} ({betting_acc*100:.2f}%) on {betting_count:,} matches")
    
    # Save matched datasets
    print(f"\n9. Saving matched datasets...")
    jeff_matched.to_csv('data/JeffSackmann/jeffsackmann_exact_matched_2023_2024.csv', index=False)
    tennis_matched.to_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_2023_2024.csv', index=False)
    
    # Check if ML-ready data exists and filter it
    ml_ready_path = 'data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv'
    try:
        ml_ready = pd.read_csv(ml_ready_path, low_memory=False)
        ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
        
        # Create a mapping from original indices to ML-ready indices
        jeff_original_to_ml = {}
        for ml_idx, row in ml_ready.iterrows():
            # Find corresponding match in original data by date, winner, loser
            matching_original = jeff_data[
                (jeff_data['tourney_date'] == row['tourney_date']) &
                (jeff_data['winner_name'] == row.get('Player1_Name', '')) &
                (jeff_data['loser_name'] == row.get('Player2_Name', ''))
            ]
            if len(matching_original) == 1:
                original_idx = matching_original.index[0]
                jeff_original_to_ml[original_idx] = ml_idx
        
        # Get ML-ready matches
        ml_matched_indices = [jeff_original_to_ml[idx] for idx in jeff_matched_indices if idx in jeff_original_to_ml]
        
        if len(ml_matched_indices) > 0:
            ml_matched = ml_ready.loc[ml_matched_indices].copy()
            ml_matched.to_csv('data/JeffSackmann/jeffsackmann_ml_ready_exact_matched_2023_2024.csv', index=False)
            print(f"   Saved ML-ready matched dataset: {len(ml_matched):,} matches")
        else:
            print(f"   Could not create ML-ready matched dataset (index mapping failed)")
            
    except FileNotFoundError:
        print(f"   ML-ready dataset not found at {ml_ready_path}")
    
    print(f"\n" + "="*80)
    print(f"FINAL COMPARISON READY")
    print(f"="*80)
    print(f"Matched dataset size: {len(exact_matches):,} matches")
    print(f"")
    print(f"Accuracy comparison on SAME MATCHES:")
    if len(jeff_clean) > 0 and len(tennis_clean) > 0:
        print(f"  ATP Ranking Baseline: {jeff_atp_accuracy*100:.2f}%")
        if betting_acc is not None:
            print(f"  Professional Betting Odds: {betting_acc*100:.2f}%")
        print(f"  Your XGBoost Model: [Run on exact matched dataset]")
        print(f"")
        print(f"Files created:")
        print(f"  - jeffsackmann_exact_matched_2023_2024.csv")
        print(f"  - tennis_data_exact_matched_2023_2024.csv")
        if len(ml_matched_indices) > 0:
            print(f"  - jeffsackmann_ml_ready_exact_matched_2023_2024.csv")

else:
    print("   Still no matches found. Let's debug further...")
    
    # Show sample surnames for debugging
    print(f"\n   Jeff Sackmann sample surnames:")
    sample_jeff = jeff_test[['winner_name', 'winner_surname', 'loser_name', 'loser_surname']].head(10)
    for _, row in sample_jeff.iterrows():
        print(f"     {row['winner_name']} -> {row['winner_surname']} vs {row['loser_name']} -> {row['loser_surname']}")
    
    print(f"\n   Tennis-Data sample surnames:")
    sample_tennis = tennis_test[['Winner', 'winner_surname', 'Loser', 'loser_surname']].head(10)
    for _, row in sample_tennis.iterrows():
        print(f"     {row['Winner']} -> {row['winner_surname']} vs {row['Loser']} -> {row['loser_surname']}")
    
    # Check for any potential surname matches
    jeff_surnames = set(jeff_test['winner_surname'].tolist() + jeff_test['loser_surname'].tolist())
    tennis_surnames = set(tennis_test['winner_surname'].tolist() + tennis_test['loser_surname'].tolist())
    
    common_surnames = jeff_surnames & tennis_surnames
    print(f"\n   Common surnames found: {len(common_surnames)} out of {len(jeff_surnames)} Jeff surnames and {len(tennis_surnames)} Tennis surnames")
    if len(common_surnames) > 0:
        print(f"   Sample common surnames: {list(common_surnames)[:10]}")