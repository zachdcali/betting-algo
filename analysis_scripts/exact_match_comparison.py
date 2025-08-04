#!/usr/bin/env python3
"""
Match exact same matches between Jeff Sackmann and Tennis-Data.co.uk datasets
for true apples-to-apples comparison
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re

def normalize_name(name):
    """Normalize player names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # Remove common suffixes and prefixes
    name = re.sub(r'\b(Jr|Sr|II|III|IV)\b', '', name)
    name = re.sub(r'\.', '', name)
    name = re.sub(r'-', ' ', name)
    name = re.sub(r'\s+', ' ', name)
    # Convert to lowercase
    return name.lower().strip()

def parse_score(score_str):
    """Parse score string to extract set scores"""
    if pd.isna(score_str):
        return None
    
    # Remove common patterns like "RET", "W/O", etc.
    score_str = str(score_str).strip()
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'DEF', 'WALKOVER']):
        return None
    
    # Extract set scores using regex
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if len(sets) >= 2:  # At least 2 sets
        return tuple(sets)
    return None

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
    
    # Favorite (lower decimal odds) should win
    betting_correct = (odds_clean[win_col] < odds_clean[lose_col]).sum()
    betting_accuracy = betting_correct / len(odds_clean)
    
    return betting_accuracy, len(odds_clean)

print("="*80)
print("EXACT MATCH COMPARISON: SAME MATCHES IN BOTH DATASETS")
print("="*80)

# Load datasets
print("\n1. Loading datasets...")
jeff_data = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

print(f"   Jeff Sackmann: {len(jeff_data):,} total matches")
print(f"   Tennis-Data: {len(tennis_data):,} total matches")

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

# Normalize player names
print("\n3. Normalizing player names...")
jeff_test['winner_name_norm'] = jeff_test['winner_name'].apply(normalize_name)
jeff_test['loser_name_norm'] = jeff_test['loser_name'].apply(normalize_name)

tennis_test['Winner_norm'] = tennis_test['Winner'].apply(normalize_name)
tennis_test['Loser_norm'] = tennis_test['Loser'].apply(normalize_name)

# Parse scores for additional verification
print("\n4. Parsing match scores...")
jeff_test['score_sets'] = jeff_test['score'].apply(parse_score)
# Tennis-Data has W1,L1,W2,L2,etc - let's create a score string
def create_tennis_data_score(row):
    sets = []
    for i in range(1, 6):  # Up to 5 sets
        w_col = f'W{i}'
        l_col = f'L{i}'
        if pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            try:
                w_score = int(float(row[w_col]))
                l_score = int(float(row[l_col]))
                sets.append(f"{w_score}-{l_score}")
            except (ValueError, TypeError):
                continue
    return ' '.join(sets) if sets else None

tennis_test['score_str'] = tennis_test.apply(create_tennis_data_score, axis=1)
tennis_test['score_sets'] = tennis_test['score_str'].apply(parse_score)

# Match players and dates
print("\n5. Matching players and dates...")
matches = []

for idx, jeff_match in jeff_test.iterrows():
    if idx % 5000 == 0:
        print(f"   Processing Jeff Sackmann match {idx:,}/{len(jeff_test):,} ({idx/len(jeff_test)*100:.1f}%)")
    
    jeff_date = jeff_match['tourney_date']
    jeff_winner = jeff_match['winner_name_norm']
    jeff_loser = jeff_match['loser_name_norm']
    
    # Look for matches within ±1 day with same players
    date_mask = (tennis_test['Date'] >= jeff_date - timedelta(days=1)) & \
                (tennis_test['Date'] <= jeff_date + timedelta(days=1))
    
    # Try both player orders (in case winner/loser are swapped)
    player_mask = ((tennis_test['Winner_norm'] == jeff_winner) & (tennis_test['Loser_norm'] == jeff_loser)) | \
                  ((tennis_test['Winner_norm'] == jeff_loser) & (tennis_test['Loser_norm'] == jeff_winner))
    
    potential_matches = tennis_test[date_mask & player_mask]
    
    if len(potential_matches) == 1:
        # Perfect match found
        tennis_match = potential_matches.iloc[0]
        matches.append({
            'jeff_idx': idx,
            'tennis_idx': tennis_match.name,
            'jeff_date': jeff_date,
            'tennis_date': tennis_match['Date'],
            'jeff_winner': jeff_match['winner_name'],
            'jeff_loser': jeff_match['loser_name'],
            'tennis_winner': tennis_match['Winner'],
            'tennis_loser': tennis_match['Loser'],
            'jeff_score': jeff_match['score'],
            'tennis_score': tennis_match['score_str'],
            'same_winner': jeff_winner == tennis_match['Winner_norm']
        })
    elif len(potential_matches) > 1:
        # Multiple matches - try to disambiguate by score or tournament
        best_match = None
        for _, tennis_match in potential_matches.iterrows():
            # Prefer exact date match
            if tennis_match['Date'] == jeff_date:
                best_match = tennis_match
                break
        
        if best_match is not None:
            matches.append({
                'jeff_idx': idx,
                'tennis_idx': best_match.name,
                'jeff_date': jeff_date,
                'tennis_date': best_match['Date'],
                'jeff_winner': jeff_match['winner_name'],
                'jeff_loser': jeff_match['loser_name'],
                'tennis_winner': best_match['Winner'],
                'tennis_loser': best_match['Loser'],
                'jeff_score': jeff_match['score'],
                'tennis_score': best_match['score_str'],
                'same_winner': jeff_winner == best_match['Winner_norm']
            })

print(f"\n6. Match results:")
print(f"   Total matches found: {len(matches):,}")

if len(matches) > 0:
    matches_df = pd.DataFrame(matches)
    same_winner_count = matches_df['same_winner'].sum()
    print(f"   Matches with same winner: {same_winner_count:,} ({same_winner_count/len(matches)*100:.1f}%)")
    
    # Show sample matches
    print(f"\n   Sample matched pairs:")
    for i in range(min(5, len(matches))):
        match = matches[i]
        print(f"     {match['jeff_date'].strftime('%Y-%m-%d')}: {match['jeff_winner']} def. {match['jeff_loser']}")
        print(f"       vs {match['tennis_date'].strftime('%Y-%m-%d')}: {match['tennis_winner']} def. {match['tennis_loser']}")
        print(f"       Score: Jeff='{match['jeff_score']}' vs Tennis='{match['tennis_score']}'")
        print()
    
    # Create datasets for model evaluation
    print(f"\n7. Creating matched datasets...")
    
    # Get Jeff Sackmann matches
    jeff_matched_indices = [m['jeff_idx'] for m in matches]
    jeff_matched = jeff_test.loc[jeff_matched_indices].copy()
    
    # Get Tennis-Data matches  
    tennis_matched_indices = [m['tennis_idx'] for m in matches]
    tennis_matched = tennis_test.loc[tennis_matched_indices].copy()
    
    print(f"   Jeff Sackmann matched dataset: {len(jeff_matched):,} matches")
    print(f"   Tennis-Data matched dataset: {len(tennis_matched):,} matches")
    
    # Calculate ATP ranking baseline on matched dataset
    print(f"\n8. ATP ranking baseline on matched dataset:")
    jeff_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
    jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
    jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
    jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])
    
    if len(jeff_clean) > 0:
        jeff_atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
        jeff_atp_accuracy = jeff_atp_correct / len(jeff_clean)
        print(f"   Jeff Sackmann ATP baseline: {jeff_atp_accuracy:.4f} ({jeff_atp_accuracy*100:.2f}%) on {len(jeff_clean):,} matches")
    
    # Calculate Tennis-Data ATP baseline and betting odds accuracy on matched dataset
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
    
    # Save matched datasets for model evaluation
    print(f"\n9. Saving matched datasets...")
    jeff_matched.to_csv('data/JeffSackmann/jeffsackmann_matched_2023_2024.csv', index=False)
    tennis_matched.to_csv('data/Tennis-Data.co.uk/tennis_data_matched_2023_2024.csv', index=False)
    
    # Save the ML-ready version for model testing
    print(f"\n10. Checking if we need to create ML-ready features...")
    ml_ready_path = 'data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv'
    if pd.io.common.file_exists(ml_ready_path):
        print(f"   Loading existing ML-ready dataset...")
        ml_ready = pd.read_csv(ml_ready_path, low_memory=False)
        ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
        
        # Filter to matched indices
        ml_matched = ml_ready[ml_ready.index.isin(jeff_matched_indices)].copy()
        
        if len(ml_matched) > 0:
            ml_matched.to_csv('data/JeffSackmann/jeffsackmann_ml_ready_matched_2023_2024.csv', index=False)
            print(f"   Saved ML-ready matched dataset: {len(ml_matched):,} matches")
            print(f"   File: data/JeffSackmann/jeffsackmann_ml_ready_matched_2023_2024.csv")
        else:
            print(f"   Warning: No ML-ready matches found. May need to regenerate ML features.")
    else:
        print(f"   ML-ready dataset not found. You'll need to run preprocessing first.")
    
    print(f"\n" + "="*80)
    print(f"NEXT STEPS")
    print(f"="*80)
    print(f"1. Run your XGBoost/Neural Net models on the matched dataset:")
    print(f"   - Use: data/JeffSackmann/jeffsackmann_ml_ready_matched_2023_2024.csv")
    print(f"   - This has {len(matches):,} matches that exist in both datasets")
    print(f"")
    print(f"2. Compare results:")
    print(f"   - Your model accuracy on these {len(matches):,} matches")
    print(f"   - vs Betting odds accuracy: {betting_acc*100:.2f}% on same matches")
    print(f"   - vs ATP baseline: ~{jeff_atp_accuracy*100:.1f}% on same matches")
    print(f"")
    print(f"This will be a true apples-to-apples comparison!")

else:
    print("   No matches found. This might indicate:")
    print("   - Different player name formats")
    print("   - Different date formats") 
    print("   - Limited overlap between datasets")
    print("   - Need to adjust matching criteria")