#!/usr/bin/env python3
"""
FIXED: Match exact same matches between Jeff Sackmann and Tennis-Data.co.uk datasets
FIX: Properly handle odds when winner/loser are swapped between datasets
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
print("FIXED EXACT MATCH COMPARISON - PROPER ODDS HANDLING")
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
tennis_test['score_str'] = tennis_test.apply(create_tennis_data_score, axis=1)
tennis_test['score_sets'] = tennis_test['score_str'].apply(parse_score)

# Match players and dates
print("\n5. Matching players and dates...")
matches = []

for i, (idx, jeff_match) in enumerate(jeff_test.iterrows()):
    if i % 5000 == 0:
        print(f"   Processing Jeff Sackmann match {i:,}/{len(jeff_test):,} ({i/len(jeff_test)*100:.1f}%)")
    
    jeff_date = jeff_match['tourney_date']
    jeff_winner = jeff_match['winner_name_norm']
    jeff_loser = jeff_match['loser_name_norm']
    
    # Look for matches within ±1 day with same players
    date_mask = (tennis_test['Date'] >= jeff_date - timedelta(days=1)) & \
                (tennis_test['Date'] <= jeff_date + timedelta(days=1))
    
    # Check for exact same winner/loser (no swapping)
    exact_match_mask = ((tennis_test['Winner_norm'] == jeff_winner) & (tennis_test['Loser_norm'] == jeff_loser))
    exact_matches = tennis_test[date_mask & exact_match_mask]
    
    # Check for swapped winner/loser  
    swapped_match_mask = ((tennis_test['Winner_norm'] == jeff_loser) & (tennis_test['Loser_norm'] == jeff_winner))
    swapped_matches = tennis_test[date_mask & swapped_match_mask]
    
    tennis_match = None
    same_winner = True
    
    if len(exact_matches) == 1:
        # Perfect match found - same winner/loser
        tennis_match = exact_matches.iloc[0]
        same_winner = True
    elif len(exact_matches) > 1:
        # Multiple exact matches - prefer exact date
        for _, candidate in exact_matches.iterrows():
            if candidate['Date'] == jeff_date:
                tennis_match = candidate
                same_winner = True
                break
        if tennis_match is None:
            tennis_match = exact_matches.iloc[0]  # Take first if no exact date
            same_winner = True
    elif len(swapped_matches) == 1:
        # Found match but winner/loser are swapped
        tennis_match = swapped_matches.iloc[0]
        same_winner = False
    elif len(swapped_matches) > 1:
        # Multiple swapped matches - prefer exact date
        for _, candidate in swapped_matches.iterrows():
            if candidate['Date'] == jeff_date:
                tennis_match = candidate
                same_winner = False
                break
        if tennis_match is None:
            tennis_match = swapped_matches.iloc[0]  # Take first if no exact date
            same_winner = False
    
    if tennis_match is not None:
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
            'same_winner': same_winner,  # TRUE = same winner, FALSE = swapped
            'winner_rank': jeff_match.get('winner_rank'),
            'loser_rank': jeff_match.get('loser_rank'),
            'tennis_wrank': tennis_match.get('WRank'),
            'tennis_lrank': tennis_match.get('LRank'),
            'tennis_avgw': tennis_match.get('AvgW'),
            'tennis_avgl': tennis_match.get('AvgL')
        })

print(f"\n6. Match results:")
print(f"   Total matches found: {len(matches):,}")

if len(matches) > 0:
    matches_df = pd.DataFrame(matches)
    same_winner_count = matches_df['same_winner'].sum()
    swapped_count = len(matches_df) - same_winner_count
    print(f"   Matches with same winner: {same_winner_count:,} ({same_winner_count/len(matches)*100:.1f}%)")
    print(f"   Matches with swapped winner/loser: {swapped_count:,} ({swapped_count/len(matches)*100:.1f}%)")
    
    # Show sample matches
    print(f"\n   Sample matched pairs:")
    for i in range(min(5, len(matches))):
        match = matches[i]
        swap_note = "" if match['same_winner'] else " [SWAPPED]"
        print(f"     {match['jeff_date'].strftime('%Y-%m-%d')}: {match['jeff_winner']} def. {match['jeff_loser']}")
        print(f"       vs {match['tennis_date'].strftime('%Y-%m-%d')}: {match['tennis_winner']} def. {match['tennis_loser']}{swap_note}")
        print(f"       Odds: W={match['tennis_avgw']}, L={match['tennis_avgl']}")
        print()
    
    # Create corrected datasets
    print(f"\n7. Creating corrected matched datasets...")
    
    # Get Jeff Sackmann matches
    jeff_matched_indices = [m['jeff_idx'] for m in matches]
    jeff_matched = jeff_test.loc[jeff_matched_indices].copy()
    
    # Create corrected Tennis-Data matches with proper odds mapping
    tennis_corrected_rows = []
    
    for match in matches:
        tennis_row = tennis_test.loc[match['tennis_idx']].copy()
        
        if match['same_winner']:
            # Same winner - keep odds as is
            # AvgW = winner odds, AvgL = loser odds
            pass  # No changes needed
        else:
            # Swapped winner/loser - swap the odds!
            # Jeff's winner was Tennis's loser, so Jeff's winner should get loser odds
            original_avgw = tennis_row['AvgW']
            original_avgl = tennis_row['AvgL']
            tennis_row['AvgW'] = original_avgl  # New winner gets old loser odds
            tennis_row['AvgL'] = original_avgw  # New loser gets old winner odds
            
            # Also swap other odds columns if they exist
            for odds_pair in [('B365W', 'B365L'), ('MaxW', 'MaxL'), ('PSW', 'PSL')]:
                w_col, l_col = odds_pair
                if w_col in tennis_row and l_col in tennis_row:
                    if pd.notna(tennis_row[w_col]) and pd.notna(tennis_row[l_col]):
                        original_w = tennis_row[w_col]
                        original_l = tennis_row[l_col]
                        tennis_row[w_col] = original_l
                        tennis_row[l_col] = original_w
        
        tennis_corrected_rows.append(tennis_row)
    
    tennis_matched = pd.DataFrame(tennis_corrected_rows)
    
    print(f"   Jeff Sackmann matched dataset: {len(jeff_matched):,} matches")
    print(f"   Tennis-Data corrected matched dataset: {len(tennis_matched):,} matches")
    
    # Verify odds correction
    print(f"\n8. Verifying odds correction...")
    print(f"   Checking for identical AvgW/AvgL after correction...")
    
    tennis_odds = tennis_matched.dropna(subset=['AvgW', 'AvgL']).copy()
    tennis_odds['AvgW'] = pd.to_numeric(tennis_odds['AvgW'], errors='coerce')
    tennis_odds['AvgL'] = pd.to_numeric(tennis_odds['AvgL'], errors='coerce')
    tennis_odds_clean = tennis_odds.dropna(subset=['AvgW', 'AvgL'])
    
    identical_odds = tennis_odds_clean[tennis_odds_clean['AvgW'] == tennis_odds_clean['AvgL']]
    print(f"   Matches with identical odds: {len(identical_odds):,} ({len(identical_odds)/len(tennis_odds_clean)*100:.2f}%)")
    
    # Calculate baselines
    print(f"\n9. Calculate baselines on corrected data...")
    
    # ATP ranking baseline
    jeff_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
    jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
    jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
    jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])
    
    if len(jeff_clean) > 0:
        jeff_atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
        jeff_atp_accuracy = jeff_atp_correct / len(jeff_clean)
        print(f"   Jeff Sackmann ATP baseline: {jeff_atp_accuracy:.4f} ({jeff_atp_accuracy*100:.2f}%) on {len(jeff_clean):,} matches")
    
    # Betting odds accuracy
    betting_acc, betting_count = calculate_betting_accuracy(tennis_odds_clean, 'AvgW', 'AvgL')
    if betting_acc is not None:
        print(f"   Tennis-Data betting odds: {betting_acc:.4f} ({betting_acc*100:.2f}%) on {betting_count:,} matches")
    
    # Save corrected datasets
    print(f"\n10. Saving corrected matched datasets...")
    jeff_matched.to_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', index=False)
    tennis_matched.to_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', index=False)
    
    print(f"   ✅ Saved corrected datasets:")
    print(f"   - data/JeffSackmann/jeffsackmann_exact_matched_final.csv")
    print(f"   - data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv")
    
    print(f"\n" + "="*80)
    print(f"ODDS CORRECTION COMPLETE")
    print(f"="*80)
    print(f"Fixed {swapped_count:,} matches where winner/loser were swapped")
    print(f"Odds now properly aligned with match outcomes")
    print(f"Ready for accurate betting analysis!")

else:
    print("   No matches found!")