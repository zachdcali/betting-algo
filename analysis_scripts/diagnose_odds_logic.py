#!/usr/bin/env python3
"""
Diagnose odds logic - trace exactly what's happening
"""
import pandas as pd
import numpy as np

print("="*80)
print("DIAGNOSING ODDS LOGIC")
print("="*80)

# Load both the matched file and original file
print("\n1. Loading files...")
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)
tennis_orig = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

print(f"   Matched file: {len(tennis_matched):,} matches")
print(f"   Original file: {len(tennis_orig):,} matches")

# Convert to numeric
tennis_matched['AvgW'] = pd.to_numeric(tennis_matched['AvgW'], errors='coerce')
tennis_matched['AvgL'] = pd.to_numeric(tennis_matched['AvgL'], errors='coerce')
tennis_orig['AvgW'] = pd.to_numeric(tennis_orig['AvgW'], errors='coerce')
tennis_orig['AvgL'] = pd.to_numeric(tennis_orig['AvgL'], errors='coerce')

# Check basic odds distribution
print("\n2. Odds distribution comparison:")
print("MATCHED FILE:")
matched_odds = tennis_matched.dropna(subset=['AvgW', 'AvgL'])
print(f"   Matches with odds: {len(matched_odds):,}")
print(f"   AvgW range: {matched_odds['AvgW'].min():.2f} to {matched_odds['AvgW'].max():.2f}")
print(f"   AvgL range: {matched_odds['AvgL'].min():.2f} to {matched_odds['AvgL'].max():.2f}")
print(f"   Winner favored (AvgW < AvgL): {(matched_odds['AvgW'] < matched_odds['AvgL']).sum():,} ({(matched_odds['AvgW'] < matched_odds['AvgL']).mean()*100:.1f}%)")

print("ORIGINAL FILE (2023-2024):")
tennis_orig['Date'] = pd.to_datetime(tennis_orig['Date'])
orig_2023_2024 = tennis_orig[tennis_orig['Date'].dt.year.isin([2023, 2024])]
orig_odds = orig_2023_2024.dropna(subset=['AvgW', 'AvgL'])
print(f"   Matches with odds: {len(orig_odds):,}")
print(f"   AvgW range: {orig_odds['AvgW'].min():.2f} to {orig_odds['AvgW'].max():.2f}")
print(f"   AvgL range: {orig_odds['AvgL'].min():.2f} to {orig_odds['AvgL'].max():.2f}")
print(f"   Winner favored (AvgW < AvgL): {(orig_odds['AvgW'] < orig_odds['AvgL']).sum():,} ({(orig_odds['AvgW'] < orig_odds['AvgL']).mean()*100:.1f}%)")

# Check accuracy of betting on favorite
print("\n3. Accuracy of betting on favorite:")
print("MATCHED FILE:")
matched_favorite_wins = (matched_odds['AvgW'] < matched_odds['AvgL']).sum()
matched_total = len(matched_odds)
matched_fav_accuracy = matched_favorite_wins / matched_total
print(f"   Favorite wins: {matched_favorite_wins:,} out of {matched_total:,} = {matched_fav_accuracy*100:.2f}%")

print("ORIGINAL FILE:")
orig_favorite_wins = (orig_odds['AvgW'] < orig_odds['AvgL']).sum()
orig_total = len(orig_odds)
orig_fav_accuracy = orig_favorite_wins / orig_total
print(f"   Favorite wins: {orig_favorite_wins:,} out of {orig_total:,} = {orig_fav_accuracy*100:.2f}%")

# Check if odds are actually from original file
print("\n4. Verifying odds source...")
matches_found_in_orig = 0
matches_odds_match = 0

for i in range(min(100, len(tennis_matched))):  # Check first 100 matches
    matched_row = tennis_matched.iloc[i]
    winner = str(matched_row['Winner']).strip()
    loser = str(matched_row['Loser']).strip()
    match_date = pd.to_datetime(matched_row['Date'])
    matched_avgw = matched_row['AvgW']
    matched_avgl = matched_row['AvgL']
    
    # Find in original
    orig_candidates = orig_2023_2024[
        (orig_2023_2024['Winner'].str.strip() == winner) &
        (orig_2023_2024['Loser'].str.strip() == loser) &
        (orig_2023_2024['Date'] == match_date)
    ]
    
    if len(orig_candidates) == 1:
        matches_found_in_orig += 1
        orig_row = orig_candidates.iloc[0]
        orig_avgw = orig_row['AvgW']
        orig_avgl = orig_row['AvgL']
        
        if (pd.notna(matched_avgw) and pd.notna(matched_avgl) and 
            pd.notna(orig_avgw) and pd.notna(orig_avgl)):
            if abs(matched_avgw - orig_avgw) < 0.01 and abs(matched_avgl - orig_avgl) < 0.01:
                matches_odds_match += 1

print(f"   First 100 matches checked:")
print(f"   Found in original: {matches_found_in_orig}/100")
print(f"   Odds match exactly: {matches_odds_match}/100")

# Show examples of model logic
print("\n5. Model betting logic examples:")
print("   (Simulating how Kelly calculation works)")

# Load ML data to see actual model logic
ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)

# Get first few ML matches
ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])

# Find first few ML matches that correspond to our tennis data
print("   Example: How odds are assigned to players")
print("   Tennis Data | ML Data | Odds Assignment")
print("   Winner/Loser AvgW/AvgL | P1/P2 P1_Wins | P1_Odds P2_Odds")
print("-" * 70)

for i in range(5):
    if i < len(tennis_matched) and i < len(jeff_matched):
        tennis_row = tennis_matched.iloc[i]
        jeff_row = jeff_matched.iloc[i]
        
        tennis_winner = str(tennis_row['Winner']).strip()
        tennis_loser = str(tennis_row['Loser']).strip()
        avgw = tennis_row['AvgW']
        avgl = tennis_row['AvgL']
        
        jeff_winner = str(jeff_row['winner_name']).strip()
        jeff_loser = str(jeff_row['loser_name']).strip()
        
        # Find corresponding ML row
        date = jeff_row['tourney_date']
        ml_candidates = ml_ready[ml_ready['tourney_date'] == date]
        
        for _, ml_row in ml_candidates.iterrows():
            player1 = str(ml_row['Player1_Name']).strip()
            player2 = str(ml_row['Player2_Name']).strip()
            
            if (player1 == jeff_winner and player2 == jeff_loser):
                p1_wins = 1
                # P1 won, so P1 gets winner odds, P2 gets loser odds
                p1_odds = avgw
                p2_odds = avgl
                print(f"   {tennis_winner[:8]}/{tennis_loser[:8]} {avgw:.2f}/{avgl:.2f} | {player1[:8]}/{player2[:8]} {p1_wins} | {p1_odds:.2f} {p2_odds:.2f}")
                break
            elif (player1 == jeff_loser and player2 == jeff_winner):
                p1_wins = 0
                # P1 lost, so P1 gets loser odds, P2 gets winner odds  
                p1_odds = avgl
                p2_odds = avgw
                print(f"   {tennis_winner[:8]}/{tennis_loser[:8]} {avgw:.2f}/{avgl:.2f} | {player1[:8]}/{player2[:8]} {p1_wins} | {p1_odds:.2f} {p2_odds:.2f}")
                break

print(f"\n" + "="*80)
print(f"DIAGNOSIS COMPLETE")
print(f"="*80)
print(f"Key questions:")
print(f"1. Are odds coming from original file? {matches_odds_match}/100 match exactly")
print(f"2. Is favorite accuracy realistic? Matched: {matched_fav_accuracy*100:.1f}%, Original: {orig_fav_accuracy*100:.1f}%") 
print(f"3. The model assigns odds based on actual outcome (hindsight)")
print(f"4. This might create artificial edges if model predictions differ from market")