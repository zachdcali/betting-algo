#!/usr/bin/env python3
"""
Debug betting odds calculation to see why returns are unrealistic
"""
import pandas as pd
import numpy as np
import sys
import os
import pickle
import torch
import torch.nn as nn
import xgboost as xgb
import re

class TennisNet(nn.Module):
    def __init__(self, input_size):
        super(TennisNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1), nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def get_feature_columns(df):
    exclude_cols = ['tourney_date', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id',
                   'winner_name', 'loser_name', 'score', 'Player1_Name', 'Player2_Name', 'Player1_Wins',
                   'best_of', 'round', 'minutes', 'data_source', 'year']
    return [col for col in df.columns if col not in exclude_cols]

def decimal_to_implied_prob(decimal_odds):
    return 1.0 / decimal_odds

def kelly_fraction(model_prob, decimal_odds):
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    kelly = (b * p - q) / b
    return max(0, kelly)

def parse_score_to_sets(score_str):
    if pd.isna(score_str):
        return None
    score_str = str(score_str).strip()
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'DEF', 'WALKOVER']):
        return None
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if len(sets) >= 2:
        return tuple(sets)
    return None

def create_tennis_data_score_sets(row):
    sets = []
    for i in range(1, 6):
        w_col = f'W{i}'
        l_col = f'L{i}'
        if pd.notna(row.get(w_col)) and pd.notna(row.get(l_col)):
            try:
                w_score = int(float(row[w_col]))
                l_score = int(float(row[l_col]))
                sets.append((str(w_score), str(l_score)))
            except (ValueError, TypeError):
                continue
    return tuple(sets) if len(sets) >= 2 else None

print("="*80)
print("DEBUG BETTING ODDS CALCULATION")
print("="*80)

# Load data (same as before)
ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)

ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
tennis_matched['Date'] = pd.to_datetime(tennis_matched['Date'])

# Get 100% verified matches
jeff_matched['score_sets'] = jeff_matched['score'].apply(parse_score_to_sets)
tennis_matched['score_sets'] = tennis_matched.apply(create_tennis_data_score_sets, axis=1)

verified_indices = []
for i in range(len(jeff_matched)):
    jeff_row = jeff_matched.iloc[i]
    tennis_row = tennis_matched.iloc[i]
    
    score_match = False
    if jeff_row['score_sets'] is not None and tennis_row['score_sets'] is not None:
        score_match = jeff_row['score_sets'] == tennis_row['score_sets']
    
    ranking_match = False
    try:
        jeff_w_rank = float(jeff_row['winner_rank']) if pd.notna(jeff_row['winner_rank']) else None
        jeff_l_rank = float(jeff_row['loser_rank']) if pd.notna(jeff_row['loser_rank']) else None
        tennis_w_rank = float(tennis_row['WRank']) if pd.notna(tennis_row['WRank']) else None
        tennis_l_rank = float(tennis_row['LRank']) if pd.notna(tennis_row['LRank']) else None
        
        if all(x is not None for x in [jeff_w_rank, jeff_l_rank, tennis_w_rank, tennis_l_rank]):
            ranking_match = (jeff_w_rank == tennis_w_rank) and (jeff_l_rank == tennis_l_rank)
    except (ValueError, TypeError):
        pass
    
    if score_match and ranking_match:
        verified_indices.append(i)

jeff_100_verified = jeff_matched.iloc[verified_indices].copy()
tennis_100_verified = tennis_matched.iloc[verified_indices].copy()

# Create ML dataset
ml_verified = []
tennis_verified_mapping = {}

for idx, verified_row in jeff_100_verified.iterrows():
    date = verified_row['tourney_date']
    winner = str(verified_row['winner_name']).strip()
    loser = str(verified_row['loser_name']).strip()
    
    ml_candidates = ml_ready[ml_ready['tourney_date'] == date]
    
    for _, ml_row in ml_candidates.iterrows():
        player1 = str(ml_row['Player1_Name']).strip()
        player2 = str(ml_row['Player2_Name']).strip()
        
        if (player1 == winner and player2 == loser):
            ml_row_copy = ml_row.copy()
            ml_row_copy['Player1_Wins'] = 1
            ml_verified.append(ml_row_copy)
            tennis_idx = list(jeff_100_verified.index).index(idx)
            tennis_verified_mapping[len(ml_verified)-1] = tennis_idx
            break
        elif (player1 == loser and player2 == winner):
            ml_row_copy = ml_row.copy()
            ml_row_copy['Player1_Wins'] = 0
            ml_verified.append(ml_row_copy)
            tennis_idx = list(jeff_100_verified.index).index(idx)
            tennis_verified_mapping[len(ml_verified)-1] = tennis_idx
            break

ml_verified_df = pd.DataFrame(ml_verified)

# Load Neural Network model
feature_cols = get_feature_columns(ml_verified_df)
numeric_cols = [col for col in feature_cols if col in ml_verified_df.columns and 
                ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]

X_test = ml_verified_df[numeric_cols].fillna(ml_verified_df[numeric_cols].median())
y_test = ml_verified_df['Player1_Wins']

nn_path = 'results/professional_tennis/Neural_Network/neural_network_model.pth'
scaler_path = 'results/professional_tennis/Neural_Network/scaler.pkl'

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

nn_model = TennisNet(X_test.shape[1])
nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
nn_model.eval()

X_scaled = scaler.transform(X_test.values)
X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    nn_probs = nn_model(X_tensor).squeeze().numpy()

print(f"Analyzing first 10 matches for debugging...")
print("="*80)

for i in range(10):
    tennis_idx = tennis_verified_mapping.get(i)
    if tennis_idx is None:
        continue
        
    tennis_row = tennis_100_verified.iloc[tennis_idx]
    ml_row = ml_verified_df.iloc[i]
    actual_p1_win = y_test.iloc[i]
    model_p1_prob = nn_probs[i]
    
    # Get the actual match details
    player1 = ml_row['Player1_Name']
    player2 = ml_row['Player2_Name']
    
    # Get odds
    avg_winner_odds = tennis_row.get('AvgW')
    avg_loser_odds = tennis_row.get('AvgL')
    
    if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
        continue
        
    avg_winner_odds = float(avg_winner_odds)
    avg_loser_odds = float(avg_loser_odds)
    
    # THIS IS THE KEY LOGIC TO DEBUG
    print(f"\nMatch {i+1}: {player1} vs {player2}")
    print(f"  Actual outcome: Player1_Wins = {actual_p1_win}")
    print(f"  Model P1 probability: {model_p1_prob:.3f}")
    print(f"  AvgW (winner odds): {avg_winner_odds:.2f}")
    print(f"  AvgL (loser odds): {avg_loser_odds:.2f}")
    
    # The current logic assigns odds based on actual outcome
    if actual_p1_win == 1:
        p1_odds = avg_winner_odds  # Player1 was the winner
        p2_odds = avg_loser_odds   # Player2 was the loser
        print(f"  Player1 won → P1 odds: {p1_odds:.2f}, P2 odds: {p2_odds:.2f}")
    else:
        p1_odds = avg_loser_odds   # Player1 was the loser
        p2_odds = avg_winner_odds  # Player2 was the winner
        print(f"  Player2 won → P1 odds: {p1_odds:.2f}, P2 odds: {p2_odds:.2f}")
    
    # Calculate Kelly
    kelly_p1 = kelly_fraction(model_p1_prob, p1_odds)
    kelly_p2 = kelly_fraction(1 - model_p1_prob, p2_odds)
    
    print(f"  Kelly P1: {kelly_p1:.4f}, Kelly P2: {kelly_p2:.4f}")
    
    # What bet would we place?
    if kelly_p1 > 0.01:
        print(f"  → Would bet on Player1 at {p1_odds:.2f} odds")
        print(f"  → Model prob: {model_p1_prob:.3f} vs Market prob: {decimal_to_implied_prob(p1_odds):.3f}")
        print(f"  → Edge: {model_p1_prob - decimal_to_implied_prob(p1_odds):+.3f}")
    elif kelly_p2 > 0.01:
        print(f"  → Would bet on Player2 at {p2_odds:.2f} odds")
        print(f"  → Model prob: {1-model_p1_prob:.3f} vs Market prob: {decimal_to_implied_prob(p2_odds):.3f}")
        print(f"  → Edge: {(1-model_p1_prob) - decimal_to_implied_prob(p2_odds):+.3f}")
    else:
        print(f"  → No bet (no positive EV)")

print(f"\n" + "="*80)
print(f"KEY ISSUE TO CHECK:")
print(f"="*80)
print(f"1. Are we correctly mapping AvgW/AvgL to Player1/Player2?")
print(f"2. AvgW should be odds for the match WINNER") 
print(f"3. AvgL should be odds for the match LOSER")
print(f"4. If Player1 actually won, then Player1 odds = AvgW")
print(f"5. If Player2 actually won, then Player1 odds = AvgL")
print(f"6. Are the odds realistic (1.5-5.0 range mostly)?")
print(f"7. Are we getting huge Kelly values due to wrong odds assignment?")