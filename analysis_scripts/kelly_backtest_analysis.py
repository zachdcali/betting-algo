#!/usr/bin/env python3
"""
Kelly Criterion Betting Backtest: Test model profitability against betting odds
"""
import pandas as pd
import numpy as np
import sys
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
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
    """Convert decimal odds to implied probability"""
    return 1.0 / decimal_odds

def kelly_fraction(model_prob, decimal_odds):
    """Calculate Kelly Criterion betting fraction"""
    # Kelly = (bp - q) / b where:
    # b = decimal odds - 1 (net odds)
    # p = model probability
    # q = 1 - p
    b = decimal_odds - 1
    p = model_prob
    q = 1 - p
    
    kelly = (b * p - q) / b
    return max(0, kelly)  # Don't bet if negative EV

def parse_score_to_sets(score_str):
    """Parse score string to standardized set format"""
    if pd.isna(score_str):
        return None
    
    score_str = str(score_str).strip()
    if any(x in score_str.upper() for x in ['RET', 'W/O', 'DEF', 'WALKOVER']):
        return None
    
    # Extract set scores using regex
    sets = re.findall(r'(\d+)-(\d+)', score_str)
    if len(sets) >= 2:  # At least 2 sets
        return tuple(sets)
    return None

def create_tennis_data_score_sets(row):
    """Create standardized score format from Tennis-Data columns"""
    sets = []
    for i in range(1, 6):  # Up to 5 sets
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
print("KELLY CRITERION BETTING BACKTEST - 100% VERIFIED DATA")
print("="*80)

# Load and prepare 100% verified data
print("\n1. Loading 100% verified data...")
ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)

ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
tennis_matched['Date'] = pd.to_datetime(tennis_matched['Date'])

# Get 100% verified matches (same logic as before)
print("\n2. Filtering to 100% verified matches...")
jeff_matched['score_sets'] = jeff_matched['score'].apply(parse_score_to_sets)
tennis_matched['score_sets'] = tennis_matched.apply(create_tennis_data_score_sets, axis=1)

verified_indices = []
for i in range(len(jeff_matched)):
    jeff_row = jeff_matched.iloc[i]
    tennis_row = tennis_matched.iloc[i]
    
    # Score verification
    score_match = False
    if jeff_row['score_sets'] is not None and tennis_row['score_sets'] is not None:
        score_match = jeff_row['score_sets'] == tennis_row['score_sets']
    
    # Ranking verification
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

print(f"   100% verified matches: {len(verified_indices):,}")

# Create ML dataset for verified matches
print("\n3. Creating ML dataset for verified matches...")
ml_verified = []
tennis_verified_mapping = {}  # Map ML index to tennis row

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
            # Map this ML match to corresponding tennis odds
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
ml_verified_df['year'] = ml_verified_df['tourney_date'].dt.year

print(f"   ML verified dataset: {len(ml_verified_df):,} matches")

# Prepare features
feature_cols = get_feature_columns(ml_verified_df)
numeric_cols = [col for col in feature_cols if col in ml_verified_df.columns and 
                ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]

X_test = ml_verified_df[numeric_cols].fillna(ml_verified_df[numeric_cols].median())
y_test = ml_verified_df['Player1_Wins']

# Load models and get probabilities
print("\n4. Loading models and generating probabilities...")
model_probs = {}

# XGBoost
xgb_path = 'results/professional_tennis/XGBoost/xgboost_model.json'
if os.path.exists(xgb_path):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]  # Probability of Player1 winning
    model_probs['XGBoost'] = xgb_probs
    print(f"   XGBoost probabilities loaded")

# Random Forest
rf_path = 'results/professional_tennis/Random_Forest/random_forest_model.pkl'
if os.path.exists(rf_path):
    with open(rf_path, 'rb') as f:
        rf_model = pickle.load(f)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    model_probs['Random_Forest'] = rf_probs
    print(f"   Random Forest probabilities loaded")

# Neural Network
nn_path = 'results/professional_tennis/Neural_Network/neural_network_model.pth'
scaler_path = 'results/professional_tennis/Neural_Network/scaler.pkl'
if os.path.exists(nn_path) and os.path.exists(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    nn_model = TennisNet(X_test.shape[1])
    nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
    nn_model.eval()
    
    X_scaled = scaler.transform(X_test.values)
    X_tensor = torch.FloatTensor(X_scaled)
    
    with torch.no_grad():
        nn_probs = nn_model(X_tensor).squeeze().numpy()
    
    model_probs['Neural_Network'] = nn_probs
    print(f"   Neural Network probabilities loaded")

# Run Kelly Criterion backtest for each model
print(f"\n5. Running Kelly Criterion backtest...")
STARTING_BANKROLL = 100.0
MIN_BET_SIZE = 1.0  # Minimum bet size
MAX_KELLY_FRACTION = 0.25  # Cap Kelly at 25% of bankroll

for model_name, model_probabilities in model_probs.items():
    print(f"\n=== {model_name} BACKTEST ===")
    
    bankroll = STARTING_BANKROLL
    total_bets = 0
    winning_bets = 0
    total_bet_amount = 0
    total_profit = 0
    
    bet_history = []
    
    for i in range(len(ml_verified_df)):
        # Get model probability for Player1 winning
        model_p1_prob = model_probabilities[i]
        
        # Get actual outcome
        actual_p1_win = y_test.iloc[i]
        
        # Get betting odds from tennis data
        tennis_idx = tennis_verified_mapping.get(i)
        if tennis_idx is None:
            continue
            
        tennis_row = tennis_100_verified.iloc[tennis_idx]
        
        # Get decimal odds for winner and loser
        avg_winner_odds = tennis_row.get('AvgW')
        avg_loser_odds = tennis_row.get('AvgL')
        
        if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
            continue
            
        avg_winner_odds = float(avg_winner_odds)
        avg_loser_odds = float(avg_loser_odds)
        
        # Determine which player to potentially bet on
        # The tennis data has AvgW (winner odds) and AvgL (loser odds)
        # We need to figure out if Player1 is the winner or loser
        
        # If Player1 won the match, then Player1 odds = AvgW, Player2 odds = AvgL
        # If Player1 lost the match, then Player1 odds = AvgL, Player2 odds = AvgW
        if actual_p1_win == 1:
            p1_odds = avg_winner_odds
            p2_odds = avg_loser_odds
        else:
            p1_odds = avg_loser_odds  
            p2_odds = avg_winner_odds
        
        # Calculate implied probabilities from odds
        market_p1_prob = decimal_to_implied_prob(p1_odds)
        market_p2_prob = decimal_to_implied_prob(p2_odds)
        
        # Check for betting opportunities on Player1
        kelly_p1 = kelly_fraction(model_p1_prob, p1_odds)
        kelly_p2 = kelly_fraction(1 - model_p1_prob, p2_odds)
        
        # Place bet if Kelly > 0 and sufficient edge
        bet_placed = False
        bet_on_p1 = False
        bet_amount = 0
        
        if kelly_p1 > 0.01:  # At least 1% Kelly
            kelly_capped = min(kelly_p1, MAX_KELLY_FRACTION)
            bet_amount = bankroll * kelly_capped
            bet_amount = max(bet_amount, MIN_BET_SIZE) if bankroll > MIN_BET_SIZE else 0
            bet_amount = min(bet_amount, bankroll)  # Never bet more than current bankroll
            bet_on_p1 = True
            bet_placed = True
        elif kelly_p2 > 0.01:  # At least 1% Kelly
            kelly_capped = min(kelly_p2, MAX_KELLY_FRACTION)
            bet_amount = bankroll * kelly_capped
            bet_amount = max(bet_amount, MIN_BET_SIZE) if bankroll > MIN_BET_SIZE else 0
            bet_amount = min(bet_amount, bankroll)  # Never bet more than current bankroll
            bet_on_p1 = False
            bet_placed = True
        
        # Stop betting if bankroll is too low
        if bankroll < MIN_BET_SIZE:
            continue
            
        if bet_placed and bet_amount > 0 and bankroll >= bet_amount and bet_amount >= MIN_BET_SIZE:
            # Ensure bet doesn't exceed current bankroll
            bet_amount = min(bet_amount, bankroll)
            
            total_bets += 1
            total_bet_amount += bet_amount
            
            # Determine if bet won
            if bet_on_p1:
                bet_won = (actual_p1_win == 1)
                odds_used = p1_odds
                model_prob_used = model_p1_prob
                market_prob_used = market_p1_prob
            else:
                bet_won = (actual_p1_win == 0)
                odds_used = p2_odds
                model_prob_used = 1 - model_p1_prob
                market_prob_used = market_p2_prob
            
            if bet_won:
                profit = bet_amount * (odds_used - 1)
                winning_bets += 1
            else:
                profit = -bet_amount
            
            bankroll += profit
            total_profit += profit
            
            bet_history.append({
                'match_idx': i,
                'player1': ml_verified_df.iloc[i]['Player1_Name'],
                'player2': ml_verified_df.iloc[i]['Player2_Name'],
                'bet_on_p1': bet_on_p1,
                'bet_amount': bet_amount,
                'odds': odds_used,
                'model_prob': model_prob_used,
                'market_prob': market_prob_used,
                'edge': model_prob_used - market_prob_used,
                'bet_won': bet_won,
                'profit': profit,
                'bankroll_after': bankroll
            })
    
    # Calculate statistics
    win_rate = winning_bets / total_bets if total_bets > 0 else 0
    roi = (total_profit / total_bet_amount) * 100 if total_bet_amount > 0 else 0
    final_bankroll = bankroll
    total_return = ((final_bankroll - STARTING_BANKROLL) / STARTING_BANKROLL) * 100
    
    print(f"   Starting bankroll: ${STARTING_BANKROLL:.2f}")
    print(f"   Final bankroll: ${final_bankroll:.2f}")
    print(f"   Total return: {total_return:+.2f}%")
    print(f"   Total bets placed: {total_bets:,}")
    print(f"   Bets won: {winning_bets:,}")
    print(f"   Betting accuracy: {win_rate:.2%}")
    print(f"   Total bet amount: ${total_bet_amount:.2f}")
    print(f"   Total profit: ${total_profit:+.2f}")
    print(f"   ROI on bets: {roi:+.2f}%")
    
    if len(bet_history) > 0:
        bet_df = pd.DataFrame(bet_history)
        avg_edge = bet_df['edge'].mean()
        avg_bet_size = bet_df['bet_amount'].mean()
        
        print(f"   Average edge: {avg_edge:+.3f}")
        print(f"   Average bet size: ${avg_bet_size:.2f}")
        
        # Show top 5 most profitable bets
        top_profits = bet_df.nlargest(5, 'profit')
        print(f"\n   Top 5 most profitable bets:")
        for _, bet in top_profits.iterrows():
            player_bet_on = bet['player1'] if bet['bet_on_p1'] else bet['player2']
            print(f"     ${bet['bet_amount']:.0f} on {player_bet_on} @ {bet['odds']:.2f} → ${bet['profit']:+.0f}")

print(f"\n" + "="*80)
print(f"KELLY CRITERION BACKTEST COMPLETE")
print(f"="*80)
print(f"Starting bankroll: ${STARTING_BANKROLL:.2f}")
print(f"All models tested on {len(ml_verified_df):,} matches with 100% verified data")
print(f"✅ Results show actual profitability of betting against market odds")