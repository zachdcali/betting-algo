#!/usr/bin/env python3
"""
Detailed Kelly backtest with CSV output of every bet
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
import matplotlib.pyplot as plt

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
print("DETAILED KELLY BACKTEST WITH CSV TRACKING")
print("="*80)

# Load and prepare data (same as before)
print("\n1. Loading 100% verified data...")
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

# Create ML dataset with mapping
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
print(f"   100% verified matches: {len(ml_verified_df):,}")

# Prepare features and load Neural Network
feature_cols = get_feature_columns(ml_verified_df)
numeric_cols = [col for col in feature_cols if col in ml_verified_df.columns and 
                ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]

X_test = ml_verified_df[numeric_cols].fillna(ml_verified_df[numeric_cols].median())
y_test = ml_verified_df['Player1_Wins']

print("\n2. Loading Neural Network model...")
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

# Run detailed Kelly backtest
print("\n3. Running detailed Kelly backtest with 2% threshold...")

STARTING_BANKROLL = 100.0
KELLY_THRESHOLD = 0.02  # 2%
MAX_KELLY = 0.05  # Cap at 5%
MIN_BET = 1.0

bankroll = STARTING_BANKROLL
all_bets = []
bet_number = 0

for i in range(len(ml_verified_df)):
    tennis_idx = tennis_verified_mapping.get(i)
    if tennis_idx is None:
        continue
        
    tennis_row = tennis_100_verified.iloc[tennis_idx]
    ml_row = ml_verified_df.iloc[i]
    actual_p1_win = y_test.iloc[i]
    model_p1_prob = nn_probs[i]
    
    player1 = ml_row['Player1_Name']
    player2 = ml_row['Player2_Name']
    match_date = ml_row['tourney_date']
    
    avg_winner_odds = tennis_row.get('AvgW')
    avg_loser_odds = tennis_row.get('AvgL')
    
    if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
        continue
        
    avg_winner_odds = float(avg_winner_odds)
    avg_loser_odds = float(avg_loser_odds)
    
    # Assign odds based on actual outcome
    if actual_p1_win == 1:
        p1_odds = avg_winner_odds  # Player1 won
        p2_odds = avg_loser_odds   # Player2 lost
    else:
        p1_odds = avg_loser_odds   # Player1 lost
        p2_odds = avg_winner_odds  # Player2 won
    
    # Calculate Kelly fractions
    kelly_p1 = kelly_fraction(model_p1_prob, p1_odds)
    kelly_p2 = kelly_fraction(1 - model_p1_prob, p2_odds)
    
    # Determine best bet
    bet_placed = False
    
    if kelly_p1 > KELLY_THRESHOLD:
        # Bet on Player1
        kelly_capped = min(kelly_p1, MAX_KELLY)
        bet_amount = bankroll * kelly_capped  # KEY: Use CURRENT bankroll
        bet_amount = max(bet_amount, MIN_BET)
        bet_amount = min(bet_amount, bankroll)  # Can't bet more than we have
        
        bet_on_player = player1
        bet_on_p1 = True
        odds_used = p1_odds
        model_prob = model_p1_prob
        market_prob = decimal_to_implied_prob(p1_odds)
        edge = model_prob - market_prob
        bet_won = (actual_p1_win == 1)
        bet_placed = True
        
    elif kelly_p2 > KELLY_THRESHOLD:
        # Bet on Player2
        kelly_capped = min(kelly_p2, MAX_KELLY)
        bet_amount = bankroll * kelly_capped  # KEY: Use CURRENT bankroll
        bet_amount = max(bet_amount, MIN_BET)
        bet_amount = min(bet_amount, bankroll)  # Can't bet more than we have
        
        bet_on_player = player2
        bet_on_p1 = False
        odds_used = p2_odds
        model_prob = 1 - model_p1_prob
        market_prob = decimal_to_implied_prob(p2_odds)
        edge = model_prob - market_prob
        bet_won = (actual_p1_win == 0)
        bet_placed = True
    
    if bet_placed and bet_amount > 0 and bankroll >= MIN_BET:
        # Calculate profit/loss
        if bet_won:
            profit = bet_amount * (odds_used - 1)
        else:
            profit = -bet_amount
        
        old_bankroll = bankroll
        bankroll += profit  # Update bankroll
        bet_number += 1
        
        # Record this bet
        all_bets.append({
            'bet_number': bet_number,
            'match_index': i,
            'date': match_date,
            'player1': player1,
            'player2': player2,
            'bet_on_player': bet_on_player,
            'bet_on_p1': bet_on_p1,
            'actual_p1_win': actual_p1_win,
            'model_p1_prob': model_p1_prob,
            'model_prob_bet_on': model_prob,
            'market_prob_bet_on': market_prob,
            'edge': edge,
            'kelly_fraction': kelly_capped,
            'bankroll_before': old_bankroll,
            'bet_amount': bet_amount,
            'bet_as_pct_bankroll': bet_amount / old_bankroll,
            'odds_used': odds_used,
            'bet_won': bet_won,
            'profit': profit,
            'bankroll_after': bankroll,
            'bankroll_growth': bankroll / old_bankroll,
            'total_return_pct': (bankroll - STARTING_BANKROLL) / STARTING_BANKROLL * 100
        })
        
        # Stop if bankroll gets too low
        if bankroll < MIN_BET:
            print(f"   Stopping: Bankroll dropped below minimum bet size")
            break

# Create results
bets_df = pd.DataFrame(all_bets)

print(f"\n4. Backtest Results:")
print(f"   Total bets placed: {len(bets_df):,}")
print(f"   Final bankroll: ${bankroll:,.2f}")
print(f"   Total return: {(bankroll-STARTING_BANKROLL)/STARTING_BANKROLL*100:+,.1f}%")
print(f"   Bets won: {bets_df['bet_won'].sum():,}")
print(f"   Win rate: {bets_df['bet_won'].mean():.1%}")
print(f"   Average edge: {bets_df['edge'].mean():+.3f}")
print(f"   Average Kelly %: {bets_df['kelly_fraction'].mean()*100:.1f}%")

# Save detailed CSV
print(f"\n5. Saving detailed betting log...")
os.makedirs('analysis_scripts/betting_logs', exist_ok=True)
bets_df.to_csv('analysis_scripts/betting_logs/detailed_kelly_bets.csv', index=False)

print(f"   ✅ Saved: analysis_scripts/betting_logs/detailed_kelly_bets.csv")
print(f"   Columns: {list(bets_df.columns)}")

# Show first 10 bets
print(f"\n6. First 10 bets:")
print("Bet# | Player Bet On    | Odds | Edge  | Bet$  | Won? | Profit | Bankroll")
print("-" * 80)
for _, bet in bets_df.head(10).iterrows():
    bet_num = int(bet['bet_number'])
    player = bet['bet_on_player'][:12]
    odds = bet['odds_used']
    edge = bet['edge']
    bet_amt = bet['bet_amount']
    won = "✓" if bet['bet_won'] else "✗"
    profit = bet['profit']
    bankroll_after = bet['bankroll_after']
    
    print(f"{bet_num:4d} | {player:16s} | {odds:4.2f} | {edge:+.3f} | ${bet_amt:5.0f} | {won:1s}  | ${profit:+6.0f} | ${bankroll_after:7.0f}")

# Create bankroll progression chart
print(f"\n7. Creating bankroll progression chart...")
plt.figure(figsize=(12, 8))
bet_numbers = [0] + list(bets_df['bet_number'])
bankrolls = [STARTING_BANKROLL] + list(bets_df['bankroll_after'])

plt.plot(bet_numbers, bankrolls, linewidth=2, color='blue')
plt.axhline(y=STARTING_BANKROLL, color='red', linestyle='--', alpha=0.7, label='Starting Bankroll')
plt.title('Neural Network Kelly Betting - Bankroll Progression')
plt.xlabel('Bet Number')
plt.ylabel('Bankroll ($)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()

os.makedirs('analysis_scripts/charts', exist_ok=True)
plt.savefig('analysis_scripts/charts/detailed_bankroll_progression.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✅ Chart saved: analysis_scripts/charts/detailed_bankroll_progression.png")

print(f"\n" + "="*80)
print(f"DETAILED KELLY TRACKING COMPLETE")
print(f"="*80)
print(f"Check the CSV file to inspect every bet and verify Kelly logic!")
print(f"Key columns to check:")
print(f"- bankroll_before: Should increase over time")
print(f"- bet_as_pct_bankroll: Should be Kelly % of CURRENT bankroll")
print(f"- edge: Model prob - Market prob")
print(f"- bankroll_growth: Should compound correctly")