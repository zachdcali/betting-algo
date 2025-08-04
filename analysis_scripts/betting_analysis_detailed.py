#!/usr/bin/env python3
"""
Detailed betting analysis: Why do models lose money despite being more accurate?
"""
import pandas as pd
import numpy as np
import sys
import os
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
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
    """Convert decimal odds to implied probability"""
    return 1.0 / decimal_odds

def kelly_fraction(model_prob, decimal_odds):
    """Calculate Kelly Criterion betting fraction"""
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
print("DETAILED BETTING ANALYSIS - WHY DO MODELS LOSE MONEY?")
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
ml_verified_df['year'] = ml_verified_df['tourney_date'].dt.year

# Prepare features
feature_cols = get_feature_columns(ml_verified_df)
numeric_cols = [col for col in feature_cols if col in ml_verified_df.columns and 
                ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]

X_test = ml_verified_df[numeric_cols].fillna(ml_verified_df[numeric_cols].median())
y_test = ml_verified_df['Player1_Wins']

print(f"   Analyzing {len(ml_verified_df):,} matches")

# Load models and get probabilities
print("\n2. Loading models and analyzing prediction quality...")
model_probs = {}
model_accuracies = {}
model_aucs = {}

# XGBoost
xgb_path = 'results/professional_tennis/XGBoost/xgboost_model.json'
if os.path.exists(xgb_path):
    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(xgb_path)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_preds = xgb_model.predict(X_test)
    model_probs['XGBoost'] = xgb_probs
    model_accuracies['XGBoost'] = accuracy_score(y_test, xgb_preds)
    model_aucs['XGBoost'] = roc_auc_score(y_test, xgb_probs)

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
    
    nn_preds = (nn_probs > 0.5).astype(int)
    model_probs['Neural_Network'] = nn_probs
    model_accuracies['Neural_Network'] = accuracy_score(y_test, nn_preds)
    model_aucs['Neural_Network'] = roc_auc_score(y_test, nn_probs)

# Calculate market accuracy and calibration
print("\n3. Analyzing market vs model performance...")
market_probs = []
market_correct = []

for i in range(len(ml_verified_df)):
    tennis_idx = tennis_verified_mapping.get(i)
    if tennis_idx is None:
        continue
        
    tennis_row = tennis_100_verified.iloc[tennis_idx]
    actual_p1_win = y_test.iloc[i]
    
    avg_winner_odds = tennis_row.get('AvgW')
    avg_loser_odds = tennis_row.get('AvgL')
    
    if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
        continue
        
    avg_winner_odds = float(avg_winner_odds)
    avg_loser_odds = float(avg_loser_odds)
    
    # Get market probability for Player1
    if actual_p1_win == 1:
        p1_odds = avg_winner_odds
    else:
        p1_odds = avg_loser_odds
    
    market_p1_prob = decimal_to_implied_prob(p1_odds)
    market_probs.append(market_p1_prob)
    market_correct.append(actual_p1_win)

market_probs = np.array(market_probs)
market_correct = np.array(market_correct)
market_accuracy = np.mean((market_probs > 0.5) == market_correct)
market_auc = roc_auc_score(market_correct, market_probs)

print(f"   Market accuracy: {market_accuracy:.4f} ({market_accuracy*100:.2f}%)")
print(f"   Market AUC: {market_auc:.4f}")

for model_name in model_probs:
    print(f"   {model_name} accuracy: {model_accuracies[model_name]:.4f} ({model_accuracies[model_name]*100:.2f}%)")
    print(f"   {model_name} AUC: {model_aucs[model_name]:.4f}")

# Analyze betting edge distribution
print("\n4. Analyzing betting edge distribution...")

for model_name, model_probabilities in model_probs.items():
    print(f"\n=== {model_name} EDGE ANALYSIS ===")
    
    edges = []
    positive_ev_count = 0
    
    for i in range(len(ml_verified_df)):
        tennis_idx = tennis_verified_mapping.get(i)
        if tennis_idx is None:
            continue
            
        tennis_row = tennis_100_verified.iloc[tennis_idx]
        actual_p1_win = y_test.iloc[i]
        
        avg_winner_odds = tennis_row.get('AvgW')
        avg_loser_odds = tennis_row.get('AvgL')
        
        if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
            continue
            
        avg_winner_odds = float(avg_winner_odds)
        avg_loser_odds = float(avg_loser_odds)
        
        if actual_p1_win == 1:
            p1_odds = avg_winner_odds
            p2_odds = avg_loser_odds
        else:
            p1_odds = avg_loser_odds  
            p2_odds = avg_winner_odds
        
        market_p1_prob = decimal_to_implied_prob(p1_odds)
        market_p2_prob = decimal_to_implied_prob(p2_odds)
        
        model_p1_prob = model_probabilities[i]
        model_p2_prob = 1 - model_p1_prob
        
        # Calculate Kelly criterion for each bet to determine actual edge
        kelly_p1 = kelly_fraction(model_p1_prob, p1_odds)
        kelly_p2 = kelly_fraction(model_p2_prob, p2_odds)
        
        # Use the bet we would actually place
        if kelly_p1 > kelly_p2 and kelly_p1 > 0:
            edges.append(model_p1_prob - market_p1_prob)
            positive_ev_count += 1
        elif kelly_p2 > 0:
            edges.append(model_p2_prob - market_p2_prob)  
            positive_ev_count += 1
        else:
            # No positive EV bet
            max_edge = max(model_p1_prob - market_p1_prob, model_p2_prob - market_p2_prob)
            edges.append(max_edge)
    
    edges = np.array(edges)
    
    print(f"   Matches with positive EV: {positive_ev_count:,} ({positive_ev_count/len(edges)*100:.1f}%)")
    print(f"   Average edge: {edges.mean():+.4f}")
    
    positive_edges = edges[edges > 0]
    negative_edges = edges[edges < 0]
    
    if len(positive_edges) > 0:
        print(f"   Average positive edge: {positive_edges.mean():+.4f}")
    else:
        print(f"   Average positive edge: No positive edges found")
        
    if len(negative_edges) > 0:
        print(f"   Average negative edge: {negative_edges.mean():+.4f}")
    else:
        print(f"   Average negative edge: No negative edges found")
        
    print(f"   Max edge: {edges.max():+.4f}")
    print(f"   Min edge: {edges.min():+.4f}")

# Test alternative betting strategies
print("\n5. Testing alternative betting strategies...")

STRATEGIES = {
    'Kelly_Capped_10%': {'type': 'kelly', 'max_bet': 0.10},
    'Kelly_Capped_5%': {'type': 'kelly', 'max_bet': 0.05},
    'Fixed_2%': {'type': 'fixed', 'bet_pct': 0.02},
    'Fixed_1%': {'type': 'fixed', 'bet_pct': 0.01},
}

strategy_results = {}

for model_name, model_probabilities in model_probs.items():
    print(f"\n--- {model_name} STRATEGY COMPARISON ---")
    strategy_results[model_name] = {}
    
    for strategy_name, strategy_config in STRATEGIES.items():
        bankroll = 100.0
        total_bets = 0
        winning_bets = 0
        bankroll_history = [bankroll]
        
        for i in range(len(ml_verified_df)):
            tennis_idx = tennis_verified_mapping.get(i)
            if tennis_idx is None:
                continue
                
            tennis_row = tennis_100_verified.iloc[tennis_idx]
            actual_p1_win = y_test.iloc[i]
            
            avg_winner_odds = tennis_row.get('AvgW')
            avg_loser_odds = tennis_row.get('AvgL')
            
            if pd.isna(avg_winner_odds) or pd.isna(avg_loser_odds):
                continue
                
            avg_winner_odds = float(avg_winner_odds)
            avg_loser_odds = float(avg_loser_odds)
            
            if actual_p1_win == 1:
                p1_odds = avg_winner_odds
                p2_odds = avg_loser_odds
            else:
                p1_odds = avg_loser_odds  
                p2_odds = avg_winner_odds
            
            model_p1_prob = model_probabilities[i]
            
            # Check for betting opportunities
            kelly_p1 = kelly_fraction(model_p1_prob, p1_odds)
            kelly_p2 = kelly_fraction(1 - model_p1_prob, p2_odds)
            
            bet_placed = False
            bet_amount = 0
            
            # Only bet if we have positive EV (Kelly > 0)
            if kelly_p1 > 0.01:  # Positive EV on Player1
                if strategy_config['type'] == 'kelly':
                    bet_amount = bankroll * min(kelly_p1, strategy_config['max_bet'])
                else:  # fixed - only bet fixed amount if Kelly is positive
                    bet_amount = bankroll * strategy_config['bet_pct']
                bet_on_p1 = True
                odds_used = p1_odds
                bet_won = (actual_p1_win == 1)
                bet_placed = True
            elif kelly_p2 > 0.01:  # Positive EV on Player2
                if strategy_config['type'] == 'kelly':
                    bet_amount = bankroll * min(kelly_p2, strategy_config['max_bet'])
                else:  # fixed - only bet fixed amount if Kelly is positive
                    bet_amount = bankroll * strategy_config['bet_pct']
                bet_on_p1 = False
                odds_used = p2_odds
                bet_won = (actual_p1_win == 0)
                bet_placed = True
            
            if bet_placed and bet_amount > 0 and bankroll >= bet_amount:
                total_bets += 1
                
                if bet_won:
                    profit = bet_amount * (odds_used - 1)
                    winning_bets += 1
                else:
                    profit = -bet_amount
                
                bankroll += profit
                bankroll_history.append(bankroll)
                
                # Stop if bankroll too low
                if bankroll < 1.0:
                    break
        
        win_rate = winning_bets / total_bets if total_bets > 0 else 0
        total_return = ((bankroll - 100) / 100) * 100
        
        strategy_results[model_name][strategy_name] = {
            'final_bankroll': bankroll,
            'total_return': total_return,
            'total_bets': total_bets,
            'win_rate': win_rate,
            'bankroll_history': bankroll_history
        }
        
        print(f"   {strategy_name:15s}: ${bankroll:6.2f} ({total_return:+6.1f}%) | {total_bets:3d} bets | {win_rate:.1%} win rate")

# Create bankroll charts
print("\n6. Creating bankroll progression charts...")
os.makedirs('analysis_scripts/charts', exist_ok=True)

for model_name in model_probs:
    plt.figure(figsize=(12, 8))
    
    for strategy_name in STRATEGIES:
        history = strategy_results[model_name][strategy_name]['bankroll_history']
        plt.plot(range(len(history)), history, label=strategy_name, linewidth=2)
    
    plt.axhline(y=100, color='black', linestyle='--', alpha=0.5, label='Starting Bankroll')
    plt.title(f'{model_name} - Bankroll Progression by Strategy')
    plt.xlabel('Bets Placed')
    plt.ylabel('Bankroll ($)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale to see all curves
    plt.tight_layout()
    plt.savefig(f'analysis_scripts/charts/{model_name}_bankroll_progression.png', dpi=300, bbox_inches='tight')
    plt.close()

print(f"   Bankroll charts saved to: analysis_scripts/charts/")

print(f"\n" + "="*80)
print(f"KEY INSIGHTS:")
print(f"="*80)
print(f"1. Small prediction edge ({model_accuracies['Neural_Network']-market_accuracy:+.1%}) doesn't guarantee betting profit")
print(f"2. Market odds already incorporate most predictive information")
print(f"3. Even positive EV bets can lose due to variance and bet sizing")
print(f"4. AUC shows probability calibration quality (higher = better)")
print(f"5. Check bankroll charts to see if it's consistent losses or big drawdowns")
print(f"✅ Analysis complete!")