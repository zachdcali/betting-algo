#!/usr/bin/env python3
"""
FINAL FIXED: Properly match ML-ready (Player1/Player2) with matched games (winner/loser)
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

def calculate_betting_accuracy(data, win_col, lose_col):
    odds_data = data.dropna(subset=[win_col, lose_col]).copy()
    if len(odds_data) == 0:
        return None, 0
    
    odds_data[win_col] = pd.to_numeric(odds_data[win_col], errors='coerce')
    odds_data[lose_col] = pd.to_numeric(odds_data[lose_col], errors='coerce')
    odds_clean = odds_data.dropna(subset=[win_col, lose_col])
    
    if len(odds_clean) == 0:
        return None, 0
    
    betting_correct = (odds_clean[win_col] < odds_clean[lose_col]).sum()
    return betting_correct / len(odds_clean), len(odds_clean)

print("="*80)
print("FINAL FIXED COMPARISON - PROPER PLAYER1/PLAYER2 vs WINNER/LOSER MATCHING")
print("="*80)

# Load datasets
print("\n1. Loading datasets...")
ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)

ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
tennis_matched['Date'] = pd.to_datetime(tennis_matched['Date'])

print(f"   ML-ready: {len(ml_ready):,}, Matched Jeff: {len(jeff_matched):,}, Matched Tennis: {len(tennis_matched):,}")

# Create proper matching between Player1/Player2 and winner/loser
print("\n2. Matching ML-ready data with identical games...")
ml_filtered = []

for _, matched_row in jeff_matched.iterrows():
    date = matched_row['tourney_date']
    winner = str(matched_row['winner_name']).strip()
    loser = str(matched_row['loser_name']).strip()
    
    # Find corresponding ML-ready row on same date
    ml_candidates = ml_ready[ml_ready['tourney_date'] == date]
    
    for _, ml_row in ml_candidates.iterrows():
        player1 = str(ml_row['Player1_Name']).strip()
        player2 = str(ml_row['Player2_Name']).strip()
        
        # Check if players match (either order)
        if (player1 == winner and player2 == loser):
            # Player1 = winner, so Player1_Wins should be 1
            ml_row_copy = ml_row.copy()
            ml_row_copy['Player1_Wins'] = 1
            ml_filtered.append(ml_row_copy)
            break
        elif (player1 == loser and player2 == winner):
            # Player1 = loser, so Player1_Wins should be 0  
            ml_row_copy = ml_row.copy()
            ml_row_copy['Player1_Wins'] = 0
            ml_filtered.append(ml_row_copy)
            break

ml_matched_df = pd.DataFrame(ml_filtered)
print(f"   Successfully matched {len(ml_matched_df):,} games")

if len(ml_matched_df) == 0:
    print("   ❌ No matches found!")
    sys.exit(1)

# Check target distribution
ml_matched_df['year'] = ml_matched_df['tourney_date'].dt.year
target_dist = ml_matched_df['Player1_Wins'].mean()
print(f"   Target distribution: {target_dist:.3f} (should be ~0.5)")

year_counts = ml_matched_df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"   {year}: {count:,} matches")

# Prepare features
print("\n3. Preparing features...")
feature_cols = get_feature_columns(ml_matched_df)
numeric_cols = [col for col in feature_cols if col in ml_matched_df.columns and 
                ml_matched_df[col].dtype in ['int64', 'float64', 'bool']]

X_test = ml_matched_df[numeric_cols].fillna(ml_matched_df[numeric_cols].median())
y_test = ml_matched_df['Player1_Wins']

print(f"   Features: {len(numeric_cols)}, Shape: {X_test.shape}")
print(f"   Target distribution: {y_test.mean():.3f}")

# Calculate baselines
print("\n4. Calculating baselines...")

# ATP ranking baseline - use ACTUAL winner/loser ranks from original data
atp_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
atp_clean['winner_rank'] = pd.to_numeric(atp_clean['winner_rank'], errors='coerce')
atp_clean['loser_rank'] = pd.to_numeric(atp_clean['loser_rank'], errors='coerce')
atp_clean = atp_clean.dropna(subset=['winner_rank', 'loser_rank'])

# Winner should have better (lower) rank than loser
atp_correct = (atp_clean['winner_rank'] < atp_clean['loser_rank']).sum()
atp_accuracy = atp_correct / len(atp_clean)
print(f"   ATP Baseline: {atp_accuracy:.4f} ({atp_accuracy*100:.2f}%) on {len(atp_clean):,} matches")

# Betting odds
betting_acc, betting_count = calculate_betting_accuracy(tennis_matched, 'AvgW', 'AvgL')
print(f"   Betting Odds: {betting_acc:.4f} ({betting_acc*100:.2f}%) on {betting_count:,} matches")

# Test models
print(f"\n5. Testing models...")
results = {}

# XGBoost
xgb_path = 'results/professional_tennis/XGBoost/xgboost_model.json'
if os.path.exists(xgb_path):
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_path)
        xgb_pred = xgb_model.predict(X_test)
        xgb_acc = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = xgb_acc
        print(f"   XGBoost: {xgb_acc:.4f} ({xgb_acc*100:.2f}%)")
    except Exception as e:
        print(f"   XGBoost failed: {e}")

# Random Forest
rf_path = 'results/professional_tennis/Random_Forest/random_forest_model.pkl'
if os.path.exists(rf_path):
    try:
        with open(rf_path, 'rb') as f:
            rf_model = pickle.load(f)
        rf_pred = rf_model.predict(X_test)
        rf_acc = accuracy_score(y_test, rf_pred)
        results['Random_Forest'] = rf_acc
        print(f"   Random Forest: {rf_acc:.4f} ({rf_acc*100:.2f}%)")
    except Exception as e:
        print(f"   Random Forest failed: {e}")

# Neural Network
nn_path = 'results/professional_tennis/Neural_Network/neural_network_model.pth'
scaler_path = 'results/professional_tennis/Neural_Network/scaler.pkl'
if os.path.exists(nn_path) and os.path.exists(scaler_path):
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        nn_model = TennisNet(X_test.shape[1])
        nn_model.load_state_dict(torch.load(nn_path, map_location='cpu'))
        nn_model.eval()
        
        X_scaled = scaler.transform(X_test.values)
        X_tensor = torch.FloatTensor(X_scaled)
        
        with torch.no_grad():
            nn_pred_proba = nn_model(X_tensor).squeeze().numpy()
            nn_pred = (nn_pred_proba > 0.5).astype(int)
        
        nn_acc = accuracy_score(y_test, nn_pred)
        results['Neural_Network'] = nn_acc
        print(f"   Neural Network: {nn_acc:.4f} ({nn_acc*100:.2f}%)")
    except Exception as e:
        print(f"   Neural Network failed: {e}")

# Year-by-year breakdown
print(f"\n6. Year-by-year breakdown...")
for year in [2023, 2024]:
    year_data = ml_matched_df[ml_matched_df['year'] == year]
    year_tennis = tennis_matched[tennis_matched['Date'].dt.year == year]
    
    if len(year_data) == 0:
        continue
        
    print(f"\n   {year} ({len(year_data):,} matches):")
    
    # ATP baseline - use actual winner/loser ranks
    year_jeff = jeff_matched[jeff_matched['tourney_date'].dt.year == year]
    year_atp = year_jeff.dropna(subset=['winner_rank', 'loser_rank']).copy()
    year_atp['winner_rank'] = pd.to_numeric(year_atp['winner_rank'], errors='coerce')
    year_atp['loser_rank'] = pd.to_numeric(year_atp['loser_rank'], errors='coerce')
    year_atp = year_atp.dropna(subset=['winner_rank', 'loser_rank'])
    
    if len(year_atp) > 0:
        year_atp_correct = (year_atp['winner_rank'] < year_atp['loser_rank']).sum()
        year_atp_acc = year_atp_correct / len(year_atp)
        print(f"     ATP Baseline: {year_atp_acc:.4f} ({year_atp_acc*100:.2f}%)")
    
    # Betting odds
    year_betting_acc, _ = calculate_betting_accuracy(year_tennis, 'AvgW', 'AvgL')
    if year_betting_acc is not None:
        print(f"     Betting Odds: {year_betting_acc:.4f} ({year_betting_acc*100:.2f}%)")
    
    # Models
    year_X = year_data[numeric_cols].fillna(year_data[numeric_cols].median())
    year_y = year_data['Player1_Wins']
    
    for model_name in results:
        if model_name == 'XGBoost':
            year_pred = xgb_model.predict(year_X)
        elif model_name == 'Random_Forest':
            year_pred = rf_model.predict(year_X)
        elif model_name == 'Neural_Network':
            year_X_scaled = scaler.transform(year_X.values)
            year_X_tensor = torch.FloatTensor(year_X_scaled)
            with torch.no_grad():
                year_pred_proba = nn_model(year_X_tensor).squeeze().numpy()
                year_pred = (year_pred_proba > 0.5).astype(int)
        
        year_acc = accuracy_score(year_y, year_pred)
        print(f"     {model_name}: {year_acc:.4f} ({year_acc*100:.2f}%)")

# Final results
print(f"\n" + "="*80)
print(f"FINAL COMPARISON - {len(ml_matched_df):,} IDENTICAL MATCHES")
print(f"="*80)

all_results = {'ATP_Baseline': atp_accuracy, 'Betting_Odds': betting_acc, **results}
sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

print(f"Performance Rankings:")
for i, (method, acc) in enumerate(sorted_results):
    if method == 'Betting_Odds':
        name = "Professional Betting Odds"
    elif method == 'ATP_Baseline':
        name = "ATP Ranking Baseline"
    else:
        name = f"Your {method.replace('_', ' ')} Model"
    
    print(f"   {i+1}. {name}: {acc:.4f} ({acc*100:.2f}%)")

# Winners
winners = [m for m, acc in results.items() if acc > betting_acc]
if winners:
    print(f"\n🎉 MODELS BEATING BETTING ODDS:")
    for model in winners:
        improvement = (results[model] - betting_acc) * 100
        print(f"   {model}: +{improvement:.2f} percentage points")
else:
    print(f"\n📊 No models beat betting odds ({betting_acc*100:.2f}%)")
    if results:
        best = max(results.items(), key=lambda x: x[1])
        gap = (betting_acc - best[1]) * 100
        print(f"   Best model ({best[0]}): -{gap:.2f} percentage points vs betting odds")

print(f"\nApples-to-apples comparison complete! 🎯")