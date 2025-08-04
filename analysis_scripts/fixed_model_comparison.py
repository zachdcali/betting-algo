#!/usr/bin/env python3
"""
Fixed comparison: Filter ML-ready dataset to matched games and test models
"""
import pandas as pd
import numpy as np
import sys
import os
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier

# Add the models directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'Models', 'professional_tennis'))

class TennisNet(nn.Module):
    """Neural Network architecture (same as in train_nn.py)"""
    def __init__(self, input_size):
        super(TennisNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def get_feature_columns(df):
    """Get feature columns (copied from preprocess.py logic)"""
    exclude_cols = ['tourney_date', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id',
                   'winner_name', 'loser_name', 'score', 'Player1_Name', 'Player2_Name', 'Player1_Wins',
                   'best_of', 'round', 'minutes', 'data_source', 'year']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    return feature_cols

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
print("FIXED MODEL COMPARISON ON IDENTICAL MATCHES")
print("="*80)

# 1. Load datasets
print("\n1. Loading datasets...")
ml_ready_path = 'data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv'
jeff_matched_path = 'data/JeffSackmann/jeffsackmann_exact_matched_final.csv'
tennis_matched_path = 'data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv'

# Load ML-ready dataset
ml_ready = pd.read_csv(ml_ready_path, low_memory=False)
ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])

# Load matched datasets
jeff_matched = pd.read_csv(jeff_matched_path, low_memory=False)
jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
tennis_matched = pd.read_csv(tennis_matched_path, low_memory=False)
tennis_matched['Date'] = pd.to_datetime(tennis_matched['Date'])

print(f"   ML-ready dataset: {len(ml_ready):,} matches")
print(f"   Matched Jeff Sackmann: {len(jeff_matched):,} matches")
print(f"   Matched Tennis-Data: {len(tennis_matched):,} matches")

# 2. Filter ML-ready dataset to matched games
print("\n2. Filtering ML-ready dataset to matched games...")

# Create lookup for matched games
matched_lookup = set()
for _, row in jeff_matched.iterrows():
    # Create a key using date, winner, loser
    key = (row['tourney_date'].strftime('%Y-%m-%d'), 
           str(row['winner_name']).strip().lower(), 
           str(row['loser_name']).strip().lower())
    matched_lookup.add(key)

print(f"   Created lookup with {len(matched_lookup):,} matched games")

# Filter ML-ready data
ml_matched = []
for idx, row in ml_ready.iterrows():
    # Create same key format
    key = (row['tourney_date'].strftime('%Y-%m-%d'),
           str(row.get('Player1_Name', '')).strip().lower(),
           str(row.get('Player2_Name', '')).strip().lower())
    
    # Also try swapped order
    key_swapped = (row['tourney_date'].strftime('%Y-%m-%d'),
                   str(row.get('Player2_Name', '')).strip().lower(),
                   str(row.get('Player1_Name', '')).strip().lower())
    
    if key in matched_lookup:
        ml_matched.append(row)
    elif key_swapped in matched_lookup:
        # Swap the target since player order is swapped
        row_copy = row.copy()
        row_copy['Player1_Wins'] = 1 - row_copy['Player1_Wins']
        ml_matched.append(row_copy)

ml_matched_df = pd.DataFrame(ml_matched)
print(f"   Filtered to {len(ml_matched_df):,} ML-ready matched games")

if len(ml_matched_df) == 0:
    print("   ❌ No matches found in ML-ready data. Check name matching logic.")
    sys.exit(1)

# Add year column
ml_matched_df['year'] = ml_matched_df['tourney_date'].dt.year

print(f"   Year breakdown:")
year_counts = ml_matched_df['year'].value_counts().sort_index()
for year, count in year_counts.items():
    print(f"     {year}: {count:,} matches")

# 3. Prepare features
print("\n3. Preparing features...")
feature_cols = get_feature_columns(ml_matched_df)

# Filter to numeric columns only
numeric_cols = []
for col in feature_cols:
    if col in ml_matched_df.columns:
        if ml_matched_df[col].dtype in ['int64', 'float64', 'bool']:
            numeric_cols.append(col)

feature_cols = numeric_cols
X_test = ml_matched_df[feature_cols]
y_test = ml_matched_df['Player1_Wins']

# Fill missing values
X_test = X_test.fillna(X_test.median())

print(f"   Features: {len(feature_cols)}")
print(f"   Test shape: {X_test.shape}")
print(f"   Target distribution: {y_test.mean():.3f}")

# 4. Calculate baseline accuracies
print("\n4. Calculating baseline accuracies...")

# ATP ranking accuracy on ML-ready data
ml_atp_clean = ml_matched_df.dropna(subset=['Player1_Rank', 'Player2_Rank']).copy()
ml_atp_clean['Player1_Rank'] = pd.to_numeric(ml_atp_clean['Player1_Rank'], errors='coerce')
ml_atp_clean['Player2_Rank'] = pd.to_numeric(ml_atp_clean['Player2_Rank'], errors='coerce')
ml_atp_clean = ml_atp_clean.dropna(subset=['Player1_Rank', 'Player2_Rank'])

if len(ml_atp_clean) > 0:
    # Player1 wins if they have better (lower) rank
    atp_correct = (ml_atp_clean['Player1_Rank'] < ml_atp_clean['Player2_Rank']).sum()
    atp_accuracy = atp_correct / len(ml_atp_clean)
    print(f"   ATP Ranking Baseline: {atp_accuracy:.4f} ({atp_accuracy*100:.2f}%) on {len(ml_atp_clean):,} matches")

# Betting odds accuracy on Tennis-Data
tennis_clean = tennis_matched.dropna(subset=['WRank', 'LRank']).copy()
tennis_clean['WRank'] = pd.to_numeric(tennis_clean['WRank'], errors='coerce')
tennis_clean['LRank'] = pd.to_numeric(tennis_clean['LRank'], errors='coerce')
tennis_clean = tennis_clean.dropna(subset=['WRank', 'LRank'])

betting_acc, betting_count = calculate_betting_accuracy(tennis_clean, 'AvgW', 'AvgL')
if betting_acc is not None:
    print(f"   Professional Betting Odds: {betting_acc:.4f} ({betting_acc*100:.2f}%) on {betting_count:,} matches")

# 5. Test models
print(f"\n5. Testing trained models...")
results = {}

# Test XGBoost
print(f"\n   Testing XGBoost...")
xgb_model_path = 'results/professional_tennis/XGBoost/xgboost_model.json'
if os.path.exists(xgb_model_path):
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model(xgb_model_path)
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = xgb_accuracy
        print(f"   ✅ XGBoost Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"   ❌ XGBoost prediction failed: {e}")
else:
    print(f"   ❌ XGBoost model not found")

# Test Random Forest
print(f"\n   Testing Random Forest...")
rf_model_path = 'results/professional_tennis/Random_Forest/random_forest_model.pkl'
if os.path.exists(rf_model_path):
    try:
        with open(rf_model_path, 'rb') as f:
            rf_model = pickle.load(f)
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        results['Random_Forest'] = rf_accuracy
        print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"   ❌ Random Forest prediction failed: {e}")
else:
    print(f"   ❌ Random Forest model not found")

# Test Neural Network
print(f"\n   Testing Neural Network...")
nn_model_path = 'results/professional_tennis/Neural_Network/neural_network_model.pth'
nn_scaler_path = 'results/professional_tennis/Neural_Network/scaler.pkl'
if os.path.exists(nn_model_path) and os.path.exists(nn_scaler_path):
    try:
        # Load scaler
        with open(nn_scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Load neural network
        input_size = X_test.shape[1]
        nn_model = TennisNet(input_size)
        nn_model.load_state_dict(torch.load(nn_model_path, map_location='cpu'))
        nn_model.eval()
        
        # Scale and predict
        X_test_scaled = scaler.transform(X_test.values)
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        
        with torch.no_grad():
            nn_pred_proba = nn_model(X_test_tensor).squeeze().numpy()
            nn_pred = (nn_pred_proba > 0.5).astype(int)
        
        nn_accuracy = accuracy_score(y_test, nn_pred)
        results['Neural_Network'] = nn_accuracy
        print(f"   ✅ Neural Network Accuracy: {nn_accuracy:.4f} ({nn_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"   ❌ Neural Network prediction failed: {e}")
else:
    print(f"   ❌ Neural Network model files not found")

# 6. Year-by-year breakdown
print(f"\n6. Year-by-year breakdown...")
for year in [2023, 2024]:
    year_mask = ml_matched_df['year'] == year
    year_data = ml_matched_df[year_mask]
    year_tennis = tennis_matched[tennis_matched['Date'].dt.year == year]
    
    if len(year_data) == 0:
        continue
        
    print(f"\n   {year} ({len(year_data):,} matches):")
    
    # ATP baseline for year
    year_atp = year_data.dropna(subset=['Player1_Rank', 'Player2_Rank']).copy()
    year_atp['Player1_Rank'] = pd.to_numeric(year_atp['Player1_Rank'], errors='coerce')
    year_atp['Player2_Rank'] = pd.to_numeric(year_atp['Player2_Rank'], errors='coerce')
    year_atp = year_atp.dropna(subset=['Player1_Rank', 'Player2_Rank'])
    
    if len(year_atp) > 0:
        year_atp_correct = (year_atp['Player1_Rank'] < year_atp['Player2_Rank']).sum()
        year_atp_accuracy = year_atp_correct / len(year_atp)
        print(f"     ATP Baseline: {year_atp_accuracy:.4f} ({year_atp_accuracy*100:.2f}%)")
    
    # Betting odds for year
    year_betting_acc, year_betting_count = calculate_betting_accuracy(year_tennis, 'AvgW', 'AvgL')
    if year_betting_acc is not None:
        print(f"     Betting Odds: {year_betting_acc:.4f} ({year_betting_acc*100:.2f}%)")
    
    # Model accuracies for year
    year_X = year_data[feature_cols].fillna(year_data[feature_cols].median())
    year_y = year_data['Player1_Wins']
    
    for model_name, _ in results.items():
        if model_name == 'XGBoost' and 'XGBoost' in results:
            year_pred = xgb_model.predict(year_X)
            year_acc = accuracy_score(year_y, year_pred)
            print(f"     {model_name}: {year_acc:.4f} ({year_acc*100:.2f}%)")
        elif model_name == 'Random_Forest' and 'Random_Forest' in results:
            year_pred = rf_model.predict(year_X)
            year_acc = accuracy_score(year_y, year_pred)
            print(f"     {model_name}: {year_acc:.4f} ({year_acc*100:.2f}%)")
        elif model_name == 'Neural_Network' and 'Neural_Network' in results:
            year_X_scaled = scaler.transform(year_X.values)
            year_X_tensor = torch.FloatTensor(year_X_scaled)
            with torch.no_grad():
                year_pred_proba = nn_model(year_X_tensor).squeeze().numpy()
                year_pred = (year_pred_proba > 0.5).astype(int)
            year_acc = accuracy_score(year_y, year_pred)
            print(f"     {model_name}: {year_acc:.4f} ({year_acc*100:.2f}%)")

# 7. Final comparison
print(f"\n" + "="*80)
print(f"FINAL APPLES-TO-APPLES COMPARISON")
print(f"="*80)
print(f"Dataset: {len(ml_matched_df):,} identical matches")
print(f"")

# Compile all results
all_results = {
    'ATP_Baseline': atp_accuracy,
    'Betting_Odds': betting_acc,
    **results
}

# Sort by accuracy
sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)

print(f"Overall Performance Rankings:")
for i, (method, accuracy) in enumerate(sorted_results):
    if method == 'Betting_Odds':
        method_name = "Professional Betting Odds"
        note = " (Tennis-Data.co.uk)"
    elif method == 'ATP_Baseline':
        method_name = "ATP Ranking Baseline"
        note = ""
    else:
        method_name = f"Your {method.replace('_', ' ')} Model"
        note = ""
    
    print(f"   {i+1}. {method_name}: {accuracy:.4f} ({accuracy*100:.2f}%){note}")

# Check which models beat betting odds
models_beat_betting = []
for method, accuracy in results.items():
    if accuracy > betting_acc:
        improvement = (accuracy - betting_acc) * 100
        models_beat_betting.append((method, improvement))

print(f"")
if models_beat_betting:
    print(f"🎉 MODELS THAT BEAT PROFESSIONAL BETTING ODDS:")
    for method, improvement in models_beat_betting:
        print(f"   {method.replace('_', ' ')}: +{improvement:.2f} percentage points")
else:
    print(f"📊 No models beat professional betting odds ({betting_acc*100:.2f}%)")
    if len(results) > 0:
        best_model = max(results.items(), key=lambda x: x[1])
        gap = (betting_acc - best_model[1]) * 100
        print(f"   Best model ({best_model[0].replace('_', ' ')}): -{gap:.2f} percentage points vs betting odds")

print(f"\nTrue apples-to-apples comparison complete! 🎯")