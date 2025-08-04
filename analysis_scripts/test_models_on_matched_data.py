#!/usr/bin/env python3
"""
Test trained XGBoost, Neural Network, and Random Forest models on the 1,646 matched identical games
to compare against 68.34% betting odds accuracy
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

def load_model_if_exists(model_path, model_type):
    """Load a saved model if it exists"""
    if os.path.exists(model_path):
        print(f"   ✅ Loading {model_type} from {model_path}")
        if model_type == "XGBoost":
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            return model
        elif model_type in ["Random Forest", "Scaler"]:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        return True  # For Neural Network
    else:
        print(f"   ❌ {model_type} not found at {model_path}")
        return None

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

print("="*80)
print("TESTING TRAINED MODELS ON 1,646 IDENTICAL MATCHES")
print("="*80)

# 1. Load the matched dataset
print("\n1. Loading matched dataset...")
jeff_matched_path = 'data/JeffSackmann/jeffsackmann_exact_matched_final.csv'
tennis_matched_path = 'data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv'

if not os.path.exists(jeff_matched_path):
    print(f"   ❌ Matched Jeff Sackmann data not found at {jeff_matched_path}")
    sys.exit(1)

jeff_matched = pd.read_csv(jeff_matched_path, low_memory=False)
tennis_matched = pd.read_csv(tennis_matched_path, low_memory=False)

print(f"   Loaded {len(jeff_matched):,} matched games")

# 2. Check if we need ML-ready features
print("\n2. Checking for ML-ready features...")
ml_ready_path = 'data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv'

if os.path.exists(ml_ready_path):
    print(f"   Loading full ML-ready dataset to get features...")
    ml_ready_full = pd.read_csv(ml_ready_path, low_memory=False)
    
    # Create a mapping to match our subset
    jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
    ml_ready_full['tourney_date'] = pd.to_datetime(ml_ready_full['tourney_date'])
    
    # Try to match by date, winner, loser
    matched_ml_data = []
    
    for idx, row in jeff_matched.iterrows():
        # Find corresponding ML-ready row
        matching_rows = ml_ready_full[
            (ml_ready_full['tourney_date'] == row['tourney_date']) &
            (ml_ready_full.get('Player1_Name', '') == row['winner_name']) &
            (ml_ready_full.get('Player2_Name', '') == row['loser_name'])
        ]
        
        if len(matching_rows) == 1:
            matched_ml_data.append(matching_rows.iloc[0])
        elif len(matching_rows) == 0:
            # Try swapped names (Player1 might be loser)
            matching_rows = ml_ready_full[
                (ml_ready_full['tourney_date'] == row['tourney_date']) &
                (ml_ready_full.get('Player1_Name', '') == row['loser_name']) &
                (ml_ready_full.get('Player2_Name', '') == row['winner_name'])
            ]
            if len(matching_rows) == 1:
                # Need to swap the target since Player1 and Player2 are swapped
                ml_row = matching_rows.iloc[0].copy()
                ml_row['Player1_Wins'] = 1 - ml_row['Player1_Wins']  # Flip the target
                matched_ml_data.append(ml_row)
    
    if len(matched_ml_data) > 0:
        matched_ml_df = pd.DataFrame(matched_ml_data)
        print(f"   ✅ Matched {len(matched_ml_df):,} games with ML features")
    else:
        print(f"   ❌ Could not match any games with ML features")
        sys.exit(1)
        
else:
    print(f"   ❌ ML-ready dataset not found at {ml_ready_path}")
    print(f"   You need to run preprocessing first to create ML features")
    sys.exit(1)

# 3. Prepare features for prediction
print("\n3. Preparing features...")
feature_cols = get_feature_columns(matched_ml_df)

# Filter to only numeric columns
numeric_cols = []
for col in feature_cols:
    if col in matched_ml_df.columns:
        if matched_ml_df[col].dtype in ['int64', 'float64', 'bool']:
            numeric_cols.append(col)

feature_cols = numeric_cols
print(f"   Using {len(feature_cols)} features")

X_test = matched_ml_df[feature_cols]
y_test = matched_ml_df['Player1_Wins']

# Fill missing values
X_test = X_test.fillna(X_test.median())
print(f"   Test features shape: {X_test.shape}")
print(f"   Test target distribution: {y_test.mean():.3f}")

# 4. Calculate baseline accuracy
print("\n4. Baseline accuracy on matched games...")
# ATP ranking baseline
jeff_clean = jeff_matched.dropna(subset=['winner_rank', 'loser_rank']).copy()
jeff_clean['winner_rank'] = pd.to_numeric(jeff_clean['winner_rank'], errors='coerce')
jeff_clean['loser_rank'] = pd.to_numeric(jeff_clean['loser_rank'], errors='coerce')
jeff_clean = jeff_clean.dropna(subset=['winner_rank', 'loser_rank'])

atp_correct = (jeff_clean['winner_rank'] < jeff_clean['loser_rank']).sum()
atp_accuracy = atp_correct / len(jeff_clean)
print(f"   ATP Ranking Baseline: {atp_accuracy:.4f} ({atp_accuracy*100:.2f}%)")

# Professional betting odds (from previous analysis)
betting_accuracy = 0.6834  # 68.34% from verification
print(f"   Professional Betting Odds: {betting_accuracy:.4f} ({betting_accuracy*100:.2f}%)")

# 5. Test models
print(f"\n5. Testing trained models...")
results = {
    'ATP_Baseline': atp_accuracy,
    'Betting_Odds': betting_accuracy
}

# Test XGBoost
print(f"\n   Testing XGBoost...")
xgb_model_path = os.path.join('results', 'professional_tennis', 'XGBoost', 'xgboost_model.json')
alt_xgb_path = os.path.join('src', 'Models', 'professional_tennis', 'xgboost_model.pkl')

xgb_model = load_model_if_exists(xgb_model_path, "XGBoost")
if xgb_model is None:
    xgb_model = load_model_if_exists(alt_xgb_path, "XGBoost")

if xgb_model is not None:
    try:
        xgb_pred = xgb_model.predict(X_test)
        xgb_accuracy = accuracy_score(y_test, xgb_pred)
        results['XGBoost'] = xgb_accuracy
        print(f"   ✅ XGBoost Accuracy: {xgb_accuracy:.4f} ({xgb_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"   ❌ XGBoost prediction failed: {e}")

# Test Random Forest
print(f"\n   Testing Random Forest...")
rf_model_path = os.path.join('results', 'professional_tennis', 'Random_Forest', 'random_forest_model.pkl')
alt_rf_path = os.path.join('src', 'Models', 'professional_tennis', 'rf_model.pkl')

rf_model = load_model_if_exists(rf_model_path, "Random Forest")
if rf_model is None:
    rf_model = load_model_if_exists(alt_rf_path, "Random Forest")

if rf_model is not None:
    try:
        rf_pred = rf_model.predict(X_test)
        rf_accuracy = accuracy_score(y_test, rf_pred)
        results['Random_Forest'] = rf_accuracy
        print(f"   ✅ Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
    except Exception as e:
        print(f"   ❌ Random Forest prediction failed: {e}")

# Test Neural Network
print(f"\n   Testing Neural Network...")
nn_model_path = os.path.join('results', 'professional_tennis', 'Neural_Network', 'neural_network_model.pth')
nn_scaler_path = os.path.join('results', 'professional_tennis', 'Neural_Network', 'scaler.pkl')
alt_nn_path = os.path.join('src', 'Models', 'professional_tennis', 'nn_model.pth')
alt_scaler_path = os.path.join('src', 'Models', 'professional_tennis', 'nn_scaler.pkl')

nn_exists = load_model_if_exists(nn_model_path, "Neural Network")
if nn_exists is None:
    nn_exists = load_model_if_exists(alt_nn_path, "Neural Network")

scaler = load_model_if_exists(nn_scaler_path, "Scaler")
if scaler is None:
    scaler = load_model_if_exists(alt_scaler_path, "Scaler")

if nn_exists and scaler is not None:
    try:
        # Load the actual neural network
        input_size = X_test.shape[1]
        nn_model = TennisNet(input_size)
        
        model_path = nn_model_path if os.path.exists(nn_model_path) else alt_nn_path
        nn_model.load_state_dict(torch.load(model_path, map_location='cpu'))
        nn_model.eval()
        
        # Scale features and predict
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

# 6. Final comparison
print(f"\n" + "="*80)
print(f"FINAL APPLES-TO-APPLES COMPARISON")
print(f"="*80)
print(f"Dataset: {len(matched_ml_df):,} identical matches (verified by scores + rankings)")
print(f"")

# Sort results by accuracy
sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

print(f"Performance Rankings:")
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

print(f"")

# Check if any model beats betting odds
models_beat_betting = []
for method, accuracy in results.items():
    if method not in ['ATP_Baseline', 'Betting_Odds'] and accuracy > betting_accuracy:
        improvement = (accuracy - betting_accuracy) * 100
        models_beat_betting.append((method, improvement))

if models_beat_betting:
    print(f"🎉 MODELS THAT BEAT PROFESSIONAL BETTING ODDS:")
    for method, improvement in models_beat_betting:
        print(f"   {method.replace('_', ' ')}: +{improvement:.2f} percentage points better")
else:
    print(f"📊 No models beat the professional betting odds on identical matches")
    best_model = max([x for x in results.items() if x[0] not in ['ATP_Baseline', 'Betting_Odds']], 
                    key=lambda x: x[1])
    if best_model:
        gap = (betting_accuracy - best_model[1]) * 100
        print(f"   Best model ({best_model[0].replace('_', ' ')}): -{gap:.2f} percentage points vs betting odds")

print(f"\nThis is the true apples-to-apples comparison on identical matches! 🎯")