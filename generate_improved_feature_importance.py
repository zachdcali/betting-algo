#!/usr/bin/env python3
"""
Generate feature importance for the improved Neural Network model
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
import os
import sys

# Add the current directory to path to import preprocessing function
sys.path.append('src/Models/professional_tennis')
from preprocess import get_feature_columns

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

class SklearnWrapper(BaseEstimator):
    def __init__(self, model, device, scaler):
        self.model = model
        self.device = device
        self.scaler = scaler
    
    def predict_proba(self, X):
        self.model.eval()
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().cpu().numpy()
        return np.vstack((1 - probs, probs)).T
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
    
    def score(self, X, y):
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def main():
    print("=" * 80)
    print("GENERATING FEATURE IMPORTANCE FOR IMPROVED NEURAL NETWORK")
    print("=" * 80)
    
    device = torch.device("cpu")
    
    # Load the improved model and scaler
    print("\n1. Loading improved Neural Network model and scaler...")
    model_path = 'results/professional_tennis/Neural_Network/neural_network_model_improved_features_early_stopping.pth'
    scaler_path = 'results/professional_tennis/Neural_Network/scaler_improved_features_early_stopping.pkl'
    
    # Load scaler
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"   Scaler loaded: {scaler.n_features_in_} features")
    
    # Load model
    model = TennisNet(scaler.n_features_in_)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    print(f"   Model loaded: {scaler.n_features_in_} input features")
    
    # Load test data (same as training script)
    print("\n2. Loading test data...")
    ml_ready_path = "data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv"
    ml_df = pd.read_csv(ml_ready_path, low_memory=False)
    
    ml_df['tourney_date'] = pd.to_datetime(ml_df['tourney_date'])
    ml_df['year'] = ml_df['tourney_date'].dt.year
    ml_df = ml_df[ml_df['year'] >= 1990].copy()
    
    # Get test set (2023-2024)
    test_df = ml_df[ml_df['tourney_date'] >= '2023-01-01'].copy()
    test_df = test_df.dropna(subset=['Player1_Rank', 'Player2_Rank'])
    
    print(f"   Test data: {len(test_df)} matches")
    
    # Get feature columns (same logic as training)
    feature_cols = get_feature_columns(ml_df)
    numeric_cols = []
    for col in feature_cols:
        if col in ml_df.columns:
            if ml_df[col].dtype in ['int64', 'float64', 'bool']:
                numeric_cols.append(col)
            elif ml_df[col].dtype == 'object':
                try:
                    pd.to_numeric(ml_df[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    continue
    
    feature_cols = numeric_cols
    
    # Remove same low-importance features as training
    low_importance_features = [
        'P2_Country_FRA', 'P2_Rank_Change_90d', 'P2_BigMatch_WinRate', 'P1_BigMatch_WinRate', 
        'H2H_P1_Wins', 'P1_Country_Other', 'H2H_P1_WinRate', 'P2_Peak_Age', 'P1_Peak_Age', 
        'P2_Days_Since_Last', 'P2_Semifinals_WinRate', 'Level_C', 'P1_Country_ARG', 
        'P2_Country_RUS', 'Handedness_Matchup_LL', 'P2_Country_GER', 'Handedness_Matchup_RR', 
        'P1_Rank_Change_30d', 'P2_Country_ARG', 'P2_Country_GBR', 'P2_Country_SRB', 
        'P2_Country_AUS', 'H2H_Recent_P1_Advantage', 'P2_Hand_A', 'P1_Country_ESP', 
        'P2_Rank_Change_30d', 'P1_Country_GER', 'Clay_Season', 'P1_Country_SUI', 
        'Rank_Ratio', 'Round_Q3',
        'Round_Q4', 'Peak_Age_P1', 'Level_G', 'Round_ER', 'Level_S', 'Round_BR', 'Peak_Age_P2'
    ]
    
    feature_cols = [col for col in feature_cols if col not in low_importance_features]
    print(f"   Final features: {len(feature_cols)} (after removing low-importance)")
    
    # Prepare test data
    X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
    y_test = test_df['Player1_Wins']
    
    print(f"   Test features shape: {X_test.shape}")
    
    # Create sklearn wrapper
    sklearn_model = SklearnWrapper(model, device, scaler)
    
    # Calculate permutation importance (use subset for speed)
    print("\n3. Calculating permutation importance...")
    subsample_size = min(5000, len(X_test))
    idx = np.random.choice(len(X_test), subsample_size, replace=False)
    X_sub = X_test.iloc[idx]
    y_sub = y_test.iloc[idx]
    
    print(f"   Using {subsample_size} samples for permutation importance")
    
    r = permutation_importance(sklearn_model, X_sub.values, y_sub.values, n_repeats=5, random_state=42, n_jobs=-1)
    
    # Create feature importance dataframe
    perm_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': r.importances_mean
    }).sort_values('importance', ascending=False)
    
    # Save results
    output_path = 'results/professional_tennis/Neural_Network/feature_importance.csv'
    perm_importance.to_csv(output_path, index=False)
    
    print(f"\n4. Results saved to: {output_path}")
    print(f"   Total features: {len(perm_importance)}")
    
    # Show summary
    positive_features = perm_importance[perm_importance['importance'] > 0]
    negative_features = perm_importance[perm_importance['importance'] < 0]
    zero_features = perm_importance[perm_importance['importance'] == 0]
    
    print(f"\n=== IMPROVED NEURAL NETWORK FEATURE IMPORTANCE SUMMARY ===")
    print(f"  Positive importance: {len(positive_features)} features")
    print(f"  Negative importance: {len(negative_features)} features")
    print(f"  Zero importance: {len(zero_features)} features")
    print(f"  Range: {perm_importance['importance'].min():.6f} to {perm_importance['importance'].max():.6f}")
    
    print(f"\nTop 10 most important features:")
    for i, (_, row) in enumerate(perm_importance.head(10).iterrows()):
        print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    if len(negative_features) > 0:
        print(f"\nMost negative features:")
        for i, (_, row) in enumerate(negative_features.head(5).iterrows()):
            print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.6f}")
    
    print(f"\n✅ Feature importance calculated for improved model!")
    print(f"   Now run: python show_all_feature_importance.py")

if __name__ == "__main__":
    main()