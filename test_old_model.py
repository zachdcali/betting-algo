#!/usr/bin/env python3
"""
Test script to verify if the old 143-feature Neural Network model is corrupted
Tests it using the same approach as the new model to isolate the issue
"""
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import pickle
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from torch.utils.data import TensorDataset, DataLoader
import os
import sys

# Add path for preprocessing functions
sys.path.append('src/Models/professional_tennis')
from preprocess import get_feature_columns

class TennisNet(nn.Module):
    """Neural Network architecture"""
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

def test_model_direct():
    """Test the old model by loading it exactly as the new model would be loaded"""
    print("=" * 80)
    print("TESTING OLD 143-FEATURE MODEL DIRECTLY")
    print("=" * 80)
    
    device = torch.device("cpu")
    
    # Load the old model and scaler
    model_path = 'results/professional_tennis/Neural_Network/neural_network_model.pth'
    scaler_path = 'results/professional_tennis/Neural_Network/scaler.pkl'
    
    print("\\n1. Loading old model files...")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    input_size = scaler.n_features_in_
    print(f"   Scaler expects: {input_size} features")
    
    model = TennisNet(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"   Model loaded with {input_size} input features")
    
    # Load data
    print("\\n2. Loading test data...")
    ml_ready_path = "data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv"
    ml_df = pd.read_csv(ml_ready_path, low_memory=False)
    ml_df['tourney_date'] = pd.to_datetime(ml_df['tourney_date'])
    ml_df['year'] = ml_df['tourney_date'].dt.year
    ml_df = ml_df[ml_df['year'] >= 1990].copy()
    ml_df = ml_df.dropna(subset=['Player1_Rank', 'Player2_Rank'])
    
    # Get test set (2023-2024)  
    test_df = ml_df[ml_df['tourney_date'] >= '2023-01-01'].copy()
    print(f"   Test data: {len(test_df):,} matches")
    
    # Test different feature approaches
    approaches = [
        ("Current preprocessing", "current"),
        ("Take first N features", "first_n"),
        ("Try saved feature order", "saved_order")
    ]
    
    for approach_name, approach_type in approaches:
        print(f"\\n3. Testing with {approach_name}...")
        
        try:
            if approach_type == "current":
                # Current preprocessing approach
                feature_cols = get_feature_columns(ml_df)
                numeric_cols = []
                for col in feature_cols:
                    if col in ml_df.columns:
                        if ml_df[col].dtype in ['int64', 'float64', 'bool']:
                            numeric_cols.append(col)
                feature_cols = numeric_cols[:input_size]
                
            elif approach_type == "first_n":
                # Just take first N columns that are numeric
                all_cols = [col for col in ml_df.columns if ml_df[col].dtype in ['int64', 'float64', 'bool']]
                exclude_cols = ['tourney_date', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id',
                               'winner_name', 'loser_name', 'score', 'Player1_Name', 'Player2_Name', 'Player1_Wins',
                               'best_of', 'round', 'minutes', 'data_source', 'year']
                feature_cols = [col for col in all_cols if col not in exclude_cols][:input_size]
                
            elif approach_type == "saved_order":
                # Try to reconstruct the exact order from the old feature importance
                old_importance_path = 'results/professional_tennis/Neural_Network/feature_importance.csv'
                if os.path.exists(old_importance_path):
                    old_features_df = pd.read_csv(old_importance_path)
                    if len(old_features_df) >= input_size:
                        feature_cols = old_features_df['feature'].tolist()[:input_size]
                    else:
                        print(f"   Not enough features in importance file, skipping...")
                        continue
                else:
                    print(f"   Feature importance file not found, skipping...")
                    continue
            
            # Check if all features exist
            missing_features = [col for col in feature_cols if col not in test_df.columns]
            if missing_features:
                print(f"   Missing features: {len(missing_features)} - {missing_features[:5]}...")
                continue
                
            print(f"   Using {len(feature_cols)} features")
            
            # Prepare test data  
            X_test = test_df[feature_cols].fillna(test_df[feature_cols].median())
            y_test = test_df['Player1_Wins']
            
            print(f"   Feature shape: {X_test.shape}")
            print(f"   Features: {feature_cols[:5]}...")
            
            # Scale and predict
            X_test_scaled = scaler.transform(X_test.values)
            test_dataset = TensorDataset(torch.FloatTensor(X_test_scaled), torch.FloatTensor(y_test.values))
            test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)
            
            y_pred_proba = []
            with torch.no_grad():
                for batch_x, _ in test_loader:
                    outputs = model(batch_x).squeeze().cpu().numpy()
                    y_pred_proba.extend(outputs)
            
            y_pred_proba = np.array(y_pred_proba)
            y_pred = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test.values, y_pred)
            auc = roc_auc_score(y_test.values, y_pred_proba)
            logloss = log_loss(y_test.values, y_pred_proba)
            
            print(f"   Results:")
            print(f"     Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
            print(f"     AUC-ROC: {auc:.4f}")
            print(f"     Log Loss: {logloss:.4f}")
            
            if accuracy > 0.65:
                print(f"   ✅ This approach works! Model is not corrupted.")
                print(f"   Problem is likely feature mismatch in calibration script.")
                return feature_cols
            else:
                print(f"   ❌ Still low accuracy with this approach.")
                
        except Exception as e:
            print(f"   Error with {approach_name}: {str(e)}")
            continue
    
    print(f"\\n   🚨 All approaches failed - model may be corrupted or incompatible")
    return None

if __name__ == "__main__":
    working_features = test_model_direct()
    if working_features:
        print(f"\\n✅ SOLUTION FOUND:")
        print(f"Working features: {working_features[:10]}...")
        print(f"Total features: {len(working_features)}")
    else:
        print(f"\\n❌ MODEL APPEARS TO BE CORRUPTED OR INCOMPATIBLE")
        print(f"Recommendation: Use the new 109-feature model instead")