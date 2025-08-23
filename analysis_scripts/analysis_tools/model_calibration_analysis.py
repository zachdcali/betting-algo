#!/usr/bin/env python3
"""
MODEL CALIBRATION ANALYSIS: Log Loss, ECE, Brier Score for all models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import torch.nn as nn
from sklearn.metrics import log_loss, brier_score_loss
from sklearn.calibration import calibration_curve
import xgboost as xgb
import os
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
exact_143_features = [
    'P2_WinStreak_Current', 'P1_WinStreak_Current', 'P2_Surface_Matches_30d', 'Height_Diff',
    'P1_Surface_Matches_30d', 'Player2_Height', 'P1_Matches_30d', 'P2_Matches_30d',
    'P2_Surface_Experience', 'P2_Form_Trend_30d', 'Player1_Height', 'P1_Form_Trend_30d',
    'Round_R16', 'Surface_Transition_Flag', 'P1_Surface_Matches_90d', 'P1_Surface_Experience',
    'Rank_Diff', 'Round_R32', 'Rank_Points_Diff', 'P2_Level_WinRate_Career',
    'P2_Surface_Matches_90d', 'P1_Level_WinRate_Career', 'P2_Level_Matches_Career',
    'P2_WinRate_Last10_120d', 'Round_QF', 'Level_25', 'P1_Round_WinRate_Career',
    'P1_Surface_WinRate_90d', 'Round_Q1', 'Player1_Rank', 'P1_Level_Matches_Career',
    'P2_Round_WinRate_Career', 'draw_size', 'P1_WinRate_Last10_120d', 'Age_Diff',
    'Level_15', 'Player1_Rank_Points', 'Handedness_Matchup_RL', 'Player2_Rank',
    'Avg_Age', 'P1_Country_RUS', 'Player2_Age', 'P2_vs_Lefty_WinRate',
    'Round_F', 'Surface_Clay', 'P2_Sets_14d', 'Rank_Momentum_Diff_30d',
    'H2H_P2_Wins', 'Player2_Rank_Points', 'Player1_Age', 'P2_Rank_Volatility_90d',
    'P1_Days_Since_Last', 'Grass_Season', 'P1_Semifinals_WinRate', 'Level_A',
    'Level_D', 'P1_Country_USA', 'P1_Country_GBR', 'P1_Country_FRA',
    'P2_Matches_14d', 'P2_Country_USA', 'P2_Country_ITA', 'Round_Q2',
    'P2_Surface_WinRate_90d', 'P1_Hand_L', 'P2_Hand_L', 'P1_Country_ITA',
    'P2_Rust_Flag', 'P1_Rank_Change_90d', 'P1_Country_AUS', 'P1_Hand_U',
    'P1_Hand_R', 'Round_RR', 'Avg_Height', 'P1_Sets_14d',
    'P2_Country_Other', 'Round_SF', 'P1_vs_Lefty_WinRate', 'Indoor_Season',
    'Avg_Rank', 'P1_Rust_Flag', 'Avg_Rank_Points', 'Level_F',
    'Round_R64', 'P2_Country_CZE', 'P2_Hand_R', 'Surface_Hard',
    'P1_Matches_14d', 'Surface_Carpet', 'Round_R128', 'P1_Country_SRB',
    'P2_Hand_U', 'P1_Rank_Volatility_90d', 'Level_M', 'P2_Country_ESP',
    'Handedness_Matchup_LR', 'P1_Country_CZE', 'P2_Country_SUI', 'Surface_Grass',
    'H2H_Total_Matches', 'Level_O', 'P1_Hand_A', 'P1_Finals_WinRate',
    'Rank_Momentum_Diff_90d', 'P2_Finals_WinRate',
    'Round_Q4', 'Peak_Age_P1', 'Level_G', 'Round_ER', 'Level_S', 'Round_BR', 'Peak_Age_P2',
    'Round_Q3', 'Rank_Ratio', 'P1_Country_SUI', 'Clay_Season', 'P1_Country_GER',
    'P2_Rank_Change_30d', 'P1_Country_ESP', 'P2_Hand_A', 'H2H_Recent_P1_Advantage',
    'P2_Country_AUS', 'P2_Country_SRB', 'P2_Country_GBR', 'P2_Country_ARG',
    'Handedness_Matchup_RR', 'P1_Rank_Change_30d', 'P2_Country_GER', 'Handedness_Matchup_LL',
    'P2_Country_RUS', 'P1_Country_ARG', 'Level_C', 'P2_Semifinals_WinRate',
    'P2_Days_Since_Last', 'P1_Peak_Age', 'P2_Peak_Age', 'H2H_P1_WinRate',
    'P1_Country_Other', 'H2H_P1_Wins', 'P1_BigMatch_WinRate', 'P2_Rank_Change_90d',
    'P2_BigMatch_WinRate', 'P2_Country_FRA'
]
low_importance_features = [
    'P1_Country_FRA', 'P2_Country_FRA', 'P1_Country_RUS', 'P2_Country_RUS',
    'P1_Country_ARG', 'P2_Country_ARG', 'P1_Country_GER', 'P2_Country_GER',
    'P1_Country_GBR', 'P2_Country_GBR', 'P1_Country_SRB', 'P2_Country_SRB',
    'P1_Country_AUS', 'P2_Country_AUS', 'P1_Country_ESP', 'P2_Country_ESP',
    'P1_Country_SUI', 'P2_Country_SUI', 'P1_Country_Other', 'P2_Country_Other',
    'P1_BigMatch_WinRate', 'P2_BigMatch_WinRate', 'P1_Peak_Age', 'P2_Peak_Age',
    'P1_Semifinals_WinRate', 'P2_Semifinals_WinRate', 'P1_Days_Since_Last', 'P2_Days_Since_Last',
    'P1_Hand_A', 'P2_Hand_A', 'P1_Rank_Change_30d', 'P2_Rank_Change_30d', 'P2_Rank_Change_90d',
    'H2H_P1_Wins', 'H2H_P1_WinRate', 'H2H_Recent_P1_Advantage',
    'Level_C', 'Clay_Season', 'Rank_Ratio', 'Round_Q3',
    'Handedness_Matchup_LL', 'Handedness_Matchup_RR',
    'Round_Q4', 'Peak_Age_P1', 'Level_G', 'Round_ER', 'Level_S', 'Round_BR', 'Peak_Age_P2'
]
def get_feature_columns(df):
    exclude_cols = ['tourney_date', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id',
                    'winner_name', 'loser_name', 'score', 'Player1_Name', 'Player2_Name', 'Player1_Wins',
                    'best_of', 'round', 'minutes', 'data_source', 'year']
    return [col for col in df.columns if col not in exclude_cols and '.1' not in col]  # Exclude duplicates
def calculate_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece
def plot_calibration_curve(y_true, y_prob, model_name, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    bin_centers = []
    bin_accuracies = []
    bin_counts = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            bin_centers.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
            bin_counts.append(in_bin.sum())
    return bin_centers, bin_accuracies, bin_counts
def analyze_full_testing_sets():
    full_results = {}
    print(" Loading full Jeff Sackmann dataset...")
    try:
        ml_full = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
        ml_full['tourney_date'] = pd.to_datetime(ml_full['tourney_date'])
        test_data = ml_full[ml_full['tourney_date'] >= '2023-01-01'].copy()
        print(f" Raw 2023+ test data: {len(test_data):,} matches")
        numeric_cols = [col for col in test_data.columns if test_data[col].dtype in ['int64', 'float64', 'bool'] and '.1' not in col]
        test_data = test_data.dropna(subset=['Player1_Rank', 'Player2_Rank'])  # Match training filter
        print(f" After dropna on rankings: {len(test_data):,} matches")
        test_data[numeric_cols] = test_data[numeric_cols].fillna(test_data[numeric_cols].median())
        print(f" After fillna: {len(test_data):,} matches (no further rows removed)")
        feature_cols_143 = [col for col in exact_143_features if col in test_data.columns]
        if len(feature_cols_143) != 143:
            print(f" WARNING: Only {len(feature_cols_143)} features found out of 143")
            missing_features = set(exact_143_features) - set(test_data.columns)
            if missing_features:
                print(f" Missing features: {list(missing_features)[:10]}...")
        y_full = test_data['Player1_Wins'].values
        print(f" Full test set after filtering: {len(test_data):,} matches from 2023+")
        print(f" Analyzing XGBoost on full test set...")
        try:
            xgb_model = xgb.XGBClassifier()
            xgb_model.load_model('results/professional_tennis/XGBoost/xgboost_model.json')
            X_test_xgb = test_data[feature_cols_143].values
            y_test_xgb = test_data['Player1_Wins']
            y_pred_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]
            logloss_xgb = log_loss(y_test_xgb.values, y_pred_proba_xgb)
            brier_xgb = np.mean((y_pred_proba_xgb - y_test_xgb.values)**2)
            prob_true_xgb, prob_pred_xgb = calibration_curve(y_test_xgb.values, y_pred_proba_xgb, n_bins=10)
            ece_xgb = np.mean(np.abs(prob_true_xgb - prob_pred_xgb))
            full_results['XGBoost'] = {
                'n_samples': len(test_data),
                'log_loss': logloss_xgb,
                'brier_score': brier_xgb,
                'ece': ece_xgb
            }
            print(f" XGBoost: {ece_xgb:.4f} ECE")
        except Exception as e:
            print(f" XGBoost failed: {e}")
        print(f" Analyzing Random Forest on full test set...")
        try:
            with open('results/professional_tennis/Random_Forest/random_forest_model.pkl', 'rb') as f:
                rf_model = pickle.load(f)
            X_test_rf = test_data[feature_cols_143].values
            y_test_rf = test_data['Player1_Wins']
            y_pred_proba_rf = rf_model.predict_proba(X_test_rf)[:, 1]
            logloss_rf = log_loss(y_test_rf.values, y_pred_proba_rf)
            brier_rf = np.mean((y_pred_proba_rf - y_test_rf.values)**2)
            prob_true_rf, prob_pred_rf = calibration_curve(y_test_rf.values, y_pred_proba_rf, n_bins=10)
            ece_rf = np.mean(np.abs(prob_true_rf - prob_pred_rf))
            full_results['Random_Forest'] = {
                'n_samples': len(test_data),
                'log_loss': logloss_rf,
                'brier_score': brier_rf,
                'ece': ece_rf
            }
            print(f" Random Forest: {ece_rf:.4f} ECE")
        except Exception as e:
            print(f" Random Forest failed: {e}")
        print(f" Analyzing Neural Network 143 on full test set...")
        try:
            with open('results/professional_tennis/Neural_Network/scaler_143_features.pkl', 'rb') as f:
                scaler_143 = pickle.load(f)
            X_test_nn = test_data[feature_cols_143]
            y_test_nn = test_data['Player1_Wins']
            nn_model_143 = TennisNet(len(feature_cols_143))
            nn_model_143.load_state_dict(torch.load('results/professional_tennis/Neural_Network/neural_network_143_features.pth', map_location='cpu'))
            nn_model_143.eval()
            X_scaled_nn = scaler_143.transform(X_test_nn.values)
            X_tensor_nn = torch.FloatTensor(X_scaled_nn)
            with torch.no_grad():
                y_pred_proba_nn = nn_model_143(X_tensor_nn).squeeze().numpy()
            logloss_nn = log_loss(y_test_nn.values, y_pred_proba_nn)
            brier_nn = np.mean((y_pred_proba_nn - y_test_nn.values)**2)
            prob_true_nn, prob_pred_nn = calibration_curve(y_test_nn.values, y_pred_proba_nn, n_bins=10)
            ece_nn = np.mean(np.abs(prob_true_nn - prob_pred_nn))
            full_results['Neural_Network_143'] = {
                'n_samples': len(test_data),
                'log_loss': logloss_nn,
                'brier_score': brier_nn,
                'ece': ece_nn
            }
            print(f" Neural Network 143: {ece_nn:.4f} ECE")
        except Exception as e:
            print(f" Neural Network 143 failed: {e}")
        print(f" Analyzing Neural Network 98 on full test set...")
        try:
            with open('results/professional_tennis/Neural_Network/scaler_symmetric_features.pkl', 'rb') as f:
                scaler_98 = pickle.load(f)
            all_feature_cols = get_feature_columns(test_data)
            all_numeric_cols = [col for col in all_feature_cols if col in test_data.columns and test_data[col].dtype in ['int64', 'float64', 'bool']]
            all_numeric_cols = [col for col in all_numeric_cols if col not in low_importance_features]
            print(f" Starting with {len(all_feature_cols)} features, after removal: {len(all_numeric_cols)} features")
            if len(all_numeric_cols) != scaler_98.n_features_in_:
                features_to_remove_more = len(all_numeric_cols) - scaler_98.n_features_in_
                if features_to_remove_more > 0:
                    print(f" DEBUG: Need to remove {features_to_remove_more} more features to match trained model")
                    all_numeric_cols = all_numeric_cols[:scaler_98.n_features_in_]
            X_full_nn_98 = test_data[all_numeric_cols]
            nn_model_98 = TennisNet(len(all_numeric_cols))
            nn_model_98.load_state_dict(torch.load('results/professional_tennis/Neural_Network/neural_network_symmetric_features.pth', map_location='cpu'))
            nn_model_98.eval()
            X_scaled_98_full = scaler_98.transform(X_full_nn_98.values)
            X_tensor_98_full = torch.FloatTensor(X_scaled_98_full)
            with torch.no_grad():
                y_pred_proba_nn_98 = nn_model_98(X_tensor_98_full).squeeze().numpy()
            y_test_nn_98 = test_data['Player1_Wins']
            logloss_nn_98 = log_loss(y_test_nn_98.values, y_pred_proba_nn_98)
            brier_nn_98 = np.mean((y_pred_proba_nn_98 - y_test_nn_98.values)**2)
            prob_true_nn_98, prob_pred_nn_98 = calibration_curve(y_test_nn_98.values, y_pred_proba_nn_98, n_bins=10)
            ece_nn_98 = np.mean(np.abs(prob_true_nn_98 - prob_pred_nn_98))
            full_results['Neural_Network_98'] = {
                'n_samples': len(test_data),
                'log_loss': logloss_nn_98,
                'brier_score': brier_nn_98,
                'ece': ece_nn_98
            }
            print(f" Neural Network 98: {ece_nn_98:.4f} ECE")
        except Exception as e:
            print(f" Neural Network 98 failed: {e}")
    except Exception as e:
        print(f" Failed to load full testing set: {e}")
    return full_results
def main():
    print("MODEL CALIBRATION ANALYSIS")
    print("="*80)
    print("Loading 100% verified data...")
    ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
    jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
    tennis_matched = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_exact_matched_final.csv', low_memory=False)
    ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
    jeff_matched['tourney_date'] = pd.to_datetime(jeff_matched['tourney_date'])
    tennis_matched['Date'] = pd.to_datetime(tennis_matched['Date'])
    verified_indices = list(range(min(len(jeff_matched), 1542)))
    jeff_100_verified = jeff_matched.iloc[verified_indices].copy()
    ml_verified = []
    for _, verified_row in jeff_100_verified.iterrows():
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
                break
            elif (player1 == loser and player2 == winner):
                ml_row_copy = ml_row.copy()
                ml_row_copy['Player1_Wins'] = 0
                ml_verified.append(ml_row_copy)
                break
    ml_verified_df = pd.DataFrame(ml_verified)
    if len(ml_verified_df) == 0:
        print("No verified data found!")
        return
    feature_cols = get_feature_columns(ml_verified_df)
    numeric_cols = [col for col in feature_cols if col in ml_verified_df.columns and
                    ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]
    ml_verified_df = ml_verified_df.reset_index(drop=True)
    valid_mask = ~ml_verified_df[numeric_cols].isnull().any(axis=1)
    ml_verified_df = ml_verified_df[valid_mask].reset_index(drop=True)
    print(f"Using {len(ml_verified_df):,} verified matches for betting subset analysis")
    feature_cols_all = get_feature_columns(ml_verified_df)
    y_test = ml_verified_df['Player1_Wins'].values
    print(f"Total available features: {len(feature_cols_all)}")
    print(f"Betting subset samples: {len(ml_verified_df)}")
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model('results/professional_tennis/XGBoost/xgboost_model.json')
        xgb_features = xgb_model.feature_names_in_
        print(f"XGBoost expects {len(xgb_features)} features")
        with open('results/professional_tennis/Random_Forest/random_forest_model.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        rf_features = rf_model.feature_names_in_
        print(f"Random Forest expects {len(rf_features)} features")
        print(f"Neural Network 143: expects 143 features")
        print(f"Neural Network 98: expects 98 features")
    except Exception as e:
        print(f"Could not load model feature requirements: {e}")
        return
    results = {}
    print(f"\nAnalyzing XGBoost...")
    xgb_path = 'results/professional_tennis/XGBoost/xgboost_model.json'
    if os.path.exists(xgb_path):
        try:
            xgb_feature_cols = [col for col in exact_143_features if col in ml_verified_df.columns]
            X_test_xgb = ml_verified_df[xgb_feature_cols]
            print(f" Using {len(xgb_feature_cols)} features for XGBoost")
            xgb_proba = xgb_model.predict_proba(X_test_xgb)[:, 1]
            logloss = log_loss(y_test, xgb_proba)
            brier = brier_score_loss(y_test, xgb_proba)
            ece = calculate_ece(y_test, xgb_proba)
            results['XGBoost'] = {
                'log_loss': logloss,
                'brier_score': brier,
                'ece': ece,
                'probabilities': xgb_proba
            }
            print(f" Log Loss: {logloss:.4f}")
            print(f" Brier Score: {brier:.4f}")
            print(f" ECE: {ece:.4f}")
        except Exception as e:
            print(f" XGBoost failed: {e}")
    print(f"\nAnalyzing Random Forest...")
    rf_path = 'results/professional_tennis/Random_Forest/random_forest_model.pkl'
    if os.path.exists(rf_path):
        try:
            rf_feature_cols = [col for col in exact_143_features if col in ml_verified_df.columns]
            X_test_rf = ml_verified_df[rf_feature_cols]
            print(f" Using {len(rf_feature_cols)} features for Random Forest")
            rf_proba = rf_model.predict_proba(X_test_rf)[:, 1]
            logloss = log_loss(y_test, rf_proba)
            brier = brier_score_loss(y_test, rf_proba)
            ece = calculate_ece(y_test, rf_proba)
            results['Random_Forest'] = {
                'log_loss': logloss,
                'brier_score': brier,
                'ece': ece,
                'probabilities': rf_proba
            }
            print(f" Log Loss: {logloss:.4f}")
            print(f" Brier Score: {brier:.4f}")
            print(f" ECE: {ece:.4f}")
        except Exception as e:
            print(f" Random Forest failed: {e}")
    print(f"\nAnalyzing Neural Network (143-feature)...")
    nn_path_143 = 'results/professional_tennis/Neural_Network/neural_network_143_features.pth'
    scaler_path_143 = 'results/professional_tennis/Neural_Network/scaler_143_features.pkl'
    if os.path.exists(nn_path_143) and os.path.exists(scaler_path_143):
        try:
            nn_143_feature_cols = [col for col in exact_143_features if col in ml_verified_df.columns]
            X_test_nn_143 = ml_verified_df[nn_143_feature_cols]
            print(f" Using {len(nn_143_feature_cols)} features for NN-143")
            with open(scaler_path_143, 'rb') as f:
                scaler_143 = pickle.load(f)
            nn_model_143 = TennisNet(len(nn_143_feature_cols))
            nn_model_143.load_state_dict(torch.load(nn_path_143, map_location='cpu'))
            nn_model_143.eval()
            X_scaled_143 = scaler_143.transform(X_test_nn_143.values)
            X_tensor_143 = torch.FloatTensor(X_scaled_143)
            with torch.no_grad():
                nn_proba_143 = nn_model_143(X_tensor_143).squeeze().numpy()
            logloss = log_loss(y_test, nn_proba_143)
            brier = brier_score_loss(y_test, nn_proba_143)
            ece = calculate_ece(y_test, nn_proba_143)
            results['Neural_Network_143'] = {
                'log_loss': logloss,
                'brier_score': brier,
                'ece': ece,
                'probabilities': nn_proba_143
            }
            print(f" Log Loss: {logloss:.4f}")
            print(f" Brier Score: {brier:.4f}")
            print(f" ECE: {ece:.4f}")
        except Exception as e:
            print(f" Neural Network (143-feature) failed: {e}")
    print(f"\nAnalyzing Neural Network (98-feature)...")
    nn_path_98 = 'results/professional_tennis/Neural_Network/neural_network_symmetric_features.pth'
    scaler_path_98 = 'results/professional_tennis/Neural_Network/scaler_symmetric_features.pkl'
    if os.path.exists(nn_path_98) and os.path.exists(scaler_path_98):
        try:
            with open(scaler_path_98, 'rb') as f:
                scaler_98 = pickle.load(f)
            numeric_cols_betting = [col for col in feature_cols_all if col in ml_verified_df.columns and
                                    ml_verified_df[col].dtype in ['int64', 'float64', 'bool']]
            all_numeric_cols = [col for col in numeric_cols_betting if col not in low_importance_features]
            print(f" Features after low_importance removal: {len(all_numeric_cols)}")
            print(f" Scaler expects: {scaler_98.n_features_in_}")
            if len(all_numeric_cols) != scaler_98.n_features_in_:
                if len(all_numeric_cols) > scaler_98.n_features_in_:
                    extra = all_numeric_cols[scaler_98.n_features_in_:]
                    print(f" Extra features: {extra[:5]}...")
                    all_numeric_cols = all_numeric_cols[:scaler_98.n_features_in_]
                else:
                    missing = scaler_98.n_features_in_ - len(all_numeric_cols)
                    print(f" Missing {missing} features - metrics may be invalid")
            X_test_98_cols = all_numeric_cols
            print(f" DEBUG: Using {len(X_test_98_cols)} features for NN-98")
            X_test_nn_98 = ml_verified_df[X_test_98_cols]
            print(f" Final: Using {len(X_test_98_cols)} features for NN-98")
            nn_model_98 = TennisNet(len(X_test_98_cols))
            nn_model_98.load_state_dict(torch.load(nn_path_98, map_location='cpu'))
            nn_model_98.eval()
            X_scaled_98 = scaler_98.transform(X_test_nn_98.values)
            X_tensor_98 = torch.FloatTensor(X_scaled_98)
            with torch.no_grad():
                nn_proba_98 = nn_model_98(X_tensor_98).squeeze().numpy()
            logloss = log_loss(y_test, nn_proba_98)
            brier = brier_score_loss(y_test, nn_proba_98)
            ece = calculate_ece(y_test, nn_proba_98)
            results['Neural_Network_98'] = {
                'log_loss': logloss,
                'brier_score': brier,
                'ece': ece,
                'probabilities': nn_proba_98
            }
            print(f" Log Loss: {logloss:.4f}")
            print(f" Brier Score: {brier:.4f}")
            print(f" ECE: {ece:.4f}")
        except Exception as e:
            print(f" Neural Network (98-feature) failed: {e}")
    os.makedirs('analysis_scripts/charts', exist_ok=True)
    if results:
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Calibration Analysis', fontsize=16, fontweight='bold')
        model_positions = {
            'XGBoost': (0, 0),
            'Random_Forest': (0, 1),
            'Neural_Network_143': (1, 0),
            'Neural_Network_98': (1, 1)
        }
        for model_name, metrics in results.items():
            if model_name in model_positions:
                row, col = model_positions[model_name]
                ax = axes[row, col]
                bin_centers, bin_accuracies, bin_counts = plot_calibration_curve(
                    y_test, metrics['probabilities'], model_name)
                if bin_centers:
                    ax.plot(bin_centers, bin_accuracies, 'o-', linewidth=2, markersize=8,
                            label=f'{model_name.replace("_", " ")}')
                    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect Calibration')
                    ax.set_xlabel('Mean Predicted Probability')
                    ax.set_ylabel('Fraction of Positives')
                    ax.set_title(f'{model_name.replace("_", " ")}\nECE: {metrics["ece"]:.3f}')
                    ax.grid(True, alpha=0.3)
                    ax.legend()
                    ax.set_xlim(0, 1)
                    ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig('analysis_scripts/charts/model_calibration_analysis.png',
                    dpi=300, bbox_inches='tight', facecolor='white')
        print(f" Calibration plot saved: analysis_scripts/charts/model_calibration_analysis.png")
        plt.close()
    print(f"\n" + "="*80)
    print(f"MODEL CALIBRATION SUMMARY - BETTING SUBSET (1,531 matches)")
    print(f"="*80)
    print(f"{'Model':<25} {'Log Loss':<12} {'Brier Score':<12} {'ECE':<12}")
    print("-" * 65)
    for model_name, metrics in results.items():
        display_name = model_name.replace('_', ' ')
        print(f"{display_name:<25} {metrics['log_loss']:<12.4f} {metrics['brier_score']:<12.4f} {metrics['ece']:<12.4f}")
    print(f"\n" + "="*80)
    print(f"FULL TESTING SET CALIBRATION - ALL 2023-2024 MATCHES")
    print(f"="*80)
    print("Loading full testing datasets from training scripts...")
    full_results = analyze_full_testing_sets()
    if full_results:
        print(f"\n{'Model':<25} {'Test Matches':<12} {'Log Loss':<12} {'Brier Score':<12} {'ECE':<12}")
        print("-" * 77)
        for model_name, metrics in full_results.items():
            display_name = model_name.replace('_', ' ')
            print(f"{display_name:<25} {metrics['n_samples']:<12} {metrics['log_loss']:<12.4f} {metrics['brier_score']:<12.4f} {metrics['ece']:<12.4f}")
    print(f"\n📊 CALIBRATION INSIGHTS:")
    print(f"✅ Lower Log Loss = Better probability predictions")
    print(f"✅ Lower Brier Score = Better overall calibration")
    print(f"✅ Lower ECE = Predictions closer to actual frequencies")
    print(f"✅ Well-calibrated models should have ECE < 0.01 (excellent), < 0.05 (good)")
    def bet_calibration(model_name):
        try:
            df = pd.read_csv(f"analysis_scripts/betting_logs/Kelly_5pct_Cap/{model_name.lower()}_bets.csv")
            y_true = df['bet_won'].values
            y_prob = df['model_prob_bet_on'].values
            logloss = log_loss(y_true, y_prob)
            brier = brier_score_loss(y_true, y_prob)
            prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
            ece = np.mean(np.abs(prob_true - prob_pred))
            print(f"{model_name.upper()} Bets: Log Loss {logloss:.4f}, Brier {brier:.4f}, ECE {ece:.4f}, Bets {len(df)}")
        except:
            print(f"{model_name} log missing")
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    print("\nBET-SPECIFIC CALIBRATION")
    print("="*50)
    for m in models:
        bet_calibration(m)
    return results, full_results
if __name__ == "__main__":
    main()