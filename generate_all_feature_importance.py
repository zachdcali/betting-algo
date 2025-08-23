#!/usr/bin/env python3
"""
Generate comprehensive feature importance plots for all three models
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import torch
import torch.nn as nn
from sklearn.inspection import permutation_importance
from sklearn.base import BaseEstimator
import xgboost as xgb
import os

# Set style for better plots
plt.style.use('default')
sns.set_palette("husl")

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
    def __init__(self, model, device, batch_size=1024):
        self.model = model
        self.device = device
        self.batch_size = batch_size
    
    def predict_proba(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            probs = self.model(X_tensor).squeeze().cpu().numpy()
        return np.vstack((1 - probs, probs)).T
    
    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs > 0.5).astype(int)
    
    def score(self, X, y):
        from sklearn.metrics import accuracy_score
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

def load_test_data():
    """Load the test data for feature importance analysis"""
    # Load the leak-free dataset
    ml_ready_path = "data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv"
    ml_df = pd.read_csv(ml_ready_path, low_memory=False)
    
    # Convert date and filter
    ml_df['tourney_date'] = pd.to_datetime(ml_df['tourney_date'])
    ml_df['year'] = ml_df['tourney_date'].dt.year
    ml_df = ml_df[ml_df['year'] >= 1990].copy()
    
    # Get test set (2023-2024)
    test_df = ml_df[ml_df['tourney_date'] >= '2023-01-01'].copy()
    
    # Filter to matches with required features
    test_df = test_df.dropna(subset=['Player1_Rank', 'Player2_Rank'])
    
    return test_df

def get_feature_columns(df):
    """Get feature columns for training"""
    exclude_cols = ['tourney_date', 'tourney_name', 'tourney_id', 'match_num', 'winner_id', 'loser_id',
                   'winner_name', 'loser_name', 'score', 'Player1_Name', 'Player2_Name', 'Player1_Wins',
                   'best_of', 'round', 'minutes', 'data_source', 'year']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Filter to numeric columns only
    numeric_cols = []
    for col in feature_cols:
        if col in df.columns:
            if df[col].dtype in ['int64', 'float64', 'bool']:
                numeric_cols.append(col)
            elif df[col].dtype == 'object':
                try:
                    pd.to_numeric(df[col], errors='raise')
                    numeric_cols.append(col)
                except:
                    continue
    
    # Exclude binary advantage features
    exclude_advantages = ['Player1_Rank_Advantage', 'Player1_Height_Advantage', 'Player1_Age_Advantage', 'Player1_Points_Advantage']
    numeric_cols = [col for col in numeric_cols if col not in exclude_advantages]
    
    return numeric_cols

def create_comprehensive_comparison_plot():
    """Create comprehensive plots showing ALL feature importances for each model"""
    
    print("Loading feature importance data...")
    
    # Load all feature importance data
    xgb_importance = pd.read_csv('results/professional_tennis/XGBoost/feature_importance.csv')
    rf_importance = pd.read_csv('results/professional_tennis/Random_Forest/feature_importance.csv')
    nn_importance = pd.read_csv('results/professional_tennis/Neural_Network/feature_importance.csv')
    
    print(f"XGBoost features: {len(xgb_importance)}")
    print(f"Random Forest features: {len(rf_importance)}")
    print(f"Neural Network features: {len(nn_importance)}")
    
    # Create individual plots for each model showing ALL features
    fig_height = max(len(xgb_importance) * 0.3, 20)  # Dynamic height based on number of features
    
    # 1. XGBoost - ALL Features
    plt.figure(figsize=(12, fig_height))
    colors = ['darkred' if x < 0 else 'lightcoral' for x in xgb_importance['importance']]
    bars = plt.barh(range(len(xgb_importance)), xgb_importance['importance'], color=colors)
    plt.yticks(range(len(xgb_importance)), xgb_importance['feature'], fontsize=8)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'XGBoost Feature Importance - ALL {len(xgb_importance)} Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels for significant features
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if abs(width) > 0.01:  # Only label significant features
            plt.text(width + (0.005 if width > 0 else -0.005), bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=7)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/xgboost_all_features_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/xgboost_all_features_importance.pdf', bbox_inches='tight')
    print(f"XGBoost ALL features plot saved (showing {len(xgb_importance)} features)")
    plt.close()
    
    # 2. Random Forest - ALL Features  
    plt.figure(figsize=(12, fig_height))
    colors = ['darkgreen' if x < 0 else 'lightgreen' for x in rf_importance['importance']]
    bars = plt.barh(range(len(rf_importance)), rf_importance['importance'], color=colors)
    plt.yticks(range(len(rf_importance)), rf_importance['feature'], fontsize=8)
    plt.xlabel('Importance', fontsize=12)
    plt.title(f'Random Forest Feature Importance - ALL {len(rf_importance)} Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels for significant features
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if abs(width) > 0.01:  # Only label significant features
            plt.text(width + (0.005 if width > 0 else -0.005), bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=7)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/random_forest_all_features_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/random_forest_all_features_importance.pdf', bbox_inches='tight')
    print(f"Random Forest ALL features plot saved (showing {len(rf_importance)} features)")
    plt.close()
    
    # 3. Neural Network - ALL Features
    plt.figure(figsize=(12, fig_height))
    colors = ['darkblue' if x < 0 else 'lightblue' for x in nn_importance['importance']]
    bars = plt.barh(range(len(nn_importance)), nn_importance['importance'], color=colors)
    plt.yticks(range(len(nn_importance)), nn_importance['feature'], fontsize=8)
    plt.xlabel('Permutation Importance', fontsize=12)
    plt.title(f'Neural Network Permutation Importance - ALL {len(nn_importance)} Features', fontsize=14, fontweight='bold')
    plt.grid(axis='x', alpha=0.3)
    
    # Add value labels for significant features
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if abs(width) > 0.01:  # Only label significant features
            plt.text(width + (0.005 if width > 0 else -0.005), bar.get_y() + bar.get_height()/2, 
                    f'{width:.3f}', ha='left' if width > 0 else 'right', va='center', fontsize=7)
    
    plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/neural_network_all_features_importance.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/neural_network_all_features_importance.pdf', bbox_inches='tight')
    print(f"Neural Network ALL features plot saved (showing {len(nn_importance)} features)")
    plt.close()
    
    # 4. Summary statistics about negative/positive importance
    print(f"\n=== FEATURE IMPORTANCE ANALYSIS ===")
    
    for model_name, importance_df in [("XGBoost", xgb_importance), ("Random Forest", rf_importance), ("Neural Network", nn_importance)]:
        positive_features = importance_df[importance_df['importance'] > 0]
        negative_features = importance_df[importance_df['importance'] < 0]
        zero_features = importance_df[importance_df['importance'] == 0]
        
        print(f"\n{model_name}:")
        print(f"  Positive importance: {len(positive_features)} features")
        print(f"  Negative importance: {len(negative_features)} features") 
        print(f"  Zero importance: {len(zero_features)} features")
        print(f"  Range: {importance_df['importance'].min():.4f} to {importance_df['importance'].max():.4f}")
        
        if len(negative_features) > 0:
            print(f"  Most negative features:")
            for _, row in negative_features.head(5).iterrows():
                print(f"    {row['feature']}: {row['importance']:.4f}")
                
    # 5. Create a summary comparison of the models
    create_model_summary_comparison(xgb_importance, rf_importance, nn_importance)

def create_model_summary_comparison(xgb_importance, rf_importance, nn_importance):
    """Create a summary comparison showing top positive and negative features"""
    
    fig, axes = plt.subplots(3, 2, figsize=(20, 18))
    
    models = [
        ("XGBoost", xgb_importance, 'lightcoral', 'darkred'),
        ("Random Forest", rf_importance, 'lightgreen', 'darkgreen'), 
        ("Neural Network", nn_importance, 'lightblue', 'darkblue')
    ]
    
    for i, (model_name, importance_df, pos_color, neg_color) in enumerate(models):
        # Top 15 positive features
        top_positive = importance_df[importance_df['importance'] > 0].head(15)
        if len(top_positive) > 0:
            bars = axes[i, 0].barh(range(len(top_positive)), top_positive['importance'], color=pos_color)
            axes[i, 0].set_yticks(range(len(top_positive)))
            axes[i, 0].set_yticklabels(top_positive['feature'], fontsize=9)
            axes[i, 0].set_xlabel('Importance')
            axes[i, 0].set_title(f'{model_name} - Top 15 Positive Features')
            axes[i, 0].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                axes[i, 0].text(width + width*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='left', va='center', fontsize=8)
        
        # Top 15 negative features (most negative)
        negative_features = importance_df[importance_df['importance'] < 0].tail(15)  # Most negative
        if len(negative_features) > 0:
            bars = axes[i, 1].barh(range(len(negative_features)), negative_features['importance'], color=neg_color)
            axes[i, 1].set_yticks(range(len(negative_features)))
            axes[i, 1].set_yticklabels(negative_features['feature'], fontsize=9)
            axes[i, 1].set_xlabel('Importance')
            axes[i, 1].set_title(f'{model_name} - Most Negative 15 Features')
            axes[i, 1].grid(axis='x', alpha=0.3)
            
            # Add value labels
            for j, bar in enumerate(bars):
                width = bar.get_width()
                axes[i, 1].text(width - abs(width)*0.01, bar.get_y() + bar.get_height()/2, 
                        f'{width:.3f}', ha='right', va='center', fontsize=8)
        else:
            axes[i, 1].text(0.5, 0.5, 'No negative features', ha='center', va='center', transform=axes[i, 1].transAxes)
            axes[i, 1].set_title(f'{model_name} - No Negative Features')
    
    plt.tight_layout()
    plt.savefig('results/model_comparison_positive_negative_features.png', dpi=300, bbox_inches='tight')
    plt.savefig('results/model_comparison_positive_negative_features.pdf', bbox_inches='tight')
    print("Model comparison (positive/negative) plot saved")
    plt.close()

def generate_all_features_list():
    """Generate a complete list of all features used"""
    print("\nGenerating complete feature list...")
    
    # Load one of the feature importance files to get all features
    nn_importance = pd.read_csv('results/professional_tennis/Neural_Network/feature_importance.csv')
    
    # Save complete feature list
    all_features = nn_importance['feature'].tolist()
    
    print(f"Total features used in models: {len(all_features)}")
    
    # Save to text file
    with open('results/all_features_list.txt', 'w') as f:
        f.write("Complete List of Features Used in Tennis Prediction Models\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total Features: {len(all_features)}\n\n")
        
        # Group features by category
        temporal_features = [f for f in all_features if any(x in f for x in ['_30d', '_90d', '_14d', '_120d', 'WinStreak', 'Form_Trend', 'Momentum', 'Matches_', 'Sets_'])]
        ranking_features = [f for f in all_features if any(x in f for x in ['Rank', 'Points'])]
        physical_features = [f for f in all_features if any(x in f for x in ['Height', 'Age', 'Hand'])]
        match_context = [f for f in all_features if any(x in f for x in ['Level_', 'Round_', 'Surface_', 'draw_size', 'Season'])]
        country_features = [f for f in all_features if 'Country_' in f]
        h2h_features = [f for f in all_features if 'H2H_' in f]
        other_features = [f for f in all_features if f not in temporal_features + ranking_features + physical_features + match_context + country_features + h2h_features]
        
        categories = [
            ("Temporal Features", temporal_features),
            ("Ranking Features", ranking_features),
            ("Physical Features", physical_features),
            ("Match Context Features", match_context),
            ("Country Features", country_features),
            ("Head-to-Head Features", h2h_features),
            ("Other Features", other_features)
        ]
        
        for category_name, features in categories:
            if features:
                f.write(f"{category_name} ({len(features)} features):\n")
                f.write("-" * 40 + "\n")
                for feature in sorted(features):
                    f.write(f"  - {feature}\n")
                f.write("\n")
    
    print("Complete feature list saved to: results/all_features_list.txt")
    
    # Also print summary to console
    print("\nFeature Categories Summary:")
    for category_name, features in categories:
        if features:
            print(f"  {category_name}: {len(features)} features")

def main():
    print("=" * 80)
    print("COMPREHENSIVE FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    # Generate comprehensive comparison plot
    create_comprehensive_comparison_plot()
    
    # Generate complete feature list
    generate_all_features_list()
    
    print(f"\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("Files generated:")
    print("  - results/xgboost_all_features_importance.png (ALL XGBoost features)")
    print("  - results/random_forest_all_features_importance.png (ALL Random Forest features)")
    print("  - results/neural_network_all_features_importance.png (ALL Neural Network features)")
    print("  - results/model_comparison_positive_negative_features.png (Summary comparison)")
    print("  - results/all_features_list.txt (Complete feature list)")
    print("  - PDF versions of all plots also generated")

if __name__ == "__main__":
    main()