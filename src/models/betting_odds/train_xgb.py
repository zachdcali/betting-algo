import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
import os
import sys

# Add the src directory to the path to import our preprocessing function
sys.path.append('/app/src')
from preprocess_tennis_data_ml import get_feature_columns

# Create output directory
base_output_dir = "/app/data/output"
output_dir = os.path.join(base_output_dir, "XGBoost")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("XGBOOST MODEL TRAINING")
print("=" * 60)

# Load the ML-ready dataset
print("\n1. Loading ML-ready dataset...")
df = pd.read_csv("/app/data/tennis_data_ml_ready.csv")
print(f"   Dataset shape: {df.shape}")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# Get feature columns automatically
feature_cols = get_feature_columns(df)
print(f"   Features for training: {len(feature_cols)}")

# Split into train/test based on date
print("\n2. Creating train/test split...")
train_df = df[df['Date'] < '2023-01-01'].copy()
test_df = df[df['Date'] >= '2023-01-01'].copy()

print(f"   Training set: {len(train_df)} matches (2000-2022)")
print(f"   Test set: {len(test_df)} matches (2023-2025)")

# Prepare features and target
X_train = train_df[feature_cols]
y_train = train_df['Player1_Wins']
X_test = test_df[feature_cols]
y_test = test_df['Player1_Wins']

print(f"   Training features shape: {X_train.shape}")
print(f"   Test features shape: {X_test.shape}")
print(f"   Target distribution - Train: {y_train.mean():.3f}, Test: {y_test.mean():.3f}")

# Check for missing values
print("\n3. Checking data quality...")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"   Training missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("   Filling missing values with median...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training median for test set

# Display feature information
print(f"\n4. Feature information:")
print(f"   Core features: {[col for col in feature_cols if not any(x in col for x in ['Surface_', 'Court_', 'Round_', 'Series_'])]}")
print(f"   Surface features: {[col for col in feature_cols if 'Surface_' in col]}")
print(f"   Court features: {[col for col in feature_cols if 'Court_' in col]}")
print(f"   Round features: {[col for col in feature_cols if 'Round_' in col]}")
print(f"   Series features: {[col for col in feature_cols if 'Series_' in col]}")

# Train XGBoost model
print("\n5. Training XGBoost model...")
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss',
    n_jobs=-1
)

# Train with validation to prevent overfitting
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)
print("   ✅ Model training complete!")

# Make predictions
print("\n6. Making predictions...")
y_pred = xgb_model.predict(X_test)
y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")


# Feature importance analysis
print("\n7. Feature importance analysis...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

# Calculate ATP ranking baseline for comparison
print("\n8. Calculating ATP ranking baseline...")
# Check if we have ranking columns in the test data
ranking_cols = [col for col in test_df.columns if 'Rank' in col and 'Player' in col]
if len(ranking_cols) >= 2:
    # Find Player1 and Player2 ranking columns
    p1_rank_col = None
    p2_rank_col = None
    for col in ranking_cols:
        if 'Player1' in col and 'Rank' in col and not any(x in col for x in ['Points', 'Movement']):
            p1_rank_col = col
        elif 'Player2' in col and 'Rank' in col and not any(x in col for x in ['Points', 'Movement']):
            p2_rank_col = col
    
    if p1_rank_col and p2_rank_col:
        # Correct baseline: predict the better-ranked player wins (regardless of Player1/Player2 position)
        baseline_correct = (
            ((test_df[p1_rank_col] < test_df[p2_rank_col]) & (test_df['Player1_Wins'] == 1)) |
            ((test_df[p1_rank_col] > test_df[p2_rank_col]) & (test_df['Player1_Wins'] == 0))
        ).sum()
        baseline_total = len(test_df)
        baseline_accuracy = baseline_correct / baseline_total
        
        print(f"   ATP Ranking Baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
        print(f"   XGBoost vs ATP Ranking: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
    else:
        print("   Warning: Could not find both Player1 and Player2 ranking columns")
        baseline_accuracy = None
else:
    print("   Warning: Ranking data not available for baseline comparison")
    baseline_accuracy = None

# Detailed performance analysis
print("\n9. Detailed Performance Analysis...")

# Prepare results dataframe for analysis
results_df = test_df[['Date', 'Tournament', 'Player1_Name', 'Player2_Name', 'Player1_Wins']].copy()
results_df['XGB_Prediction'] = y_pred
results_df['XGB_Probability'] = y_pred_proba
results_df['Correct'] = (y_pred == y_test).astype(int)
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df['Year'] = results_df['Date'].dt.year

# Add ranking data if available for subset analysis
if p1_rank_col and p2_rank_col:
    results_df['Player1_Rank'] = test_df[p1_rank_col]
    results_df['Player2_Rank'] = test_df[p2_rank_col]

# Calculate subset metrics helper function
def calculate_subset_metrics(subset_df):
    if len(subset_df) == 0:
        return None, None, 0
    
    # Model accuracy
    model_acc = subset_df['Correct'].mean()
    
    # ATP baseline accuracy for this subset (if ranking data available)
    if 'Player1_Rank' in subset_df.columns and 'Player2_Rank' in subset_df.columns:
        baseline_correct = (
            ((subset_df['Player1_Rank'] < subset_df['Player2_Rank']) & (subset_df['Player1_Wins'] == 1)) |
            ((subset_df['Player1_Rank'] > subset_df['Player2_Rank']) & (subset_df['Player1_Wins'] == 0))
        ).sum()
        baseline_acc = baseline_correct / len(subset_df)
    else:
        baseline_acc = None
    
    return model_acc, baseline_acc, len(subset_df)

print("\n=== YEAR-BY-YEAR BREAKDOWN ===")
for year in sorted(results_df['Year'].unique()):
    year_data = results_df[results_df['Year'] == year]
    model_acc, baseline_acc, count = calculate_subset_metrics(year_data)
    if model_acc is not None:
        if baseline_acc is not None:
            improvement = (model_acc - baseline_acc) * 100
            print(f"{year}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
        else:
            print(f"{year}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

# Check for tournament level and surface data
if 'Series' in test_df.columns:
    print("\n=== TOURNAMENT LEVEL BREAKDOWN ===")
    # Show all series/levels in test data
    series_counts = test_df['Series'].value_counts()
    print("All tournament series in test data:")
    for series, count in series_counts.items():
        print(f"  '{series}': {count:,} matches")
    print()
    
    # Group tournament levels (adapt based on available data)
    series_groups = {
        'ATP-Level': ['ATP Masters 1000', 'Grand Slam', 'ATP 500', 'ATP 250', 'ATP Finals'],
        'Challengers': ['ATP Challenger', 'Challenger'],
        'ITF-Futures': ['ITF', 'Futures'],
        'Other': ['Davis Cup', 'Exhibition']
    }
    
    for group_name, series_list in series_groups.items():
        group_data = results_df[test_df['Series'].isin(series_list)]
        model_acc, baseline_acc, count = calculate_subset_metrics(group_data)
        if model_acc is not None and count > 0:
            if baseline_acc is not None:
                improvement = (model_acc - baseline_acc) * 100
                print(f"{group_name:12s}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
            else:
                print(f"{group_name:12s}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

if 'Surface' in test_df.columns:
    print("\n=== SURFACE BREAKDOWN ===")
    surfaces = [s for s in test_df['Surface'].unique() if pd.notna(s)]
    for surface in sorted(surfaces):
        surface_data = results_df[test_df['Surface'] == surface]
        model_acc, baseline_acc, count = calculate_subset_metrics(surface_data)
        if model_acc is not None and count > 100:  # Only show if substantial sample
            if baseline_acc is not None:
                improvement = (model_acc - baseline_acc) * 100
                print(f"{surface:12s}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
            else:
                print(f"{surface:12s}: XGB {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

# Save detailed results
print("\n10. Saving results...")

# Save predictions (reuse the detailed results_df from analysis)
results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save feature importance
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Save model metrics including baseline
metrics_list = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC']
values_list = [accuracy, precision, recall, f1, auc]
if baseline_accuracy is not None:
    metrics_list.append('ATP_Baseline')
    values_list.append(baseline_accuracy)

metrics_df = pd.DataFrame({
    'Metric': metrics_list,
    'Value': values_list
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

# Create visualizations
print("\n11. Creating visualizations...")

# Feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('XGBoost - Top 15 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Prediction probability distribution
plt.figure(figsize=(10, 6))
plt.hist(y_pred_proba[y_test == 1], bins=50, alpha=0.7, label='Player1 Won', density=True)
plt.hist(y_pred_proba[y_test == 0], bins=50, alpha=0.7, label='Player1 Lost', density=True)
plt.xlabel('Predicted Probability (Player1 Wins)')
plt.ylabel('Density')
plt.title('XGBoost - Prediction Probability Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Monthly accuracy analysis
results_df['Date'] = pd.to_datetime(results_df['Date'])
results_df['Year_Month'] = results_df['Date'].dt.to_period('M')
monthly_acc = results_df.groupby('Year_Month')['Correct'].mean()

plt.figure(figsize=(12, 6))
monthly_acc.plot(kind='line', marker='o')
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy ({accuracy:.3f})')
if baseline_accuracy is not None:
    plt.axhline(y=baseline_accuracy, color='orange', linestyle='--', label=f'ATP Ranking Baseline ({baseline_accuracy:.3f})')
plt.title('XGBoost - Monthly Accuracy Over Time')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

# Training curve (if available)
if hasattr(xgb_model, 'evals_result_'):
    results = xgb_model.evals_result_
    if 'validation_0' in results:
        plt.figure(figsize=(10, 6))
        plt.plot(results['validation_0']['logloss'], label='Validation Loss')
        plt.title('XGBoost - Training Curve')
        plt.xlabel('Boosting Rounds')
        plt.ylabel('Log Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'training_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

print(f"   Results saved to: {output_dir}")

print("\n" + "=" * 60)
print("XGBOOST TRAINING COMPLETE")
print("=" * 60)
print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
if baseline_accuracy is not None:
    print(f"ATP Ranking baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"Improvement: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
print(f"Results saved to: {output_dir}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Player2_Wins', 'Player1_Wins']))

