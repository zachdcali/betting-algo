import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
import seaborn as sns
import os
import sys

# Add the current directory to path to import preprocessing function
sys.path.append(os.path.dirname(__file__))
from preprocess import get_feature_columns

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "../../..", "results", "professional_tennis", "Random_Forest")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("PROFESSIONAL TENNIS RANDOM FOREST MODEL TRAINING")
print("=" * 60)

# Parameters
MIN_YEAR = 1990  # Starting from 1990 for better data quality - can be adjusted as needed

# Check if ML-ready dataset exists, if not create it
ml_ready_path = os.path.join(os.path.dirname(__file__), "../../..", "data", "professional_tennis", "ml_ready_all_years.csv")
if not os.path.exists(ml_ready_path):
    print("ML-ready dataset not found. Creating it...")
    from preprocess import preprocess_data_for_ml
    ml_df, feature_columns = preprocess_jeffsackmann_data_for_ml()
else:
    print("\n1. Loading ML-ready Jeffsackmann dataset...")
    ml_df = pd.read_csv(ml_ready_path, low_memory=False)
    print(f"   Dataset shape: {ml_df.shape}")
    print(f"   Date range: {ml_df['tourney_date'].min()} to {ml_df['tourney_date'].max()}")

# Convert date column and add year
ml_df['tourney_date'] = pd.to_datetime(ml_df['tourney_date'])
ml_df['year'] = ml_df['tourney_date'].dt.year

# Filter by minimum year
print(f"\n2. Filtering data from {MIN_YEAR} onwards...")
initial_count = len(ml_df)
ml_df = ml_df[ml_df['year'] >= MIN_YEAR].copy()
print(f"   After year filtering: {len(ml_df)} matches ({initial_count - len(ml_df)} removed)")

# Filter to matches with required features (only ATP rankings)
print("\n3. Filtering to matches with required features...")
required_features = ['Player1_Rank', 'Player2_Rank']  # Only rankings are required

# Check for required features
feature_missing = 0
for feature in required_features:
    if feature in ml_df.columns:
        missing_count = ml_df[feature].isnull().sum()
        if missing_count > 0:
            print(f"   Missing {feature}: {missing_count} matches")
            feature_missing += missing_count
    else:
        print(f"   Warning: {feature} not found in dataset")

# Remove matches with missing required features
before_filter = len(ml_df)
ml_df = ml_df.dropna(subset=[col for col in required_features if col in ml_df.columns])
print(f"   After feature filtering: {len(ml_df)} matches ({before_filter - len(ml_df)} removed)")

# Show missing data for optional features
optional_features = ['Player1_Height', 'Player2_Height', 'Player1_Age', 'Player2_Age', 
                    'Player1_Rank_Points', 'Player2_Rank_Points']
print("   Optional features missing data:")
for feature in optional_features:
    if feature in ml_df.columns:
        missing_count = ml_df[feature].isnull().sum()
        missing_pct = (missing_count / len(ml_df)) * 100
        print(f"   - {feature}: {missing_count} matches ({missing_pct:.1f}%)")
    else:
        print(f"   - {feature}: not found in dataset")

# Show handedness data availability (one-hot encoded)
hand_cols = [col for col in ml_df.columns if 'P1_Hand_' in col or 'P2_Hand_' in col]
if hand_cols:
    print("   Handedness features (one-hot encoded):")
    for col in hand_cols:
        non_zero = (ml_df[col] == 1).sum()
        print(f"   - {col}: {non_zero:,} matches")
else:
    print("   - Handedness: not found in dataset")

# Get feature columns automatically
feature_cols = get_feature_columns(ml_df)

# Filter to only numeric columns to avoid the median error
numeric_cols = []
for col in feature_cols:
    if col in ml_df.columns:
        if ml_df[col].dtype in ['int64', 'float64', 'bool']:
            numeric_cols.append(col)
        elif ml_df[col].dtype == 'object':
            # Check if it's actually numeric but stored as object
            try:
                pd.to_numeric(ml_df[col], errors='raise')
                numeric_cols.append(col)
            except:
                print(f"   Skipping non-numeric column: {col} (dtype: {ml_df[col].dtype})")

feature_cols = numeric_cols
print(f"   Features for training: {len(feature_cols)}")

# Split into train/test based on date
print("\n4. Creating train/test split...")
train_df = ml_df[ml_df['tourney_date'] < '2023-01-01'].copy()
test_df = ml_df[ml_df['tourney_date'] >= '2023-01-01'].copy()

print(f"   Training set: {len(train_df)} matches ({MIN_YEAR}-2022)")
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
print("\n5. Checking data quality...")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()
print(f"   Training missing values: {train_missing}")
print(f"   Test missing values: {test_missing}")

if train_missing > 0 or test_missing > 0:
    print("   Filling missing values with median...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training median for test set

# Display feature information
print(f"\n6. Feature information:")
core_features = [col for col in feature_cols if not any(x in col for x in ['Surface_', 'Level_', 'Round_', 'P1_Hand', 'P2_Hand', 'P1_Country', 'P2_Country'])]
surface_features = [col for col in feature_cols if 'Surface_' in col]
level_features = [col for col in feature_cols if 'Level_' in col]
round_features = [col for col in feature_cols if 'Round_' in col]
hand_features = [col for col in feature_cols if 'P1_Hand' in col or 'P2_Hand' in col]
country_features = [col for col in feature_cols if 'P1_Country' in col or 'P2_Country' in col]

print(f"   Core features: {len(core_features)}")
print(f"   Surface features: {len(surface_features)}")
print(f"   Level features: {len(level_features)}")
print(f"   Round features: {len(round_features)}")
print(f"   Hand features: {len(hand_features)}")
print(f"   Country features: {len(country_features)}")

# Train Random Forest model
print("\n7. Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,  # Slightly deeper for more complex Jeffsackmann data
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# Make predictions
print("\n8. Making predictions...")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("JEFFSACKMANN RANDOM FOREST RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# Calculate ATP ranking baseline for comparison
print("\n9. Calculating ATP ranking baseline...")
# Correct baseline: predict the better-ranked player wins (regardless of Player1/Player2 position)
baseline_correct = (
    ((test_df['Player1_Rank'] < test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 1)) |
    ((test_df['Player1_Rank'] > test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 0))
).sum()
baseline_total = len(test_df)
baseline_accuracy = baseline_correct / baseline_total

print(f"   ATP Ranking Baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"   Random Forest vs ATP Ranking: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")

# Feature importance analysis
print("\n10. Feature importance analysis...")
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 15 most important features:")
for i, (_, row) in enumerate(feature_importance.head(15).iterrows()):
    print(f"{i+1:2d}. {row['feature']:30s} {row['importance']:.4f}")

# Detailed performance analysis
print("\n11. Detailed Performance Analysis...")

# Prepare results dataframe
results_df = test_df[['tourney_date', 'tourney_name', 'tourney_level', 'surface', 'Player1_Name', 'Player2_Name', 'Player1_Rank', 'Player2_Rank', 'Player1_Wins']].copy()
results_df['RF_Prediction'] = y_pred
results_df['RF_Probability'] = y_pred_proba
results_df['Correct'] = (y_pred == y_test).astype(int)
results_df['tourney_date'] = pd.to_datetime(results_df['tourney_date'])
results_df['Year'] = results_df['tourney_date'].dt.year

# Calculate ATP baseline for each subset
def calculate_subset_metrics(subset_df):
    if len(subset_df) == 0:
        return None, None, 0
    
    # Model accuracy
    model_acc = subset_df['Correct'].mean()
    
    # ATP baseline accuracy for this subset
    baseline_correct = (
        ((subset_df['Player1_Rank'] < subset_df['Player2_Rank']) & (subset_df['Player1_Wins'] == 1)) |
        ((subset_df['Player1_Rank'] > subset_df['Player2_Rank']) & (subset_df['Player1_Wins'] == 0))
    ).sum()
    baseline_acc = baseline_correct / len(subset_df)
    
    return model_acc, baseline_acc, len(subset_df)

print("\n=== YEAR-BY-YEAR BREAKDOWN ===")
for year in sorted(results_df['Year'].unique()):
    year_data = results_df[results_df['Year'] == year]
    model_acc, baseline_acc, count = calculate_subset_metrics(year_data)
    if model_acc is not None:
        improvement = (model_acc - baseline_acc) * 100
        print(f"{year}: RF {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

print("\n=== TOURNAMENT LEVEL BREAKDOWN ===")
# First show ALL tournament levels to see what we have
print("All tournament levels in test data:")
level_counts = results_df['tourney_level'].value_counts()
for level, count in level_counts.items():
    print(f"  '{level}': {count:,} matches")
print()

# Group tournament levels
level_groups = {
    'ATP-Level': ['G', 'M', 'A', 'F'],  # Grand Slams, Masters, ATP, Finals
    'Challengers': ['C'],
    'ITF-Futures': ['15', '25', 'S'],  # ITF $15K, $25K, Satellites
    'Other': ['D', 'O']  # Davis Cup, Other
}

for group_name, levels in level_groups.items():
    group_data = results_df[results_df['tourney_level'].isin(levels)]
    model_acc, baseline_acc, count = calculate_subset_metrics(group_data)
    if model_acc is not None and count > 0:  # Show all groups with any data
        improvement = (model_acc - baseline_acc) * 100
        print(f"{group_name:12s}: RF {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

print("\n=== SURFACE BREAKDOWN ===")
surfaces = [s for s in results_df['surface'].unique() if pd.notna(s)]
for surface in sorted(surfaces):
    surface_data = results_df[results_df['surface'] == surface]
    model_acc, baseline_acc, count = calculate_subset_metrics(surface_data)
    if model_acc is not None and count > 100:  # Only show if substantial sample
        improvement = (model_acc - baseline_acc) * 100
        print(f"{surface:12s}: RF {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

print("\n12. Saving results...")

# Save predictions with detailed info
results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save feature importance
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Save model metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'ATP_Baseline'],
    'Value': [accuracy, precision, recall, f1, auc, baseline_accuracy]
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

# Create visualizations
print("\n13. Creating visualizations...")

# Feature importance plot
plt.figure(figsize=(12, 10))
sns.barplot(data=feature_importance.head(20), x='importance', y='feature')
plt.title('Jeffsackmann Random Forest - Top 20 Feature Importances')
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
plt.title('Jeffsackmann Random Forest - Prediction Probability Distribution')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'probability_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Monthly accuracy analysis
results_df['tourney_date'] = pd.to_datetime(results_df['tourney_date'])
results_df['Year_Month'] = results_df['tourney_date'].dt.to_period('M')
monthly_acc = results_df.groupby('Year_Month')['Correct'].mean()

plt.figure(figsize=(12, 6))
monthly_acc.plot(kind='line', marker='o')
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy ({accuracy:.3f})')
plt.axhline(y=baseline_accuracy, color='orange', linestyle='--', label=f'ATP Ranking Baseline ({baseline_accuracy:.3f})')
plt.title('Jeffsackmann Random Forest - Monthly Accuracy Over Time')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   Results saved to: {output_dir}")

print("\n" + "=" * 60)
print("JEFFSACKMANN RANDOM FOREST TRAINING COMPLETE")
print("=" * 60)
print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ATP Ranking baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Improvement: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
print(f"Results saved to: {output_dir}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Player2_Wins', 'Player1_Wins']))