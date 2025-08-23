import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import sys

# Add the current directory to path to import preprocessing function
sys.path.append(os.path.dirname(__file__))
from preprocess import get_feature_columns

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "../../..", "results", "betting_odds", "Random_Forest")
os.makedirs(output_dir, exist_ok=True)

print("=" * 60)
print("BETTING ODDS RANDOM FOREST MODEL TRAINING")
print("=" * 60)

# Load the ML-ready dataset
print("\n1. Loading ML-ready betting odds dataset...")
ml_ready_path = os.path.join(os.path.dirname(__file__), "../../..", "data", "betting_odds", "ml_ready.csv")
if not os.path.exists(ml_ready_path):
    print("ML-ready dataset not found. Please run preprocessing first.")
    sys.exit(1)
df = pd.read_csv(ml_ready_path)
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

# Train Random Forest model
print("\n5. Training Random Forest model...")
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("   ✅ Model training complete!")

# Make predictions
print("\n6. Making predictions...")
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
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
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 most important features:")
for i, (_, row) in enumerate(feature_importance.head(10).iterrows()):
    print(f"{i+1:2d}. {row['feature']:25s} {row['importance']:.4f}")

# Save detailed results
print("\n8. Saving results...")

# Save predictions
results_df = test_df[['Date', 'Tournament', 'Player1_Name', 'Player2_Name', 'Player1_Wins']].copy()
results_df['RF_Prediction'] = y_pred
results_df['RF_Probability'] = y_pred_proba
results_df['Correct'] = (y_pred == y_test).astype(int)
results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save feature importance
feature_importance.to_csv(os.path.join(output_dir, 'feature_importance.csv'), index=False)

# Save model metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC'],
    'Value': [accuracy, precision, recall, f1, auc]
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

# Create visualizations
print("\n9. Creating visualizations...")

# Feature importance plot
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
plt.title('Random Forest - Top 15 Feature Importances')
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
plt.title('Random Forest - Prediction Probability Distribution')
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
plt.title('Random Forest - Monthly Accuracy Over Time')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   Results saved to: {output_dir}")

print("\n" + "=" * 60)
print("RANDOM FOREST TRAINING COMPLETE")
print("=" * 60)
print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Results saved to: {output_dir}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Player2_Wins', 'Player1_Wins']))