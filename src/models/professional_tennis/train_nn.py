import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, log_loss
from sklearn.calibration import calibration_curve
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the current directory to path to import preprocessing function
sys.path.append(os.path.dirname(__file__))
from preprocess import get_feature_columns

# Create output directory
output_dir = os.path.join(os.path.dirname(__file__), "../../..", "results", "professional_tennis", "Neural_Network")
os.makedirs(output_dir, exist_ok=True)

# Set device (use CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TennisNet(nn.Module):
    """
    Neural Network for tennis match prediction with more capacity for Jeffsackmann data
    """
    def __init__(self, input_size):
        super(TennisNet, self).__init__()
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layer 1
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layer 2
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Hidden layer 3
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001, patience=10):
    """
    Train the neural network model with early stopping
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_epoch = 0
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x).squeeze()
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predicted = (outputs > 0.5).float()
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        # Calculate averages
        train_loss_avg = train_loss / len(train_loader)
        val_loss_avg = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Early stopping check
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            best_epoch = epoch
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'   Early stopping at epoch {epoch} (best epoch: {best_epoch}, best val loss: {best_val_loss:.4f})')
                model.load_state_dict(best_model_state)
                break
        
        print(f'   Epoch {epoch:3d}/{num_epochs}: Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

print("=" * 60)
print("PROFESSIONAL TENNIS NEURAL NETWORK MODEL TRAINING")
print("=" * 60)

# Parameters
MIN_YEAR = 1990  # Starting from 1990 for better data quality - can be adjusted as needed

# Check if ML-ready dataset exists, if not create it
ml_ready_path = os.path.join(os.path.dirname(__file__), "../../..", "data", "JeffSackmann", "jeffsackmann_ml_ready_LEAK_FREE.csv")
if not os.path.exists(ml_ready_path):
    print("ML-ready dataset not found. Creating it...")
    from preprocess import preprocess_data_for_ml
    ml_df, feature_columns = preprocess_data_for_ml()
else:
    print("\n1. Loading ML-ready professional tennis dataset...")
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

# Use the exact 143 features from the feature importance analysis (in order of importance)
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
    # Zero importance features (7 total)
    'Round_Q4', 'Peak_Age_P1', 'Level_G', 'Round_ER', 'Level_S', 'Round_BR', 'Peak_Age_P2',
    # Negative importance features (31 total)
    'Round_Q3', 'Rank_Ratio', 'P1_Country_SUI', 'Clay_Season', 'P1_Country_GER',
    'P2_Rank_Change_30d', 'P1_Country_ESP', 'P2_Hand_A', 'H2H_Recent_P1_Advantage',
    'P2_Country_AUS', 'P2_Country_SRB', 'P2_Country_GBR', 'P2_Country_ARG',
    'Handedness_Matchup_RR', 'P1_Rank_Change_30d', 'P2_Country_GER', 'Handedness_Matchup_LL',
    'P2_Country_RUS', 'P1_Country_ARG', 'Level_C', 'P2_Semifinals_WinRate',
    'P2_Days_Since_Last', 'P1_Peak_Age', 'P2_Peak_Age', 'H2H_P1_WinRate',
    'P1_Country_Other', 'H2H_P1_Wins', 'P1_BigMatch_WinRate', 'P2_Rank_Change_90d',
    'P2_BigMatch_WinRate', 'P2_Country_FRA'
]

# Filter to only features that exist in the dataset
feature_cols = [col for col in exact_143_features if col in ml_df.columns]

print(f"   Features for training: {len(feature_cols)} (exact 143-feature model from importance analysis)")
if len(feature_cols) != 143:
    print(f"   WARNING: Expected 143 features but got {len(feature_cols)} - some features may be missing from dataset")
    missing_features = [col for col in exact_143_features if col not in ml_df.columns]
    if missing_features:
        print(f"   Missing features: {missing_features[:10]}...")  # Show first 10 missing

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

# Check for missing values and fill
print("\n5. Checking data quality...")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

if train_missing > 0 or test_missing > 0:
    print("   Filling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training median for test set

print(f"   Training missing values: {X_train.isnull().sum().sum()}")
print(f"   Test missing values: {X_test.isnull().sum().sum()}")

# Standardize features
print("\n6. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

print(f"   Feature scaling complete")

# Create PyTorch datasets
print("\n7. Creating PyTorch datasets...")
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train.values)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.FloatTensor(y_test.values)
)

# Create data loaders
batch_size = 1024  # Larger batch size for larger dataset
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Create and train model
print("\n8. Training Neural Network model...")
input_size = X_train_scaled.shape[1]
model = TennisNet(input_size).to(device)

print(f"   Model architecture:")
print(f"   - Input size: {input_size}")
print(f"   - Hidden layers: 128 -> 64 -> 32 -> 16")
print(f"   - Output size: 1 (sigmoid)")
print(f"   - Dropout: 0.3, 0.3, 0.2, 0.2")

# Train the model
num_epochs = 100
learning_rate = 0.001

train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, test_loader, num_epochs, learning_rate
)

print("   ✅ Model training complete!")

# Make predictions
print("\n9. Making predictions...")
model.eval()
all_predictions = []
all_probabilities = []

with torch.no_grad():
    for batch_x, _ in test_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x).squeeze()
        probabilities = outputs.cpu().numpy()
        predictions = (outputs > 0.5).float().cpu().numpy()
        
        all_predictions.extend(predictions)
        all_probabilities.extend(probabilities)

y_pred = np.array(all_predictions)
y_pred_proba = np.array(all_probabilities)

# Calculate metrics
accuracy = accuracy_score(y_test.values, y_pred)
precision = precision_score(y_test.values, y_pred)
recall = recall_score(y_test.values, y_pred)
f1 = f1_score(y_test.values, y_pred)
auc = roc_auc_score(y_test.values, y_pred_proba)

print("\n" + "=" * 60)
print("PROFESSIONAL TENNIS NEURAL NETWORK RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

# Calculate additional calibration metrics
logloss = log_loss(y_test.values, y_pred_proba)
brier = np.mean((y_pred_proba - y_test.values)**2)
prob_true, prob_pred = calibration_curve(y_test.values, y_pred_proba, n_bins=10)
ece = np.mean(np.abs(prob_true - prob_pred))
print(f"Log Loss: {logloss:.4f}")
print(f"Brier Score: {brier:.4f}")
print(f"Expected Calibration Error (ECE): {ece:.4f}")

# Calculate ATP ranking baseline for comparison
print("\n10. Calculating ATP ranking baseline...")
# Correct baseline: predict the better-ranked player wins (regardless of Player1/Player2 position)
baseline_correct = (
    ((test_df['Player1_Rank'] < test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 1)) |
    ((test_df['Player1_Rank'] > test_df['Player2_Rank']) & (test_df['Player1_Wins'] == 0))
).sum()
baseline_total = len(test_df)
baseline_accuracy = baseline_correct / baseline_total

print(f"   ATP Ranking Baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"   Neural Network vs ATP Ranking: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")

# Detailed performance analysis
print("\n11. Detailed Performance Analysis...")

# Prepare results dataframe
results_df = test_df[['tourney_date', 'tourney_name', 'tourney_level', 'surface', 'Player1_Name', 'Player2_Name', 'Player1_Rank', 'Player2_Rank', 'Player1_Wins']].copy()
results_df['NN_Prediction'] = y_pred
results_df['NN_Probability'] = y_pred_proba
results_df['Correct'] = (y_pred == y_test.values).astype(int)
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
        print(f"{year}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

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
        print(f"{group_name:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

print("\n=== SURFACE BREAKDOWN ===")
surfaces = [s for s in results_df['surface'].unique() if pd.notna(s)]
for surface in sorted(surfaces):
    surface_data = results_df[results_df['surface'] == surface]
    model_acc, baseline_acc, count = calculate_subset_metrics(surface_data)
    if model_acc is not None and count > 100:  # Only show if substantial sample
        improvement = (model_acc - baseline_acc) * 100
        print(f"{surface:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")

print("\n12. Saving results...")

# Save predictions with detailed info
results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

# Save model metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC-ROC', 'Log_Loss', 'Brier_Score', 'ECE', 'ATP_Baseline'],
    'Value': [accuracy, precision, recall, f1, auc, logloss, brier, ece, baseline_accuracy]
})
metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

# Create visualizations
print("\n12. Creating visualizations...")

# Training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Professional Tennis Neural Network - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Professional Tennis Neural Network - Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[y_test.values == 1], bins=50, alpha=0.7, label='Player1 Won', density=True)
plt.hist(y_pred_proba[y_test.values == 0], bins=50, alpha=0.7, label='Player1 Lost', density=True)
plt.title('Professional Tennis Neural Network - Prediction Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
plt.close()

# Monthly accuracy analysis
results_df['tourney_date'] = pd.to_datetime(results_df['tourney_date'])
results_df['Year_Month'] = results_df['tourney_date'].dt.to_period('M')
monthly_acc = results_df.groupby('Year_Month')['Correct'].mean()

plt.figure(figsize=(12, 6))
monthly_acc.plot(kind='line', marker='o')
plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall Accuracy ({accuracy:.3f})')
plt.axhline(y=baseline_accuracy, color='orange', linestyle='--', label=f'ATP Ranking Baseline ({baseline_accuracy:.3f})')
plt.title('Professional Tennis Neural Network - Monthly Accuracy Over Time')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   Results saved to: {output_dir}")

print("\n" + "=" * 60)
print("PROFESSIONAL TENNIS NEURAL NETWORK TRAINING COMPLETE")
print("=" * 60)
print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"ATP Ranking baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
print(f"Improvement: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
print(f"Results saved to: {output_dir}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test.values, y_pred, target_names=['Player2_Wins', 'Player1_Wins']))

# Save the corrected 143-feature model (replaces corrupted original)
print(f"\nSaving corrected 143-feature Neural Network model...")
original_model_path = os.path.join(output_dir, 'neural_network_model.pth')
original_scaler_path = os.path.join(output_dir, 'scaler.pkl')

torch.save(model.state_dict(), original_model_path)
with open(original_scaler_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"   Corrected Neural Network model saved to: {original_model_path}")
print(f"   Corrected scaler saved to: {original_scaler_path}")
print(f"   Features used: {len(feature_cols)} (full feature set)")
print(f"   This replaces the corrupted 143-feature model")