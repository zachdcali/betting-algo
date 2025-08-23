import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Add the src directory to the path to import our preprocessing function
sys.path.append('/app/src')
from preprocess_tennis_data_ml import get_feature_columns

# Create output directory
base_output_dir = "/app/data/output"
output_dir = os.path.join(base_output_dir, "Neural_Network")
os.makedirs(output_dir, exist_ok=True)

# Set device (use CPU in Docker)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class TennisNet(nn.Module):
    """
    Neural Network for tennis match prediction
    """
    def __init__(self, input_size):
        super(TennisNet, self).__init__()
        self.network = nn.Sequential(
            # Input layer
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layer 1
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layer 2
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x)

def train_model(model, train_loader, val_loader, num_epochs=100, learning_rate=0.001):
    """
    Train the neural network model
    """
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
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
        
        if epoch % 10 == 0:
            print(f'   Epoch {epoch:3d}/{num_epochs}: Train Loss: {train_loss_avg:.4f}, Val Loss: {val_loss_avg:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_losses, train_accuracies, val_accuracies

print("=" * 60)
print("NEURAL NETWORK MODEL TRAINING")
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

# Check for missing values and fill
print("\n3. Checking data quality...")
train_missing = X_train.isnull().sum().sum()
test_missing = X_test.isnull().sum().sum()

if train_missing > 0 or test_missing > 0:
    print("   Filling missing values...")
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_train.median())  # Use training median for test set

print(f"   Training missing values: {X_train.isnull().sum().sum()}")
print(f"   Test missing values: {X_test.isnull().sum().sum()}")

# Standardize features
print("\n4. Standardizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)

print(f"   Feature scaling complete")

# Create PyTorch datasets
print("\n5. Creating PyTorch datasets...")
train_dataset = TensorDataset(
    torch.FloatTensor(X_train_scaled),
    torch.FloatTensor(y_train.values)
)
test_dataset = TensorDataset(
    torch.FloatTensor(X_test_scaled),
    torch.FloatTensor(y_test.values)
)

# Create data loaders
batch_size = 512
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"   Batch size: {batch_size}")
print(f"   Train batches: {len(train_loader)}")
print(f"   Test batches: {len(test_loader)}")

# Create and train model
print("\n6. Training Neural Network model...")
input_size = X_train_scaled.shape[1]
model = TennisNet(input_size).to(device)

print(f"   Model architecture:")
print(f"   - Input size: {input_size}")
print(f"   - Hidden layers: 64 -> 32 -> 16")
print(f"   - Output size: 1 (sigmoid)")
print(f"   - Dropout: 0.3, 0.3, 0.2")

# Train the model
num_epochs = 100
learning_rate = 0.001

train_losses, val_losses, train_accuracies, val_accuracies = train_model(
    model, train_loader, test_loader, num_epochs, learning_rate
)

print("   ✅ Model training complete!")

# Make predictions
print("\n7. Making predictions...")
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
print("NEURAL NETWORK RESULTS")
print("=" * 60)
print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print(f"AUC-ROC: {auc:.4f}")

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
        print(f"   Neural Network vs ATP Ranking: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
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
results_df['NN_Prediction'] = y_pred
results_df['NN_Probability'] = y_pred_proba
results_df['Correct'] = (y_pred == y_test.values).astype(int)
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
            print(f"{year}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
        else:
            print(f"{year}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

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
                print(f"{group_name:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
            else:
                print(f"{group_name:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

if 'Surface' in test_df.columns:
    print("\n=== SURFACE BREAKDOWN ===")
    surfaces = [s for s in test_df['Surface'].unique() if pd.notna(s)]
    for surface in sorted(surfaces):
        surface_data = results_df[test_df['Surface'] == surface]
        model_acc, baseline_acc, count = calculate_subset_metrics(surface_data)
        if model_acc is not None and count > 100:  # Only show if substantial sample
            if baseline_acc is not None:
                improvement = (model_acc - baseline_acc) * 100
                print(f"{surface:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | ATP {baseline_acc:.3f} ({baseline_acc*100:.1f}%) | Edge: {improvement:+.1f}pp | N: {count:,}")
            else:
                print(f"{surface:12s}: NN {model_acc:.3f} ({model_acc*100:.1f}%) | N: {count:,}")

# Save detailed results
print("\n10. Saving results...")

# Save predictions (reuse the detailed results_df from analysis)
results_df.to_csv(os.path.join(output_dir, 'predictions.csv'), index=False)

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

# Training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Neural Network - Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(train_accuracies, label='Training Accuracy')
plt.plot(val_accuracies, label='Validation Accuracy')
plt.title('Neural Network - Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 3, 3)
plt.hist(y_pred_proba[y_test.values == 1], bins=50, alpha=0.7, label='Player1 Won', density=True)
plt.hist(y_pred_proba[y_test.values == 0], bins=50, alpha=0.7, label='Player1 Lost', density=True)
plt.title('Neural Network - Prediction Probabilities')
plt.xlabel('Predicted Probability')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'training_analysis.png'), dpi=300, bbox_inches='tight')
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
plt.title('Neural Network - Monthly Accuracy Over Time')
plt.xlabel('Month')
plt.ylabel('Accuracy')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'monthly_accuracy.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"   Results saved to: {output_dir}")

print("\n" + "=" * 60)
print("NEURAL NETWORK TRAINING COMPLETE")
print("=" * 60)
print(f"Final accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
if baseline_accuracy is not None:
    print(f"ATP Ranking baseline: {baseline_accuracy:.4f} ({baseline_accuracy*100:.2f}%)")
    print(f"Improvement: {(accuracy - baseline_accuracy)*100:+.2f} percentage points")
print(f"Results saved to: {output_dir}")

# Print classification report
print("\nDetailed Classification Report:")
print(classification_report(y_test.values, y_pred, target_names=['Player2_Wins', 'Player1_Wins']))

