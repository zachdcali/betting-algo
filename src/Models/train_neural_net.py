import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
import seaborn as sns
import os
from datetime import datetime
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")
pd.options.mode.chained_assignment = None  # default='warn'

# Create output directory if it doesn't exist
base_output_dir = "/Users/zachdodson/Documents/betting-algo/data/output"
output_dir = os.path.join(base_output_dir, "Neural Network")
os.makedirs(output_dir, exist_ok=True)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load the same dataset as bet_heuristic.py
processed_data_path = "/Users/zachdodson/Documents/betting-algo/data/Old Data/processed_atp_data.csv"
df = pd.read_csv(processed_data_path, low_memory=False)
# Try to parse dates with mixed formats
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
print(f"Loaded processed ATP data with shape: {df.shape}")

# Create a copy to work with
df_randomized = df.copy()

# The processed_atp_data.csv file should already be properly formatted
import numpy as np
# Use a deterministic seed for any numpy operations
np.random.seed(42)

# Verify data format
print(f"Checking dataset format - Label distribution: {df_randomized['Label'].value_counts(normalize=True)}")

# Use the dataset directly
df = df_randomized.copy()

# No additional swapping needed for processed_atp_data.csv
# It should already have the appropriate structure
print("Using processed ATP data, skipping player order randomization")

# Verify the label distribution one more time
print(f"Final label distribution: {df['Label'].value_counts(normalize=True)}")

# Features adjusted based on available columns in processed_atp_data.csv
features_to_keep = [
    # Basic features
    'P1Rank', 'P2Rank', 'P1Pts', 'P2Pts',
    'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
    'Round_2nd Round', 'Round_3rd Round', 'Round_4th Round', 'Round_Quarterfinals',
    'Court_Outdoor',
    'Series_ATP250', 'Series_Grand Slam', 'Series_International', 'Series_International Gold', 'Series_Masters', 'Series_Masters Cup',
    
    # H2H features
    'H2H_P1_Wins', 'H2H_Total', 'H2H_P1_WinRate', 'H2H_P1_GamesWonRatio', 'H2H_P1_SetsWonRatio',
    
    # Fatigue and form features
    'P1_Matches_7Days', 'P2_Matches_7Days', 'P1_Matches_14Days', 'P2_Matches_14Days',
    'P1_Games_7Days', 'P2_Games_7Days',
    'P1_WinLoss_7Days', 'P2_WinLoss_7Days',
    'P1_GamesWonRatio_7Days', 'P2_GamesWonRatio_7Days', 'P1_GamesWonRatio_14Days', 'P2_GamesWonRatio_14Days',
    'P1_GamesPlayed_7Days', 'P2_GamesPlayed_7Days', 'P1_GamesPlayed_14Days', 'P2_GamesPlayed_14Days',
    'P1_SetsPlayed_7Days', 'P2_SetsPlayed_7Days', 'P1_SetsPlayed_14Days', 'P2_SetsPlayed_14Days',
    'P1_Days_Since_Last', 'P2_Days_Since_Last',
    
    # Log-transformed features
    'Log_P1Rank', 'Log_P2Rank'
]
numerical_features = [f for f in features_to_keep if 'Series_' not in f and 'Surface_' not in f and 'Round_' not in f and 'Court_' not in f]

# Train-test split with explicit date range
start_date = '2000-01-01'  # Set the start date for training data (adjust as needed)
split_date = '2023-01-01'  # Date to split train/test

# Create train/test sets
train_df = df[(df['Date'] >= start_date) & (df['Date'] < split_date)].copy()
test_df = df[df['Date'] >= split_date].copy()

print(f"Using data from {start_date} to present")

# No physical attributes in this dataset, so no need to handle them specifically

# Fill remaining missing values with 0
X_train = train_df[features_to_keep].fillna(0)
y_train = train_df['Label']
X_test = test_df[features_to_keep].fillna(0)
y_test = test_df['Label']

# Convert boolean columns to numeric
bool_columns = X_train.select_dtypes(include=['bool']).columns
X_train[bool_columns] = X_train[bool_columns].astype(int)
X_test[bool_columns] = X_test[bool_columns].astype(int)

# Convert all columns to numeric
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

# Standardize
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"Training set (2000-2012): {X_train.shape}")
print(f"Testing set (2023-2025): {X_test.shape}")
print(f"Total number of features: {X_train.shape[1]}")

# Betting data
test_indices = X_test.index
test_betting_data = df.loc[test_indices, ['Player1', 'Player2', 'AvgW', 'AvgL', 'Date', 'Tournament']].copy()
test_betting_data['Label'] = y_test
test_betting_data = test_betting_data.dropna(subset=['AvgW', 'AvgL'])
print(f"Test matches with odds: {test_betting_data.shape[0]}")

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train.values).to(device)
y_train_tensor = torch.FloatTensor(y_train.values).to(device)
X_test_tensor = torch.FloatTensor(X_test.values).to(device)
y_test_tensor = torch.FloatTensor(y_test.values).to(device)

# DataLoaders - Increased batch size for faster training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Define the neural network model with two hidden layers (32 and 16 neurons)
class TennisPredictor(nn.Module):
    def __init__(self, input_size):
        super(TennisPredictor, self).__init__()
        # First hidden layer with 32 neurons
        self.layer1 = nn.Linear(input_size, 32)
        self.batch_norm1 = nn.BatchNorm1d(32)
        self.dropout1 = nn.Dropout(0.4)
        
        # Second hidden layer with 16 neurons
        self.layer2 = nn.Linear(32, 16)
        self.batch_norm2 = nn.BatchNorm1d(16)
        self.dropout2 = nn.Dropout(0.3)
        
        # Output layer
        self.layer3 = nn.Linear(16, 1)
        self.leaky_relu = nn.LeakyReLU(0.1)  # LeakyReLU activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # First hidden layer
        x = self.leaky_relu(self.batch_norm1(self.layer1(x)))
        x = self.dropout1(x)
        
        # Second hidden layer
        x = self.leaky_relu(self.batch_norm2(self.layer2(x)))
        x = self.dropout2(x)
        
        # Output layer
        x = self.sigmoid(self.layer3(x))
        return x

# Create model
model = TennisPredictor(input_size=X_train.shape[1]).to(device)

# Calculate and print the total number of parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total model parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")
print(f"Model architecture:")
print(f"  Input layer: {X_train.shape[1]} features")
print(f"  Hidden layer 1: 32 neurons with BatchNorm and Dropout(0.4)")
print(f"  Hidden layer 2: 16 neurons with BatchNorm and Dropout(0.3)")
print(f"  Output layer: 1 neuron with Sigmoid activation")

criterion = nn.BCELoss()
# More conservative learning rate for better convergence
learning_rate = 0.0005
weight_decay = 1e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# More sophisticated learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

# Print model architecture summary
def print_model_summary():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    print("\n" + "="*50)
    print(f"TENNIS NEURAL NETWORK MODEL SUMMARY - {timestamp}")
    print("="*50)
    print(f"Input Features: {X_train.shape[1]}")
    print(f"Training Samples: {X_train.shape[0]}")
    print(f"Testing Samples: {X_test.shape[0]}")
    print("\nArchitecture:")
    print(f"- Input Layer: {X_train.shape[1]} features")
    print(f"- Hidden Layer 1: 32 neurons with BatchNorm and Dropout(0.4)")
    print(f"- Hidden Layer 2: 16 neurons with BatchNorm and Dropout(0.3)")
    print(f"- Output Layer: 1 neuron with Sigmoid activation")
    
    print("\nTraining Parameters:")
    print(f"- Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})")
    print(f"- Learning Rate Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)")
    print(f"- Loss Function: Binary Cross Entropy")
    print(f"- Batch Size: 256")
    print(f"- Early Stopping Patience: 15 epochs")
    print(f"- Max Epochs: 100")
    
    # Feature groups summary
    feature_groups = {
        "Basic": ['P1Rank', 'P2Rank', 'P1Pts', 'P2Pts', 'Surface_Clay', 'Surface_Grass', 'Surface_Hard',
                 'Round_2nd Round', 'Round_3rd Round', 'Round_4th Round', 'Round_Quarterfinals',
                 'Court_Outdoor', 'Series_ATP250', 'Series_Grand Slam', 'Series_International', 
                 'Series_International Gold', 'Series_Masters', 'Series_Masters Cup'],
        "Physical": ['P1_Hand_R', 'P1_Hand_L', 'P2_Hand_R', 'P2_Hand_L', 'Matchup_RR', 'Matchup_RL', 
                    'Matchup_LR', 'Matchup_LL', 'P1_Height_cm', 'P2_Height_cm', 'Height_Diff_cm',
                    'P1_Age', 'P2_Age', 'Age_Diff_Years'],
        "H2H": ['H2H_P1_Wins', 'H2H_Total', 'H2H_P1_WinRate', 'H2H_P1_GamesWonRatio', 'H2H_P1_SetsWonRatio'],
        "Fatigue/Form": ['P1_Matches_7Days', 'P2_Matches_7Days', 'P1_Matches_14Days', 'P2_Matches_14Days',
                        'P1_Games_7Days', 'P2_Games_7Days', 'P1_WinLoss_7Days', 'P2_WinLoss_7Days',
                        'P1_GamesWonRatio_7Days', 'P2_GamesWonRatio_7Days', 'P1_GamesWonRatio_14Days', 
                        'P2_GamesWonRatio_14Days', 'P1_GamesPlayed_7Days', 'P2_GamesPlayed_7Days', 
                        'P1_GamesPlayed_14Days', 'P2_GamesPlayed_14Days', 'P1_SetsPlayed_7Days', 
                        'P2_SetsPlayed_7Days', 'P1_SetsPlayed_14Days', 'P2_SetsPlayed_14Days',
                        'P1_Days_Since_Last', 'P2_Days_Since_Last']
    }
    
    print("\nFeature Categories:")
    for group, features in feature_groups.items():
        count = sum(1 for f in features if f in X_train.columns)
        print(f"- {group}: {count} features")
    
    print("="*50)
    
    # Also save this summary to a file
    with open(f"{output_dir}/model_summary_{timestamp}.txt", "w") as f:
        f.write(f"TENNIS NEURAL NETWORK MODEL SUMMARY - {timestamp}\n")
        f.write("="*50 + "\n")
        f.write(f"Input Features: {X_train.shape[1]}\n")
        f.write(f"Training Samples: {X_train.shape[0]}\n")
        f.write(f"Testing Samples: {X_test.shape[0]}\n")
        f.write("\nArchitecture:\n")
        f.write(f"- Input Layer: {X_train.shape[1]} features\n")
        f.write(f"- Hidden Layer 1: 32 neurons with BatchNorm and Dropout(0.4)\n")
        f.write(f"- Hidden Layer 2: 16 neurons with BatchNorm and Dropout(0.3)\n")
        f.write(f"- Output Layer: 1 neuron with Sigmoid activation\n")
        
        f.write("\nTraining Parameters:\n")
        f.write(f"- Optimizer: Adam(lr={learning_rate}, weight_decay={weight_decay})\n")
        f.write(f"- Learning Rate Scheduler: ReduceLROnPlateau(patience=5, factor=0.5)\n")
        f.write(f"- Loss Function: Binary Cross Entropy\n")
        f.write(f"- Batch Size: 256\n")
        f.write(f"- Early Stopping Patience: 15 epochs\n")
        f.write(f"- Max Epochs: 100\n")
        
        f.write("\nFeature Categories:\n")
        for group, features in feature_groups.items():
            count = sum(1 for f in features if f in X_train.columns)
            f.write(f"- {group}: {count} features\n")
            
        f.write("\nFeature List:\n")
        for feature in X_train.columns:
            f.write(f"- {feature}\n")
        
        f.write("="*50 + "\n")

# Print model summary before training
print_model_summary()

# Training with more epochs for better convergence
num_epochs = 100  # Full training run
train_losses, test_losses = [], []
best_test_loss = float('inf')
patience = 15  # Increased patience for better chance at finding optimal model
patience_counter = 0
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_losses.append(train_loss / len(train_loader))

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            test_loss += loss.item()
    test_losses.append(test_loss / len(test_loader))
    
    # Learning rate scheduling for ReduceLROnPlateau
    scheduler.step(test_losses[-1])
    
    # Print progress every 5 epochs
    print(f"Epoch {epoch}, Train Loss: {train_losses[-1]:.4f}, Test Loss: {test_losses[-1]:.4f}")
    
    if test_losses[-1] < best_test_loss:
        best_test_loss = test_losses[-1]
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

model.load_state_dict(best_model_state)

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/loss_plot_nn.png')
plt.close()

# Evaluation
model.eval()
with torch.no_grad():
    y_pred_prob = model(X_test_tensor).squeeze().cpu().numpy()
    y_pred = (y_pred_prob > 0.5).astype(int)
    y_true = y_test_tensor.cpu().numpy()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
roc_auc = roc_auc_score(y_true, y_pred_prob)

print(f"\nMetrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nROC AUC: {roc_auc:.4f}")

# Betting simulation
X_test_betting = X_test.loc[test_betting_data.index]
with torch.no_grad():
    X_test_betting_tensor = torch.FloatTensor(X_test_betting.values).to(device)
    test_betting_data['Predicted_Prob'] = model(X_test_betting_tensor).squeeze().cpu().numpy()

test_betting_data['Predicted'] = (test_betting_data['Predicted_Prob'] > 0.5).astype(int)
test_betting_data['Implied_Prob_P1'] = 1 / test_betting_data['AvgW']
test_betting_data['Implied_Prob_P2'] = 1 / test_betting_data['AvgL']
total_implied_prob = test_betting_data['Implied_Prob_P1'] + test_betting_data['Implied_Prob_P2']
test_betting_data['Implied_Prob_P1'] /= total_implied_prob
test_betting_data['Implied_Prob_P2'] /= total_implied_prob
test_betting_data['Edge'] = test_betting_data['Predicted_Prob'] - test_betting_data['Implied_Prob_P1']
test_betting_data['Kelly_Fraction'] = np.where(
    test_betting_data['Predicted_Prob'] > test_betting_data['Implied_Prob_P1'],
    ((test_betting_data['Predicted_Prob'] - test_betting_data['Implied_Prob_P1']) / 
     (test_betting_data['Implied_Prob_P1'] * (1 - test_betting_data['Predicted_Prob']))),
    0
)

# Print prediction statistics
print("\nPrediction Statistics:")
print(f"Average predicted probability: {test_betting_data['Predicted_Prob'].mean():.4f}")
print(f"Max edge found: {test_betting_data['Edge'].max():.4f}")
print(f"Min edge found: {test_betting_data['Edge'].min():.4f}")
print(f"Max Kelly fraction: {test_betting_data['Kelly_Fraction'].max():.4f}")
print(f"Matches with positive edge: {(test_betting_data['Edge'] > 0).sum()}")
print(f"Average edge on positive edge bets: {test_betting_data.loc[test_betting_data['Edge'] > 0, 'Edge'].mean():.4f}")

# Initialize betting simulation
initial_bankroll = 100000  # Match RF and XGBoost
min_kelly_fraction = 0.01  # Lower threshold to get more bets
max_bet_fraction = 0.01    # Conservative bet sizing
min_edge = 0.02            # Minimum edge for betting
bankroll = initial_bankroll
bankroll_history = [bankroll]
num_bets = 0
cumulative_bets = [0]
total_bets = 0
total_winnings = 0
wins = 0
losses = 0

# Qualifying bets
qualifying_bets = test_betting_data[(test_betting_data['Kelly_Fraction'] >= min_kelly_fraction) & 
                                   (test_betting_data['Edge'] >= min_edge)]
print(f"\nNumber of qualifying bets with current thresholds: {len(qualifying_bets)}")

# Prepare bet history
bet_history = []

# Simulation
test_betting_data = test_betting_data.sort_values('Date')
for idx, row in test_betting_data.iterrows():
    if bankroll <= 0:
        print(f"Bankroll depleted at match {idx}")
        break
    
    # Only bet when criteria are met
    if row['Kelly_Fraction'] >= min_kelly_fraction and row['Edge'] >= min_edge:
        # Use Half Kelly criterion - bet size is half of the Kelly optimal fraction
        kelly_fraction = row['Kelly_Fraction']
        half_kelly = 0.5 * kelly_fraction  # Half Kelly
        
        # Cap at 5% of current bankroll
        bet_size = min(half_kelly * bankroll, 0.05 * bankroll)
        if bet_size > bankroll:
            bet_size = bankroll
        
        num_bets += 1
        total_bets += bet_size
        bankroll -= bet_size
        
        is_win = (row['Predicted'] == 1 and row['Label'] == 1) or (row['Predicted'] == 0 and row['Label'] == 0)
        bet_outcome = "WIN" if is_win else "LOSS"
        
        # Update win/loss counter
        if is_win:
            wins += 1
            if row['Predicted'] == 1:
                winnings = bet_size * (row['AvgW'] - 1)
            else:
                winnings = bet_size * (row['AvgL'] - 1)
            total_winnings += winnings
            bankroll += bet_size + winnings
        else:
            losses += 1
            winnings = -bet_size
        
        # Record bet details
        bet_record = {
            'Date': row['Date'],
            'Tournament': row['Tournament'],
            'Player1': row['Player1'],
            'Player2': row['Player2'],
            'Odds_P1': row['AvgW'],
            'Odds_P2': row['AvgL'],
            'Predicted_Prob': row['Predicted_Prob'],
            'Implied_Prob': row['Implied_Prob_P1'],
            'Edge': row['Edge'],
            'Kelly_Fraction': row['Kelly_Fraction'],
            'Bet_Size': bet_size,
            'Bet_On': 'Player1' if row['Predicted'] == 1 else 'Player2',
            'Outcome': bet_outcome,
            'Profit': winnings,
            'Current_Bankroll': bankroll
        }
        bet_history.append(bet_record)
    
    bankroll_history.append(bankroll)
    cumulative_bets.append(num_bets)

# Save bet history
if bet_history:
    bet_history_df = pd.DataFrame(bet_history)
    bet_history_df.to_csv(f"{output_dir}/nn_bet_history.csv", index=False)
    print(f"Saved {len(bet_history)} bets to 'nn_bet_history.csv'")

# Plot bankroll
plt.figure(figsize=(12, 7))
plt.plot(bankroll_history, color='blue', linewidth=2)
plt.xlabel('Matches Processed (Chronological)')
plt.ylabel('Bankroll ($)')
plt.title('Neural Network Model with Half-Kelly Betting: Bankroll Over Time')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/bankroll_plot_nn.png')
plt.close()

# Calculate performance metrics
profit = bankroll - initial_bankroll
roi = profit / total_bets if total_bets > 0 else 0  # Corrected ROI formula (profit divided by amount wagered)
profit_percentage = (profit / initial_bankroll) * 100
win_rate = wins / num_bets if num_bets > 0 else 0

# Calculate additional betting metrics
avg_bet_size = total_bets / num_bets if num_bets > 0 else 0
avg_win = total_winnings / wins if wins > 0 else 0
avg_loss = total_bets / losses if losses > 0 else 0
win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else 0

# Calculate Profit Factor
profit_factor = total_winnings / (total_bets - total_winnings) if (total_bets - total_winnings) > 0 else float('inf')

# Calculate max drawdown
max_bankroll = initial_bankroll
max_drawdown = 0
for b in bankroll_history:
    if b > max_bankroll:
        max_bankroll = b
    current_drawdown = (max_bankroll - b) / max_bankroll if max_bankroll > 0 else 0
    if current_drawdown > max_drawdown:
        max_drawdown = current_drawdown

# Calculate Volatility and Sharpe Ratio if we have bet results
if bet_history:
    returns = [record['Profit'] / record['Bet_Size'] for record in bet_history if record['Bet_Size'] > 0]
    volatility = np.std(returns) if len(returns) > 1 else 0
    avg_return = np.mean(returns) if returns else 0
    sharpe_ratio = avg_return / volatility if volatility > 0 else 0
    
    # Calculate Sortino Ratio (downside deviation only)
    negative_returns = [r for r in returns if r < 0]
    downside_deviation = np.std(negative_returns) if len(negative_returns) > 1 else 0
    sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Calmar Ratio
    calmar_ratio = abs(roi / max_drawdown) if max_drawdown > 0 else None
    
    # Calculate Ulcer Index
    rolling_max = np.maximum.accumulate(bankroll_history)
    drawdowns = np.zeros_like(bankroll_history)
    for i in range(len(bankroll_history)):
        if rolling_max[i] > 0:
            drawdowns[i] = (rolling_max[i] - bankroll_history[i]) / rolling_max[i]
    ulcer_index = np.sqrt(np.mean(np.square(drawdowns)))
    
    # Risk of Ruin (approximate)
    risk_of_ruin = np.exp(-2 * initial_bankroll * abs(roi) / (avg_bet_size * volatility)) if volatility > 0 else 0
    risk_of_ruin = min(risk_of_ruin, 1.0)  # Cap at 100%
    
    # Z-Score (statistical significance)
    z_score = (roi * np.sqrt(num_bets)) / volatility if volatility > 0 and num_bets > 30 else None
else:
    volatility = 0
    sharpe_ratio = 0
    sortino_ratio = 0
    calmar_ratio = 0
    ulcer_index = 0
    risk_of_ruin = 0
    z_score = None

print(f"\nBetting Results (Neural Network):")
print(f"Total Bets: ${total_bets:.2f}")
print(f"Total Winnings: ${total_winnings:.2f}")
print(f"Bets Placed: {num_bets}")
print(f"Wins: {wins}, Losses: {losses}")
print(f"Final Bankroll: ${bankroll:.2f}")
print(f"Profit: ${profit:.2f} ({profit_percentage:.2f}%)")
print(f"ROI: {roi:.4f}")
print(f"Win Rate: {win_rate:.4f}")

print("\nAdvanced Performance Metrics:")
print(f"Average Bet Size: ${avg_bet_size:.2f}")
print(f"Win/Loss Ratio: {win_loss_ratio:.4f}")
print(f"Average Win: ${avg_win:.2f}")
print(f"Average Loss: ${avg_loss:.2f}")
print(f"Profit Factor: {profit_factor:.4f}")
print(f"Max Drawdown: {max_drawdown:.4f}")
print(f"Volatility: {volatility:.4f}")
print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
print(f"Sortino Ratio: {sortino_ratio:.4f}")
print(f"Calmar Ratio: {calmar_ratio if calmar_ratio is not None else 'N/A'}")
print(f"Ulcer Index: {ulcer_index:.4f}")
print(f"Risk of Ruin: {risk_of_ruin:.6f}")
print(f"Z-Score: {z_score if z_score is not None else 'N/A'}")

# Calibration analysis
print("\n========== MODEL CALIBRATION ANALYSIS ==========")
test_betting_data['Implied_Prob_Bin'] = pd.cut(test_betting_data['Implied_Prob_P1'], bins=10)

calibration_analysis = test_betting_data.groupby('Implied_Prob_Bin', observed=True).agg({
    'Label': lambda x: (x == test_betting_data.loc[x.index, 'Predicted']).mean(),
    'Implied_Prob_P1': 'mean',
    'Predicted_Prob': 'mean',
    'Date': 'count'
}).rename(columns={'Label': 'Accuracy', 'Date': 'Count'})

print(calibration_analysis)

plt.figure(figsize=(10, 6))
plt.plot(calibration_analysis['Implied_Prob_P1'], calibration_analysis['Accuracy'], 'o-', label='Model Accuracy')
plt.plot(calibration_analysis['Implied_Prob_P1'], calibration_analysis['Implied_Prob_P1'], '--', label='Perfect Calibration')
plt.plot(calibration_analysis['Implied_Prob_P1'], calibration_analysis['Predicted_Prob'], 'o-', label='Model Predicted Prob')
plt.xlabel('Implied Probability from Odds')
plt.ylabel('Actual Accuracy / Predicted Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Neural Network Calibration vs. Betting Market')
plt.savefig(f'{output_dir}/nn_calibration.png')
plt.close()

# Calculate and plot feature importances using manual permutation importance
print("\n========== FEATURE IMPORTANCE ANALYSIS ==========")
print("Calculating feature importance using permutation method (this may take a few minutes)...")

# Baseline performance
model.eval()
with torch.no_grad():
    baseline_preds = model(X_test_tensor).cpu().squeeze().numpy()
    baseline_score = roc_auc_score(y_test.values, baseline_preds)
    print(f"Baseline score (AUC): {baseline_score:.4f}")

# Calculate feature importance manually
feature_importance = {}
X_test_np = X_test.values.copy()

for i, feature_name in enumerate(X_test.columns):
    # Store original values
    orig_values = X_test_np[:, i].copy()
    
    # Shuffle the feature values
    np.random.seed(42)
    X_test_np[:, i] = np.random.permutation(orig_values)
    
    # Create tensor and get predictions
    permuted_tensor = torch.FloatTensor(X_test_np).to(device)
    with torch.no_grad():
        permuted_preds = model(permuted_tensor).cpu().squeeze().numpy()
        permuted_score = roc_auc_score(y_test.values, permuted_preds)
    
    # Importance is the decrease in performance
    importance = baseline_score - permuted_score
    feature_importance[feature_name] = importance
    
    # Restore original values
    X_test_np[:, i] = orig_values
    
    # Progress indicator every 10 features
    if (i + 1) % 10 == 0 or i == len(X_test.columns) - 1:
        print(f"Processed {i+1}/{len(X_test.columns)} features")
    
# Sort features by importance
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
importances = np.array([imp for _, imp in sorted_importance])
feature_names = [name for name, _ in sorted_importance]

# Create a DataFrame for easier visualization
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Save feature importance data
feature_importance_df.to_csv(f"{output_dir}/feature_importance_nn.csv", index=False)
print(f"Saved feature importance data to 'feature_importance_nn.csv'")

# Plot feature importance
plt.figure(figsize=(12, 10))
plt.barh(range(len(importances)), importances, align='center')
plt.yticks(range(len(importances)), feature_names)
plt.xlabel('Feature Importance (decrease in AUC when permuted)')
plt.title('Neural Network - Feature Importance (Permutation Method)')
plt.tight_layout()
plt.savefig(f'{output_dir}/feature_importance_nn.png')
plt.close()

# Display top 15 most important features
print("\nTop 15 Most Important Features:")
for i in range(min(15, len(feature_importance_df))):
    print(f"{feature_importance_df.iloc[i]['Feature']}: {feature_importance_df.iloc[i]['Importance']:.4f}")

print("\nAll outputs saved to:", output_dir)