import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
import os

# Create output directory if it doesn't exist
base_output_dir = "/Users/zachdodson/Documents/betting-algo/data/output"
output_dir = os.path.join(base_output_dir, "XGBoost")
os.makedirs(output_dir, exist_ok=True)

# Load the same dataset as bet_heuristic.py
processed_data_path = "/Users/zachdodson/Documents/betting-algo/data/Old Data/processed_atp_data.csv"
df = pd.read_csv(processed_data_path, low_memory=False)
# Try to parse dates with mixed formats
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
print(f"Loaded processed ATP data with shape: {df.shape}")

# Create a copy to work with
df_randomized = df.copy()

# NOTE: The processed_atp_data.csv file should already be properly randomized,
# so we don't need to do the player swapping operation

# The processed_atp_data.csv file should already be properly formatted
import numpy as np
# Use a deterministic seed for any numpy operations
np.random.seed(42)

# Verify data format
print(f"Checking dataset format - Label distribution: {df_randomized['Label'].value_counts(normalize=True)}")

# Use the dataset directly - remove references to unused variables
df = df_randomized.copy()
# No need for orig_columns or swap_mask variables

# No special handling needed for processed_atp_data.csv
# It should already have the appropriate structure
print("Using processed ATP data, skipping randomization")

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

# Train-test split (same as Random Forest and bet_points.py: 2000-2019 for training, 2020-2025 for testing)
train_df = df[df['Date'] < '2023-01-01'].copy()
test_df = df[df['Date'] >= '2023-01-01'].copy()

# No physical attributes in this dataset, so no need to handle them specifically

# Fill remaining missing values with 0
X_train = train_df[features_to_keep].fillna(0)
y_train = train_df['Label']
X_test = test_df[features_to_keep].fillna(0)
y_test = test_df['Label']

# Convert boolean columns to numeric (TRUE/FALSE to 1/0)
bool_columns = X_train.select_dtypes(include=['bool']).columns
X_train[bool_columns] = X_train[bool_columns].astype(int)
X_test[bool_columns] = X_test[bool_columns].astype(int)

# Convert all columns to numeric, coercing errors to NaN, then fill with 0
for col in X_train.columns:
    X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
    X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

# Standardize
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])

print(f"Training set (2000-2022): {X_train.shape}")
print(f"Testing set (2023-2025): {X_test.shape}")

# Betting data
test_indices = X_test.index
test_betting_data = df.loc[test_indices, ['Player1', 'Player2', 'AvgW', 'AvgL', 'Date', 'Tournament']].copy()
test_betting_data['Label'] = y_test
test_betting_data = test_betting_data.dropna(subset=['AvgW', 'AvgL'])
print(f"Test matches with odds: {test_betting_data.shape[0]}")

# Convert data to DMatrix for XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Define improved parameters for XGBoost
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'eta': 0.05,           # Lower learning rate for better generalization
    'max_depth': 8,        # Less complex trees to avoid overfitting
    'min_child_weight': 2, # Increased to avoid overfitting
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'colsample_bylevel': 0.7, # Additional feature sampling at each level
    'gamma': 0.1,          # Minimum loss reduction for split
    'alpha': 0.1,          # L1 regularization
    'lambda': 1.0,         # L2 regularization
    'scale_pos_weight': 1, # For any potential class imbalance
    'seed': 42
}

# Train XGBoost with early stopping
num_boost_round = 1000     # More rounds, with early stopping
early_stopping_rounds = 50 # More patience
evals = [(dtrain, 'train'), (dtest, 'eval')]
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=50        # Print every 50 iterations
)
print(f"Best iteration: {bst.best_iteration}")

# Create an XGBClassifier for scikit-learn compatibility
model = xgb.XGBClassifier(
    n_estimators=bst.best_iteration,
    learning_rate=0.05,
    max_depth=8,
    min_child_weight=2,
    subsample=0.8,
    colsample_bytree=0.8,
    colsample_bylevel=0.7,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nMetrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nROC AUC: {roc_auc:.4f}")

# Feature Importance Plot
importance = model.feature_importances_
perm_importance_df = pd.DataFrame({
    'Feature': features_to_keep,
    'Importance': importance
})
perm_importance_df = perm_importance_df.sort_values(by='Importance', ascending=False)

# Show top 20 features only for better readability
top_features = perm_importance_df.head(20)
plt.figure(figsize=(12, 10))
sns.barplot(x='Importance', y='Feature', data=top_features)
plt.title('Top 20 Features by Importance (XGBoost)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig(f'{output_dir}/feature_importance_plot_xgb.png')
plt.close()
print("Saved feature importance plot to 'feature_importance_plot_xgb.png'")

# Betting simulation
X_test_betting = X_test.loc[test_betting_data.index]
test_betting_data['Predicted_Prob'] = model.predict_proba(X_test_betting)[:, 1]
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
initial_bankroll = 100000  # Changed to match RF/bet_points.py
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
    bet_history_df.to_csv(f"{output_dir}/xgb_bet_history.csv", index=False)
    print(f"Saved {len(bet_history)} bets to 'xgb_bet_history.csv'")

# Plot bankroll
plt.figure(figsize=(12, 7))
plt.plot(bankroll_history, color='blue', linewidth=2)
plt.xlabel('Matches Processed (Chronological)')
plt.ylabel('Bankroll ($)')
plt.title('XGBoost Model with Half-Kelly Betting: Bankroll Over Time')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/bankroll_plot_xgb.png')
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

print(f"\nBetting Results (XGBoost):")
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
plt.title('XGBoost Calibration vs. Betting Market')
plt.savefig(f'{output_dir}/xgb_calibration.png')
plt.close()

print("\nAll outputs saved to:", output_dir)