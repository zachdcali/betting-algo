import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import os

# Create output directory if it doesn't exist
base_output_dir = "/Users/zachdodson/Documents/betting-algo/data/output"
output_dir = os.path.join(base_output_dir, "Random Forest")
os.makedirs(output_dir, exist_ok=True)

# Load the same dataset as bet_heuristic.py
processed_data_path = "/Users/zachdodson/Documents/betting-algo/data/Old Data/processed_atp_data.csv"
df = pd.read_csv(processed_data_path, low_memory=False)
# Try to parse dates with mixed formats
df['Date'] = pd.to_datetime(df['Date'], format='mixed', dayfirst=False)
print(f"Loaded processed ATP data with shape: {df.shape}")

# Unlike before, we don't need to filter for UTR or randomize players
# The data is already filtered and randomized

# Features - Removed derived features and added UTR features
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
    
    # Log-transformed features (if available)
    'Log_P1Rank', 'Log_P2Rank'
]
numerical_features = [f for f in features_to_keep if 'Series_' not in f and 'Surface_' not in f and 'Round_' not in f and 'Court_' not in f]

# Train-test split (same as bet_points.py: 2000-2022 for training, 2023-2025 for testing)
train_df = df[df['Date'] < '2023-01-01'].copy()
test_df = df[df['Date'] >= '2023-01-01'].copy()

# Handle missing values for physical attributes specifically
for col in ['P1_Height_cm', 'P2_Height_cm', 'Height_Diff_cm', 'P1_Age', 'P2_Age', 'Age_Diff_Years']:
    # Fill with median for numerical physical attributes
    if col in train_df.columns:
        median_val = train_df[col].median()
        train_df[col] = train_df[col].fillna(median_val)
        test_df[col] = test_df[col].fillna(median_val)
        print(f"Filled missing values in {col} with median: {median_val}")

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

print(f"Training set (2000-01-01–2022-12-31): {X_train.shape}")
print(f"Testing set (2023-01-01–2025-04-05): {X_test.shape}")

# Betting data - ensuring we use properly randomized odds
test_indices = X_test.index
test_betting_data = df.loc[test_indices, ['Player1', 'Player2', 'AvgW', 'AvgL', 'Date', 'Tournament']].copy()
test_betting_data['Label'] = y_test
test_betting_data = test_betting_data.dropna(subset=['AvgW', 'AvgL'])
print(f"Test matches with odds: {test_betting_data.shape[0]}")

# Train Random Forest model
model = RandomForestClassifier(
    n_estimators=200,  # Increased from 100
    max_depth=12,      # Increased from 10
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',  # Use square root of the number of features
    class_weight='balanced',  # Account for any class imbalance
    random_state=42,
    n_jobs=-1  # Use all available cores
)
model.fit(X_train, y_train)

# First, calculate baseline heuristics for comparison
# Skip UTR heuristic as it's not in the dataset

# Rank Heuristic (better/lower rank wins)
rank_pred = (test_df['P1Rank'] < test_df['P2Rank']).astype(int)
rank_accuracy = accuracy_score(test_df['Label'], rank_pred)
rank_auc = roc_auc_score(test_df['Label'], -1 * (test_df['P1Rank'] - test_df['P2Rank']))

# Betting odds heuristic (betting favorite wins)
odds_test_df = test_df.dropna(subset=['AvgW', 'AvgL'])
p1_implied_prob = 1 / odds_test_df['AvgW']
p2_implied_prob = 1 / odds_test_df['AvgL']
total_prob = p1_implied_prob + p2_implied_prob
p1_normalized = p1_implied_prob / total_prob
odds_pred = (p1_normalized > 0.5).astype(int)
odds_accuracy = accuracy_score(odds_test_df['Label'], odds_pred)
odds_auc = roc_auc_score(odds_test_df['Label'], p1_normalized)

print("\nBaseline Heuristics Performance (Test Set):")
print(f"Rank Heuristic: Accuracy: {rank_accuracy:.4f}, AUC: {rank_auc:.4f}")
print(f"Odds Heuristic: Accuracy: {odds_accuracy:.4f}, AUC: {odds_auc:.4f}")

# Evaluation of our model
y_pred = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_prob)

print(f"\nRandom Forest Model Metrics:\nAccuracy: {accuracy:.4f}\nPrecision: {precision:.4f}\nRecall: {recall:.4f}\nF1-Score: {f1:.4f}\nROC AUC: {roc_auc:.4f}")

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
plt.title('Top 20 Features by Importance (Random Forest)')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.savefig(f'{output_dir}/feature_importance_plot_rf.png')
plt.close()
print("Saved feature importance plot to 'feature_importance_plot_rf.png'")

# Betting simulation
X_test_betting = X_test.loc[test_betting_data.index]
test_betting_data['Predicted_Prob'] = model.predict_proba(X_test_betting)[:, 1]
test_betting_data['Predicted'] = (test_betting_data['Predicted_Prob'] > 0.5).astype(int)

# Our new dataset already has P1_Odds and P2_Odds correctly assigned
# No need to manipulate the odds here, they are pre-processed during randomization

# Calculate implied probabilities
test_betting_data['Implied_Prob_P1'] = 1 / test_betting_data['AvgW']
test_betting_data['Implied_Prob_P2'] = 1 / test_betting_data['AvgL']

# Normalize implied probabilities
total_implied_prob = test_betting_data['Implied_Prob_P1'] + test_betting_data['Implied_Prob_P2']
test_betting_data['Implied_Prob_P1'] /= total_implied_prob
test_betting_data['Implied_Prob_P2'] /= total_implied_prob

# Calculate edge between model predicted probability and implied probability
test_betting_data['Edge'] = test_betting_data['Predicted_Prob'] - test_betting_data['Implied_Prob_P1']

# Calculate Kelly Criterion for bet sizing with safety mechanisms
epsilon = 0.001  # Small value to prevent division by zero
test_betting_data['Edge'] = np.maximum(0, test_betting_data['Predicted_Prob'] - test_betting_data['Implied_Prob_P1'])
test_betting_data['Kelly_Denominator'] = test_betting_data['Implied_Prob_P1'] * np.maximum(epsilon, 1 - test_betting_data['Predicted_Prob'])
# Calculate Kelly fraction with safeguards against division by zero or infinite values
test_betting_data['Kelly_Fraction'] = test_betting_data['Edge'] / test_betting_data['Kelly_Denominator']

# Cap Kelly fractions at a conservative level to prevent astronomical bet sizes
test_betting_data['Kelly_Fraction'] = test_betting_data['Kelly_Fraction'].clip(0, 0.25)
print(f"Kelly fractions after safety processing: Max={test_betting_data['Kelly_Fraction'].max():.4f}, Mean={test_betting_data['Kelly_Fraction'].mean():.4f}")

# Print prediction statistics
print("\nPrediction Statistics:")
print(f"Average predicted probability: {test_betting_data['Predicted_Prob'].mean():.4f}")
print(f"Max edge found: {test_betting_data['Edge'].max():.4f}")
print(f"Min edge found: {test_betting_data['Edge'].min():.4f}")
print(f"Max Kelly fraction: {test_betting_data['Kelly_Fraction'].max():.4f}")
print(f"Matches with positive edge: {(test_betting_data['Edge'] > 0).sum()}")
print(f"Average edge on positive edge bets: {test_betting_data.loc[test_betting_data['Edge'] > 0, 'Edge'].mean():.4f}")

# Initialize betting simulation
initial_bankroll = 100000  # Changed to match bet_points.py
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

# Simulation - sorted by date to simulate real-time betting
test_betting_data = test_betting_data.sort_values('Date')
for idx, row in test_betting_data.iterrows():
    if bankroll <= 0:
        print(f"Bankroll depleted at match {idx}")
        break
    
    # Only bet when criteria are met
    if row['Kelly_Fraction'] >= min_kelly_fraction and row['Edge'] >= min_edge:
        # Use a very conservative fraction of Kelly (1/20 Kelly)
        kelly_fraction = row['Kelly_Fraction']
        conservative_kelly = 0.05 * kelly_fraction  # 1/20 Kelly
        
        # Cap at 1% of current bankroll for extra safety
        bet_size = min(conservative_kelly * bankroll, 0.01 * bankroll)
        
        # Additional safety: Ensure bet size is reasonable
        bet_size = min(bet_size, 1000)  # Maximum $1000 per bet
        if bet_size > bankroll:
            bet_size = bankroll
        
        num_bets += 1
        total_bets += bet_size
        bankroll -= bet_size
        
        # Determine if bet won or lost - making sure to use the correct odds
        is_win = (row['Predicted'] == 1 and row['Label'] == 1) or (row['Predicted'] == 0 and row['Label'] == 0)
        bet_outcome = "WIN" if is_win else "LOSS"
        
        # Update win/loss counter
        if is_win:
            wins += 1
            # Use correct odds for calculating winnings
            if row['Predicted'] == 1:  # Bet on Player1
                winnings = bet_size * (row['AvgW'] - 1)
            else:  # Bet on Player2
                winnings = bet_size * (row['AvgL'] - 1)
            total_winnings += winnings
            bankroll += bet_size + winnings
        else:
            losses += 1
            winnings = -bet_size
        
        # Record bet details with comprehensive feature information
        bet_record = {
            # Basic match info
            'Date': row['Date'],
            'Tournament': row['Tournament'],
            'Player1': row['Player1'],
            'Player2': row['Player2'],
            
            # Player attributes
            'P1_Rank': df.loc[idx, 'P1Rank'],
            'P2_Rank': df.loc[idx, 'P2Rank'],
            'Rank_Diff': df.loc[idx, 'P1Rank'] - df.loc[idx, 'P2Rank'],
            'P1_Points': df.loc[idx, 'P1Pts'],
            'P2_Points': df.loc[idx, 'P2Pts'],
            'Points_Diff': df.loc[idx, 'P1Pts'] - df.loc[idx, 'P2Pts'],
            'Height_Diff': df.loc[idx, 'Height_Diff_cm'] if 'Height_Diff_cm' in df.columns else None,
            'P1_Age': df.loc[idx, 'P1_Age'] if 'P1_Age' in df.columns else None,
            'P2_Age': df.loc[idx, 'P2_Age'] if 'P2_Age' in df.columns else None,
            'Age_Diff': df.loc[idx, 'Age_Diff_Years'] if 'Age_Diff_Years' in df.columns else None,
            
            # H2H and recent form if available
            'H2H_P1_Wins': df.loc[idx, 'H2H_P1_Wins'] if 'H2H_P1_Wins' in df.columns else None,
            'H2H_Total': df.loc[idx, 'H2H_Total'] if 'H2H_Total' in df.columns else None,
            'H2H_P1_WinRate': df.loc[idx, 'H2H_P1_WinRate'] if 'H2H_P1_WinRate' in df.columns else None,
            
            # Match conditions
            'Surface_Clay': df.loc[idx, 'Surface_Clay'] if 'Surface_Clay' in df.columns else None,
            'Surface_Grass': df.loc[idx, 'Surface_Grass'] if 'Surface_Grass' in df.columns else None,
            'Surface_Hard': df.loc[idx, 'Surface_Hard'] if 'Surface_Hard' in df.columns else None,
            
            # Betting info
            'Odds_P1': row['AvgW'],
            'Odds_P2': row['AvgL'],
            'Predicted_Prob': row['Predicted_Prob'],
            'Implied_Prob': row['Implied_Prob_P1'],
            'Edge': row['Edge'],
            'Kelly_Fraction': row['Kelly_Fraction'],
            
            # Bet details
            'Bet_Size': bet_size,
            'Bet_On': 'Player1' if row['Predicted'] == 1 else 'Player2',
            'Outcome': bet_outcome,
            'Profit': winnings,
            'Current_Bankroll': bankroll,
            
            # Actual outcome
            'Label': row['Label']  # 1 = P1 won, 0 = P2 won
        }
        bet_history.append(bet_record)
    
    bankroll_history.append(bankroll)
    cumulative_bets.append(num_bets)

# Save bet history
if bet_history:
    bet_history_df = pd.DataFrame(bet_history)
    bet_history_df.to_csv(f"{output_dir}/rf_bet_history.csv", index=False)
    print(f"Saved {len(bet_history)} bets to 'rf_bet_history.csv'")

# Plot bankroll
plt.figure(figsize=(12, 7))
plt.plot(bankroll_history, color='blue', linewidth=2)
plt.xlabel('Matches Processed (Chronological)')
plt.ylabel('Bankroll ($)')
plt.title('Random Forest Model with Half-Kelly Betting: Bankroll Over Time')
plt.grid(True, alpha=0.3)
plt.savefig(f'{output_dir}/bankroll_plot_rf.png')
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

print(f"\nBetting Results (Random Forest with Fixed Randomization):")
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
plt.title('Random Forest Calibration vs. Betting Market')
plt.savefig(f'{output_dir}/rf_calibration.png')
plt.close()

print("\nAll outputs saved to:", output_dir)