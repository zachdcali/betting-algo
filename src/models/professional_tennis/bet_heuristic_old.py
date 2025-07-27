import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression

# Define output directory
base_output_dir = "/Users/zachdodson/Documents/betting-algo/data/output"
specific_output_dir = os.path.join(base_output_dir, "Rank_Points_Heuristic")
os.makedirs(specific_output_dir, exist_ok=True)

# Define surfaces
surfaces = ['Hard', 'Clay', 'Grass']

# Create subdirectories for better organization
folders = ['Rank_Overall', 'Rank_Hard', 'Rank_Clay', 'Rank_Grass', 
           'Points_Overall', 'Points_Hard', 'Points_Clay', 'Points_Grass']
for folder in folders:
    os.makedirs(os.path.join(specific_output_dir, folder), exist_ok=True)

# Unified simulate_betting function for both overall and surface-specific simulations
def simulate_betting(data, pred_col, odds_p1_col, odds_p2_col, label_col, prob_col=None, strategy_name="Flat", use_adjusted=False, use_kelly=False):
    # Constants
    initial_bankroll = 100000  # Starting bankroll
    bankroll = initial_bankroll
    
    # Base bet as a percentage of initial bankroll - 1% for proportional strategies
    if "Flat" in strategy_name:
        base_bet = 1000  # 1% of bankroll for flat betting
    elif use_kelly:
        base_bet = 100  # Doesn't matter for Kelly
    else:
        base_bet = 1000  # 1% of bankroll for proportional strategies
    bankroll_history = [bankroll]
    bets, wins, total_bet, total_winnings = 0, 0, 0, 0
    
    # Track predictions for all matches to compute overall accuracy
    correct_predictions = 0
    total_predictions = 0
    
    # Sort by date for chronological simulation
    data = data.sort_values('Date')
    
    # Create a list to store detailed bet information
    bet_details = []
    
    # Track max drawdown
    max_bankroll = initial_bankroll
    max_drawdown = 0
    daily_returns = []
    
    for _, row in data.iterrows():
        # Initialize variables for this bet
        bet_made = False
        bet_size = 0
        model_probability = None
        implied_probability = None
        edge = None
        bet_outcome = None
        winnings = 0
        
        # Skip if prediction is NaN (edge case like same rank/points or missing data)
        if pd.isna(row[pred_col]):
            bankroll_history.append(bankroll)
            # Extract the surface from the Surface_X columns
            surface = "Unknown"
            if 'Surface_Hard' in row and row['Surface_Hard'] == 1:
                surface = "Hard"
            elif 'Surface_Clay' in row and row['Surface_Clay'] == 1:
                surface = "Clay"
            elif 'Surface_Grass' in row and row['Surface_Grass'] == 1:
                surface = "Grass"
                
            # Format match score in tennis format (e.g., "6-4, 7-5")
            score = ""
            set_num = 1
            while f'W{set_num}' in row and f'L{set_num}' in row and not pd.isna(row[f'W{set_num}']) and not pd.isna(row[f'L{set_num}']):
                # If P1 won, use W as P1 score and L as P2 score, otherwise swap
                if row[label_col] == 1:
                    p1_score = row[f'W{set_num}']
                    p2_score = row[f'L{set_num}']
                else:
                    p1_score = row[f'L{set_num}']
                    p2_score = row[f'W{set_num}']
                    
                if set_num > 1:
                    score += ", "
                score += f"{int(p1_score)}-{int(p2_score)}"
                set_num += 1
                
            bet_details.append({
                'Date': row['Date'],
                'Player1': row['Player1'],
                'Player2': row['Player2'],
                'P1_Rank': row['P1Rank'] if 'P1Rank' in row else None,
                'P2_Rank': row['P2Rank'] if 'P2Rank' in row else None,
                'P1_Points': row['P1Pts'] if 'P1Pts' in row else None, 
                'P2_Points': row['P2Pts'] if 'P2Pts' in row else None,
                'Surface': surface,
                'Tournament': row['Tournament'] if 'Tournament' in row else 'Unknown',
                'Score': score if score else None,
                'Bet_Made': False,
                'Prediction': None,
                'Actual': row[label_col],
                'Bet_Size': 0,
                'Bet_Size_Pct': 0,
                'Odds': None,
                'Model_Probability': None,
                'Implied_Probability': None,
                'Edge': None,
                'Outcome': 'No Bet',
                'Winnings': 0,
                'Bankroll': bankroll
            })
            continue
            
        pred = row[pred_col]
        odds = row[odds_p1_col] if pred else row[odds_p2_col]
        actual = row[label_col]
        
        # Calculate implied probability from odds
        implied_probability = 1 / odds if odds > 0 else None
        
        # Track prediction accuracy for all matches
        total_predictions += 1
        if (pred and actual == 1) or (not pred and actual == 0):
            correct_predictions += 1
        
        # Default to flat betting
        bet_size = base_bet
        model_probability = None
        
        # Only calculate prob if prob_col is provided
        if prob_col is not None:
            # Skip if probability is NaN
            if pd.isna(row[prob_col]):
                bankroll_history.append(bankroll)
                # Extract the surface from the Surface_X columns
                surface = "Unknown"
                if 'Surface_Hard' in row and row['Surface_Hard'] == 1:
                    surface = "Hard"
                elif 'Surface_Clay' in row and row['Surface_Clay'] == 1:
                    surface = "Clay"
                elif 'Surface_Grass' in row and row['Surface_Grass'] == 1:
                    surface = "Grass"
                    
                # Format match score in tennis format (e.g., "6-4, 7-5")
                score = ""
                set_num = 1
                while f'W{set_num}' in row and f'L{set_num}' in row and not pd.isna(row[f'W{set_num}']) and not pd.isna(row[f'L{set_num}']):
                    # If P1 won, use W as P1 score and L as P2 score, otherwise swap
                    if row[label_col] == 1:
                        p1_score = row[f'W{set_num}']
                        p2_score = row[f'L{set_num}']
                    else:
                        p1_score = row[f'L{set_num}']
                        p2_score = row[f'W{set_num}']
                        
                    if set_num > 1:
                        score += ", "
                    score += f"{int(p1_score)}-{int(p2_score)}"
                    set_num += 1
                    
                bet_details.append({
                    'Date': row['Date'],
                    'Player1': row['Player1'],
                    'Player2': row['Player2'],
                    'P1_Rank': row['P1Rank'] if 'P1Rank' in row else None,
                    'P2_Rank': row['P2Rank'] if 'P2Rank' in row else None,
                    'P1_Points': row['P1Pts'] if 'P1Pts' in row else None, 
                    'P2_Points': row['P2Pts'] if 'P2Pts' in row else None,
                    'Surface': surface,
                    'Tournament': row['Tournament'] if 'Tournament' in row else 'Unknown',
                    'Score': score if score else None,
                    'Bet_Made': False,
                    'Prediction': 'P1' if pred else 'P2',
                    'Actual': 'P1' if row[label_col] == 1 else 'P2',
                    'Bet_Size': 0,
                    'Bet_Size_Pct': 0,
                    'Odds': odds,
                    'Model_Probability': None,
                    'Implied_Probability': implied_probability,
                    'Edge': None,
                    'Outcome': 'No Bet',
                    'Winnings': 0,
                    'Bankroll': bankroll
                })
                continue
                
            prob = row[prob_col] if pred else (1 - row[prob_col])
            model_probability = prob
            
            # Calculate edge (difference between model probability and implied probability)
            edge = model_probability - implied_probability if implied_probability is not None else None
            
            # Check for positive EV (p * odds > 1.0) for Proportional and Adjusted Proportional
            ev_condition = prob * odds > 1.0
            
            if use_kelly:
                b = odds - 1
                p = prob
                q = 1 - p
                f = (b * p - q) / b if b > 0 else 0
                # Calculate Half Kelly with a 5% bankroll maximum
                half_kelly = (f / 2) * bankroll if f > 0 and prob * odds > 1 else 0
                # Cap at 5% of bankroll
                bet_size = min(half_kelly, 0.05 * bankroll) if half_kelly > 0 else 0  # Half Kelly with max 5%
            elif use_adjusted:
                # Multiply by 2 to scale up from (prob-0.5) range of 0-0.5 to 0-1.0 range
                bet_size = base_bet * 2 * (prob - 0.5) if prob > 0.5 and ev_condition else 0  # Adjusted Proportional with positive EV
            else:
                bet_size = base_bet * prob if ev_condition else 0  # Proportional with positive EV
        else:
            bet_size = base_bet
        
        # Calculate bet size as percentage of current bankroll
        bet_size_pct = (bet_size / bankroll) * 100 if bankroll > 0 else 0
        
        if bet_size > 0:
            bet_made = True
            bets += 1
            total_bet += bet_size
            bankroll -= bet_size
            
            if (pred and actual == 1) or (not pred and actual == 0):
                wins += 1
                winnings = bet_size * (odds - 1)
                total_winnings += winnings
                bankroll += bet_size + winnings
                bet_outcome = 'Win'
                # Add daily return for Sharpe Ratio calculation
                daily_returns.append(winnings / bet_size if bet_size > 0 else 0)
            else:
                bet_outcome = 'Loss'
                winnings = -bet_size
                # Add daily return for Sharpe Ratio calculation
                daily_returns.append(-1)  # -100% return
        else:
            bet_outcome = 'No Bet'
        
        bankroll_history.append(bankroll)
        
        # Update max bankroll and drawdown tracking
        if bankroll > max_bankroll:
            max_bankroll = bankroll
        current_drawdown = (max_bankroll - bankroll) / max_bankroll if max_bankroll > 0 else 0
        if current_drawdown > max_drawdown:
            max_drawdown = current_drawdown
        
        # Extract the surface from the Surface_X columns
        surface = "Unknown"
        if 'Surface_Hard' in row and row['Surface_Hard'] == 1:
            surface = "Hard"
        elif 'Surface_Clay' in row and row['Surface_Clay'] == 1:
            surface = "Clay"
        elif 'Surface_Grass' in row and row['Surface_Grass'] == 1:
            surface = "Grass"
            
        # Format match score in tennis format (e.g., "6-4, 7-5")
        score = ""
        set_num = 1
        while f'W{set_num}' in row and f'L{set_num}' in row and not pd.isna(row[f'W{set_num}']) and not pd.isna(row[f'L{set_num}']):
            # If P1 won, use W as P1 score and L as P2 score, otherwise swap
            if row[label_col] == 1:
                p1_score = row[f'W{set_num}']
                p2_score = row[f'L{set_num}']
            else:
                p1_score = row[f'L{set_num}']
                p2_score = row[f'W{set_num}']
                
            if set_num > 1:
                score += ", "
            score += f"{int(p1_score)}-{int(p2_score)}"
            set_num += 1
        
        # Store detailed bet information
        bet_details.append({
            'Date': row['Date'],
            'Player1': row['Player1'],
            'Player2': row['Player2'],
            'P1_Rank': row['P1Rank'] if 'P1Rank' in row else None,
            'P2_Rank': row['P2Rank'] if 'P2Rank' in row else None,
            'P1_Points': row['P1Pts'] if 'P1Pts' in row else None, 
            'P2_Points': row['P2Pts'] if 'P2Pts' in row else None,
            'Surface': surface,
            'Tournament': row['Tournament'] if 'Tournament' in row else 'Unknown',
            'Score': score if score else None,
            'Bet_Made': bet_made,
            'Prediction': 'P1' if pred else 'P2',
            'Actual': 'P1' if row[label_col] == 1 else 'P2',
            'Bet_Size': bet_size,
            'Bet_Size_Pct': bet_size_pct,
            'Odds': odds,
            'Model_Probability': model_probability,
            'Implied_Probability': implied_probability,
            'Edge': edge,
            'Outcome': bet_outcome,
            'Winnings': winnings if bet_outcome == 'Win' else (-bet_size if bet_outcome == 'Loss' else 0),
            'Bankroll': bankroll
        })
        
        if bankroll <= 0:
            break  # Stop if bankroll depletes
    
    # Calculate the actual return on investment (how much profit made per dollar wagered)
    roi = (bankroll - initial_bankroll) / total_bet if total_bet > 0 else 0
    win_rate = wins / bets if bets > 0 else 0
    prediction_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    
    # Calculate Sharpe ratio and other metrics
    sharpe_ratio = None
    if len(daily_returns) > 1:
        avg_return = np.mean(daily_returns)
        std_return = np.std(daily_returns)
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    
    # Calculate additional betting metrics
    profit = bankroll - initial_bankroll
    avg_bet_size = total_bet / bets if bets > 0 else 0
    profit_per_bet = profit / bets if bets > 0 else 0
    
    # Calculate basic win/loss metrics
    avg_win = total_winnings / wins if wins > 0 else 0
    avg_loss = -total_bet / (bets - wins) if (bets - wins) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    
    # Calculate Profit Factor (gross profit / gross loss)
    profit_factor = total_winnings / (total_bet - total_winnings + winnings) if (total_bet - total_winnings + winnings) > 0 else 0
    
    # Calculate average Expected Value per bet
    # For each bet: EV = p*(odds-1) - (1-p)
    # This is the theoretical edge per dollar bet
    expected_values = []
    for _, row in data.iterrows():
        if not pd.isna(row[pred_col]) and prob_col is not None and not pd.isna(row[prob_col]):
            pred = row[pred_col]
            odds = row[odds_p1_col] if pred else row[odds_p2_col]
            prob = row[prob_col] if pred else (1 - row[prob_col])
            if odds > 0:
                ev = prob * (odds - 1) - (1 - prob)
                expected_values.append(ev)
    
    # Average EV across all potential bets
    avg_expected_value = np.mean(expected_values) if expected_values else None
    
    # Also calculate average EV of bets actually placed
    ev_bets_placed = []
    for _, row in data.iterrows():
        if not pd.isna(row[pred_col]) and prob_col is not None and not pd.isna(row[prob_col]):
            pred = row[pred_col]
            odds = row[odds_p1_col] if pred else row[odds_p2_col]
            prob = row[prob_col] if pred else (1 - row[prob_col])
            
            # Check if we would place this bet
            would_bet = False
            if use_kelly:
                b = odds - 1
                p = prob
                q = 1 - p
                f = (b * p - q) / b if b > 0 else 0
                would_bet = f > 0 and prob * odds > 1  # Half Kelly with positive edge
            elif use_adjusted:
                would_bet = prob > 0.5 and prob * odds > 1.0
            elif prob_col is not None:
                would_bet = prob * odds > 1.0
            else:
                would_bet = True
                
            if would_bet:
                ev = prob * (odds - 1) - (1 - prob)
                ev_bets_placed.append(ev)
    
    # Average EV of bets actually placed
    avg_ev_bets_placed = np.mean(ev_bets_placed) if ev_bets_placed else None
    
    # Calculate volatility (standard deviation of returns)
    volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
    
    # Calculate statistical significance (z-score assuming normal distribution)
    z_score = None
    if volatility > 0 and bets > 30:  # Need decent sample size for meaningful z-score
        z_score = (roi * np.sqrt(bets)) / volatility
    
    # Calculate Risk of Ruin (approximate using exponential formula)
    risk_of_ruin = None
    if win_rate != 0.5 and volatility > 0:
        # This is a simplified approximation based on gambler's ruin problem
        risk_of_ruin = np.exp(-2 * initial_bankroll * abs(roi) / (avg_bet_size * volatility))
        risk_of_ruin = min(risk_of_ruin, 1.0)  # Cap at 100%
    
    # Track maximum consecutive wins and losses
    max_consecutive_wins = 0
    max_consecutive_losses = 0
    current_consecutive_wins = 0
    current_consecutive_losses = 0
    
    # Calculate Sortino ratio (like Sharpe but only using negative returns)
    sortino_ratio = None
    if len(daily_returns) > 1:
        negative_returns = [r for r in daily_returns if r < 0]
        if negative_returns:
            downside_deviation = np.std(negative_returns)
            sortino_ratio = avg_return / downside_deviation if downside_deviation > 0 else 0
    
    # Calculate Calmar ratio (return divided by max drawdown)
    calmar_ratio = abs(roi / max_drawdown) if max_drawdown > 0 else None
    
    # Calculate Ulcer Index (measure of drawdown depth and duration)
    ulcer_index = None
    if len(bankroll_history) > 1:
        rolling_max = np.maximum.accumulate(bankroll_history)
        drawdowns = np.zeros_like(bankroll_history)
        for i in range(len(bankroll_history)):
            if rolling_max[i] > 0:  # Avoid division by zero
                drawdowns[i] = (rolling_max[i] - bankroll_history[i]) / rolling_max[i]
        # Square the drawdowns and take the square root of their mean
        ulcer_index = np.sqrt(np.mean(np.square(drawdowns)))
    
    # Determine EV cutoff based on strategy
    ev_cutoff = "N/A"  # Default for flat betting
    if use_kelly:
        ev_cutoff = "10% Edge (prob * odds > 1)"
    elif use_adjusted or prob_col is not None:
        ev_cutoff = "0% Edge (prob * odds > 1.0)"
    
    # Create a metrics dictionary
    metrics = {
        # Strategy information
        'Strategy_Type': strategy_name,
        'EV_Cutoff': ev_cutoff,
        
        # Basic performance metrics
        'Initial_Bankroll': initial_bankroll,
        'Final_Bankroll': bankroll,
        'Profit': profit,
        'ROI': roi,
        'Avg_EV_All_Matches': avg_expected_value,
        'Avg_EV_Bets_Placed': avg_ev_bets_placed,
        'Win_Rate': win_rate,
        'Prediction_Accuracy': prediction_accuracy,
        
        # Betting activity
        'Bets_Made': bets,
        'Wins': wins,
        'Losses': bets - wins,
        'Avg_Bet_Size': avg_bet_size,
        'Profit_Per_Bet': profit_per_bet,
        'Avg_Win': avg_win,
        'Avg_Loss': avg_loss,
        'Win_Loss_Ratio': win_loss_ratio,
        
        # Risk metrics
        'Max_Drawdown': max_drawdown,
        'Volatility': volatility,
        'Risk_of_Ruin': risk_of_ruin,
        
        # Statistical metrics
        'Z_Score': z_score,
        'Profit_Factor': profit_factor,
        
        # Risk-adjusted performance ratios
        'Sharpe_Ratio': sharpe_ratio,
        'Sortino_Ratio': sortino_ratio,
        'Calmar_Ratio': calmar_ratio,
        'Ulcer_Index': ulcer_index,
        
        # Streak information
        'Max_Consecutive_Wins': max_consecutive_wins,
        'Max_Consecutive_Losses': max_consecutive_losses
    }
    
    # Create figure with two subplots - bankroll chart on top, metrics table below
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top subplot - Bankroll chart
    ax1.plot(bankroll_history, color='blue', linewidth=2)
    ax1.set_xlabel('Matches Processed (Chronological)')
    ax1.set_ylabel('Bankroll ($)')
    
    # Use "Overall" for title if no Surface column, otherwise use surface name
    title_surface = data['Surface'].iloc[0] if 'Surface' in data.columns and len(set(data['Surface'])) == 1 else "Overall"
    ax1.set_title(f'{strategy_name}: Bankroll Over Time ({title_surface})')
    ax1.grid(True, alpha=0.3)
    
    # Helper function to format values safely
    def format_metric(value):
        if value is None:
            return "N/A"
        try:
            return f"{value:.4f}"
        except (ValueError, TypeError):
            return str(value)
    
    # Bottom subplot - Metrics table (hidden axes)
    ax2.axis('off')
    
    # Create a 2x5 grid for metrics (10 metrics per row, 2 rows)
    col_width = 0.2
    row_height = 0.4
    
    # Row 1 - Performance metrics
    metrics_row1 = [
        f"Strategy: {strategy_name}",
        f"EV Cutoff: {ev_cutoff}",
        f"Initial: ${initial_bankroll:,.0f}",
        f"Final: ${bankroll:,.0f}",
        f"Profit: ${profit:,.0f}",
        f"ROI: {roi:.4f}",
        f"Win Rate: {win_rate:.4f}",
        f"Pred Acc: {prediction_accuracy:.4f}",
        f"Bets: {bets}",
        f"Avg Bet: ${avg_bet_size:.2f}"
    ]
    
    # Row 2 - Risk and advanced metrics
    metrics_row2 = [
        f"Max Drawdown: {format_metric(max_drawdown)}",
        f"Risk of Ruin: {format_metric(risk_of_ruin)}",
        f"Profit Factor: {format_metric(profit_factor)}",
        f"Win/Loss Ratio: {format_metric(win_loss_ratio)}",
        f"Avg EV (All): {format_metric(avg_expected_value)}",
        f"Avg EV (Bets): {format_metric(avg_ev_bets_placed)}",
        f"Sharpe: {format_metric(sharpe_ratio)}",
        f"Sortino: {format_metric(sortino_ratio)}",
        f"Calmar: {format_metric(calmar_ratio)}",
        f"Volatility: {format_metric(volatility)}"
    ]
    
    # Add metrics title 
    ax2.text(0.5, 0.9, "PERFORMANCE METRICS", ha='center', fontsize=14, fontweight='bold')
    
    # Add metrics in row 1
    for i, metric in enumerate(metrics_row1):
        x_pos = 0.05 + (i % 5) * col_width
        y_pos = 0.6 if i < 5 else 0.45
        ax2.text(x_pos, y_pos, metric, fontsize=10)
    
    # Add metrics in row 2
    for i, metric in enumerate(metrics_row2):
        x_pos = 0.05 + (i % 5) * col_width
        y_pos = 0.3 if i < 5 else 0.15
        ax2.text(x_pos, y_pos, metric, fontsize=10)
        
    # Add a horizontal line between the chart and metrics
    plt.subplots_adjust(hspace=0.1)
    
    # Determine which folder to use based on strategy and surface
    if "Rank" in strategy_name:
        folder = f"Rank_{title_surface}"
    elif "Points" in strategy_name:
        folder = f"Points_{title_surface}"
    else:  # For "Favorite" strategy
        folder = f"Rank_{title_surface}"  # Default to Rank folder
    
    # Save the bankroll chart figure
    plt.savefig(os.path.join(specific_output_dir, folder, f'bankroll_plot_{strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "")}.png'))
    plt.close()
    
    # Save detailed bet information to CSV
    strategy_clean = strategy_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
    detail_df = pd.DataFrame(bet_details)
    
    # Filter out non-bet matches if requested (commented out by default as we're keeping them for now)
    # if filter_no_bets:
    #     detail_df = detail_df[detail_df['Bet_Made'] == True]
    
    # Save to CSV
    detail_df.to_csv(os.path.join(specific_output_dir, folder, f'detailed_bets_{strategy_clean}.csv'), index=False)
    
    # Also save a metrics summary file
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(specific_output_dir, folder, f'metrics_{strategy_clean}.csv'), index=False)
    
    return bankroll, roi, win_rate, bets, wins, bets - wins, bankroll_history, prediction_accuracy, max_drawdown, sharpe_ratio

# Load data
df = pd.read_csv("/Users/zachdodson/Documents/betting-algo/data/Old Data/processed_atp_data.csv", low_memory=False)
# Try different date formats
try:
    # Try ISO format (YYYY-MM-DD)
    df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
except ValueError:
    try:
        # Try MM/DD/YY format
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
    except ValueError:
        # Use default format detection
        df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year
print(f"Loaded data with shape: {df.shape}")

# Train-test split (2000-2019 for training, 2020-2025 for testing)
train_df = df[df['Date'] < '2023-01-01'].copy()
test_df = df[df['Date'] >= '2023-01-01'].copy()
print(f"Training data (2000-2012): {len(train_df)} matches")
print(f"Testing data (2023-2025): {len(test_df)} matches")

# Reconstruct the Surface column from one-hot encoded columns
df['Surface'] = np.where(df['Surface_Hard'] == 1, 'Hard',
                         np.where(df['Surface_Clay'] == 1, 'Clay',
                                  np.where(df['Surface_Grass'] == 1, 'Grass', 'Unknown')))
train_df['Surface'] = np.where(train_df['Surface_Hard'] == 1, 'Hard',
                               np.where(train_df['Surface_Clay'] == 1, 'Clay',
                                        np.where(train_df['Surface_Grass'] == 1, 'Grass', 'Unknown')))
test_df['Surface'] = np.where(test_df['Surface_Hard'] == 1, 'Hard',
                              np.where(test_df['Surface_Clay'] == 1, 'Clay',
                                       np.where(test_df['Surface_Grass'] == 1, 'Grass', 'Unknown')))

# Prepare dataframes for betting (test set)
df_odds = test_df.dropna(subset=['AvgW', 'AvgL']).copy()
df_points = test_df[(test_df['P1Pts'] > 0) & (test_df['P2Pts'] > 0)].copy()
df_rank_betting = test_df.dropna(subset=['AvgW', 'AvgL']).copy()
df_points_betting = df_points.dropna(subset=['AvgW', 'AvgL']).copy()

# Prepare training counterparts for accuracy calculations
train_df_odds = train_df.dropna(subset=['AvgW', 'AvgL']).copy()
train_df_points = train_df[(train_df['P1Pts'] > 0) & (train_df['P2Pts'] > 0)].copy()

# Add Surface column to all derived dataframes
df_odds['Surface'] = np.where(df_odds['Surface_Hard'] == 1, 'Hard',
                              np.where(df_odds['Surface_Clay'] == 1, 'Clay',
                                       np.where(df_odds['Surface_Grass'] == 1, 'Grass', 'Unknown')))
df_points['Surface'] = np.where(df_points['Surface_Hard'] == 1, 'Hard',
                                np.where(df_points['Surface_Clay'] == 1, 'Clay',
                                         np.where(df_points['Surface_Grass'] == 1, 'Grass', 'Unknown')))
df_rank_betting['Surface'] = np.where(df_rank_betting['Surface_Hard'] == 1, 'Hard',
                                      np.where(df_rank_betting['Surface_Clay'] == 1, 'Clay',
                                               np.where(df_rank_betting['Surface_Grass'] == 1, 'Grass', 'Unknown')))
df_points_betting['Surface'] = np.where(df_points_betting['Surface_Hard'] == 1, 'Hard',
                                        np.where(df_points_betting['Surface_Clay'] == 1, 'Clay',
                                                 np.where(df_points_betting['Surface_Grass'] == 1, 'Grass', 'Unknown')))

train_df_odds['Surface'] = np.where(train_df_odds['Surface_Hard'] == 1, 'Hard',
                                    np.where(train_df_odds['Surface_Clay'] == 1, 'Clay',
                                             np.where(train_df_odds['Surface_Grass'] == 1, 'Grass', 'Unknown')))
train_df_points['Surface'] = np.where(train_df_points['Surface_Hard'] == 1, 'Hard',
                                      np.where(train_df_points['Surface_Clay'] == 1, 'Clay',
                                               np.where(train_df_points['Surface_Grass'] == 1, 'Grass', 'Unknown')))

print(f"Matches with odds (test): {len(df_odds)}")
print(f"Matches with points (test): {len(df_points)}")
print(f"Matches for Higher Rank betting (with odds, test): {len(df_rank_betting)}")
print(f"Matches for More Points betting (with odds, test): {len(df_points_betting)}")

# Reassign odds to Player 1 and Player 2 based on Label (test set)
df_odds['P1_Odds'] = np.where(df_odds['Label'] == 1, df_odds['AvgW'], df_odds['AvgL'])
df_odds['P2_Odds'] = np.where(df_odds['Label'] == 1, df_odds['AvgL'], df_odds['AvgW'])
df_rank_betting['P1_Odds'] = np.where(df_rank_betting['Label'] == 1, df_rank_betting['AvgW'], df_rank_betting['AvgL'])
df_rank_betting['P2_Odds'] = np.where(df_rank_betting['Label'] == 1, df_rank_betting['AvgL'], df_rank_betting['AvgW'])
df_points_betting['P1_Odds'] = np.where(df_points_betting['Label'] == 1, df_points_betting['AvgW'], df_points_betting['AvgL'])
df_points_betting['P2_Odds'] = np.where(df_points_betting['Label'] == 1, df_points_betting['AvgL'], df_points_betting['AvgW'])

# Reassign odds for training set
train_df_odds['P1_Odds'] = np.where(train_df_odds['Label'] == 1, train_df_odds['AvgW'], train_df_odds['AvgL'])
train_df_odds['P2_Odds'] = np.where(train_df_odds['Label'] == 1, train_df_odds['AvgL'], train_df_odds['AvgW'])

# Baselines (training set)
train_df['Higher_Rank_Winner'] = ((train_df['P1Rank'] < train_df['P2Rank']) & (train_df['Label'] == 1)) | ((train_df['P1Rank'] > train_df['P2Rank']) & (train_df['Label'] == 0))
train_df_points['More_Points_Winner'] = ((train_df_points['P1Pts'] > train_df_points['P2Pts']) & (train_df_points['Label'] == 1)) | ((train_df_points['P1Pts'] < train_df_points['P2Pts']) & (train_df_points['Label'] == 0))
train_df_odds['Favorite_Winner'] = ((train_df_odds['P1_Odds'] < train_df_odds['P2_Odds']) & (train_df_odds['Label'] == 1)) | ((train_df_odds['P1_Odds'] > train_df_odds['P2_Odds']) & (train_df_odds['Label'] == 0))

overall_rank_acc_train = train_df['Higher_Rank_Winner'].mean()
overall_points_acc_train = train_df_points['More_Points_Winner'].mean()
overall_favorite_acc_train = train_df_odds['Favorite_Winner'].mean()

# Baselines (test set)
test_df['Higher_Rank_Winner'] = ((test_df['P1Rank'] < test_df['P2Rank']) & (test_df['Label'] == 1)) | ((test_df['P1Rank'] > test_df['P2Rank']) & (test_df['Label'] == 0))
df_points['More_Points_Winner'] = ((df_points['P1Pts'] > df_points['P2Pts']) & (df_points['Label'] == 1)) | ((df_points['P1Pts'] < df_points['P2Pts']) & (df_points['Label'] == 0))
df_odds['Favorite_Winner'] = ((df_odds['P1_Odds'] < df_odds['P2_Odds']) & (df_odds['Label'] == 1)) | ((df_odds['P1_Odds'] > df_odds['P2_Odds']) & (df_odds['Label'] == 0))

overall_rank_acc_test = test_df['Higher_Rank_Winner'].mean()
overall_points_acc_test = df_points['More_Points_Winner'].mean()
overall_favorite_acc_test = df_odds['Favorite_Winner'].mean()

# Yearly stats (on test set)
yearly_stats_test = []
for year in range(2020, 2026):  # Test years
    year_df = test_df[test_df['Year'] == year]
    year_df_odds = df_odds[df_odds['Year'] == year]
    year_df_points = df_points[df_points['Year'] == year]
    
    stats = {
        'Year': year,
        'Matches (Rank)': len(year_df),
        'Rank Acc': year_df['Higher_Rank_Winner'].mean() if len(year_df) > 0 else 0,
        'Matches (Points)': len(year_df_points),
        'Points Acc': year_df_points['More_Points_Winner'].mean() if len(year_df_points) > 0 else 0,
        'Matches (Odds)': len(year_df_odds),
        'Favorite Acc': year_df_odds['Favorite_Winner'].mean() if len(year_df_odds) > 0 else 0,
    }
    yearly_stats_test.append(stats)
yearly_df_test = pd.DataFrame(yearly_stats_test)

# Yearly stats (on training set, for reference)
yearly_stats_train = []
for year in range(2000, 2020):  # Training years
    year_df = train_df[train_df['Year'] == year]
    year_df_odds = train_df_odds[train_df_odds['Year'] == year]
    year_df_points = train_df_points[train_df_points['Year'] == year]
    
    stats = {
        'Year': year,
        'Matches (Rank)': len(year_df),
        'Rank Acc': year_df['Higher_Rank_Winner'].mean() if len(year_df) > 0 else 0,
        'Matches (Points)': len(year_df_points),
        'Points Acc': year_df_points['More_Points_Winner'].mean() if len(year_df_points) > 0 else 0,
        'Matches (Odds)': len(year_df_odds),
        'Favorite Acc': year_df_odds['Favorite_Winner'].mean() if len(year_df_odds) > 0 else 0,
    }
    yearly_stats_train.append(stats)
yearly_df_train = pd.DataFrame(yearly_stats_train)

# Fit logistic regression models for each surface (on training set)
surface_rank_models = {}
surface_points_models = {}

for surface in surfaces:
    # Filter training data by surface for model fitting
    train_surface_df = train_df[train_df['Surface'] == surface].copy()
    train_surface_points = train_df_points[train_df_points['Surface'] == surface].copy()
    
    # Rank model for this surface
    train_surface_df['Rank_Diff'] = train_surface_df['P2Rank'] - train_surface_df['P1Rank']
    rank_model = LogisticRegression()
    rank_model.fit(train_surface_df[['Rank_Diff']], train_surface_df['Label'])
    surface_rank_models[surface] = rank_model
    
    # Points model for this surface
    train_surface_points['Points_Diff'] = train_surface_points['P1Pts'] - train_surface_points['P2Pts']
    points_model = LogisticRegression()
    points_model.fit(train_surface_points[['Points_Diff']], train_surface_points['Label'])
    surface_points_models[surface] = points_model

# Print overall model info (for reference only)
print("\nUsing surface-specific regression models for overall predictions")

# Create Rank_Diff column for testing data
df_rank_betting['Rank_Diff'] = df_rank_betting['P2Rank'] - df_rank_betting['P1Rank']
df_points_betting['Points_Diff'] = df_points_betting['P1Pts'] - df_points_betting['P2Pts']

# Apply surface-specific models to each match based on surface
df_rank_betting['Rank_Win_Prob'] = np.nan  # Initialize with NaN
df_points_betting['Points_Win_Prob'] = np.nan  # Initialize with NaN

# Populate probabilities using surface-specific models
for surface in surfaces:
    # Apply the surface-specific rank model to matches on this surface
    surface_matches = df_rank_betting['Surface'] == surface
    df_rank_betting.loc[surface_matches, 'Rank_Win_Prob'] = surface_rank_models[surface].predict_proba(
        df_rank_betting.loc[surface_matches, ['Rank_Diff']])[:, 1]
    
    # Apply the surface-specific points model to matches on this surface
    surface_points_matches = df_points_betting['Surface'] == surface
    df_points_betting.loc[surface_points_matches, 'Points_Win_Prob'] = surface_points_models[surface].predict_proba(
        df_points_betting.loc[surface_points_matches, ['Points_Diff']])[:, 1]

# Plot overall data with surface-specific models
# We'll create a combined plot showing all three surfaces with their specific models
plt.figure(figsize=(15, 8))

# Plot rank models
ax1 = plt.subplot(1, 2, 1)
rank_diff_range = np.linspace(-500, 500, 100)
rank_diff_range_df = pd.DataFrame(rank_diff_range, columns=['Rank_Diff'])

for surface, color in zip(surfaces, ['blue', 'red', 'green']):
    # Plot the surface-specific regression line
    rank_prob_range = surface_rank_models[surface].predict_proba(rank_diff_range_df)[:, 1]
    ax1.plot(rank_diff_range, rank_prob_range, color=color, linewidth=2, 
             label=f'{surface} Surface Model')

ax1.set_xlim(-500, 500)
ax1.set_xlabel('Rank Difference (P2Rank - P1Rank)')
ax1.set_ylabel('Probability of Player 1 Winning')
ax1.set_title('Surface-Specific Rank Models')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot points models
ax2 = plt.subplot(1, 2, 2)
points_diff_range = np.linspace(-5000, 5000, 100)
points_diff_range_df = pd.DataFrame(points_diff_range, columns=['Points_Diff'])

for surface, color in zip(surfaces, ['blue', 'red', 'green']):
    # Plot the surface-specific regression line
    points_prob_range = surface_points_models[surface].predict_proba(points_diff_range_df)[:, 1]
    ax2.plot(points_diff_range, points_prob_range, color=color, linewidth=2, 
             label=f'{surface} Surface Model')

ax2.set_xlim(-5000, 5000)
ax2.set_xlabel('Points Difference (P1Pts - P2Pts)')
ax2.set_ylabel('Probability of Player 1 Winning')
ax2.set_title('Surface-Specific Points Models')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(specific_output_dir, 'surface_specific_models.png'))
plt.close()

# Initialize results dictionary
surface_results = {}

# Store overall prediction accuracies for each surface (to be computed later)
surface_rank_pred_acc = {}
surface_points_pred_acc = {}
surface_favorite_pred_acc = {}

for surface in surfaces:
    print(f"\n========== Processing Surface: {surface} ==========")
    
    # Filter data by surface
    train_df_surface = train_df[train_df['Surface'] == surface].copy()
    df_surface = test_df[test_df['Surface'] == surface].copy()
    df_odds_surface = df_odds[df_odds['Surface'] == surface].copy()
    df_points_surface = df_points[df_points['Surface'] == surface].copy()
    df_rank_betting_surface = df_rank_betting[df_rank_betting['Surface'] == surface].copy()
    df_points_betting_surface = df_points_betting[df_points_betting['Surface'] == surface].copy()
    
    print(f"Matches on {surface} (test): {len(df_odds_surface)}")
    print(f"Matches with odds on {surface} (test): {len(df_odds_surface)}")
    print(f"Matches with points on {surface} (test): {len(df_points_surface)}")
    print(f"Matches for Higher Rank betting (with odds) on {surface} (test): {len(df_rank_betting_surface)}")
    print(f"Matches for More Points betting (with odds) on {surface} (test): {len(df_points_betting_surface)}")
    
    # Reassign odds for betting dataframes
    df_odds_surface['P1_Odds'] = np.where(df_odds_surface['Label'] == 1, df_odds_surface['AvgW'], df_odds_surface['AvgL'])
    df_odds_surface['P2_Odds'] = np.where(df_odds_surface['Label'] == 1, df_odds_surface['AvgL'], df_odds_surface['AvgW'])
    df_rank_betting_surface['P1_Odds'] = np.where(df_rank_betting_surface['Label'] == 1, df_rank_betting_surface['AvgW'], df_rank_betting_surface['AvgL'])
    df_rank_betting_surface['P2_Odds'] = np.where(df_rank_betting_surface['Label'] == 1, df_rank_betting_surface['AvgL'], df_rank_betting_surface['AvgW'])
    df_points_betting_surface['P1_Odds'] = np.where(df_points_betting_surface['Label'] == 1, df_points_betting_surface['AvgW'], df_points_betting_surface['AvgL'])
    df_points_betting_surface['P2_Odds'] = np.where(df_points_betting_surface['Label'] == 1, df_points_betting_surface['AvgL'], df_points_betting_surface['AvgW'])
    
    # Baselines
    df_surface['Higher_Rank_Winner'] = ((df_surface['P1Rank'] < df_surface['P2Rank']) & (df_surface['Label'] == 1)) | ((df_surface['P1Rank'] > df_surface['P2Rank']) & (df_surface['Label'] == 0))
    df_points_surface['More_Points_Winner'] = ((df_points_surface['P1Pts'] > df_points_surface['P2Pts']) & (df_points_surface['Label'] == 1)) | ((df_points_surface['P1Pts'] < df_points_surface['P2Pts']) & (df_points_surface['Label'] == 0))
    df_odds_surface['Favorite_Winner'] = ((df_odds_surface['P1_Odds'] < df_odds_surface['P2_Odds']) & (df_odds_surface['Label'] == 1)) | ((df_odds_surface['P1_Odds'] > df_odds_surface['P2_Odds']) & (df_odds_surface['Label'] == 0))
    
    surface_rank_acc = df_surface['Higher_Rank_Winner'].mean()
    surface_points_acc = df_points_surface['More_Points_Winner'].mean()
    surface_favorite_acc = df_odds_surface['Favorite_Winner'].mean()
    
    # Fit logistic regression models
    # Higher Rank
    train_df_surface['Rank_Diff'] = train_df_surface['P2Rank'] - train_df_surface['P1Rank']
    
    # Add polynomial features for better curve fitting
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import Pipeline
    
    # Create a pipeline with polynomial features and logistic regression
    # Use L2 regularization (Ridge) to prevent overfitting
    polynomial_degree = 2  # Quadratic features
    
    rank_model_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False)),
        ('log_reg', LogisticRegression(C=1.0, solver='liblinear'))
    ])
    
    # Train the model
    rank_model_pipeline.fit(train_df_surface[['Rank_Diff']], train_df_surface['Label'])
    rank_model = rank_model_pipeline  # Use the pipeline as our model
    
    # Apply to the betting data
    df_rank_betting_surface['Rank_Diff'] = df_rank_betting_surface['P2Rank'] - df_rank_betting_surface['P1Rank']
    df_rank_betting_surface['Rank_Win_Prob'] = rank_model.predict_proba(df_rank_betting_surface[['Rank_Diff']])[:, 1]
    
    # Print information about the model
    log_reg = rank_model.named_steps['log_reg']
    coefficients = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]
    print(f"Rank Difference Polynomial Model ({surface}):")
    print(f"  Intercept: {intercept:.4f}")
    for i, coef in enumerate(coefficients):
        if i == 0:
            print(f"  Rank_Diff: {coef:.4f}")
        elif i == 1:
            print(f"  Rank_Diff^2: {coef:.4f}")
    print(f"  where p = sigmoid(logit) = 1 / (1 + exp(-logit))")
    
    # Plot Rank Difference vs. Win Probability (zoomed in)
    rank_diff_range = np.linspace(-500, 500, 100)
    rank_diff_range_df = pd.DataFrame(rank_diff_range, columns=['Rank_Diff'])
    rank_prob_range = rank_model.predict_proba(rank_diff_range_df)[:, 1]
    
    # Calculate actual probabilities by binning
    bin_size = 50  # Size of each bin
    bin_edges = np.arange(-500, 501, bin_size)
    bin_centers = bin_edges[:-1] + bin_size/2
    actual_probs = []
    bin_counts = []
    
    # Group by rank difference bins and calculate actual win probabilities
    for i in range(len(bin_edges)-1):
        bin_data = train_df_surface[(train_df_surface['Rank_Diff'] >= bin_edges[i]) & 
                                     (train_df_surface['Rank_Diff'] < bin_edges[i+1])]
        if len(bin_data) > 0:
            win_prob = bin_data['Label'].mean()
            actual_probs.append(win_prob)
            bin_counts.append(len(bin_data))
        else:
            actual_probs.append(np.nan)
            bin_counts.append(0)
    
    # Convert to arrays
    actual_probs = np.array(actual_probs)
    bin_counts = np.array(bin_counts)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Scatter plot for raw data (in background, low alpha)
    plt.scatter(train_df_surface['Rank_Diff'], train_df_surface['Label'], alpha=0.05, 
                label='Raw Data Points (0 = P2 Wins, 1 = P1 Wins)', color='lightblue')
    
    # Regression line
    plt.plot(rank_diff_range, rank_prob_range, color='red', 
             label='Logistic Regression Model', linewidth=2)
    
    # Actual probabilities with point size reflecting sample size
    valid_indices = ~np.isnan(actual_probs)
    sc = plt.scatter(bin_centers[valid_indices], actual_probs[valid_indices], 
                   s=np.minimum(bin_counts[valid_indices]/10, 100), color='darkblue', 
                   alpha=0.7, label='Actual Win Probability (binned)')
    
    plt.xlim(-500, 500)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Rank Difference (P2Rank - P1Rank)')
    plt.ylabel('Probability of Player 1 Winning')
    plt.title(f'Rank Difference vs. Win Probability ({surface}, Zoomed In)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotation for a few key points
    for i in range(len(bin_centers)):
        if valid_indices[i] and bin_counts[i] > np.percentile(bin_counts[valid_indices], 75):
            plt.annotate(f'n={bin_counts[i]}', 
                        (bin_centers[i], actual_probs[i]),
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.savefig(os.path.join(specific_output_dir, f'Rank_{surface}', 'rank_diff_regression_zoomed.png'), dpi=300)
    plt.close()

    # More Points
    train_df_points_surface = train_df_surface[(train_df_surface['P1Pts'] > 0) & (train_df_surface['P2Pts'] > 0)].copy()
    train_df_points_surface['Points_Diff'] = train_df_points_surface['P1Pts'] - train_df_points_surface['P2Pts']
    
    # Create a pipeline with polynomial features and logistic regression for points
    points_model_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=polynomial_degree, include_bias=False)),
        ('log_reg', LogisticRegression(C=1.0, solver='liblinear'))
    ])
    
    # Train the model
    points_model_pipeline.fit(train_df_points_surface[['Points_Diff']], train_df_points_surface['Label'])
    points_model = points_model_pipeline  # Use the pipeline as our model
    
    # Apply to the betting data
    df_points_betting_surface['Points_Diff'] = df_points_betting_surface['P1Pts'] - df_points_betting_surface['P2Pts']
    df_points_betting_surface['Points_Win_Prob'] = points_model.predict_proba(df_points_betting_surface[['Points_Diff']])[:, 1]

    # Print information about the points model
    log_reg = points_model.named_steps['log_reg']
    coefficients = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]
    print(f"Points Difference Polynomial Model ({surface}):")
    print(f"  Intercept: {intercept:.4f}")
    for i, coef in enumerate(coefficients):
        if i == 0:
            print(f"  Points_Diff: {coef:.4f}")
        elif i == 1:
            print(f"  Points_Diff^2: {coef:.4f}")
    print(f"  where p = sigmoid(logit) = 1 / (1 + exp(-logit))")

    # Plot Points Difference vs. Win Probability (zoomed in)
    points_diff_range = np.linspace(-5000, 5000, 100)
    points_diff_range_df = pd.DataFrame(points_diff_range, columns=['Points_Diff'])
    points_prob_range = points_model.predict_proba(points_diff_range_df)[:, 1]
    
    # Calculate actual probabilities by binning for points
    bin_size = 500  # Size of each bin for points
    bin_edges = np.arange(-5000, 5001, bin_size)
    bin_centers = bin_edges[:-1] + bin_size/2
    actual_probs = []
    bin_counts = []
    
    # Group by points difference bins and calculate actual win probabilities
    for i in range(len(bin_edges)-1):
        bin_data = train_df_points_surface[(train_df_points_surface['Points_Diff'] >= bin_edges[i]) & 
                                           (train_df_points_surface['Points_Diff'] < bin_edges[i+1])]
        if len(bin_data) > 0:
            win_prob = bin_data['Label'].mean()
            actual_probs.append(win_prob)
            bin_counts.append(len(bin_data))
        else:
            actual_probs.append(np.nan)
            bin_counts.append(0)
    
    # Convert to arrays
    actual_probs = np.array(actual_probs)
    bin_counts = np.array(bin_counts)
    
    # Plot
    plt.figure(figsize=(12, 7))
    
    # Scatter plot for raw data (in background, low alpha)
    plt.scatter(train_df_points_surface['Points_Diff'], train_df_points_surface['Label'], alpha=0.05, 
                label='Raw Data Points (0 = P2 Wins, 1 = P1 Wins)', color='lightblue')
    
    # Regression line
    plt.plot(points_diff_range, points_prob_range, color='red', 
             label='Logistic Regression Model', linewidth=2)
    
    # Actual probabilities with point size reflecting sample size
    valid_indices = ~np.isnan(actual_probs)
    sc = plt.scatter(bin_centers[valid_indices], actual_probs[valid_indices], 
                   s=np.minimum(bin_counts[valid_indices]/10, 100), color='darkblue', 
                   alpha=0.7, label='Actual Win Probability (binned)')
    
    plt.xlim(-5000, 5000)
    plt.ylim(-0.05, 1.05)
    plt.xlabel('Points Difference (P1Pts - P2Pts)')
    plt.ylabel('Probability of Player 1 Winning')
    plt.title(f'Points Difference vs. Win Probability ({surface}, Zoomed In)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add sample size annotation for a few key points
    for i in range(len(bin_centers)):
        if valid_indices[i] and bin_counts[i] > np.percentile(bin_counts[valid_indices], 75):
            plt.annotate(f'n={bin_counts[i]}', 
                        (bin_centers[i], actual_probs[i]),
                        textcoords="offset points",
                        xytext=(0,10), 
                        ha='center',
                        fontsize=8)
    
    plt.savefig(os.path.join(specific_output_dir, f'Points_{surface}', 'points_diff_regression_zoomed.png'), dpi=300)
    plt.close()
    
    # Define predictions with edge case handling
    df_rank_betting_surface['Same_Rank'] = df_rank_betting_surface['P1Rank'] == df_rank_betting_surface['P2Rank']
    df_rank_betting_surface['Missing_Rank'] = (df_rank_betting_surface['P1Rank'].isna()) | (df_rank_betting_surface['P2Rank'].isna()) | (df_rank_betting_surface['P1Rank'] <= 0) | (df_rank_betting_surface['P2Rank'] <= 0)
    df_rank_betting_surface['Valid_Rank_Bet'] = ~(df_rank_betting_surface['Same_Rank'] | df_rank_betting_surface['Missing_Rank'])
    df_rank_betting_surface['Rank_Pred'] = np.where(df_rank_betting_surface['Valid_Rank_Bet'], df_rank_betting_surface['P1Rank'] < df_rank_betting_surface['P2Rank'], np.nan)

    df_points_betting_surface['Same_Points'] = df_points_betting_surface['P1Pts'] == df_points_betting_surface['P2Pts']
    df_points_betting_surface['Missing_Points'] = (df_points_betting_surface['P1Pts'].isna()) | (df_points_betting_surface['P2Pts'].isna()) | (df_points_betting_surface['P1Pts'] <= 0) | (df_points_betting_surface['P2Pts'] <= 0)
    df_points_betting_surface['Valid_Points_Bet'] = ~(df_points_betting_surface['Same_Points'] | df_points_betting_surface['Missing_Points'])
    df_points_betting_surface['Points_Pred'] = np.where(df_points_betting_surface['Valid_Points_Bet'], df_points_betting_surface['P1Pts'] > df_points_betting_surface['P2Pts'], np.nan)

    df_odds_surface['Same_Odds'] = df_odds_surface['P1_Odds'] == df_odds_surface['P2_Odds']
    df_odds_surface['Missing_Odds'] = (df_odds_surface['P1_Odds'].isna()) | (df_odds_surface['P2_Odds'].isna()) | (df_odds_surface['P1_Odds'] <= 1.0) | (df_odds_surface['P2_Odds'] <= 1.0)
    df_odds_surface['Valid_Odds_Bet'] = ~(df_odds_surface['Same_Odds'] | df_odds_surface['Missing_Odds'])
    df_odds_surface['Favorite_Pred'] = np.where(df_odds_surface['Valid_Odds_Bet'], df_odds_surface['P1_Odds'] < df_odds_surface['P2_Odds'], np.nan)
    
    # Run simulations
    rank_flat_results = simulate_betting(df_rank_betting_surface, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="Higher Rank Flat")
    rank_prop_results = simulate_betting(df_rank_betting_surface, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Proportional")
    rank_adj_prop_results = simulate_betting(df_rank_betting_surface, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Adjusted Proportional", use_adjusted=True)
    rank_kelly_results = simulate_betting(df_rank_betting_surface, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Half Kelly (max 5%)", use_kelly=True)
    points_flat_results = simulate_betting(df_points_betting_surface, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="More Points Flat")
    points_prop_results = simulate_betting(df_points_betting_surface, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Proportional")
    points_adj_prop_results = simulate_betting(df_points_betting_surface, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Adjusted Proportional", use_adjusted=True)
    points_kelly_results = simulate_betting(df_points_betting_surface, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Half Kelly (max 5%)", use_kelly=True)
    favorite_results = simulate_betting(df_odds_surface, 'Favorite_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="Favorite Flat")
    
    # Store results
    surface_results[surface] = {
        'rank_acc': surface_rank_acc,
        'points_acc': surface_points_acc,
        'favorite_acc': surface_favorite_acc,
        'rank_flat_results': rank_flat_results,
        'rank_prop_results': rank_prop_results,
        'rank_adj_prop_results': rank_adj_prop_results,
        'rank_kelly_results': rank_kelly_results,
        'points_flat_results': points_flat_results,
        'points_prop_results': points_prop_results,
        'points_adj_prop_results': points_adj_prop_results,
        'points_kelly_results': points_kelly_results,
        'favorite_results': favorite_results,
    }

    # Store prediction accuracies for surface-specific results
    surface_rank_pred_acc[surface] = rank_flat_results[7] if len(rank_flat_results) > 7 else None  # Prediction accuracy for Higher Rank
    surface_points_pred_acc[surface] = points_flat_results[7] if len(points_flat_results) > 7 else None  # Prediction accuracy for More Points
    surface_favorite_pred_acc[surface] = favorite_results[7] if len(favorite_results) > 7 else None  # Prediction accuracy for Favorite

# Define predictions (overall) with edge case handling
df_rank_betting['Same_Rank'] = df_rank_betting['P1Rank'] == df_rank_betting['P2Rank']
df_rank_betting['Missing_Rank'] = (df_rank_betting['P1Rank'].isna()) | (df_rank_betting['P2Rank'].isna()) | (df_rank_betting['P1Rank'] <= 0) | (df_rank_betting['P2Rank'] <= 0)
df_rank_betting['Valid_Rank_Bet'] = ~(df_rank_betting['Same_Rank'] | df_rank_betting['Missing_Rank'])
df_rank_betting['Rank_Pred'] = np.where(df_rank_betting['Valid_Rank_Bet'], df_rank_betting['P1Rank'] < df_rank_betting['P2Rank'], np.nan)

df_points_betting['Same_Points'] = df_points_betting['P1Pts'] == df_points_betting['P2Pts']
df_points_betting['Missing_Points'] = (df_points_betting['P1Pts'].isna()) | (df_points_betting['P2Pts'].isna()) | (df_points_betting['P1Pts'] <= 0) | (df_points_betting['P2Pts'] <= 0)
df_points_betting['Valid_Points_Bet'] = ~(df_points_betting['Same_Points'] | df_points_betting['Missing_Points'])
df_points_betting['Points_Pred'] = np.where(df_points_betting['Valid_Points_Bet'], df_points_betting['P1Pts'] > df_points_betting['P2Pts'], np.nan)

df_odds['Same_Odds'] = df_odds['P1_Odds'] == df_odds['P2_Odds']
df_odds['Missing_Odds'] = (df_odds['P1_Odds'].isna()) | (df_odds['P2_Odds'].isna()) | (df_odds['P1_Odds'] <= 1.0) | (df_odds['P2_Odds'] <= 1.0)
df_odds['Valid_Odds_Bet'] = ~(df_odds['Same_Odds'] | df_odds['Missing_Odds'])
df_odds['Favorite_Pred'] = np.where(df_odds['Valid_Odds_Bet'], df_odds['P1_Odds'] < df_odds['P2_Odds'], np.nan)

# Run overall simulations
overall_rank_flat_results = simulate_betting(df_rank_betting, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="Higher Rank Flat")
overall_rank_prop_results = simulate_betting(df_rank_betting, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Proportional")
overall_rank_adj_prop_results = simulate_betting(df_rank_betting, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Adjusted Proportional", use_adjusted=True)
overall_rank_kelly_results = simulate_betting(df_rank_betting, 'Rank_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Rank_Win_Prob', strategy_name="Higher Rank Half Kelly (max 5%)", use_kelly=True)
overall_points_flat_results = simulate_betting(df_points_betting, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="More Points Flat")
overall_points_prop_results = simulate_betting(df_points_betting, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Proportional")
overall_points_adj_prop_results = simulate_betting(df_points_betting, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Adjusted Proportional", use_adjusted=True)
overall_points_kelly_results = simulate_betting(df_points_betting, 'Points_Pred', 'P1_Odds', 'P2_Odds', 'Label', prob_col='Points_Win_Prob', strategy_name="More Points Half Kelly (max 5%)", use_kelly=True)
overall_favorite_results = simulate_betting(df_odds, 'Favorite_Pred', 'P1_Odds', 'P2_Odds', 'Label', strategy_name="Favorite Flat")

# Output overall results
print("\n========== OVERALL BASELINE RESULTS ==========")
print("Training Set (2000-2022):")
print(f"Higher Rank Accuracy: {overall_rank_acc_train:.4f} ({len(train_df)} matches)")
print(f"More Points Accuracy: {overall_points_acc_train:.4f} ({len(train_df_points)} matches)")
print(f"Favorite Accuracy: {overall_favorite_acc_train:.4f} ({len(train_df_odds)} matches)")
print("\nTesting Set (2023-2025):")
print(f"Higher Rank Accuracy: {overall_rank_acc_test:.4f} ({len(test_df)} matches)")
print(f"More Points Accuracy: {overall_points_acc_test:.4f} ({len(df_points)} matches)")
print(f"Favorite Accuracy: {overall_favorite_acc_test:.4f} ({len(df_odds)} matches)")

print("\n========== YEARLY BASELINE ACCURACY (TEST SET) ==========")
print(yearly_df_test.to_string(index=False))

# Create a formatted table for overall betting results
print("\n========== OVERALL BETTING SIMULATION RESULTS (TEST SET) ==========")
print("Starting Bankroll: $100,000")
print("Note: 'Pred Acc (All Matches)' is the same for all strategies within a prediction method (Higher Rank, More Points, Favorite) because they use the same prediction logic, differing only in betting decisions.")
overall_results_df = pd.DataFrame({
    'Strategy': [
        'Higher Rank (Flat)', 'Higher Rank (Proportional)', 'Higher Rank (Adjusted Prop.)', 'Higher Rank (Half Kelly max 5%)',
        'More Points (Flat)', 'More Points (Proportional)', 'More Points (Adjusted Prop.)', 'More Points (Half Kelly max 5%)',
        'Favorite (Flat)'
    ],
    'Bankroll': [
        f"${overall_rank_flat_results[0]:.2f}", f"${overall_rank_prop_results[0]:.2f}", 
        f"${overall_rank_adj_prop_results[0]:.2f}", f"${overall_rank_kelly_results[0]:.2f}",
        f"${overall_points_flat_results[0]:.2f}", f"${overall_points_prop_results[0]:.2f}", 
        f"${overall_points_adj_prop_results[0]:.2f}", f"${overall_points_kelly_results[0]:.2f}",
        f"${overall_favorite_results[0]:.2f}"
    ],
    'ROI': [
        f"{overall_rank_flat_results[1]:.4f}", f"{overall_rank_prop_results[1]:.4f}", 
        f"{overall_rank_adj_prop_results[1]:.4f}", f"{overall_rank_kelly_results[1]:.4f}",
        f"{overall_points_flat_results[1]:.4f}", f"{overall_points_prop_results[1]:.4f}", 
        f"{overall_points_adj_prop_results[1]:.4f}", f"{overall_points_kelly_results[1]:.4f}",
        f"{overall_favorite_results[1]:.4f}"
    ],
    'Win Rate (Betting)': [
        f"{overall_rank_flat_results[2]:.4f}", f"{overall_rank_prop_results[2]:.4f}", 
        f"{overall_rank_adj_prop_results[2]:.4f}", f"{overall_rank_kelly_results[2]:.4f}",
        f"{overall_points_flat_results[2]:.4f}", f"{overall_points_prop_results[2]:.4f}", 
        f"{overall_points_adj_prop_results[2]:.4f}", f"{overall_points_kelly_results[2]:.4f}",
        f"{overall_favorite_results[2]:.4f}"
    ],
    'Pred Acc (All Matches)': [
        f"{overall_rank_flat_results[7]:.4f}", f"{overall_rank_prop_results[7]:.4f}", 
        f"{overall_rank_adj_prop_results[7]:.4f}", f"{overall_rank_kelly_results[7]:.4f}",
        f"{overall_points_flat_results[7]:.4f}", f"{overall_points_prop_results[7]:.4f}", 
        f"{overall_points_adj_prop_results[7]:.4f}", f"{overall_points_kelly_results[7]:.4f}",
        f"{overall_favorite_results[7]:.4f}"
    ],
    'Bets': [
        f"{overall_rank_flat_results[3]}", f"{overall_rank_prop_results[3]}", 
        f"{overall_rank_adj_prop_results[3]}", f"{overall_rank_kelly_results[3]}",
        f"{overall_points_flat_results[3]}", f"{overall_points_prop_results[3]}", 
        f"{overall_points_adj_prop_results[3]}", f"{overall_points_kelly_results[3]}",
        f"{overall_favorite_results[3]}"
    ],
    'Wins': [
        f"{overall_rank_flat_results[4]}", f"{overall_rank_prop_results[4]}", 
        f"{overall_rank_adj_prop_results[4]}", f"{overall_rank_kelly_results[4]}",
        f"{overall_points_flat_results[4]}", f"{overall_points_prop_results[4]}", 
        f"{overall_points_adj_prop_results[4]}", f"{overall_points_kelly_results[4]}",
        f"{overall_favorite_results[4]}"
    ],
    'Losses': [
        f"{overall_rank_flat_results[5]}", f"{overall_rank_prop_results[5]}", 
        f"{overall_rank_adj_prop_results[5]}", f"{overall_rank_kelly_results[5]}",
        f"{overall_points_flat_results[5]}", f"{overall_points_prop_results[5]}", 
        f"{overall_points_adj_prop_results[5]}", f"{overall_points_kelly_results[5]}",
        f"{overall_favorite_results[5]}"
    ]
})

# Print the formatted table
print(overall_results_df.to_string(index=False))

# Output surface-specific results
for surface in surfaces:
    results = surface_results[surface]
    print(f"\n========== SURFACE-SPECIFIC RESULTS: {surface} ==========")
    print(f"Higher Rank Accuracy: {results['rank_acc']:.4f} ({len(test_df[test_df['Surface'] == surface])} matches)")
    print(f"More Points Accuracy: {results['points_acc']:.4f} ({len(df_points[df_points['Surface'] == surface])} matches)")
    print(f"Favorite Accuracy: {results['favorite_acc']:.4f} ({len(df_odds[df_odds['Surface'] == surface])} matches)")
    
    # Create a formatted table for surface-specific betting results
    print(f"\nBetting Simulation Results ({surface}):")
    print("Starting Bankroll: $100,000")
    print("Note: 'Pred Acc (All Matches)' is the same for all strategies within a prediction method (Higher Rank, More Points, Favorite) because they use the same prediction logic, differing only in betting decisions.")
    surface_results_df = pd.DataFrame({
        'Strategy': [
            'Higher Rank (Flat)', 'Higher Rank (Proportional)', 'Higher Rank (Adjusted Prop.)', 'Higher Rank (Half Kelly max 5%)',
            'More Points (Flat)', 'More Points (Proportional)', 'More Points (Adjusted Prop.)', 'More Points (Half Kelly max 5%)',
            'Favorite (Flat)'
        ],
        'Bankroll': [
            f"${results['rank_flat_results'][0]:.2f}", f"${results['rank_prop_results'][0]:.2f}", 
            f"${results['rank_adj_prop_results'][0]:.2f}", f"${results['rank_kelly_results'][0]:.2f}",
            f"${results['points_flat_results'][0]:.2f}", f"${results['points_prop_results'][0]:.2f}", 
            f"${results['points_adj_prop_results'][0]:.2f}", f"${results['points_kelly_results'][0]:.2f}",
            f"${results['favorite_results'][0]:.2f}"
        ],
        'ROI': [
            f"{results['rank_flat_results'][1]:.4f}", f"{results['rank_prop_results'][1]:.4f}", 
            f"{results['rank_adj_prop_results'][1]:.4f}", f"{results['rank_kelly_results'][1]:.4f}",
            f"{results['points_flat_results'][1]:.4f}", f"{results['points_prop_results'][1]:.4f}", 
            f"{results['points_adj_prop_results'][1]:.4f}", f"{results['points_kelly_results'][1]:.4f}",
            f"{results['favorite_results'][1]:.4f}"
        ],
        'Win Rate (Betting)': [
            f"{results['rank_flat_results'][2]:.4f}", f"{results['rank_prop_results'][2]:.4f}", 
            f"{results['rank_adj_prop_results'][2]:.4f}", f"{results['rank_kelly_results'][2]:.4f}",
            f"{results['points_flat_results'][2]:.4f}", f"{results['points_prop_results'][2]:.4f}", 
            f"{results['points_adj_prop_results'][2]:.4f}", f"{results['points_kelly_results'][2]:.4f}",
            f"{results['favorite_results'][2]:.4f}"
        ],
        'Pred Acc (All Matches)': [
            f"{surface_rank_pred_acc[surface]:.4f}", f"{surface_rank_pred_acc[surface]:.4f}", 
            f"{surface_rank_pred_acc[surface]:.4f}", f"{surface_rank_pred_acc[surface]:.4f}",
            f"{surface_points_pred_acc[surface]:.4f}", f"{surface_points_pred_acc[surface]:.4f}", 
            f"{surface_points_pred_acc[surface]:.4f}", f"{surface_points_pred_acc[surface]:.4f}",
            f"{surface_favorite_pred_acc[surface]:.4f}"
        ],
        'Bets': [
            f"{results['rank_flat_results'][3]}", f"{results['rank_prop_results'][3]}", 
            f"{results['rank_adj_prop_results'][3]}", f"{results['rank_kelly_results'][3]}",
            f"{results['points_flat_results'][3]}", f"{results['points_prop_results'][3]}", 
            f"{results['points_adj_prop_results'][3]}", f"{results['points_kelly_results'][3]}",
            f"{results['favorite_results'][3]}"
        ],
        'Wins': [
            f"{results['rank_flat_results'][4]}", f"{results['rank_prop_results'][4]}", 
            f"{results['rank_adj_prop_results'][4]}", f"{results['rank_kelly_results'][4]}",
            f"{results['points_flat_results'][4]}", f"{results['points_prop_results'][4]}", 
            f"{results['points_adj_prop_results'][4]}", f"{results['points_kelly_results'][4]}",
            f"{results['favorite_results'][4]}"
        ],
        'Losses': [
            f"{results['rank_flat_results'][5]}", f"{results['rank_prop_results'][5]}", 
            f"{results['rank_adj_prop_results'][5]}", f"{results['rank_kelly_results'][5]}",
            f"{results['points_flat_results'][5]}", f"{results['points_prop_results'][5]}", 
            f"{results['points_adj_prop_results'][5]}", f"{results['points_kelly_results'][5]}",
            f"{results['favorite_results'][5]}"
        ]
    })
    
    # Print the formatted table
    print(surface_results_df.to_string(index=False))

# Create and format tabular results
# We'll load the metrics from the metrics files for more comprehensive data
metrics_files = []
strategies = [
    'higher_rank_flat', 'higher_rank_proportional', 'higher_rank_adjusted_proportional', 'higher_rank_half_kelly',
    'more_points_flat', 'more_points_proportional', 'more_points_adjusted_proportional', 'more_points_half_kelly',
    'favorite_flat'
]

# Basic metrics from simulation results
basic_metrics = pd.DataFrame({
    'Strategy': [
        'Higher Rank (Flat)', 'Higher Rank (Proportional)', 'Higher Rank (Adjusted Proportional)', 'Higher Rank (Half Kelly max 5%)',
        'More Points (Flat)', 'More Points (Proportional)', 'More Points (Adjusted Proportional)', 'More Points (Half Kelly max 5%)',
        'Favorite (Flat)'
    ],
    'EV Cutoff': [
        'N/A', '0% Edge', '0% Edge', '10% Edge',
        'N/A', '0% Edge', '0% Edge', '10% Edge',
        'N/A'
    ],
    'Final Bankroll': [
        overall_rank_flat_results[0], overall_rank_prop_results[0], overall_rank_adj_prop_results[0], overall_rank_kelly_results[0],
        overall_points_flat_results[0], overall_points_prop_results[0], overall_points_adj_prop_results[0], overall_points_kelly_results[0],
        overall_favorite_results[0]
    ],
    'ROI': [
        overall_rank_flat_results[1], overall_rank_prop_results[1], overall_rank_adj_prop_results[1], overall_rank_kelly_results[1],
        overall_points_flat_results[1], overall_points_prop_results[1], overall_points_adj_prop_results[1], overall_points_kelly_results[1],
        overall_favorite_results[1]
    ],
    'Win Rate': [
        overall_rank_flat_results[2], overall_rank_prop_results[2], overall_rank_adj_prop_results[2], overall_rank_kelly_results[2],
        overall_points_flat_results[2], overall_points_prop_results[2], overall_points_adj_prop_results[2], overall_points_kelly_results[2],
        overall_favorite_results[2]
    ],
    'Bets Made': [
        overall_rank_flat_results[3], overall_rank_prop_results[3], overall_rank_adj_prop_results[3], overall_rank_kelly_results[3],
        overall_points_flat_results[3], overall_points_prop_results[3], overall_points_adj_prop_results[3], overall_points_kelly_results[3],
        overall_favorite_results[3]
    ]
})

# Create an enhanced summary from the metrics files, if they've been created previously
try:
    # Try to load all the metrics files and combine them
    folder = "Rank_Overall"
    metrics_dfs = []
    for strategy in strategies:
        try:
            metrics_file = os.path.join(specific_output_dir, folder, f'metrics_{strategy}.csv')
            if os.path.exists(metrics_file):
                df = pd.read_csv(metrics_file)
                metrics_dfs.append(df)
        except Exception as e:
            print(f"Error loading metrics for {strategy}: {e}")
    
    if metrics_dfs:
        # Combine all metrics dataframes
        all_metrics = pd.concat(metrics_dfs, ignore_index=True)
        
        # Create a comprehensive summary with all the rich metrics
        comprehensive_summary = pd.DataFrame({
            'Strategy': basic_metrics['Strategy'],
            'EV Cutoff': basic_metrics['EV Cutoff'],
            'Final Bankroll': basic_metrics['Final Bankroll'],
            'Profit': all_metrics['Profit'] if 'Profit' in all_metrics.columns else None,
            'ROI': basic_metrics['ROI'],
            'Win Rate': basic_metrics['Win Rate'],
            'Prediction Accuracy': all_metrics['Prediction_Accuracy'] if 'Prediction_Accuracy' in all_metrics.columns else None,
            'Bets Made': basic_metrics['Bets Made'],
            
            # Betting metrics
            'Win/Loss Ratio': all_metrics['Win_Loss_Ratio'] if 'Win_Loss_Ratio' in all_metrics.columns else None,
            'Avg Win': all_metrics['Avg_Win'] if 'Avg_Win' in all_metrics.columns else None,
            'Avg Loss': all_metrics['Avg_Loss'] if 'Avg_Loss' in all_metrics.columns else None,
            'Profit Factor': all_metrics['Profit_Factor'] if 'Profit_Factor' in all_metrics.columns else None,
            
            # Risk metrics
            'Max Drawdown': all_metrics['Max_Drawdown'] if 'Max_Drawdown' in all_metrics.columns else None,
            'Volatility': all_metrics['Volatility'] if 'Volatility' in all_metrics.columns else None,
            'Risk of Ruin': all_metrics['Risk_of_Ruin'] if 'Risk_of_Ruin' in all_metrics.columns else None,
            'Z-Score': all_metrics['Z_Score'] if 'Z_Score' in all_metrics.columns else None,
            'Ulcer Index': all_metrics['Ulcer_Index'] if 'Ulcer_Index' in all_metrics.columns else None,
            
            # Risk-adjusted performance ratios
            'Sharpe Ratio': all_metrics['Sharpe_Ratio'] if 'Sharpe_Ratio' in all_metrics.columns else None,
            'Sortino Ratio': all_metrics['Sortino_Ratio'] if 'Sortino_Ratio' in all_metrics.columns else None,
            'Calmar Ratio': all_metrics['Calmar_Ratio'] if 'Calmar_Ratio' in all_metrics.columns else None
        })
        
        # Save the comprehensive summary
        comprehensive_summary.to_csv(os.path.join(specific_output_dir, "comprehensive_betting_results.csv"), index=False)
        
        # For the standard overall_summary, use a simpler version with key metrics
        overall_summary = comprehensive_summary[['Strategy', 'EV Cutoff', 'Final Bankroll', 'ROI', 'Win Rate', 
                                                'Bets Made', 'Max Drawdown', 'Risk of Ruin', 'Profit Factor',
                                                'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']]
    else:
        # If metrics files don't exist yet, use the basic metrics
        overall_summary = basic_metrics
except Exception as e:
    print(f"Error creating comprehensive summary: {e}")
    # Fall back to basic metrics
    overall_summary = basic_metrics

# Save overall results summary
overall_summary.to_csv(os.path.join(specific_output_dir, "overall_betting_results.csv"), index=False)

# Create surface-specific summaries
for surface in surfaces:
    results = surface_results[surface]
    
    # Create basic metrics for this surface
    basic_surface_metrics = pd.DataFrame({
        'Strategy': [
            'Higher Rank (Flat)', 'Higher Rank (Proportional)', 'Higher Rank (Adjusted Proportional)', 'Higher Rank (Half Kelly max 5%)',
            'More Points (Flat)', 'More Points (Proportional)', 'More Points (Adjusted Proportional)', 'More Points (Half Kelly max 5%)',
            'Favorite (Flat)'
        ],
        'EV Cutoff': [
            'N/A', '0% Edge', '0% Edge', '10% Edge',
            'N/A', '0% Edge', '0% Edge', '10% Edge',
            'N/A'
        ],
        'Final Bankroll': [
            results['rank_flat_results'][0], results['rank_prop_results'][0], results['rank_adj_prop_results'][0], results['rank_kelly_results'][0],
            results['points_flat_results'][0], results['points_prop_results'][0], results['points_adj_prop_results'][0], results['points_kelly_results'][0],
            results['favorite_results'][0]
        ],
        'ROI': [
            results['rank_flat_results'][1], results['rank_prop_results'][1], results['rank_adj_prop_results'][1], results['rank_kelly_results'][1],
            results['points_flat_results'][1], results['points_prop_results'][1], results['points_adj_prop_results'][1], results['points_kelly_results'][1],
            results['favorite_results'][1]
        ],
        'Win Rate': [
            results['rank_flat_results'][2], results['rank_prop_results'][2], results['rank_adj_prop_results'][2], results['rank_kelly_results'][2],
            results['points_flat_results'][2], results['points_prop_results'][2], results['points_adj_prop_results'][2], results['points_kelly_results'][2],
            results['favorite_results'][2]
        ],
        'Bets Made': [
            results['rank_flat_results'][3], results['rank_prop_results'][3], results['rank_adj_prop_results'][3], results['rank_kelly_results'][3],
            results['points_flat_results'][3], results['points_prop_results'][3], results['points_adj_prop_results'][3], results['points_kelly_results'][3],
            results['favorite_results'][3]
        ]
    })
    
    # Try to read metrics files for this surface, if they exist
    try:
        folder = f"Rank_{surface}"
        metrics_dfs = []
        for strategy in strategies:
            try:
                metrics_file = os.path.join(specific_output_dir, folder, f'metrics_{strategy}.csv')
                if os.path.exists(metrics_file):
                    df = pd.read_csv(metrics_file)
                    metrics_dfs.append(df)
            except Exception as e:
                # Silently continue if file doesn't exist
                pass
        
        if metrics_dfs:
            # Combine all metrics dataframes
            all_metrics = pd.concat(metrics_dfs, ignore_index=True)
            
            # Create a comprehensive summary with all the rich metrics
            comprehensive_summary = pd.DataFrame({
                'Strategy': basic_surface_metrics['Strategy'],
                'EV Cutoff': basic_surface_metrics['EV Cutoff'],
                'Final Bankroll': basic_surface_metrics['Final Bankroll'],
                'Profit': all_metrics['Profit'] if 'Profit' in all_metrics.columns else None,
                'ROI': basic_surface_metrics['ROI'],
                'Win Rate': basic_surface_metrics['Win Rate'],
                'Prediction Accuracy': all_metrics['Prediction_Accuracy'] if 'Prediction_Accuracy' in all_metrics.columns else None,
                'Bets Made': basic_surface_metrics['Bets Made'],
                
                # Betting metrics
                'Win/Loss Ratio': all_metrics['Win_Loss_Ratio'] if 'Win_Loss_Ratio' in all_metrics.columns else None,
                'Avg Win': all_metrics['Avg_Win'] if 'Avg_Win' in all_metrics.columns else None,
                'Avg Loss': all_metrics['Avg_Loss'] if 'Avg_Loss' in all_metrics.columns else None,
                'Profit Factor': all_metrics['Profit_Factor'] if 'Profit_Factor' in all_metrics.columns else None,
                
                # Risk metrics
                'Max Drawdown': all_metrics['Max_Drawdown'] if 'Max_Drawdown' in all_metrics.columns else None,
                'Volatility': all_metrics['Volatility'] if 'Volatility' in all_metrics.columns else None,
                'Risk of Ruin': all_metrics['Risk_of_Ruin'] if 'Risk_of_Ruin' in all_metrics.columns else None,
                'Z-Score': all_metrics['Z_Score'] if 'Z_Score' in all_metrics.columns else None,
                'Ulcer Index': all_metrics['Ulcer_Index'] if 'Ulcer_Index' in all_metrics.columns else None,
                
                # Risk-adjusted performance ratios
                'Sharpe Ratio': all_metrics['Sharpe_Ratio'] if 'Sharpe_Ratio' in all_metrics.columns else None,
                'Sortino Ratio': all_metrics['Sortino_Ratio'] if 'Sortino_Ratio' in all_metrics.columns else None,
                'Calmar Ratio': all_metrics['Calmar_Ratio'] if 'Calmar_Ratio' in all_metrics.columns else None
            })
            
            # Save the comprehensive summary
            comprehensive_summary.to_csv(os.path.join(specific_output_dir, f"comprehensive_{surface.lower()}_betting_results.csv"), index=False)
            
            # For the surface summary, use a simpler version with key metrics
            surface_summary = comprehensive_summary[['Strategy', 'EV Cutoff', 'Final Bankroll', 'ROI', 'Win Rate', 
                                                     'Bets Made', 'Max Drawdown', 'Risk of Ruin', 'Profit Factor',
                                                     'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio']]
        else:
            # If metrics files don't exist yet, use the basic metrics with max drawdown and sharpe ratio from the results
            surface_summary = pd.DataFrame({
                'Strategy': basic_surface_metrics['Strategy'],
                'EV Cutoff': basic_surface_metrics['EV Cutoff'],
                'Final Bankroll': basic_surface_metrics['Final Bankroll'],
                'ROI': basic_surface_metrics['ROI'],
                'Win Rate': basic_surface_metrics['Win Rate'],
                'Bets Made': basic_surface_metrics['Bets Made'],
                'Max Drawdown': [
                    results['rank_flat_results'][8] if len(results['rank_flat_results']) > 8 else None, 
                    results['rank_prop_results'][8] if len(results['rank_prop_results']) > 8 else None, 
                    results['rank_adj_prop_results'][8] if len(results['rank_adj_prop_results']) > 8 else None, 
                    results['rank_kelly_results'][8] if len(results['rank_kelly_results']) > 8 else None,
                    results['points_flat_results'][8] if len(results['points_flat_results']) > 8 else None, 
                    results['points_prop_results'][8] if len(results['points_prop_results']) > 8 else None, 
                    results['points_adj_prop_results'][8] if len(results['points_adj_prop_results']) > 8 else None, 
                    results['points_kelly_results'][8] if len(results['points_kelly_results']) > 8 else None,
                    results['favorite_results'][8] if len(results['favorite_results']) > 8 else None
                ],
                'Sharpe Ratio': [
                    results['rank_flat_results'][9] if len(results['rank_flat_results']) > 9 else None, 
                    results['rank_prop_results'][9] if len(results['rank_prop_results']) > 9 else None, 
                    results['rank_adj_prop_results'][9] if len(results['rank_adj_prop_results']) > 9 else None, 
                    results['rank_kelly_results'][9] if len(results['rank_kelly_results']) > 9 else None,
                    results['points_flat_results'][9] if len(results['points_flat_results']) > 9 else None, 
                    results['points_prop_results'][9] if len(results['points_prop_results']) > 9 else None, 
                    results['points_adj_prop_results'][9] if len(results['points_adj_prop_results']) > 9 else None, 
                    results['points_kelly_results'][9] if len(results['points_kelly_results']) > 9 else None,
                    results['favorite_results'][9] if len(results['favorite_results']) > 9 else None
                ]
            })
    except Exception as e:
        # Fall back to basic metrics
        surface_summary = basic_surface_metrics
    
    # Save the surface summary
    surface_summary.to_csv(os.path.join(specific_output_dir, f"{surface.lower()}_betting_results.csv"), index=False)

# Save yearly stats to CSV (test set)
yearly_df_test.to_csv(os.path.join(specific_output_dir, "yearly_baseline_stats_test.csv"), index=False)

# Save yearly stats to CSV (training set, for reference)
yearly_df_train.to_csv(os.path.join(specific_output_dir, "yearly_baseline_stats_train.csv"), index=False)

# Create a readme file to explain the directory structure
readme_content = """# Rank and Points Betting Analysis

This directory contains the results of betting simulations based on player ranks and ATP points.

## Directory Structure:
- **Rank_Overall/**: Analysis and plots for rank-based betting strategies across all surfaces
- **Rank_Hard/**: Analysis and plots for rank-based betting on hard courts
- **Rank_Clay/**: Analysis and plots for rank-based betting on clay courts
- **Rank_Grass/**: Analysis and plots for rank-based betting on grass courts
- **Points_Overall/**: Analysis and plots for points-based betting strategies across all surfaces
- **Points_Hard/**: Analysis and plots for points-based betting on hard courts
- **Points_Clay/**: Analysis and plots for points-based betting on clay courts
- **Points_Grass/**: Analysis and plots for points-based betting on grass courts

## CSV Files:
- **yearly_baseline_stats_test.csv**: Year-by-year accuracy for test set (2020-2025) for different prediction methods
- **yearly_baseline_stats_train.csv**: Year-by-year accuracy for training set (2000-2019) for reference
- **overall_betting_results.csv**: Summary of all betting strategies across all surfaces (test set)
- **hard_betting_results.csv**: Summary of all betting strategies on hard courts (test set)
- **clay_betting_results.csv**: Summary of all betting strategies on clay courts (test set)
- **grass_betting_results.csv**: Summary of all betting strategies on grass courts (test set)

## Betting Strategies:
- **Flat**: Constant bet size regardless of confidence
- **Proportional**: Bet size proportional to predicted win probability, only when p * odds > 1.0 (positive EV)
- **Adjusted Proportional**: Bet only when probability > 0.5, size proportional to (prob - 0.5), only when p * odds > 1.0 (positive EV)
- **Half Kelly (max 5%)**: Uses the Kelly Criterion with a half-Kelly fraction for bankroll management and a 5% maximum bet size, betting only when p * odds > 1 (positive edge)
"""

with open(os.path.join(specific_output_dir, "README.md"), 'w') as f:
    f.write(readme_content)

print(f"\nResults saved to: {specific_output_dir}")
print("Data is organized into folders by prediction method (Rank/Points) and surface type.")
print("Summary CSV files have been created for easier analysis.")