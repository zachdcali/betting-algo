import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LogisticRegression

# Define output directory
base_output_dir = "/app/data/output"
specific_output_dir = os.path.join(base_output_dir, "Rank_Heuristic_Baseline")
os.makedirs(specific_output_dir, exist_ok=True)

# Define surfaces
surfaces = ['Hard', 'Clay', 'Grass']

# Create subdirectories for better organization
folders = ['Rank_Overall', 'Rank_Hard', 'Rank_Clay', 'Rank_Grass']
for folder in folders:
    os.makedirs(os.path.join(specific_output_dir, folder), exist_ok=True)

def simulate_betting(data, pred_col, odds_p1_col, odds_p2_col, label_col, strategy_name="Flat"):
    """
    Simulate betting based on predictions and odds
    """
    # Constants
    initial_bankroll = 100000  # Starting bankroll
    bankroll = initial_bankroll
    base_bet = 1000  # 1% of initial bankroll for flat betting
    
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
    
    for idx, row in data.iterrows():
        prediction = row[pred_col]
        actual = row[label_col]
        
        # Track overall accuracy
        if prediction == actual:
            correct_predictions += 1
        total_predictions += 1
        
        # Only bet if we have odds data
        if pd.notna(row[odds_p1_col]) and pd.notna(row[odds_p2_col]):
            # Determine which odds to use based on prediction
            if prediction == 1:  # Betting on winner (higher ranked player)
                odds = row[odds_p1_col]
                bet_on = "Winner"
            else:  # Betting on loser (lower ranked player)
                odds = row[odds_p2_col]
                bet_on = "Loser"
            
            # Calculate bet amount (flat betting)
            bet_amount = min(base_bet, bankroll * 0.1)  # Never bet more than 10% of current bankroll
            
            if bet_amount > 0:
                bets += 1
                total_bet += bet_amount
                
                # Check if bet won
                bet_won = (prediction == actual)
                
                if bet_won:
                    winnings = bet_amount * (odds - 1)  # Profit = bet * (odds - 1)
                    bankroll += winnings
                    total_winnings += winnings
                    wins += 1
                else:
                    bankroll -= bet_amount
                
                # Track bankroll history
                bankroll_history.append(bankroll)
                
                # Update max drawdown
                if bankroll > max_bankroll:
                    max_bankroll = bankroll
                else:
                    drawdown = (max_bankroll - bankroll) / max_bankroll
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Store bet details
                bet_details.append({
                    'Date': row['Date'],
                    'Tournament': row['Tournament'],
                    'Winner': row['Winner'],
                    'Loser': row['Loser'],
                    'Prediction': prediction,
                    'Actual': actual,
                    'Bet_On': bet_on,
                    'Odds': odds,
                    'Bet_Amount': bet_amount,
                    'Won': bet_won,
                    'Profit': winnings if bet_won else -bet_amount,
                    'Bankroll': bankroll
                })
    
    # Calculate metrics
    win_rate = wins / bets if bets > 0 else 0
    roi = (bankroll - initial_bankroll) / initial_bankroll
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    avg_odds = data[[odds_p1_col, odds_p2_col]].mean().mean()
    
    # Calculate Sharpe ratio (simplified)
    if len(bankroll_history) > 1:
        returns = np.diff(bankroll_history) / bankroll_history[:-1]
        sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    results = {
        'strategy': strategy_name,
        'total_bets': bets,
        'wins': wins,
        'win_rate': win_rate,
        'total_bet': total_bet,
        'total_winnings': total_winnings,
        'net_profit': bankroll - initial_bankroll,
        'roi': roi,
        'final_bankroll': bankroll,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'overall_accuracy': overall_accuracy,
        'avg_odds': avg_odds,
        'bankroll_history': bankroll_history,
        'bet_details': bet_details
    }
    
    return results

def create_heuristic_predictions(df):
    """
    Create predictions based on ATP ranking heuristic and bookmaker odds
    """
    df = df.copy()
    
    # Create prediction based on ranking (1 = bet on winner, 0 = bet on loser)
    # Winner has lower rank number (better ranking), so we predict 1 when winner rank < loser rank
    df['rank_prediction'] = np.where(
        (df['WRank'] < df['LRank']) & (df['WRank'] != 'NR') & (df['LRank'] != 'NR'),
        1, 0
    )
    
    # Create bookmaker prediction based on odds (1 = bet on winner, 0 = bet on loser)
    # Lower odds = bookmaker favorite, so if AvgW < AvgL, bookmaker favors winner
    df['bookmaker_prediction'] = np.where(
        (pd.notna(df['AvgW']) & pd.notna(df['AvgL']) & (df['AvgW'] < df['AvgL'])),
        1, 0
    )
    
    # Create actual label (1 = winner won, 0 = loser won)
    # Since data is structured with Winner/Loser columns, winner always won
    df['actual_result'] = 1
    
    return df

def run_heuristic_analysis(df, surface_filter=None, surface_name="Overall"):
    """
    Run ranking heuristic analysis for a specific surface or overall
    """
    print(f"\n=== {surface_name} Analysis ===")
    
    # Filter data if surface specified
    if surface_filter:
        df_filtered = df[df['Surface'] == surface_filter].copy()
        print(f"Filtered to {surface_filter} surface: {len(df_filtered)} matches")
    else:
        df_filtered = df.copy()
        print(f"Using all surfaces: {len(df_filtered)} matches")
    
    if len(df_filtered) == 0:
        print(f"No data available for {surface_name}")
        return
    
    # Create predictions
    df_filtered = create_heuristic_predictions(df_filtered)
    
    # Filter out matches without ranking data
    df_filtered = df_filtered[
        (df_filtered['WRank'] != 'NR') & 
        (df_filtered['LRank'] != 'NR') & 
        pd.notna(df_filtered['WRank']) & 
        pd.notna(df_filtered['LRank'])
    ]
    
    print(f"After filtering for ranking data: {len(df_filtered)} matches")
    
    if len(df_filtered) == 0:
        print(f"No matches with ranking data for {surface_name}")
        return
    
    # Convert rankings to numeric
    df_filtered['WRank'] = pd.to_numeric(df_filtered['WRank'], errors='coerce')
    df_filtered['LRank'] = pd.to_numeric(df_filtered['LRank'], errors='coerce')
    
    # Remove matches with NaN rankings
    df_filtered = df_filtered.dropna(subset=['WRank', 'LRank'])
    
    print(f"After converting rankings to numeric: {len(df_filtered)} matches")
    
    # Split into train/test
    train_df = df_filtered[df_filtered['Date'] < '2023-01-01'].copy()
    test_df = df_filtered[df_filtered['Date'] >= '2023-01-01'].copy()
    
    print(f"Training period (2000-2022): {len(train_df)} matches")
    print(f"Testing period (2023-2025): {len(test_df)} matches")
    
    if len(test_df) == 0:
        print("No test data available")
        return
    
    # Filter test data to only include matches with odds (apples-to-apples comparison)
    test_df_with_odds = test_df[
        pd.notna(test_df['AvgW']) & pd.notna(test_df['AvgL'])
    ].copy()
    
    print(f"Test matches with both rankings and odds: {len(test_df_with_odds)} matches")
    
    if len(test_df_with_odds) == 0:
        print("No test data with odds available")
        return
    
    # Calculate bookmaker accuracy on the same test set
    bookmaker_correct = (test_df_with_odds['bookmaker_prediction'] == test_df_with_odds['actual_result']).sum()
    bookmaker_total = len(test_df_with_odds)
    bookmaker_accuracy = bookmaker_correct / bookmaker_total if bookmaker_total > 0 else 0
    
    # Use average odds for betting simulation
    results = simulate_betting(
        test_df_with_odds, 
        'rank_prediction', 
        'AvgW',  # Average odds for winner
        'AvgL',  # Average odds for loser
        'actual_result',
        strategy_name=f"Ranking_Heuristic_{surface_name}"
    )
    
    # Print results
    print(f"\n=== Results for {surface_name} ===")
    print(f"Total bets: {results['total_bets']}")
    print(f"Wins: {results['wins']}")
    print(f"ATP Ranking accuracy: {results['overall_accuracy']:.3f}")
    print(f"Bookmaker accuracy: {bookmaker_accuracy:.3f}")
    print(f"Win rate: {results['win_rate']:.3f}")
    print(f"ROI: {results['roi']:.3f}")
    print(f"Final bankroll: ${results['final_bankroll']:.2f}")
    print(f"Net profit: ${results['net_profit']:.2f}")
    print(f"Max drawdown: {results['max_drawdown']:.3f}")
    print(f"Sharpe ratio: {results['sharpe_ratio']:.3f}")
    
    # Save results
    output_folder = os.path.join(specific_output_dir, f"Rank_{surface_name}")
    
    # Save summary
    summary_df = pd.DataFrame([results])
    summary_df.to_csv(os.path.join(output_folder, "summary_results.csv"), index=False)
    
    # Save detailed bet history
    if results['bet_details']:
        bet_df = pd.DataFrame(results['bet_details'])
        bet_df.to_csv(os.path.join(output_folder, "bet_history.csv"), index=False)
    
    # Create bankroll plot
    plt.figure(figsize=(12, 6))
    plt.plot(results['bankroll_history'])
    plt.title(f'Bankroll Over Time - {surface_name} Ranking Heuristic')
    plt.xlabel('Bet Number')
    plt.ylabel('Bankroll ($)')
    plt.grid(True)
    plt.savefig(os.path.join(output_folder, 'bankroll_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return results

# Load Tennis-Data.co.uk data
print("Loading Tennis-Data.co.uk data...")
df = pd.read_csv("/app/data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv")

# Convert date column
df['Date'] = pd.to_datetime(df['Date'])
df['Year'] = df['Date'].dt.year

print(f"Loaded data with shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Check for ranking data availability
print(f"\nRanking data availability:")
print(f"WRank non-null: {df['WRank'].notna().sum()}")
print(f"LRank non-null: {df['LRank'].notna().sum()}")

# Check for odds data availability
print(f"\nOdds data availability:")
print(f"AvgW non-null: {df['AvgW'].notna().sum()}")
print(f"AvgL non-null: {df['AvgL'].notna().sum()}")

# Run analysis for overall and each surface
print("\n" + "="*50)
print("TENNIS RANKING HEURISTIC BASELINE ANALYSIS")
print("="*50)

# Overall analysis
overall_results = run_heuristic_analysis(df, surface_name="Overall")

# Surface-specific analysis
for surface in surfaces:
    surface_results = run_heuristic_analysis(df, surface_filter=surface, surface_name=surface)

print("\n" + "="*50)
print("BASELINE ANALYSIS COMPLETE")
print("="*50)
print(f"Results saved to: {specific_output_dir}")