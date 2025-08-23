#!/usr/bin/env python3
"""
BETTING ANALYSIS VERIFICATION: Deep dive into model betting differences
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_betting_logs():
    """Load all betting log files"""
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    logs = {}
    
    for model in models:
        try:
            filename = f"analysis_scripts/betting_logs/kelly_{model}_bets.csv"
            df = pd.read_csv(filename)
            df['date'] = pd.to_datetime(df['date'])
            logs[model] = df
            print(f"Loaded {len(df):,} bets for {model}")
        except FileNotFoundError:
            print(f"⚠️  {model} betting log not found")
    
    return logs

def verify_same_match_same_odds(logs):
    """Verify that same matches have same odds across all models"""
    print(f"\n" + "="*80)
    print(f"ODDS VERIFICATION: Same match should have same odds")
    print(f"="*80)
    
    # Find common matches across models
    all_matches = {}
    
    for model_name, df in logs.items():
        for _, row in df.iterrows():
            match_key = f"{row['player1']}|{row['player2']}|{row['date'].strftime('%Y-%m-%d')}"
            
            if match_key not in all_matches:
                all_matches[match_key] = {}
            
            all_matches[match_key][model_name] = {
                'bet_on_player': row['bet_on_player'],
                'odds_used': row['odds_used'],
                'market_prob_bet_on': row['market_prob_bet_on']
            }
    
    # Check for discrepancies
    discrepancies = []
    
    for match_key, model_data in all_matches.items():
        if len(model_data) > 1:  # Only check matches bet on by multiple models
            # Group by player bet on
            player_odds = {}
            
            for model, data in model_data.items():
                player = data['bet_on_player']
                odds = data['odds_used']
                market_prob = data['market_prob_bet_on']
                
                if player not in player_odds:
                    player_odds[player] = []
                player_odds[player].append((model, odds, market_prob))
            
            # Check if same player has different odds across models
            for player, odds_list in player_odds.items():
                if len(odds_list) > 1:
                    odds_values = [o[1] for o in odds_list]
                    market_probs = [o[2] for o in odds_list]
                    
                    if len(set(odds_values)) > 1 or len(set(market_probs)) > 1:
                        discrepancies.append({
                            'match': match_key,
                            'player': player,
                            'model_odds': odds_list
                        })
    
    if discrepancies:
        print(f"🚨 FOUND {len(discrepancies)} ODDS DISCREPANCIES:")
        for disc in discrepancies[:10]:  # Show first 10
            print(f"Match: {disc['match']}")
            print(f"Player: {disc['player']}")
            for model, odds, prob in disc['model_odds']:
                print(f"  {model}: odds={odds}, market_prob={prob:.4f}")
            print()
    else:
        print(f"✅ No odds discrepancies found! All models see same odds for same matches.")
    
    return len(discrepancies) == 0

def analyze_specific_date(logs, date_str="2023-01-02"):
    """Analyze betting differences on a specific date"""
    print(f"\n" + "="*80)
    print(f"DATE ANALYSIS: {date_str} betting patterns")
    print(f"="*80)
    
    target_date = pd.to_datetime(date_str)
    date_bets = {}
    
    for model_name, df in logs.items():
        day_bets = df[df['date'].dt.date == target_date.date()]
        date_bets[model_name] = day_bets
        print(f"{model_name}: {len(day_bets)} bets on {date_str}")
    
    # Find all unique matches on this date
    all_matches = set()
    for model_name, day_bets in date_bets.items():
        for _, row in day_bets.iterrows():
            match = f"{row['player1']} vs {row['player2']}"
            all_matches.add(match)
    
    print(f"\nTotal unique matches with bets on {date_str}: {len(all_matches)}")
    
    # Show match-by-match comparison
    print(f"\nMatch-by-Match Comparison:")
    print(f"{'Match':<40} {'NN143':<8} {'NN98':<8} {'XGB':<8} {'RF':<8}")
    print("-" * 80)
    
    for match in sorted(all_matches):
        bet_indicators = {}
        
        for model_name, day_bets in date_bets.items():
            match_bets = day_bets[
                (day_bets['player1'] + ' vs ' + day_bets['player2'] == match) |
                (day_bets['player2'] + ' vs ' + day_bets['player1'] == match)
            ]
            
            if len(match_bets) > 0:
                bet_on = match_bets.iloc[0]['bet_on_player']
                bet_indicators[model_name] = bet_on[:3]  # First 3 chars of name
            else:
                bet_indicators[model_name] = "---"
        
        print(f"{match[:38]:<40} {bet_indicators.get('neural_network_143', '---'):<8} {bet_indicators.get('neural_network_98', '---'):<8} {bet_indicators.get('xgboost', '---'):<8} {bet_indicators.get('random_forest', '---'):<8}")

def analyze_betting_behavior(logs):
    """Analyze why models make different betting decisions"""
    print(f"\n" + "="*80)
    print(f"BETTING BEHAVIOR ANALYSIS")
    print(f"="*80)
    
    for model_name, df in logs.items():
        print(f"\n{model_name.replace('_', ' ').title()}:")
        print(f"  Total bets: {len(df):,}")
        print(f"  Win rate: {df['bet_won'].mean():.1%}")
        print(f"  Average edge: {df['edge'].mean():.3f}")
        print(f"  Average Kelly fraction: {df['kelly_fraction'].mean():.3f}")
        print(f"  Average bet size: ${df['bet_amount'].mean():.2f}")
        
        # Edge distribution
        edge_bins = [0, 0.05, 0.10, 0.20, 0.50, 1.0]
        edge_counts = pd.cut(df['edge'], bins=edge_bins, right=False).value_counts().sort_index()
        print(f"  Edge distribution:")
        for interval, count in edge_counts.items():
            print(f"    {interval}: {count} bets ({count/len(df)*100:.1f}%)")

def compare_model_probabilities(logs):
    """Compare model probability predictions for the same matches"""
    print(f"\n" + "="*80)
    print(f"PROBABILITY COMPARISON: Why different betting decisions?")
    print(f"="*80)
    
    # Focus on Neural Networks since they have biggest return difference
    if 'neural_network_143' not in logs or 'neural_network_98' not in logs:
        print("Need both neural network models for comparison")
        return
    
    nn143 = logs['neural_network_143']
    nn98 = logs['neural_network_98']
    
    # Find matches both models bet on
    common_matches = []
    
    for _, row143 in nn143.iterrows():
        match_key = f"{row143['player1']}|{row143['player2']}|{row143['date']}"
        
        # Find corresponding match in NN98
        nn98_match = nn98[
            (nn98['player1'] == row143['player1']) & 
            (nn98['player2'] == row143['player2']) & 
            (nn98['date'] == row143['date'])
        ]
        
        if len(nn98_match) > 0:
            row98 = nn98_match.iloc[0]
            
            common_matches.append({
                'match': f"{row143['player1']} vs {row143['player2']}",
                'date': row143['date'],
                'nn143_bet_on': row143['bet_on_player'],
                'nn98_bet_on': row98['bet_on_player'],
                'nn143_prob': row143['model_prob_bet_on'],
                'nn98_prob': row98['model_prob_bet_on'],
                'nn143_edge': row143['edge'],
                'nn98_edge': row98['edge'],
                'nn143_kelly': row143['kelly_fraction'],
                'nn98_kelly': row98['kelly_fraction'],
                'nn143_won': row143['bet_won'],
                'nn98_won': row98['bet_won']
            })
    
    if len(common_matches) == 0:
        print("No common matches found between NN-143 and NN-98")
        return
    
    common_df = pd.DataFrame(common_matches)
    
    print(f"Found {len(common_matches):,} matches both models bet on")
    
    # Find matches where they bet on different players
    different_bets = common_df[common_df['nn143_bet_on'] != common_df['nn98_bet_on']]
    print(f"Matches where they bet on different players: {len(different_bets)}")
    
    if len(different_bets) > 0:
        print(f"\nFirst 10 disagreement matches:")
        for _, row in different_bets.head(10).iterrows():
            print(f"Match: {row['match']} ({row['date'].strftime('%Y-%m-%d')})")
            print(f"  NN-143 bet on {row['nn143_bet_on']} (prob: {row['nn143_prob']:.3f}, edge: {row['nn143_edge']:.3f}) - {'WON' if row['nn143_won'] else 'LOST'}")
            print(f"  NN-98  bet on {row['nn98_bet_on']} (prob: {row['nn98_prob']:.3f}, edge: {row['nn98_edge']:.3f}) - {'WON' if row['nn98_won'] else 'LOST'}")
            print()
    
    # Analyze Kelly fraction differences
    print(f"\nKelly Fraction Analysis:")
    print(f"NN-143 average Kelly: {common_df['nn143_kelly'].mean():.4f}")
    print(f"NN-98 average Kelly: {common_df['nn98_kelly'].mean():.4f}")
    
    # High Kelly fraction bets (>2% of bankroll)
    high_kelly_143 = common_df[common_df['nn143_kelly'] > 0.02]
    high_kelly_98 = common_df[common_df['nn98_kelly'] > 0.02]
    
    print(f"NN-143 high-conviction bets (>2%): {len(high_kelly_143)} ({len(high_kelly_143)/len(common_df)*100:.1f}%)")
    print(f"NN-98 high-conviction bets (>2%): {len(high_kelly_98)} ({len(high_kelly_98)/len(common_df)*100:.1f}%)")
    
    if len(high_kelly_143) > 0:
        print(f"NN-143 high-conviction win rate: {high_kelly_143['nn143_won'].mean():.1%}")
    if len(high_kelly_98) > 0:
        print(f"NN-98 high-conviction win rate: {high_kelly_98['nn98_won'].mean():.1%}")

def analyze_early_performance(logs):
    """Analyze first 100 bets to see if early luck explains compounding"""
    print(f"\n" + "="*80)
    print(f"EARLY PERFORMANCE ANALYSIS: First 100 bets impact")
    print(f"="*80)
    
    for model_name, df in logs.items():
        if len(df) >= 100:
            first_100 = df.head(100).copy()
            
            # Calculate cumulative return for first 100 bets
            cumulative_return = (first_100['bankroll_after'].iloc[-1] / first_100['bankroll_before'].iloc[0] - 1) * 100
            
            print(f"\n{model_name.replace('_', ' ').title()} - First 100 bets:")
            print(f"  Win rate: {first_100['bet_won'].mean():.1%}")
            print(f"  Cumulative return: {cumulative_return:+.1f}%")
            print(f"  Final bankroll after 100 bets: ${first_100['bankroll_after'].iloc[-1]:.2f}")
            print(f"  Biggest single win: ${first_100['profit'].max():.2f}")
            print(f"  Biggest single loss: ${first_100['profit'].min():.2f}")

def create_probability_calibration_analysis(logs):
    """Analyze model calibration without retraining"""
    print(f"\n" + "="*80)
    print(f"RETROACTIVE CALIBRATION ANALYSIS")
    print(f"="*80)
    
    for model_name, df in logs.items():
        if len(df) == 0:
            continue
            
        print(f"\n{model_name.replace('_', ' ').title()}:")
        
        # Bin predictions and calculate actual win rates
        prob_bins = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        df['prob_bin'] = pd.cut(df['model_prob_bet_on'], bins=prob_bins)
        
        calibration_data = df.groupby('prob_bin').agg({
            'bet_won': ['count', 'sum', 'mean'],
            'model_prob_bet_on': 'mean'
        }).round(3)
        
        print(f"  Calibration by probability bins:")
        print(f"  {'Prob Range':<15} {'Count':<8} {'Wins':<6} {'Win%':<8} {'Avg Pred':<10} {'Calibration':<12}")
        print("  " + "-" * 75)
        
        for bin_range, data in calibration_data.iterrows():
            count = data[('bet_won', 'count')]
            wins = data[('bet_won', 'sum')]
            win_rate = data[('bet_won', 'mean')]
            avg_pred = data[('model_prob_bet_on', 'mean')]
            
            if count > 0:
                calibration_error = abs(win_rate - avg_pred)
                print(f"  {str(bin_range):<15} {count:<8} {wins:<6} {win_rate:<8.1%} {avg_pred:<10.3f} {calibration_error:<12.3f}")
        
        # Overall Brier Score approximation
        brier_score = np.mean((df['model_prob_bet_on'] - df['bet_won'].astype(float)) ** 2)
        print(f"  Approximate Brier Score: {brier_score:.4f}")

def main():
    print("BETTING ANALYSIS VERIFICATION")
    print("="*80)
    
    # Load all betting logs
    logs = load_betting_logs()
    
    if len(logs) == 0:
        print("No betting logs found. Run the main analysis first.")
        return
    
    # 1. Verify same matches have same odds
    odds_verified = verify_same_match_same_odds(logs)
    
    # 2. Analyze specific date
    analyze_specific_date(logs, "2023-01-02")
    
    # 3. Analyze betting behavior patterns
    analyze_betting_behavior(logs)
    
    # 4. Compare model probabilities
    compare_model_probabilities(logs)
    
    # 5. Analyze early performance impact
    analyze_early_performance(logs)
    
    # 6. Calibration analysis
    create_probability_calibration_analysis(logs)
    
    print(f"\n" + "="*80)
    print(f"SUMMARY OF FINDINGS:")
    print(f"✅ Odds verification: {'PASSED' if odds_verified else 'FAILED'}")
    print(f"✅ Detailed analysis complete - check output above for insights")
    print(f"="*80)

if __name__ == "__main__":
    main()