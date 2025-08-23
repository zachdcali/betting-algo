#!/usr/bin/env python3
"""
LOSS SEGMENT ANALYSIS FOR BETTING STRATEGIES

Analyzes patterns in losing bets to identify:
- Edge ranges where most losses occur
- Odds ranges with high loss concentration  
- Confidence levels that lead to big losses
- Common denominators in bust-causing bets

This helps determine optimal min/max edge thresholds and betting filters.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse

def load_betting_data(strategy_folder, model_name):
    """Load betting CSV for a specific model and strategy"""
    csv_path = f"analysis_scripts/betting_logs/{strategy_folder}/{model_name.lower()}_bets.csv"
    
    if not os.path.exists(csv_path):
        print(f"   WARNING: {csv_path} not found")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"   ERROR loading {csv_path}: {e}")
        return None

def analyze_losses(df, model_name, strategy_name):
    """Comprehensive loss analysis by edge, odds, confidence"""
    if df is None or len(df) == 0:
        return None
    
    # Split into wins/losses
    wins = df[df['profit'] > 0].copy()
    losses = df[df['profit'] <= 0].copy()
    
    print(f"\n{'='*60}")
    print(f"{model_name.upper()} - {strategy_name}")
    print(f"{'='*60}")
    print(f"Total bets: {len(df):,}")
    print(f"Wins: {len(wins):,} ({len(wins)/len(df)*100:.1f}%)")
    print(f"Losses: {len(losses):,} ({len(losses)/len(df)*100:.1f}%)")
    print(f"Total profit: ${df['profit'].sum():,.0f}")
    print(f"Win profit: ${wins['profit'].sum():,.0f}")
    print(f"Loss amount: ${losses['profit'].sum():,.0f}")
    
    if len(losses) == 0:
        print("No losses to analyze!")
        return None
    
    # Create bins for analysis
    losses['edge_bin'] = pd.qcut(losses['edge'], 
                                q=5, 
                                labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                                duplicates='drop')
    
    losses['odds_bin'] = pd.qcut(losses['odds_used'], 
                                q=5, 
                                labels=['Low', 'Med-Low', 'Medium', 'Med-High', 'High'],
                                duplicates='drop')
    
    losses['conf_bin'] = pd.qcut(losses['model_prob_bet_on'], 
                                q=5, 
                                labels=['Low Conf', 'Med-Low', 'Medium', 'Med-High', 'High Conf'],
                                duplicates='drop')
    
    # Analyze by edge ranges
    print(f"\n📊 LOSS ANALYSIS BY EDGE RANGE:")
    print(f"{'Range':<15} {'Count':<8} {'% of Losses':<12} {'Avg Loss':<12} {'Total Loss':<15}")
    print("-" * 70)
    
    edge_analysis = losses.groupby('edge_bin')['profit'].agg(['count', 'mean', 'sum'])
    for edge_range, stats in edge_analysis.iterrows():
        pct_losses = stats['count'] / len(losses) * 100
        print(f"{str(edge_range):<15} {stats['count']:<8} {pct_losses:<11.1f}% "
              f"${stats['mean']:<11.0f} ${stats['sum']:<14.0f}")
    
    # Find edge threshold recommendations
    print(f"\n💡 EDGE THRESHOLD ANALYSIS:")
    edge_thresholds = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    for threshold in edge_thresholds:
        low_edge_losses = losses[losses['edge'] < threshold]
        if len(low_edge_losses) > 0:
            pct_of_losses = len(low_edge_losses) / len(losses) * 100
            loss_amount = low_edge_losses['profit'].sum()
            bets_filtered = len(df[df['edge'] < threshold])
            pct_bets_filtered = bets_filtered / len(df) * 100
            
            print(f"Min edge {threshold:.1%}: Would filter {bets_filtered:,} bets ({pct_bets_filtered:.1f}%), "
                  f"eliminating ${abs(loss_amount):,.0f} losses ({pct_of_losses:.1f}% of all losses)")
    
    # Analyze by odds ranges  
    print(f"\n📊 LOSS ANALYSIS BY ODDS RANGE:")
    print(f"{'Range':<15} {'Count':<8} {'% of Losses':<12} {'Avg Loss':<12} {'Total Loss':<15}")
    print("-" * 70)
    
    odds_analysis = losses.groupby('odds_bin')['profit'].agg(['count', 'mean', 'sum'])
    for odds_range, stats in odds_analysis.iterrows():
        pct_losses = stats['count'] / len(losses) * 100
        print(f"{str(odds_range):<15} {stats['count']:<8} {pct_losses:<11.1f}% "
              f"${stats['mean']:<11.0f} ${stats['sum']:<14.0f}")
    
    # Analyze by confidence ranges
    print(f"\n📊 LOSS ANALYSIS BY CONFIDENCE RANGE:")
    print(f"{'Range':<15} {'Count':<8} {'% of Losses':<12} {'Avg Loss':<12} {'Total Loss':<15}")
    print("-" * 70)
    
    conf_analysis = losses.groupby('conf_bin')['profit'].agg(['count', 'mean', 'sum'])
    for conf_range, stats in conf_analysis.iterrows():
        pct_losses = stats['count'] / len(losses) * 100
        print(f"{str(conf_range):<15} {stats['count']:<8} {pct_losses:<11.1f}% "
              f"${stats['mean']:<11.0f} ${stats['sum']:<14.0f}")
    
    # Find biggest single losses
    print(f"\n💸 TOP 10 BIGGEST SINGLE LOSSES:")
    print(f"{'Date':<12} {'Player Bet On':<20} {'Edge':<8} {'Odds':<8} {'Confidence':<12} {'Bet Size':<10} {'Loss':<12}")
    print("-" * 90)
    
    biggest_losses = losses.nsmallest(10, 'profit')
    for _, loss in biggest_losses.iterrows():
        player_bet_on = loss['player1'] if loss['bet_on_p1'] else loss['player2']
        print(f"{loss['date'].strftime('%Y-%m-%d'):<12} "
              f"{player_bet_on[:18]:<20} "
              f"{loss['edge']:<7.1%} "
              f"{loss['odds_used']:<7.2f} "
              f"{loss['model_prob_bet_on']:<11.1%} "
              f"${loss['bet_amount']:<9.0f} "
              f"${loss['profit']:<11.0f}")
    
    # Analyze players involved in biggest losses
    print(f"\n👤 PLAYERS INVOLVED IN BIGGEST LOSSES:")
    big_loss_threshold = losses['profit'].quantile(0.1)  # Bottom 10% of losses
    big_losses = losses[losses['profit'] <= big_loss_threshold]
    
    # Count losses by player bet on
    big_losses['player_bet_on'] = big_losses.apply(lambda x: x['player1'] if x['bet_on_p1'] else x['player2'], axis=1)
    player_loss_counts = big_losses.groupby('player_bet_on').agg({
        'profit': ['count', 'sum', 'mean'],
        'edge': 'mean',
        'model_prob_bet_on': 'mean'
    }).round(3)
    
    player_loss_counts.columns = ['Loss_Count', 'Total_Loss', 'Avg_Loss', 'Avg_Edge', 'Avg_Confidence']
    player_loss_counts = player_loss_counts.sort_values('Total_Loss').head(10)
    
    print(f"{'Player':<25} {'Count':<8} {'Total Loss':<12} {'Avg Loss':<10} {'Avg Edge':<10} {'Avg Conf':<10}")
    print("-" * 85)
    for player, stats in player_loss_counts.iterrows():
        print(f"{player[:23]:<25} {stats['Loss_Count']:<8} ${stats['Total_Loss']:<11.0f} "
              f"${stats['Avg_Loss']:<9.0f} {stats['Avg_Edge']:<9.1%} {stats['Avg_Confidence']:<9.1%}")
    
    # Analyze temporal patterns
    print(f"\n📅 TEMPORAL LOSS PATTERNS:")
    losses['month'] = losses['date'].dt.month
    losses['weekday'] = losses['date'].dt.dayofweek
    
    monthly_losses = losses.groupby('month')['profit'].agg(['count', 'sum', 'mean'])
    print(f"\nLosses by Month:")
    print(f"{'Month':<8} {'Count':<8} {'Total Loss':<12} {'Avg Loss':<10}")
    print("-" * 40)
    for month, stats in monthly_losses.iterrows():
        print(f"{month:<8} {stats['count']:<8} ${stats['sum']:<11.0f} ${stats['mean']:<9.0f}")
    
    # Analyze match-up patterns (P1 vs P2 betting)
    print(f"\n🥎 BETTING PATTERN ANALYSIS:")
    p1_losses = losses[losses['bet_on_p1'] == True]
    p2_losses = losses[losses['bet_on_p1'] == False]
    
    print(f"Losses betting on Player 1: {len(p1_losses):,} (${p1_losses['profit'].sum():,.0f})")
    print(f"Losses betting on Player 2: {len(p2_losses):,} (${p2_losses['profit'].sum():,.0f})")
    print(f"Player 1 avg loss: ${p1_losses['profit'].mean():.0f}")
    print(f"Player 2 avg loss: ${p2_losses['profit'].mean():.0f}")
    
    if len(p1_losses) > 0 and len(p2_losses) > 0:
        from scipy.stats import ttest_ind
        t_stat, p_val = ttest_ind(p1_losses['profit'], p2_losses['profit'])
        if p_val < 0.05:
            print(f"⚠️  Significant difference in P1 vs P2 losses (p={p_val:.3f})")
        else:
            print(f"No significant difference in P1 vs P2 losses (p={p_val:.3f})")
    
    # Statistical tests for edge bins
    print(f"\n🔬 EDGE BIN PERFORMANCE TEST:")
    print(f"Testing if losses exceed expected based on model probabilities...")
    print(f"{'Edge Range':<15} {'Expected Losses':<15} {'Actual Losses':<15} {'Excess':<10} {'P-Value':<10}")
    print("-" * 75)
    
    from scipy.stats import binomtest
    
    for edge_range in edge_analysis.index:
        if pd.isna(edge_range):
            continue
            
        bin_bets = df[df['edge'].between(losses[losses['edge_bin'] == edge_range]['edge'].min(),
                                       losses[losses['edge_bin'] == edge_range]['edge'].max())]
        if len(bin_bets) == 0:
            continue
            
        expected_win_rate = bin_bets['model_prob_bet_on'].mean()
        actual_losses = len(bin_bets[bin_bets['profit'] <= 0])
        expected_losses = len(bin_bets) * (1 - expected_win_rate)
        
        if expected_losses > 0:
            p_value = binomtest(actual_losses, len(bin_bets), 1 - expected_win_rate, alternative='greater').pvalue
            excess = actual_losses - expected_losses
            
            print(f"{str(edge_range):<15} {expected_losses:<14.1f} {actual_losses:<14} "
                  f"{excess:<9.1f} {p_value:<9.3f}")
    
    print(f"\nP-value < 0.05 indicates significant miscalibration (losses exceed model predictions)")
    print(f"P-value >= 0.05 suggests losses are within expected variance")
    
    # Analyze bust timing for fixed strategies
    bust_analysis = None
    if 'Fixed' in strategy_name:
        print(f"\n🕐 BUST TIMING ANALYSIS (Fixed Strategy):")
        print(f"Analyzing when $100 starting bankroll would go bust...")
        
        # Simulate bankroll progression with $1 fixed bets
        bankroll = 100.0
        bust_bet = None
        
        for i, (_, bet) in enumerate(df.iterrows()):
            if bankroll <= 0:
                break
            if bet['profit'] < 0 and bankroll + bet['profit'] <= 0:
                bust_bet = i + 1
                break
            bankroll += bet['profit']
        
        if bust_bet:
            bust_pct = bust_bet / len(df) * 100
            print(f"   Would bust at bet #{bust_bet:,} of {len(df):,} ({bust_pct:.1f}% through sequence)")
        else:
            print(f"   No bust - final bankroll: ${bankroll:,.0f}")
        
        # Analyze early vs late losses
        early_cutoff = len(df) // 4  # First 25% of bets
        early_losses = df.iloc[:early_cutoff]
        late_losses = df.iloc[early_cutoff:]
        
        early_loss_count = len(early_losses[early_losses['profit'] < 0])
        late_loss_count = len(late_losses[late_losses['profit'] < 0])
        early_loss_amount = early_losses[early_losses['profit'] < 0]['profit'].sum()
        late_loss_amount = late_losses[late_losses['profit'] < 0]['profit'].sum()
        
        print(f"   Early losses (first 25% of bets): {early_loss_count} bets, ${early_loss_amount:,.0f}")
        print(f"   Later losses (remaining 75%): {late_loss_count} bets, ${late_loss_amount:,.0f}")
        
        bust_analysis = {
            'bust_bet': bust_bet,
            'early_loss_count': early_loss_count,
            'late_loss_count': late_loss_count,
            'early_loss_amount': early_loss_amount,
            'late_loss_amount': late_loss_amount
        }

    return {
        'model': model_name,
        'strategy': strategy_name, 
        'total_bets': len(df),
        'losses': len(losses),
        'total_loss': losses['profit'].sum(),
        'edge_analysis': edge_analysis,
        'odds_analysis': odds_analysis,
        'conf_analysis': conf_analysis,
        'biggest_losses': biggest_losses,
        'bust_analysis': bust_analysis
    }

def create_loss_visualization(all_results, save_folder):
    """Create visualizations of loss patterns"""
    os.makedirs(save_folder, exist_ok=True)
    
    # Combine all data for cross-model comparison
    combined_data = []
    
    for strategy, models in all_results.items():
        for model, analysis in models.items():
            if analysis is not None:
                # Add model/strategy info to loss records
                df_path = f"analysis_scripts/betting_logs/{strategy}/{model.lower()}_bets.csv"
                if os.path.exists(df_path):
                    df = pd.read_csv(df_path)
                    losses = df[df['profit'] <= 0].copy()
                    losses['model'] = model.replace('_', ' ')
                    losses['strategy'] = all_strategies[strategy]
                    combined_data.append(losses)
    
    if not combined_data:
        print("No data available for visualization")
        return
    
    all_losses = pd.concat(combined_data, ignore_index=True)
    
    # Create loss distribution plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Edge distribution
    sns.boxplot(data=all_losses, x='strategy', y='edge', hue='model', ax=axes[0,0])
    axes[0,0].set_title('Loss Distribution by Edge Range', fontweight='bold')
    axes[0,0].set_ylabel('Edge')
    axes[0,0].tick_params(axis='x', rotation=45)
    
    # Odds distribution  
    sns.boxplot(data=all_losses, x='strategy', y='odds_used', hue='model', ax=axes[0,1])
    axes[0,1].set_title('Loss Distribution by Odds Range', fontweight='bold') 
    axes[0,1].set_ylabel('Odds Used')
    axes[0,1].tick_params(axis='x', rotation=45)
    
    # Confidence distribution
    sns.boxplot(data=all_losses, x='strategy', y='model_prob_bet_on', hue='model', ax=axes[1,0])
    axes[1,0].set_title('Loss Distribution by Model Confidence', fontweight='bold')
    axes[1,0].set_ylabel('Model Probability') 
    axes[1,0].tick_params(axis='x', rotation=45)
    
    # Loss amount distribution
    sns.boxplot(data=all_losses, x='strategy', y='profit', hue='model', ax=axes[1,1])
    axes[1,1].set_title('Loss Amount Distribution', fontweight='bold')
    axes[1,1].set_ylabel('Loss Amount ($)')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f"{save_folder}/loss_patterns_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Loss visualization saved: {save_folder}/loss_patterns_analysis.png")

def main(strategy=None, model=None):
    print("=" * 80)
    print("LOSS SEGMENT ANALYSIS - BETTING PATTERN IDENTIFICATION")
    print("=" * 80)
    print("Analyzing patterns in losing bets to optimize strategy parameters")
    print()
    
    # Define strategies and models
    global all_strategies
    all_strategies = {
        'Kelly_5pct_Cap': 'Kelly 5% Cap',
        'Full_Kelly': 'Full Kelly',
        'Half_Kelly': 'Half Kelly',
        'Quarter_Kelly': 'Quarter Kelly', 
        'Fixed_Bankroll': 'Fixed Bankroll'
    }
    
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    
    # Create output directory
    output_dir = 'analysis_scripts/variance_analysis/loss_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    all_results = {}
    
    # Analyze specific strategy/model or all combinations
    if strategy and model:
        strategies_to_analyze = {strategy: all_strategies[strategy]}
        models_to_analyze = [model]
    else:
        strategies_to_analyze = all_strategies
        models_to_analyze = models
    
    for strategy_folder, strategy_name in strategies_to_analyze.items():
        strategy_results = {}
        
        for model_name in models_to_analyze:
            print(f"\nAnalyzing {model_name} - {strategy_name}...")
            
            # Load and analyze data
            df = load_betting_data(strategy_folder, model_name)
            analysis = analyze_losses(df, model_name, strategy_name)
            
            strategy_results[model_name] = analysis
        
        all_results[strategy_folder] = strategy_results
    
    # Create visualizations
    print(f"\n📊 Creating loss pattern visualizations...")
    create_loss_visualization(all_results, output_dir)
    
    print(f"\n✅ Loss segment analysis complete!")
    print(f"📁 Results saved to: {output_dir}/")
    
    # Key recommendations
    print(f"\n🎯 KEY RECOMMENDATIONS:")
    print(f"1. Check edge threshold analysis above - if low edges (<5%) show high loss concentration, set minimum")
    print(f"2. Look for p-values < 0.05 in statistical tests - indicates miscalibration in that range")  
    print(f"3. High-odds losses may indicate overconfidence on underdogs - consider caps")
    print(f"4. Biggest single losses show patterns - edge/odds/confidence combinations to avoid")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Loss Segment Analysis for Betting Strategies')
    parser.add_argument('--strategy', choices=['Kelly_5pct_Cap', 'Full_Kelly', 'Half_Kelly', 'Quarter_Kelly', 'Fixed_Bankroll'],
                       help='Analyze specific strategy only')
    parser.add_argument('--model', choices=['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98'],
                       help='Analyze specific model only')
    
    args = parser.parse_args()
    
    main(strategy=args.strategy, model=args.model)