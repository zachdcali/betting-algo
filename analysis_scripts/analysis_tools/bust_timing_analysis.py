#!/usr/bin/env python3
"""
BUST TIMING ANALYSIS FOR MONTE CARLO SIMULATIONS

Analyzes when busts occur across all 10,000 Monte Carlo simulations
to test hypothesis that Fixed strategy busts happen early.

Key Questions:
1. What percentage of bet sequence do most busts occur?
2. Does Fixed strategy bust risk decrease as bankroll grows?
3. Are busts clustered in early portions of sequences?
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import argparse
from tqdm import tqdm

def simulate_bust_timing(period_profits, n_simulations=10000, starting_bankroll=100.0):
    """
    Run Monte Carlo simulations and track bust timing for Fixed strategy.
    Uses actual profit/loss values from betting logs, same as original Monte Carlo.
    
    Parameters:
    - period_profits: Array of weekly profit batches (actual profit/loss amounts)
    - n_simulations: Number of Monte Carlo runs
    - starting_bankroll: Starting bankroll
    
    Returns:
    - Dictionary with bust timing statistics
    """
    bust_times = []  # Period number when bust occurred
    bust_percentages = []  # Percentage through sequence when bust occurred
    total_periods = len(period_profits)
    
    print(f"   Running {n_simulations:,} simulations to analyze bust timing...")
    
    for sim in tqdm(range(n_simulations), desc="   Bust timing analysis", leave=False):
        # Resample periods with replacement (same as original Monte Carlo)
        sim_period_profits = np.random.choice(period_profits, len(period_profits), replace=True)
        
        bankroll = starting_bankroll
        
        for period_num, period_profit in enumerate(sim_period_profits):
            # Add the period profit/loss directly (matches original Monte Carlo logic)
            bankroll += period_profit
            
            # Check for bust
            if bankroll <= 0:
                bust_times.append(period_num + 1)
                bust_percentage = (period_num + 1) / total_periods * 100
                bust_percentages.append(bust_percentage)
                break
    
    bust_count = len(bust_times)
    bust_rate = bust_count / n_simulations * 100
    
    print(f"   Bust rate: {bust_rate:.1f}% ({bust_count:,} of {n_simulations:,} simulations)")
    
    if bust_count == 0:
        print(f"   No busts occurred - strategy appears completely safe")
        return {
            'bust_rate': 0.0,
            'bust_times': [],
            'bust_percentages': [],
            'early_bust_rate': 0.0,
            'avg_bust_percentage': 0.0,
            'n_simulations': n_simulations
        }
    
    bust_times = np.array(bust_times)
    bust_percentages = np.array(bust_percentages)
    
    # Analyze timing patterns
    early_busts = np.sum(bust_percentages <= 25)  # First quarter of bets
    early_bust_rate = early_busts / bust_count * 100
    
    print(f"   Average bust occurs at period {bust_times.mean():.0f} of {total_periods} ({bust_percentages.mean():.1f}% through sequence)")
    print(f"   Early busts (first 25%): {early_busts:,} of {bust_count:,} busts ({early_bust_rate:.1f}%)")
    print(f"   Median bust: {np.median(bust_percentages):.1f}% through sequence")
    
    return {
        'bust_rate': bust_rate,
        'bust_times': bust_times,
        'bust_percentages': bust_percentages,
        'early_bust_rate': early_bust_rate,
        'avg_bust_percentage': bust_percentages.mean(),
        'median_bust_percentage': np.median(bust_percentages),
        'total_periods': total_periods,
        'n_simulations': n_simulations
    }

def create_bust_timing_plots(results, model_name, save_path):
    """Create visualizations of bust timing patterns"""
    
    if results['bust_rate'] == 0:
        # Create empty plot for no-bust strategies
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, f'{model_name} - Fixed Bankroll\n\nNo busts in {results["n_simulations"]:,} simulations\nStrategy appears completely safe', 
                transform=ax.transAxes, ha='center', va='center', fontsize=14, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        ax.set_title(f'{model_name} - Bust Timing Analysis', fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    bust_percentages = results['bust_percentages']
    
    # Histogram of bust timing
    ax1.hist(bust_percentages, bins=20, alpha=0.7, color='red', edgecolor='black')
    ax1.axvline(25, color='blue', linestyle='--', linewidth=2, label='First 25% of bets')
    ax1.axvline(results['avg_bust_percentage'], color='orange', linestyle='-', linewidth=2, 
                label=f'Average: {results["avg_bust_percentage"]:.1f}%')
    ax1.axvline(results['median_bust_percentage'], color='green', linestyle='-', linewidth=2, 
                label=f'Median: {results["median_bust_percentage"]:.1f}%')
    
    ax1.set_xlabel('Percentage Through Period Sequence When Bust Occurred', fontsize=12)
    ax1.set_ylabel('Number of Busts', fontsize=12)
    ax1.set_title(f'{model_name} - When Busts Occur\n({results["bust_rate"]:.1f}% bust rate)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Cumulative bust probability over time
    sorted_percentages = np.sort(bust_percentages)
    cumulative_prob = np.arange(1, len(sorted_percentages) + 1) / results['n_simulations'] * 100
    
    ax2.plot(sorted_percentages, cumulative_prob, linewidth=2, color='red')
    ax2.axvline(25, color='blue', linestyle='--', linewidth=2, label='First 25% of bets')
    ax2.axhline(results['early_bust_rate'] * results['bust_rate'] / 100, 
                color='blue', linestyle=':', alpha=0.7,
                label=f'{results["early_bust_rate"]:.1f}% of busts occur early')
    
    ax2.set_xlabel('Percentage Through Period Sequence', fontsize=12)
    ax2.set_ylabel('Cumulative Bust Probability (%)', fontsize=12)
    ax2.set_title(f'{model_name} - Cumulative Bust Risk Over Time', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

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

def calculate_temporal_profits(df, grouping='weekly'):
    """Calculate profits grouped by time period to preserve temporal clustering"""
    if df is None or len(df) == 0:
        return np.array([])
    
    df = df.copy()
    
    if grouping == 'weekly':
        df['period'] = df['date'].dt.isocalendar().week.astype(str) + "-" + df['date'].dt.year.astype(str)
    elif grouping == 'daily':
        df['period'] = df['date'].dt.strftime('%Y-%m-%d')
    elif grouping == 'monthly':
        df['period'] = df['date'].dt.to_period('M').astype(str)
    
    # Calculate profit per period (this maintains the bet clustering)
    period_profits = df.groupby('period')['profit'].sum().values
    
    return period_profits

def main():
    print("=" * 80)
    print("BUST TIMING ANALYSIS - FIXED BANKROLL STRATEGIES")
    print("=" * 80)
    print("Testing hypothesis: Most busts in Fixed strategies occur early")
    print("Analyzing bust timing across 10,000 Monte Carlo simulations")
    print()
    
    # Focus on Fixed Bankroll strategies only
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    
    # Create output directories
    os.makedirs('analysis_scripts/variance_analysis/bust_timing', exist_ok=True)
    
    all_results = {}
    
    for model_name in models:
        print(f"\nAnalyzing {model_name} - Fixed Bankroll:")
        print("-" * 50)
        
        # Load betting data
        df = load_betting_data('Fixed_Bankroll', model_name)
        
        if df is None:
            print(f"   SKIPPED (no data)")
            continue
        
        # Calculate temporal profits (weekly batches)
        period_profits = calculate_temporal_profits(df, grouping='weekly')
        
        if len(period_profits) == 0:
            print(f"   SKIPPED (no profits)")
            continue
        
        # Run bust timing analysis
        results = simulate_bust_timing(period_profits, n_simulations=10000, 
                                     starting_bankroll=100.0)
        
        all_results[model_name] = results
        
        # Create visualization
        plot_path = f"analysis_scripts/variance_analysis/bust_timing/{model_name}_bust_timing.png"
        create_bust_timing_plots(results, model_name, plot_path)
        print(f"   📈 Plot saved: {plot_path}")
    
    # Summary comparison
    print(f"\n" + "=" * 80)
    print(f"BUST TIMING SUMMARY - FIXED BANKROLL STRATEGIES")
    print(f"=" * 80)
    print(f"{'Model':<20} {'Bust Rate':<12} {'Avg Bust %':<12} {'Early Bust %':<12} {'Hypothesis':<15}")
    print("-" * 80)
    
    for model_name, results in all_results.items():
        if results['bust_rate'] == 0:
            hypothesis = "N/A (No busts)"
        elif results['early_bust_rate'] > 50:
            hypothesis = "✅ CONFIRMED"
        else:
            hypothesis = "❌ REJECTED"
            
        print(f"{model_name:<20} {results['bust_rate']:<11.1f}% {results.get('avg_bust_percentage', 0):<11.1f}% "
              f"{results.get('early_bust_rate', 0):<11.1f}% {hypothesis:<15}")
    
    print(f"\n🎯 HYPOTHESIS TEST:")
    print(f"If >50% of busts occur in first 25% of sequence → Fixed gets safer over time")
    print(f"If busts are evenly distributed → Risk stays constant")
    
    print(f"\n✅ Analysis complete!")
    print(f"📁 Charts saved to: analysis_scripts/variance_analysis/bust_timing/")

if __name__ == "__main__":
    main()