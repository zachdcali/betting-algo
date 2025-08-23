#!/usr/bin/env python3
"""
COMPREHENSIVE KELLY FRACTION OPTIMIZATION

This script runs the COMPLETE analysis:
- 4 models × 100 Kelly fractions (1% to 100%) × 10,000 Monte Carlo simulations each
- Total: 4 million simulations
- Uses the proper approach: loads actual betting strategy results and resamples them
- Creates smooth, accurate curves showing the true Kelly fraction relationships

This will take time but gives us definitive results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tqdm import tqdm
import time

# Set random seed for reproducibility
np.random.seed(42)

def load_betting_strategy(model_name, strategy_folder):
    """Load betting results for a specific model and strategy"""
    csv_path = f"analysis_scripts/betting_logs/{strategy_folder}/{model_name.lower()}_bets.csv"
    
    if not os.path.exists(csv_path):
        return None
    
    try:
        df = pd.read_csv(csv_path)
        df['date'] = pd.to_datetime(df['date'])
        return df
    except Exception as e:
        print(f"   ERROR loading {csv_path}: {e}")
        return None

def calculate_weekly_profits(betting_df):
    """Calculate weekly profits preserving temporal clustering"""
    if betting_df is None or len(betting_df) == 0:
        return np.array([])
    
    df = betting_df.copy()
    df['week'] = df['date'].dt.isocalendar().week.astype(str) + "-" + df['date'].dt.year.astype(str)
    
    # Calculate profit per week
    weekly_profits = df.groupby('week')['profit'].sum().values
    return weekly_profits

def simulate_kelly_fraction(weekly_profits, target_kelly_fraction, base_kelly_fraction, n_simulations=10000, starting_bankroll=100.0):
    """
    Simulate a specific Kelly fraction by scaling the base strategy results.
    
    Args:
        weekly_profits: Weekly profits from base strategy
        target_kelly_fraction: Target Kelly fraction to simulate (e.g., 0.03 for 3%)
        base_kelly_fraction: Kelly fraction of the base strategy (e.g., 0.02 for Tenth_Kelly which is 1/50)
        n_simulations: Number of Monte Carlo simulations
        starting_bankroll: Starting bankroll
        
    Returns:
        Dictionary with simulation results
    """
    if len(weekly_profits) == 0:
        return None
    
    # Scale profits to match target Kelly fraction
    scaling_factor = target_kelly_fraction / base_kelly_fraction
    scaled_weekly_profits = weekly_profits * scaling_factor
    
    sim_bankrolls = []
    sim_returns = []
    sim_drawdowns = []
    bust_count = 0
    
    for sim in range(n_simulations):
        # Resample weekly profits with replacement to preserve temporal clustering
        sim_weekly_profits = np.random.choice(scaled_weekly_profits, len(scaled_weekly_profits), replace=True)
        
        bankroll = starting_bankroll
        max_bankroll = starting_bankroll
        max_drawdown = 0
        
        for weekly_profit in sim_weekly_profits:
            bankroll += weekly_profit
            max_bankroll = max(max_bankroll, bankroll)
            
            # Calculate drawdown from peak
            if max_bankroll > 0:
                drawdown = (max_bankroll - bankroll) / max_bankroll
                max_drawdown = max(max_drawdown, drawdown)
            
            # Check for bust
            if bankroll <= 0:
                bankroll = 0
                bust_count += 1
                break
        
        sim_bankrolls.append(bankroll)
        sim_returns.append((bankroll - starting_bankroll) / starting_bankroll)
        sim_drawdowns.append(max_drawdown)
    
    sim_bankrolls = np.array(sim_bankrolls)
    sim_returns = np.array(sim_returns)
    sim_drawdowns = np.array(sim_drawdowns)
    
    # Calculate comprehensive metrics
    mean_return = np.mean(sim_returns)
    std_return = np.std(sim_returns)
    
    # Risk-adjusted metrics
    sharpe_ratio = mean_return / std_return if std_return > 0 else 0
    
    # Risk metrics
    bust_rate = bust_count / n_simulations
    profit_rate = np.sum(sim_returns > 0) / n_simulations
    
    return {
        'kelly_fraction': target_kelly_fraction,
        'mean_bankroll': np.mean(sim_bankrolls),
        'mean_return': mean_return * 100,  # Convert to percentage
        'std_return': std_return * 100,
        'sharpe_ratio': sharpe_ratio,
        'bust_rate': bust_rate * 100,  # Convert to percentage
        'profit_rate': profit_rate * 100,
        'max_drawdown': np.mean(sim_drawdowns) * 100,
        'median_return': np.median(sim_returns) * 100,
        'p95_return': np.percentile(sim_returns, 95) * 100,
        'p5_return': np.percentile(sim_returns, 5) * 100,
        'model': None  # Will be set later
    }

def load_all_existing_strategies(model_name):
    """Load all existing betting strategy results for a model"""
    
    # Generate all 100 Kelly fractions dynamically (same as 100% verified analysis)
    strategy_mappings = {}
    
    # Generate all 100 Kelly fractions (1%, 2%, 3%, ..., 100%)
    for i in range(1, 101):
        multiplier = i / 100.0  # Convert to decimal (1% = 0.01, 2% = 0.02, etc.)
        
        if i == 100:
            # Special case for Full Kelly
            strategy_name = 'Full_Kelly'
        else:
            strategy_name = f'Kelly_{i}pct'
        
        strategy_mappings[strategy_name] = multiplier
    
    # Also include legacy strategies that might still exist
    legacy_strategies = {
        'Kelly_5pct_Cap': 0.05,  # Kelly with 5% cap (similar to 1/20)
        'Tenth_Kelly': 0.02,     # Legacy name for 1/50 Kelly = 2%
        # Note: These might not exist but we'll try to load them anyway
    }
    strategy_mappings.update(legacy_strategies)
    
    results = []
    
    print(f"\n🔍 Loading all existing strategies for {model_name.replace('_', ' ')}...")
    
    for strategy_folder, kelly_fraction in strategy_mappings.items():
        betting_data = load_betting_strategy(model_name, strategy_folder)
        
        if betting_data is not None:
            weekly_profits = calculate_weekly_profits(betting_data)
            
            if len(weekly_profits) > 0:
                print(f"   ✅ Loaded {strategy_folder} ({kelly_fraction:.1%})")
                results.append({
                    'kelly_fraction': kelly_fraction,
                    'strategy_name': strategy_folder,
                    'weekly_profits': weekly_profits,
                    'betting_data': betting_data
                })
            else:
                print(f"   ⚠️  {strategy_folder} has no weekly profits")
        else:
            print(f"   ❌ {strategy_folder} not found")
    
    return results

def run_monte_carlo_on_existing_strategies(model_name, n_simulations=10000):
    """
    Run Monte Carlo analysis on all existing betting strategies.
    This matches the methodology of monte_carlo_variance_analysis.py exactly.
    """
    
    # Load all available strategies for this model
    strategy_results = load_all_existing_strategies(model_name)
    
    if not strategy_results:
        print(f"   ❌ No strategies found for {model_name}")
        return None
    
    print(f"   Running {n_simulations:,} Monte Carlo simulations on {len(strategy_results)} strategies...")
    
    results = []
    
    for strategy_info in tqdm(strategy_results, desc=f"   {model_name} strategies"):
        weekly_profits = strategy_info['weekly_profits']
        
        # Run Monte Carlo simulation (same as monte_carlo_variance_analysis.py)
        sim_bankrolls = []
        sim_returns = []
        sim_drawdowns = []
        bust_count = 0
        starting_bankroll = 100.0
        
        for sim in range(n_simulations):
            # Resample periods with replacement to preserve temporal clustering
            sim_period_profits = np.random.choice(weekly_profits, len(weekly_profits), replace=True)
            
            bankroll = starting_bankroll
            max_bankroll = starting_bankroll
            max_drawdown = 0
            
            for period_profit in sim_period_profits:
                bankroll += period_profit
                max_bankroll = max(max_bankroll, bankroll)
                
                # Calculate drawdown from peak
                if max_bankroll > 0:
                    drawdown = (max_bankroll - bankroll) / max_bankroll
                    max_drawdown = max(max_drawdown, drawdown)
                
                # Check for bust
                if bankroll <= 0:
                    bankroll = 0
                    bust_count += 1
                    break
            
            sim_bankrolls.append(bankroll)
            sim_returns.append((bankroll - starting_bankroll) / starting_bankroll)
            sim_drawdowns.append(max_drawdown)
        
        sim_bankrolls = np.array(sim_bankrolls)
        sim_returns = np.array(sim_returns)
        sim_drawdowns = np.array(sim_drawdowns)
        
        # Calculate comprehensive statistics
        mean_return = np.mean(sim_returns)
        std_return = np.std(sim_returns)
        
        # Risk-adjusted metrics
        sharpe_ratio = mean_return / std_return if std_return > 0 else 0
        
        # Risk metrics
        bust_rate = bust_count / n_simulations
        profit_rate = np.sum(sim_returns > 0) / n_simulations
        
        results.append({
            'kelly_fraction': strategy_info['kelly_fraction'],
            'strategy_name': strategy_info['strategy_name'],
            'mean_bankroll': np.mean(sim_bankrolls),
            'mean_return': mean_return * 100,  # Convert to percentage
            'std_return': std_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'bust_rate': bust_rate * 100,  # Convert to percentage
            'profit_rate': profit_rate * 100,
            'max_drawdown': np.mean(sim_drawdowns) * 100,
            'median_return': np.median(sim_returns) * 100,
            'p95_return': np.percentile(sim_returns, 95) * 100,
            'p5_return': np.percentile(sim_returns, 5) * 100,
            'model': model_name
        })
    
    if results:
        results_df = pd.DataFrame(results).sort_values('kelly_fraction')
        print(f"   ✅ Completed Monte Carlo analysis on {len(results)} strategies for {model_name.replace('_', ' ')}")
        return results_df
    else:
        print(f"   ❌ No results generated for {model_name}")
        return None

def create_comprehensive_visualizations(all_results_df, save_dir):
    """Create comprehensive smooth visualizations with 100 data points per model"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up plotting style
    plt.style.use('default')
    sns.set_palette("husl")
    
    models = all_results_df['model'].unique()
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    model_colors = dict(zip(models, colors[:len(models)]))
    
    # 1. Comprehensive Kelly Fraction vs Returns
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        model_data = all_results_df[all_results_df['model'] == model].copy()
        model_data = model_data.sort_values('kelly_fraction')
        
        kelly_fractions = model_data['kelly_fraction'].values
        returns = model_data['mean_return'].values
        
        # Plot smooth curve
        ax.plot(kelly_fractions, returns, color=model_colors[model], linewidth=3, alpha=0.8)
        
        # Add markers at key points
        key_fractions = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
        for key_frac in key_fractions:
            if key_frac <= kelly_fractions.max():
                # Find closest point
                closest_idx = np.argmin(np.abs(kelly_fractions - key_frac))
                ax.scatter(kelly_fractions[closest_idx], returns[closest_idx], 
                         color=model_colors[model], s=120, edgecolor='white', linewidth=2, zorder=5)
                
                # Label key points
                ax.annotate(f'{int(1/key_frac)}/1', 
                          (kelly_fractions[closest_idx], returns[closest_idx]),
                          xytext=(0, 10), textcoords='offset points', 
                          ha='center', fontsize=9, fontweight='bold',
                          bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Kelly Fraction', fontsize=14)
        ax.set_ylabel('Mean Return (%)', fontsize=14)
        ax.set_title(f'{model.replace("_", " ").title()} - Complete Kelly Analysis\n(100 data points)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Use log scale if returns are very large
        max_return = max(returns) if len(returns) > 0 else 0
        if max_return > 10000:
            ax.set_yscale('symlog', linthresh=100)
        
        # Set x-axis labels
        ax.set_xlim(0, 1.0)
        key_labels = ['1/100', '1/50', '1/20', '1/10', '1/4', '1/2', 'Full']
        ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00])
        ax.set_xticklabels(key_labels)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_kelly_returns.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Comprehensive Kelly Fraction vs Bust Rate
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        model_data = all_results_df[all_results_df['model'] == model].copy()
        model_data = model_data.sort_values('kelly_fraction')
        
        kelly_fractions = model_data['kelly_fraction'].values
        bust_rates = model_data['bust_rate'].values
        
        # Plot smooth curve
        ax.plot(kelly_fractions, bust_rates, color=model_colors[model], linewidth=3, alpha=0.8)
        
        # Add markers and labels at key points
        key_fractions = [0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
        for key_frac in key_fractions:
            if key_frac <= kelly_fractions.max():
                closest_idx = np.argmin(np.abs(kelly_fractions - key_frac))
                ax.scatter(kelly_fractions[closest_idx], bust_rates[closest_idx], 
                         color=model_colors[model], s=120, edgecolor='white', linewidth=2, zorder=5)
        
        ax.set_xlabel('Kelly Fraction', fontsize=14)
        ax.set_ylabel('Bust Rate (%)', fontsize=14)
        ax.set_title(f'{model.replace("_", " ").title()} - Risk Analysis\n(100 data points)', 
                    fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Add danger zone markers
        ax.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% Risk')
        ax.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='20% Risk')
        ax.legend(loc='upper left')
        
        ax.set_xlim(0, 1.0)
        ax.set_xticks([0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00])
        ax.set_xticklabels(key_labels)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_kelly_bust_rates.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Combined Risk-Return Efficiency Frontier
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    
    for model in models:
        model_data = all_results_df[all_results_df['model'] == model].copy()
        model_data = model_data.sort_values('kelly_fraction')
        
        returns = model_data['mean_return'].values
        bust_rates = model_data['bust_rate'].values
        kelly_fractions = model_data['kelly_fraction'].values
        
        # Plot efficiency frontier
        ax.plot(bust_rates, returns, color=model_colors[model], linewidth=3, 
               label=f'{model.replace("_", " ").title()}', alpha=0.8)
        
        # Mark optimal points (highest Sharpe ratio)
        sharpe_ratios = model_data['sharpe_ratio'].values
        if len(sharpe_ratios) > 0 and max(sharpe_ratios) > 0:
            max_sharpe_idx = np.argmax(sharpe_ratios)
            ax.scatter(bust_rates[max_sharpe_idx], returns[max_sharpe_idx], 
                      color=model_colors[model], s=200, marker='*', 
                      edgecolor='black', linewidth=2, zorder=10)
            
            optimal_kelly = kelly_fractions[max_sharpe_idx]
            ax.annotate(f'Optimal: {int(1/optimal_kelly)}/1 Kelly\n{returns[max_sharpe_idx]:.0f}%, {bust_rates[max_sharpe_idx]:.1f}% bust', 
                       (bust_rates[max_sharpe_idx], returns[max_sharpe_idx]),
                       xytext=(10, 10), textcoords='offset points', 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    ax.set_xlabel('Bust Rate (%)', fontsize=14)
    ax.set_ylabel('Mean Return (%)', fontsize=14)
    ax.set_title('Comprehensive Kelly Efficiency Frontier\n(100 Kelly fractions per model)', 
                fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add quadrant guides
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axvline(x=20, color='red', linestyle='--', alpha=0.5)
    
    # Fix axis limits based on data - focus on reasonable ranges
    # Filter out extreme outliers for better visualization
    reasonable_returns = []
    reasonable_bust_rates = []
    
    for model in models:
        model_data = all_results_df[all_results_df['model'] == model]
        # Only include strategies with <50% bust rate for main frontier
        reasonable_data = model_data[model_data['bust_rate'] < 50]
        if len(reasonable_data) > 0:
            reasonable_returns.extend(reasonable_data['mean_return'].values)
            reasonable_bust_rates.extend(reasonable_data['bust_rate'].values)
    
    if reasonable_returns:
        max_reasonable_return = max(reasonable_returns)
        max_reasonable_bust = max(reasonable_bust_rates)
        
        # Set sensible limits
        ax.set_ylim(-200, min(max_reasonable_return * 1.1, 2000))  # Cap at 2000% for readability
        ax.set_xlim(-1, min(max_reasonable_bust * 1.1, 60))  # Cap at 60% bust rate
    else:
        ax.set_ylim(-200, 2000)
        ax.set_xlim(0, 50)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/comprehensive_efficiency_frontier.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. CONSERVATIVE RANGE FOCUS (1/100 to 1/20 Kelly) - NEW!
    print("📊 Creating focused charts for practical Kelly ranges...")
    
    # Create multiple focused ranges
    # 1. Ultra-conservative: 1% to 5% Kelly
    ultra_conservative_df = all_results_df[(all_results_df['kelly_fraction'] >= 0.01) & 
                                         (all_results_df['kelly_fraction'] <= 0.05)].copy()
    
    # 2. Practical range: 1% to 10% Kelly (where most real strategies operate)
    practical_df = all_results_df[(all_results_df['kelly_fraction'] >= 0.01) & 
                                (all_results_df['kelly_fraction'] <= 0.10)].copy()
    
    # 3. Reasonable range: 1% to 25% Kelly (excludes extreme high-Kelly strategies)
    reasonable_df = all_results_df[(all_results_df['kelly_fraction'] >= 0.01) & 
                                 (all_results_df['kelly_fraction'] <= 0.25)].copy()
    
    # Filter to conservative range (1% to 5% Kelly) - keep original for backward compatibility
    conservative_df = ultra_conservative_df
    
    if len(conservative_df) > 0:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Returns in conservative range
        ax1 = axes[0, 0]
        for model in models:
            model_data = conservative_df[conservative_df['model'] == model].sort_values('kelly_fraction')
            kelly_fracs = model_data['kelly_fraction'].values
            returns = model_data['mean_return'].values
            
            ax1.plot(kelly_fracs, returns, 'o-', color=model_colors[model], 
                    linewidth=3, markersize=8, label=model.replace('_', ' ').title())
            
            # Mark optimal points in this range
            if len(returns) > 0:
                max_idx = np.argmax(returns)
                ax1.scatter(kelly_fracs[max_idx], returns[max_idx], 
                           color=model_colors[model], s=200, marker='*', 
                           edgecolor='black', linewidth=2, zorder=10)
        
        ax1.set_xlabel('Kelly Fraction', fontsize=14)
        ax1.set_ylabel('Mean Return (%)', fontsize=14)
        ax1.set_title('CONSERVATIVE KELLY RANGE: Returns\n(1/100 to 1/20 Kelly)', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0.009, 0.051)
        
        # Custom x-axis labels for conservative range
        conservative_kelly_fracs = sorted(conservative_df['kelly_fraction'].unique())
        ax1.set_xticks(conservative_kelly_fracs)
        ax1.set_xticklabels([f"1/{int(1/k)}" for k in conservative_kelly_fracs], rotation=45)
        
        # Bust rates in conservative range
        ax2 = axes[0, 1]
        for model in models:
            model_data = conservative_df[conservative_df['model'] == model].sort_values('kelly_fraction')
            kelly_fracs = model_data['kelly_fraction'].values
            bust_rates = model_data['bust_rate'].values
            
            ax2.plot(kelly_fracs, bust_rates, 'o-', color=model_colors[model], 
                    linewidth=3, markersize=8, label=model.replace('_', ' ').title())
        
        ax2.set_xlabel('Kelly Fraction', fontsize=14)
        ax2.set_ylabel('Bust Rate (%)', fontsize=14)
        ax2.set_title('CONSERVATIVE KELLY RANGE: Risk\n(1/100 to 1/20 Kelly)', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.009, 0.051)
        ax2.set_xticks(conservative_kelly_fracs)
        ax2.set_xticklabels([f"1/{int(1/k)}" for k in conservative_kelly_fracs], rotation=45)
        
        # Add safety lines
        ax2.axhline(y=1, color='green', linestyle='--', alpha=0.7, label='1% Risk')
        ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5% Risk')
        ax2.legend(fontsize=10)
        
        # Sharpe ratios in conservative range
        ax3 = axes[1, 0]
        for model in models:
            model_data = conservative_df[conservative_df['model'] == model].sort_values('kelly_fraction')
            kelly_fracs = model_data['kelly_fraction'].values
            sharpe_ratios = model_data['sharpe_ratio'].values
            
            ax3.plot(kelly_fracs, sharpe_ratios, 'o-', color=model_colors[model], 
                    linewidth=3, markersize=8, label=model.replace('_', ' ').title())
        
        ax3.set_xlabel('Kelly Fraction', fontsize=14)
        ax3.set_ylabel('Sharpe Ratio', fontsize=14)
        ax3.set_title('CONSERVATIVE KELLY RANGE: Risk-Adjusted Returns\n(1/100 to 1/20 Kelly)', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.009, 0.051)
        ax3.set_xticks(conservative_kelly_fracs)
        ax3.set_xticklabels([f"1/{int(1/k)}" for k in conservative_kelly_fracs], rotation=45)
        
        # Conservative range risk-return scatter
        ax4 = axes[1, 1]
        for model in models:
            model_data = conservative_df[conservative_df['model'] == model]
            
            # Size bubbles by Kelly conservativeness (1/kelly_fraction)
            sizes = [(1/k) * 3 for k in model_data['kelly_fraction']]
            
            scatter = ax4.scatter(model_data['bust_rate'], model_data['mean_return'], 
                                s=sizes, color=model_colors[model], alpha=0.8,
                                edgecolors='black', linewidth=1, 
                                label=model.replace('_', ' ').title())
            
            # Connect with lines
            sorted_data = model_data.sort_values('kelly_fraction')
            ax4.plot(sorted_data['bust_rate'], sorted_data['mean_return'], 
                    '-', color=model_colors[model], alpha=0.4, linewidth=2)
        
        ax4.set_xlabel('Bust Rate (%)', fontsize=14)
        ax4.set_ylabel('Mean Return (%)', fontsize=14)
        ax4.set_title('CONSERVATIVE RANGE: Risk vs Return\n(Bubble size ∝ conservativeness)', fontsize=16, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-0.5, 10)  # Focus on low bust rates
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/conservative_kelly_focus.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 Conservative range focus chart created!")
    else:
        print("   ⚠️  No data in conservative range (1% to 5%) found")
    
    # 5. PRACTICAL RANGE CHARTS (1% to 10% Kelly)
    if len(practical_df) > 0:
        print("📊 Creating practical range charts (1% to 10% Kelly)...")
        
        fig, axes = plt.subplots(2, 2, figsize=(18, 14))
        
        # Practical range efficiency frontier
        ax1 = axes[0, 0]
        for model in models:
            model_data = practical_df[practical_df['model'] == model].sort_values('kelly_fraction')
            if len(model_data) > 0:
                ax1.plot(model_data['bust_rate'], model_data['mean_return'], 
                        'o-', color=model_colors[model], linewidth=3, markersize=6,
                        label=model.replace('_', ' ').title())
        
        ax1.set_xlabel('Bust Rate (%)', fontsize=14)
        ax1.set_ylabel('Mean Return (%)', fontsize=14)
        ax1.set_title('Practical Kelly Range: Efficiency Frontier\\n(1% to 10% Kelly)', fontsize=16, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax1.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5% Risk')
        
        # Kelly fraction vs Return
        ax2 = axes[0, 1]
        for model in models:
            model_data = practical_df[practical_df['model'] == model].sort_values('kelly_fraction')
            if len(model_data) > 0:
                ax2.plot(model_data['kelly_fraction'], model_data['mean_return'], 
                        'o-', color=model_colors[model], linewidth=3, markersize=6,
                        label=model.replace('_', ' ').title())
        
        ax2.set_xlabel('Kelly Fraction', fontsize=14)
        ax2.set_ylabel('Mean Return (%)', fontsize=14)
        ax2.set_title('Practical Kelly Range: Returns\\n(1% to 10% Kelly)', fontsize=16, fontweight='bold')
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0.005, 0.105)
        
        # Kelly fraction vs Bust Rate
        ax3 = axes[1, 0]
        for model in models:
            model_data = practical_df[practical_df['model'] == model].sort_values('kelly_fraction')
            if len(model_data) > 0:
                ax3.plot(model_data['kelly_fraction'], model_data['bust_rate'], 
                        'o-', color=model_colors[model], linewidth=3, markersize=6,
                        label=model.replace('_', ' ').title())
        
        ax3.set_xlabel('Kelly Fraction', fontsize=14)
        ax3.set_ylabel('Bust Rate (%)', fontsize=14)
        ax3.set_title('Practical Kelly Range: Risk\\n(1% to 10% Kelly)', fontsize=16, fontweight='bold')
        ax3.legend(fontsize=12)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim(0.005, 0.105)
        ax3.axhline(y=5, color='orange', linestyle='--', alpha=0.7)
        
        # Kelly fraction vs Sharpe Ratio
        ax4 = axes[1, 1]
        for model in models:
            model_data = practical_df[practical_df['model'] == model].sort_values('kelly_fraction')
            if len(model_data) > 0:
                ax4.plot(model_data['kelly_fraction'], model_data['sharpe_ratio'], 
                        'o-', color=model_colors[model], linewidth=3, markersize=6,
                        label=model.replace('_', ' ').title())
        
        ax4.set_xlabel('Kelly Fraction', fontsize=14)
        ax4.set_ylabel('Sharpe Ratio', fontsize=14)
        ax4.set_title('Practical Kelly Range: Risk-Adjusted Returns\\n(1% to 10% Kelly)', fontsize=16, fontweight='bold')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0.005, 0.105)
        ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Sharpe = 1.0')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/practical_kelly_range_1_to_10_percent.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 Practical range charts (1-10% Kelly) created!")
    
    # 6. REASONABLE RANGE EFFICIENCY FRONTIER (1% to 25% Kelly)
    if len(reasonable_df) > 0:
        print("📊 Creating reasonable range efficiency frontier (1% to 25% Kelly)...")
        
        fig, ax = plt.subplots(1, 1, figsize=(14, 10))
        
        for model in models:
            model_data = reasonable_df[reasonable_df['model'] == model].sort_values('kelly_fraction')
            if len(model_data) > 0:
                # Plot efficiency frontier
                ax.plot(model_data['bust_rate'], model_data['mean_return'], 
                       color=model_colors[model], linewidth=3, alpha=0.8,
                       label=f'{model.replace("_", " ").title()}')
                
                # Mark key Kelly fractions
                key_fractions = [0.01, 0.02, 0.05, 0.10, 0.25]
                for frac in key_fractions:
                    closest_row = model_data.iloc[(model_data['kelly_fraction'] - frac).abs().argsort()[:1]]
                    if len(closest_row) > 0:
                        ax.scatter(closest_row['bust_rate'].iloc[0], closest_row['mean_return'].iloc[0],
                                 color=model_colors[model], s=80, edgecolor='white', linewidth=2, zorder=5)
        
        ax.set_xlabel('Bust Rate (%)', fontsize=14)
        ax.set_ylabel('Mean Return (%)', fontsize=14)
        ax.set_title('Reasonable Kelly Range: Efficiency Frontier\\n(1% to 25% Kelly - Excludes Extreme High-Risk)', 
                    fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax.axvline(x=5, color='orange', linestyle='--', alpha=0.7, label='5% Risk')
        ax.axvline(x=20, color='red', linestyle='--', alpha=0.7, label='20% Risk')
        
        plt.tight_layout()
        plt.savefig(f"{save_dir}/reasonable_efficiency_frontier_1_to_25_percent.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("   📊 Reasonable range efficiency frontier created!")
        
    print("   ✅ All focused charts created for practical Kelly ranges!")

def find_optimal_strategies(all_results_df):
    """Find optimal Kelly fractions using different criteria"""
    
    optimal_results = []
    
    for model in all_results_df['model'].unique():
        model_data = all_results_df[all_results_df['model'] == model].copy()
        
        # 1. Maximum Sharpe ratio (overall best risk-adjusted)
        max_sharpe_idx = model_data['sharpe_ratio'].idxmax()
        max_sharpe = model_data.loc[max_sharpe_idx]
        
        # 2. Maximum return with bust rate < 5% (safe high-return)
        safe_data = model_data[model_data['bust_rate'] <= 5.0]
        max_safe_return = None
        if len(safe_data) > 0:
            max_safe_idx = safe_data['mean_return'].idxmax()
            max_safe_return = safe_data.loc[max_safe_idx]
        
        # 3. Best return/risk ratio (custom metric)
        positive_returns = model_data[model_data['mean_return'] > 0].copy()
        best_ratio = None
        if len(positive_returns) > 0:
            positive_returns['return_risk_ratio'] = positive_returns['mean_return'] / (positive_returns['bust_rate'] + 0.1)
            best_ratio_idx = positive_returns['return_risk_ratio'].idxmax()
            best_ratio = positive_returns.loc[best_ratio_idx]
        
        optimal_results.append({
            'model': model,
            'max_sharpe_kelly': max_sharpe['kelly_fraction'],
            'max_sharpe_return': max_sharpe['mean_return'],
            'max_sharpe_bust': max_sharpe['bust_rate'],
            'max_sharpe_sharpe': max_sharpe['sharpe_ratio'],
            'safe_kelly': max_safe_return['kelly_fraction'] if max_safe_return is not None else None,
            'safe_return': max_safe_return['mean_return'] if max_safe_return is not None else None,
            'safe_bust': max_safe_return['bust_rate'] if max_safe_return is not None else None,
            'ratio_kelly': best_ratio['kelly_fraction'] if best_ratio is not None else None,
            'ratio_return': best_ratio['mean_return'] if best_ratio is not None else None,
            'ratio_bust': best_ratio['bust_rate'] if best_ratio is not None else None
        })
    
    return pd.DataFrame(optimal_results)

def main():
    print("=" * 80)
    print("COMPREHENSIVE KELLY FRACTION ANALYSIS")
    print("=" * 80)
    print("🎯 USING ACTUAL BETTING STRATEGIES: Load existing betting logs and run Monte Carlo")
    print("📊 Same methodology as monte_carlo_variance_analysis.py")
    print("⏱️  Expected runtime: 2-5 minutes")
    print()
    
    start_time = time.time()
    
    # Configuration
    models = ['xgboost', 'random_forest', 'neural_network_143', 'neural_network_98']
    n_simulations = 10000   # Monte Carlo simulations per strategy
    
    # Create output directories
    output_dir = "analysis_scripts/comprehensive_kelly_optimization"
    charts_dir = f"{output_dir}/charts"
    results_dir = f"{output_dir}/results"
    
    for dir_path in [output_dir, charts_dir, results_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Run analysis on existing betting strategies for each model
    all_results = []
    
    for i, model in enumerate(models):
        print(f"\n{'='*60}")
        print(f"MODEL {i+1}/{len(models)}: {model.replace('_', ' ').upper()}")
        print(f"{'='*60}")
        
        model_results = run_monte_carlo_on_existing_strategies(model, n_simulations)
        
        if model_results is not None:
            all_results.append(model_results)
            
            # Save individual model results
            model_results.to_csv(f"{results_dir}/{model}_comprehensive_kelly.csv", index=False)
            print(f"   💾 Saved results: {results_dir}/{model}_comprehensive_kelly.csv")
        else:
            print(f"   ❌ Failed to generate results for {model}")
    
    if all_results:
        # Combine all results
        print(f"\n📊 Combining results from all models...")
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_results.to_csv(f"{results_dir}/all_models_comprehensive_kelly.csv", index=False)
        
        # Create comprehensive visualizations
        print(f"📈 Creating comprehensive visualizations...")
        create_comprehensive_visualizations(combined_results, charts_dir)
        
        # Find optimal strategies
        print(f"🏆 Finding optimal strategies...")
        optimal_df = find_optimal_strategies(combined_results)
        optimal_df.to_csv(f"{results_dir}/optimal_strategies_comprehensive.csv", index=False)
        
        # Display results
        elapsed_hours = (time.time() - start_time) / 3600
        print(f"\n{'='*80}")
        print(f"🎉 COMPREHENSIVE KELLY OPTIMIZATION COMPLETE!")
        print(f"{'='*80}")
        print(f"⏱️  Total runtime: {elapsed_hours:.2f} hours")
        print(f"📊 Total simulations: {len(combined_results) * n_simulations:,}")
        print(f"📁 Results saved to: {results_dir}")
        print(f"📈 Charts saved to: {charts_dir}")
        
        print(f"\n🏆 OPTIMAL KELLY STRATEGIES:")
        print("-" * 80)
        for _, row in optimal_df.iterrows():
            model_name = row['model'].replace('_', ' ').title()
            if pd.notna(row['max_sharpe_kelly']):
                kelly_name = f"1/{int(1/row['max_sharpe_kelly'])}" if row['max_sharpe_kelly'] < 1 else "Full"
                print(f"{model_name:<20} {kelly_name} Kelly ({row['max_sharpe_kelly']:.1%}) - "
                      f"Return: {row['max_sharpe_return']:+.1f}%, Bust: {row['max_sharpe_bust']:.1f}%, "
                      f"Sharpe: {row['max_sharpe_sharpe']:.2f}")
            
            if pd.notna(row['safe_kelly']):
                safe_kelly_name = f"1/{int(1/row['safe_kelly'])}" if row['safe_kelly'] < 1 else "Full"
                print(f"{'  Safe (<5% bust):':<20} {safe_kelly_name} Kelly ({row['safe_kelly']:.1%}) - "
                      f"Return: {row['safe_return']:+.1f}%, Bust: {row['safe_bust']:.1f}%")
        
        print(f"\n✅ Perfect! Now you have smooth curves with 100 data points per model!")
        
    else:
        print("❌ No results generated. Check that betting log files exist.")
        elapsed = time.time() - start_time
        print(f"⏱️  Failed after {elapsed/60:.1f} minutes")

if __name__ == "__main__":
    main()