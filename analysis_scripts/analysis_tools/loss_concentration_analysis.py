#!/usr/bin/env python3
"""
EXPECTED VALUE ANALYSIS

Analyzes betting performance by Expected Value per bet, not absolute dollars.
Identifies bet characteristics yielding negative EV despite positive predicted edge.

Key Questions:
1. Which edge/confidence ranges have negative EV per bet?
2. Are there statistically significant patterns of miscalibration?
3. What bet characteristics should be eliminated to improve EV?

Confidence = model_prob_bet_on (probability that the player we bet on wins)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
from scipy.stats import binomtest

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

def analyze_expected_value_patterns(df, model_name, strategy_name):
    """Analyze EV patterns using meaningful ranges and statistical tests"""
    
    if df is None or len(df) == 0:
        return None
    
    print(f"\n{'='*70}")
    print(f"{model_name.upper()} - {strategy_name}")
    print(f"{'='*70}")
    
    # Basic stats
    wins = df[df['bet_won'] == True]
    losses = df[df['bet_won'] == False]
    total_profit = df['profit'].sum()
    
    print(f"Total bets: {len(df):,}")
    print(f"Win rate: {len(wins)/len(df)*100:.1f}% ({len(wins):,} wins, {len(losses):,} losses)")
    print(f"Total profit: ${total_profit:,.0f}")
    print(f"Average profit per bet: ${total_profit/len(df):.2f}")
    
    # Define meaningful edge ranges  
    edge_ranges = [
        (0.00, 0.03, "0-3%"),
        (0.03, 0.05, "3-5%"),
        (0.05, 0.08, "5-8%"), 
        (0.08, 0.12, "8-12%"),
        (0.12, 0.20, "12-20%"),
        (0.20, 1.00, ">20%")
    ]
    
    # Define confidence ranges (probability model thinks our bet will win)
    conf_ranges = [
        (0.0, 0.55, "Low Conf (<55%)"),
        (0.55, 0.65, "Med-Low (55-65%)"),
        (0.65, 0.75, "Med-High (65-75%)"),
        (0.75, 0.85, "High (75-85%)"),
        (0.85, 1.00, "Very High (>85%)")
    ]
    
    print(f"\n📊 EXPECTED VALUE ANALYSIS BY EDGE RANGE:")
    print(f"{'Range':<12} {'Bets':<8} {'Win Rate':<10} {'Expected':<10} {'EV/Bet':<10} {'P-Value':<10} {'Verdict':<15}")
    print("-" * 90)
    
    edge_analysis = []
    
    for min_edge, max_edge, range_name in edge_ranges:
        range_bets = df[(df['edge'] >= min_edge) & (df['edge'] < max_edge)]
        
        if len(range_bets) < 20:  # Need minimum sample size
            continue
            
        range_wins = len(range_bets[range_bets['bet_won'] == True])
        range_losses = len(range_bets[range_bets['bet_won'] == False])
        actual_win_rate = range_wins / len(range_bets)
        
        # Expected win rate based on model probabilities
        expected_win_rate = range_bets['model_prob_bet_on'].mean()
        
        # EV per bet
        ev_per_bet = range_bets['profit'].sum() / len(range_bets)
        
        # Statistical test - are we losing more than expected?
        p_value = binomtest(range_losses, len(range_bets), 1 - expected_win_rate, alternative='greater').pvalue
        
        # Verdict
        if ev_per_bet < -0.50 and p_value < 0.05:
            verdict = "❌ ELIMINATE"
        elif ev_per_bet < 0 and p_value < 0.05:
            verdict = "⚠️  MISCALIBRATED"
        elif ev_per_bet > 0:
            verdict = "✅ PROFITABLE"
        else:
            verdict = "🔄 VARIANCE"
        
        print(f"{range_name:<12} {len(range_bets):<8} {actual_win_rate:<9.1%} {expected_win_rate:<9.1%} "
              f"${ev_per_bet:<9.2f} {p_value:<9.3f} {verdict:<15}")
        
        edge_analysis.append({
            'range': range_name,
            'bets': len(range_bets),
            'actual_win_rate': actual_win_rate,
            'expected_win_rate': expected_win_rate,
            'ev_per_bet': ev_per_bet,
            'p_value': p_value,
            'verdict': verdict
        })
    
    print(f"\n📊 EXPECTED VALUE ANALYSIS BY CONFIDENCE RANGE:")
    print(f"{'Range':<18} {'Bets':<8} {'Win Rate':<10} {'Expected':<10} {'EV/Bet':<10} {'P-Value':<10} {'Verdict':<15}")
    print("-" * 95)
    
    conf_analysis = []
    
    for min_conf, max_conf, range_name in conf_ranges:
        range_bets = df[(df['model_prob_bet_on'] >= min_conf) & (df['model_prob_bet_on'] < max_conf)]
        
        if len(range_bets) < 20:  # Need minimum sample size
            continue
            
        range_wins = len(range_bets[range_bets['bet_won'] == True])
        range_losses = len(range_bets[range_bets['bet_won'] == False])
        actual_win_rate = range_wins / len(range_bets)
        
        # Expected win rate should match confidence range
        expected_win_rate = range_bets['model_prob_bet_on'].mean()
        
        # EV per bet
        ev_per_bet = range_bets['profit'].sum() / len(range_bets)
        
        # Statistical test
        p_value = binomtest(range_losses, len(range_bets), 1 - expected_win_rate, alternative='greater').pvalue
        
        # Verdict
        if ev_per_bet < -0.50 and p_value < 0.05:
            verdict = "❌ ELIMINATE"
        elif ev_per_bet < 0 and p_value < 0.05:
            verdict = "⚠️  MISCALIBRATED"
        elif ev_per_bet > 0:
            verdict = "✅ PROFITABLE"
        else:
            verdict = "🔄 VARIANCE"
        
        print(f"{range_name:<18} {len(range_bets):<8} {actual_win_rate:<9.1%} {expected_win_rate:<9.1%} "
              f"${ev_per_bet:<9.2f} {p_value:<9.3f} {verdict:<15}")
        
        conf_analysis.append({
            'range': range_name,
            'bets': len(range_bets),
            'actual_win_rate': actual_win_rate,
            'expected_win_rate': expected_win_rate,
            'ev_per_bet': ev_per_bet,
            'p_value': p_value,
            'verdict': verdict
        })
    
    # Find eliminable patterns
    eliminable_edges = [x for x in edge_analysis if x['verdict'] == "❌ ELIMINATE"]
    eliminable_confs = [x for x in conf_analysis if x['verdict'] == "❌ ELIMINATE"]
    miscalibrated_edges = [x for x in edge_analysis if x['verdict'] == "⚠️  MISCALIBRATED"]
    miscalibrated_confs = [x for x in conf_analysis if x['verdict'] == "⚠️  MISCALIBRATED"]
    
    print(f"\n💡 ACTIONABLE RECOMMENDATIONS:")
    
    if eliminable_edges:
        print(f"🚫 ELIMINATE EDGE RANGES (Negative EV + Significant):")
        for pattern in eliminable_edges:
            eliminated_bets = pattern['bets']
            total_bets = len(df)
            pct_filtered = eliminated_bets / total_bets * 100
            print(f"   • {pattern['range']} edge: {eliminated_bets:,} bets ({pct_filtered:.1f}%), "
                  f"EV: ${pattern['ev_per_bet']:.2f}/bet")
    
    if eliminable_confs:
        print(f"🚫 ELIMINATE CONFIDENCE RANGES (Negative EV + Significant):")
        for pattern in eliminable_confs:
            eliminated_bets = pattern['bets']
            total_bets = len(df)
            pct_filtered = eliminated_bets / total_bets * 100
            print(f"   • {pattern['range']}: {eliminated_bets:,} bets ({pct_filtered:.1f}%), "
                  f"EV: ${pattern['ev_per_bet']:.2f}/bet")
    
    if miscalibrated_edges:
        print(f"⚠️  MISCALIBRATED EDGE RANGES (Consider limits):")
        for pattern in miscalibrated_edges:
            print(f"   • {pattern['range']} edge: Expected {pattern['expected_win_rate']:.1%} win, "
                  f"Actual {pattern['actual_win_rate']:.1%}, EV: ${pattern['ev_per_bet']:.2f}/bet")
    
    if miscalibrated_confs:
        print(f"⚠️  MISCALIBRATED CONFIDENCE RANGES (Consider limits):")
        for pattern in miscalibrated_confs:
            print(f"   • {pattern['range']}: Expected {pattern['expected_win_rate']:.1%} win, "
                  f"Actual {pattern['actual_win_rate']:.1%}, EV: ${pattern['ev_per_bet']:.2f}/bet")
    
    if not eliminable_edges and not eliminable_confs and not miscalibrated_edges and not miscalibrated_confs:
        print(f"✅ No statistically significant negative EV patterns found!")
        print(f"   All bet ranges appear to have positive or neutral expected value.")
    
    # Calculate impact of eliminating negative patterns
    total_negative_bets = sum(x['bets'] for x in eliminable_edges + eliminable_confs)
    if total_negative_bets > 0:
        total_negative_ev = sum(x['ev_per_bet'] * x['bets'] for x in eliminable_edges + eliminable_confs)
        pct_bets_eliminated = total_negative_bets / len(df) * 100
        print(f"\n📈 FILTERING IMPACT:")
        print(f"   Eliminating {total_negative_bets:,} bets ({pct_bets_eliminated:.1f}%)")
        print(f"   Would remove ${total_negative_ev:,.0f} in negative EV")
        print(f"   Estimated improvement: ${-total_negative_ev/len(df):.2f} per bet")
    
    return {
        'model': model_name,
        'strategy': strategy_name,
        'edge_analysis': edge_analysis,
        'conf_analysis': conf_analysis,
        'total_bets': len(df),
        'overall_ev': total_profit / len(df)
    }

def main():
    print("=" * 80)
    print("EXPECTED VALUE ANALYSIS - IDENTIFYING NEGATIVE EV PATTERNS")
    print("=" * 80)
    print("Analyzing EV per bet to find statistically significant miscalibration")
    print("Confidence = probability model thinks our bet will win")
    print()
    
    # Analyze key model/strategy combinations
    test_cases = [
        ('neural_network_143', 'Fixed_Bankroll'),
        ('neural_network_98', 'Fixed_Bankroll'), 
        ('neural_network_143', 'Kelly_5pct_Cap'),
        ('neural_network_143', 'Quarter_Kelly')
    ]
    
    all_results = {}
    
    for model_name, strategy_folder in test_cases:
        strategy_name = strategy_folder.replace('_', ' ')
        
        # Load data
        df = load_betting_data(strategy_folder, model_name)
        
        if df is not None:
            results = analyze_expected_value_patterns(df, model_name, strategy_name)
            all_results[f"{model_name}_{strategy_folder}"] = results
    
    print(f"\n" + "=" * 80)
    print(f"STATISTICAL METHODOLOGY")
    print(f"=" * 80)
    print(f"✅ EV/Bet = Average profit per bet in each range")
    print(f"✅ P-Value = Binomial test if losses exceed model predictions")
    print(f"✅ P < 0.05 = Statistically significant miscalibration")
    print(f"✅ Eliminate ranges with negative EV + p < 0.05")
    print(f"✅ This identifies genuine negative EV patterns, not just variance")

if __name__ == "__main__":
    main()