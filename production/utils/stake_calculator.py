#!/usr/bin/env python3
"""
Kelly Betting Stake Calculator for Live Tennis Betting
Implements block betting strategy with risk management
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import math

class KellyStakeCalculator:
    """Calculate Kelly stakes for tennis betting with block execution"""
    
    def __init__(self, 
                 kelly_multiplier: float = 0.18,
                 edge_threshold: float = 0.02, 
                 max_stake_fraction: float = 0.05,
                 min_stake_dollars: float = 1.0,
                 allow_leverage: bool = False):
        """
        Initialize stake calculator
        
        Args:
            kelly_multiplier: Kelly fraction multiplier (e.g., 0.18 for 18% Kelly)
            edge_threshold: Minimum edge required to place bet
            max_stake_fraction: Maximum fraction of bankroll to risk per bet
            min_stake_dollars: Minimum bet size in dollars
            allow_leverage: Whether to allow total stakes > bankroll
        """
        self.kelly_multiplier = kelly_multiplier
        self.edge_threshold = edge_threshold
        self.max_stake_fraction = max_stake_fraction
        self.min_stake_dollars = min_stake_dollars
        self.allow_leverage = allow_leverage
    
    def kelly_fraction(self, win_prob: float, decimal_odds: float) -> float:
        """Calculate optimal Kelly fraction for a single bet"""
        if win_prob <= 0 or decimal_odds <= 1.01:
            return 0.0
        
        b = decimal_odds - 1.0  # Net odds (profit per dollar)
        p = win_prob            # Win probability
        q = 1.0 - p            # Loss probability
        
        # Kelly formula: f* = (bp - q) / b
        kelly_frac = (b * p - q) / b
        return max(0.0, kelly_frac)
    
    def filter_betting_opportunities(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Filter matches that meet betting criteria"""
        # Only consider matches with positive edge above threshold
        betting_opportunities = predictions_df[
            predictions_df['best_edge'] >= self.edge_threshold
        ].copy()
        
        if betting_opportunities.empty:
            print("📊 No betting opportunities found above edge threshold")
            return pd.DataFrame()
        
        print(f"📊 Found {len(betting_opportunities)} betting opportunities")
        return betting_opportunities
    
    def allocate_block_stakes(self, 
                            matches_df: pd.DataFrame, 
                            bankroll: float,
                            grouping: str = 'day') -> pd.DataFrame:
        """
        Allocate stakes across a block of simultaneous matches
        
        Args:
            matches_df: DataFrame with betting opportunities
            bankroll: Current bankroll amount
            grouping: How to group matches ('day', 'event', 'all')
            
        Returns:
            DataFrame with stake allocations
        """
        if matches_df.empty:
            return pd.DataFrame()
        
        # Group matches by specified method
        if grouping == 'day':
            # Group all matches happening on same day, handling unknown times
            if 'match_time' in matches_df.columns:
                match_times = matches_df['match_time'].fillna('today').replace('Unknown', 'today')
            else:
                match_times = pd.Series(['today'] * len(matches_df), index=matches_df.index)
            matches_df['group_key'] = pd.to_datetime(match_times, errors='coerce').dt.date
        elif grouping == 'event':
            # Group by tournament event
            matches_df['group_key'] = matches_df['event']
        else:
            # Treat all matches as one big block
            matches_df['group_key'] = 'all'
        
        stake_results = []
        
        for group_key, group_matches in matches_df.groupby('group_key'):
            print(f"\n💰 Calculating stakes for group: {group_key} ({len(group_matches)} matches)")
            
            # Calculate block budget (Kelly multiplier * bankroll)
            block_budget = self.kelly_multiplier * bankroll
            
            # Calculate individual Kelly fractions
            kelly_fractions = []
            for _, match in group_matches.iterrows():
                kelly_frac = self.kelly_fraction(match['bet_prob'], match['bet_odds'])
                # Apply max fraction cap
                kelly_frac = min(kelly_frac, self.max_stake_fraction)
                kelly_fractions.append(kelly_frac)
            
            # Calculate proportional stakes
            total_kelly = sum(kelly_fractions)
            
            if total_kelly <= 0:
                print("   ⚠️  No positive Kelly fractions in this group")
                continue
            
            individual_stakes = []
            for i, (_, match) in enumerate(group_matches.iterrows()):
                if kelly_fractions[i] > 0:
                    # Proportional allocation based on Kelly fraction
                    proportional_stake = block_budget * (kelly_fractions[i] / total_kelly)
                else:
                    proportional_stake = 0.0
                
                individual_stakes.append(proportional_stake)
            
            # Check if total stakes exceed bankroll
            total_stakes = sum(individual_stakes)
            
            if total_stakes > bankroll and not self.allow_leverage:
                # Scale down all stakes proportionally
                scale_factor = bankroll / total_stakes
                individual_stakes = [stake * scale_factor for stake in individual_stakes]
                total_stakes = sum(individual_stakes)
                print(f"   📉 Scaled down stakes by {scale_factor:.3f} (total: ${total_stakes:.2f})")
            
            # Apply minimum stake filter and build results
            for i, (_, match) in enumerate(group_matches.iterrows()):
                stake = individual_stakes[i]
                
                # Skip bets below minimum stake
                if stake < self.min_stake_dollars:
                    stake = 0.0
                
                stake_info = {
                    **match.to_dict(),
                    'kelly_fraction': kelly_fractions[i],
                    'proportional_stake': individual_stakes[i],
                    'final_stake': stake,
                    'stake_dollars': stake,  # Explicit dollar amount
                    'stake_fraction': stake / bankroll if bankroll > 0 else 0.0,
                    'stake_percentage': (stake / bankroll * 100) if bankroll > 0 else 0.0,
                    'group_key': group_key,
                    'block_budget': block_budget,
                    'block_budget_dollars': block_budget,
                    'total_block_stakes': total_stakes,
                    'bankroll_snapshot': bankroll
                }
                
                if stake > 0:
                    stake_results.append(stake_info)
        
        if stake_results:
            stakes_df = pd.DataFrame(stake_results)
            total_stakes = stakes_df['final_stake'].sum()
            print(f"\n💵 Final allocation:")
            print(f"   Bankroll: ${bankroll:.2f}")
            print(f"   Total stakes: ${total_stakes:.2f} ({total_stakes/bankroll:.1%})")
            print(f"   Number of bets: {len(stakes_df)}")
            return stakes_df
        else:
            print("📊 No bets meet minimum stake requirements")
            return pd.DataFrame()
    
    def generate_bet_slips(self, stakes_df: pd.DataFrame) -> pd.DataFrame:
        """Generate final bet slips with all necessary information"""
        if stakes_df.empty:
            return pd.DataFrame()
        
        bet_slips = []
        
        for _, stake_info in stakes_df.iterrows():
            # Calculate potential profit/loss
            potential_profit = stake_info['final_stake'] * (stake_info['bet_odds'] - 1.0)
            potential_loss = stake_info['final_stake']
            
            bet_slip = {
                'bet_id': f"bet_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(bet_slips)}",
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event': stake_info.get('event', 'Unknown Event'),
                'match': f"{stake_info['player1_raw']} vs {stake_info['player2_raw']}",
                'match_uid': stake_info.get('match_uid'),
                'feature_snapshot_id': stake_info.get('feature_snapshot_id'),
                'run_id': stake_info.get('run_id'),
                'bet_on': stake_info['bet_player'],
                'bet_on_player1': stake_info['bet_on_player1'],
                'odds_decimal': stake_info['bet_odds'],
                'stake': stake_info['final_stake'],
                'stake_fraction': stake_info['stake_fraction'],
                'model_prob': stake_info['bet_prob'],
                'market_prob': stake_info['market_prob'],
                'edge': stake_info['best_edge'],
                'kelly_fraction': stake_info['kelly_fraction'],
                'potential_profit': potential_profit,
                'potential_loss': potential_loss,
                'group_key': stake_info['group_key'],
                'bankroll': stake_info['bankroll_snapshot'],
                'model_version': stake_info.get('model_version', 'NN-143'),
                'match_date': stake_info.get('meta_match_date', stake_info.get('match_date', '')),
                'match_start_time': stake_info.get('match_time', ''),
                'status': 'pending'
            }
            bet_slips.append(bet_slip)
        
        return pd.DataFrame(bet_slips)

def simulate_daily_betting(bankroll: float = 1000.0,
                         kelly_multiplier: float = 0.18,
                         sample_opportunities: int = 5) -> pd.DataFrame:
    """Simulate a day of betting for testing"""
    print(f"🎲 Simulating daily betting with ${bankroll} bankroll and {kelly_multiplier:.1%} Kelly")
    
    # Create sample betting opportunities
    np.random.seed(42)  # For reproducible results
    
    sample_data = []
    for i in range(sample_opportunities):
        # Random match data
        odds1 = np.random.uniform(1.5, 3.0)
        odds2 = np.random.uniform(1.5, 3.0)
        
        model_prob1 = np.random.uniform(0.3, 0.7)
        model_prob2 = 1.0 - model_prob1
        
        market_prob1 = 1.0 / odds1
        market_prob2 = 1.0 / odds2
        
        edge1 = model_prob1 - market_prob1
        edge2 = model_prob2 - market_prob2
        
        best_edge = max(edge1, edge2)
        bet_on_player1 = edge1 >= edge2
        
        if best_edge > 0.02:  # Only include profitable opportunities
            sample_data.append({
                'player1_raw': f'Player {i*2+1}',
                'player2_raw': f'Player {i*2+2}',
                'event': f'ATP Tournament {(i % 3) + 1}',
                'match_time': '2025-08-24 14:00:00',
                'player1_win_prob': model_prob1,
                'player2_win_prob': model_prob2,
                'player1_odds_decimal': odds1,
                'player2_odds_decimal': odds2,
                'player1_implied_prob': market_prob1,
                'player2_implied_prob': market_prob2,
                'edge_player1': edge1,
                'edge_player2': edge2,
                'best_edge': best_edge,
                'bet_on_player1': bet_on_player1,
                'bet_player': f'Player {i*2+1}' if bet_on_player1 else f'Player {i*2+2}',
                'bet_odds': odds1 if bet_on_player1 else odds2,
                'bet_prob': model_prob1 if bet_on_player1 else model_prob2,
                'market_prob': market_prob1 if bet_on_player1 else market_prob2,
                'model_version': 'NN-143'
            })
    
    if not sample_data:
        print("📊 No profitable opportunities in simulation")
        return pd.DataFrame()
    
    opportunities_df = pd.DataFrame(sample_data)
    
    # Calculate stakes
    calculator = KellyStakeCalculator(
        kelly_multiplier=kelly_multiplier,
        edge_threshold=0.02,
        max_stake_fraction=0.05
    )
    
    filtered_df = calculator.filter_betting_opportunities(opportunities_df)
    stakes_df = calculator.allocate_block_stakes(filtered_df, bankroll, grouping='day')
    bet_slips_df = calculator.generate_bet_slips(stakes_df)
    
    return bet_slips_df

def main():
    """Test the stake calculator"""
    bet_slips = simulate_daily_betting(bankroll=1000.0, kelly_multiplier=0.18)
    
    if not bet_slips.empty:
        print("\n📋 Generated bet slips:")
        print(bet_slips[['match', 'bet_on', 'odds_decimal', 'stake', 'edge']].to_string(index=False))
        
        total_risk = bet_slips['stake'].sum()
        total_potential_profit = bet_slips['potential_profit'].sum()
        print(f"\n💰 Summary:")
        print(f"   Total risk: ${total_risk:.2f}")
        print(f"   Total potential profit: ${total_potential_profit:.2f}")
        print(f"   Number of bets: {len(bet_slips)}")
    else:
        print("❌ No bet slips generated")

if __name__ == "__main__":
    main()
