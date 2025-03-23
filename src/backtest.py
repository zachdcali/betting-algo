# src/backtest.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime
import os
from betting_model import TennisUTRModel

class TennisBetTracker:
    def __init__(self, initial_bankroll=1000, stake_method="fixed_percentage", stake_param=0.02):
        """
        Initialize the bet tracker
        
        Parameters:
        initial_bankroll: Starting bankroll
        stake_method: "fixed_percentage", "kelly", or "fixed_amount"
        stake_param: Percentage of bankroll or fixed amount depending on stake_method
        """
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.stake_method = stake_method
        self.stake_param = stake_param
        
        # Create directories for data
        base_dir = Path(__file__).parent.parent
        self.data_dir = base_dir / "data"
        self.data_dir.mkdir(exist_ok=True)
        
        self.betting_log_file = self.data_dir / "betting_log.csv"
        
        # Initialize betting log if it doesn't exist
        if not self.betting_log_file.exists():
            self.betting_log = pd.DataFrame(columns=[
                'date', 'match', 'player1', 'player2', 'player1_utr', 'player2_utr', 
                'utr_diff', 'utr_win_prob', 'player1_odds', 'player2_odds', 
                'bet_on', 'bet_amount', 'bet_odds', 'result', 'payout', 'profit',
                'bankroll', 'roi', 'notes'
            ])
            self.betting_log.to_csv(self.betting_log_file, index=False)
        else:
            self.betting_log = pd.read_csv(self.betting_log_file)
            # Update current bankroll
            if not self.betting_log.empty:
                self.current_bankroll = self.betting_log.iloc[-1]['bankroll']
    
    def calculate_bet_amount(self, odds, win_prob):
        """Calculate bet amount based on chosen staking method"""
        if self.stake_method == "fixed_percentage":
            return self.current_bankroll * self.stake_param
        
        elif self.stake_method == "kelly":
            # Convert American odds to decimal odds
            if odds > 0:
                decimal_odds = (odds / 100) + 1
            else:
                decimal_odds = (100 / abs(odds)) + 1
            
            # Kelly formula: f* = (p * (b+1) - 1) / b 
            # where f* is fraction of bankroll to bet, p is probability of winning, b is decimal odds-1
            b = decimal_odds - 1
            kelly = (win_prob * (b + 1) - 1) / b
            
            # Apply a Kelly fraction for safety (often 1/4 or 1/2 Kelly)
            kelly_fraction = 0.25
            
            # Ensure positive stake and cap at 10% of bankroll for safety
            stake = max(0, min(kelly * kelly_fraction, 0.1)) * self.current_bankroll
            return stake
        
        elif self.stake_method == "fixed_amount":
            return self.stake_param
        
        else:
            return self.current_bankroll * 0.01  # Default to 1% of bankroll
    
    def record_bet(self, match_data, bet_on, bet_amount, result=None, notes=""):
        """
        Record a bet in the betting log
        
        Parameters:
        match_data: Dictionary with match details
        bet_on: Name of player bet on
        bet_amount: Amount wagered
        result: 'win', 'loss', or None if pending
        notes: Additional notes
        """
        # Determine which player is being bet on
        is_player1 = (bet_on == match_data['player1'])
        bet_odds = match_data['player1_odds'] if is_player1 else match_data['player2_odds']
        
        # Calculate payout and profit if result is known
        payout = 0
        profit = 0
        new_bankroll = self.current_bankroll
        
        if result == 'win':
            # Calculate payout based on American odds
            if bet_odds > 0:
                payout = bet_amount * (bet_odds / 100) + bet_amount
            else:
                payout = bet_amount * (100 / abs(bet_odds)) + bet_amount
            profit = payout - bet_amount
            new_bankroll = self.current_bankroll + profit
        elif result == 'loss':
            payout = 0
            profit = -bet_amount
            new_bankroll = self.current_bankroll + profit
        
        # Create bet record
        bet_record = {
            'date': datetime.now().strftime('%Y-%m-%d'),
            'match': f"{match_data['player1']} vs {match_data['player2']}",
            'player1': match_data['player1'],
            'player2': match_data['player2'],
            'player1_utr': match_data.get('player1_utr', None),
            'player2_utr': match_data.get('player2_utr', None),
            'utr_diff': match_data.get('utr_diff', None),
            'utr_win_prob': match_data.get('utr_win_prob', None),
            'player1_odds': match_data['player1_odds'],
            'player2_odds': match_data['player2_odds'],
            'bet_on': bet_on,
            'bet_amount': bet_amount,
            'bet_odds': bet_odds,
            'result': result,
            'payout': payout,
            'profit': profit,
            'bankroll': new_bankroll,
            'roi': (profit / bet_amount) if bet_amount > 0 else 0,
            'notes': notes
        }
        
        # Update current bankroll if result is known
        if result in ['win', 'loss']:
            self.current_bankroll = new_bankroll
        
        # Append to betting log
        self.betting_log = pd.concat([self.betting_log, pd.DataFrame([bet_record])], ignore_index=True)
        
        # Save updated log
        self.betting_log.to_csv(self.betting_log_file, index=False)
        
        print(f"Bet recorded: {bet_on} in {match_data['player1']} vs {match_data['player2']}")
        if result:
            print(f"Result: {result.upper()}, Profit: ${profit:.2f}, New bankroll: ${new_bankroll:.2f}")
    
    def update_bet_result(self, match_id, winner, notes=""):
        """
        Update a pending bet with its result
        
        Parameters:
        match_id: Index or match identifier
        winner: Name of player who won the match
        notes: Additional notes
        """
        # Find the bet
        if isinstance(match_id, int):
            bet_idx = match_id
        else:
            # Find by match identifier
            matches = self.betting_log[self.betting_log['match'] == match_id]
            if matches.empty:
                print(f"No bet found for match: {match_id}")
                return
            bet_idx = matches.index[-1]
        
        # Get bet details
        bet = self.betting_log.loc[bet_idx]
        bet_on = bet['bet_on']
        bet_amount = bet['bet_amount']
        bet_odds = bet['bet_odds']
        
        # Determine result
        result = 'win' if bet_on == winner else 'loss'
        
        # Calculate payout and profit
        if result == 'win':
            if bet_odds > 0:
                payout = bet_amount * (bet_odds / 100) + bet_amount
            else:
                payout = bet_amount * (100 / abs(bet_odds)) + bet_amount
            profit = payout - bet_amount
        else:
            payout = 0
            profit = -bet_amount
        
        # Update bankroll
        new_bankroll = self.betting_log.iloc[bet_idx-1]['bankroll'] + profit if bet_idx > 0 else self.initial_bankroll + profit
        
        # Update bet record
        self.betting_log.at[bet_idx, 'result'] = result
        self.betting_log.at[bet_idx, 'payout'] = payout
        self.betting_log.at[bet_idx, 'profit'] = profit
        self.betting_log.at[bet_idx, 'bankroll'] = new_bankroll
        self.betting_log.at[bet_idx, 'roi'] = profit / bet_amount if bet_amount > 0 else 0
        self.betting_log.at[bet_idx, 'notes'] = notes
        
        # Update current bankroll
        self.current_bankroll = new_bankroll
        
        # Save updated log
        self.betting_log.to_csv(self.betting_log_file, index=False)
        
        print(f"Bet updated: {bet_on} in {bet['match']}")
        print(f"Result: {result.upper()}, Profit: ${profit:.2f}, New bankroll: ${new_bankroll:.2f}")
    
    def get_performance_stats(self):
        """
        Calculate performance statistics from betting log
        
        Returns:
        Dictionary of performance metrics
        """
        completed_bets = self.betting_log[self.betting_log['result'].isin(['win', 'loss'])]
        
        if completed_bets.empty:
            return {"status": "No completed bets yet"}
        
        total_bets = len(completed_bets)
        winning_bets = len(completed_bets[completed_bets['result'] == 'win'])
        losing_bets = len(completed_bets[completed_bets['result'] == 'loss'])
        
        win_rate = winning_bets / total_bets
        
        total_wagered = completed_bets['bet_amount'].sum()
        total_profit = completed_bets['profit'].sum()
        roi = total_profit / total_wagered if total_wagered > 0 else 0
        
        # Calculate profit by UTR difference bins
        completed_bets['utr_diff_bin'] = pd.cut(
            completed_bets['utr_diff'], 
            bins=[-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5],
            labels=['< -2', '-2 to -1', '-1 to -0.5', '-0.5 to 0', 
                   '0 to 0.5', '0.5 to 1', '1 to 2', '> 2']
        )
        
        profit_by_diff = completed_bets.groupby('utr_diff_bin')['profit'].sum()
        
        stats = {
            'total_bets': total_bets,
            'winning_bets': winning_bets,
            'losing_bets': losing_bets,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'roi': roi,
            'initial_bankroll': self.initial_bankroll,
            'current_bankroll': self.current_bankroll,
            'profit_by_utr_diff': profit_by_diff.to_dict()
        }
        
        return stats
    
    def plot_bankroll_history(self):
        """Plot bankroll history over time"""
        if self.betting_log.empty:
            print("No betting data available")
            return
        
        # Ensure date column is properly formatted
        self.betting_log['date'] = pd.to_datetime(self.betting_log['date'])
        
        # Get completed bets
        completed_bets = self.betting_log[self.betting_log['result'].isin(['win', 'loss'])].copy()
        
        if completed_bets.empty:
            print("No completed bets yet")
            return
        
        # Sort by date
        completed_bets = completed_bets.sort_values('date')
        
        plt.figure(figsize=(12, 6))
        plt.plot(completed_bets['date'], completed_bets['bankroll'])
        plt.grid(True, alpha=0.3)
        plt.xlabel('Date')
        plt.ylabel('Bankroll ($)')
        plt.title('Bankroll History')
        
        # Add initial bankroll line
        plt.axhline(y=self.initial_bankroll, color='r', linestyle='--', 
                   label=f'Initial: ${self.initial_bankroll}')
        
        # Format the y-axis as currency
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        
        # Add win and loss markers
        wins = completed_bets[completed_bets['result'] == 'win']
        losses = completed_bets[completed_bets['result'] == 'loss']
        
        plt.scatter(wins['date'], wins['bankroll'], color='g', alpha=0.6, label='Win')
        plt.scatter(losses['date'], losses['bankroll'], color='r', alpha=0.6, label='Loss')
        
        plt.legend()
        
        # Save plot
        base_dir = Path(__file__).parent.parent
        plots_dir = base_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / 'bankroll_history.png')
        
        plt.show()
    
    def plot_profit_by_utr_diff(self):
        """Plot profit by UTR difference bins"""
        completed_bets = self.betting_log[self.betting_log['result'].isin(['win', 'loss'])].copy()
        
        if len(completed_bets) < 5:
            print("Not enough completed bets for meaningful analysis")
            return
        
        # Create UTR difference bins if not already present
        if 'utr_diff_bin' not in completed_bets.columns:
            completed_bets['utr_diff_bin'] = pd.cut(
                completed_bets['utr_diff'], 
                bins=[-5, -2, -1, -0.5, 0, 0.5, 1, 2, 5],
                labels=['< -2', '-2 to -1', '-1 to -0.5', '-0.5 to 0', 
                       '0 to 0.5', '0.5 to 1', '1 to 2', '> 2']
            )
        
        # Summarize profit by UTR difference
        profit_by_diff = completed_bets.groupby('utr_diff_bin')['profit'].sum()
        count_by_diff = completed_bets.groupby('utr_diff_bin').size()
        
        # Plot
        plt.figure(figsize=(12, 6))
        
        # Bar chart
        bars = plt.bar(profit_by_diff.index, profit_by_diff.values)
        
        # Add count annotations
        for i, bar in enumerate(bars):
            count = count_by_diff.iloc[i] if i < len(count_by_diff) else 0
            plt.text(bar.get_x() + bar.get_width()/2, 
                    bar.get_height() + (5 if bar.get_height() >= 0 else -20),
                    f"n={count}", 
                    ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('UTR Difference')
        plt.ylabel('Total Profit ($)')
        plt.title('Profit by UTR Difference')
        
        # Format the y-axis as currency
        plt.gca().yaxis.set_major_formatter('${x:,.0f}')
        
        # Save plot
        base_dir = Path(__file__).parent.parent
        plots_dir = base_dir / "plots"
        plots_dir.mkdir(exist_ok=True)
        plt.savefig(plots_dir / 'profit_by_utr_diff.png')
        
        plt.show()

# Simple test if run directly
if __name__ == "__main__":
    # Create bet tracker
    tracker = TennisBetTracker(initial_bankroll=1000)
    
    # Check if we have any saved bets
    stats = tracker.get_performance_stats()
    
    if stats.get('status') == "No completed bets yet":
        print("No bet history found.")
        
        # Find most recent enhanced odds file with value bets
        model = TennisUTRModel()
        
        base_dir = Path(__file__).parent.parent
        utr_dir = base_dir / "data" / "utr_data"
        odds_files = list(utr_dir.glob("odds_with_utr_*.csv"))
        
        if odds_files:
            # Sort by modification time (most recent first)
            latest_file = sorted(odds_files, key=os.path.getmtime)[-1]
            print(f"Finding value bets in: {latest_file}")
            
            # Find value bets
            value_bets = model.find_value_bets(latest_file)
            
            if not value_bets.empty:
                # Example of recording a bet
                first_bet = value_bets.iloc[0]
                match_data = {
                    'player1': first_bet['player1'],
                    'player2': first_bet['player2'],
                    'player1_utr': first_bet['player1_utr'],
                    'player2_utr': first_bet['player2_utr'],
                    'utr_diff': first_bet['utr_diff'],
                    'utr_win_prob': first_bet['utr_win_prob'],
                    'player1_odds': first_bet['player1_odds'],
                    'player2_odds': first_bet['player2_odds']
                }
                
                # Calculate bet amount using our model
                bet_amount = tracker.calculate_bet_amount(
                    odds=first_bet['bet_odds'],
                    win_prob=first_bet['utr_win_prob'] if first_bet['value_bet_on'] == first_bet['player1'] else 1 - first_bet['utr_win_prob']
                )
                
                print(f"\nExample bet recording (no actual bet is placed):")
                print(f"Would bet ${bet_amount:.2f} on {first_bet['value_bet_on']}")
                
                # Uncomment to actually record a pending bet
                # tracker.record_bet(
                #     match_data=match_data,
                #     bet_on=first_bet['value_bet_on'],
                #     bet_amount=bet_amount
                # )
                
                print("\nTo record actual bets, edit this file and uncomment the tracker.record_bet line")
                print("Then after the match completes, update the result using update_bet_result()")
    else:
        # Show performance stats
        print("\nBetting Performance:")
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"{key}:")
                for sub_key, sub_value in value.items():
                    print(f"  {sub_key}: {sub_value}")
            else:
                print(f"{key}: {value}")
        
        # Plot bankroll history
        tracker.plot_bankroll_history()