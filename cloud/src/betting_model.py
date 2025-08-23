# src/betting_model.py
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import json
import os

class TennisUTRModel:
    def __init__(self, k_parameter=2.2, model_file=None):
        """
        Initialize the UTR-based tennis betting model
        
        Parameters:
        k_parameter: Controls steepness of the logistic function
        model_file: Path to saved model parameters (if None, uses default parameters)
        """
        self.k = k_parameter
        
        # Default win probabilities based on binned UTR differences
        self.default_win_probs = {
            # UTR difference range: win probability
            (-5.0, -2.0): 0.05,   # Large negative difference
            (-2.0, -1.5): 0.10,   # Very significant negative difference
            (-1.5, -1.0): 0.15,   # Significant negative difference
            (-1.0, -0.5): 0.25,   # Moderate negative difference
            (-0.5, -0.2): 0.35,   # Small negative difference
            (-0.2, 0.2): 0.50,    # Essentially even match
            (0.2, 0.5): 0.65,     # Small positive difference
            (0.5, 1.0): 0.75,     # Moderate positive difference
            (1.0, 1.5): 0.85,     # Significant positive difference
            (1.5, 2.0): 0.90,     # Very significant positive difference
            (2.0, 5.0): 0.95      # Large positive difference
        }
        
        # Initialize win probability bins
        self.win_probs = self.default_win_probs.copy()
        
        # Load saved model if provided
        if model_file and os.path.exists(model_file):
            with open(model_file, 'r') as f:
                saved_model = json.load(f)
                self.k = saved_model.get('k', self.k)
                self.win_probs = {tuple(eval(k)): v for k, v in saved_model.get('win_probs', {}).items()}
                print(f"Loaded model parameters from {model_file}")
    
    def calculate_win_probability(self, utr_diff):
        """
        Calculate win probability based on UTR differential
        
        Parameters:
        utr_diff: UTR rating of player A minus UTR rating of player B
        
        Returns:
        Probability of player A winning
        """
        # Method 1: Use logistic function
        logistic_prob = 1 / (1 + np.exp(-self.k * utr_diff))
        
        # Method 2: Use binned probabilities based on collected data
        binned_prob = None
        for (lower, upper), prob in self.win_probs.items():
            if lower <= utr_diff < upper:
                binned_prob = prob
                break
        
        # If no bin found, use the logistic function
        if binned_prob is None:
            return logistic_prob
        
        # Gradually blend logistic function with empirical data as we collect more
        # This weight would increase as we collect more data
        empirical_weight = 0.5  # Start with equal weight
        
        blended_prob = (empirical_weight * binned_prob) + ((1 - empirical_weight) * logistic_prob)
        return blended_prob
    
    def find_value_bets(self, odds_file, edge_threshold=0.05):
        """
        Identify value betting opportunities in the provided odds file
        
        Parameters:
        odds_file: Path to enhanced odds file with UTR ratings
        edge_threshold: Minimum edge required to consider a bet valuable
        
        Returns:
        DataFrame containing only the value bets
        """
        # Load the odds data
        df = pd.read_csv(odds_file)
        
        # Skip matches without UTR data for both players
        df = df.dropna(subset=['player1_utr', 'player2_utr'])
        
        if len(df) == 0:
            print("No matches with complete UTR data found")
            return pd.DataFrame()
        
        print(f"Analyzing {len(df)} matches with complete UTR data")
        
        # Calculate UTR win probability where UTR data is available
        df['utr_diff'] = df['player1_utr'] - df['player2_utr']
        df['utr_win_prob'] = df['utr_diff'].apply(self.calculate_win_probability)
        
        # Calculate implied probabilities from odds
        df['p1_implied_prob'] = df['player1_odds'].apply(
            lambda x: 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)
        )
        df['p2_implied_prob'] = df['player2_odds'].apply(
            lambda x: 100 / (x + 100) if x > 0 else abs(x) / (abs(x) + 100)
        )
        
        # Calculate edge (UTR probability - implied probability)
        df['p1_edge'] = df['utr_win_prob'] - df['p1_implied_prob']
        
        # Calculate expected value for potential bets
        # EV = (probability * payoff) - (1 - probability) for a $1 bet
        df['p1_ev'] = df.apply(
            lambda row: (row['utr_win_prob'] * row['player1_odds'] / 100) - (1 - row['utr_win_prob']) 
            if row['player1_odds'] > 0 else 
            (row['utr_win_prob'] * (100 / abs(row['player1_odds']))) - (1 - row['utr_win_prob']),
            axis=1
        )
        
        df['p2_ev'] = df.apply(
            lambda row: ((1 - row['utr_win_prob']) * row['player2_odds'] / 100) - row['utr_win_prob'] 
            if row['player2_odds'] > 0 else 
            ((1 - row['utr_win_prob']) * (100 / abs(row['player2_odds']))) - row['utr_win_prob'],
            axis=1
        )
        
        # Identify value bets
        df['value_bet_on'] = None
        df['bet_odds'] = None
        df['bet_ev'] = None
        df['edge'] = None
        
        # Player 1 value bets
        mask = df['p1_edge'] > edge_threshold
        df.loc[mask, 'value_bet_on'] = df.loc[mask, 'player1']
        df.loc[mask, 'bet_odds'] = df.loc[mask, 'player1_odds']
        df.loc[mask, 'bet_ev'] = df.loc[mask, 'p1_ev']
        df.loc[mask, 'edge'] = df.loc[mask, 'p1_edge']
        
        # Player 2 value bets
        mask = df['p1_edge'] < -edge_threshold
        df.loc[mask, 'value_bet_on'] = df.loc[mask, 'player2']
        df.loc[mask, 'bet_odds'] = df.loc[mask, 'player2_odds']
        df.loc[mask, 'bet_ev'] = df.loc[mask, 'p2_ev']
        df.loc[mask, 'edge'] = -df.loc[mask, 'p1_edge']  # Make edge positive for display
        
        # Return only value bets
        value_bets = df[pd.notna(df['value_bet_on'])].copy()
        
        # Sort by edge (highest first)
        value_bets = value_bets.sort_values('edge', ascending=False)
        
        if len(value_bets) > 0:
            print(f"Found {len(value_bets)} value betting opportunities:")
            for idx, row in value_bets.iterrows():
                print(f"  {row['value_bet_on']} vs {row['player1'] if row['value_bet_on'] != row['player1'] else row['player2']}")
                print(f"    UTR: {row['player1_utr']:.2f} vs {row['player2_utr']:.2f} (diff: {row['utr_diff']:.2f})")
                print(f"    Win probability: {row['utr_win_prob']:.2f} vs implied {row['p1_implied_prob']:.2f}")
                print(f"    Edge: {row['edge']*100:.1f}%, EV: ${row['bet_ev']:.2f}, Odds: {row['bet_odds']}")
                print()
        else:
            print("No value betting opportunities found")
        
        return value_bets
    
    def update_model_with_results(self, match_results):
        """
        Update model parameters based on match results
        
        Parameters:
        match_results: DataFrame with match results including UTR differences
        
        Returns:
        Updated win probability bins
        """
        # Group matches by UTR difference bins
        bins = list(self.win_probs.keys())
        
        # Create bin labels for easier reading
        bin_labels = [f"{lower:.1f} to {upper:.1f}" for lower, upper in bins]
        
        # Create a new column with binned UTR differences
        match_results['utr_diff_bin'] = pd.cut(match_results['utr_diff'], 
                                             bins=[b[0] for b in bins] + [bins[-1][1]],
                                             labels=bin_labels)
        
        # Calculate win rates for each bin
        win_rates = match_results.groupby('utr_diff_bin')['player1_won'].mean()
        
        # Update win probabilities based on observed data
        for i, (bin_range, win_rate) in enumerate(zip(bins, win_rates)):
            # If we have at least 5 matches in this bin, use the empirical win rate
            bin_count = match_results['utr_diff_bin'].value_counts().get(bin_labels[i], 0)
            
            if bin_count >= 5:
                # Blend with existing probability (80% new data, 20% old estimate)
                self.win_probs[bin_range] = 0.8 * win_rate + 0.2 * self.win_probs[bin_range]
                print(f"Updated win probability for UTR diff {bin_range}: {self.win_probs[bin_range]:.2f} (based on {bin_count} matches)")
        
        return self.win_probs
    
    def save_model(self, filename="utr_model.json"):
        """Save model parameters to file"""
        base_dir = Path(__file__).parent.parent
        models_dir = base_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'k': self.k,
            'win_probs': {str(k): v for k, v in self.win_probs.items()}
        }
        
        with open(models_dir / filename, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {models_dir / filename}")
    
    def plot_probability_curve(self):
        """Generate a plot showing the win probability curve"""
        utr_diffs = np.linspace(-3, 3, 100)
        
        # Logistic function probabilities
        logistic_probs = [1 / (1 + np.exp(-self.k * diff)) for diff in utr_diffs]
        
        # Binned probabilities
        binned_probs = [self.calculate_win_probability(diff) for diff in utr_diffs]
        
        plt.figure(figsize=(12, 8))
        
        # Plot logistic function
        plt.plot(utr_diffs, logistic_probs, 'b-', alpha=0.7, label='Logistic Function')
        
        # Plot binned probabilities
        plt.plot(utr_diffs, binned_probs, 'r-', label='Model Predictions')
        
        # Plot bin midpoints
        for (lower, upper), prob in self.win_probs.items():
            midpoint = (lower + upper) / 2
            plt.plot(midpoint, prob, 'ro', markersize=8)
            plt.text(midpoint, prob + 0.03, f"{prob:.2f}", ha='center')
        
        plt.grid(True, alpha=0.3)
        plt.xlabel('UTR Rating Differential')
        plt.ylabel('Win Probability')
        plt.title('UTR Win Probability Model')
        plt.legend()
        
        plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        
        plt.ylim(0, 1)
        
        # Save the plot
        base_dir = Path(__file__).parent.parent
        plots_dir = base_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plots_dir / 'utr_probability_curve.png')
        
        plt.show()

# Simple test if run directly
if __name__ == "__main__":
    model = TennisUTRModel()
    model.plot_probability_curve()
    
    # Find the most recent enhanced odds file
    base_dir = Path(__file__).parent.parent
    utr_dir = base_dir / "data" / "utr_data"
    odds_files = list(utr_dir.glob("odds_with_utr_*.csv"))
    
    if odds_files:
        # Sort by modification time (most recent first)
        latest_file = sorted(odds_files, key=os.path.getmtime)[-1]
        print(f"Analyzing latest enhanced odds file: {latest_file}")
        
        # Find value bets
        value_bets = model.find_value_bets(latest_file)
    else:
        print("No enhanced odds files found. Run fetch_data.py first.")