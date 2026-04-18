#!/usr/bin/env python3
"""
Comprehensive Bet Tracking and Results Settlement System
Tracks all bets, settles results, and maintains bankroll history
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json

from logging_utils import ensure_csv_columns, normalize_name


BETS_COLUMNS = [
    'bet_id', 'session_id', 'timestamp', 'event', 'match', 'match_uid', 'feature_snapshot_id',
    'run_id', 'bet_on', 'bet_on_player1', 'odds_decimal', 'stake', 'stake_fraction',
    'model_prob', 'market_prob', 'edge', 'kelly_fraction', 'potential_profit',
    'potential_loss', 'bankroll_before', 'model_version', 'status', 'outcome',
    'actual_profit', 'bankroll_after', 'settled_timestamp', 'match_date',
    'match_start_time', 'notes'
]

BANKROLL_COLUMNS = [
    'timestamp', 'session_id', 'bankroll', 'change_amount', 'change_reason',
    'total_staked', 'num_pending_bets', 'num_settled_bets'
]

SESSION_COLUMNS = [
    'session_id', 'start_time', 'end_time', 'initial_bankroll', 'final_bankroll',
    'total_bets_placed', 'total_staked', 'total_profit_loss', 'win_rate',
    'avg_odds', 'avg_edge', 'kelly_multiplier_used', 'notes'
]

class BetTracker:
    """Track bets, settle results, and maintain bankroll history"""
    
    def __init__(self, logs_dir: str = "./logs"):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths
        self.bets_file = self.logs_dir / "all_bets.csv"
        self.bankroll_file = self.logs_dir / "bankroll_history.csv"
        self.session_file = self.logs_dir / "betting_sessions.csv"
        
        # Initialize files if they don't exist
        self._initialize_files()
    
    def _initialize_files(self):
        """Initialize tracking files with headers"""
        ensure_csv_columns(self.bets_file, BETS_COLUMNS).to_csv(self.bets_file, index=False)
        ensure_csv_columns(self.bankroll_file, BANKROLL_COLUMNS).to_csv(self.bankroll_file, index=False)
        ensure_csv_columns(self.session_file, SESSION_COLUMNS).to_csv(self.session_file, index=False)
    
    def start_session(self, initial_bankroll: float, kelly_multiplier: float, notes: str = "") -> str:
        """Start a new betting session"""
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Add to sessions file
        session_data = {
            'session_id': session_id,
            'start_time': timestamp,
            'end_time': None,
            'initial_bankroll': initial_bankroll,
            'final_bankroll': None,
            'total_bets_placed': 0,
            'total_staked': 0.0,
            'total_profit_loss': 0.0,
            'win_rate': None,
            'avg_odds': None,
            'avg_edge': None,
            'kelly_multiplier_used': kelly_multiplier,
            'notes': notes
        }
        
        # Append to sessions file
        sessions_df = pd.read_csv(self.session_file)
        sessions_df = pd.concat([sessions_df, pd.DataFrame([session_data])], ignore_index=True)
        sessions_df.to_csv(self.session_file, index=False)
        
        # Log bankroll at session start
        self.log_bankroll_change(session_id, initial_bankroll, 0.0, "Session started")
        
        print(f"🎯 Started betting session: {session_id}")
        print(f"   Initial bankroll: ${initial_bankroll:.2f}")
        print(f"   Kelly multiplier: {kelly_multiplier:.1%}")
        
        return session_id
    
    def log_bets(self, bet_slips_df: pd.DataFrame, session_id: str, current_bankroll: float):
        """Log new bets from bet slips"""
        if bet_slips_df.empty:
            return
        
        # Prepare bet records
        bet_records = []
        for _, bet in bet_slips_df.iterrows():
            bet_record = {
                'bet_id': bet.get('bet_id', f"bet_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{len(bet_records)}"),
                'session_id': session_id,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'event': bet['event'],
                'match': bet['match'],
                'match_uid': bet.get('match_uid'),
                'feature_snapshot_id': bet.get('feature_snapshot_id'),
                'run_id': bet.get('run_id'),
                'bet_on': bet['bet_on'],
                'bet_on_player1': bet['bet_on_player1'],
                'odds_decimal': bet['odds_decimal'],
                'stake': bet['stake'],
                'stake_fraction': bet['stake_fraction'],
                'model_prob': bet['model_prob'],
                'market_prob': bet['market_prob'],
                'edge': bet['edge'],
                'kelly_fraction': bet['kelly_fraction'],
                'potential_profit': bet['potential_profit'],
                'potential_loss': bet['potential_loss'],
                'bankroll_before': bet.get('bankroll', current_bankroll),
                'model_version': bet.get('model_version', 'NN-143'),
                'status': 'pending',
                'outcome': None,
                'actual_profit': None,
                'bankroll_after': None,
                'settled_timestamp': None,
                'match_date': bet.get('match_date', ''),
                'match_start_time': bet.get('match_start_time', ''),
                'notes': ''
            }
            bet_records.append(bet_record)
        
        # Append to all bets file
        all_bets_df = ensure_csv_columns(self.bets_file, BETS_COLUMNS)
        new_bets_df = pd.DataFrame(bet_records)
        all_bets_df = pd.concat([all_bets_df, new_bets_df], ignore_index=True)
        all_bets_df.to_csv(self.bets_file, index=False)
        
        # Update bankroll with total staked
        total_staked = sum(bet['stake'] for bet in bet_records)
        new_bankroll = current_bankroll  # Bankroll doesn't decrease until bet settles
        self.log_bankroll_change(session_id, new_bankroll, -total_staked, 
                                f"Placed {len(bet_records)} bets")
        self._refresh_session_record(session_id)
        
        print(f"📝 Logged {len(bet_records)} new bets")
        print(f"   Total stakes: ${total_staked:.2f}")
    
    def settle_bet(self, bet_id: str, won: bool, notes: str = "") -> float:
        """Settle a specific bet and return profit/loss"""
        all_bets_df = pd.read_csv(self.bets_file)
        for col in ['status', 'outcome', 'settled_timestamp', 'notes']:
            if col in all_bets_df.columns:
                all_bets_df[col] = all_bets_df[col].astype(object)
        
        # Find the bet
        bet_idx = all_bets_df[all_bets_df['bet_id'] == bet_id].index
        if len(bet_idx) == 0:
            print(f"⚠️  Bet {bet_id} not found")
            return 0.0
        
        idx = bet_idx[0]
        bet = all_bets_df.loc[idx]
        
        if bet['status'] != 'pending':
            print(f"⚠️  Bet {bet_id} already settled")
            return 0.0
        
        # Calculate actual profit/loss
        if won:
            actual_profit = bet['stake'] * (bet['odds_decimal'] - 1.0)
            outcome = 'win'
        else:
            actual_profit = -bet['stake']
            outcome = 'loss'
        
        # Update bet record
        all_bets_df.loc[idx, 'status'] = 'settled'
        all_bets_df.loc[idx, 'outcome'] = outcome
        all_bets_df.loc[idx, 'actual_profit'] = actual_profit
        all_bets_df.loc[idx, 'settled_timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        all_bets_df.loc[idx, 'notes'] = notes
        
        # Calculate new bankroll
        current_bankroll = self.get_current_bankroll()
        new_bankroll = current_bankroll + actual_profit
        all_bets_df.loc[idx, 'bankroll_after'] = new_bankroll
        
        # Save updated bets
        all_bets_df.to_csv(self.bets_file, index=False)
        
        # Log bankroll change
        session_id = bet['session_id']
        result_desc = f"Settled bet: {bet['match']} ({'WIN' if won else 'LOSS'})"
        self.log_bankroll_change(session_id, new_bankroll, actual_profit, result_desc)
        self._refresh_session_record(session_id)
        
        print(f"✅ Settled bet {bet_id}: {outcome.upper()}")
        print(f"   Profit/Loss: ${actual_profit:+.2f}")
        print(f"   New bankroll: ${new_bankroll:.2f}")
        
        return actual_profit
    
    def settle_bets_batch(self, results: List[Dict]) -> float:
        """
        Settle multiple bets at once
        results: [{'bet_id': 'bet_123', 'won': True, 'notes': '...'}]
        """
        total_profit = 0.0
        for result in results:
            profit = self.settle_bet(result['bet_id'], result['won'], result.get('notes', ''))
            total_profit += profit
        
        return total_profit
    
    def log_bankroll_change(self, session_id: str, new_bankroll: float, change_amount: float, reason: str):
        """Log a bankroll change"""
        bankroll_df = pd.read_csv(self.bankroll_file)
        
        # Count pending and settled bets
        all_bets_df = pd.read_csv(self.bets_file)
        session_bets = all_bets_df[all_bets_df['session_id'] == session_id]
        num_pending = len(session_bets[session_bets['status'] == 'pending'])
        num_settled = len(session_bets[session_bets['status'] == 'settled'])
        total_staked = session_bets['stake'].sum()
        
        bankroll_record = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'session_id': session_id,
            'bankroll': new_bankroll,
            'change_amount': change_amount,
            'change_reason': reason,
            'total_staked': total_staked,
            'num_pending_bets': num_pending,
            'num_settled_bets': num_settled
        }
        
        bankroll_df = pd.concat([bankroll_df, pd.DataFrame([bankroll_record])], ignore_index=True)
        bankroll_df.to_csv(self.bankroll_file, index=False)
    
    def get_current_bankroll(self) -> float:
        """Get the current bankroll amount"""
        bankroll_df = pd.read_csv(self.bankroll_file)
        if bankroll_df.empty:
            return 1000.0  # Default
        return float(bankroll_df.iloc[-1]['bankroll'])
    
    def get_pending_bets(self) -> pd.DataFrame:
        """Get all pending bets"""
        all_bets_df = pd.read_csv(self.bets_file)
        return all_bets_df[all_bets_df['status'] == 'pending']
    
    def get_session_summary(self, session_id: str) -> Dict:
        """Get summary statistics for a session"""
        all_bets_df = pd.read_csv(self.bets_file)
        session_bets = all_bets_df[all_bets_df['session_id'] == session_id]
        
        if session_bets.empty:
            return {'error': 'No bets found for session'}
        
        settled_bets = session_bets[session_bets['status'] == 'settled']
        pending_bets = session_bets[session_bets['status'] == 'pending']
        
        summary = {
            'session_id': session_id,
            'total_bets': len(session_bets),
            'pending_bets': len(pending_bets),
            'settled_bets': len(settled_bets),
            'total_staked': session_bets['stake'].sum(),
            'total_profit_loss': settled_bets['actual_profit'].sum() if not settled_bets.empty else 0.0,
            'win_rate': (settled_bets['outcome'] == 'win').mean() if not settled_bets.empty else None,
            'avg_odds': session_bets['odds_decimal'].mean(),
            'avg_edge': session_bets['edge'].mean(),
            'roi': None
        }
        
        # Calculate ROI
        if summary['total_staked'] > 0 and not settled_bets.empty:
            summary['roi'] = (summary['total_profit_loss'] / summary['total_staked']) * 100
        
        return summary

    def _refresh_session_record(self, session_id: str):
        """Synchronize cached session summary columns with the underlying bets file."""
        summary = self.get_session_summary(session_id)
        if 'error' in summary:
            return

        sessions_df = pd.read_csv(self.session_file)
        if 'end_time' in sessions_df.columns:
            sessions_df['end_time'] = sessions_df['end_time'].astype(object)
        session_idx = sessions_df[sessions_df['session_id'] == session_id].index
        if len(session_idx) == 0:
            return

        idx = session_idx[0]
        sessions_df.loc[idx, 'total_bets_placed'] = summary['total_bets']
        sessions_df.loc[idx, 'total_staked'] = summary['total_staked']
        sessions_df.loc[idx, 'total_profit_loss'] = summary['total_profit_loss']
        sessions_df.loc[idx, 'win_rate'] = summary['win_rate']
        sessions_df.loc[idx, 'avg_odds'] = summary['avg_odds']
        sessions_df.loc[idx, 'avg_edge'] = summary['avg_edge']

        if summary['pending_bets'] == 0 and summary['settled_bets'] > 0:
            sessions_df.loc[idx, 'end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sessions_df.loc[idx, 'final_bankroll'] = self.get_current_bankroll()

        sessions_df.to_csv(self.session_file, index=False)

    def settle_pending_bets_for_match(
        self,
        match_uid: str = None,
        p1: str = None,
        p2: str = None,
        actual_winner: int = None,
        notes: str = "",
    ) -> int:
        """Auto-settle any pending bets linked to a finished match."""
        if actual_winner not in (1, 2):
            return 0

        pending = self.get_pending_bets()
        if pending.empty:
            return 0

        candidates = pd.DataFrame()
        if match_uid and 'match_uid' in pending.columns:
            candidates = pending[pending['match_uid'].fillna('') == match_uid]

        if candidates.empty and p1 and p2:
            p1_norm = normalize_name(p1)
            p2_norm = normalize_name(p2)

            def _match_pair(match_text: str) -> bool:
                parts = [normalize_name(part) for part in str(match_text).split(' vs ')]
                if len(parts) != 2:
                    return False
                return parts == [p1_norm, p2_norm] or parts == [p2_norm, p1_norm]

            candidates = pending[pending['match'].apply(_match_pair)]

        settled = 0
        auto_note = notes or "Auto-settled from Tennis Abstract"
        for _, bet in candidates.iterrows():
            won = bool((actual_winner == 1 and bool(bet['bet_on_player1'])) or (actual_winner == 2 and not bool(bet['bet_on_player1'])))
            self.settle_bet(bet['bet_id'], won=won, notes=auto_note)
            settled += 1

        return settled
    
    def end_session(self, session_id: str) -> Dict:
        """End a betting session and update summary"""
        self._refresh_session_record(session_id)
        summary = self.get_session_summary(session_id)
        
        # Update sessions file
        sessions_df = pd.read_csv(self.session_file)
        session_idx = sessions_df[sessions_df['session_id'] == session_id].index
        
        if len(session_idx) > 0:
            idx = session_idx[0]
            sessions_df.loc[idx, 'end_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            sessions_df.loc[idx, 'final_bankroll'] = self.get_current_bankroll()
            sessions_df.loc[idx, 'total_bets_placed'] = summary['total_bets']
            sessions_df.loc[idx, 'total_staked'] = summary['total_staked']
            sessions_df.loc[idx, 'total_profit_loss'] = summary['total_profit_loss']
            sessions_df.loc[idx, 'win_rate'] = summary['win_rate']
            sessions_df.loc[idx, 'avg_odds'] = summary['avg_odds']
            sessions_df.loc[idx, 'avg_edge'] = summary['avg_edge']
            sessions_df.to_csv(self.session_file, index=False)
        
        print(f"🏁 Ended session {session_id}")
        print(f"   Final P&L: ${summary['total_profit_loss']:+.2f}")
        print(f"   Win rate: {summary['win_rate']:.1%}" if summary['win_rate'] is not None else "   Win rate: N/A")
        print(f"   ROI: {summary['roi']:+.1f}%" if summary['roi'] is not None else "   ROI: N/A")
        
        return summary

def main():
    """Test the bet tracker"""
    tracker = BetTracker()
    
    # Start a test session
    session_id = tracker.start_session(1000.0, 0.18, "Test session")
    
    # Create sample bet slips
    sample_bets = pd.DataFrame([
        {
            'event': 'ATP Test Tournament',
            'match': 'Player A vs Player B',
            'bet_on': 'Player A',
            'bet_on_player1': True,
            'odds_decimal': 2.0,
            'stake': 50.0,
            'stake_fraction': 0.05,
            'model_prob': 0.55,
            'market_prob': 0.5,
            'edge': 0.05,
            'kelly_fraction': 0.1,
            'potential_profit': 50.0,
            'potential_loss': 50.0,
            'bankroll': 1000.0,
            'model_version': 'NN-143'
        }
    ])
    
    # Log bets
    tracker.log_bets(sample_bets, session_id, 1000.0)
    
    # Get pending bets
    pending = tracker.get_pending_bets()
    print(f"\n📋 Pending bets: {len(pending)}")
    
    # Simulate settling a bet (win)
    if not pending.empty:
        bet_id = pending.iloc[0]['bet_id']
        tracker.settle_bet(bet_id, won=True, notes="Simulated win")
    
    # Get session summary
    summary = tracker.get_session_summary(session_id)
    print(f"\n📊 Session summary: {summary}")
    
    # End session
    tracker.end_session(session_id)

if __name__ == "__main__":
    main()
