#!/usr/bin/env python3
"""
Bet Settlement Script
- Show pending bets
- Interactive settle (win/loss/skip)
- Session summary
"""

import sys
import argparse
from pathlib import Path
import pandas as pd

# Add production modules to path
PROD_ROOT = Path(__file__).parent.resolve()
if str(PROD_ROOT) not in sys.path:
    sys.path.append(str(PROD_ROOT))

from utils.bet_tracker import BetTracker  # type: ignore

def main():
    parser = argparse.ArgumentParser(description='Settle Tennis Bets')
    parser.add_argument('session_id', nargs='?', help='Session ID to settle (optional)')
    parser.add_argument('--bet-id', help='Settle a specific bet by ID')
    parser.add_argument('--show-pending', action='store_true', help='Show all pending bets')
    parser.add_argument('--interactive', action='store_true', help='Interactive settlement mode')
    args = parser.parse_args()

    tracker = BetTracker()

    # Show pending
    if args.show_pending or args.interactive:
        pending = tracker.get_pending_bets()
        if pending.empty:
            print("📊 No pending bets to settle")
            return
        print(f"📋 Pending bets ({len(pending)}):")
        print("-" * 100)
        for _, bet in pending.iterrows():
            print(f"ID: {bet['bet_id']}")
            print(f"   Match: {bet['match']}")
            print(f"   Bet on: {bet['bet_on']} @ {bet['odds_decimal']:.2f}")
            print(f"   Stake: ${bet['stake']:.2f}")
            print(f"   Edge: {bet.get('edge', float('nan')):.1%}")
            print(f"   Model: {bet.get('bet_prob', float('nan')):.1%} | Market: {bet.get('market_prob', float('nan')):.1%}")
            print(f"   Event: {bet.get('event', 'Unknown Event')}")
            print(f"   Date: {bet['timestamp']}")
            print()

    # Interactive settle
    if args.interactive:
        pending = tracker.get_pending_bets()
        if pending.empty:
            return
        print("🎯 Interactive settlement (w=win, l=loss, s=skip, q=quit)")
        print("-" * 60)
        for _, bet in pending.iterrows():
            print(f"\nBet: {bet['match']}")
            print(f"Betting on: {bet['bet_on']} @ {bet['odds_decimal']:.2f}")
            print(f"Stake: ${bet['stake']:.2f}")
            while True:
                choice = input("Result (w/l/s/q): ").lower().strip()
                if choice == 'q':
                    print("Exiting.")
                    return
                if choice == 's':
                    print("Skipped.")
                    break
                if choice in ('w', 'l'):
                    notes = input("Notes (optional): ").strip()
                    _ = tracker.settle_bet(bet['bet_id'], won=(choice == 'w'), notes=notes)
                    break
                print("Invalid choice. Use w/l/s/q.")
        if args.session_id:
            summary = tracker.get_session_summary(args.session_id)
            print("\n📊 Session summary:")
            print(f"   Win rate: {summary.get('win_rate', 0):.1%}")
            print(f"   Total P&L: ${summary.get('total_profit_loss', 0):+.2f}")
            print(f"   ROI: {summary.get('roi', 0):+.1f}%")
        return

    # Settle one bet
    if args.bet_id:
        pending = tracker.get_pending_bets()
        bet = pending[pending['bet_id'] == args.bet_id]
        if bet.empty:
            print(f"❌ Bet {args.bet_id} not found or already settled")
            return
        bet = bet.iloc[0]
        print(f"Settling bet: {bet['match']}")
        print(f"Betting on: {bet['bet_on']} @ {bet['odds_decimal']:.2f}")
        print(f"Stake: ${bet['stake']:.2f}")
        result = input("Did the bet win? (y/n): ").lower().strip()
        notes = input("Notes (optional): ").strip()
        won = result in ('y', 'yes', '1', 'true')
        _ = tracker.settle_bet(args.bet_id, won=won, notes=notes)
        return

    # Session summary only
    if args.session_id:
        summary = tracker.get_session_summary(args.session_id)
        print(f"📊 Session Summary: {args.session_id}")
        print("-" * 50)
        print(f"Total bets: {summary['total_bets']}")
        print(f"Settled bets: {summary['settled_bets']}")
        print(f"Pending bets: {summary['pending_bets']}")
        print(f"Total staked: ${summary['total_staked']:.2f}")
        if summary['settled_bets'] > 0:
            print(f"Total P&L: ${summary['total_profit_loss']:+.2f}")
            print(f"Win rate: {summary['win_rate']:.1%}")
            print(f"ROI: {summary['roi']:+.1f}%")
        if summary['pending_bets'] > 0:
            print("\nUse --interactive to settle pending bets")
        return

    print("Usage:")
    print("  python settle_bets.py --show-pending")
    print("  python settle_bets.py --interactive")
    print("  python settle_bets.py SESSION_ID")
    print("  python settle_bets.py --bet-id BET_ID")

if __name__ == "__main__":
    main()
