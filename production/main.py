#!/usr/bin/env python3
"""
Live Tennis Betting System - Main Orchestrator
Coordinates odds fetching, feature extraction, model inference, and stake calculation
"""

import argparse
import sys
import re
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import os

# Add production modules to path
sys.path.append(str(Path(__file__).parent))

from odds.fetch_bovada import fetch_bovada_tennis_odds, save_odds_data
from features.ta_feature_calculator import TAFeatureCalculator
from models.inference import TennisPredictor, calculate_betting_edges, EXACT_141_FEATURES
from utils.stake_calculator import KellyStakeCalculator
from utils.bet_tracker import BetTracker
from tournaments.resolve_tournament import TournamentResolver, level_hint_from_title
from prediction_logger import log_prediction
from scraping.atp_rankings_scraper import fetch_atp_rankings, save_rankings

class LiveBettingOrchestrator:
    """Main orchestrator for live tennis betting system"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.predictor = TennisPredictor()
        self.calculator = KellyStakeCalculator(
            kelly_multiplier=self.config.get('kelly_multiplier', 0.18),
            edge_threshold=self.config.get('edge_threshold', 0.02),
            max_stake_fraction=self.config.get('max_stake_fraction', 0.05),
            min_stake_dollars=self.config.get('min_stake_dollars', 1.0)
        )
        self.logs_dir = Path(self.config.get('logs_dir', './logs'))
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.bet_tracker = BetTracker(str(self.logs_dir))
        self.feature_engine = TAFeatureCalculator()

        # Tournament resolver for surface/level/draw/round
        tournaments_map_path = Path(self.config.get('data_dir', '../data')) / 'tournaments_map.csv'
        self.tournament_resolver = TournamentResolver(str(tournaments_map_path)) if tournaments_map_path.exists() else None

    @staticmethod
    def parse_match_date(match_time_str: str) -> datetime:
        """
        Parse Bovada match_time string to a datetime for feature computation.
        Examples: "Today 7:30 PM", "Tomorrow 3:00 AM", "3/16/26 2:00 PM"
        Falls back to today if unparseable.
        """
        if not match_time_str or match_time_str == "Unknown":
            return datetime.now()

        t = match_time_str.strip()
        now = datetime.now()

        # Absolute date: "3/16/26 2:00 PM" or "03/16/2026 2:00 PM"
        import re as _re
        m = _re.search(r"(\d{1,2})/(\d{1,2})/(\d{2,4})", t)
        if m:
            mo, dy, yr = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if yr < 100:
                yr += 2000
            try:
                return datetime(yr, mo, dy, now.hour, now.minute)
            except ValueError:
                return now

        # Relative: "Today ..." or "Tomorrow ..."
        if t.lower().startswith("today"):
            return now
        if t.lower().startswith("tomorrow"):
            from datetime import timedelta
            return now + timedelta(days=1)

        # Day-of-week: "Sat 7:30 PM" — default to today
        return now

    @staticmethod
    def parse_round_from_text(text: str) -> str:
        """
        Extract round code from Bovada event text.
        Returns standard codes: F, SF, QF, R16, R32, R64, R128, Q1-Q4, RR, or None
        """
        if not text:
            return None
        t = text.lower()

        # Finals (but not semifinals)
        if " final" in t and "semi" not in t:
            return "F"

        # Semifinals
        if "semifinal" in t or "semi-final" in t or "semifinals" in t:
            return "SF"

        # Quarterfinals
        if "quarterfinal" in t or "quarter-final" in t or "quarterfinals" in t:
            return "QF"

        # Round of N
        m = re.search(r"round\s+of\s+(128|64|32|16)", t)
        if m:
            return f"R{m.group(1)}"

        # Qualifying rounds
        if re.search(r"\bq(?:ualifying)?\s*1\b", t):
            return "Q1"
        if re.search(r"\bq(?:ualifying)?\s*2\b", t):
            return "Q2"
        if re.search(r"\bq(?:ualifying)?\s*3\b", t):
            return "Q3"
        if re.search(r"\bq(?:ualifying)?\s*4\b", t):
            return "Q4"

        # Round Robin / Group
        if "round robin" in t or "group" in t:
            return "RR"

        return None
        
    def _load_config(self, config_path: str) -> dict:
        """Load configuration from file or use defaults"""
        default_config = {
            'kelly_multiplier': 0.18,
            'edge_threshold': 0.02,
            'max_stake_fraction': 0.05,
            'min_stake_dollars': 1.0,
            'bankroll': 1000.0,
            'logs_dir': './logs',
            'models_dir': '../results/professional_tennis/Neural_Network'
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def fetch_odds(self) -> pd.DataFrame:
        """Fetch current odds from Bovada"""
        print("🎾 Fetching odds from Bovada...")
        odds_df = fetch_bovada_tennis_odds(headless=True)
        
        if not odds_df.empty:
            # Save odds data
            odds_file = save_odds_data(odds_df, self.logs_dir / "odds")
            print(f"✅ Fetched {len(odds_df)} ATP/Challenger matches")
        else:
            print("❌ No matches found")
            
        return odds_df
    
    def extract_features(self, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Extract 141 features for all matches via Tennis Abstract scraper"""
        if odds_df.empty:
            return pd.DataFrame()

        print("🔧 Extracting features from Tennis Abstract...")

        # Shared session cache: avoids duplicate TA requests within a single run
        session_cache = {}

        feature_rows = []
        for idx, row in odds_df.iterrows():
            p1 = row.get('player1_normalized') or row.get('player1_raw', '')
            p2 = row.get('player2_normalized') or row.get('player2_raw', '')
            print(f"   🎯 Processing {p1} vs {p2}")

            # Resolve tournament metadata
            surface, tournament_level, draw_size = "Hard", "A", 32
            round_code = None
            resolver_source = "default"

            if self.tournament_resolver:
                tournament_meta = self.tournament_resolver.resolve_soft(row['event'])
                if tournament_meta:
                    meta, score = tournament_meta
                    surface = meta.surface
                    tournament_level = meta.level
                    draw_size = meta.draw_size
                    round_code = meta.round_code
                    resolver_source = "resolved"
                    print(f"      📍 {surface}, Level:{tournament_level}, Draw:{draw_size} (score:{score:.3f})")
                else:
                    tournament_level = level_hint_from_title(row['event']) or "A"
                    resolver_source = "level_hint"
                    print(f"      ⚠️  Inferred level: {tournament_level}")
            else:
                tournament_level = level_hint_from_title(row['event']) or "A"

            if not round_code:
                round_code = self.parse_round_from_text(row.get('event', ''))

            # Use exact match date from Bovada where available
            match_time_str = row.get('match_time', '')
            match_date = self.parse_match_date(match_time_str)
            # Flag whether Bovada gave us a real absolute date (e.g. "3/22/26 8:00 AM")
            # vs. just a time or no date (defaults to today)
            import re as _re2
            match_date_is_explicit = bool(_re2.search(r'\d{1,2}/\d{1,2}/\d{2,4}', match_time_str))

            has_defaulted = False
            try:
                features = self.feature_engine.build_141_features(
                    player1_name=p1,
                    player2_name=p2,
                    match_date=match_date,
                    surface=surface,
                    tournament_level=tournament_level,
                    draw_size=draw_size,
                    round_code=round_code,
                    force_refresh=True,
                    persist=False,
                    session_cache=session_cache,
                    match_date_is_explicit=match_date_is_explicit,
                )
                # features_complete=False if ANY meaningful feature was defaulted
                # (includes ATP points fallback, round=None, structural defaults)
                has_defaulted = bool(features.get('_defaulted_features', ''))
                status = "ok"
            except Exception as e:
                print(f"      ⚠️ Feature extraction failed: {e}")
                features = {}
                status = "skip"

            # Ordered 141 features for the model — flag any that are absent from the returned dict
            missing_from_dict = [k for k in EXACT_141_FEATURES if k not in features]
            if missing_from_dict and status == "ok":
                print(f"      ⚠️  MISSING FROM FEATURE DICT (filled 0): {missing_from_dict}")
            ordered_features = {k: features.get(k, 0.0) for k in EXACT_141_FEATURES}

            # Metadata (not fed to model)
            ordered_features.update({
                '_has_defaulted_features': has_defaulted,
                'match_id': idx,
                'player1_raw': row.get('player1_raw', p1),
                'player2_raw': row.get('player2_raw', p2),
                'event': row.get('event', ''),
                'timestamp': row.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'match_time': row.get('match_time', ''),
                'status': status,
                'meta_level_input': tournament_level,
                'meta_surface_input': surface,
                'meta_round_input': features.get('_resolved_round_code') or round_code or '',
                'meta_match_date': features.get('_resolved_match_date') or '',
                'meta_defaulted_features': features.get('_defaulted_features') or '',
                'meta_draw_input': draw_size,
                'meta_resolver_source': resolver_source
            })

            feature_rows.append(ordered_features)

        features_df = pd.DataFrame(feature_rows)

        if not features_df.empty:
            features_file = self.logs_dir / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            features_df.to_csv(features_file, index=False)
            ok_count = (features_df['status'] == 'ok').sum()
            skip_count = (features_df['status'] == 'skip').sum()
            print(f"✅ Feature extraction: {ok_count} ok, {skip_count} skipped")

        return features_df
    
    def generate_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate model predictions"""
        if features_df.empty:
            return pd.DataFrame()
            
        print("🎯 Generating predictions...")
        predictions_df = self.predictor.predict_slate(features_df)
        
        if not predictions_df.empty:
            print(f"✅ Generated predictions for {len(predictions_df)} matches")
        
        return predictions_df
    
    def calculate_edges_and_stakes(self, predictions_df: pd.DataFrame, odds_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate betting edges and stakes"""
        if predictions_df.empty or odds_df.empty:
            return pd.DataFrame()
            
        print("💰 Calculating edges and stakes...")
        
        # Filter to only successful predictions before calculating edges
        successful_predictions = predictions_df[predictions_df.get('prediction_status') == 'success'].copy()
        
        if successful_predictions.empty:
            print("📊 No successful predictions to calculate edges for")
            return pd.DataFrame()
        
        print(f"   📈 Using {len(successful_predictions)} successful predictions out of {len(predictions_df)} total")
        
        # Calculate edges
        edges_df = calculate_betting_edges(successful_predictions, odds_df)
        
        # Filter betting opportunities
        opportunities_df = self.calculator.filter_betting_opportunities(edges_df)
        
        if opportunities_df.empty:
            print("📊 No profitable betting opportunities found")
            return pd.DataFrame()
        
        # Calculate stakes
        bankroll = self.config.get('bankroll', 1000.0)
        stakes_df = self.calculator.allocate_block_stakes(opportunities_df, bankroll)
        
        # Generate bet slips
        bet_slips_df = self.calculator.generate_bet_slips(stakes_df)
        
        return bet_slips_df
    
    def save_bet_slips(self, bet_slips_df: pd.DataFrame) -> str:
        """Save bet slips to file"""
        if bet_slips_df.empty:
            return ""
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        bet_slips_file = self.logs_dir / f"bet_slips_{timestamp}.csv"
        bet_slips_df.to_csv(bet_slips_file, index=False)
        
        # Also save as latest for easy access
        latest_file = self.logs_dir / "bet_slips_latest.csv"
        bet_slips_df.to_csv(latest_file, index=False)
        
        print(f"💾 Bet slips saved to: {bet_slips_file}")
        return str(bet_slips_file)
    
    def print_summary(self, bet_slips_df: pd.DataFrame):
        """Print betting summary"""
        if bet_slips_df.empty:
            print("\n📊 No bets to place today")
            return
            
        print(f"\n📋 BETTING SUMMARY - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)
        
        total_stake = bet_slips_df['stake'].sum()
        total_potential_profit = bet_slips_df['potential_profit'].sum()
        bankroll = self.config.get('bankroll', 1000.0)
        
        print(f"Bankroll: ${bankroll:.2f}")
        print(f"Total stakes: ${total_stake:.2f} ({total_stake/bankroll:.1%} of bankroll)")
        print(f"Number of bets: {len(bet_slips_df)}")
        print(f"Potential profit: ${total_potential_profit:.2f}")
        print(f"Average edge: {bet_slips_df['edge'].mean():.1%}")
        
        print("\nBET DETAILS:")
        print("-" * 80)
        
        for _, bet in bet_slips_df.iterrows():
            print(f"{bet['match']}")
            print(f"  Bet on: {bet['bet_on']} @ {bet['odds_decimal']:.2f}")
            # Show both dollar amount and percentage
            stake_pct = bet.get('stake_percentage', bet.get('stake_fraction', 0) * 100)
            print(f"  Stake: ${bet['stake']:.2f} ({stake_pct:.1f}% of bankroll)")
            # Use model_prob (renamed from bet_prob in stake calculator)
            model_prob = bet.get('model_prob', bet.get('bet_prob', 0.5))
            print(f"  Edge: {bet['edge']:.1%} | Model: {model_prob:.1%} | Market: {bet['market_prob']:.1%}")
            print(f"  Potential: ${bet['potential_profit']:.2f} | Event: {bet['event']}")
            print()
    
    def _log_all_predictions(self, predictions_df, odds_df, features_df):
        """Log every prediction to prediction_log.csv for later accuracy tracking."""
        try:
            today = datetime.now().date().isoformat()

            for _, pred_row in predictions_df.iterrows():
                p1 = pred_row.get('player1_normalized') or pred_row.get('player1_raw', '')
                p2 = pred_row.get('player2_normalized') or pred_row.get('player2_raw', '')
                model_p1 = pred_row.get('player1_win_prob') or pred_row.get('p1_win_prob')
                if model_p1 is None:
                    continue
                model_p2 = 1.0 - float(model_p1)

                # features_complete: False if the calculator reported any noisy defaults
                features_complete = not bool(pred_row.get('_has_defaulted_features', False))
                if not features_complete:
                    defaulted = pred_row.get('meta_defaulted_features', '')
                    print(f"  ⛔ Skipping prediction log for {p1} vs {p2} — incomplete features: {defaulted}")
                    continue

                # Try to get market odds from odds_df — match BOTH players to avoid
                # picking up futures/outright lines (e.g. Alcaraz vs The Field)
                # instead of the correct H2H matchup.
                p1_lower = str(p1).lower()
                p2_lower = str(p2).lower()
                match_odds = odds_df[
                    (
                        (odds_df['player1_normalized'].str.lower() == p1_lower) &
                        (odds_df['player2_normalized'].str.lower() == p2_lower)
                    ) | (
                        (odds_df['player1_normalized'].str.lower() == p2_lower) &
                        (odds_df['player2_normalized'].str.lower() == p1_lower)
                    )
                ]
                if not match_odds.empty:
                    o_row = match_odds.iloc[0]
                    mkt_p1_raw = float(o_row.get('player1_implied_prob', 0.5))
                    mkt_p2_raw = float(o_row.get('player2_implied_prob', 0.5))
                    # De-vig: normalize so probs sum to 1.0
                    mkt_total = mkt_p1_raw + mkt_p2_raw
                    mkt_p1 = mkt_p1_raw / mkt_total if mkt_total > 0 else 0.5
                    mkt_p2 = mkt_p2_raw / mkt_total if mkt_total > 0 else 0.5
                    o1 = o_row.get('player1_odds_american')
                    o2 = o_row.get('player2_odds_american')
                    tournament = o_row.get('tourney_name', '')
                    surface = pred_row.get('meta_surface_input', o_row.get('surface', 'Hard'))
                    level = pred_row.get('meta_level_input', o_row.get('tourney_level', ''))
                    match_time = o_row.get('match_time', '')
                else:
                    mkt_p1, mkt_p2, o1, o2 = 0.5, 0.5, None, None
                    tournament = ''
                    surface = pred_row.get('meta_surface_input', 'Hard')
                    level = pred_row.get('meta_level_input', '')
                    match_time = ''

                # Use TA-inferred match date (tourney_start + round_offset) — Bovada only gives clock time
                match_date = pred_row.get('meta_match_date') or today
                model_version = pred_row.get('model_version', 'NN-SURFACE_FIX')
                log_prediction(
                    p1=p1, p2=p2,
                    tournament=tournament, surface=surface, level=level,
                    round_code=pred_row.get('meta_round_input', '') or None,
                    match_date=match_date,
                    model_p1_prob=float(model_p1), model_p2_prob=model_p2,
                    market_p1_prob=mkt_p1, market_p2_prob=mkt_p2,
                    p1_odds_american=o1, p2_odds_american=o2,
                    model_version=model_version,
                    features_complete=features_complete,
                    defaulted_features=pred_row.get('meta_defaulted_features', ''),
                )
        except Exception as e:
            print(f"  ⚠️ Prediction logging failed (non-fatal): {e}")
            import traceback; traceback.print_exc()

    def _refresh_atp_rankings(self):
        """Fetch and cache current ATP rankings (rank + points) from atptour.com."""
        print("📊 Refreshing ATP rankings...")
        try:
            df = fetch_atp_rankings(headless=True)
            if df.empty:
                print("  ⚠️  ATP rankings scrape returned no data — Rank_Points will default to 500")
                return
            save_rankings(df)
            # Reload into feature engine so it uses the freshly scraped data
            self.feature_engine._atp_rankings = df
            print(f"  ✅ Loaded {len(df)} ranked players")
        except Exception as e:
            print(f"  ⚠️  ATP rankings refresh failed (non-fatal): {e}")
            print("       Rank_Points will default to 500 for this run")

    def run_full_pipeline(self, start_session: bool = True) -> bool:
        """Run the complete betting pipeline with bet tracking"""
        session_id = None
        try:
            print("🚀 Starting live tennis betting pipeline...")
            kelly_mult = self.config.get('kelly_multiplier', 0.18)
            bankroll = self.config.get('bankroll', 1000.0)
            
            print(f"⚙️  Kelly multiplier: {kelly_mult:.1%}")
            print(f"⚙️  Edge threshold: {self.config.get('edge_threshold', 0.02):.1%}")
            print(f"⚙️  Bankroll: ${bankroll:.2f}")
            
            # Start betting session
            if start_session:
                session_id = self.bet_tracker.start_session(
                    bankroll, kelly_mult, f"Auto session {datetime.now().strftime('%Y-%m-%d %H:%M')}"
                )
            
            # Step 0a: Auto-settle any pending predictions with known results
            try:
                from auto_settle import run as auto_settle_run
                print("\n📋 Auto-settling pending predictions...")
                auto_settle_run(dry_run=False)
            except Exception as e:
                print(f"  ⚠️  Auto-settle failed (non-fatal): {e}")

            # Step 0b: Refresh ATP rankings (rank + points)
            self._refresh_atp_rankings()

            # Step 1: Fetch odds
            odds_df = self.fetch_odds()
            if odds_df.empty:
                print("⚠️  No odds available, stopping pipeline")
                return False
            
            # Step 2: Extract features
            features_df = self.extract_features(odds_df)
            if features_df.empty:
                print("⚠️  Feature extraction failed, stopping pipeline")
                return False
            
            # Step 3: Generate predictions
            predictions_df = self.generate_predictions(features_df)
            if predictions_df.empty:
                print("⚠️  Prediction generation failed, stopping pipeline")
                return False
            
            # Step 4: Calculate stakes
            bet_slips_df = self.calculate_edges_and_stakes(predictions_df, odds_df)

            # Step 4b: Log all predictions (regardless of edge threshold)
            self._log_all_predictions(predictions_df, odds_df, features_df)

            # Step 5: Save, log, and display results
            if not bet_slips_df.empty:
                self.save_bet_slips(bet_slips_df)
                
                # Log bets to tracking system
                if session_id:
                    self.bet_tracker.log_bets(bet_slips_df, session_id, bankroll)
                
                self.print_summary(bet_slips_df)
                
                # Show tracking info
                if session_id:
                    pending_bets = self.bet_tracker.get_pending_bets()
                    print(f"\n📊 Session tracking:")
                    print(f"   Session ID: {session_id}")
                    print(f"   Total pending bets: {len(pending_bets)}")
                    print(f"   Use 'python settle_bets.py {session_id}' to settle results later")
                
                return True
            else:
                print("📊 No profitable betting opportunities found")
                return False
                
        except Exception as e:
            print(f"💥 Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Live Tennis Betting System')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--bankroll', type=float, help='Override bankroll amount')
    parser.add_argument('--kelly-multiplier', type=float, help='Override Kelly multiplier')
    parser.add_argument('--edge-threshold', type=float, help='Override edge threshold')
    parser.add_argument('--dry-run', action='store_true', help='Run without actually placing bets')
    parser.add_argument('--skip-rankings-refresh', action='store_true', help='Skip ATP rankings scrape (use cached data/atp_rankings.csv)')

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = LiveBettingOrchestrator(args.config)

    # Apply command line overrides
    if args.bankroll:
        orchestrator.config['bankroll'] = args.bankroll
    if args.kelly_multiplier:
        orchestrator.config['kelly_multiplier'] = args.kelly_multiplier
    if args.edge_threshold:
        orchestrator.config['edge_threshold'] = args.edge_threshold
    # Patch refresh flag onto orchestrator
    if args.skip_rankings_refresh:
        orchestrator._refresh_atp_rankings = lambda: print("📊 ATP rankings refresh skipped (--skip-rankings-refresh)")

    # Run pipeline
    success = orchestrator.run_full_pipeline()
    
    if args.dry_run:
        print("\n🧪 DRY RUN - No bets would actually be placed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()