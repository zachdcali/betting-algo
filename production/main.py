#!/usr/bin/env python3
"""
Live Tennis Betting System - Main Orchestrator
Coordinates odds fetching, feature extraction, model inference, and stake calculation
"""

import argparse
import sys
import re
from collections import Counter
import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
import os

# Add production modules to path
sys.path.append(str(Path(__file__).parent))

from odds.fetch_bovada import fetch_bovada_tennis_odds, save_odds_data
from features.ta_feature_calculator import TAFeatureCalculator, UnsafeToInferError
from models.inference import (
    EXACT_141_FEATURES,
    MODEL_VERSION,
    RF_MODEL_VERSION,
    XGB_MODEL_VERSION,
    RandomForestPredictor,
    TennisPredictor,
    XGBoostPredictor,
    calculate_betting_edges,
)
from utils.stake_calculator import KellyStakeCalculator
from utils.bet_tracker import BetTracker
from tournaments.resolve_tournament import TournamentResolver, level_hint_from_title
from prediction_logger import log_prediction
from audit_logger import log_skipped_live_match, upsert_run_history
from logging_utils import build_feature_snapshot_id, build_match_uid, make_run_id, utc_now
from scraping.atp_rankings_scraper import fetch_atp_rankings, save_rankings

class LiveBettingOrchestrator:
    """Main orchestrator for live tennis betting system"""

    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.predictor = TennisPredictor()
        self.xgb_predictor = XGBoostPredictor()
        self.rf_predictor = RandomForestPredictor()
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
        self.run_id = None
        self.run_started_at = None
        self.run_metrics = {}
        self.rankings_refresh_enabled = True

        # Tournament resolver for surface/level/draw/round
        tournaments_map_path = Path(self.config.get('data_dir', '../data')) / 'tournaments_map.csv'
        self.tournament_resolver = TournamentResolver(str(tournaments_map_path)) if tournaments_map_path.exists() else None

    def _start_run_context(self):
        """Create per-run metadata shared across odds, features, and predictions."""
        started = utc_now()
        self.run_id = make_run_id(started)
        self.run_started_at = started.replace(microsecond=0).isoformat()
        self.run_metrics = {
            'run_id': self.run_id,
            'run_kind': 'prediction_pipeline',
            'started_at': self.run_started_at,
            'completed_at': '',
            'status': 'running',
            'auto_settle_enabled': '',
            'rankings_refresh_enabled': '',
            'odds_rows_fetched': 0,
            'odds_rows_candidate': 0,
            'feature_rows_total': 0,
            'feature_rows_ok': 0,
            'feature_rows_skipped': 0,
            'feature_skip_reason_summary': {},
            'prediction_rows_total': 0,
            'prediction_rows_success': 0,
            'prediction_rows_error': 0,
            'prediction_log_attempts': 0,
            'prediction_log_created': 0,
            'prediction_log_updated': 0,
            'prediction_log_skipped_incomplete': 0,
            'bet_opportunities': 0,
            'bets_logged': 0,
            'settlement_candidates': 0,
            'settlement_newly_settled': 0,
            'settlement_auto_settled_bets': 0,
            'settlement_reason_summary': {},
            'error_message': '',
        }
        upsert_run_history(self.run_metrics)

    def _flush_run_history(self, status: str | None = None, error_message: str = ""):
        """Write the current run summary for dashboards and ops debugging."""
        if not self.run_metrics:
            return
        payload = dict(self.run_metrics)
        if status is not None:
            payload['status'] = status
            self.run_metrics['status'] = status
        if error_message:
            payload['error_message'] = error_message
            self.run_metrics['error_message'] = error_message
        if payload.get('status') not in {'running', ''}:
            completed_at = utc_now().replace(microsecond=0).isoformat()
            payload['completed_at'] = completed_at
            self.run_metrics['completed_at'] = completed_at
        upsert_run_history(payload)

    @staticmethod
    def parse_match_date(match_time_str: str) -> datetime:
        """
        Parse Bovada match_time string to a datetime for feature computation.
        Examples: "Today 7:30 PM", "Tomorrow 3:00 AM", "3/16/26 2:00 PM"
        Falls back to today if unparseable.
        """
        return LiveBettingOrchestrator.parse_match_start_datetime(match_time_str) or datetime.now()

    @staticmethod
    def parse_match_start_datetime(match_time_str: str, now: datetime = None):
        """
        Parse Bovada match_time into a local naive datetime when possible.

        Supported examples:
        - "4/21/26 7:30 PM"
        - "4/21/2026 7:30 PM"
        - "Today 7:30 PM"
        - "Tomorrow 3:00 AM"
        - "Sat 7:30 PM"
        """
        if not match_time_str or match_time_str == "Unknown":
            return None

        now = now or datetime.now()
        text = str(match_time_str).strip()

        for fmt in ("%m/%d/%y %I:%M %p", "%m/%d/%Y %I:%M %p", "%m/%d/%y", "%m/%d/%Y"):
            try:
                parsed = datetime.strptime(text, fmt)
                if "%I:%M %p" not in fmt:
                    parsed = parsed.replace(hour=now.hour, minute=now.minute)
                return parsed
            except ValueError:
                pass

        time_match = re.search(r"(\d{1,2}:\d{2}\s*(?:AM|PM))", text, re.I)
        parsed_time = None
        if time_match:
            try:
                parsed_time = datetime.strptime(time_match.group(1).upper(), "%I:%M %p")
            except ValueError:
                parsed_time = None

        lower = text.lower()
        if lower.startswith("today") and parsed_time:
            return now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )
        if lower.startswith("tomorrow") and parsed_time:
            base = now + timedelta(days=1)
            return base.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )

        dow_match = re.match(r"^(mon|tue|wed|thu|fri|sat|sun)\b", lower)
        if dow_match and parsed_time:
            day_map = {
                "mon": 0,
                "tue": 1,
                "wed": 2,
                "thu": 3,
                "fri": 4,
                "sat": 5,
                "sun": 6,
            }
            target = day_map[dow_match.group(1)]
            days_ahead = (target - now.weekday()) % 7
            base = now + timedelta(days=days_ahead)
            candidate = base.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )
            if candidate < now and days_ahead == 0:
                candidate = candidate + timedelta(days=7)
            return candidate

        if parsed_time:
            return now.replace(
                hour=parsed_time.hour,
                minute=parsed_time.minute,
                second=0,
                microsecond=0,
            )

        return None

    def get_inference_guard_reason(self, match_time_str: str) -> tuple[datetime | None, str]:
        """
        Return a skip reason when a match is too close to start or already in progress.

        This keeps the TA-driven feature path out of post-start territory where the
        current match could already appear in player history.
        """
        start_dt = self.parse_match_start_datetime(match_time_str)
        if start_dt is None:
            return None, ""

        buffer_minutes = int(self.config.get('pre_match_inference_buffer_minutes', 5))
        cutoff = start_dt - timedelta(minutes=buffer_minutes)
        now = datetime.now()
        if now >= start_dt:
            return start_dt, "scheduled_start_passed"
        if now >= cutoff:
            return start_dt, f"inside_pre_match_buffer_{buffer_minutes}m"
        return start_dt, ""

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
            'models_dir': '../results/professional_tennis/Neural_Network',
            'pre_match_inference_buffer_minutes': 5,
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
        self.run_metrics['odds_rows_fetched'] = len(odds_df)
        
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

        if not self.run_id:
            self._start_run_context()

        print("🔧 Extracting features from Tennis Abstract...")

        # Shared session cache: avoids duplicate TA requests within a single run
        session_cache = {}

        # Filter out futures/outrights before feature extraction
        odds_df = odds_df[~odds_df.apply(
            lambda r: 'field' in str(r.get('player1_raw', '')).lower()
                   or 'field' in str(r.get('player2_raw', '')).lower()
                   or 'futures' in str(r.get('event', '')).lower()
                   or 'outright' in str(r.get('event', '')).lower(),
            axis=1
        )].copy()
        self.run_metrics['odds_rows_candidate'] = len(odds_df)

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
            match_start_dt, guard_reason = self.get_inference_guard_reason(match_time_str)
            # Flag whether Bovada gave us a real absolute date (e.g. "3/22/26 8:00 AM")
            # vs. just a time or no date (defaults to today)
            import re as _re2
            match_date_is_explicit = bool(_re2.search(r'\d{1,2}/\d{1,2}/\d{2,4}', match_time_str))

            has_defaulted = False
            status_detail = ""
            try:
                if guard_reason:
                    start_label = match_start_dt.isoformat(sep=' ', timespec='minutes') if match_start_dt else match_time_str
                    print(f"      ⏭️  Skipping pre-match inference for {p1} vs {p2} — {guard_reason} (scheduled {start_label})")
                    features = {}
                    status = "skip"
                    status_detail = guard_reason
                else:
                    features = self.feature_engine.build_141_features(
                        player1_name=p1,
                        player2_name=p2,
                        match_date=match_date,
                        surface=surface,
                        tournament_level=tournament_level,
                        draw_size=draw_size,
                        round_code=round_code,
                        expected_event_title=row.get('event', ''),
                        force_refresh=True,
                        persist=False,
                        session_cache=session_cache,
                        match_date_is_explicit=match_date_is_explicit,
                    )
                    # features_complete=False if ANY meaningful feature was defaulted
                    # (includes ATP points fallback, round=None, structural defaults)
                    has_defaulted = bool(features.get('_defaulted_features', ''))
                    status = "ok"
            except UnsafeToInferError as e:
                print(f"      ⏭️  Skipping pre-match inference for {p1} vs {p2} — {e}")
                features = {}
                status = "skip"
                status_detail = str(e)
            except Exception as e:
                print(f"      ⚠️ Feature extraction failed: {e}")
                features = {}
                status = "skip"
                status_detail = f"feature_error:{e}"

            # Ordered 141 features for the model — flag any that are absent from the returned dict
            missing_from_dict = [k for k in EXACT_141_FEATURES if k not in features]
            if missing_from_dict and status == "ok":
                print(f"      ⚠️  MISSING FROM FEATURE DICT (filled 0): {missing_from_dict}")
                # Treat missing-from-dict features as defaulted too
                existing_defaults = features.get('_defaulted_features', '')
                missing_str = ','.join(missing_from_dict)
                features['_defaulted_features'] = f"{existing_defaults},{missing_str}" if existing_defaults else missing_str
                has_defaulted = True
            ordered_features = {k: features.get(k, 0.0) for k in EXACT_141_FEATURES}

            resolved_round = features.get('_resolved_round_code') or round_code or ''
            resolved_match_date = features.get('_resolved_match_date') or ''
            if not resolved_match_date:
                fallback_dt = pd.to_datetime(
                    row.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                    errors='coerce',
                )
                resolved_match_date = (
                    fallback_dt.date().isoformat()
                    if pd.notna(fallback_dt)
                    else match_date.date().isoformat()
                )

            match_uid = build_match_uid(
                row.get('player1_raw', p1),
                row.get('player2_raw', p2),
                resolved_match_date,
                row.get('event', ''),
                resolved_round,
                surface,
            )
            feature_snapshot_id = build_feature_snapshot_id(
                match_uid,
                self.run_id,
                row.get('player1_raw', p1),
                row.get('player2_raw', p2),
            )

            # Metadata (not fed to model)
            ordered_features.update({
                'run_id': self.run_id,
                'run_started_at': self.run_started_at,
                '_has_defaulted_features': has_defaulted,
                'match_id': idx,
                'match_uid': match_uid,
                'feature_snapshot_id': feature_snapshot_id,
                'player1_raw': row.get('player1_raw', p1),
                'player2_raw': row.get('player2_raw', p2),
                'event': row.get('event', ''),
                'timestamp': row.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'match_time': row.get('match_time', ''),
                'match_start_dt_local': match_start_dt.isoformat() if match_start_dt else '',
                'status': status,
                'status_detail': status_detail,
                'meta_level_input': tournament_level,
                'meta_surface_input': surface,
                'meta_round_input': resolved_round,
                'meta_match_date': resolved_match_date,
                'meta_defaulted_features': features.get('_defaulted_features') or '',
                'meta_draw_input': draw_size,
                'meta_resolver_source': resolver_source
            })

            if status == "skip":
                skip_reason_code = (status_detail or "feature_skip_unknown").split(':', 1)[0]
                log_skipped_live_match(
                    run_id=self.run_id,
                    run_started_at=self.run_started_at,
                    stage='feature_extraction',
                    skip_reason_code=skip_reason_code,
                    skip_reason_detail=status_detail,
                    match_uid=match_uid,
                    feature_snapshot_id=feature_snapshot_id,
                    match_date=resolved_match_date,
                    match_start_time=row.get('match_time', ''),
                    match_start_dt_local=match_start_dt.isoformat() if match_start_dt else '',
                    odds_scraped_at=row.get('scrape_time_utc', '') or row.get('timestamp', ''),
                    tournament=row.get('tourney_name', '') or row.get('event', ''),
                    event_title=row.get('event', ''),
                    surface=surface,
                    level=tournament_level,
                    round_code=resolved_round,
                    resolver_source=resolver_source,
                    p1=p1,
                    p2=p2,
                    defaulted_features=features.get('_defaulted_features', '') or '',
                )

            feature_rows.append(ordered_features)

        features_df = pd.DataFrame(feature_rows)

        if not features_df.empty:
            features_file = self.logs_dir / f"features_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            features_df.to_csv(features_file, index=False)
            ok_count = (features_df['status'] == 'ok').sum()
            skip_count = (features_df['status'] == 'skip').sum()
            skip_counter = Counter(
                features_df.loc[features_df['status'] == 'skip', 'status_detail']
                .fillna('feature_skip_unknown')
                .replace('', 'feature_skip_unknown')
                .tolist()
            )
            self.run_metrics['feature_rows_total'] = len(features_df)
            self.run_metrics['feature_rows_ok'] = int(ok_count)
            self.run_metrics['feature_rows_skipped'] = int(skip_count)
            self.run_metrics['feature_skip_reason_summary'] = dict(skip_counter)
            print(f"✅ Feature extraction: {ok_count} ok, {skip_count} skipped")

        return features_df
    
    def generate_predictions(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Generate model predictions (NN + XGBoost)"""
        if features_df.empty:
            return pd.DataFrame()

        print("🎯 Generating predictions...")
        predictions_df = self.predictor.predict_slate(features_df)

        # Also run XGBoost on each match
        if not self.xgb_predictor.is_loaded:
            self.xgb_predictor.load_model()
        if self.xgb_predictor.is_loaded:
            xgb_p1_probs = []
            xgb_p2_probs = []
            for _, row in predictions_df.iterrows():
                if row.get('prediction_status') != 'success':
                    xgb_p1_probs.append(None)
                    xgb_p2_probs.append(None)
                    continue
                xgb_result = self.xgb_predictor.predict_match_probability(row.to_dict())
                if 'error' not in xgb_result:
                    xgb_p1_probs.append(xgb_result['xgb_p1_prob'])
                    xgb_p2_probs.append(xgb_result['xgb_p2_prob'])
                else:
                    xgb_p1_probs.append(None)
                    xgb_p2_probs.append(None)
            predictions_df['xgb_p1_prob'] = xgb_p1_probs
            predictions_df['xgb_p2_prob'] = xgb_p2_probs

        # Also run Random Forest on each match
        if not self.rf_predictor.is_loaded:
            self.rf_predictor.load_model()
        if self.rf_predictor.is_loaded:
            rf_p1_probs = []
            rf_p2_probs = []
            for _, row in predictions_df.iterrows():
                if row.get('prediction_status') != 'success':
                    rf_p1_probs.append(None)
                    rf_p2_probs.append(None)
                    continue
                rf_result = self.rf_predictor.predict_match_probability(row.to_dict())
                if 'error' not in rf_result:
                    rf_p1_probs.append(rf_result['rf_p1_prob'])
                    rf_p2_probs.append(rf_result['rf_p2_prob'])
                else:
                    rf_p1_probs.append(None)
                    rf_p2_probs.append(None)
            predictions_df['rf_p1_prob'] = rf_p1_probs
            predictions_df['rf_p2_prob'] = rf_p2_probs

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
        stats = {
            'attempts': 0,
            'created': 0,
            'updated': 0,
            'skipped_incomplete': 0,
        }
        try:
            today = datetime.now().date().isoformat()

            for _, pred_row in predictions_df.iterrows():
                p1 = pred_row.get('player1_normalized') or pred_row.get('player1_raw', '')
                p2 = pred_row.get('player2_normalized') or pred_row.get('player2_raw', '')

                # Skip futures/outrights ("vs The Field", "futures" in event name)
                if 'field' in str(p2).lower() or 'field' in str(p1).lower():
                    continue
                event = str(pred_row.get('event', '') or '')
                if 'futures' in event.lower() or 'outright' in event.lower():
                    continue

                model_p1 = pred_row.get('player1_win_prob') or pred_row.get('p1_win_prob')
                if model_p1 is None:
                    continue
                stats['attempts'] += 1
                model_p2 = 1.0 - float(model_p1)

                # features_complete: False if the calculator reported any noisy defaults
                features_complete = not bool(pred_row.get('_has_defaulted_features', False))
                if not features_complete:
                    defaulted = pred_row.get('meta_defaulted_features', '')
                    print(f"  ⛔ Skipping prediction log for {p1} vs {p2} — incomplete features: {defaulted}")
                    stats['skipped_incomplete'] += 1
                    log_skipped_live_match(
                        run_id=pred_row.get('run_id', self.run_id),
                        run_started_at=self.run_started_at,
                        stage='prediction_logging',
                        skip_reason_code='incomplete_features',
                        skip_reason_detail=defaulted,
                        match_uid=pred_row.get('match_uid', ''),
                        feature_snapshot_id=pred_row.get('feature_snapshot_id', ''),
                        match_date=pred_row.get('meta_match_date', '') or today,
                        match_start_time=pred_row.get('match_time', ''),
                        match_start_dt_local=pred_row.get('match_start_dt_local', ''),
                        odds_scraped_at=pred_row.get('timestamp', ''),
                        tournament=pred_row.get('event', ''),
                        event_title=pred_row.get('event', ''),
                        surface=pred_row.get('meta_surface_input', ''),
                        level=pred_row.get('meta_level_input', ''),
                        round_code=pred_row.get('meta_round_input', ''),
                        resolver_source=pred_row.get('meta_resolver_source', ''),
                        p1=p1,
                        p2=p2,
                        defaulted_features=defaulted,
                    )
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
                    od1 = o_row.get('player1_odds_decimal')
                    od2 = o_row.get('player2_odds_decimal')
                    sph = o_row.get('spread_handicap')
                    sp1 = o_row.get('spread_odds_p1')
                    sp2 = o_row.get('spread_odds_p2')
                    tg = o_row.get('total_games')
                    tov = o_row.get('total_odds_over')
                    tun = o_row.get('total_odds_under')
                    tournament = o_row.get('tourney_name', '')
                    surface = pred_row.get('meta_surface_input', o_row.get('surface', 'Hard'))
                    level = pred_row.get('meta_level_input', o_row.get('tourney_level', ''))
                    match_time = o_row.get('match_time', '')
                    odds_scraped_at = o_row.get('scrape_time_utc', '') or o_row.get('timestamp', '')
                else:
                    mkt_p1, mkt_p2, o1, o2 = 0.5, 0.5, None, None
                    od1, od2 = None, None
                    sph, sp1, sp2 = None, None, None
                    tg, tov, tun = None, None, None
                    tournament = ''
                    surface = pred_row.get('meta_surface_input', 'Hard')
                    level = pred_row.get('meta_level_input', '')
                    match_time = ''
                    odds_scraped_at = ''

                # Use TA-inferred match date (tourney_start + round_offset) — Bovada only gives clock time
                match_date = pred_row.get('meta_match_date') or today
                model_version = pred_row.get('model_version', 'NN-SURFACE_FIX')
                nn_probability_source = pred_row.get('probability_source', 'raw')
                # Get ranks from features
                p1_rank = pred_row.get('Player1_Rank')
                p2_rank = pred_row.get('Player2_Rank')
                if pd.notna(p1_rank) and float(p1_rank) > 0:
                    p1_rank = float(p1_rank)
                else:
                    p1_rank = None
                if pd.notna(p2_rank) and float(p2_rank) > 0:
                    p2_rank = float(p2_rank)
                else:
                    p2_rank = None

                # XGBoost prediction (if available)
                xgb_p1 = pred_row.get('xgb_p1_prob')
                xgb_p2 = pred_row.get('xgb_p2_prob')
                if pd.notna(xgb_p1):
                    xgb_p1 = float(xgb_p1)
                    xgb_p2 = float(xgb_p2)
                else:
                    xgb_p1, xgb_p2 = None, None

                # Random Forest prediction (if available)
                rf_p1 = pred_row.get('rf_p1_prob')
                rf_p2 = pred_row.get('rf_p2_prob')
                if pd.notna(rf_p1):
                    rf_p1 = float(rf_p1)
                    rf_p2 = float(rf_p2)
                else:
                    rf_p1, rf_p2 = None, None

                action = log_prediction(
                    p1=p1, p2=p2,
                    tournament=tournament, surface=surface, level=level,
                    round_code=pred_row.get('meta_round_input', '') or None,
                    match_date=match_date,
                    run_id=pred_row.get('run_id', self.run_id),
                    match_uid=pred_row.get('match_uid'),
                    feature_snapshot_id=pred_row.get('feature_snapshot_id'),
                    model_p1_prob=float(model_p1), model_p2_prob=model_p2,
                    market_p1_prob=mkt_p1, market_p2_prob=mkt_p2,
                    p1_rank=p1_rank, p2_rank=p2_rank,
                    p1_odds_american=o1, p2_odds_american=o2,
                    p1_odds_decimal=od1, p2_odds_decimal=od2,
                    spread_handicap=sph, spread_odds_p1=sp1, spread_odds_p2=sp2,
                    total_games=tg, total_odds_over=tov, total_odds_under=tun,
                    xgb_p1_prob=xgb_p1, xgb_p2_prob=xgb_p2,
                    rf_p1_prob=rf_p1, rf_p2_prob=rf_p2,
                    model_version=model_version,
                    nn_model_version=model_version or MODEL_VERSION,
                    xgb_model_version=XGB_MODEL_VERSION if xgb_p1 is not None else '',
                    rf_model_version=RF_MODEL_VERSION if rf_p1 is not None else '',
                    nn_probability_source=nn_probability_source,
                    odds_scraped_at=odds_scraped_at,
                    match_start_time=match_time,
                    features_complete=features_complete,
                    defaulted_features=pred_row.get('meta_defaulted_features', ''),
                )
                if action == 'created':
                    stats['created'] += 1
                elif action == 'updated':
                    stats['updated'] += 1
        except Exception as e:
            print(f"  ⚠️ Prediction logging failed (non-fatal): {e}")
            import traceback; traceback.print_exc()
        return stats

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

    def run_full_pipeline(self, start_session: bool = True, auto_settle: bool = True) -> bool:
        """Run the complete betting pipeline with bet tracking"""
        session_id = None
        try:
            self._start_run_context()
            self.run_metrics['auto_settle_enabled'] = bool(auto_settle)
            self.run_metrics['rankings_refresh_enabled'] = bool(self.rankings_refresh_enabled)
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
            if auto_settle:
                try:
                    from auto_settle import run as auto_settle_run
                    print("\n📋 Auto-settling pending predictions...")
                    settle_summary = auto_settle_run(dry_run=False, run_id=self.run_id, record_run_history=False)
                    if isinstance(settle_summary, dict):
                        self.run_metrics['settlement_candidates'] = settle_summary.get('settlement_candidates', 0)
                        self.run_metrics['settlement_newly_settled'] = settle_summary.get('settlement_newly_settled', 0)
                        self.run_metrics['settlement_auto_settled_bets'] = settle_summary.get('settlement_auto_settled_bets', 0)
                        self.run_metrics['settlement_reason_summary'] = settle_summary.get('settlement_reason_summary', {})
                except Exception as e:
                    print(f"  ⚠️  Auto-settle failed (non-fatal): {e}")
                    self.run_metrics['settlement_reason_summary'] = {'auto_settle_error': 1}
            else:
                print("\n⏭️  Auto-settle skipped for this run")

            # Step 0b: Refresh ATP rankings (rank + points)
            self._refresh_atp_rankings()

            # Step 1: Fetch odds
            odds_df = self.fetch_odds()
            if odds_df.empty:
                print("⚠️  No odds available, stopping pipeline")
                self._flush_run_history(status='no_odds')
                return False
            
            # Step 2: Extract features
            features_df = self.extract_features(odds_df)
            if features_df.empty:
                print("⚠️  Feature extraction failed, stopping pipeline")
                self._flush_run_history(status='no_features')
                return False
            
            # Step 3: Generate predictions
            predictions_df = self.generate_predictions(features_df)
            self.run_metrics['prediction_rows_total'] = len(predictions_df)
            if 'prediction_status' in predictions_df.columns:
                success_count = int((predictions_df['prediction_status'] == 'success').sum())
            else:
                success_count = len(predictions_df)
            self.run_metrics['prediction_rows_success'] = success_count
            self.run_metrics['prediction_rows_error'] = len(predictions_df) - success_count
            if predictions_df.empty:
                print("⚠️  Prediction generation failed, stopping pipeline")
                self._flush_run_history(status='no_predictions')
                return False
            
            # Step 4: Calculate stakes
            bet_slips_df = self.calculate_edges_and_stakes(predictions_df, odds_df)

            # Step 4b: Log all predictions (regardless of edge threshold)
            prediction_log_stats = self._log_all_predictions(predictions_df, odds_df, features_df)
            self.run_metrics['prediction_log_attempts'] = prediction_log_stats.get('attempts', 0)
            self.run_metrics['prediction_log_created'] = prediction_log_stats.get('created', 0)
            self.run_metrics['prediction_log_updated'] = prediction_log_stats.get('updated', 0)
            self.run_metrics['prediction_log_skipped_incomplete'] = prediction_log_stats.get('skipped_incomplete', 0)
            self.run_metrics['bet_opportunities'] = len(bet_slips_df)

            # Step 5: Save, log, and display results
            if not bet_slips_df.empty:
                self.save_bet_slips(bet_slips_df)
                
                # Log bets to tracking system
                if session_id:
                    self.bet_tracker.log_bets(bet_slips_df, session_id, bankroll)
                    self.run_metrics['bets_logged'] = len(bet_slips_df)
                
                self.print_summary(bet_slips_df)
                
                # Show tracking info
                if session_id:
                    pending_bets = self.bet_tracker.get_pending_bets()
                    print(f"\n📊 Session tracking:")
                    print(f"   Session ID: {session_id}")
                    print(f"   Total pending bets: {len(pending_bets)}")
                    print(f"   Use 'python settle_bets.py {session_id}' to settle results later")
                
                self._flush_run_history(status='success')
                return True
            else:
                print("📊 No profitable betting opportunities found")
                self._flush_run_history(status='success')
                return False
                
        except Exception as e:
            print(f"💥 Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            self._flush_run_history(status='failed', error_message=str(e))
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
    parser.add_argument('--skip-auto-settle', action='store_true', help='Skip pre-run auto-settlement and go straight to prediction generation')

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
        orchestrator.rankings_refresh_enabled = False
        orchestrator._refresh_atp_rankings = lambda: print("📊 ATP rankings refresh skipped (--skip-rankings-refresh)")

    # Run pipeline
    success = orchestrator.run_full_pipeline(auto_settle=not args.skip_auto_settle)
    
    if args.dry_run:
        print("\n🧪 DRY RUN - No bets would actually be placed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
