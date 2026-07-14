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
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import os

# Add production modules to path
sys.path.append(str(Path(__file__).parent))

from odds.fetch_bovada import fetch_bovada_tennis_odds, save_odds_data
from features.ta_feature_calculator import TAFeatureCalculator, UnsafeToInferError
from features.performance_v1 import PERFORMANCE_FEATURES, build_match_performance_features
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
from tournaments.fallback_heuristics import get_fallback_tournament_meta
from tournaments.resolve_tournament import TournamentResolver, level_hint_from_title
from prediction_logger import log_prediction
from shadow.performance_v1_shadow import (
    PerformanceV1ShadowEnsemble,
    log_shadow_predictions,
    shadow_row_from_prediction,
)
from audit_logger import log_skipped_live_match, upsert_run_history
from logging_utils import build_feature_snapshot_id, build_match_uid, make_run_id, utc_now
from scraping.atp_rankings_scraper import resolve_rankings, save_rankings

BOVADA_TIMEZONE = ZoneInfo("America/New_York")


def prediction_terminal_status(predictions: pd.DataFrame) -> tuple[str, int, int]:
    """Return terminal health plus success/error counts for inference output."""
    total = len(predictions)
    if total == 0:
        return "no_predictions", 0, 0
    if "prediction_status" not in predictions.columns:
        return "success", total, 0
    success = int((predictions["prediction_status"] == "success").sum())
    errors = total - success
    if success == 0:
        return "no_predictions", 0, errors
    return ("partial" if errors else "success"), success, errors


def _atp_live_surface(event_label: str, session_cache: dict):
    """Surface from the live ATP challenger calendar (official, current season)."""
    try:
        cal = (session_cache.get("atp_calendar") or {}).get("df")
        if cal is None or cal.empty or "surface" not in cal.columns:
            return None
        city = str(event_label).split(",")[0].split("(")[0].strip().lower()
        city = " ".join(w for w in city.split()
                        if w.isalpha() and w not in ("atp", "challenger", "itf", "men", "mens", "wta"))
        if len(city) < 4:
            return None
        hits = cal[cal["event"].astype(str).str.lower().str.contains(city, regex=False) & cal["surface"].notna()]
        return hits.iloc[0]["surface"] if len(hits) == 1 else None
    except Exception:
        return None

def _itf_surface(event_label: str, session_cache: dict):
    """Surface from the ITF calendar API (surfaceDesc) for ITF Men <City> labels."""
    if "itf" not in str(event_label).lower():
        return None
    try:
        from features.history_stitch import _itf_event_for
        from datetime import date
        ev = _itf_event_for(event_label, date.today(), session_cache)
        return (ev or {}).get("surface") or None
    except Exception:
        return None

def _hand_of(pred_row: dict, pref: str) -> str:
    """Handedness one-hot -> 'R'/'L'/'U' for the prediction row (dashboard badge)."""
    try:
        if float(pred_row.get(f"{pref}_Hand_U", 0) or 0) == 1:
            return "U"
        if float(pred_row.get(f"{pref}_Hand_L", 0) or 0) == 1:
            return "L"
        if float(pred_row.get(f"{pref}_Hand_R", 0) or 0) == 1:
            return "R"
    except (TypeError, ValueError):
        pass
    return ""


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
        self.bet_tracker = BetTracker(
            str(self.logs_dir), initial_bankroll=self.config.get('bankroll', 1000.0)
        )
        self.feature_engine = TAFeatureCalculator()
        self.performance_shadow_enabled = bool(self.config.get('performance_shadow_enabled', True))
        self.performance_shadow_predictor = PerformanceV1ShadowEnsemble()
        # One cache per run.  Prefetch warms this before feature extraction;
        # extract_features must reuse it rather than replacing it.
        self._session_cache = {}
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
        self._session_cache = {}
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
            'auto_settle_status': 'not_started',
            'auto_settle_error': '',
            'canonical_ingest_status': 'not_started',
            'canonical_ingest_rows': 0,
            'canonical_ingest_error': '',
            'reconcile_status': 'not_started',
            'reconcile_error': '',
            'account_equity': 0,
            'pending_exposure': 0,
            'available_bankroll': 0,
            'exposure_gate_status': 'not_checked',
            'performance_shadow_attempts': 0,
            'performance_shadow_logged': 0,
            'performance_shadow_models_loaded': 0,
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

    def _persist_run_state(self, status: str | None = None, error_message: str = ""):
        """Write run metadata before publishing the state snapshot.

        The old order synced the dashboard first and only then marked the run
        complete, leaving every mirrored run stuck at ``running`` with its
        initial zero counters.  This ordering also provides an explicit
        checkpoint after settlement, before slower scraping starts.
        """
        self._flush_run_history(status=status, error_message=error_message)
        self._refresh_database()

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

        # The browser is pinned to America/New_York, so relative labels and
        # absolute clocks must be interpreted in that same display zone on
        # every machine (including UTC GitHub runners).
        now = now or datetime.now(BOVADA_TIMEZONE).replace(tzinfo=None)
        if now.tzinfo is not None:
            now = now.astimezone(BOVADA_TIMEZONE).replace(tzinfo=None)
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

    @staticmethod
    def match_start_at_utc(match_time_str: str) -> str:
        parsed = LiveBettingOrchestrator.parse_match_start_datetime(match_time_str)
        if parsed is None:
            return ""
        eastern = parsed.replace(tzinfo=BOVADA_TIMEZONE)
        return eastern.astimezone(timezone.utc).isoformat(timespec="seconds")

    def get_inference_guard_reason(self, match_time_str: str) -> tuple[datetime | None, str]:
        """
        Return a skip reason when a match is too close to start or already in progress.

        This keeps the TA-driven feature path out of post-start territory where the
        current match could already appear in player history.
        """
        start_dt = self.parse_match_start_datetime(match_time_str)
        if start_dt is None:
            # Without a scheduled start we cannot prove this is pre-match.
            # Skip before touching player history so a completed/current match
            # can never leak into its own feature vector.
            return None, "match_start_time_missing"

        buffer_minutes = int(self.config.get('pre_match_inference_buffer_minutes', 5))
        cutoff = start_dt - timedelta(minutes=buffer_minutes)
        now = datetime.now(BOVADA_TIMEZONE).replace(tzinfo=None)
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
            'performance_shadow_enabled': True,
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
                default_config.update(config)
        
        return default_config
    
    def fetch_odds(self) -> pd.DataFrame:
        """Fetch current odds from Bovada"""
        print("🎾 Fetching odds from Bovada...")
        self._odds_fetch_failed = False
        try:
            odds_df = fetch_bovada_tennis_odds(headless=True)
        except Exception as exc:
            # Fetch CRASH is not the same as an empty board: settlement/ingest
            # work still persists, but the failure stays loud (status
            # 'odds_fetch_error'), and two consecutive crashes hard-fail the
            # run so persistent breakage alerts instead of scrolling by.
            print(f"❌ Bovada odds fetch failed (continuing to persist settlement work): {exc}")
            self.run_metrics['odds_fetch_error'] = str(exc)[:200]
            self._odds_fetch_failed = True
            odds_df = pd.DataFrame()
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

        # Shared session cache: prefetch has already warmed this object.  Keep
        # it on self so post-run hooks can ingest fetched event results without
        # refetching.
        session_cache = self._session_cache

        # Filter out futures/outrights before feature extraction
        odds_df = odds_df[~odds_df.apply(
            lambda r: 'field' in str(r.get('player1_raw', '')).lower()
                   or 'field' in str(r.get('player2_raw', '')).lower()
                   or 'futures' in str(r.get('event', '')).lower()
                   or 'outright' in str(r.get('event', '')).lower(),
            axis=1
        )].copy()

        # Cloud runs skip ITF futures (SKIP_ITF_MATCHES=1): no round source exists
        # for ITF yet (itftennis.com scraper unbuilt) so they are never bettable,
        # and their per-player fetches were blowing the hourly runner's budget.
        if os.environ.get("SKIP_ITF_MATCHES") == "1":
            n_before = len(odds_df)
            odds_df = odds_df[~odds_df['event'].astype(str).str.contains("ITF", case=False, na=False)].copy()
            if n_before - len(odds_df):
                print(f"   ⏭️  Skipping {n_before - len(odds_df)} ITF match(es) (SKIP_ITF_MATCHES=1)")

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
                    surface_is_guess = False
                    from canonical_store import surface_from_store
                    _live = _itf_surface(row['event'], session_cache) or _atp_live_surface(row['event'], session_cache)
                    if _live and surface and _live != surface:
                        print(f"      🔁 registry={surface} overridden by LIVE official source: {_live}")
                        surface = _live
                    elif not _live:
                        _corr = surface_from_store(row['event'], session_cache)
                        if _corr and surface and _corr != surface:
                            print(f"      🚨 SURFACE DISAGREEMENT: registry={surface} vs store={_corr} — flagging as guess")
                            surface_is_guess = True
                    print(f"      📍 {surface}, Level:{tournament_level}, Draw:{draw_size} (score:{score:.3f})")
                else:
                    fallback_meta = get_fallback_tournament_meta(row['event'])
                    from canonical_store import surface_from_store
                    _store_surf = fallback_meta.surface or _itf_surface(row['event'], session_cache) or _atp_live_surface(row['event'], session_cache) or surface_from_store(row['event'], session_cache)
                    surface_is_guess = _store_surf is None
                    surface = _store_surf or surface
                    # smart alias learning: if the registry HAD a near-miss candidate whose
                    # surface the store independently confirms, adopt it and remember the
                    # sponsor-mangled title permanently (two sources agree -> safe to learn)
                    if _store_surf and self.tournament_resolver is not None:
                        # two independent confirmations before learning: the city token
                        # must identify exactly one registry tournament AND that entry's
                        # surface must equal the store's answer
                        try:
                            _city = str(row['event']).split(',')[0].split('(')[0].strip().lower()
                            _city = " ".join(w for w in _city.split()
                                             if w.isalpha() and w not in ("atp","challenger","itf","men","mens","wta"))
                            if len(_city) >= 4:
                                _df = self.tournament_resolver.df
                                _hits = _df[_df['canonical_name'].astype(str).str.lower().str.contains(_city, regex=False)]
                                if len(_hits) == 1 and str(_hits.iloc[0].get('surface')) == str(_store_surf):
                                    _idx = _hits.index[0]
                                    self.tournament_resolver.learn_alias(row['event'], _idx, 1.0)
                                    tournament_level = _hits.iloc[0].get('level') or tournament_level
                                    print(f"      🧠 Learned alias: '{str(row['event'])[:40]}' -> {_hits.iloc[0].get('canonical_name','?')} (city+surface corroborated: {_store_surf})")
                        except Exception as _la_exc:
                            print(f"      ⚠️ alias learning skipped: {_la_exc}")
                    tournament_level = fallback_meta.level or level_hint_from_title(row['event']) or "A"
                    draw_size = fallback_meta.draw_size or draw_size
                    resolver_source = "store_surface" if (_store_surf and not fallback_meta.surface) else "fallback_heuristic"
                    print(f"      {'🎾' if not surface_is_guess else '⚠️ '} Fallback metadata: {surface}{'' if not surface_is_guess else ' (GUESS — flagged)'} Level:{tournament_level}, Draw:{draw_size}")
            else:
                fallback_meta = get_fallback_tournament_meta(row['event'])
                from canonical_store import surface_from_store
                _store_surf = fallback_meta.surface or _itf_surface(row['event'], session_cache) or _atp_live_surface(row['event'], session_cache) or surface_from_store(row['event'], session_cache)
                surface_is_guess = _store_surf is None
                surface = _store_surf or surface
                tournament_level = fallback_meta.level or level_hint_from_title(row['event']) or "A"
                draw_size = fallback_meta.draw_size or draw_size
                resolver_source = "store_surface" if (_store_surf and not fallback_meta.surface) else "fallback_heuristic"

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
                        metadata_source=resolver_source,
                    )
                    if locals().get('surface_is_guess'):
                        _d = features.get('_defaulted_features') or ''
                        features['_defaulted_features'] = (_d + ',' if _d else '') + f'Surface={surface}(guess)'
                    # A match with no identifiable tournament can't have its
                    # surface/level verified — never silently bettable (JJ Wolf
                    # was showing complete with tournament=None).
                    event_label = row.get('event', '')
                    if pd.isna(event_label) or not str(event_label).strip():
                        _d = features.get('_defaulted_features') or ''
                        features['_defaulted_features'] = (_d + ',' if _d else '') + 'tournament=missing'
                    if match_start_dt is None:
                        _d = features.get('_defaulted_features') or ''
                        features['_defaulted_features'] = (_d + ',' if _d else '') + 'match_start_time=missing'
                    # features_complete=False if ANY meaningful feature was defaulted
                    # (includes ATP points fallback, round=None, structural defaults)
                    has_defaulted = bool(features.get('_defaulted_features', ''))
                    if match_start_dt is None:
                        status = "skip"
                        status_detail = "match_start_time_missing"
                    else:
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
            from feature_vector_log import feature_fingerprint, feature_validation_issues
            structural_issues = feature_validation_issues(ordered_features)
            if status == "ok" and structural_issues:
                status = "skip"
                status_detail = "feature_schema_invalid:" + ";".join(structural_issues)
                has_defaulted = True
                existing_defaults = features.get('_defaulted_features', '')
                marker = "structural_validation"
                features['_defaulted_features'] = (
                    f"{existing_defaults},{marker}" if existing_defaults else marker
                )
            feature_schema_sha256, feature_vector_sha256, feature_count = (
                feature_fingerprint(ordered_features)
                if status == "ok"
                else ("", "", len(EXACT_141_FEATURES))
            )

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
            resolved_surface = features.get('_resolved_surface') or surface

            performance_status = "not_attempted"
            performance_error = ""
            if status == "ok" and self.performance_shadow_enabled:
                try:
                    slug1 = self.feature_engine.find_slug(p1)
                    slug2 = self.feature_engine.find_slug(p2)
                    if not slug1 or not slug2:
                        raise RuntimeError(f"missing_slug:{p1 if not slug1 else p2}")
                    matches1 = self.feature_engine.scraper.get_player_matches(
                        slug1,
                        years=[],
                        force_refresh=False,
                        persist=False,
                        session_cache=session_cache,
                    )
                    matches2 = self.feature_engine.scraper.get_player_matches(
                        slug2,
                        years=[],
                        force_refresh=False,
                        persist=False,
                        session_cache=session_cache,
                    )
                    performance_features = build_match_performance_features(
                        matches1,
                        matches2,
                        resolved_match_date,
                    )
                    ordered_features.update(performance_features)
                    performance_status = "ok"
                except Exception as e:
                    performance_error = str(e)
                    performance_status = "error"
                    for feature_name in PERFORMANCE_FEATURES:
                        ordered_features[feature_name] = pd.NA
                    print(f"      ⚠️ performance_v1 shadow feature extraction failed: {e}")
            else:
                for feature_name in PERFORMANCE_FEATURES:
                    ordered_features[feature_name] = pd.NA

            match_uid = build_match_uid(
                row.get('player1_raw', p1),
                row.get('player2_raw', p2),
                resolved_match_date,
                row.get('event', ''),
                resolved_round,
                resolved_surface,
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
                'feature_schema_sha256': feature_schema_sha256,
                'feature_vector_sha256': feature_vector_sha256,
                'feature_count': feature_count,
                'player1_raw': row.get('player1_raw', p1),
                'player2_raw': row.get('player2_raw', p2),
                'event': row.get('event', ''),
                'timestamp': row.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                'match_time': row.get('match_time', ''),
                'match_start_dt_local': match_start_dt.isoformat() if match_start_dt else '',
                'status': status,
                'status_detail': status_detail,
                'meta_level_input': tournament_level,
                'meta_surface_input': resolved_surface,
                'meta_round_input': resolved_round,
                'meta_match_date': resolved_match_date,
                'meta_defaulted_features': features.get('_defaulted_features') or '',
                'meta_draw_input': draw_size,
                'meta_resolver_source': resolver_source,
                'performance_v1_features_available': performance_status == "ok",
                'performance_v1_status': performance_status,
                'performance_v1_error': performance_error,
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
                    match_start_at_utc=self.match_start_at_utc(row.get('match_time', '')),
                    odds_scraped_at=row.get('scrape_time_utc', '') or row.get('timestamp', ''),
                    tournament=row.get('tourney_name', '') or row.get('event', ''),
                    event_title=row.get('event', ''),
                    surface=resolved_surface,
                    level=tournament_level,
                    round_code=resolved_round,
                    resolver_source=resolver_source,
                    p1=p1,
                    p2=p2,
                    defaulted_features=features.get('_defaulted_features', '') or '',
                )

            if status == "ok":
                try:
                    from feature_audit import validate_features
                    validate_features(features, p1, p2, self.run_metrics.get('run_id', ''))
                except Exception as _fa_exc:
                    print(f"      ⚠️ feature audit failed (non-fatal): {_fa_exc}")
            try:
                from feature_vector_log import save_feature_vector
                features['_regime'] = MODEL_VERSION  # regime bump unfreezes complete vectors
                save_feature_vector(p1, p2, resolved_match_date, self.run_metrics.get('run_id', ''),
                                    features,
                                    status == "ok" and not bool(features.get('_defaulted_features', '')),
                                    match_uid=match_uid,
                                    feature_snapshot_id=feature_snapshot_id,
                                    build_status=status)
            except Exception as _fv_exc:
                print(f"      ⚠️ feature-vector log failed (non-fatal): {_fv_exc}")

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
    
    def calculate_edges_and_stakes(self, predictions_df: pd.DataFrame,
                                   odds_df: pd.DataFrame,
                                   bankroll: float = None,
                                   available_bankroll: float = None) -> pd.DataFrame:
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

        # Never stake on incomplete-feature matches. Rows whose features were
        # defaulted/missing (e.g. matchup not yet in Tennis Abstract, unresolved
        # round) are still logged for the record + settlement, but must not
        # produce a bet — we only bet matches with complete, real features.
        if '_has_defaulted_features' in successful_predictions.columns:
            n_before = len(successful_predictions)
            successful_predictions = successful_predictions[
                ~successful_predictions['_has_defaulted_features'].astype(bool)
            ].copy()
            n_excluded = n_before - len(successful_predictions)
            if n_excluded:
                print(f"   🚫 Excluded {n_excluded} incomplete-feature match(es) from betting "
                      f"(logged for the record, not staked)")
            if successful_predictions.empty:
                print("📊 No complete-feature matches available to bet")
                return pd.DataFrame()

        # Calculate edges
        edges_df = calculate_betting_edges(successful_predictions, odds_df)
        
        # Filter betting opportunities
        opportunities_df = self.calculator.filter_betting_opportunities(edges_df)
        
        if opportunities_df.empty:
            print("📊 No profitable betting opportunities found")
            return pd.DataFrame()
        
        # Calculate stakes
        bankroll = float(
            bankroll if bankroll is not None else self.bet_tracker.get_current_bankroll()
        )
        available_bankroll = float(
            available_bankroll
            if available_bankroll is not None
            else self.bet_tracker.get_available_bankroll()
        )
        if available_bankroll < self.config.get('min_stake_dollars', 1.0):
            print(f"🛑 Portfolio exposure gate: ${available_bankroll:.2f} unreserved; no new bets")
            return pd.DataFrame()
        stakes_df = self.calculator.allocate_block_stakes(
            opportunities_df,
            bankroll,
            available_bankroll=available_bankroll,
        )
        
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
        bankroll = self.bet_tracker.get_current_bankroll()
        
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

                # features_complete: False if the calculator reported any noisy defaults.
                # Still log these rows below; analysis excludes incomplete-feature rows
                # from clean accuracy, but operational settlement needs the prediction.
                features_complete = not bool(pred_row.get('_has_defaulted_features', False))
                if not features_complete:
                    defaulted = pred_row.get('meta_defaulted_features', '')
                    print(f"  ⚠️ Logging prediction for {p1} vs {p2} with incomplete features: {defaulted}")

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
                    mkt_p1_raw = pd.to_numeric(o_row.get('player1_implied_prob'), errors='coerce')
                    mkt_p2_raw = pd.to_numeric(o_row.get('player2_implied_prob'), errors='coerce')
                    # De-vig: normalize so probs sum to 1.0
                    mkt_total = mkt_p1_raw + mkt_p2_raw
                    if pd.notna(mkt_p1_raw) and pd.notna(mkt_p2_raw) and mkt_total > 0:
                        mkt_p1 = float(mkt_p1_raw / mkt_total)
                        mkt_p2 = float(mkt_p2_raw / mkt_total)
                    else:
                        mkt_p1 = mkt_p2 = None
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
                    # Missing market evidence is null, never a synthetic 50/50.
                    # A real even-money price remains a valid observed 0.5.
                    mkt_p1, mkt_p2, o1, o2 = None, None, None, None
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
                    match_start_at_utc=self.match_start_at_utc(match_time),
                    features_complete=features_complete,
                    defaulted_features=pred_row.get('meta_defaulted_features', ''),
                    feature_schema_sha256=pred_row.get('feature_schema_sha256', ''),
                    feature_vector_sha256=pred_row.get('feature_vector_sha256', ''),
                    # data-quality caveats surfaced on the slate (still bettable,
                    # but visible): handedness and unranked status are honest
                    # values the model uses, not defaults — the user should see them
                    p1_hand=_hand_of(pred_row, 'P1'), p2_hand=_hand_of(pred_row, 'P2'),
                )
                if action == 'created':
                    stats['created'] += 1
                elif action == 'updated':
                    stats['updated'] += 1
        except Exception as e:
            required = os.environ.get('REQUIRE_DURABLE_STATE') == '1'
            print(f"  ⚠️ Prediction logging failed ({'fatal' if required else 'non-fatal'}): {e}")
            import traceback; traceback.print_exc()
            if required:
                raise RuntimeError(f"prediction lineage logging failed: {e}") from e
        return stats

    def _find_odds_row(self, p1: str, p2: str, odds_df: pd.DataFrame):
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
        return None if match_odds.empty else match_odds.iloc[0]

    def _log_performance_shadow_predictions(self, predictions_df: pd.DataFrame, odds_df: pd.DataFrame):
        """Run and log performance_v1 shadow predictions without affecting bets."""
        stats = {'attempts': 0, 'logged': 0, 'models_loaded': 0}
        if not self.performance_shadow_enabled:
            return stats
        if predictions_df.empty:
            return stats
        if not self.performance_shadow_predictor.is_loaded and not self.performance_shadow_predictor.load_model():
            return stats
        stats['models_loaded'] = len(self.performance_shadow_predictor.loaded_predictors)

        shadow_rows = []
        for _, pred_row in predictions_df.iterrows():
            if pred_row.get('prediction_status') != 'success':
                continue
            if not bool(pred_row.get('performance_v1_features_available', False)):
                continue
            p1 = pred_row.get('player1_normalized') or pred_row.get('player1_raw', '')
            p2 = pred_row.get('player2_normalized') or pred_row.get('player2_raw', '')
            odds_row = self._find_odds_row(p1, p2, odds_df)
            for predictor, result in self.performance_shadow_predictor.predict_match_probabilities(pred_row.to_dict()):
                stats['attempts'] += 1
                shadow_rows.append(
                    shadow_row_from_prediction(
                        pred_row,
                        odds_row,
                        result,
                        model_version=predictor.model_version,
                        model_family=predictor.family,
                        feature_set=predictor.feature_set,
                        n_features=len(predictor.feature_names),
                    )
                )

        if shadow_rows:
            path = self.logs_dir / "performance_v1_shadow_predictions.csv"
            logged = log_shadow_predictions(path, shadow_rows)
            stats['logged'] = logged
            print(
                f"🧪 performance_v1 shadow: {logged}/{len(shadow_rows)} rows logged "
                f"across {stats['models_loaded']} model(s) to {path}"
            )
        return stats

    def _refresh_atp_rankings(self):
        """Refresh ATP rankings (rank + points), falling back to cache — never silently to 500."""
        print("📊 Refreshing ATP rankings...")
        df, source = resolve_rankings(headless=True)
        if source == "fresh":
            save_rankings(df)
            self.feature_engine._atp_rankings = df
            print(f"  ✅ Loaded {len(df)} ranked players (fresh scrape)")
        elif source.startswith("cached"):
            # Live scrape failed but cached rankings exist — use them rather than
            # degrading every Rank_Points to 500. The cache date is surfaced so a
            # stale cache is visible.
            self.feature_engine._atp_rankings = df
            print(f"  ↩️  Live scrape unavailable; using {source} ({len(df)} players). "
                  f"Run scraping/atp_rankings_scraper.py to refresh the cache.")
        else:
            print("  ⚠️  No live or cached rankings available — Rank_Points will default to 500 this run")

    def _refresh_database(self):
        """Refresh local SQL and publish the durable remote state generation.

        SQLite is a local convenience and remains non-fatal. In cloud operation
        ``REQUIRE_DURABLE_STATE=1`` makes remote publication a correctness gate:
        Actions must not report success after losing the runner's state.
        """
        try:
            import db as _db
            prod_dir = os.path.dirname(os.path.abspath(__file__))
            db_path = os.path.join(prod_dir, "logs", "betting.db")
            summary = _db.build_database(prod_dir, db_path)
            print(f"  🗄️  SQLite refreshed: {summary.get('predictions', 0)} predictions, "
                  f"{summary.get('shadow_predictions', 0)} shadow rows → logs/betting.db")
        except Exception as e:
            print(f"  ⚠️  SQLite refresh skipped (non-fatal): {e}")

        try:
            from dashboard_sync import sync_dashboard_tables
            sync_dashboard_tables()
        except Exception as dash_exc:
            required = os.environ.get('REQUIRE_DURABLE_STATE') == '1'
            severity = "fatal" if required else "non-fatal"
            print(f"  ⚠️ Durable state sync failed ({severity}): {dash_exc}")
            if required:
                raise RuntimeError(f"durable state publication failed: {dash_exc}") from dash_exc

    def _ingest_store_events(self):
        """Append this run's fetched event results into the canonical Supabase
        store (additive, non-fatal — skipped offline or without .env.supabase)."""
        cache = getattr(self, "_session_cache", None) or {}
        frames = cache.get("atp_event_results") or {}
        metas = cache.get("atp_event_meta") or {}
        itf_frames = cache.get("itf_event_matches") or {}
        itf_cal = cache.get("itf_calendar")
        if not frames and not itf_frames:
            self.run_metrics['canonical_ingest_status'] = 'no_event_frames'
            self.run_metrics['reconcile_status'] = 'not_attempted'
            return
        try:
            import canonical_store as cs
            with cs.connect() as conn:
                total = 0
                for url, df in frames.items():
                    ev = metas.get(url)
                    if ev is None or df is None or df.empty:
                        continue
                    # per-event transaction: one bad event must not abort the rest
                    with conn.transaction():
                        r = cs.ingest_event_results(
                            conn, df, event=ev["event"], start_date=ev["start_date"],
                            surface=ev.get("surface") or None, level=ev.get("level") or None,
                            tourney_id=ev.get("id"),
                        )
                    total += r["inserted"]
                # ITF events fetched this run (rounds/history) carry completed
                # results too — persist them so ITF histories accumulate weekly
                # instead of freezing at Sackmann's last drop
                if itf_frames and itf_cal is not None and not itf_cal.empty:
                    for key, em in itf_frames.items():
                        if em is None or em.empty:
                            continue
                        evrow = itf_cal[itf_cal["key"] == key]
                        if evrow.empty:
                            continue
                        ev = evrow.iloc[0]
                        with conn.transaction():
                            r = cs.ingest_itf_results(
                                conn, em, event=str(ev["event"]), start_date=str(ev["start_date"]),
                                surface=(str(ev.get("surface")) or None),
                                level="25" if "25" in str(ev.get("category", "")) else "15",
                                tourney_id=str(ev.get("key") or "") or None,
                            )
                        total += r["inserted"]
                print(f"  🗄️  Canonical store: +{total} event-result rows ingested")
                self.run_metrics['canonical_ingest_status'] = 'success'
                self.run_metrics['canonical_ingest_rows'] = total
                # cross-source reconciliation: conflicts loud, curated-level
                # repair, capped stats gap-fill for the active registry events
                try:
                    import reconcile_store as rcs
                    from features.history_stitch import CURRENT_EVENT_REGISTRY
                    import pandas as _pd
                    urls = [ev["url"] for ev in CURRENT_EVENT_REGISTRY
                            if _pd.Timestamp(ev["window"][0]) <= _pd.Timestamp.now() <= _pd.Timestamp(ev["window"][1])]
                    rcs.run(conn=conn, since_days=75, stats_urls=urls, stats_cap=10)
                    self.run_metrics['reconcile_status'] = 'success'
                except Exception as _rc_exc:
                    self.run_metrics['reconcile_status'] = 'error'
                    self.run_metrics['reconcile_error'] = str(_rc_exc)
                    print(f"  ⚠️  reconcile skipped (non-fatal): {_rc_exc}")
        except Exception as e:
            self.run_metrics['canonical_ingest_status'] = 'error'
            self.run_metrics['canonical_ingest_error'] = str(e)
            if self.run_metrics.get('reconcile_status') == 'not_started':
                self.run_metrics['reconcile_status'] = 'not_attempted'
            print(f"  ⚠️  Canonical store ingest skipped (non-fatal): {e}")

    def _stage_adjusted_terminal_status(self, status: str) -> str:
        if status != 'success':
            return status
        error_fields = (
            self.run_metrics.get('auto_settle_status'),
            self.run_metrics.get('canonical_ingest_status'),
            self.run_metrics.get('reconcile_status'),
        )
        if 'error' in error_fields:
            return 'partial'
        if self.run_metrics.get('exposure_gate_status') == 'blocked_pending_exposure':
            return 'partial'
        return status

    def _previous_run_status(self) -> str:
        """Status of the run before this one, from the Supabase run mirror
        (visible even for runs whose git log-commit was skipped)."""
        try:
            import canonical_store as cs
            with cs.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """SELECT status FROM dash_runs
                           WHERE run_id <> %s ORDER BY started_at DESC LIMIT 1""",
                        (self.run_metrics.get('run_id', ''),),
                    )
                    row = cur.fetchone()
                    return str(row[0]) if row else ''
        except Exception as exc:
            print(f"  ⚠️ previous-run status unavailable (non-fatal): {exc}")
            return ''

    def run_full_pipeline(self, start_session: bool = True, auto_settle: bool = True, dry_run: bool = False) -> bool:
        """Run the complete betting pipeline with bet tracking"""
        session_id = None
        try:
            self._start_run_context()
            self.run_metrics['auto_settle_enabled'] = bool(auto_settle)
            self.run_metrics['rankings_refresh_enabled'] = bool(self.rankings_refresh_enabled)
            print("🚀 Starting live tennis betting pipeline...")
            kelly_mult = self.config.get('kelly_multiplier', 0.18)
            configured_bankroll = self.config.get('bankroll', 1000.0)
            bankroll = self.bet_tracker.get_current_bankroll()
            pending_exposure = self.bet_tracker.get_pending_exposure()
            available_bankroll = self.bet_tracker.get_available_bankroll()
            
            print(f"⚙️  Kelly multiplier: {kelly_mult:.1%}")
            print(f"⚙️  Edge threshold: {self.config.get('edge_threshold', 0.02):.1%}")
            print(f"⚙️  Account equity: ${bankroll:.2f} "
                  f"(configured start ${configured_bankroll:.2f})")
            print(f"⚙️  Pending exposure: ${pending_exposure:.2f}; "
                  f"available: ${available_bankroll:.2f}")
            if dry_run:
                print("🧪 Dry run: no betting session or tracked bets will be written")
            
            # Step 0a: Auto-settle any pending predictions with known results
            if auto_settle:
                try:
                    from auto_settle import run as auto_settle_run
                    print("\n📋 Auto-settling pending predictions...")
                    settle_summary = auto_settle_run(dry_run=dry_run, run_id=self.run_id, record_run_history=False)
                    if isinstance(settle_summary, dict):
                        self.run_metrics['settlement_candidates'] = settle_summary.get('settlement_candidates', 0)
                        self.run_metrics['settlement_newly_settled'] = settle_summary.get('settlement_newly_settled', 0)
                        self.run_metrics['settlement_auto_settled_bets'] = settle_summary.get('settlement_auto_settled_bets', 0)
                        self.run_metrics['settlement_reason_summary'] = settle_summary.get('settlement_reason_summary', {})
                    self.run_metrics['auto_settle_status'] = 'success'
                except Exception as e:
                    print(f"  ⚠️  Auto-settle failed (non-fatal): {e}")
                    self.run_metrics['settlement_reason_summary'] = {'auto_settle_error': 1}
                    self.run_metrics['auto_settle_status'] = 'error'
                    self.run_metrics['auto_settle_error'] = str(e)
            else:
                print("\n⏭️  Auto-settle skipped for this run")
                self.run_metrics['auto_settle_status'] = 'skipped'

            # Settlement may release exposure or change account equity. All
            # sizing below uses this reconciled account state, never a fresh
            # $1,000 session reset.
            bankroll = self.bet_tracker.get_current_bankroll()
            pending_exposure = self.bet_tracker.get_pending_exposure()
            available_bankroll = self.bet_tracker.get_available_bankroll()
            self.run_metrics['account_equity'] = bankroll
            self.run_metrics['pending_exposure'] = pending_exposure
            self.run_metrics['available_bankroll'] = available_bankroll
            self.run_metrics['exposure_gate_status'] = (
                'blocked_pending_exposure'
                if available_bankroll < self.config.get('min_stake_dollars', 1.0)
                else 'open'
            )

            # Settlement is durable before rankings/scraping begins.  A later
            # cancellation or source timeout must not erase completed work.
            if not dry_run:
                self._persist_run_state()

            # Step 0b: Refresh ATP rankings (rank + points)
            self._refresh_atp_rankings()

            # Step 1: Fetch odds
            odds_df = self.fetch_odds()
            if odds_df.empty:
                # settlement already ran; persist its work first, THEN decide
                # health: a parsed-but-empty board is a genuine no-odds hour;
                # a fetch crash is tolerated once (transient) and hard-fails
                # on the second consecutive occurrence so real breakage alerts.
                if not getattr(self, '_odds_fetch_failed', False):
                    print("⚠️  Board parsed but empty — closing out as a no-odds hour")
                    self._ingest_store_events()
                    self._persist_run_state(status='no_odds')
                    return True
                self._ingest_store_events()
                self._persist_run_state(status='odds_fetch_error')
                if self._previous_run_status() == 'odds_fetch_error':
                    print("💥 Second consecutive odds-fetch failure — failing loudly")
                    return False
                print("⚠️  Odds fetch failed once — settlement work persisted; will alert if it repeats")
                return True
            
            # Step 1.5: warm event-page caches in parallel (thread-local
            # browsers) — week-boundary runs discover ~15 fresh tournaments and
            # sequential first-touch fetches were the 40-minute Sundays
            try:
                from prefetch import prefetch_event_pages
                prefetch_event_pages(self._session_cache)
            except Exception as _pf_exc:
                print(f"  ⚠️ prefetch skipped (non-fatal): {_pf_exc}")

            # Step 2: Extract features
            features_df = self.extract_features(odds_df)
            if features_df.empty:
                print("⚠️  Feature extraction failed, stopping pipeline")
                self._ingest_store_events()
                self._persist_run_state(status='no_features')
                return False
            
            # Step 3: Generate predictions
            predictions_df = self.generate_predictions(features_df)
            self.run_metrics['prediction_rows_total'] = len(predictions_df)
            terminal_status, success_count, error_count = prediction_terminal_status(predictions_df)
            self.run_metrics['prediction_rows_success'] = success_count
            self.run_metrics['prediction_rows_error'] = error_count
            if terminal_status == 'no_predictions':
                print("⚠️  Prediction generation failed, stopping pipeline")
                self._ingest_store_events()
                self._persist_run_state(status='no_predictions')
                return False
            
            # Step 4: Calculate stakes
            bet_slips_df = self.calculate_edges_and_stakes(
                predictions_df,
                odds_df,
                bankroll=bankroll,
                available_bankroll=available_bankroll,
            )

            # Step 4b: Log all predictions (regardless of edge threshold)
            prediction_log_stats = self._log_all_predictions(predictions_df, odds_df, features_df)
            self.run_metrics['prediction_log_attempts'] = prediction_log_stats.get('attempts', 0)
            self.run_metrics['prediction_log_created'] = prediction_log_stats.get('created', 0)
            self.run_metrics['prediction_log_updated'] = prediction_log_stats.get('updated', 0)
            self.run_metrics['prediction_log_skipped_incomplete'] = prediction_log_stats.get('skipped_incomplete', 0)
            shadow_stats = self._log_performance_shadow_predictions(predictions_df, odds_df)
            self.run_metrics['performance_shadow_attempts'] = shadow_stats.get('attempts', 0)
            self.run_metrics['performance_shadow_logged'] = shadow_stats.get('logged', 0)
            self.run_metrics['performance_shadow_models_loaded'] = shadow_stats.get('models_loaded', 0)
            self.run_metrics['bet_opportunities'] = len(bet_slips_df)

            # Step 5: Save, log, and display results
            if not bet_slips_df.empty:
                self.save_bet_slips(bet_slips_df)

                # A session represents actual paper exposure, not merely a
                # pipeline attempt.  Create it only when bets will be logged.
                if start_session and not dry_run:
                    session_id = self.bet_tracker.start_session(
                        bankroll, kelly_mult,
                        f"Auto session {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                    )
                
                # Log bets to tracking system
                if session_id:
                    self.run_metrics['bets_logged'] = self.bet_tracker.log_bets(bet_slips_df, session_id, bankroll)
                    if self.run_metrics['bets_logged'] == 0:
                        self.bet_tracker.discard_empty_session(session_id)
                        session_id = None

                # Placement changes reserved capital, not account equity. Make
                # the terminal run row reflect the post-placement state that
                # the next hourly run will inherit.
                self.run_metrics['account_equity'] = self.bet_tracker.get_current_bankroll()
                self.run_metrics['pending_exposure'] = self.bet_tracker.get_pending_exposure()
                self.run_metrics['available_bankroll'] = self.bet_tracker.get_available_bankroll()
                self.run_metrics['exposure_gate_status'] = (
                    'blocked_pending_exposure'
                    if self.run_metrics['available_bankroll']
                    < self.config.get('min_stake_dollars', 1.0)
                    else 'open'
                )
                
                self.print_summary(bet_slips_df)
                
                # Show tracking info
                if session_id:
                    pending_bets = self.bet_tracker.get_pending_bets()
                    print(f"\n📊 Session tracking:")
                    print(f"   Session ID: {session_id}")
                    print(f"   Total pending bets: {len(pending_bets)}")
                    print(f"   Use 'python settle_bets.py {session_id}' to settle results later")
                
                self._ingest_store_events()
                self._persist_run_state(
                    status=self._stage_adjusted_terminal_status(terminal_status)
                )
                return True
            else:
                print("📊 No profitable betting opportunities found")
                self._ingest_store_events()
                self._persist_run_state(
                    status=self._stage_adjusted_terminal_status(terminal_status)
                )
                # a no-bets hour is a HEALTHY run — exit 0 or the cloud job
                # reports failure and skips committing this run's logs
                return True
                
        except Exception as e:
            print(f"💥 Pipeline error: {e}")
            import traceback
            traceback.print_exc()
            try:
                self._persist_run_state(status='failed', error_message=str(e))
            except Exception as persist_exc:
                # Preserve the terminal row locally for the always-run git
                # checkpoint even when the remote store itself is unavailable.
                self._flush_run_history(
                    status='failed',
                    error_message=f"{e}; terminal publication failed: {persist_exc}",
                )
                print(f"💥 Terminal durable-state publication also failed: {persist_exc}")
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
    if args.bankroll is not None:
        orchestrator.config['bankroll'] = args.bankroll
        orchestrator.bet_tracker.initial_bankroll = float(args.bankroll)
    if args.kelly_multiplier is not None:
        orchestrator.config['kelly_multiplier'] = args.kelly_multiplier
        orchestrator.calculator.kelly_multiplier = float(args.kelly_multiplier)
    if args.edge_threshold is not None:
        orchestrator.config['edge_threshold'] = args.edge_threshold
        orchestrator.calculator.edge_threshold = float(args.edge_threshold)
    # Patch refresh flag onto orchestrator
    if args.skip_rankings_refresh:
        orchestrator.rankings_refresh_enabled = False
        orchestrator._refresh_atp_rankings = lambda: print("📊 ATP rankings refresh skipped (--skip-rankings-refresh)")

    # Run pipeline
    success = orchestrator.run_full_pipeline(
        start_session=not args.dry_run,
        auto_settle=not args.skip_auto_settle,
        dry_run=args.dry_run,
    )
    
    if args.dry_run:
        print("\n🧪 DRY RUN - No bets would actually be placed")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
