#!/usr/bin/env python3
"""
Tennis Abstract Feature Calculator
Mirrors LiveFeatureEngine logic but uses Tennis Abstract as data source.
Returns the exact 143 features expected by NN-143 model.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re
import sys

# Import TA scraper
sys.path.insert(0, str(Path(__file__).parent.parent / "scraping"))
from ta_scraper import TennisAbstractScraper

# Import schema contract
import json
SCHEMA_PATH = Path(__file__).parent / "schema_143.json"
with open(SCHEMA_PATH) as f:
    SCHEMA = json.load(f)
    EXACT_143_FEATURES = [feat["name"] for feat in SCHEMA["features"]]


class TAFeatureCalculator:
    """
    Calculate the exact 143 features from Tennis Abstract data.
    Mirrors LiveFeatureEngine but uses TA as primary data source.
    """

    def __init__(self, ta_scraper: Optional[TennisAbstractScraper] = None):
        self.scraper = ta_scraper or TennisAbstractScraper()
        self.player_slug_map = self._load_player_mapping()

    def _load_player_mapping(self) -> Dict[str, str]:
        """Load player name -> TA slug mapping from CSV"""
        mapping_file = Path(__file__).parent.parent / "ta_player_mapping.csv"
        if not mapping_file.exists():
            return {}

        df = pd.read_csv(mapping_file)
        name_to_slug = {}

        for _, row in df.iterrows():
            slug = str(row.get('ta_slug', '')).strip()
            primary = str(row.get('primary_name', '')).strip()
            bovada = str(row.get('bovada_name', '')).strip()
            variants = str(row.get('name_variants', ''))

            if slug:
                if primary:
                    name_to_slug[self._norm(primary)] = slug
                if bovada:
                    name_to_slug[self._norm(bovada)] = slug
                if variants:
                    for v in variants.split('|'):
                        v = v.strip()
                        if v:
                            name_to_slug[self._norm(v)] = slug

        return name_to_slug

    @staticmethod
    def _norm(s: str) -> str:
        """Normalize name for fuzzy matching"""
        s = (s or "").strip().lower()
        try:
            import unicodedata
            s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        except Exception:
            pass
        s = re.sub(r'[^a-z\s\-]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    def find_slug(self, player_name: str) -> Optional[str]:
        """
        Find TA slug for player name.
        Priority: 1) explicit mapping CSV  2) CamelCase derivation  3) TA HTTP search
        """
        normalized = self._norm(player_name)
        if normalized in self.player_slug_map:
            return self.player_slug_map[normalized]

        # Derive slug from name (works for most players without a network request)
        derived = self.scraper.name_to_slug(player_name)
        # Cache and return derived slug — the caller will validate via HTTP when fetching profile
        if derived:
            self.player_slug_map[normalized] = derived
            return derived

        # Last resort: try HTTP search
        slug = self.scraper.search_player(player_name)
        if slug:
            self.player_slug_map[normalized] = slug
        return slug

    # ========== Laplace-smoothed win rate helper ==========

    @staticmethod
    def _laplace(wins: int, total: int, alpha: float = 3.0) -> float:
        """
        Bayesian (Laplace) smoothed win rate.
        Formula: (wins + alpha/2) / (total + alpha)
        Prior is 0.5 (neutral); alpha controls regularization strength.
        alpha=3 → need ~3 observations to trust data 50% over prior.

        Replaces both min_n threshold AND hard 0.5 default with a
        continuous, well-behaved estimate. Requires retraining to take
        full effect in the model — applied consistently with preprocess.py.

        Examples (alpha=3):
          0/0  → 0.500   1/1  → 0.625   2/2  → 0.700
          3/3  → 0.750   0/3  → 0.250  10/15 → 0.639
        """
        return (wins + alpha / 2.0) / (total + alpha)

    # ========== Temporal Features from Match History ==========

    @staticmethod
    def _count_period(df: pd.DataFrame, ref: datetime, days: int) -> int:
        """Count matches in time window"""
        if df.empty or 'date' not in df.columns:
            return 0
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        return int((df['date'] >= cut).sum())

    @staticmethod
    def _surface_mask(df: pd.DataFrame, surface: str) -> pd.Series:
        """Boolean mask for surface matches"""
        if 'surface' not in df.columns:
            return pd.Series(False, index=df.index)
        return df['surface'].astype(str).str.lower() == (surface or '').lower()

    def _count_surface(self, df: pd.DataFrame, ref: datetime, surface: str, days: int) -> int:
        """Count surface-specific matches in time window"""
        if df.empty:
            return 0
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = (df['date'] >= cut) & self._surface_mask(df, surface)
        return int(m.sum())

    def _surface_winrate(self, df: pd.DataFrame, ref: datetime, surface: str, days: int) -> float:
        """Surface-specific win rate — Laplace smoothed (alpha=3)."""
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = df[(df['date'] >= cut) & self._surface_mask(df, surface)]
        wins = (m['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(m))

    def _winrate_lastN_within(self, df: pd.DataFrame, ref: datetime, N: int, days: int) -> float:
        """Last N matches within window win rate — Laplace smoothed (alpha=3)."""
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = df[df['date'] >= cut].head(N)
        wins = (m['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(m))

    def _form_trend_ewm(self, df: pd.DataFrame, ref: datetime, days: int = 30) -> float:
        """
        Exponentially weighted moving average form trend — mirrors training.
        Half-life 15 days. Requires >=3 matches, else returns 0.5.
        """
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        m = df[(df['date'] >= cut) & (df['date'] < ref)]
        if len(m) < 3:
            return 0.5
        wins_w, total_w = 0.0, 0.0
        for _, r in m.iterrows():
            days_ago = (ref - r['date']).days
            w = float(np.exp(-days_ago / 15.0))
            total_w += w
            if str(r.get('result', '')).upper() == 'W':
                wins_w += w
        return wins_w / total_w if total_w > 0 else 0.5

    @staticmethod
    def _streak(df: pd.DataFrame) -> int:
        """
        Current win/loss streak — mirrors training preprocess.py.
        Positive = win streak (+N), negative = loss streak (-N), 0 = no history.
        df must be sorted most-recent first.
        """
        if df.empty:
            return 0
        results = df['result'].astype(str).str.upper().tolist()
        first = results[0]
        if first not in ('W', 'L'):
            return 0
        s = 0
        for r in results:
            if r == first:
                s += 1
            else:
                break
        return s if first == 'W' else -s

    @staticmethod
    def _monday_of(dt: pd.Timestamp) -> pd.Timestamp:
        """Get Monday of week"""
        return dt - pd.Timedelta(days=int(dt.weekday()))

    def _days_since_last_tournament(self, df: pd.DataFrame, ref: datetime) -> Optional[int]:
        """Days since last tournament (week-based)"""
        if df.empty or 'date' not in df.columns:
            return None
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        d = df[df['date'] < ref].copy()
        if d.empty:
            return None
        d['week_monday'] = d['date'].apply(lambda x: self._monday_of(pd.Timestamp(x)))
        current_week = self._monday_of(pd.Timestamp(ref))
        prev_weeks = d[d['week_monday'] < current_week]['week_monday']
        if prev_weeks.empty:
            return None
        last_week = prev_weeks.max()
        return int((current_week - last_week).days)

    @staticmethod
    def _count_sets_from_score(score: str) -> int:
        """Count number of sets from score string (e.g., '6-3 6-4' → 2 sets)"""
        if not score or pd.isna(score):
            return 0
        # Score format: "6-3 6-4" or "6-4 6-7(3) 6-1"
        # Count space-separated set scores
        sets = [s.strip() for s in str(score).split() if s.strip()]
        return len(sets)

    def _sets_14d(self, df: pd.DataFrame, ref: datetime) -> int:
        """Count actual sets played in last 14 days from scores"""
        if df.empty or 'date' not in df.columns:
            return 0
        cut = ref - timedelta(days=14)
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        recent = df[df['date'] >= cut]

        if recent.empty:
            return 0

        # Count actual sets from scores if available
        if 'score' in recent.columns:
            total_sets = recent['score'].apply(self._count_sets_from_score).sum()
            return int(total_sets)

        # Fallback to estimate if no scores
        return int(len(recent) * 2.5)

    # ========== Level & Round Stats from TA Match History ==========

    @staticmethod
    def _level_code(level: str) -> Optional[str]:
        """Normalize level to Sackmann code"""
        s = (level or "").strip().upper()
        if s in {"A", "ATP", "ATP 250", "ATP 500"}: return "A"
        if s in {"M", "MASTERS", "MASTERS 1000"}: return "M"
        if s in {"G", "GRAND SLAM", "SLAM"}: return "G"
        if s in {"C", "CHALLENGER"}: return "C"
        if s in {"25", "ITF M25", "M25", "25K"}: return "25"
        if s in {"15", "ITF M15", "M15", "15K"}: return "15"
        if s in {"F", "ATP FINALS"}: return "F"
        if s in {"S", "ITF", "FUTURES"}: return "S"
        return None

    def _level_stats(self, df: pd.DataFrame, level_code: Optional[str]) -> Tuple[float, int]:
        """Win rate and match count at specific level — Laplace smoothed (alpha=3)."""
        if df.empty or not level_code or 'level' not in df.columns:
            return (0.5, 0)
        sub = df[df['level'].astype(str).str.upper() == level_code]
        total = len(sub)
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return (self._laplace(int(wins), total), total)

    def _round_winrate(self, df: pd.DataFrame, round_code: Optional[str]) -> float:
        """Win rate at specific round — Laplace smoothed (alpha=3)."""
        if not round_code or df.empty or 'round' not in df.columns:
            return 0.5
        sub = df[df['round'].astype(str).str.upper() == str(round_code).upper()]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    def _semis_finals_winrates(self, df: pd.DataFrame) -> Tuple[float, float]:
        """Win rates in SF and F — Laplace smoothed (alpha=3)."""
        if df.empty or 'round' not in df.columns:
            return (0.5, 0.5)

        def wr(code):
            sub = df[df['round'].astype(str).str.upper() == code]
            wins = (sub['result'].astype(str).str.upper() == 'W').sum()
            return self._laplace(int(wins), len(sub))

        return (wr('SF'), wr('F'))

    def _big_match_wr(self, df: pd.DataFrame) -> float:
        """Big match win rate (Grand Slams + Masters pooled) — Laplace smoothed (alpha=3)."""
        if df.empty or 'level' not in df.columns:
            return 0.5
        sub = df[df['level'].astype(str).str.upper().isin(['G', 'M'])]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    # ========== Rank Features from TA Match History ==========

    def _rank_change(self, df: pd.DataFrame, ref_date: datetime, days: int) -> float:
        """Rank change over time window (positive = improved)"""
        if df.empty or 'rank' not in df.columns or 'date' not in df.columns:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

        past = df[df['date'] < ref_date].sort_values('date')
        if past.empty:
            return 0.0

        t_now = past.iloc[-1]
        t_then_cut = ref_date - timedelta(days=days)
        past_then = past[past['date'] < t_then_cut]
        if past_then.empty:
            return 0.0
        t_then = past_then.iloc[-1]

        r_now = t_now.get('rank')
        r_then = t_then.get('rank')
        if pd.notna(r_now) and pd.notna(r_then):
            return float(r_then) - float(r_now)
        return 0.0

    def _rank_volatility(self, df: pd.DataFrame, ref_date: datetime, days: int) -> float:
        """Standard deviation of rank over time window"""
        if df.empty or 'rank' not in df.columns or 'date' not in df.columns:
            return 0.0

        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['rank'] = pd.to_numeric(df['rank'], errors='coerce')

        cut = ref_date - timedelta(days=days)
        window = df[(df['date'] < ref_date) & (df['date'] >= cut)]
        if window.empty:
            return 0.0

        ranks = window['rank'].dropna()
        if ranks.empty:
            return 0.0
        return float(ranks.std())

    # ========== Opponent-Specific Features ==========

    def _winrate_vs_hand(self, df: pd.DataFrame, opp_hand: str) -> float:
        """Win rate vs specific opponent handedness — Laplace smoothed (alpha=3)."""
        if df.empty or 'opp_hand' not in df.columns:
            return 0.5
        sub = df[df['opp_hand'].astype(str).str.upper() == opp_hand.upper()]
        wins = (sub['result'].astype(str).str.upper() == 'W').sum()
        return self._laplace(int(wins), len(sub))

    # ========== H2H from Tennis Abstract H2H Page ==========

    def _get_h2h_stats(self, slug1: str, slug2: str, session_cache: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get H2H stats from Tennis Abstract H2H page.
        For now, calculate from match histories until H2H page parser is added.
        """
        # Load both players' FULL CAREER matches for H2H (years=[] means all years)
        matches1 = self.scraper.get_player_matches(slug1, years=[], force_refresh=False, persist=False, session_cache=session_cache)
        matches2 = self.scraper.get_player_matches(slug2, years=[], force_refresh=False, persist=False, session_cache=session_cache)

        if matches1.empty and matches2.empty:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }

        # Filter for matches against each other (by opponent name matching)
        # This is approximate - ideally we'd parse the dedicated H2H page
        profile1 = self.scraper.get_player_profile(slug1, force_refresh=False, persist=False, session_cache=session_cache)
        profile2 = self.scraper.get_player_profile(slug2, force_refresh=False, persist=False, session_cache=session_cache)

        if not profile1 or not profile2:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }

        name1 = profile1.get('name', '')
        name2 = profile2.get('name', '')

        # Get H2H matches from player1's perspective only
        # (Don't concatenate both perspectives - that counts each match twice!)
        if not matches1.empty and 'opp_name' in matches1.columns:
            h2h = matches1[matches1['opp_name'].str.contains(name2, case=False, na=False)].copy()
        else:
            h2h = pd.DataFrame()

        if h2h.empty:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }
        if 'date' in h2h.columns:
            h2h['date'] = pd.to_datetime(h2h['date'], errors='coerce')
            h2h = h2h.sort_values('date', ascending=False)

        total = len(h2h)
        p1_wins = (h2h['result'].astype(str).str.upper() == 'W').sum()
        p2_wins = total - p1_wins

        # Recent advantage (last 3 meetings) — Laplace smoothed, centered at 0
        recent = h2h.head(3)
        recent_wins = (recent['result'].astype(str).str.upper() == 'W').sum()
        adv = float(self._laplace(int(recent_wins), len(recent)) - 0.5) if len(recent) >= 2 else 0.0

        # Career H2H win rate — Laplace smoothed
        wr = self._laplace(int(p1_wins), total)

        return {
            'H2H_Total_Matches': int(total),
            'H2H_P1_Wins': int(p1_wins),
            'H2H_P2_Wins': int(p2_wins),
            'H2H_P1_WinRate': float(wr),
            'H2H_Recent_P1_Advantage': float(adv)
        }

    # ========== One-Hot Encodings ==========

    @staticmethod
    def _season_flags(surface: str, when: datetime) -> Dict[str, int]:
        """Seasonal flags based on month"""
        m = when.month
        return {
            'Clay_Season': 1 if 4 <= m <= 6 else 0,
            'Grass_Season': 1 if m in (6, 7) else 0,
            'Indoor_Season': 1 if (m >= 10 or m <= 2) else 0
        }

    @staticmethod
    def _levels_onehot(level: str) -> Dict[str, int]:
        """One-hot encoding for tournament level"""
        s = (level or "").strip().upper()

        # Normalize text variants to codes
        if s in {"GRAND SLAM", "SLAM"}: s = "G"
        elif s in {"MASTERS 1000", "MASTERS"}: s = "M"
        elif s in {"ATP", "ATP 250", "ATP 500"}: s = "A"
        elif s in {"CHALLENGER"}: s = "C"
        elif s in {"ITF", "FUTURES"}: s = "S"
        elif s in {"ITF M25", "M25", "25K", "25"}: s = "25"
        elif s in {"ITF M15", "M15", "15K", "15"}: s = "15"
        elif s in {"ATP FINALS", "NITTO ATP FINALS"}: s = "F"

        return {
            "Level_G": 1 if s == "G" else 0,
            "Level_M": 1 if s == "M" else 0,
            "Level_A": 1 if s == "A" else 0,
            "Level_C": 1 if s == "C" else 0,
            "Level_S": 1 if s == "S" else 0,
            "Level_F": 1 if s == "F" else 0,
            "Level_25": 1 if s == "25" else 0,
            "Level_15": 1 if s == "15" else 0,
            "Level_O": 1 if s == "O" else 0,
            "Level_D": 1 if s == "D" else 0,
        }

    @staticmethod
    def _rounds_onehot(round_code: Optional[str]) -> Dict[str, int]:
        """One-hot encoding for round"""
        rc = (round_code or '').upper()
        keys = ['R128', 'R64', 'R32', 'R16', 'Q1', 'Q2', 'Q3', 'Q4', 'QF', 'SF', 'F', 'RR', 'ER', 'BR']
        return {f'Round_{k}': (1 if rc == k else 0) for k in keys}

    @staticmethod
    def _hand_onehot(hand: str, prefix: str) -> Dict[str, int]:
        """One-hot encoding for handedness"""
        h = (hand or 'R').upper()
        return {
            f'{prefix}_Hand_R': 1 if h == 'R' else 0,
            f'{prefix}_Hand_L': 1 if h == 'L' else 0,
            f'{prefix}_Hand_U': 1 if h == 'U' else 0,
            f'{prefix}_Hand_A': 1 if h == 'A' else 0,
        }

    @staticmethod
    def _country_onehot(country: str, prefix: str) -> Dict[str, int]:
        """One-hot encoding for country"""
        c = (country or 'Other').upper()
        keys = ['USA', 'GBR', 'FRA', 'ITA', 'AUS', 'SRB', 'CZE', 'ESP', 'SUI', 'GER', 'ARG', 'RUS']
        vals = {f'{prefix}_Country_{cc}': int(c == cc) for cc in keys}
        vals[f'{prefix}_Country_Other'] = 0 if any(vals.values()) else 1
        return vals

    @staticmethod
    def _handedness_matchup(p1_hand: str, p2_hand: str) -> Dict[str, int]:
        """One-hot encoding for handedness matchup"""
        a = (p1_hand or 'R').upper()
        b = (p2_hand or 'R').upper()
        combos = ['RR', 'RL', 'LR', 'LL']
        out = {f'Handedness_Matchup_{cmb}': 0 for cmb in combos}
        key = f'{a}{b}'
        if key in combos:
            out[f'Handedness_Matchup_{key}'] = 1
        return out

    # ========== Main Feature Builder ==========

    def build_143_features(
        self,
        player1_name: Optional[str] = None,
        player2_name: Optional[str] = None,
        slug1: Optional[str] = None,
        slug2: Optional[str] = None,
        match_date: Optional[datetime] = None,
        surface: str = "Hard",
        tournament_level: str = "A",
        draw_size: int = 32,
        round_code: Optional[str] = None,
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Build exactly 143 features from Tennis Abstract data.

        Args:
            player1_name: Player 1 name (if slug1 not provided)
            player2_name: Player 2 name (if slug2 not provided)
            slug1: Player 1 TA slug (if provided, skips name resolution)
            slug2: Player 2 TA slug (if provided, skips name resolution)
            match_date: Match date (default: now)
            surface: Surface (Hard/Clay/Grass/Carpet)
            tournament_level: Level code (G/M/A/C/25/15)
            draw_size: Tournament draw size
            round_code: Round code (R32/QF/SF/F/etc)
            force_refresh: If True, always fetch fresh data (default: True)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns:
            Dict with exactly 143 features in correct order
        """
        # Resolve slugs if names provided
        if not slug1:
            if not player1_name:
                raise ValueError("Must provide either player1_name or slug1")

            # Check session cache for slug resolution
            if session_cache is not None:
                slug_map = session_cache.setdefault('slug_resolutions', {})
                if player1_name in slug_map:
                    slug1 = slug_map[player1_name]
                else:
                    slug1 = self.find_slug(player1_name)
                    if slug1:
                        slug_map[player1_name] = slug1
            else:
                slug1 = self.find_slug(player1_name)

            if not slug1:
                raise RuntimeError(f"Could not resolve TA slug for: {player1_name}")

        if not slug2:
            if not player2_name:
                raise ValueError("Must provide either player2_name or slug2")

            # Check session cache for slug resolution
            if session_cache is not None:
                slug_map = session_cache.setdefault('slug_resolutions', {})
                if player2_name in slug_map:
                    slug2 = slug_map[player2_name]
                else:
                    slug2 = self.find_slug(player2_name)
                    if slug2:
                        slug_map[player2_name] = slug2
            else:
                slug2 = self.find_slug(player2_name)

            if not slug2:
                raise RuntimeError(f"Could not resolve TA slug for: {player2_name}")

        # Build features from slugs, passing cache params through
        return self.build_143_features_from_slugs(
            slug1=slug1,
            slug2=slug2,
            match_date=match_date,
            surface=surface,
            tournament_level=tournament_level,
            draw_size=draw_size,
            round_code=round_code,
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache
        )

    def build_143_features_from_slugs(
        self,
        slug1: str,
        slug2: str,
        match_date: Optional[datetime] = None,
        surface: str = "Hard",
        tournament_level: str = "A",
        draw_size: int = 32,
        round_code: Optional[str] = None,
        force_refresh: bool = True,
        persist: bool = False,
        session_cache: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Build exactly 143 features from Tennis Abstract slugs.
        Bypasses name→slug search for tests/warmers.
        Returns dict with features in exact order expected by model.

        Args:
            slug1: Player 1 TA slug
            slug2: Player 2 TA slug
            match_date: Match date (default: now)
            surface: Surface (Hard/Clay/Grass/Carpet)
            tournament_level: Level code (G/M/A/C/25/15)
            draw_size: Tournament draw size
            round_code: Round code (R32/QF/SF/F/etc)
            force_refresh: If True, always fetch fresh data (default: True)
            persist: If True, read/write disk cache (default: False)
            session_cache: Dict for in-run memoization (default: None)

        Returns:
            Dict with exactly 143 features in correct order
        """
        when = match_date or datetime.utcnow()
        surface = (surface or "Hard").strip().title()

        # Get profiles with cache params
        profile1 = self.scraper.get_player_profile(
            slug1,
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache
        )
        profile2 = self.scraper.get_player_profile(
            slug2,
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache
        )

        if not profile1 or not profile2:
            missing_slug = slug1 if not profile1 else slug2
            raise RuntimeError(f"TA profile load failed for slug: {missing_slug}")

        # Get match histories (years=[] means ALL years for career stats)
        matches1 = self.scraper.get_player_matches(
            slug1,
            years=[],  # All years for career stats
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache
        )
        matches2 = self.scraper.get_player_matches(
            slug2,
            years=[],  # All years for career stats
            force_refresh=force_refresh,
            persist=persist,
            session_cache=session_cache
        )

        # For profile-only testing, allow empty match histories
        # (temporal features will use defaults)
        if matches1.empty and matches2.empty:
            print(f"⚠️ No match history available (will use profile-only features)")
            # Don't return defaults - proceed with profile-only features

        def _fractional_age(profile: dict, ref: datetime) -> float:
            """Fractional age in years at match date — matches Sackmann's (match_date - dob) / 365.25"""
            bd = profile.get('birthdate')  # 'YYYY-MM-DD'
            if bd:
                try:
                    dob = datetime.strptime(str(bd), '%Y-%m-%d')
                    return (ref - dob).days / 365.25
                except Exception:
                    pass
            return float(profile.get('age') or 25.0)

        # Validate required fields — raise explicitly so caller knows what's missing
        for label, profile in [('P1', profile1), ('P2', profile2)]:
            if profile.get('current_rank') is None:
                raise RuntimeError(f"{label} rank is None (not ranked / not found on TA) for slug: {profile.get('slug','unknown')}")

        s1 = {
            'height': profile1.get('height_cm'),
            'age': _fractional_age(profile1, when),
            'hand': profile1.get('hand'),
            'country': profile1.get('country'),
            'rank': float(profile1['current_rank']),
            'rank_points': 500,  # only remaining gap — not available on TA
        }
        s2 = {
            'height': profile2.get('height_cm'),
            'age': _fractional_age(profile2, when),
            'hand': profile2.get('hand'),
            'country': profile2.get('country'),
            'rank': float(profile2['current_rank']),
            'rank_points': 500,
        }

        # Calculate temporal features
        def temporal(df):
            return {
                'matches_14d': self._count_period(df, when, 14),
                'matches_30d': self._count_period(df, when, 30),
                'matches_90d': self._count_period(df, when, 90),
                'surface_matches_30d': self._count_surface(df, when, surface, 30),
                'surface_matches_90d': self._count_surface(df, when, surface, 90),
                'surface_experience': self._count_surface(df, when, surface, 9999),
                'surface_winrate_90d': self._surface_winrate(df, when, surface, 90),
                'winrate_last10_120d': self._winrate_lastN_within(df, when, 10, 120),
                'streak': self._streak(df),
                'form_trend_30d': self._form_trend_ewm(df, when, 30),
                'days_since_last': (self._days_since_last_tournament(df, when) or 60),
                'sets_14d': self._sets_14d(df, when),
                'last_surface': str(df.iloc[0]['surface']).title() if not df.empty and pd.notna(df.iloc[0].get('surface')) else None
            }

        t1 = temporal(matches1)
        t2 = temporal(matches2)

        # Rank changes and volatility
        p1_rc30 = self._rank_change(matches1, when, 30)
        p1_rc90 = self._rank_change(matches1, when, 90)
        p2_rc30 = self._rank_change(matches2, when, 30)
        p2_rc90 = self._rank_change(matches2, when, 90)
        p1_vol90 = self._rank_volatility(matches1, when, 90)
        p2_vol90 = self._rank_volatility(matches2, when, 90)

        # Level and round stats
        level_code = self._level_code(tournament_level)
        p1_level_wr, p1_level_matches = self._level_stats(matches1, level_code)
        p2_level_wr, p2_level_matches = self._level_stats(matches2, level_code)
        p1_round_wr = self._round_winrate(matches1, round_code)
        p2_round_wr = self._round_winrate(matches2, round_code)
        p1_sf_wr, p1_f_wr = self._semis_finals_winrates(matches1)
        p2_sf_wr, p2_f_wr = self._semis_finals_winrates(matches2)

        # Vs-lefty win rates
        p1_vs_lefty = self._winrate_vs_hand(matches1, 'L')
        p2_vs_lefty = self._winrate_vs_hand(matches2, 'L')

        # H2H stats (pass session_cache to avoid duplicate requests)
        h2h = self._get_h2h_stats(slug1, slug2, session_cache=session_cache)

        # One-hot encodings
        seasons = self._season_flags(surface, when)
        levels = self._levels_onehot(tournament_level)
        rounds = self._rounds_onehot(round_code)

        p1_hand = s1['hand']
        p2_hand = s2['hand']
        hand1 = self._hand_onehot(p1_hand, 'P1')
        hand2 = self._hand_onehot(p2_hand, 'P2')
        matchup = self._handedness_matchup(p1_hand, p2_hand)
        c1 = self._country_onehot(s1['country'], 'P1')
        c2 = self._country_onehot(s2['country'], 'P2')

        # Surface transition flag
        st_flag = 1 if ((t1['last_surface'] and t1['last_surface'].lower() != surface.lower()) or
                        (t2['last_surface'] and t2['last_surface'].lower() != surface.lower())) else 0

        # Assemble all features
        features: Dict[str, float] = {}

        # Direct player attributes
        features.update({
            'Player1_Height': s1['height'],
            'Player2_Height': s2['height'],
            'Player1_Age': s1['age'],
            'Player2_Age': s2['age'],
            'Player1_Rank': s1['rank'],
            'Player2_Rank': s2['rank'],
            'Player1_Rank_Points': s1['rank_points'],
            'Player2_Rank_Points': s2['rank_points'],

            'P1_Matches_14d': t1['matches_14d'],
            'P1_Matches_30d': t1['matches_30d'],
            'P1_Surface_Matches_30d': t1['surface_matches_30d'],
            'P1_Surface_Matches_90d': t1['surface_matches_90d'],
            'P1_Surface_Experience': t1['surface_experience'],
            'P1_Surface_WinRate_90d': t1['surface_winrate_90d'],
            'P1_WinRate_Last10_120d': t1['winrate_last10_120d'],
            'P1_WinStreak_Current': t1['streak'],
            'P1_Form_Trend_30d': t1['form_trend_30d'],
            'P1_Days_Since_Last': t1['days_since_last'],
            'P1_Sets_14d': t1['sets_14d'],
            'P1_Rust_Flag': 1 if t1['days_since_last'] > 21 else 0,
            'P1_Rank_Change_30d': p1_rc30,
            'P1_Rank_Change_90d': p1_rc90,
            'P1_Rank_Volatility_90d': p1_vol90,

            'P2_Matches_14d': t2['matches_14d'],
            'P2_Matches_30d': t2['matches_30d'],
            'P2_Surface_Matches_30d': t2['surface_matches_30d'],
            'P2_Surface_Matches_90d': t2['surface_matches_90d'],
            'P2_Surface_Experience': t2['surface_experience'],
            'P2_Surface_WinRate_90d': t2['surface_winrate_90d'],
            'P2_WinRate_Last10_120d': t2['winrate_last10_120d'],
            'P2_WinStreak_Current': t2['streak'],
            'P2_Form_Trend_30d': t2['form_trend_30d'],
            'P2_Days_Since_Last': t2['days_since_last'],
            'P2_Sets_14d': t2['sets_14d'],
            'P2_Rust_Flag': 1 if t2['days_since_last'] > 21 else 0,
            'P2_Rank_Change_30d': p2_rc30,
            'P2_Rank_Change_90d': p2_rc90,
            'P2_Rank_Volatility_90d': p2_vol90,
        })

        # Derived features
        features.update({
            'Height_Diff': s1['height'] - s2['height'],
            'Age_Diff': s1['age'] - s2['age'],
            'Avg_Height': (s1['height'] + s2['height']) / 2,
            'Avg_Age': (s1['age'] + s2['age']) / 2,
            'Rank_Diff': s1['rank'] - s2['rank'],
            'Rank_Points_Diff': s1['rank_points'] - s2['rank_points'],
            'Avg_Rank': (s1['rank'] + s2['rank']) / 2,
            'Avg_Rank_Points': (s1['rank_points'] + s2['rank_points']) / 2,
            'draw_size': int(draw_size),
            'Rank_Ratio': (
                max(s1['rank'], s2['rank']) / min(s1['rank'], s2['rank'])
                if min(s1['rank'], s2['rank']) > 0 else 1.0
            ),
            'Surface_Transition_Flag': st_flag,
        })

        # Peak age flags
        features['Peak_Age_P1'] = 1 if 24 <= float(s1['age']) <= 28 else 0
        features['Peak_Age_P2'] = 1 if 24 <= float(s2['age']) <= 28 else 0
        features['P1_Peak_Age'] = features['Peak_Age_P1']
        features['P2_Peak_Age'] = features['Peak_Age_P2']

        # Surfaces
        features.update({
            'Surface_Hard': 1 if surface == 'Hard' else 0,
            'Surface_Clay': 1 if surface == 'Clay' else 0,
            'Surface_Grass': 1 if surface == 'Grass' else 0,
            'Surface_Carpet': 1 if surface == 'Carpet' else 0
        })

        # Seasons, levels, rounds
        features.update(seasons)
        features.update(levels)
        features.update(rounds)

        # Level & round win rates
        features.update({
            'P1_Level_WinRate_Career': p1_level_wr,
            'P1_Level_Matches_Career': p1_level_matches,
            'P2_Level_WinRate_Career': p2_level_wr,
            'P2_Level_Matches_Career': p2_level_matches,
            'P1_Round_WinRate_Career': p1_round_wr,
            'P2_Round_WinRate_Career': p2_round_wr,
            'P1_Semifinals_WinRate': p1_sf_wr,
            'P1_Finals_WinRate': p1_f_wr,
            'P2_Semifinals_WinRate': p2_sf_wr,
            'P2_Finals_WinRate': p2_f_wr,
            'P1_BigMatch_WinRate': self._big_match_wr(matches1),
            'P2_BigMatch_WinRate': self._big_match_wr(matches2),
        })

        # Lefty & handedness & country
        features.update({
            'P1_vs_Lefty_WinRate': p1_vs_lefty,
            'P2_vs_Lefty_WinRate': p2_vs_lefty
        })
        features.update(hand1)
        features.update(hand2)
        features.update(matchup)
        features.update(c1)
        features.update(c2)

        # H2H
        features.update(h2h)

        # Momentum diffs
        features['Rank_Momentum_Diff_30d'] = p1_rc30 - p2_rc30
        features['Rank_Momentum_Diff_90d'] = p1_rc90 - p2_rc90

        # Ensure all 143 features exist in correct order
        final = {}
        for k in EXACT_143_FEATURES:
            if k in features and pd.notna(features[k]):
                final[k] = float(features[k]) if isinstance(features[k], (int, float, np.floating)) else features[k]
            else:
                final[k] = self._default_for(k, p1=s1, p2=s2, surface=surface)

        # Guard against NaNs
        for k, v in list(final.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                final[k] = self._default_for(k, p1=s1, p2=s2, surface=surface)

        return final

    # ========== Defaults ==========

    def _default_for(self, feature_name: str, p1=None, p2=None, surface='Hard') -> float:
        """Default value for missing feature"""
        if 'Rank' in feature_name and 'Points' not in feature_name:
            return 100.0
        if 'Points' in feature_name:
            return 500.0
        if 'Age' in feature_name:
            return 25.0
        if 'Height' in feature_name:
            return 180.0
        if 'WinRate' in feature_name:
            return 0.5
        if any(x in feature_name for x in [
            'Country_', 'Hand_', 'Round_', 'Level_', 'Surface_', 'Handedness_',
            'Season'
        ]):
            return 0.0
        if feature_name in ('draw_size', 'H2H_Total_Matches',
                            'P1_Matches_14d', 'P1_Matches_30d', 'P1_Sets_14d',
                            'P2_Matches_14d', 'P2_Matches_30d', 'P2_Sets_14d',
                            'P1_Days_Since_Last', 'P2_Days_Since_Last',
                            'P1_WinStreak_Current', 'P2_WinStreak_Current',
                            'P1_Surface_Matches_30d', 'P2_Surface_Matches_30d',
                            'P1_Surface_Matches_90d', 'P2_Surface_Matches_90d',
                            'P1_Surface_Experience', 'P2_Surface_Experience'):
            return 0.0
        if feature_name.endswith('_Flag'):
            return 0.0
        return 0.0

    def _defaults_143(self) -> Dict[str, float]:
        """Return dict of all 143 features with default values"""
        return {k: self._default_for(k) for k in EXACT_143_FEATURES}


def main():
    """Test the TA feature calculator"""
    calc = TAFeatureCalculator()

    # Test with two players
    features = calc.build_143_features(
        player1_name="Arthur Cazaux",
        player2_name="Mackenzie McDonald",
        match_date=datetime(2025, 10, 13),
        surface="Hard",
        tournament_level="C",
        draw_size=32,
        round_code="F"
    )

    print(f"✅ Built {len(features)} features from Tennis Abstract")
    print("\nFirst 10 features:")
    for i, k in enumerate(EXACT_143_FEATURES[:10]):
        print(f"  {k}: {features[k]}")

    print(f"\n✅ All 143 features present: {len(features) == 143}")
    print(f"✅ Feature names match schema: {list(features.keys()) == EXACT_143_FEATURES}")


if __name__ == "__main__":
    main()
