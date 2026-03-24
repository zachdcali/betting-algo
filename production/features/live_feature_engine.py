#!/usr/bin/env python3
"""
Live Feature Engineering System for Tennis Betting
Combines static features (JeffSackmann ML-ready) with live temporal features (UTR CSVs)
and returns the exact 141 features expected by NN-141 in the correct order.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import re

class MissingData(RuntimeError):
    def __init__(self, category, players):
        super().__init__(f"missing:{category}:{','.join(players)}")
        self.category = category
        self.players = players

class LiveFeatureEngine:
    """Single source of truth for building the EXACT 141 live features."""

    def __init__(self, data_dir: str = "../../data", require_round: bool = False):
        self.data_dir = Path(data_dir)
        self.require_round = bool(require_round)
        self.ml_ready_file = self.data_dir / "JeffSackmann" / "jeffsackmann_ml_ready_LEAK_FREE.csv"
        self.player_mapping_file = self.data_dir / "player_mapping.csv"
        self.matches_dir = self.data_dir / "matches"
        self.ratings_dir = self.data_dir / "ratings"

        self.EXACT_141_FEATURES = [
            'P2_WinStreak_Current','P1_WinStreak_Current','P2_Surface_Matches_30d','Height_Diff',
            'P1_Surface_Matches_30d','Player2_Height','P1_Matches_30d','P2_Matches_30d',
            'P2_Surface_Experience','P2_Form_Trend_30d','Player1_Height','P1_Form_Trend_30d',
            'Round_R16','Surface_Transition_Flag','P1_Surface_Matches_90d','P1_Surface_Experience',
            'Rank_Diff','Round_R32','Rank_Points_Diff','P2_Level_WinRate_Career',
            'P2_Surface_Matches_90d','P1_Level_WinRate_Career','P2_Level_Matches_Career',
            'P2_WinRate_Last10_120d','Round_QF','Level_25','P1_Round_WinRate_Career',
            'P1_Surface_WinRate_90d','Round_Q1','Player1_Rank','P1_Level_Matches_Career',
            'P2_Round_WinRate_Career','draw_size','P1_WinRate_Last10_120d','Age_Diff',
            'Level_15','Player1_Rank_Points','Handedness_Matchup_RL','Player2_Rank',
            'Avg_Age','P1_Country_RUS','Player2_Age','P2_vs_Lefty_WinRate',
            'Round_F','Surface_Clay','P2_Sets_14d','Rank_Momentum_Diff_30d',
            'H2H_P2_Wins','Player2_Rank_Points','Player1_Age','P2_Rank_Volatility_90d',
            'P1_Days_Since_Last','Grass_Season','P1_Semifinals_WinRate','Level_A',
            'Level_D','P1_Country_USA','P1_Country_GBR','P1_Country_FRA',
            'P2_Matches_14d','P2_Country_USA','P2_Country_ITA','Round_Q2',
            'P2_Surface_WinRate_90d','P1_Hand_L','P2_Hand_L','P1_Country_ITA',
            'P2_Rust_Flag','P1_Rank_Change_90d','P1_Country_AUS','P1_Hand_U',
            'P1_Hand_R','Round_RR','Avg_Height','P1_Sets_14d',
            'P2_Country_Other','Round_SF','P1_vs_Lefty_WinRate','Indoor_Season',
            'Avg_Rank','P1_Rust_Flag','Avg_Rank_Points','Level_F',
            'Round_R64','P2_Country_CZE','P2_Hand_R','Surface_Hard',
            'P1_Matches_14d','Surface_Carpet','Round_R128','P1_Country_SRB',
            'P2_Hand_U','P1_Rank_Volatility_90d','Level_M','P2_Country_ESP',
            'Handedness_Matchup_LR','P1_Country_CZE','P2_Country_SUI','Surface_Grass',
            'H2H_Total_Matches','Level_O','P1_Hand_A','P1_Finals_WinRate',
            'Rank_Momentum_Diff_90d','P2_Finals_WinRate',
            'Round_Q4','Level_G','Round_ER','Level_S','Round_BR',
            'Round_Q3','Rank_Ratio','P1_Country_SUI','Clay_Season','P1_Country_GER',
            'P2_Rank_Change_30d','P1_Country_ESP','P2_Hand_A','H2H_Recent_P1_Advantage',
            'P2_Country_AUS','P2_Country_SRB','P2_Country_GBR','P2_Country_ARG',
            'Handedness_Matchup_RR','P1_Rank_Change_30d','P2_Country_GER','Handedness_Matchup_LL',
            'P2_Country_RUS','P1_Country_ARG','Level_C','P2_Semifinals_WinRate',
            'P2_Days_Since_Last','P1_Peak_Age','P2_Peak_Age','H2H_P1_WinRate',
            'P1_Country_Other','H2H_P1_Wins','P1_BigMatch_WinRate','P2_Rank_Change_90d',
            'P2_BigMatch_WinRate','P2_Country_FRA'
        ]

        self._load_static()

    # ---------- loading & mapping ----------

    def _load_static(self):
        # ML-ready dataset
        if self.ml_ready_file.exists():
            self.ml_df = pd.read_csv(self.ml_ready_file, low_memory=False)
            # standardize dates
            for col in ('tourney_date', 'Date', 'date'):
                if col in self.ml_df.columns:
                    self.ml_df[col] = pd.to_datetime(self.ml_df[col], errors='coerce')
            # ensure round/level columns exist-ish
            if 'Round' in self.ml_df.columns:
                self.ml_df['Round'] = self.ml_df['Round'].astype(str)
            if 'Level' in self.ml_df.columns:
                self.ml_df['Level'] = self.ml_df['Level'].astype(str)
        else:
            self.ml_df = pd.DataFrame()

        # Player mappings (normalized → utr_id)
        self.name_to_id: Dict[str, str] = {}
        self.utr_id_to_primary: Dict[str, str] = {}
        self.mappings_df = pd.DataFrame()
        if self.player_mapping_file.exists():
            m = pd.read_csv(self.player_mapping_file)
            self.mappings_df = m
            for _, row in m.iterrows():
                pid = str(row.get('utr_id', '') or '').strip()
                primary = str(row.get('primary_name', '') or '').strip()
                variants = str(row.get('name_variants', '') or '')
                bovada = str(row.get('bovada_name', '') or '').strip()
                if pid:
                    if primary:
                        self.utr_id_to_primary[pid] = primary
                        self.name_to_id[self._norm(primary)] = pid
                    if bovada:
                        self.name_to_id[self._norm(bovada)] = pid
                    if variants:
                        for v in variants.split('|'):
                            v = v.strip()
                            if v:
                                self.name_to_id[self._norm(v)] = pid

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").strip().lower()
        # strip accents
        try:
            import unicodedata
            s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
        except Exception:
            pass
        s = re.sub(r'[^a-z\s\-]', '', s)
        return re.sub(r'\s+', ' ', s).strip()

    def find_utr_id(self, player_name: str) -> Optional[str]:
        return self.name_to_id.get(self._norm(player_name))

    # ---------- IO helpers ----------

    def _load_player_matches(self, utr_id: str) -> pd.DataFrame:
        f = self.matches_dir / f"player_{utr_id}_matches.csv"
        if not f.exists():
            return pd.DataFrame()
        df = pd.read_csv(f)
        # flexible date/surface/result columns
        date_col = 'date' if 'date' in df.columns else ('match_date' if 'match_date' in df.columns else None)
        if date_col:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df = df.dropna(subset=[date_col])
            df = df.rename(columns={date_col: 'date'})
        else:
            df['date'] = pd.NaT
        for col in ('surface', 'result', 'opponent_id', 'score'):
            if col not in df.columns:
                df[col] = np.nan
        df = df.sort_values('date', ascending=False)
        # Enrich surface if missing
        df = self._enrich_surface_from_title(df)
        return df

    def _enrich_surface_from_title(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Infer surface from tournament name using tournaments_map.csv when surface column is missing/empty.
        Lightweight heuristic to recover surface data for temporal features.
        """
        # Only enrich if surface is missing or mostly empty
        if 'surface' in df.columns and df['surface'].notna().sum() > len(df) * 0.5:
            return df  # Already have enough surface data

        # Load tournaments map
        mapf = self.data_dir / "tournaments_map.csv"
        if not mapf.exists() or df.empty:
            if 'surface' not in df.columns:
                df['surface'] = np.nan
            return df

        try:
            tm = pd.read_csv(mapf)
            # Build fuzzy index on lowercased tokens
            tm['_aliases'] = (
                tm['aliases'].fillna('') + '|' +
                tm['canonical_name'].fillna('') + '|' +
                tm['city_sig'].fillna('')
            ).str.lower()

            def guess_surface(text: str) -> Optional[str]:
                if pd.isna(text) or not text:
                    return np.nan
                t = str(text).lower()
                # Extract event name (before ' - ' if present)
                event_name = t.split(' - ')[0][:40]  # First 40 chars
                # Find matches in tournaments_map
                hits = tm[tm['_aliases'].str.contains(re.escape(event_name), na=False, regex=True)]
                if hits.empty:
                    return np.nan
                # Use first match's surface
                s = str(hits.iloc[0]['surface']).title()
                return s if s in ('Hard', 'Clay', 'Grass', 'Carpet') else np.nan

            # Look for tournament column (various names)
            for col_name in ('tournament', 'event', 'title'):
                if col_name in df.columns:
                    df['surface'] = df[col_name].apply(guess_surface)
                    break
            else:
                # No tournament column found
                if 'surface' not in df.columns:
                    df['surface'] = np.nan

        except Exception as e:
            # Silently fail - just means we won't have surface data
            if 'surface' not in df.columns:
                df['surface'] = np.nan

        return df

    # ---------- temporal features (UTR) ----------

    @staticmethod
    def _count_period(df: pd.DataFrame, ref: datetime, days: int) -> int:
        if df.empty or 'date' not in df.columns:
            return 0
        cut = ref - timedelta(days=days)
        return int((df['date'] >= cut).sum())

    @staticmethod
    def _surface_mask(df: pd.DataFrame, surface: str) -> pd.Series:
        if 'surface' not in df.columns:
            return pd.Series(False, index=df.index)
        return df['surface'].astype(str).str.lower() == (surface or '').lower()

    def _count_surface(self, df: pd.DataFrame, ref: datetime, surface: str, days: int) -> int:
        if df.empty:
            return 0
        cut = ref - timedelta(days=days)
        m = (df['date'] >= cut) & self._surface_mask(df, surface)
        return int(m.sum())

    def _surface_winrate(self, df: pd.DataFrame, ref: datetime, surface: str, days: int, min_matches: int = 5) -> float:
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        m = df[(df['date'] >= cut) & self._surface_mask(df, surface)]
        if len(m) < min_matches:
            return 0.5
        wins = (m['result'].astype(str) == 'W').sum()
        return float(wins) / float(len(m))

    @staticmethod
    def _winrate_lastN_within(df: pd.DataFrame, ref: datetime, N: int, days: int, min_matches: int = 3) -> float:
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        m = df[df['date'] >= cut].head(N)
        if len(m) < min_matches:
            return 0.5
        wins = (m['result'].astype(str) == 'W').sum()
        return float(wins) / float(len(m))

    def _form_trend_ewm(self, df: pd.DataFrame, ref: datetime, days: int = 30) -> float:
        if df.empty:
            return 0.5
        cut = ref - timedelta(days=days)
        m = df[(df['date'] >= cut) & (df['date'] < ref)]
        if m.empty:
            return 0.5
        wins_w, total_w = 0.0, 0.0
        for _, r in m.iterrows():
            days_ago = (ref - r['date']).days
            w = float(np.exp(-days_ago / 15.0))
            total_w += w
            if str(r.get('result', '')) == 'W':
                wins_w += w
        return wins_w / total_w if total_w > 0 else 0.5

    def _winrate_vs_lefty(self, name: str, min_matches: int = 3) -> float:
        if self.ml_df.empty:
            return 0.5
        df = self._player_slice(name)
        if df.empty:
            return 0.5
        wins, total = 0, 0
        for _, r in df.iterrows():
            if r.get('Player1_Name') == name:
                opp_hand = r.get('Player2_Hand', 'R')
                if str(opp_hand).upper() == 'L':
                    total += 1
                    if r.get('Winner') in (name, r.get('Player1_Name')):
                        wins += 1
            elif r.get('Player2_Name') == name:
                opp_hand = r.get('Player1_Hand', 'R')
                if str(opp_hand).upper() == 'L':
                    total += 1
                    if r.get('Winner') in (name, r.get('Player2_Name')):
                        wins += 1
        if total < min_matches:
            return 0.5
        return wins / total

    @staticmethod
    def _level_code(level: str) -> Optional[str]:
        s = (level or "").strip().upper()
        if s in {"A", "ATP", "ATP 250", "ATP 500"}: return "A"
        if s in {"M", "MASTERS", "MASTERS 1000"}:  return "M"
        if s in {"G", "GRAND SLAM", "SLAM"}:       return "G"
        if s in {"C", "CHALLENGER"}:               return "C"
        if s in {"25", "ITF M25", "M25", "25K"}:   return "25"
        if s in {"15", "ITF M15", "M15", "15K"}:   return "15"
        if s in {"F", "ATP FINALS"}:               return "F"   # (rare; fine if unused in level stats)
        if s in {"S", "ITF", "FUTURES"}:           return "S"
        return None

    def _level_stats(self, name: str, level_code: Optional[str]) -> Tuple[float, int]:
        """Winrate & matches at *this* level; 0.5 prior if total <5."""
        if self.ml_df.empty or not level_code:
            return (0.5, 0)
        df = self._player_slice(name)
        if df.empty or 'Level' not in df.columns:
            return (0.5, 0)
        sub = df[df['Level'].astype(str) == level_code]
        total = len(sub)
        if total < 5:
            return (0.5, total)
        wins = (sub.get('Winner') == name).sum()
        return (wins / total, total)

    def _round_winrate(self, name: str, round_code: Optional[str]) -> float:
        if not round_code or self.ml_df.empty:
            return 0.5
        df = self._player_slice(name)
        if df.empty or 'Round' not in df.columns:
            return 0.5
        sub = df[df['Round'].astype(str) == str(round_code)]
        total = len(sub)
        if total == 0:
            return 0.5
        # training thresholds: F ≥1, SF ≥2, others ≥3
        min_n = {'F': 1, 'SF': 2}.get(str(round_code).upper(), 3)
        if total < min_n:
            return 0.5
        wins = (sub.get('Winner') == name).sum()
        return float(wins) / float(total)

    def _semis_finals_winrates(self, name: str) -> Tuple[float, float]:
        if self.ml_df.empty or 'Round' not in self.ml_df.columns:
            return (0.5, 0.5)
        df = self._player_slice(name)
        if df.empty:
            return (0.5, 0.5)
        def wr(code, min_n):
            sub = df[df['Round'].astype(str) == code]
            if len(sub) < min_n:
                return 0.5
            wins = (sub.get('Winner') == name).sum()
            return wins / len(sub)
        return (wr('SF', 2), wr('F', 1))

    def _big_match_wr(self, name: str) -> float:
        """Big matches: G+M pooled; min total 3 else 0.5"""
        g_wr, g_n = self._level_stats(name, 'G')
        m_wr, m_n = self._level_stats(name, 'M')
        tot = g_n + m_n
        if tot < 3: return 0.5
        return (g_wr * g_n + m_wr * m_n) / tot

    @staticmethod
    def _streak(df: pd.DataFrame) -> int:
        s = 0
        for _, r in df.iterrows():
            if str(r.get('result', '')) == 'W':
                s += 1
            else:
                break
        return s

    @staticmethod
    def _monday_of(dt: pd.Timestamp) -> pd.Timestamp:
        return dt - pd.Timedelta(days=int(dt.weekday()))

    def _days_since_last_tournament(self, df: pd.DataFrame, ref: datetime) -> Optional[int]:
        if df.empty or 'date' not in df.columns:
            return None
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


    def _sets_14d(self, df: pd.DataFrame, ref: datetime) -> int:
        return int(self._count_period(df, ref, 14) * 2.5)

    # ---------- static features (Jeff) ----------

    def _primary_name_for(self, utr_id: str, fallback: str) -> str:
        return self.utr_id_to_primary.get(utr_id, fallback)

    def _player_slice(self, name: str) -> pd.DataFrame:
        if self.ml_df.empty:
            return pd.DataFrame()
        return self.ml_df[(self.ml_df.get('Player1_Name') == name) | (self.ml_df.get('Player2_Name') == name)]

    def _static_snapshot(self, name: str) -> Dict[str, float]:
        """Most recent snapshot for height/age/hand/country/rank/rank_points."""
        if self.ml_df.empty:
            return {'height':180,'age':25,'hand':'R','country':'Other',
                    'rank':100,'rank_points':500}
        df = self._player_slice(name)
        if df.empty:
            return {'height':180,'age':25,'hand':'R','country':'Other',
                    'rank':100,'rank_points':500}

        # pick most recent by tourney_date (fallback any date col)
        date_col = 'tourney_date' if 'tourney_date' in df.columns else ( 'date' if 'date' in df.columns else None)
        if date_col:
            df = df.sort_values(date_col)
        row = df.iloc[-1]

        def pick(prefix):
            # choose columns by role
            if row.get('Player1_Name') == name:
                return {
                    'height': row.get('Player1_Height', 180),
                    'age': row.get('Player1_Age', 25),
                    'hand': row.get('Player1_Hand', 'R'),
                    'country': row.get('Player1_IOC', 'Other'),
                    'rank': row.get('Player1_Rank', 100),
                    'rank_points': row.get('Player1_Rank_Points', 500),
                }
            else:
                return {
                    'height': row.get('Player2_Height', 180),
                    'age': row.get('Player2_Age', 25),
                    'hand': row.get('Player2_Hand', 'R'),
                    'country': row.get('Player2_IOC', 'Other'),
                    'rank': row.get('Player2_Rank', 100),
                    'rank_points': row.get('Player2_Rank_Points', 500),
                }

        snap = pick('Player1' if row.get('Player1_Name') == name else 'Player2')
        # normalize country flag names
        snap['country'] = str(snap['country'] or 'Other').upper()
        if snap['country'] not in {'USA','GBR','FRA','ITA','AUS','SRB','CZE','ESP','SUI','GER','ARG','RUS'}:
            snap['country'] = 'Other'
        snap['hand'] = str(snap['hand'] or 'R').upper()
        if snap['hand'] not in {'R','L','U','A'}:
            snap['hand'] = 'R'
        # coerce numeric
        for k in ('height','age','rank','rank_points'):
            try:
                snap[k] = float(snap[k])
            except Exception:
                snap[k] = {'height':180,'age':25,'rank':100,'rank_points':500}[k]
        return snap

    def _rank_change(self, name: str, ref_date: datetime, days: int) -> float:
        """Rank(t0 - days) - Rank(t0); positive => improved (rank number decreased)."""
        if self.ml_df.empty:
            return 0.0
        df = self._player_slice(name)
        if df.empty:
            return 0.0
        date_col = 'tourney_date' if 'tourney_date' in df.columns else ('date' if 'date' in df.columns else None)
        if not date_col:
            return 0.0
        df = df.sort_values(date_col)
        past = df[df[date_col] < ref_date]
        if past.empty:
            return 0.0
        t_now = past.iloc[-1]
        t_then_cut = ref_date - timedelta(days=days)
        past_then = past[past[date_col] < t_then_cut]
        if past_then.empty:
            return 0.0
        t_then = past_then.iloc[-1]

        def get_rank(row):
            return row.get('Player1_Rank') if row.get('Player1_Name') == name else row.get('Player2_Rank')

        r_now = pd.to_numeric(get_rank(t_now), errors='coerce')
        r_then = pd.to_numeric(get_rank(t_then), errors='coerce')
        if pd.notna(r_now) and pd.notna(r_then):
            return float(r_then) - float(r_now)
        return 0.0

    def _rank_volatility(self, name: str, ref_date: datetime, days: int) -> float:
        """Std dev of rank over the last 'days' days."""
        if self.ml_df.empty:
            return 0.0
        df = self._player_slice(name)
        if df.empty:
            return 0.0
        date_col = 'tourney_date' if 'tourney_date' in df.columns else ( 'date' if 'date' in df.columns else None)
        if not date_col:
            return 0.0
        cut = ref_date - timedelta(days=days)
        window = df[(df[date_col] < ref_date) & (df[date_col] >= cut)].copy()
        if window.empty:
            return 0.0
        # collect ranks per row role
        ranks = []
        for _, r in window.iterrows():
            if r.get('Player1_Name') == name:
                ranks.append(r.get('Player1_Rank', np.nan))
            elif r.get('Player2_Name') == name:
                ranks.append(r.get('Player2_Rank', np.nan))
        ranks = pd.to_numeric(pd.Series(ranks), errors='coerce').dropna()
        if ranks.empty:
            return 0.0
        return float(ranks.std())


    # ---------- H2H via UTR CSVs ----------

    def _h2h_stats(self, p1_id: str, p2_id: str) -> Dict[str, float]:
        df1 = self._load_player_matches(p1_id)
        df2 = self._load_player_matches(p2_id)
        if df1.empty and df2.empty:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }
        # filter df1 rows vs opponent_id == p2_id, and vice versa
        a = df1[df1.get('opponent_id').astype(str) == str(p2_id)].copy()
        b = df2[df2.get('opponent_id').astype(str) == str(p1_id)].copy()
        # outcomes are from each player's perspective
        total = len(a) + len(b)
        p1_w = (a.get('result').astype(str) == 'W').sum() + (b.get('result').astype(str) == 'L').sum()
        p2_w = total - p1_w
        if total == 0:
            return {
                'H2H_Total_Matches': 0,
                'H2H_P1_Wins': 0,
                'H2H_P2_Wins': 0,
                'H2H_P1_WinRate': 0.5,
                'H2H_Recent_P1_Advantage': 0.0
            }
        # Recent advantage from P1 perspective over last up to 3 meetings
        # Flip P2's results to P1's perspective
        bb = b[['date', 'result']].copy()
        bb['result'] = bb['result'].map({'W': 'L', 'L': 'W'})
        recent = pd.concat([a[['date', 'result']], bb], ignore_index=True).sort_values('date', ascending=False).head(3)

        if len(recent) >= 2:
            recent_p1_win_frac = (recent['result'] == 'W').mean()
            adv = float(recent_p1_win_frac - 0.5)   # −0.5 … +0.5
        else:
            adv = 0.0

        wr = (p1_w / total) if total >= 3 else 0.5

        return {
            'H2H_Total_Matches': int(total),
            'H2H_P1_Wins': int(p1_w),
            'H2H_P2_Wins': int(p2_w),
            'H2H_P1_WinRate': float(wr),
            'H2H_Recent_P1_Advantage': float(adv)
        }

    # ---------- helpers ----------

    @staticmethod
    def _season_flags(surface: str, when: datetime) -> Dict[str, int]:
        m = when.month
        return {
            'Clay_Season':   1 if 4 <= m <= 6 else 0,         # Apr–Jun
            'Grass_Season':  1 if m in (6, 7) else 0,         # Jun–Jul
            'Indoor_Season': 1 if (m >= 10 or m <= 2) else 0  # Oct–Feb
        }

    @staticmethod
    def _levels_onehot(level: str) -> Dict[str, int]:
        """
        Accept Sackmann-style codes or text. Map to training-consistent one-hots.
        Sackmann (men): G, M, A, C, S, 25, 15, F, D, O
        """
        s = (level or "").strip().upper()

        # Normalize a few common text variants to codes
        if s in {"GRAND SLAM", "SLAM"}: s = "G"
        elif s in {"MASTERS 1000", "MASTERS"}: s = "M"
        elif s in {"ATP", "ATP 250", "ATP 500"}: s = "A"
        elif s in {"CHALLENGER"}: s = "C"
        elif s in {"ITF", "FUTURES"}: s = "S"        # Sackmann uses S for Satellites/ITF
        elif s in {"ITF M25", "M25", "25K", "25"}: s = "25"
        elif s in {"ITF M15", "M15", "15K", "15"}: s = "15"
        elif s in {"ATP FINALS", "NITTO ATP FINALS"}: s = "F"

        flags = {
            "Level_G": 1 if s == "G" else 0,
            "Level_M": 1 if s == "M" else 0,
            "Level_A": 1 if s == "A" else 0,
            "Level_C": 1 if s == "C" else 0,
            "Level_S": 1 if s == "S" else 0,  # generic ITF/Satellites bucket
            "Level_F": 1 if s == "F" else 0,  # season-ending finals
            "Level_25": 1 if s == "25" else 0,
            "Level_15": 1 if s == "15" else 0,
            "Level_O": 1 if s == "O" else 0,  # "Other" (rare)
            "Level_D": 1 if s == "D" else 0,  # Davis Cup, etc. (rare)
        }
        return flags

    @staticmethod
    def _rounds_onehot(round_code: Optional[str]) -> Dict[str, int]:
        rc = (round_code or '').upper()
        keys = ['R128','R64','R32','R16','Q1','Q2','Q3','Q4','QF','SF','F','RR','ER','BR']
        return {f'Round_{k}': (1 if rc == k else 0) for k in keys}

    @staticmethod
    def _hand_onehot(hand: str, prefix: str) -> Dict[str, int]:
        h = (hand or 'R').upper()
        return {
            f'{prefix}_Hand_R': 1 if h=='R' else 0,
            f'{prefix}_Hand_L': 1 if h=='L' else 0,
            f'{prefix}_Hand_U': 1 if h=='U' else 0,
            f'{prefix}_Hand_A': 1 if h=='A' else 0,
        }

    @staticmethod
    def _country_onehot(country: str, prefix: str) -> Dict[str, int]:
        c = (country or 'Other').upper()
        keys = ['USA','GBR','FRA','ITA','AUS','SRB','CZE','ESP','SUI','GER','ARG','RUS']
        vals = {f'{prefix}_Country_{cc}': int(c == cc) for cc in keys}
        vals[f'{prefix}_Country_Other'] = 0 if any(vals.values()) else 1
        return vals

    @staticmethod
    def _handedness_matchup(p1_hand: str, p2_hand: str) -> Dict[str, int]:
        a = (p1_hand or 'R').upper()
        b = (p2_hand or 'R').upper()
        combos = ['RR','RL','LR','LL']
        out = {f'Handedness_Matchup_{cmb}': 0 for cmb in combos}
        key = f'{a}{b}'
        if key in combos:
            out[f'Handedness_Matchup_{key}'] = 1
        return out

    @staticmethod
    def _peak_age_flag(age: float) -> int:
        # simple tennis peak band ~24-28
        try:
            return 1 if 24 <= float(age) <= 28 else 0
        except Exception:
            return 0

    # ---------- public API ----------

    def build_match_features(
        self,
        player1_name: str,
        player2_name: str,
        match_date: datetime = None,
        surface: str = "Hard",
        tournament_level: str = "ATP",
        draw_size: int = 32,
        round_code: Optional[str] = None,
        strict: bool = True
    ) -> Dict[str, float]:
        """
        Returns dict of exactly 141 features in the right order (we'll re-order at the end).
        """
        when = match_date or datetime.utcnow()
        surface = (surface or "Hard").strip().title()

        # resolve UTR ids
        p1_id = self.find_utr_id(player1_name)
        p2_id = self.find_utr_id(player2_name)
        
        # Strict mode gates - fail fast on missing data
        if strict:
            missing_players = []
            if not p1_id:
                missing_players.append(player1_name)
            if not p2_id:
                missing_players.append(player2_name)
            if missing_players:
                raise MissingData("mapping", missing_players)
        
        if not p1_id or not p2_id:
            return self._defaults_141()

        # primary names for static datasets
        p1_primary = self._primary_name_for(p1_id, player1_name)
        p2_primary = self._primary_name_for(p2_id, player2_name)

        # Strict mode: validate static data exists in Jeff Sackmann dataset
        if strict:
            missing_static = []
            p1_slice = self._player_slice(p1_primary)
            p2_slice = self._player_slice(p2_primary)
            if p1_slice.empty:
                missing_static.append(p1_primary)
            if p2_slice.empty:
                missing_static.append(p2_primary)
            if missing_static:
                raise MissingData("static", missing_static)

        # load UTR temporal
        p1_matches = self._load_player_matches(p1_id)
        p2_matches = self._load_player_matches(p2_id)

        # Strict mode: validate match history exists and is non-empty
        if strict:
            missing_history = []
            if p1_matches.empty:
                missing_history.append(p1_primary)
            if p2_matches.empty:
                missing_history.append(p2_primary)
            if missing_history:
                raise MissingData("history", missing_history)

        # Strict mode: validate tournament metadata is complete
        if strict:
            tm_issues = []
            if surface not in ("Hard", "Clay", "Grass", "Carpet"):
                tm_issues.append("surface")
            level_code = self._level_code(tournament_level)
            if level_code is None:
                tm_issues.append("level")
            if draw_size is None or int(draw_size) <= 0:
                tm_issues.append("draw_size")
            if not round_code and self.require_round:
                tm_issues.append("round_code")
            if tm_issues:
                raise MissingData("tournament_meta", tm_issues)

        # Strict mode: validate rankings are present (numeric ranks > 0)
        if strict:
            # Check static snapshots for both players
            temp_s1 = self._static_snapshot(p1_primary)
            temp_s2 = self._static_snapshot(p2_primary)
            missing_ranks = []
            if not (isinstance(temp_s1['rank'], (int, float)) and temp_s1['rank'] > 0):
                missing_ranks.append(p1_primary)
            if not (isinstance(temp_s2['rank'], (int, float)) and temp_s2['rank'] > 0):
                missing_ranks.append(p2_primary)
            if missing_ranks:
                raise MissingData("rankings", missing_ranks)

        # temporal (per player)
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

        t1 = temporal(p1_matches)
        t2 = temporal(p2_matches)

        # static snapshots
        s1 = self._static_snapshot(p1_primary)
        s2 = self._static_snapshot(p2_primary)

        # momentum & volatility from Jeff
        p1_rc30 = self._rank_change(p1_primary, when, 30)
        p1_rc90 = self._rank_change(p1_primary, when, 90)
        p2_rc30 = self._rank_change(p2_primary, when, 30)
        p2_rc90 = self._rank_change(p2_primary, when, 90)
        p1_vol90 = self._rank_volatility(p1_primary, when, 90)
        p2_vol90 = self._rank_volatility(p2_primary, when, 90)

        # level / round stats with proper level code
        level_code = self._level_code(tournament_level)
        p1_level_wr, p1_level_matches = self._level_stats(p1_primary, level_code)
        p2_level_wr, p2_level_matches = self._level_stats(p2_primary, level_code)
        p1_round_wr = self._round_winrate(p1_primary, round_code)
        p2_round_wr = self._round_winrate(p2_primary, round_code)
        p1_sf_wr, p1_f_wr = self._semis_finals_winrates(p1_primary)
        p2_sf_wr, p2_f_wr = self._semis_finals_winrates(p2_primary)

        # vs-lefty
        p1_vs_lefty = self._winrate_vs_lefty(p1_primary)
        p2_vs_lefty = self._winrate_vs_lefty(p2_primary)

        # H2H
        h2h = self._h2h_stats(p1_id, p2_id)

        # seasons & levels & rounds
        seasons = self._season_flags(surface, when)
        levels = self._levels_onehot(tournament_level)
        rounds = self._rounds_onehot(round_code)

        # handedness & countries
        p1_hand = s1['hand']; p2_hand = s2['hand']
        hand1 = self._hand_onehot(p1_hand, 'P1')
        hand2 = self._hand_onehot(p2_hand, 'P2')
        matchup = self._handedness_matchup(p1_hand, p2_hand)
        c1 = self._country_onehot(s1['country'], 'P1')
        c2 = self._country_onehot(s2['country'], 'P2')

        # surface transition flag (did either player’s last surface differ from current?)
        st_flag = 1 if ((t1['last_surface'] and t1['last_surface'].lower() != surface.lower()) or
                        (t2['last_surface'] and t2['last_surface'].lower() != surface.lower())) else 0

        # assemble
        features: Dict[str, float] = {}

        # direct player attributes / temporal
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

        # derived
        features.update({
            'Height_Diff': s1['height'] - s2['height'],
            'Age_Diff': s1['age'] - s2['age'],
            'Avg_Height': (s1['height'] + s2['height'])/2,
            'Avg_Age': (s1['age'] + s2['age'])/2,
            'Rank_Diff': s1['rank'] - s2['rank'],
            'Rank_Points_Diff': s1['rank_points'] - s2['rank_points'],
            'Avg_Rank': (s1['rank'] + s2['rank'])/2,
            'Avg_Rank_Points': (s1['rank_points'] + s2['rank_points'])/2,
            'draw_size': int(draw_size),
            'Rank_Ratio': (
                max(s1['rank'], s2['rank']) / min(s1['rank'], s2['rank'])
                if min(s1['rank'], s2['rank']) > 0 else 1.0
            ),
            'Surface_Transition_Flag': st_flag,
        })

        # Peak age flags
        features['P1_Peak_Age'] = 1 if 24 <= float(s1['age']) <= 28 else 0
        features['P2_Peak_Age'] = 1 if 24 <= float(s2['age']) <= 28 else 0

        # surfaces
        features.update({
            'Surface_Hard': 1 if surface == 'Hard' else 0,
            'Surface_Clay': 1 if surface == 'Clay' else 0,
            'Surface_Grass': 1 if surface == 'Grass' else 0,
            'Surface_Carpet': 1 if surface == 'Carpet' else 0
        })

        # seasons, levels, rounds
        features.update(seasons)
        features.update(levels)
        features.update(rounds)

        # level & round win rates / matches
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
            'P1_BigMatch_WinRate': self._big_match_wr(p1_primary),
            'P2_BigMatch_WinRate': self._big_match_wr(p2_primary),
        })

        # lefty & handedness & country one-hots
        features.update({
            'P1_vs_Lefty_WinRate': p1_vs_lefty,
            'P2_vs_Lefty_WinRate': p2_vs_lefty
        })
        features.update(hand1); features.update(hand2); features.update(matchup)
        features.update(c1); features.update(c2)

        # H2H
        features.update(h2h)
        # momentum diffs
        features['Rank_Momentum_Diff_30d'] = p1_rc30 - p2_rc30
        features['Rank_Momentum_Diff_90d'] = p1_rc90 - p2_rc90

        # ensure all required keys exist; fill safe defaults for any still missing
        final = {}
        for k in self.EXACT_141_FEATURES:
            if k in features and pd.notna(features[k]):
                final[k] = float(features[k]) if isinstance(features[k], (int,float,np.floating)) else features[k]
            else:
                final[k] = self._default_for(k, p1=s1, p2=s2, surface=surface)

        # guard against NaNs
        for k, v in list(final.items()):
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                final[k] = self._default_for(k, p1=s1, p2=s2, surface=surface)

        return final

    # ---------- defaults ----------

    def _default_for(self, feature_name: str, p1=None, p2=None, surface='Hard') -> float:
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
                            'P1_Matches_14d','P1_Matches_30d','P1_Sets_14d',
                            'P2_Matches_14d','P2_Matches_30d','P2_Sets_14d',
                            'P1_Days_Since_Last','P2_Days_Since_Last',
                            'P1_WinStreak_Current','P2_WinStreak_Current',
                            'P1_Surface_Matches_30d','P2_Surface_Matches_30d',
                            'P1_Surface_Matches_90d','P2_Surface_Matches_90d',
                            'P1_Surface_Experience','P2_Surface_Experience'):
            return 0.0
        if feature_name.endswith('_Flag'):
            return 0.0
        return 0.0

    def _defaults_141(self) -> Dict[str, float]:
        return {k: self._default_for(k) for k in self.EXACT_141_FEATURES}


def main():
    engine = LiveFeatureEngine()
    f = engine.build_match_features(
        "novak djokovic", "rafael nadal",
        match_date=datetime.now(),
        surface="Hard",
        tournament_level="ATP",
        draw_size=64,
        round_code="R32"
    )
    print("✅ Features built:", len(f))
    # print a small subset deterministically
    keys = engine.EXACT_141_FEATURES[:10]
    for k in keys:
        print(k, "=", f[k])

def _contract_test():
    # Paste the exact array from train_nn_143.py here to lock names/order.
    exact_from_train = [
        'P2_WinStreak_Current','P1_WinStreak_Current','P2_Surface_Matches_30d','Height_Diff',
        'P1_Surface_Matches_30d','Player2_Height','P1_Matches_30d','P2_Matches_30d',
        'P2_Surface_Experience','P2_Form_Trend_30d','Player1_Height','P1_Form_Trend_30d',
        'Round_R16','Surface_Transition_Flag','P1_Surface_Matches_90d','P1_Surface_Experience',
        'Rank_Diff','Round_R32','Rank_Points_Diff','P2_Level_WinRate_Career',
        'P2_Surface_Matches_90d','P1_Level_WinRate_Career','P2_Level_Matches_Career',
        'P2_WinRate_Last10_120d','Round_QF','Level_25','P1_Round_WinRate_Career',
        'P1_Surface_WinRate_90d','Round_Q1','Player1_Rank','P1_Level_Matches_Career',
        'P2_Round_WinRate_Career','draw_size','P1_WinRate_Last10_120d','Age_Diff',
        'Level_15','Player1_Rank_Points','Handedness_Matchup_RL','Player2_Rank',
        'Avg_Age','P1_Country_RUS','Player2_Age','P2_vs_Lefty_WinRate',
        'Round_F','Surface_Clay','P2_Sets_14d','Rank_Momentum_Diff_30d',
        'H2H_P2_Wins','Player2_Rank_Points','Player1_Age','P2_Rank_Volatility_90d',
        'P1_Days_Since_Last','Grass_Season','P1_Semifinals_WinRate','Level_A',
        'Level_D','P1_Country_USA','P1_Country_GBR','P1_Country_FRA',
        'P2_Matches_14d','P2_Country_USA','P2_Country_ITA','Round_Q2',
        'P2_Surface_WinRate_90d','P1_Hand_L','P2_Hand_L','P1_Country_ITA',
        'P2_Rust_Flag','P1_Rank_Change_90d','P1_Country_AUS','P1_Hand_U',
        'P1_Hand_R','Round_RR','Avg_Height','P1_Sets_14d',
        'P2_Country_Other','Round_SF','P1_vs_Lefty_WinRate','Indoor_Season',
        'Avg_Rank','P1_Rust_Flag','Avg_Rank_Points','Level_F',
        'Round_R64','P2_Country_CZE','P2_Hand_R','Surface_Hard',
        'P1_Matches_14d','Surface_Carpet','Round_R128','P1_Country_SRB',
        'P2_Hand_U','P1_Rank_Volatility_90d','Level_M','P2_Country_ESP',
        'Handedness_Matchup_LR','P1_Country_CZE','P2_Country_SUI','Surface_Grass',
        'H2H_Total_Matches','Level_O','P1_Hand_A','P1_Finals_WinRate',
        'Rank_Momentum_Diff_90d','P2_Finals_WinRate',
        'Round_Q4','Level_G','Round_ER','Level_S','Round_BR',
        'Round_Q3','Rank_Ratio','P1_Country_SUI','Clay_Season','P1_Country_GER',
        'P2_Rank_Change_30d','P1_Country_ESP','P2_Hand_A','H2H_Recent_P1_Advantage',
        'P2_Country_AUS','P2_Country_SRB','P2_Country_GBR','P2_Country_ARG',
        'Handedness_Matchup_RR','P1_Rank_Change_30d','P2_Country_GER','Handedness_Matchup_LL',
        'P2_Country_RUS','P1_Country_ARG','Level_C','P2_Semifinals_WinRate',
        'P2_Days_Since_Last','P1_Peak_Age','P2_Peak_Age','H2H_P1_WinRate',
        'P1_Country_Other','H2H_P1_Wins','P1_BigMatch_WinRate','P2_Rank_Change_90d',
        'P2_BigMatch_WinRate','P2_Country_FRA'
    ]

    engine = LiveFeatureEngine()
    assert engine.EXACT_141_FEATURES == exact_from_train, "⚠️ Name/order drift from training!"
    
    try:
        from datetime import datetime
        v = engine.build_match_features("Novak Djokovic", "Rafael Nadal", datetime.utcnow(), "Hard", "ATP", 64, "R32")
        assert set(v.keys()) == set(exact_from_train), f"Missing features: {set(exact_from_train) - set(v.keys())}"
        print("✅ Contract test passed - all 141 features present and accounted for!")
        
        # Sanity ranges for rates/flags
        for k in exact_from_train:
            x = v[k]
            if any(t in k for t in ("WinRate","Form_Trend","H2H_P1_WinRate")):
                assert 0.0 <= float(x) <= 1.0, f"{k} = {x} not in [0,1]"
        print("✅ All feature ranges look reasonable!")
        return True
    except Exception as e:
        print(f"❌ Contract test failed: {e}")
        return False

if __name__ == "__main__":
    print("Running main test...")
    main()
    print("\nRunning contract test...")
    _contract_test()
