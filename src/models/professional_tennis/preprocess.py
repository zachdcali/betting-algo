import pandas as pd
import numpy as np
import json
import os
import sys
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path

# Import production feature contracts shared by training and serving.
_PRODUCTION_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..', '..', 'production'
)
_FEATURES_DIR = os.path.join(_PRODUCTION_DIR, 'features')
_ACTIVE_LEGACY_DATASET = (
    Path(__file__).resolve().parents[3]
    / 'data' / 'JeffSackmann' / 'jeffsackmann_ml_ready_SURFACE_FIX.csv'
).resolve()
for _path in (_PRODUCTION_DIR, _FEATURES_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)
from round_offsets import get_round_day_offset
from base_141_shared import (
    SEMANTICS_ID as SHARED_SEMANTICS_ID,
    as_of_day,
    feature_default as shared_feature_default,
    h2h_features_from_history,
    normalize_player_snapshot,
    observations_from_records,
    player_temporal_features,
    surface_transition_flag as shared_surface_transition_flag,
)
from versioning import HISTORICAL_SEMANTICS_ID


def _paths_alias(candidate, active) -> bool:
    """Fail-closed path alias check for candidate side outputs.

    Resolved/case-folded comparison catches symlink and case-only aliases;
    ``samefile`` catches hard links when both paths exist.
    """

    candidate_path = Path(candidate).expanduser()
    active_path = Path(active).expanduser()
    candidate_resolved = candidate_path.resolve(strict=False)
    active_resolved = active_path.resolve(strict=False)
    if str(candidate_resolved).casefold() == str(active_resolved).casefold():
        return True
    try:
        return candidate_path.exists() and active_path.exists() and os.path.samefile(
            candidate_path, active_path
        )
    except OSError:
        return False

def laplace(wins: int, total: int, alpha: float = 3.0) -> float:
    """Bayesian (Laplace) smoothing: (wins + α/2) / (total + α). Returns 0.5 when total=0."""
    return (wins + alpha / 2.0) / (total + alpha)


def get_feature_columns(df):
    """
    Helper function to identify feature columns for ML models.
    Returns only the columns that should be used as features (no metadata, no target).
    """
    # Metadata columns to exclude
    metadata_cols = [
        'tourney_id', 'tourney_name', 'tourney_date', 'match_num', 'score', 
        'best_of', 'round', 'minutes', 'data_source',
        'winner_id', 'winner_name', 'loser_id', 'loser_name',
        'Player1_ID', 'Player1_Name', 'Player2_ID', 'Player2_Name',
        'winner_seed', 'winner_entry', 'loser_seed', 'loser_entry',
        'winner_ioc', 'loser_ioc',
        # Original categorical columns (we use one-hot encoded versions instead)
        'tourney_level', 'surface', 'round'
    ]
    
    # Target column
    target_col = 'Player1_Wins'
    
    # Get all columns except metadata and target
    feature_cols = [col for col in df.columns if col not in metadata_cols + [target_col]]
    
    return feature_cols

def calculate_temporal_features(
    df,
    feature_semantics_id=HISTORICAL_SEMANTICS_ID,
):
    """
    Calculate all temporal derived features from high-coverage base data.
    This implements ALL recommendations from ChatGPT/Grok analysis.
    """
    
    if feature_semantics_id not in {HISTORICAL_SEMANTICS_ID, SHARED_SEMANTICS_ID}:
        raise ValueError(f"unsupported historical feature semantics: {feature_semantics_id}")
    use_shared_semantics = feature_semantics_id == SHARED_SEMANTICS_ID

    print("=" * 60)
    print("CALCULATING ENHANCED TEMPORAL FEATURES")
    print("=" * 60)
    
    # Sort by inferred match date (tourney_date + round offset) then match_num for
    # within-tournament ordering. This is CRITICAL — Sackmann stores tournament start
    # date for all rounds, so without this two tournaments starting the same week
    # would be interleaved and within-tournament features would be computed out of order.
    if 'inferred_match_date' in df.columns:
        df_sorted = df.sort_values(['inferred_match_date', 'match_num']).reset_index(drop=True).copy()
    else:
        df_sorted = df.sort_values(['tourney_date', 'match_num']).reset_index(drop=True).copy()
    df_sorted['tourney_date'] = pd.to_datetime(df_sorted['tourney_date'])
    
    # Initialize player tracking dictionaries
    player_stats = defaultdict(lambda: {
        'last_match_date': None,
        'match_history': deque(maxlen=50),  # Store last 50 matches per player
        'rank_history': deque(maxlen=50),
        'surface_stats': defaultdict(lambda: {'matches': deque(maxlen=20), 'wins': deque(maxlen=20)}),
        'surface_career_count': defaultdict(int),  # Unbounded career match count per surface (fixes deque(maxlen=20) cap bug)
        'level_stats': defaultdict(lambda: {'total': 0, 'wins': 0}),
        'round_stats': defaultdict(lambda: {'total': 0, 'wins': 0}),
        # Frozen one-direction legacy tracker. Shared H2H is derived from the
        # source-neutral match history so tied-day ambiguity uses the kernel's
        # exact same policy as live serving.
        'h2h_stats': defaultdict(lambda: {'total': 0, 'wins': 0}),
    })
    
    # Initialize new feature columns
    enhanced_features = [
        # PHASE 1: Ranking Dynamics (highest ROI)
        'P1_Rank_Change_30d', 'P2_Rank_Change_30d',
        'P1_Rank_Change_90d', 'P2_Rank_Change_90d', 
        'P1_Rank_Volatility_90d', 'P2_Rank_Volatility_90d',
        'Rank_Momentum_Diff_30d', 'Rank_Momentum_Diff_90d',
        
        # PHASE 1: Activity/Fatigue (rust/rest factors)
        'P1_Days_Since_Last', 'P2_Days_Since_Last',
        'P1_Rust_Flag', 'P2_Rust_Flag',  # >21 days
        'P1_Matches_14d', 'P2_Matches_14d',
        'P1_Matches_30d', 'P2_Matches_30d',
        'P1_Sets_14d', 'P2_Sets_14d',
        
        # PHASE 1: Age/Physical advantages
        'Age_Diff', 'Height_Diff',
        
        # PHASE 2: Surface-Specific Form (high value)
        'P1_Surface_Matches_30d', 'P2_Surface_Matches_30d',
        'P1_Surface_Matches_90d', 'P2_Surface_Matches_90d',
        'P1_Surface_WinRate_90d', 'P2_Surface_WinRate_90d',
        'P1_Surface_Experience', 'P2_Surface_Experience',  # Career matches on surface
        
        # PHASE 2: Tournament Level Experience
        'P1_Level_WinRate_Career', 'P2_Level_WinRate_Career',
        'P1_Level_Matches_Career', 'P2_Level_Matches_Career',
        'P1_BigMatch_WinRate', 'P2_BigMatch_WinRate',  # GS/Masters only
        
        # PHASE 2: Round Performance
        'P1_Round_WinRate_Career', 'P2_Round_WinRate_Career',
        'P1_Finals_WinRate', 'P2_Finals_WinRate',
        'P1_Semifinals_WinRate', 'P2_Semifinals_WinRate',
        
        # PHASE 2: Head-to-Head Temporal
        'H2H_Total_Matches', 'H2H_P1_Wins', 'H2H_P2_Wins', 
        'H2H_P1_WinRate', 'H2H_Recent_P1_Advantage',  # Last 3 H2H meetings
        
        # PHASE 2: Handedness Matchups
        'Handedness_Matchup_RR', 'Handedness_Matchup_RL', 
        'Handedness_Matchup_LR', 'Handedness_Matchup_LL',
        'P1_vs_Lefty_WinRate', 'P2_vs_Lefty_WinRate',
        
        # PHASE 2: Seasonal Patterns
        'Clay_Season', 'Grass_Season', 'Indoor_Season',
        'Surface_Transition_Flag',  # Different surface from last match
        
        # PHASE 2: Form Metrics (with recency caps)
        'P1_WinRate_Last10_120d', 'P2_WinRate_Last10_120d',
        'P1_WinStreak_Current', 'P2_WinStreak_Current',
        'P1_Form_Trend_30d', 'P2_Form_Trend_30d'  # Exponential weighted recent form
    ]
    
    # Initialize feature columns with NaN
    for feature in enhanced_features:
        df_sorted[feature] = np.nan
    
    print(f"Processing {len(df_sorted)} matches chronologically...")
    print("This ensures no data leakage - each match only uses prior information")

    pending_shared_updates = []
    pending_shared_day = None

    def _update_sort_key(update):
        try:
            match_order = float(update['match_order'])
        except (TypeError, ValueError):
            match_order = 0.0
        return (
            match_order,
            str(update['tourney_id']),
            str(update['p1_id']),
            str(update['p2_id']),
            str(update['p1_rank']),
            str(update['p2_rank']),
            str(update['match_result']),
            str(update['surface']),
            str(update['level']),
            str(update['round_name']),
            str(update['score']),
        )

    def _apply_match_update(update):
        for player_num, player_id in enumerate([update['p1_id'], update['p2_id']], 1):
            stats = player_stats[player_id]
            opponent_id = update['p2_id'] if player_num == 1 else update['p1_id']
            opponent_hand = update['p2_hand'] if player_num == 1 else update['p1_hand']
            won = (
                update['match_result'] == 1
                if player_num == 1 else update['match_result'] == 0
            )
            current_rank = update['p1_rank'] if player_num == 1 else update['p2_rank']

            match_info = {
                'date': update['match_date'],
                'opponent': opponent_id,
                'opponent_hand': opponent_hand,
                'surface': update['surface'],
                'level': update['level'],
                'round': update['round_name'],
                'won': won,
                'rank': current_rank,
            }
            if use_shared_semantics:
                match_info['score'] = update['score']
                match_info['source_key'] = update['source_key']
                if update['chronological_order'] is not None:
                    match_info['chronological_order'] = update['chronological_order']
                    match_info['chronological_order_scope'] = (
                        update['chronological_order_scope'] or '__global__'
                    )
                elif update['round_ord'] is not None:
                    match_info['round_ord'] = update['round_ord']
                    match_info['order_scope'] = update['tourney_id']
            stats['match_history'].append(match_info)
            stats['rank_history'].append(current_rank)
            stats['last_match_date'] = update['match_date']

            surface_stats = stats['surface_stats'][update['surface']]
            surface_stats['matches'].append(update['match_date'])
            surface_stats['wins'].append(won)
            stats['surface_career_count'][update['surface']] += 1

            level_stats = stats['level_stats'][update['level']]
            level_stats['total'] += 1
            level_stats['wins'] += int(won)
            round_stats = stats['round_stats'][update['round_name']]
            round_stats['total'] += 1
            round_stats['wins'] += int(won)

            if player_num == 1:
                legacy_h2h = stats['h2h_stats'][opponent_id]
                legacy_h2h['total'] += 1
                legacy_h2h['wins'] += int(won)

    def _flush_pending_shared_updates():
        nonlocal pending_shared_updates
        for update in sorted(pending_shared_updates, key=_update_sort_key):
            _apply_match_update(update)
        pending_shared_updates = []
    
    # Process each match chronologically
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(df_sorted):,} matches ({idx/len(df_sorted)*100:.1f}%)")
        
        # Get player IDs and basic info
        p1_id = row['Player1_ID'] if pd.notna(row['Player1_ID']) else row['Player1_Name']
        p2_id = row['Player2_ID'] if pd.notna(row['Player2_ID']) else row['Player2_Name']
        # Use inferred match date (tourney_date + round offset) if available,
        # otherwise fall back to raw tourney_date.
        match_date = row['inferred_match_date'] if 'inferred_match_date' in df_sorted.columns else row['tourney_date']
        if use_shared_semantics:
            match_date = as_of_day(match_date)
            if pending_shared_day is not None and match_date != pending_shared_day:
                _flush_pending_shared_updates()
            pending_shared_day = match_date
        surface = row['surface'] if pd.notna(row['surface']) else 'Hard'
        level = row['tourney_level'] if pd.notna(row['tourney_level']) else 'A'
        round_name = (
            row['round'] if pd.notna(row['round'])
            else (None if use_shared_semantics else 'R32')
        )
        
        # Extract current match info
        p1_rank = row['Player1_Rank'] if pd.notna(row['Player1_Rank']) else 999
        p2_rank = row['Player2_Rank'] if pd.notna(row['Player2_Rank']) else 999
        p1_age = row['Player1_Age'] if pd.notna(row['Player1_Age']) else 25
        p2_age = row['Player2_Age'] if pd.notna(row['Player2_Age']) else 25
        p1_height = row['Player1_Height'] if pd.notna(row['Player1_Height']) else 180
        p2_height = row['Player2_Height'] if pd.notna(row['Player2_Height']) else 180
        default_hand = 'U' if use_shared_semantics else 'R'
        p1_hand = row.get('Player1_Hand', default_hand) if pd.notna(row.get('Player1_Hand', default_hand)) else default_hand
        p2_hand = row.get('Player2_Hand', default_hand) if pd.notna(row.get('Player2_Hand', default_hand)) else default_hand

        shared_temporal = {}
        if use_shared_semantics:
            for player_num, player_id in enumerate([p1_id, p2_id], 1):
                stats = player_stats[player_id]
                current_rank = p1_rank if player_num == 1 else p2_rank
                shared_temporal[player_num] = player_temporal_features(
                    observations_from_records(stats['match_history']),
                    match_date,
                    surface,
                    rank_as_of=current_rank,
                    surface_experience=stats['surface_career_count'][surface],
                )
        
        # Calculate features for both players
        for player_num, player_id in enumerate([p1_id, p2_id], 1):
            prefix = f'P{player_num}_'
            stats = player_stats[player_id]
            
            # === PHASE 1: RANKING DYNAMICS ===
            if use_shared_semantics:
                temporal = shared_temporal[player_num]
                df_sorted.loc[row.name, f'{prefix}Rank_Change_30d'] = temporal['rank_change_30d']
                df_sorted.loc[row.name, f'{prefix}Rank_Change_90d'] = temporal['rank_change_90d']
                df_sorted.loc[row.name, f'{prefix}Rank_Volatility_90d'] = temporal['rank_volatility_90d']
            elif len(stats['rank_history']) >= 2:
                recent_ranks = list(stats['rank_history'])
                dates = [match['date'] for match in stats['match_history']]
                
                # Ranking changes over time
                rank_30d_ago = get_rank_at_date(recent_ranks, dates, match_date, days_back=30)
                rank_90d_ago = get_rank_at_date(recent_ranks, dates, match_date, days_back=90)
                
                current_rank = p1_rank if player_num == 1 else p2_rank
                
                if rank_30d_ago is not None:
                    df_sorted.loc[row.name, f'{prefix}Rank_Change_30d'] = rank_30d_ago - current_rank
                if rank_90d_ago is not None:
                    df_sorted.loc[row.name, f'{prefix}Rank_Change_90d'] = rank_90d_ago - current_rank
                
                # Ranking volatility
                recent_90d_ranks = get_ranks_in_window(recent_ranks, dates, match_date, days_back=90)
                if len(recent_90d_ranks) >= 3:
                    df_sorted.loc[row.name, f'{prefix}Rank_Volatility_90d'] = np.std(recent_90d_ranks)
            
            # === PHASE 1: ACTIVITY/FATIGUE ===
            # Calculate days since last TOURNAMENT, not last match (prevents leakage)
            if use_shared_semantics:
                days_since = shared_temporal[player_num]['days_since_last']
            else:
                last_tournament_date = get_last_tournament_date(stats['match_history'], match_date)
                days_since = (
                    (match_date - last_tournament_date).days
                    if last_tournament_date is not None else None
                )
            if days_since is not None:
                df_sorted.loc[row.name, f'{prefix}Days_Since_Last'] = days_since
                df_sorted.loc[row.name, f'{prefix}Rust_Flag'] = (
                    shared_temporal[player_num]['rust_flag']
                    if use_shared_semantics else (1 if days_since > 21 else 0)
                )
            
            # Match frequency
            if use_shared_semantics:
                matches_14d = shared_temporal[player_num]['matches_14d']
                matches_30d = shared_temporal[player_num]['matches_30d']
            else:
                matches_14d = count_matches_in_window(stats['match_history'], match_date, days=14)
                matches_30d = count_matches_in_window(stats['match_history'], match_date, days=30)
            df_sorted.loc[row.name, f'{prefix}Matches_14d'] = matches_14d
            df_sorted.loc[row.name, f'{prefix}Matches_30d'] = matches_30d
            
            # Candidate semantics uses score lineage; legacy keeps its frozen
            # 2.5-sets-per-match approximation.
            sets_14d = (
                shared_temporal[player_num]['sets_14d']
                if use_shared_semantics else matches_14d * 2.5
            )
            df_sorted.loc[row.name, f'{prefix}Sets_14d'] = sets_14d
            
            # === PHASE 1: AGE/PHYSICAL ===
            age = p1_age if player_num == 1 else p2_age
            if player_num == 1:
                df_sorted.loc[row.name, 'Age_Diff'] = p1_age - p2_age
                df_sorted.loc[row.name, 'Height_Diff'] = p1_height - p2_height
            df_sorted.loc[row.name, f'{prefix}Peak_Age'] = 1 if 24 <= age <= 28 else 0
            
        # === PHASE 2: SURFACE-SPECIFIC FEATURES ===
        for player_num, player_id in enumerate([p1_id, p2_id], 1):
            prefix = f'P{player_num}_'
            stats = player_stats[player_id]
            surface_stats = stats['surface_stats'][surface]
            
            # Surface activity
            if use_shared_semantics:
                surface_matches_30d = shared_temporal[player_num]['surface_matches_30d']
                surface_matches_90d = shared_temporal[player_num]['surface_matches_90d']
            else:
                surface_matches_30d = count_surface_matches(surface_stats['matches'], match_date, days=30)
                surface_matches_90d = count_surface_matches(surface_stats['matches'], match_date, days=90)
            df_sorted.loc[row.name, f'{prefix}Surface_Matches_30d'] = surface_matches_30d
            df_sorted.loc[row.name, f'{prefix}Surface_Matches_90d'] = surface_matches_90d
            
            # Surface win rate (with minimum threshold)
            surface_win_rate = (
                shared_temporal[player_num]['surface_winrate_90d']
                if use_shared_semantics
                else calculate_surface_win_rate(surface_stats, match_date, days=90, min_matches=5)
            )
            df_sorted.loc[row.name, f'{prefix}Surface_WinRate_90d'] = surface_win_rate
            
            # Career surface experience — use unbounded counter, not deque length (which was capped at 20)
            career_surface_matches = (
                shared_temporal[player_num]['surface_experience']
                if use_shared_semantics else stats['surface_career_count'][surface]
            )
            df_sorted.loc[row.name, f'{prefix}Surface_Experience'] = career_surface_matches
            
            # === PHASE 2: TOURNAMENT LEVEL EXPERIENCE ===
            level_stats = stats['level_stats'][level]
            level_win_rate = laplace(level_stats['wins'], level_stats['total'])
            df_sorted.loc[row.name, f'{prefix}Level_WinRate_Career'] = level_win_rate
            df_sorted.loc[row.name, f'{prefix}Level_Matches_Career'] = level_stats['total']
            
            # Big match performance (Grand Slams + Masters)
            big_match_wins = stats['level_stats']['G']['wins'] + stats['level_stats']['M']['wins']
            big_match_total = stats['level_stats']['G']['total'] + stats['level_stats']['M']['total']
            big_match_rate = laplace(big_match_wins, big_match_total)
            df_sorted.loc[row.name, f'{prefix}BigMatch_WinRate'] = big_match_rate
            
            # === PHASE 2: ROUND PERFORMANCE ===
            round_stats = stats['round_stats'][round_name]
            round_win_rate = laplace(round_stats['wins'], round_stats['total'])
            df_sorted.loc[row.name, f'{prefix}Round_WinRate_Career'] = round_win_rate
            
            # Specific round performance
            finals_rate = laplace(stats['round_stats']['F']['wins'], stats['round_stats']['F']['total'])
            sf_rate = laplace(stats['round_stats']['SF']['wins'], stats['round_stats']['SF']['total'])
            df_sorted.loc[row.name, f'{prefix}Finals_WinRate'] = finals_rate
            df_sorted.loc[row.name, f'{prefix}Semifinals_WinRate'] = sf_rate
            
            # === PHASE 2: FORM METRICS ===
            win_rate_10_120d = (
                shared_temporal[player_num]['winrate_last10_120d']
                if use_shared_semantics
                else calculate_recent_form(stats['match_history'], match_date, max_matches=10, max_days=120)
            )
            df_sorted.loc[row.name, f'{prefix}WinRate_Last10_120d'] = win_rate_10_120d
            
            # Current win streak
            win_streak = (
                shared_temporal[player_num]['streak']
                if use_shared_semantics else calculate_current_streak(stats['match_history'])
            )
            df_sorted.loc[row.name, f'{prefix}WinStreak_Current'] = win_streak
            
            # Form trend (exponential weighted)
            form_trend = (
                shared_temporal[player_num]['form_trend_30d']
                if use_shared_semantics
                else calculate_form_trend(stats['match_history'], match_date, days=30)
            )
            df_sorted.loc[row.name, f'{prefix}Form_Trend_30d'] = form_trend
        
        # === PHASE 2: HEAD-TO-HEAD ===
        if use_shared_semantics:
            h2h = h2h_features_from_history(
                observations_from_records(player_stats[p1_id]['match_history']),
                p2_id,
                match_date,
            )
            for feature_name, value in h2h.items():
                df_sorted.loc[row.name, feature_name] = value
        else:
            h2h_stats = player_stats[p1_id]['h2h_stats'][p2_id]
            df_sorted.loc[row.name, 'H2H_Total_Matches'] = h2h_stats['total']
            df_sorted.loc[row.name, 'H2H_P1_Wins'] = h2h_stats['wins']
            df_sorted.loc[row.name, 'H2H_P2_Wins'] = h2h_stats['total'] - h2h_stats['wins']

            h2h_win_rate = laplace(h2h_stats['wins'], h2h_stats['total'])
            df_sorted.loc[row.name, 'H2H_P1_WinRate'] = h2h_win_rate

            # Recent H2H advantage (last 3 meetings)
            recent_h2h = calculate_recent_h2h(player_stats[p1_id]['match_history'], p2_id, max_meetings=3)
            df_sorted.loc[row.name, 'H2H_Recent_P1_Advantage'] = recent_h2h
        
        # === PHASE 2: HANDEDNESS MATCHUPS ===
        handedness_key = f"{p1_hand}{p2_hand}"
        for combo in ['RR', 'RL', 'LR', 'LL']:
            df_sorted.loc[row.name, f'Handedness_Matchup_{combo}'] = 1 if handedness_key == combo else 0
        
        # Performance vs lefties
        p1_vs_lefty = calculate_vs_handedness_rate(player_stats[p1_id]['match_history'], 'L')
        p2_vs_lefty = calculate_vs_handedness_rate(player_stats[p2_id]['match_history'], 'L')
        df_sorted.loc[row.name, 'P1_vs_Lefty_WinRate'] = p1_vs_lefty
        df_sorted.loc[row.name, 'P2_vs_Lefty_WinRate'] = p2_vs_lefty
        
        # === PHASE 2: SEASONAL PATTERNS ===
        month = match_date.month
        df_sorted.loc[row.name, 'Clay_Season'] = 1 if 4 <= month <= 6 else 0
        df_sorted.loc[row.name, 'Grass_Season'] = 1 if month == 6 or month == 7 else 0
        df_sorted.loc[row.name, 'Indoor_Season'] = 1 if month >= 10 or month <= 2 else 0
        
        # Surface transition flag
        if use_shared_semantics:
            df_sorted.loc[row.name, 'Surface_Transition_Flag'] = shared_surface_transition_flag(
                shared_temporal[1]['last_surface'],
                shared_temporal[2]['last_surface'],
                surface,
            )
        else:
            p1_last_surface = get_last_surface(player_stats[p1_id]['match_history'])
            p2_last_surface = get_last_surface(player_stats[p2_id]['match_history'])
            df_sorted.loc[row.name, 'Surface_Transition_Flag'] = 1 if (p1_last_surface != surface or p2_last_surface != surface) else 0
        
        # === PHASE 1: RANKING MOMENTUM DIFFERENCES ===
        p1_momentum_30d = df_sorted.loc[row.name, 'P1_Rank_Change_30d']
        p2_momentum_30d = df_sorted.loc[row.name, 'P2_Rank_Change_30d']
        if pd.notna(p1_momentum_30d) and pd.notna(p2_momentum_30d):
            df_sorted.loc[row.name, 'Rank_Momentum_Diff_30d'] = p1_momentum_30d - p2_momentum_30d
        
        p1_momentum_90d = df_sorted.loc[row.name, 'P1_Rank_Change_90d']
        p2_momentum_90d = df_sorted.loc[row.name, 'P2_Rank_Change_90d']
        if pd.notna(p1_momentum_90d) and pd.notna(p2_momentum_90d):
            df_sorted.loc[row.name, 'Rank_Momentum_Diff_90d'] = p1_momentum_90d - p2_momentum_90d
        
        # Candidate rows sharing a calendar day read one immutable state
        # snapshot.  Their updates are applied only when the next day begins.
        update = {
            'tourney_id': row.get('tourney_id', ''),
            'match_order': row.get('match_num', 0),
            'match_date': match_date,
            'p1_id': p1_id,
            'p2_id': p2_id,
            'p1_hand': p1_hand,
            'p2_hand': p2_hand,
            'p1_rank': p1_rank,
            'p2_rank': p2_rank,
            'surface': surface,
            'level': level,
            'round_name': round_name,
            'match_result': row['Player1_Wins'],
            'score': row.get('score'),
            'source_key': (
                str(row.get('match_id'))
                if pd.notna(row.get('match_id')) and str(row.get('match_id')).strip()
                else (
                    f"{row.get('tourney_id', '')}:{p1_id}:{p2_id}:"
                    f"{row.get('Player1_Wins')}:{row.get('score', '')}"
                )
            ),
            'chronological_order': (
                row.get('chronological_order')
                if pd.notna(row.get('chronological_order')) else None
            ),
            'chronological_order_scope': (
                row.get('chronological_order_scope')
                if pd.notna(row.get('chronological_order_scope')) else None
            ),
            'round_ord': (
                row.get('round_ord') if pd.notna(row.get('round_ord')) else None
            ),
        }
        if use_shared_semantics:
            pending_shared_updates.append(update)
        else:
            _apply_match_update(update)

    if use_shared_semantics:
        _flush_pending_shared_updates()
    
    print("\n✅ Temporal feature calculation complete!")
    print(f"Added {len(enhanced_features)} new temporal features")
    
    # Fill remaining NaN values with sensible defaults
    print("\nFilling missing values with defaults...")
    
    # Days since last match: 60 days for first match
    df_sorted['P1_Days_Since_Last'] = df_sorted['P1_Days_Since_Last'].fillna(60)
    df_sorted['P2_Days_Since_Last'] = df_sorted['P2_Days_Since_Last'].fillna(60)
    
    # Activity counts: 0 for new players
    activity_cols = [col for col in enhanced_features if 'Matches_' in col or 'Sets_' in col]
    for col in activity_cols:
        df_sorted[col] = df_sorted[col].fillna(0)
    
    # Win rates: 0.5 (neutral) for insufficient data
    rate_cols = [col for col in enhanced_features if 'WinRate' in col or 'Rate' in col]
    for col in rate_cols:
        df_sorted[col] = df_sorted[col].fillna(0.5)
    
    # Ranking changes: 0 for insufficient history
    rank_cols = [col for col in enhanced_features if 'Rank_Change' in col or 'Volatility' in col or 'Momentum' in col]
    for col in rank_cols:
        df_sorted[col] = df_sorted[col].fillna(0)
    
    # Binary flags: 0 as default
    flag_cols = [col for col in enhanced_features if 'Flag' in col or 'Season' in col or 'Peak_Age' in col or 'Handedness_Matchup' in col]
    for col in flag_cols:
        df_sorted[col] = df_sorted[col].fillna(0)
    
    # Streaks and trends: 0 as neutral
    misc_cols = [col for col in enhanced_features if 'Streak' in col or 'Trend' in col or 'Advantage' in col]
    for col in misc_cols:
        df_sorted[col] = df_sorted[col].fillna(0)
    
    return df_sorted

# === HELPER FUNCTIONS ===

def get_rank_at_date(rank_history, date_history, target_date, days_back):
    """Get rank approximately N days before target date."""
    target_date_back = target_date - pd.Timedelta(days=days_back)
    
    # Find closest match before target date
    best_rank = None
    min_diff = float('inf')
    
    for rank, date in zip(rank_history, date_history):
        if date <= target_date_back:
            diff = abs((target_date_back - date).days)
            if diff < min_diff:
                min_diff = diff
                best_rank = rank
    
    return best_rank if min_diff <= days_back // 2 else None

def get_ranks_in_window(rank_history, date_history, target_date, days_back):
    """Get all ranks within the specified window."""
    cutoff_date = target_date - pd.Timedelta(days=days_back)
    
    ranks = []
    for rank, date in zip(rank_history, date_history):
        if cutoff_date <= date < target_date:
            ranks.append(rank)
    
    return ranks

def count_matches_in_window(match_history, target_date, days):
    """Count matches in the last N days."""
    cutoff_date = target_date - pd.Timedelta(days=days)
    
    count = 0
    for match in match_history:
        if cutoff_date <= match['date'] < target_date:
            count += 1
    
    return count

def count_surface_matches(surface_matches, target_date, days):
    """Count matches on specific surface in last N days."""
    cutoff_date = target_date - pd.Timedelta(days=days)
    
    count = 0
    for match_date in surface_matches:
        if cutoff_date <= match_date < target_date:
            count += 1
    
    return count

def calculate_surface_win_rate(surface_stats, target_date, days, min_matches=5):
    """Calculate win rate on surface using Laplace smoothing."""
    cutoff_date = target_date - pd.Timedelta(days=days)

    matches = []
    wins = []

    for match_date, won in zip(surface_stats['matches'], surface_stats['wins']):
        if cutoff_date <= match_date < target_date:
            matches.append(match_date)
            wins.append(won)

    return laplace(sum(wins), len(matches))

def calculate_recent_form(match_history, target_date, max_matches=10, max_days=120):
    """Calculate win rate over last N matches within M days."""
    cutoff_date = target_date - pd.Timedelta(days=max_days)
    
    # Get recent matches within time window
    recent_matches = []
    for match in match_history:
        if cutoff_date <= match['date'] < target_date:
            recent_matches.append(match)
    
    # Sort by date and take most recent N matches
    recent_matches.sort(key=lambda x: x['date'], reverse=True)
    recent_matches = recent_matches[:max_matches]
    
    wins = sum(1 for match in recent_matches if match['won'])
    return laplace(wins, len(recent_matches))

def calculate_current_streak(match_history):
    """Calculate current win/loss streak."""
    if not match_history:
        return 0
    
    # Sort by date, most recent first
    sorted_matches = sorted(match_history, key=lambda x: x['date'], reverse=True)
    
    if not sorted_matches:
        return 0
    
    # Check if most recent match was won or lost
    last_result = sorted_matches[0]['won']
    streak = 1 if last_result else -1
    
    # Count consecutive same results
    for match in sorted_matches[1:]:
        if match['won'] == last_result:
            if last_result:
                streak += 1
            else:
                streak -= 1
        else:
            break
    
    return streak

def calculate_form_trend(match_history, target_date, days=30):
    """Calculate exponential weighted form trend."""
    cutoff_date = target_date - pd.Timedelta(days=days)
    
    # Get matches in window
    recent_matches = []
    for match in match_history:
        if cutoff_date <= match['date'] < target_date:
            recent_matches.append(match)
    
    if not recent_matches:
        return 0.5
    
    # Calculate exponential weights (more recent = higher weight)
    total_weight = 0
    weighted_wins = 0
    
    for match in recent_matches:
        days_ago = (target_date - match['date']).days
        weight = np.exp(-days_ago / 15)  # Half-life of 15 days
        
        total_weight += weight
        if match['won']:
            weighted_wins += weight
    
    return weighted_wins / total_weight if total_weight > 0 else 0.5

def calculate_recent_h2h(match_history, opponent_id, max_meetings=3):
    """Calculate advantage in recent H2H meetings."""
    h2h_matches = []
    for match in match_history:
        if match['opponent'] == opponent_id:
            h2h_matches.append(match)
    
    # Sort by date, most recent first
    h2h_matches.sort(key=lambda x: x['date'], reverse=True)
    recent_h2h = h2h_matches[:max_meetings]
    
    wins = sum(1 for match in recent_h2h if match['won'])
    return laplace(wins, len(recent_h2h)) - 0.5  # Advantage relative to 50%

def calculate_vs_handedness_rate(match_history, target_handedness):
    """Calculate win rate vs specific handedness."""
    vs_hand_matches = []
    for match in match_history:
        if match.get('opponent_hand') == target_handedness:
            vs_hand_matches.append(match)
    
    wins = sum(1 for match in vs_hand_matches if match['won'])
    return laplace(wins, len(vs_hand_matches))

def get_last_surface(match_history):
    """Get surface of most recent match."""
    if not match_history:
        return 'Hard'  # Default
    
    # Sort by date, most recent first
    sorted_matches = sorted(match_history, key=lambda x: x['date'], reverse=True)
    return sorted_matches[0]['surface'] if sorted_matches else 'Hard'

def get_last_tournament_date(match_history, current_date):
    """Get the start date of the most recent completed tournament."""
    if not match_history:
        return None
    
    # Group matches by tournament (same week = same tournament)
    tournaments = {}
    for match in match_history:
        # Use Monday of the tournament week as key
        monday = match['date'] - pd.Timedelta(days=match['date'].weekday())
        if monday not in tournaments:
            tournaments[monday] = []
        tournaments[monday].append(match['date'])
    
    # Find most recent tournament that ended before current date
    current_monday = current_date - pd.Timedelta(days=current_date.weekday())
    
    recent_tournaments = [date for date in tournaments.keys() if date < current_monday]
    if recent_tournaments:
        return max(recent_tournaments)
    else:
        return None

def preprocess_jeffsackmann_data_for_ml(
    output_path=None,
    include_performance_features=False,
    feature_semantics_id=HISTORICAL_SEMANTICS_ID,
):
    """
    Enhanced preprocessing with comprehensive temporal features.
    This replaces the old preprocessing with all AI-recommended features.
    """
    
    if feature_semantics_id not in {HISTORICAL_SEMANTICS_ID, SHARED_SEMANTICS_ID}:
        raise ValueError(f"unsupported historical feature semantics: {feature_semantics_id}")
    if feature_semantics_id == SHARED_SEMANTICS_ID and output_path is None:
        raise ValueError(
            "base_141_shared is a reserved candidate; provide an explicit side-output path"
        )
    if (
        feature_semantics_id == SHARED_SEMANTICS_ID
        and _paths_alias(output_path, _ACTIVE_LEGACY_DATASET)
    ):
        raise ValueError(
            "base_141_shared candidate cannot overwrite the active legacy dataset; "
            "use an explicit versioned side-output path"
        )
    if feature_semantics_id == SHARED_SEMANTICS_ID and include_performance_features:
        raise ValueError("base_141_shared candidate currently covers the base 141 fields only")

    print("=" * 60)
    print("ENHANCED JEFF SACKMANN ATP DATA ML PREPROCESSING")
    print("=" * 60)
    
    # Load the master combined Jeffsackmann data
    print(f"\n1. Loading Jeff Sackmann master combined data...")
    _base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..", "data", "JeffSackmann")
    df = pd.read_csv(os.path.join(_base, "jeffsackmann_master_combined.csv"), low_memory=False)
    print(f"   Original data shape: {df.shape}")
    print(f"   Data source distribution:")
    print(f"   {df['data_source'].value_counts()}")
    
    # Convert date column
    df['tourney_date'] = pd.to_datetime(df['tourney_date'], format='%Y%m%d')
    df['year'] = df['tourney_date'].dt.year
    df['month'] = df['tourney_date'].dt.month
    df['day_of_year'] = df['tourney_date'].dt.dayofyear
    
    print(f"\n2. Processing ALL data...")
    initial_count = len(df)
    print(f"   Total matches: {initial_count}")
    print(f"   Date range: {df['tourney_date'].min().date()} to {df['tourney_date'].max().date()}")
    print(f"   Year range: {df['year'].min()} to {df['year'].max()}")
    
    if len(df) == 0:
        print("   ERROR: No matches in dataset!")
        return
    
    # 3. Randomize player positions
    print("\n3. Randomizing player positions...")
    
    # Create a random boolean mask for swapping
    np.random.seed(42)  # For reproducibility
    swap_mask = np.random.random(len(df)) < 0.5
    
    # Initialize Player1/Player2 columns
    df['Player1_ID'] = df['winner_id'].copy()
    df['Player1_Name'] = df['winner_name'].copy()
    df['Player1_Hand'] = df['winner_hand'].copy()
    df['Player1_Height'] = df['winner_ht'].copy()
    df['Player1_IOC'] = df['winner_ioc'].copy()
    df['Player1_Age'] = df['winner_age'].copy()
    df['Player1_Seed'] = df['winner_seed'].copy()
    df['Player1_Entry'] = df['winner_entry'].copy()
    df['Player1_Rank'] = df['winner_rank'].copy()
    df['Player1_Rank_Points'] = df['winner_rank_points'].copy()
    
    df['Player2_ID'] = df['loser_id'].copy()
    df['Player2_Name'] = df['loser_name'].copy()
    df['Player2_Hand'] = df['loser_hand'].copy()
    df['Player2_Height'] = df['loser_ht'].copy()
    df['Player2_IOC'] = df['loser_ioc'].copy()
    df['Player2_Age'] = df['loser_age'].copy()
    df['Player2_Seed'] = df['loser_seed'].copy()
    df['Player2_Entry'] = df['loser_entry'].copy()
    df['Player2_Rank'] = df['loser_rank'].copy()
    df['Player2_Rank_Points'] = df['loser_rank_points'].copy()
    
    df['Player1_Wins'] = 1  # Winner always wins initially
    
    # Swap players where swap_mask is True
    df.loc[swap_mask, 'Player1_ID'] = df.loc[swap_mask, 'loser_id']
    df.loc[swap_mask, 'Player1_Name'] = df.loc[swap_mask, 'loser_name']
    df.loc[swap_mask, 'Player1_Hand'] = df.loc[swap_mask, 'loser_hand']
    df.loc[swap_mask, 'Player1_Height'] = df.loc[swap_mask, 'loser_ht']
    df.loc[swap_mask, 'Player1_IOC'] = df.loc[swap_mask, 'loser_ioc']
    df.loc[swap_mask, 'Player1_Age'] = df.loc[swap_mask, 'loser_age']
    df.loc[swap_mask, 'Player1_Seed'] = df.loc[swap_mask, 'loser_seed']
    df.loc[swap_mask, 'Player1_Entry'] = df.loc[swap_mask, 'loser_entry']
    df.loc[swap_mask, 'Player1_Rank'] = df.loc[swap_mask, 'loser_rank']
    df.loc[swap_mask, 'Player1_Rank_Points'] = df.loc[swap_mask, 'loser_rank_points']
    
    df.loc[swap_mask, 'Player2_ID'] = df.loc[swap_mask, 'winner_id']
    df.loc[swap_mask, 'Player2_Name'] = df.loc[swap_mask, 'winner_name']
    df.loc[swap_mask, 'Player2_Hand'] = df.loc[swap_mask, 'winner_hand']
    df.loc[swap_mask, 'Player2_Height'] = df.loc[swap_mask, 'winner_ht']
    df.loc[swap_mask, 'Player2_IOC'] = df.loc[swap_mask, 'winner_ioc']
    df.loc[swap_mask, 'Player2_Age'] = df.loc[swap_mask, 'winner_age']
    df.loc[swap_mask, 'Player2_Seed'] = df.loc[swap_mask, 'winner_seed']
    df.loc[swap_mask, 'Player2_Entry'] = df.loc[swap_mask, 'winner_entry']
    df.loc[swap_mask, 'Player2_Rank'] = df.loc[swap_mask, 'winner_rank']
    df.loc[swap_mask, 'Player2_Rank_Points'] = df.loc[swap_mask, 'winner_rank_points']
    
    df.loc[swap_mask, 'Player1_Wins'] = 0  # If we swapped, Player1 lost
    
    print(f"   Swapped {swap_mask.sum()} matches ({swap_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Player1 wins: {df['Player1_Wins'].sum()} ({df['Player1_Wins'].mean()*100:.1f}%)")
    print(f"   Player2 wins: {(1-df['Player1_Wins']).sum()} ({(1-df['Player1_Wins']).mean()*100:.1f}%)")

    if feature_semantics_id == SHARED_SEMANTICS_ID:
        # Normalize raw profile evidence before any derived arithmetic so the
        # full historical adapter follows the same finite missingness contract
        # as live serving.
        for prefix in ('Player1', 'Player2'):
            snapshots = [
                normalize_player_snapshot(
                    height=height,
                    age=age,
                    rank=rank,
                    rank_points=points,
                    hand=hand,
                    country=country,
                )
                for height, age, rank, points, hand, country in zip(
                    df[f'{prefix}_Height'],
                    df[f'{prefix}_Age'],
                    df[f'{prefix}_Rank'],
                    df[f'{prefix}_Rank_Points'],
                    df[f'{prefix}_Hand'],
                    df[f'{prefix}_IOC'],
                )
            ]
            df[f'{prefix}_Height'] = [item['height'] for item in snapshots]
            df[f'{prefix}_Age'] = [item['age'] for item in snapshots]
            df[f'{prefix}_Rank'] = [item['rank'] for item in snapshots]
            df[f'{prefix}_Rank_Points'] = [item['rank_points'] for item in snapshots]
            df[f'{prefix}_Hand'] = [item['hand'] for item in snapshots]
            df[f'{prefix}_IOC'] = [item['country'] for item in snapshots]

        df['surface'] = df['surface'].where(df['surface'].notna(), 'Hard')
        df['surface'] = df['surface'].astype(str).str.strip().replace('', 'Hard').str.title()
        df['tourney_level'] = df['tourney_level'].where(df['tourney_level'].notna(), 'A')
        df['tourney_level'] = df['tourney_level'].astype(str).str.strip().replace('', 'A').str.upper()
        df['draw_size'] = pd.to_numeric(df['draw_size'], errors='coerce').fillna(32).astype(int)
    
    # 4. Create basic derived features
    print("\n4. Creating basic derived features...")
    
    # Ranking features
    df['Rank_Diff'] = df['Player1_Rank'] - df['Player2_Rank']  # Negative means Player1 better ranked
    df['Player1_Rank_Advantage'] = (df['Player1_Rank'] < df['Player2_Rank']).astype(int)
    df['Avg_Rank'] = (df['Player1_Rank'] + df['Player2_Rank']) / 2
    df['Rank_Ratio'] = np.maximum(df['Player1_Rank'], df['Player2_Rank']) / np.minimum(df['Player1_Rank'], df['Player2_Rank'])
    
    # Height features
    df['Height_Diff'] = df['Player1_Height'] - df['Player2_Height']  # Positive means Player1 taller
    df['Player1_Height_Advantage'] = (df['Player1_Height'] > df['Player2_Height']).astype(int)
    df['Avg_Height'] = (df['Player1_Height'] + df['Player2_Height']) / 2
    
    # Age features
    df['Age_Diff'] = df['Player1_Age'] - df['Player2_Age']  # Positive means Player1 older
    df['Player1_Age_Advantage'] = (df['Player1_Age'] < df['Player2_Age']).astype(int)  # Younger is better
    df['Avg_Age'] = (df['Player1_Age'] + df['Player2_Age']) / 2
    
    # Ranking points features (if available)
    if 'Player1_Rank_Points' in df.columns and 'Player2_Rank_Points' in df.columns:
        df['Rank_Points_Diff'] = df['Player1_Rank_Points'] - df['Player2_Rank_Points']
        df['Player1_Points_Advantage'] = (df['Player1_Rank_Points'] > df['Player2_Rank_Points']).astype(int)
        df['Avg_Rank_Points'] = (df['Player1_Rank_Points'] + df['Player2_Rank_Points']) / 2
    
    # 5. One-hot encode categorical features
    print("\n5. One-hot encoding categorical features...")
    
    # Surface encoding
    if 'surface' in df.columns:
        surface_dummies = pd.get_dummies(df['surface'], prefix='Surface')
        df = pd.concat([df, surface_dummies], axis=1)
        surface_features = list(surface_dummies.columns)
    else:
        surface_features = []
    
    # Tournament level encoding
    if 'tourney_level' in df.columns:
        level_dummies = pd.get_dummies(df['tourney_level'], prefix='Level')
        df = pd.concat([df, level_dummies], axis=1)
        level_features = list(level_dummies.columns)
    else:
        level_features = []
    
    # Round encoding
    if 'round' in df.columns:
        round_dummies = pd.get_dummies(df['round'], prefix='Round')
        df = pd.concat([df, round_dummies], axis=1)
        round_features = list(round_dummies.columns)
    else:
        round_features = []
    
    # Handedness encoding
    hand_dummies = pd.get_dummies(df['Player1_Hand'], prefix='P1_Hand')
    df = pd.concat([df, hand_dummies], axis=1)
    hand_dummies_p2 = pd.get_dummies(df['Player2_Hand'], prefix='P2_Hand')
    df = pd.concat([df, hand_dummies_p2], axis=1)
    hand_features = list(hand_dummies.columns) + list(hand_dummies_p2.columns)
    
    # Country encoding (simplified - only major tennis countries)
    if 'Player1_IOC' in df.columns:
        major_countries = ['USA', 'ESP', 'FRA', 'GER', 'ITA', 'GBR', 'ARG', 'AUS', 'SUI', 'RUS', 'SRB', 'CZE']
        df['Player1_Major_Country'] = df['Player1_IOC'].apply(lambda x: x if x in major_countries else 'Other')
        df['Player2_Major_Country'] = df['Player2_IOC'].apply(lambda x: x if x in major_countries else 'Other')
        
        country_dummies_p1 = pd.get_dummies(df['Player1_Major_Country'], prefix='P1_Country')
        df = pd.concat([df, country_dummies_p1], axis=1)
        country_dummies_p2 = pd.get_dummies(df['Player2_Major_Country'], prefix='P2_Country')
        df = pd.concat([df, country_dummies_p2], axis=1)
        country_features = list(country_dummies_p1.columns) + list(country_dummies_p2.columns)
    else:
        country_features = []
    
    print(f"   Created {len(surface_features)} surface features")
    print(f"   Created {len(level_features)} tournament level features")
    print(f"   Created {len(round_features)} round features")
    print(f"   Created {len(hand_features)} handedness features")
    print(f"   Created {len(country_features)} country features")
    
    # 6. Infer per-match dates using round day offsets, then calculate temporal features
    print("\n6. Inferring match dates from round + tournament type...")
    # Sackmann stores tournament START DATE for every round. We derive an estimated
    # actual match date so that within-tournament rounds are processed in the correct
    # chronological order and rolling windows (30d/90d) are computed accurately.

    # Pre-compute number of qualifier rounds per tournament so qualifier offsets are accurate.
    # e.g. if a tournament only has Q1,Q2 → Q1=-2,Q2=-1 (not Q1=-3,Q2=-2,Q3=-1)
    _qual_rounds_per_tourney = df.groupby('tourney_id')['round'].apply(
        lambda rounds: sum(1 for r in rounds.unique()
                           if str(r).upper().startswith('Q') and str(r).upper() not in ('QF',))
    ).to_dict()
    df['_num_qual_rounds'] = df['tourney_id'].map(_qual_rounds_per_tourney).fillna(0).astype(int)
    print(f"   Qualifier rounds per tournament: min={df['_num_qual_rounds'].min()}, "
          f"max={df['_num_qual_rounds'].max()}, "
          f"mean={df['_num_qual_rounds'].mean():.1f}")

    df['inferred_match_date'] = df.apply(
        lambda row: row['tourney_date'] + pd.Timedelta(days=get_round_day_offset(
            row.get('tourney_level'), row.get('draw_size'), row.get('round'),
            tourney_date=row['tourney_date'],
            num_qual_rounds=row.get('_num_qual_rounds') or None
        )),
        axis=1
    )
    df.drop(columns=['_num_qual_rounds'], inplace=True)
    print(f"   Sample inferred dates (first 3 rows):")
    for _, r in df[['tourney_name', 'round', 'tourney_date', 'inferred_match_date']].head(3).iterrows():
        print(f"     {r['tourney_name']} {r['round']}: {r['tourney_date'].date()} -> {r['inferred_match_date'].date()}")

    performance_features = []
    if include_performance_features:
        print("\n   Calculating performance feature-set add-ons...")
        from performance_features import PERFORMANCE_FEATURES, add_performance_features

        df = add_performance_features(df)
        performance_features = [col for col in PERFORMANCE_FEATURES if col in df.columns]
        print(f"   Added {len(performance_features)} score/stat performance features")

    print("\n   Calculating enhanced temporal features...")
    print("   This is the major enhancement - adding 60+ temporal features!")

    df_with_temporal = calculate_temporal_features(
        df,
        feature_semantics_id=feature_semantics_id,
    )
    
    # 7. Select final features for ML
    print("\n7. Selecting final ML features...")
    
    # Core features (player attributes and derived features)
    core_features = [
        'Player1_Rank', 'Player2_Rank', 'Rank_Diff', 'Player1_Rank_Advantage',
        'Avg_Rank', 'Rank_Ratio',
        'Player1_Height', 'Player2_Height', 'Height_Diff', 'Player1_Height_Advantage', 'Avg_Height',
        'Player1_Age', 'Player2_Age', 'Age_Diff', 'Player1_Age_Advantage', 'Avg_Age',
        'draw_size', 'best_of'
    ]
    
    # Add ranking points features if available
    if 'Player1_Rank_Points' in df_with_temporal.columns:
        core_features.extend(['Player1_Rank_Points', 'Player2_Rank_Points', 'Rank_Points_Diff', 
                             'Player1_Points_Advantage', 'Avg_Rank_Points'])
    
    # Get all temporal features
    temporal_features = [col for col in df_with_temporal.columns if any(x in col for x in [
        'Days_Since_Last', 'Rust_Flag', 'Matches_', 'Sets_', 'Rank_Change', 'Rank_Volatility',
        'Surface_Matches', 'Surface_WinRate', 'Surface_Experience', 'Level_WinRate', 'Level_Matches',
        'BigMatch_WinRate', 'Round_WinRate', 'Finals_WinRate', 'Semifinals_WinRate',
        'H2H_', 'Handedness_Matchup', 'vs_Lefty_WinRate', 'Clay_Season', 'Grass_Season',
        'Indoor_Season', 'Surface_Transition_Flag', 'WinRate_Last10', 'WinStreak_Current',
        'Form_Trend', 'Peak_Age', 'Momentum_Diff'
    ])]
    
    print(f"   Added {len(temporal_features)} temporal features!")
    
    # Combine all features
    feature_columns = (
        core_features
        + surface_features
        + level_features
        + round_features
        + hand_features
        + country_features
        + temporal_features
        + performance_features
    )
    
    # Keep only features that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df_with_temporal.columns]

    if feature_semantics_id == SHARED_SEMANTICS_ID:
        with open(os.path.join(_FEATURES_DIR, 'schema_141.json')) as schema_file:
            shared_schema = json.load(schema_file)
        ordered_shared_features = [item['name'] for item in shared_schema['features']]
        structural_zero_prefixes = (
            'Level_', 'Round_', 'P1_Hand_', 'P2_Hand_',
            'P1_Country_', 'P2_Country_', 'Handedness_Matchup_',
        )
        structural_zero_fields = {
            name for name in ordered_shared_features
            if name.startswith(structural_zero_prefixes)
            or (name.startswith('Surface_') and name != 'Surface_Transition_Flag')
        }
        absent_structural_fields = sorted(
            name for name in structural_zero_fields
            if name not in df_with_temporal.columns
        )
        if absent_structural_fields:
            structural_defaults = pd.DataFrame(
                0,
                index=df_with_temporal.index,
                columns=absent_structural_fields,
            )
            df_with_temporal = pd.concat(
                [df_with_temporal, structural_defaults], axis=1
            )
        missing_shared_features = [
            name for name in ordered_shared_features if name not in df_with_temporal.columns
        ]
        if missing_shared_features:
            raise ValueError(
                "base_141_shared historical candidate is missing ordered fields: "
                + ",".join(missing_shared_features)
            )
        for name in ordered_shared_features:
            numeric = pd.to_numeric(df_with_temporal[name], errors='coerce')
            numeric = numeric.replace([np.inf, -np.inf], np.nan)
            df_with_temporal[name] = numeric.fillna(shared_feature_default(name))

        one_hot_groups = {
            'surface': [
                name for name in ordered_shared_features
                if name.startswith('Surface_') and name != 'Surface_Transition_Flag'
            ],
            'level': [name for name in ordered_shared_features if name.startswith('Level_')],
            'round': [name for name in ordered_shared_features if name.startswith('Round_')],
            'p1_hand': [name for name in ordered_shared_features if name.startswith('P1_Hand_')],
            'p2_hand': [name for name in ordered_shared_features if name.startswith('P2_Hand_')],
            'p1_country': [name for name in ordered_shared_features if name.startswith('P1_Country_')],
            'p2_country': [name for name in ordered_shared_features if name.startswith('P2_Country_')],
            'hand_matchup': [
                name for name in ordered_shared_features
                if name.startswith('Handedness_Matchup_')
            ],
        }
        for label, names in one_hot_groups.items():
            values = df_with_temporal[names]
            nonbinary = ~values.isin([0.0, 1.0]).all(axis=1)
            totals = values.sum(axis=1)
            bad_cardinality = (
                ~totals.isin([0.0, 1.0])
                if label == 'hand_matchup' else totals.ne(1.0)
            )
            invalid = nonbinary | bad_cardinality
            if invalid.any():
                bad_rows = df_with_temporal.index[invalid].tolist()[:10]
                raise ValueError(
                    f"base_141_shared invalid {label} one-hot rows: {bad_rows}"
                )
        feature_columns = ordered_shared_features
    
    # Add metadata columns for tracking
    metadata_columns = [
        'tourney_date', 'tourney_name', 'tourney_level', 'surface', 'round',
        'Player1_Name', 'Player2_Name', 'Player1_Rank', 'Player2_Rank'
    ]
    if feature_semantics_id == SHARED_SEMANTICS_ID:
        metadata_columns = [name for name in metadata_columns if name not in feature_columns]
    
    # Target variable
    target_column = 'Player1_Wins'
    
    # Create final dataset
    final_columns = feature_columns + metadata_columns + [target_column]
    final_columns = [col for col in final_columns if col in df_with_temporal.columns]
    
    ml_df = df_with_temporal[final_columns].copy()
    
    print(f"   Selected {len(feature_columns)} total features for ML:")
    print(f"   - Core features: {len(core_features)}")
    print(f"   - Surface features: {len(surface_features)}")
    print(f"   - Level features: {len(level_features)}")
    print(f"   - Round features: {len(round_features)}")
    print(f"   - Hand features: {len(hand_features)}")
    print(f"   - Country features: {len(country_features)}")
    print(f"   - TEMPORAL features: {len(temporal_features)}")
    
    # 8. Save the enhanced ML-ready dataset
    print("\n8. Saving enhanced ML-ready dataset...")
    
    if output_path is None:
        output_path = os.path.join(_base, "jeffsackmann_ml_ready_SURFACE_FIX.csv")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ml_df.to_csv(output_path, index=False)
    
    if feature_semantics_id == SHARED_SEMANTICS_ID:
        print(f"   ✅ Saved reserved shared-semantics candidate side output: {output_path}")
    elif include_performance_features:
        print(f"   ✅ Saved side feature-set dataset: {output_path}")
    else:
        print(f"   ✅ REPLACED old dataset with enhanced version: {output_path}")
    print(f"   Final dataset shape: {ml_df.shape}")
    
    # 9. Summary statistics
    print("\n" + "=" * 60)
    print("ENHANCED JEFFSACKMANN PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original matches: {initial_count:,}")
    print(f"Final ML dataset: {len(ml_df):,}")
    print(f"Total features for training: {len(feature_columns)}")
    print(f"  -> Basic features: {len(core_features + surface_features + level_features + round_features + hand_features + country_features)}")
    print(f"  -> NEW temporal features: {len(temporal_features)}")
    if include_performance_features:
        print(f"  -> Performance feature-set add-ons: {len(performance_features)}")
    print(f"Target balance: {ml_df['Player1_Wins'].mean()*100:.1f}% Player1 wins")
    print(f"Data spans: {ml_df['tourney_date'].min().date()} to {ml_df['tourney_date'].max().date()}")
    if 'year' in ml_df.columns:
        print(f"Years covered: {ml_df['year'].min()} to {ml_df['year'].max()}")
    
    # Show sample of NEW temporal features
    print(f"\nSample of NEW temporal features:")
    for i, feature in enumerate(temporal_features[:20]):
        print(f"  {i+1:2d}. {feature}")
    if len(temporal_features) > 20:
        print(f"  ... and {len(temporal_features)-20} more temporal features")
    
    print("\n✅ Enhanced Jeff Sackmann ML preprocessing complete!")
    print("MAJOR IMPROVEMENTS:")
    print("1. Added 60+ temporal features based on AI recommendations")
    print("2. Ranking momentum, form trends, surface adaptation")
    print("3. Head-to-head patterns, fatigue metrics, seasonal factors")
    print("4. Expected improvement: +3-6% accuracy (64.7% → 67-70%)")
    print("\nNext steps:")
    print("1. Run enhanced models and compare to baseline")
    print("2. Expected significant improvement over ATP ranking baseline")
    print("3. Models should now capture temporal patterns rankings miss")
    
    return ml_df, feature_columns

if __name__ == "__main__":
    # Run enhanced preprocessing - replaces old ML dataset
    ml_df, feature_columns = preprocess_jeffsackmann_data_for_ml()
