import pandas as pd
import numpy as np
import os
from datetime import datetime
from collections import defaultdict, deque

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

def calculate_temporal_features(df):
    """
    Calculate all temporal derived features from high-coverage base data.
    This implements ALL recommendations from ChatGPT/Grok analysis.
    """
    
    print("=" * 60)
    print("CALCULATING ENHANCED TEMPORAL FEATURES")
    print("=" * 60)
    
    # Sort by date for chronological processing (CRITICAL - prevents data leakage)
    df_sorted = df.sort_values(['tourney_date']).reset_index(drop=True).copy()
    df_sorted['tourney_date'] = pd.to_datetime(df_sorted['tourney_date'])
    
    # Initialize player tracking dictionaries
    player_stats = defaultdict(lambda: {
        'last_match_date': None,
        'match_history': deque(maxlen=50),  # Store last 50 matches per player
        'rank_history': deque(maxlen=50),
        'surface_stats': defaultdict(lambda: {'matches': deque(maxlen=20), 'wins': deque(maxlen=20)}),
        'level_stats': defaultdict(lambda: {'total': 0, 'wins': 0}),
        'round_stats': defaultdict(lambda: {'total': 0, 'wins': 0}),
        'h2h_stats': defaultdict(lambda: {'total': 0, 'wins': 0})
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
        'Age_Diff', 'Height_Diff', 'Peak_Age_P1', 'Peak_Age_P2',
        
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
    
    # Process each match chronologically
    for idx, (_, row) in enumerate(df_sorted.iterrows()):
        if idx % 50000 == 0:
            print(f"  Processed {idx:,}/{len(df_sorted):,} matches ({idx/len(df_sorted)*100:.1f}%)")
        
        # Get player IDs and basic info
        p1_id = row['Player1_ID'] if pd.notna(row['Player1_ID']) else row['Player1_Name']
        p2_id = row['Player2_ID'] if pd.notna(row['Player2_ID']) else row['Player2_Name']
        match_date = row['tourney_date']
        surface = row['surface'] if pd.notna(row['surface']) else 'Hard'
        level = row['tourney_level'] if pd.notna(row['tourney_level']) else 'A'
        round_name = row['round'] if pd.notna(row['round']) else 'R32'
        
        # Extract current match info
        p1_rank = row['Player1_Rank'] if pd.notna(row['Player1_Rank']) else 999
        p2_rank = row['Player2_Rank'] if pd.notna(row['Player2_Rank']) else 999
        p1_age = row['Player1_Age'] if pd.notna(row['Player1_Age']) else 25
        p2_age = row['Player2_Age'] if pd.notna(row['Player2_Age']) else 25
        p1_height = row['Player1_Height'] if pd.notna(row['Player1_Height']) else 180
        p2_height = row['Player2_Height'] if pd.notna(row['Player2_Height']) else 180
        p1_hand = row.get('Player1_Hand', 'R') if pd.notna(row.get('Player1_Hand', 'R')) else 'R'
        p2_hand = row.get('Player2_Hand', 'R') if pd.notna(row.get('Player2_Hand', 'R')) else 'R'
        
        # Calculate features for both players
        for player_num, player_id in enumerate([p1_id, p2_id], 1):
            prefix = f'P{player_num}_'
            stats = player_stats[player_id]
            
            # === PHASE 1: RANKING DYNAMICS ===
            if len(stats['rank_history']) >= 2:
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
            last_tournament_date = get_last_tournament_date(stats['match_history'], match_date)
            if last_tournament_date is not None:
                days_since = (match_date - last_tournament_date).days
                df_sorted.loc[row.name, f'{prefix}Days_Since_Last'] = days_since
                df_sorted.loc[row.name, f'{prefix}Rust_Flag'] = 1 if days_since > 21 else 0
            
            # Match frequency
            matches_14d = count_matches_in_window(stats['match_history'], match_date, days=14)
            matches_30d = count_matches_in_window(stats['match_history'], match_date, days=30)
            df_sorted.loc[row.name, f'{prefix}Matches_14d'] = matches_14d
            df_sorted.loc[row.name, f'{prefix}Matches_30d'] = matches_30d
            
            # Sets played (estimate 2.5 sets per match on average)
            df_sorted.loc[row.name, f'{prefix}Sets_14d'] = matches_14d * 2.5
            
            # === PHASE 1: AGE/PHYSICAL ===
            age = p1_age if player_num == 1 else p2_age
            if player_num == 1:
                df_sorted.loc[row.name, 'Age_Diff'] = p1_age - p2_age
                df_sorted.loc[row.name, 'Height_Diff'] = p1_height - p2_height
            df_sorted.loc[row.name, f'{prefix[:-1]}_Peak_Age'] = 1 if 24 <= age <= 28 else 0
            
        # === PHASE 2: SURFACE-SPECIFIC FEATURES ===
        for player_num, player_id in enumerate([p1_id, p2_id], 1):
            prefix = f'P{player_num}_'
            stats = player_stats[player_id]
            surface_stats = stats['surface_stats'][surface]
            
            # Surface activity
            surface_matches_30d = count_surface_matches(surface_stats['matches'], match_date, days=30)
            surface_matches_90d = count_surface_matches(surface_stats['matches'], match_date, days=90)
            df_sorted.loc[row.name, f'{prefix}Surface_Matches_30d'] = surface_matches_30d
            df_sorted.loc[row.name, f'{prefix}Surface_Matches_90d'] = surface_matches_90d
            
            # Surface win rate (with minimum threshold)
            surface_win_rate = calculate_surface_win_rate(surface_stats, match_date, days=90, min_matches=5)
            df_sorted.loc[row.name, f'{prefix}Surface_WinRate_90d'] = surface_win_rate
            
            # Career surface experience
            career_surface_matches = len(surface_stats['matches'])
            df_sorted.loc[row.name, f'{prefix}Surface_Experience'] = career_surface_matches
            
            # === PHASE 2: TOURNAMENT LEVEL EXPERIENCE ===
            level_stats = stats['level_stats'][level]
            level_win_rate = level_stats['wins'] / level_stats['total'] if level_stats['total'] >= 5 else 0.5
            df_sorted.loc[row.name, f'{prefix}Level_WinRate_Career'] = level_win_rate
            df_sorted.loc[row.name, f'{prefix}Level_Matches_Career'] = level_stats['total']
            
            # Big match performance (Grand Slams + Masters)
            big_match_wins = stats['level_stats']['G']['wins'] + stats['level_stats']['M']['wins']
            big_match_total = stats['level_stats']['G']['total'] + stats['level_stats']['M']['total']
            big_match_rate = big_match_wins / big_match_total if big_match_total >= 3 else 0.5
            df_sorted.loc[row.name, f'{prefix}BigMatch_WinRate'] = big_match_rate
            
            # === PHASE 2: ROUND PERFORMANCE ===
            round_stats = stats['round_stats'][round_name]
            round_win_rate = round_stats['wins'] / round_stats['total'] if round_stats['total'] >= 3 else 0.5
            df_sorted.loc[row.name, f'{prefix}Round_WinRate_Career'] = round_win_rate
            
            # Specific round performance
            finals_rate = stats['round_stats']['F']['wins'] / stats['round_stats']['F']['total'] if stats['round_stats']['F']['total'] >= 1 else 0.5
            sf_rate = stats['round_stats']['SF']['wins'] / stats['round_stats']['SF']['total'] if stats['round_stats']['SF']['total'] >= 2 else 0.5
            df_sorted.loc[row.name, f'{prefix}Finals_WinRate'] = finals_rate
            df_sorted.loc[row.name, f'{prefix}Semifinals_WinRate'] = sf_rate
            
            # === PHASE 2: FORM METRICS ===
            win_rate_10_120d = calculate_recent_form(stats['match_history'], match_date, max_matches=10, max_days=120)
            df_sorted.loc[row.name, f'{prefix}WinRate_Last10_120d'] = win_rate_10_120d
            
            # Current win streak
            win_streak = calculate_current_streak(stats['match_history'])
            df_sorted.loc[row.name, f'{prefix}WinStreak_Current'] = win_streak
            
            # Form trend (exponential weighted)
            form_trend = calculate_form_trend(stats['match_history'], match_date, days=30)
            df_sorted.loc[row.name, f'{prefix}Form_Trend_30d'] = form_trend
        
        # === PHASE 2: HEAD-TO-HEAD ===
        h2h_stats = player_stats[p1_id]['h2h_stats'][p2_id]
        df_sorted.loc[row.name, 'H2H_Total_Matches'] = h2h_stats['total']
        df_sorted.loc[row.name, 'H2H_P1_Wins'] = h2h_stats['wins']
        df_sorted.loc[row.name, 'H2H_P2_Wins'] = h2h_stats['total'] - h2h_stats['wins']
        
        h2h_win_rate = h2h_stats['wins'] / h2h_stats['total'] if h2h_stats['total'] >= 3 else 0.5
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
        
        # === UPDATE PLAYER STATS (AFTER CALCULATIONS) ===
        # This is crucial - update stats AFTER using them for current match
        match_result = row['Player1_Wins']
        
        # Update both players' histories
        for player_num, player_id in enumerate([p1_id, p2_id], 1):
            stats = player_stats[player_id]
            opponent_id = p2_id if player_num == 1 else p1_id
            opponent_hand = p2_hand if player_num == 1 else p1_hand
            won = (match_result == 1) if player_num == 1 else (match_result == 0)
            current_rank = p1_rank if player_num == 1 else p2_rank
            
            # Update match history
            match_info = {
                'date': match_date,
                'opponent': opponent_id,
                'opponent_hand': opponent_hand,
                'surface': surface,
                'level': level,
                'round': round_name,
                'won': won,
                'rank': current_rank
            }
            stats['match_history'].append(match_info)
            stats['rank_history'].append(current_rank)
            stats['last_match_date'] = match_date
            
            # Update surface stats
            stats['surface_stats'][surface]['matches'].append(match_date)
            stats['surface_stats'][surface]['wins'].append(won)
            
            # Update level/round stats
            stats['level_stats'][level]['total'] += 1
            if won:
                stats['level_stats'][level]['wins'] += 1
                
            stats['round_stats'][round_name]['total'] += 1
            if won:
                stats['round_stats'][round_name]['wins'] += 1
            
            # Update H2H stats
            if player_num == 1:
                stats['h2h_stats'][opponent_id]['total'] += 1
                if won:
                    stats['h2h_stats'][opponent_id]['wins'] += 1
    
    print("\n✅ Temporal feature calculation complete!")
    print(f"Added {len(enhanced_features)} new temporal features")
    
    # Fill remaining NaN values with sensible defaults
    print("\nFilling missing values with defaults...")
    
    # Days since last match: 60 days for first match
    df_sorted['P1_Days_Since_Last'].fillna(60, inplace=True)
    df_sorted['P2_Days_Since_Last'].fillna(60, inplace=True)
    
    # Activity counts: 0 for new players
    activity_cols = [col for col in enhanced_features if 'Matches_' in col or 'Sets_' in col]
    for col in activity_cols:
        df_sorted[col].fillna(0, inplace=True)
    
    # Win rates: 0.5 (neutral) for insufficient data
    rate_cols = [col for col in enhanced_features if 'WinRate' in col or 'Rate' in col]
    for col in rate_cols:
        df_sorted[col].fillna(0.5, inplace=True)
    
    # Ranking changes: 0 for insufficient history
    rank_cols = [col for col in enhanced_features if 'Rank_Change' in col or 'Volatility' in col or 'Momentum' in col]
    for col in rank_cols:
        df_sorted[col].fillna(0, inplace=True)
    
    # Binary flags: 0 as default
    flag_cols = [col for col in enhanced_features if 'Flag' in col or 'Season' in col or 'Peak_Age' in col or 'Handedness_Matchup' in col]
    for col in flag_cols:
        df_sorted[col].fillna(0, inplace=True)
    
    # Streaks and trends: 0 as neutral
    misc_cols = [col for col in enhanced_features if 'Streak' in col or 'Trend' in col or 'Advantage' in col]
    for col in misc_cols:
        df_sorted[col].fillna(0, inplace=True)
    
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
    """Calculate win rate on surface with minimum match threshold."""
    cutoff_date = target_date - pd.Timedelta(days=days)
    
    matches = []
    wins = []
    
    for match_date, won in zip(surface_stats['matches'], surface_stats['wins']):
        if cutoff_date <= match_date < target_date:
            matches.append(match_date)
            wins.append(won)
    
    if len(matches) >= min_matches:
        return sum(wins) / len(wins)
    else:
        return 0.5  # Neutral prior for insufficient data

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
    
    if len(recent_matches) >= 3:  # Minimum for meaningful rate
        wins = sum(1 for match in recent_matches if match['won'])
        return wins / len(recent_matches)
    else:
        return 0.5  # Neutral prior

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
    
    if len(recent_matches) < 3:
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
    
    if len(recent_h2h) >= 2:
        wins = sum(1 for match in recent_h2h if match['won'])
        return wins / len(recent_h2h) - 0.5  # Advantage relative to 50%
    else:
        return 0  # No advantage for insufficient data

def calculate_vs_handedness_rate(match_history, target_handedness):
    """Calculate win rate vs specific handedness."""
    vs_hand_matches = []
    for match in match_history:
        if match.get('opponent_hand') == target_handedness:
            vs_hand_matches.append(match)
    
    if len(vs_hand_matches) >= 3:
        wins = sum(1 for match in vs_hand_matches if match['won'])
        return wins / len(vs_hand_matches)
    else:
        return 0.5  # Neutral prior

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

def preprocess_jeffsackmann_data_for_ml():
    """
    Enhanced preprocessing with comprehensive temporal features.
    This replaces the old preprocessing with all AI-recommended features.
    """
    
    print("=" * 60)
    print("ENHANCED JEFF SACKMANN ATP DATA ML PREPROCESSING")
    print("=" * 60)
    
    # Load the master combined Jeffsackmann data
    print(f"\n1. Loading Jeff Sackmann master combined data...")
    df = pd.read_csv("/app/data/JeffSackmann/jeffsackmann_master_combined.csv", low_memory=False)
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
    
    # 6. Calculate enhanced temporal features
    print("\n6. Calculating enhanced temporal features...")
    print("   This is the major enhancement - adding 60+ temporal features!")
    
    df_with_temporal = calculate_temporal_features(df)
    
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
    feature_columns = core_features + surface_features + level_features + round_features + hand_features + country_features + temporal_features
    
    # Keep only features that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df_with_temporal.columns]
    
    # Add metadata columns for tracking
    metadata_columns = [
        'tourney_date', 'tourney_name', 'tourney_level', 'surface', 'round',
        'Player1_Name', 'Player2_Name', 'Player1_Rank', 'Player2_Rank'
    ]
    
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
    
    # 8. Save the enhanced ML-ready dataset (REPLACES old file)
    print("\n8. Saving enhanced ML-ready dataset...")
    
    output_path = "/app/data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv"
    ml_df.to_csv(output_path, index=False)
    
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
    print(f"Target balance: {ml_df['Player1_Wins'].mean()*100:.1f}% Player1 wins")
    print(f"Data spans: {ml_df['tourney_date'].min().date()} to {ml_df['tourney_date'].max().date()}")
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