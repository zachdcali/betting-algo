"""
Dynamic round-day offset heuristic — single source of truth for training and production.

Anchored on "finals are always Sunday": computes the offset from tournament start date
to each round's approximate match date, adapting to non-Monday starts.
"""

import pandas as pd
from datetime import datetime


# Days BEFORE the final for each round, by tournament category
_ROUND_RELATIVE = {
    'G': {'R128': -13, 'R64': -11, 'R32': -9, 'R16': -6, 'QF': -4, 'SF': -2, 'F': 0, 'BR': 0},
    'M_large': {'R128': -11, 'R64': -9, 'R32': -7, 'R16': -5, 'QF': -4, 'SF': -2, 'F': 0, 'BR': 0},
    # M/64 2-week style (Madrid, Rome, Shanghai — non-Monday starts)
    'M_medium_long': {'R64': -12, 'R32': -10, 'R16': -8, 'QF': -4, 'SF': -2, 'F': 0, 'BR': 0},
    # M/64 1-week style (Canada, Cincinnati — Monday starts)
    'M_medium_short': {'R64': -6, 'R32': -4, 'R16': -3, 'QF': -2, 'SF': -1, 'F': 0, 'BR': 0},
    'M_small': {'R64': -6, 'R32': -5, 'R16': -3, 'QF': -2, 'SF': -1, 'F': 0, 'BR': 0},
    'standard': {'R32': -6, 'R16': -4, 'QF': -2, 'SF': -1, 'F': 0, 'BR': 0},
}

# Qualifier offsets: days relative to main-draw start (tourney_date), NOT final-relative
_QUAL_OFFSETS = {
    'G': {'Q1': -7, 'Q2': -5, 'Q3': -3, 'Q4': -3},
    'M': {'Q1': -2, 'Q2': -1},
    'C': {'Q1': -3, 'Q2': -2, 'Q3': -1, 'Q4': -1},
    'default': {'Q1': -3, 'Q2': -2, 'Q3': -1, 'Q4': -1},
}

# Static offsets (returned directly, not final-relative)
_STATIC = {
    'O': {'RR': 1, 'SF': 6, 'F': 7},
    'D': {'RR': 0, 'QF': 0, 'SF': 0, 'F': 2, 'BR': 2},
}


def get_round_day_offset(tourney_level, draw_size, round_code, tourney_date=None,
                         num_qual_rounds=None) -> int:
    """
    Estimate the day offset from tourney_date for a given round.

    Uses a "finals are always Sunday" anchor: computes the number of days from
    tournament start to the final (based on day-of-week of start date), then
    applies round-relative offsets backwards from the final.

    Args:
        tourney_level: Tournament level code (G, M, A, C, S, O, D, etc.)
        draw_size: Draw size (128, 96, 64, 48, 32, etc.)
        round_code: Round code (R128, R64, R32, R16, QF, SF, F, Q1, Q2, etc.)
        tourney_date: Tournament start date (datetime or pd.Timestamp). If None,
                      assumes Monday start (dow=0) as fallback.
        num_qual_rounds: Number of qualifier rounds for this tournament (1, 2, or 3).
                         If provided, qualifier offsets are computed dynamically:
                         each round is 1 day, counting backwards from day -1.
                         If None, uses level-based defaults.

    Returns:
        Integer day offset from tourney_date to approximate match date.
    """
    level = str(tourney_level).strip().upper() if pd.notna(tourney_level) else ''
    try:
        draw = int(draw_size)
    except (TypeError, ValueError):
        draw = 32
    rnd = str(round_code).strip().upper() if pd.notna(round_code) else 'R32'

    # --- Static-offset levels (Tour Finals, Davis Cup) ---
    if level in _STATIC:
        return _STATIC[level].get(rnd, 0)

    # --- Qualifier rounds: offset from tourney_date directly ---
    # Match Q1, Q2, Q3, Q4 but NOT QF (quarterfinal)
    if rnd in ('Q1', 'Q2', 'Q3', 'Q4'):
        if num_qual_rounds is not None:
            # Dynamic: each qual round is 1 day, last qual is day -1
            # e.g. 2 rounds: Q1=-2, Q2=-1.  3 rounds: Q1=-3, Q2=-2, Q3=-1
            qual_num = int(rnd[1])  # Q1->1, Q2->2, Q3->3
            return -(num_qual_rounds - qual_num + 1)
        # Fallback: use level-based defaults
        qual_table = _QUAL_OFFSETS.get(level, _QUAL_OFFSETS['default'])
        return qual_table.get(rnd, 0)

    # --- Main draw rounds: anchor on "finals are Sunday" ---
    # Compute day-of-week of tournament start
    if tourney_date is not None and pd.notna(tourney_date):
        if isinstance(tourney_date, str):
            tourney_date = pd.to_datetime(tourney_date)
        dow = tourney_date.weekday()  # 0=Monday, 6=Sunday
    else:
        dow = 0  # Assume Monday if unknown

    days_to_sunday = (6 - dow) % 7
    if days_to_sunday == 0:
        days_to_sunday = 7  # Sunday start → final is NEXT Sunday

    # Two-week events: Grand Slams, Masters with draw >= 96 (IW/Miami)
    # M/64 is tricky: Madrid/Rome/Shanghai are ~12-day but Canada/Cincinnati are 7-day.
    # Heuristic: if the tournament start is NOT a Monday, it's likely a 2-week M/64.
    # (Canada/Cincinnati start Monday; Madrid/Rome/Shanghai start Tue-Wed)
    if level == 'G' or (level == 'M' and draw >= 96):
        final_offset = days_to_sunday + 7
    elif level == 'M' and draw == 64 and dow != 0:
        # Non-Monday start M/64 = Madrid/Rome/Shanghai style (~12-day event)
        final_offset = days_to_sunday + 7
    else:
        final_offset = days_to_sunday

    # Select round-relative table
    if level == 'G':
        table = _ROUND_RELATIVE['G']
    elif level == 'M' and draw >= 96:
        table = _ROUND_RELATIVE['M_large']
    elif level == 'M' and draw == 64 and dow != 0:
        table = _ROUND_RELATIVE['M_medium_long']
    elif level == 'M' and draw == 64:
        table = _ROUND_RELATIVE['M_medium_short']
    elif level == 'M':
        table = _ROUND_RELATIVE['M_small']
    else:
        # ATP 250/500 (A), Challenger (C), ITF (S), and anything else
        table = _ROUND_RELATIVE['standard']

    round_delta = table.get(rnd, 0)
    return final_offset + round_delta


def infer_draw_size(event_name, level) -> int:
    """Infer draw size from event name + level for offset lookup.

    Used by production code when draw_size isn't explicitly available
    (e.g. historical TA match rows).
    """
    evt = str(event_name).lower() if event_name else ''
    level = str(level).strip().upper() if level else ''
    if level == 'G':
        return 128
    if level == 'M':
        if 'indian wells' in evt or 'miami' in evt:
            return 96
        if any(x in evt for x in ['madrid', 'rome', 'roma', 'montreal',
                                   'toronto', 'canadian', 'rogers',
                                   'cincinnati', 'shanghai']):
            return 64
        return 48  # Monte Carlo, Paris, etc.
    return 32  # ATP 250/500, Challenger, ITF
