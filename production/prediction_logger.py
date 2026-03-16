#!/usr/bin/env python3
"""Append a prediction to prediction_log.csv."""
import pandas as pd
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'prediction_log.csv')

COLUMNS = [
    'logged_at', 'match_date', 'tournament', 'surface', 'level', 'round',
    'p1', 'p2',
    'model_p1_prob', 'model_p2_prob',
    'market_p1_prob', 'market_p2_prob',
    'p1_odds_american', 'p2_odds_american',
    'edge_p1', 'model_version',
    'actual_winner', 'settled_at', 'model_correct', 'market_correct',
]

def log_prediction(
    p1: str, p2: str,
    tournament: str, surface: str, level: str, round_code: str,
    match_date,
    model_p1_prob: float, model_p2_prob: float,
    market_p1_prob: float, market_p2_prob: float,
    p1_odds_american: float = None, p2_odds_american: float = None,
    model_version: str = 'laplace_20260315',
):
    row = {
        'logged_at': datetime.now().isoformat(),
        'match_date': str(match_date)[:10],
        'tournament': tournament,
        'surface': surface,
        'level': level,
        'round': round_code,
        'p1': p1, 'p2': p2,
        'model_p1_prob': round(model_p1_prob, 4),
        'model_p2_prob': round(model_p2_prob, 4),
        'market_p1_prob': round(market_p1_prob, 4),
        'market_p2_prob': round(market_p2_prob, 4),
        'p1_odds_american': p1_odds_american,
        'p2_odds_american': p2_odds_american,
        'edge_p1': round(model_p1_prob - market_p1_prob, 4),
        'model_version': model_version,
        'actual_winner': None,
        'settled_at': None,
        'model_correct': None,
        'market_correct': None,
    }

    if os.path.exists(LOG_PATH):
        df = pd.read_csv(LOG_PATH)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row], columns=COLUMNS)

    df.to_csv(LOG_PATH, index=False)
