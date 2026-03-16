#!/usr/bin/env python3
"""Export all 141 features + predictions + market odds to CSV for inspection."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
from features.ta_feature_calculator import TAFeatureCalculator
from models.inference import TennisPredictor, EXACT_141_FEATURES
from datetime import datetime

pred = TennisPredictor()
pred.load_model()
calc = TAFeatureCalculator()

all_matches = [
    ('Cristian Garin',       'Liam Draxl',                  'Miami',   'M', 'Hard', -130,  110),
    ('Jay Clarke',           'Shintaro Mochizuki',           'Miami',   'M', 'Hard',  220, -275),
    ('Patrick Kypson',       'Nikoloz Basilashvili',         'Miami',   'M', 'Hard', -260,  210),
    ('Alexander Blockx',     'Chun Hsin Tseng',              'Miami',   'M', 'Hard', -400,  300),
    ('Aleksandar Vukic',     'Billy Harris',                 'Miami',   'M', 'Hard', -140,  115),
    ('Karl Poling',          'Takeru Yuzuki',                'Morelos', 'C', 'Clay', -200,  150),
    ('Quinn Vandecasteele',  'Miguel Tobon',                 'Morelos', 'C', 'Clay', -200,  150),
    ('Robin Catry',          'Juan Sebastian Osorio',        'Morelos', 'C', 'Clay',-1200,  650),
    ('Tibo Colson',          'Jesse Flores',                 'Morelos', 'C', 'Clay', -460,  305),
    ('Jake Delaney',         'Guillaume Dalmasso',           'Morelos', 'C', 'Clay', -175,  135),
    ('Samuel Heredia',       'Alan Fernando Rubio Fierros',  'Morelos', 'C', 'Clay',  120, -155),
    ('Andrea Fiorentini',    'Matyas Cerny',                 'Zadar',   'C', 'Hard',  255, -360),
    ('Laurent Lokoli',       'Ergi Kirkin',                  'Zadar',   'C', 'Hard', -210,  160),
    ('Samuele Pieri',        'Gerard Campana Lee',           'Zadar',   'C', 'Hard', -130,    0),
    ('Christian Langmo',     'Oleksii Krutykh',              'Zadar',   'C', 'Hard',  190, -260),
    ('Mirza Basic',          'Sergi Perez Contri',           'Zadar',   'C', 'Hard', -115, -115),
    ('Stefan Palosi',        'John Sperle',                  'Zadar',   'C', 'Hard', -185,  140),
    ('Andrej Nedic',         'Emanuel Ivanisevic',           'Zadar',   'C', 'Hard',-5000, 1400),
    ('Zsombor Piros',        'Enrico Dalla Valle',           'Zadar',   'C', 'Hard', -650,  400),
    ('Jelle Sels',           'Kyrian Jacquet',               'Zadar',   'C', 'Hard',  425, -700),
    ('Lorenzo Giustino',     'Jonas Forejtek',               'Zadar',   'C', 'Hard',  105, -135),
]

def to_prob(odds):
    if odds == 0: return 0.5
    if odds > 0: return 100 / (odds + 100)
    return -odds / (-odds + 100)

match_date = datetime(2026, 3, 15)
global_cache = {}
rows = []

for p1, p2, tourney, level, surface, o1, o2 in all_matches:
    row = {
        'P1': p1, 'P2': p2, 'Tournament': tourney,
        'Surface': surface, 'Level': level,
        'P1_Odds_American': o1, 'P2_Odds_American': o2,
        'Market_P1_ImpliedProb': round(to_prob(o1), 4),
        'Market_P2_ImpliedProb': round(to_prob(o2), 4),
    }
    try:
        f = calc.build_143_features(
            player1_name=p1, player2_name=p2,
            match_date=match_date, surface=surface,
            tournament_level=level, draw_size=96,
            round_code='R32', force_refresh=True, persist=False,
            session_cache=global_cache
        )
        ordered = {k: f.get(k, 0.0) for k in EXACT_141_FEATURES}
        r = pred.predict_match_probability(ordered)
        row['Model_P1_WinProb'] = round(r['player1_win_prob'], 4)
        row['Model_P2_WinProb'] = round(r['player2_win_prob'], 4)
        row['Edge_P1'] = round(r['player1_win_prob'] - to_prob(o1), 4)
        row['Status'] = 'OK'
        # Add all 141 features
        for feat in EXACT_141_FEATURES:
            row[feat] = f.get(feat, None)
    except Exception as e:
        row['Model_P1_WinProb'] = None
        row['Model_P2_WinProb'] = None
        row['Edge_P1'] = None
        row['Status'] = f'SKIP: {str(e)[:120]}'

    rows.append(row)
    status = row['Status']
    prob = f"{row['Model_P1_WinProb']:.0%}" if row['Model_P1_WinProb'] else 'N/A'
    print(f"  {p1} vs {p2}: {prob} | {status}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'match_features_export.csv')
df = pd.DataFrame(rows)
# Put metadata columns first, then features
meta_cols = ['P1','P2','Tournament','Surface','Level',
             'P1_Odds_American','P2_Odds_American',
             'Market_P1_ImpliedProb','Market_P2_ImpliedProb',
             'Model_P1_WinProb','Model_P2_WinProb','Edge_P1','Status']
feat_cols = [c for c in df.columns if c not in meta_cols]
df = df[meta_cols + feat_cols]
df.to_csv(out_path, index=False)
print(f"\nSaved to: {out_path}")
print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
