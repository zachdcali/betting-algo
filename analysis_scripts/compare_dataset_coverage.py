#!/usr/bin/env python3
"""
Compare dataset coverage between Jeff Sackmann and Tennis-Data.co.uk for 2023+ period
"""
import pandas as pd
import numpy as np
from datetime import datetime

def normalize_name(name):
    """Normalize player names for matching"""
    if pd.isna(name):
        return ""
    # Remove common suffixes and normalize
    name = str(name).strip()
    name = name.replace('.', '').replace('-', ' ')
    # Handle common name variations
    name = name.replace('Jr', '').replace('Sr', '')
    name = name.strip()
    return name.lower()

def normalize_tournament_name(name):
    """Normalize tournament names for matching"""
    if pd.isna(name):
        return ""
    name = str(name).strip().lower()
    # Common tournament name variations
    name = name.replace('championships', 'championship')
    name = name.replace('masters cup', 'atp finals')
    name = name.replace('masters 1000', 'masters')
    return name

print("==" * 40)
print("DATASET COVERAGE COMPARISON - 2023+ PERIOD")
print("==" * 40)

# Load both datasets
print("\n1. Loading datasets...")
jeff_data = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
tennis_data = pd.read_csv('data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv', low_memory=False)

print(f"   Jeff Sackmann total matches: {len(jeff_data):,}")
print(f"   Tennis-Data total matches: {len(tennis_data):,}")

# Filter to 2023+ and ATP-level tournaments
print("\n2. Filtering to test period (2023+)...")

# Jeff Sackmann filtering - dates are in YYYYMMDD integer format
jeff_data['tourney_date'] = pd.to_datetime(jeff_data['tourney_date'], format='%Y%m%d')
jeff_data['year'] = jeff_data['tourney_date'].dt.year
jeff_test = jeff_data[jeff_data['year'] >= 2023].copy()

# Filter to ATP-level tournaments (G, M, A, F)
atp_levels = ['G', 'M', 'A', 'F']
jeff_test_atp = jeff_test[jeff_test['tourney_level'].isin(atp_levels)].copy()

print(f"   Jeff Sackmann 2023+: {len(jeff_test):,} matches")
print(f"   Jeff Sackmann 2023+ ATP-level: {len(jeff_test_atp):,} matches")

# Tennis-Data filtering
tennis_data['Date'] = pd.to_datetime(tennis_data['Date'])
tennis_data['year'] = tennis_data['Date'].dt.year
tennis_test = tennis_data[tennis_data['year'] >= 2023].copy()

# Filter to ATP-level tournaments
atp_series = ['Masters Cup', 'Masters 1000', 'Grand Slam', 'ATP250', 'ATP500']
tennis_test_atp = tennis_test[tennis_test['Series'].isin(atp_series)].copy()

print(f"   Tennis-Data 2023+: {len(tennis_test):,} matches")
print(f"   Tennis-Data 2023+ ATP-level: {len(tennis_test_atp):,} matches")

# Year breakdown
print("\n3. Year-by-year breakdown:")
print("   Jeff Sackmann ATP-level:")
jeff_year_counts = jeff_test_atp['year'].value_counts().sort_index()
for year, count in jeff_year_counts.items():
    print(f"     {year}: {count:,} matches")

print("\n   Tennis-Data ATP-level:")
tennis_year_counts = tennis_test_atp['year'].value_counts().sort_index()
for year, count in tennis_year_counts.items():
    print(f"     {year}: {count:,} matches")

# Tournament level breakdown
print("\n4. Tournament level breakdown:")
print("   Jeff Sackmann:")
jeff_levels = jeff_test_atp['tourney_level'].value_counts()
for level, count in jeff_levels.items():
    level_name = {'G': 'Grand Slams', 'M': 'Masters', 'A': 'ATP', 'F': 'Finals'}
    print(f"     {level_name.get(level, level)}: {count:,} matches")

print("\n   Tennis-Data:")
tennis_levels = tennis_test_atp['Series'].value_counts()
for level, count in tennis_levels.items():
    print(f"     {level}: {count:,} matches")

# Tournament name comparison
print("\n5. Unique tournaments in each dataset:")
jeff_tournaments = set(jeff_test_atp['tourney_name'].unique())
tennis_tournaments = set(tennis_test_atp['Tournament'].unique())

print(f"   Jeff Sackmann unique tournaments: {len(jeff_tournaments)}")
print(f"   Tennis-Data unique tournaments: {len(tennis_tournaments)}")

# Try to find overlapping tournaments by normalizing names
jeff_norm_tournaments = {normalize_tournament_name(t): t for t in jeff_tournaments}
tennis_norm_tournaments = {normalize_tournament_name(t): t for t in tennis_tournaments}

common_normalized = set(jeff_norm_tournaments.keys()) & set(tennis_norm_tournaments.keys())
print(f"   Normalized name overlap: {len(common_normalized)} tournaments")

print("\n   Sample tournaments only in Jeff Sackmann:")
jeff_only_norm = set(jeff_norm_tournaments.keys()) - set(tennis_norm_tournaments.keys())
for i, norm_name in enumerate(sorted(jeff_only_norm)[:10]):
    print(f"     {jeff_norm_tournaments[norm_name]}")

print("\n   Sample tournaments only in Tennis-Data:")
tennis_only_norm = set(tennis_norm_tournaments.keys()) - set(jeff_norm_tournaments.keys())
for i, norm_name in enumerate(sorted(tennis_only_norm)[:10]):
    print(f"     {tennis_norm_tournaments[norm_name]}")

# Date range analysis
print("\n6. Date range analysis:")
jeff_min_date = jeff_test_atp['tourney_date'].min()
jeff_max_date = jeff_test_atp['tourney_date'].max()
tennis_min_date = tennis_test_atp['Date'].min()
tennis_max_date = tennis_test_atp['Date'].max()

print(f"   Jeff Sackmann date range: {jeff_min_date.strftime('%Y-%m-%d')} to {jeff_max_date.strftime('%Y-%m-%d')}")
print(f"   Tennis-Data date range: {tennis_min_date.strftime('%Y-%m-%d')} to {tennis_max_date.strftime('%Y-%m-%d')}")

# Check 2025 data specifically
jeff_2025 = jeff_test_atp[jeff_test_atp['year'] == 2025]
tennis_2025 = tennis_test_atp[tennis_test_atp['year'] == 2025]

print(f"\n   2025 matches:")
print(f"     Jeff Sackmann: {len(jeff_2025):,} matches")
print(f"     Tennis-Data: {len(tennis_2025):,} matches")

if len(jeff_2025) > 0:
    print(f"   Jeff Sackmann 2025 date range: {jeff_2025['tourney_date'].min().strftime('%Y-%m-%d')} to {jeff_2025['tourney_date'].max().strftime('%Y-%m-%d')}")
if len(tennis_2025) > 0:
    print(f"   Tennis-Data 2025 date range: {tennis_2025['Date'].min().strftime('%Y-%m-%d')} to {tennis_2025['Date'].max().strftime('%Y-%m-%d')}")

print("\n7. Sample recent matches for comparison:")
print("   Jeff Sackmann recent ATP matches:")
recent_jeff = jeff_test_atp.sort_values('tourney_date', ascending=False).head(5)
for _, match in recent_jeff.iterrows():
    print(f"     {match['tourney_date'].strftime('%Y-%m-%d')}: {match['tourney_name']} - {match['winner_name']} def. {match['loser_name']}")

print("\n   Tennis-Data recent ATP matches:")
recent_tennis = tennis_test_atp.sort_values('Date', ascending=False).head(5)
for _, match in recent_tennis.iterrows():
    print(f"     {match['Date'].strftime('%Y-%m-%d')}: {match['Tournament']} - {match['Winner']} def. {match['Loser']}")

print("\n" + "==" * 40)
print("CONCLUSION")
print("==" * 40)
print(f"Jeff Sackmann has {len(jeff_test_atp):,} ATP-level matches for 2023+")
print(f"Tennis-Data has {len(tennis_test_atp):,} ATP-level matches for 2023+")
print(f"Difference: {len(jeff_test_atp) - len(tennis_test_atp):,} matches ({((len(jeff_test_atp) - len(tennis_test_atp))/len(jeff_test_atp)*100):.1f}% more in Jeff Sackmann)")

if len(jeff_2025) > len(tennis_2025):
    print(f"\nMain difference appears to be 2025 coverage:")
    print(f"  Jeff Sackmann: {len(jeff_2025):,} matches in 2025")
    print(f"  Tennis-Data: {len(tennis_2025):,} matches in 2025")
    print(f"  2025 difference: {len(jeff_2025) - len(tennis_2025):,} matches")