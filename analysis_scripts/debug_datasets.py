#!/usr/bin/env python3
"""
Debug what's actually in the datasets to understand the data structure
"""
import pandas as pd

print("DEBUGGING DATASETS")
print("="*50)

# 1. Check ML-ready dataset structure
print("\n1. ML-ready dataset structure:")
ml_ready = pd.read_csv('data/JeffSackmann/jeffsackmann_ml_ready_all_years.csv', low_memory=False)
print(f"   Shape: {ml_ready.shape}")
print(f"   Columns with 'name' or 'Name':")
name_cols = [col for col in ml_ready.columns if 'name' in col.lower()]
for col in name_cols:
    print(f"     {col}")

print(f"\n   Sample data:")
print(ml_ready[['tourney_date'] + name_cols + ['Player1_Wins']].head())

# 2. Check original Jeff Sackmann structure  
print(f"\n2. Original Jeff Sackmann structure:")
jeff_orig = pd.read_csv('data/JeffSackmann/jeffsackmann_master_combined.csv', low_memory=False)
print(f"   Shape: {jeff_orig.shape}")
print(f"   Name columns:")
jeff_name_cols = [col for col in jeff_orig.columns if 'name' in col.lower()]
for col in jeff_name_cols:
    print(f"     {col}")

print(f"\n   Sample data:")
print(jeff_orig[['tourney_date'] + jeff_name_cols].head())

# 3. Check matched dataset structure
print(f"\n3. Matched Jeff Sackmann structure:")
jeff_matched = pd.read_csv('data/JeffSackmann/jeffsackmann_exact_matched_final.csv', low_memory=False)
print(f"   Shape: {jeff_matched.shape}")
print(f"   Name columns:")
matched_name_cols = [col for col in jeff_matched.columns if 'name' in col.lower()]
for col in matched_name_cols:
    print(f"     {col}")

print(f"\n   Sample data:")
print(jeff_matched[['tourney_date'] + matched_name_cols].head())

# 4. Filter ML-ready to 2023-2024 and see target distribution
print(f"\n4. ML-ready 2023-2024 analysis:")
ml_ready['tourney_date'] = pd.to_datetime(ml_ready['tourney_date'])
ml_ready['year'] = ml_ready['tourney_date'].dt.year
ml_2023_2024 = ml_ready[ml_ready['year'].isin([2023, 2024])]

print(f"   ML-ready 2023-2024 shape: {ml_2023_2024.shape}")
print(f"   Target (Player1_Wins) distribution: {ml_2023_2024['Player1_Wins'].mean():.3f}")
print(f"   Target value counts:")
print(ml_2023_2024['Player1_Wins'].value_counts())

# 5. Check if there's a relationship issue
print(f"\n5. Checking Player1_Wins logic:")
if 'Player1_Name' in ml_ready.columns and 'Player2_Name' in ml_ready.columns:
    sample = ml_2023_2024.head(10)
    for _, row in sample.iterrows():
        print(f"   {row['Player1_Name']} vs {row['Player2_Name']} -> Player1_Wins: {row['Player1_Wins']}")