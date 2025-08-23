import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
print("Loading dataset...")
df = pd.read_csv('/Users/zachdodson/Documents/betting-algo/data/JeffSackmann/jeffsackmann_ml_ready_LEAK_FREE.csv')

print(f"Dataset shape: {df.shape}")
print("\nColumn names:")
print(df.columns.tolist())

# Check the key features mentioned
key_features = ['Player1_Wins', 'Player1_Rank_Advantage', 'Player1_Height_Advantage', 
                'Player1_Age_Advantage', 'Player1_Points_Advantage']

print("\n" + "="*50)
print("ANALYZING KEY FEATURES FOR DATA LEAKAGE")
print("="*50)

# 1. Check mean values of key features
print("\n1. MEAN VALUES OF KEY FEATURES (should be ~0.5):")
print("-" * 50)
for feature in key_features:
    if feature in df.columns:
        mean_val = df[feature].mean()
        print(f"{feature}: {mean_val:.4f}")
    else:
        print(f"{feature}: NOT FOUND in dataset")

# 2. Check Player1_Wins distribution specifically
if 'Player1_Wins' in df.columns:
    print(f"\nPlayer1_Wins value counts:")
    print(df['Player1_Wins'].value_counts().sort_index())
    print(f"Player1_Wins proportion: {df['Player1_Wins'].mean():.4f}")

# 3. Calculate correlations between advantage features and Player1_Wins
print("\n2. CORRELATIONS WITH Player1_Wins:")
print("-" * 50)
if 'Player1_Wins' in df.columns:
    for feature in key_features[1:]:  # Skip Player1_Wins itself
        if feature in df.columns:
            corr = df[feature].corr(df['Player1_Wins'])
            print(f"{feature} vs Player1_Wins: {corr:.4f}")
        else:
            print(f"{feature}: NOT FOUND in dataset")

# 4. Check if randomization worked - look at variance in advantage features
print("\n3. VARIANCE CHECK (higher variance indicates better randomization):")
print("-" * 50)
for feature in key_features[1:]:  # Skip Player1_Wins
    if feature in df.columns:
        variance = df[feature].var()
        std = df[feature].std()
        print(f"{feature} - Variance: {variance:.4f}, Std: {std:.4f}")

# 5. Look for any extreme values or patterns
print("\n4. DESCRIPTIVE STATISTICS:")
print("-" * 50)
for feature in key_features:
    if feature in df.columns:
        print(f"\n{feature}:")
        print(df[feature].describe())

# 6. Check for any other suspicious patterns
print("\n5. CHECKING FOR SUSPICIOUS PATTERNS:")
print("-" * 50)

# Check if Player1 always wins when they have advantages
if all(col in df.columns for col in ['Player1_Wins', 'Player1_Rank_Advantage']):
    # Create a combined advantage score
    advantage_features = [col for col in key_features[1:] if col in df.columns]
    if len(advantage_features) > 0:
        df_temp = df[advantage_features + ['Player1_Wins']].copy()
        df_temp['Total_Advantage'] = df_temp[advantage_features].sum(axis=1)
        
        print("Win rate by total advantage score:")
        advantage_groups = pd.cut(df_temp['Total_Advantage'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
        win_rates = df_temp.groupby(advantage_groups)['Player1_Wins'].mean()
        print(win_rates)

print("\n" + "="*50)
print("ANALYSIS COMPLETE")
print("="*50)