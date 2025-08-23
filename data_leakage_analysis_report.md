# Data Leakage Analysis Report: Jeff Sackmann Dataset

## Executive Summary

The Random Forest model's unexpectedly high accuracy (89%) is **confirmed to be caused by data leakage**. Despite the randomization process working correctly, there is a critical flaw in the advantage feature calculation logic that creates perfect predictive signals.

## Key Findings

### 1. Randomization Working Correctly
- Mean values are all ~0.5 as expected:
  - Player1_Wins: 0.5006
  - Player1_Rank_Advantage: 0.5005
  - Player1_Height_Advantage: 0.4997
  - Player1_Age_Advantage: 0.4994
  - Player1_Points_Advantage: 0.5005
- Variance = 0.25, Std = 0.5 for all features (perfect binary distribution)

### 2. Critical Data Leakage Issue Identified

**Player1_Height_Advantage shows a massive NEGATIVE correlation (-0.5452) with Player1_Wins**

This means:
- When Player1_Height_Advantage = 1.0: Win rate = 22.78%
- When Player1_Height_Advantage = 0.0: Win rate = 77.30%

This is the **opposite** of what should happen and indicates a fundamental error.

### 3. Root Cause Analysis

Located in `/Users/zachdodson/Documents/betting-algo/src/Models/professional_tennis/preprocess.py` line 654:

```python
df['Player1_Height_Advantage'] = (df['Player1_Height'] > df['Player2_Height']).astype(int)
```

**The Problem**: This calculation happens BEFORE randomization, but the randomization logic is flawed.

### 4. The Randomization Logic Flaw

The randomization process (lines 835-841) attempts to fix advantage features:

```python
# CRITICAL: For swapped matches, flip the advantage features (0->1, 1->0)
advantage_features = ['Player1_Rank_Advantage', 'Player1_Height_Advantage', 
                     'Player1_Age_Advantage', 'Player1_Points_Advantage']

for feat in advantage_features:
    if feat in ml_df.columns:
        ml_df.loc[swap_mask, feat] = 1 - ml_df.loc[swap_mask, feat]
```

**However, this approach is fundamentally flawed because:**

1. Advantage features are calculated based on winner/loser data (Player1=winner initially)
2. When we randomize and swap players, we flip the advantage features
3. But the advantage features were originally correlated with the win outcome
4. This creates an inverted signal: advantage features now predict losses instead of wins

### 5. Why Other Correlations Are Smaller

- Player1_Rank_Advantage vs Player1_Wins: 0.0368 (small positive)
- Player1_Age_Advantage vs Player1_Wins: -0.0987 (moderate negative)
- Player1_Points_Advantage vs Player1_Wins: -0.0530 (small negative)

Height has the strongest correlation because height differences are more consistent/stable than rank or age differences.

## The Correct Fix

The advantage features should be calculated **AFTER** randomization, not before:

```python
# WRONG (current approach):
# 1. Calculate advantages based on winner/loser
# 2. Randomize player positions 
# 3. Try to fix advantages by flipping

# CORRECT approach:
# 1. Randomize player positions first
# 2. Calculate advantages based on randomized Player1/Player2
```

## Impact on Model Performance

The current "leak-free" dataset is actually worse than the original because:
- Advantage features now have **inverted signals**
- Player1_Height_Advantage = 1 predicts Player1 will **lose** (22.78% win rate)
- This explains why Random Forest still achieves 89% accuracy - it's just learned the inverted patterns

## Recommendation

**Immediate action required**: Recreate the dataset with advantages calculated AFTER randomization, not before. The current "leak-free" dataset is actually more leaky than before due to the inverted signals.