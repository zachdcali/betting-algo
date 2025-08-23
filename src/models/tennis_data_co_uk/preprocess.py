import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder
from datetime import datetime

def get_feature_columns(df):
    """
    Helper function to identify feature columns for ML models.
    Returns only the columns that should be used as features (no metadata, no target, no odds).
    """
    # Metadata columns to exclude
    metadata_cols = [
        'Date', 'Tournament', 'Location', 'Winner', 'Loser', 
        'Player1_Name', 'Player2_Name', 'WRank', 'LRank'
    ]
    
    # Odds columns to exclude (would be cheating)
    odds_cols = [col for col in df.columns if any(x in col for x in ['CBW', 'CBL', 'GBW', 'GBL', 'IWW', 'IWL', 'SBW', 'SBL', 'B365W', 'B365L', 'B&WW', 'B&WL', 'EXW', 'EXL', 'PSW', 'PSL', 'WPts', 'LPts', 'UBW', 'UBL', 'LBW', 'LBL', 'SJW', 'SJL', 'MaxW', 'MaxL', 'AvgW', 'AvgL'])]
    
    # Target column
    target_col = 'Player1_Wins'
    
    # Get all columns except metadata, odds, and target
    feature_cols = [col for col in df.columns if col not in metadata_cols + odds_cols + [target_col]]
    
    return feature_cols

def preprocess_tennis_data_for_ml():
    """
    Preprocess Tennis-Data.co.uk data for machine learning:
    1. Randomize player positions (Player1/Player2)
    2. Create binary target (player1_wins)
    3. One-hot encode categorical features
    4. Create additional features
    5. Filter to matches with both rankings and odds
    6. Save ML-ready dataset
    """
    
    print("=" * 60)
    print("TENNIS DATA ML PREPROCESSING")
    print("=" * 60)
    
    # Load the combined Tennis-Data.co.uk data
    print("\n1. Loading Tennis-Data.co.uk data...")
    df = pd.read_csv("/app/data/Tennis-Data.co.uk/tennis_data_combined_2000_2025.csv", low_memory=False)
    print(f"   Original data shape: {df.shape}")
    print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Convert date column
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day_of_Year'] = df['Date'].dt.dayofyear
    
    # Filter to matches with both rankings and odds (same as baseline)
    print("\n2. Filtering data...")
    initial_count = len(df)
    
    # Remove matches without proper ranking data
    df = df[
        (df['WRank'] != 'NR') & 
        (df['LRank'] != 'NR') & 
        pd.notna(df['WRank']) & 
        pd.notna(df['LRank'])
    ].copy()
    
    # Convert rankings to numeric
    df['WRank'] = pd.to_numeric(df['WRank'], errors='coerce')
    df['LRank'] = pd.to_numeric(df['LRank'], errors='coerce')
    
    # Remove matches with NaN rankings
    df = df.dropna(subset=['WRank', 'LRank'])
    
    # Keep only matches with odds data (for fair comparison with baseline)
    df = df[pd.notna(df['AvgW']) & pd.notna(df['AvgL'])].copy()
    
    print(f"   After filtering: {len(df)} matches ({initial_count - len(df)} removed)")
    print(f"   Filtering removed {((initial_count - len(df)) / initial_count * 100):.1f}% of matches")
    
    if len(df) == 0:
        print("   ERROR: No matches left after filtering!")
        return
    
    # 3. Randomize player positions
    print("\n3. Randomizing player positions...")
    
    # Create a random boolean mask for swapping
    np.random.seed(42)  # For reproducibility
    swap_mask = np.random.random(len(df)) < 0.5
    
    # Initialize Player1/Player2 columns
    df['Player1_Name'] = df['Winner'].copy()
    df['Player2_Name'] = df['Loser'].copy()
    df['Player1_Rank'] = df['WRank'].copy()
    df['Player2_Rank'] = df['LRank'].copy()
    df['Player1_Wins'] = 1  # Winner always wins initially
    
    # Swap players where swap_mask is True
    df.loc[swap_mask, 'Player1_Name'] = df.loc[swap_mask, 'Loser']
    df.loc[swap_mask, 'Player2_Name'] = df.loc[swap_mask, 'Winner']
    df.loc[swap_mask, 'Player1_Rank'] = df.loc[swap_mask, 'LRank']
    df.loc[swap_mask, 'Player2_Rank'] = df.loc[swap_mask, 'WRank']
    df.loc[swap_mask, 'Player1_Wins'] = 0  # If we swapped, Player1 lost
    
    print(f"   Swapped {swap_mask.sum()} matches ({swap_mask.sum()/len(df)*100:.1f}%)")
    print(f"   Player1 wins: {df['Player1_Wins'].sum()} ({df['Player1_Wins'].mean()*100:.1f}%)")
    print(f"   Player2 wins: {(1-df['Player1_Wins']).sum()} ({(1-df['Player1_Wins']).mean()*100:.1f}%)")
    
    # 4. Create derived features
    print("\n4. Creating derived features...")
    
    # Ranking difference (Player1 rank - Player2 rank)
    # Negative means Player1 is better ranked
    df['Rank_Diff'] = df['Player1_Rank'] - df['Player2_Rank']
    
    # Ranking advantage (1 if Player1 better ranked, 0 otherwise)
    df['Player1_Rank_Advantage'] = (df['Player1_Rank'] < df['Player2_Rank']).astype(int)
    
    # Average ranking
    df['Avg_Rank'] = (df['Player1_Rank'] + df['Player2_Rank']) / 2
    
    # Ranking ratio (higher rank / lower rank)
    df['Rank_Ratio'] = np.maximum(df['Player1_Rank'], df['Player2_Rank']) / np.minimum(df['Player1_Rank'], df['Player2_Rank'])
    
    # 5. One-hot encode categorical features
    print("\n5. One-hot encoding categorical features...")
    
    # Surface encoding
    surface_dummies = pd.get_dummies(df['Surface'], prefix='Surface')
    df = pd.concat([df, surface_dummies], axis=1)
    
    # Court type encoding  
    court_dummies = pd.get_dummies(df['Court'], prefix='Court')
    df = pd.concat([df, court_dummies], axis=1)
    
    # Tournament level encoding (if Series column exists)
    if 'Series' in df.columns:
        series_dummies = pd.get_dummies(df['Series'], prefix='Series')
        df = pd.concat([df, series_dummies], axis=1)
    
    # Round encoding
    round_dummies = pd.get_dummies(df['Round'], prefix='Round')
    df = pd.concat([df, round_dummies], axis=1)
    
    print(f"   Created {len(surface_dummies.columns)} surface features")
    print(f"   Created {len(court_dummies.columns)} court features")
    print(f"   Created {len(round_dummies.columns)} round features")
    
    # 6. Select final features for ML
    print("\n6. Selecting final ML features...")
    
    # Core features (NO ODDS - that would be cheating!)
    core_features = [
        'Player1_Rank', 'Player2_Rank', 'Rank_Diff', 'Player1_Rank_Advantage',
        'Avg_Rank', 'Rank_Ratio', 'Year', 'Month', 'Day_of_Year'
    ]
    
    # Get all one-hot encoded features
    surface_features = [col for col in df.columns if col.startswith('Surface_')]
    court_features = [col for col in df.columns if col.startswith('Court_')]
    round_features = [col for col in df.columns if col.startswith('Round_')]
    series_features = [col for col in df.columns if col.startswith('Series_')]
    
    # Combine all features
    feature_columns = core_features + surface_features + court_features + round_features + series_features
    
    # Keep only features that actually exist in the dataframe
    feature_columns = [col for col in feature_columns if col in df.columns]
    
    # Add metadata columns for tracking
    metadata_columns = [
        'Date', 'Tournament', 'Location', 'Winner', 'Loser', 
        'Player1_Name', 'Player2_Name', 'WRank', 'LRank',
        'AvgW', 'AvgL'  # Keep odds for evaluation but NOT as features
    ]
    
    # Target variable
    target_column = 'Player1_Wins'
    
    # Create final dataset
    final_columns = feature_columns + metadata_columns + [target_column]
    final_columns = [col for col in final_columns if col in df.columns]
    
    ml_df = df[final_columns].copy()
    
    print(f"   Selected {len(feature_columns)} features for ML:")
    print(f"   - Core features: {len(core_features)}")
    print(f"   - Surface features: {len(surface_features)}")
    print(f"   - Court features: {len(court_features)}")
    print(f"   - Round features: {len(round_features)}")
    print(f"   - Series features: {len(series_features)}")
    
    # 7. Train/Test split
    print("\n7. Creating train/test split...")
    
    train_df = ml_df[ml_df['Date'] < '2023-01-01'].copy()
    test_df = ml_df[ml_df['Date'] >= '2023-01-01'].copy()
    
    print(f"   Training set (2000-2022): {len(train_df)} matches")
    print(f"   Test set (2023-2025): {len(test_df)} matches")
    print(f"   Train/test split: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% / {len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
    
    # Check target distribution
    print(f"   Training set - Player1 wins: {train_df['Player1_Wins'].mean()*100:.1f}%")
    print(f"   Test set - Player1 wins: {test_df['Player1_Wins'].mean()*100:.1f}%")
    
    # 8. Save the ML-ready dataset
    print("\n8. Saving ML-ready dataset...")
    
    output_path = "/app/data/tennis_data_ml_ready.csv"
    ml_df.to_csv(output_path, index=False)
    
    print(f"   Saved to: {output_path}")
    print(f"   Final dataset shape: {ml_df.shape}")
    
    # 9. Summary statistics
    print("\n" + "=" * 60)
    print("PREPROCESSING SUMMARY")
    print("=" * 60)
    print(f"Original matches: {initial_count:,}")
    print(f"Final ML dataset: {len(ml_df):,}")
    print(f"Features for training: {len(feature_columns)}")
    print(f"Training matches: {len(train_df):,}")
    print(f"Test matches: {len(test_df):,}")
    print(f"Target balance: {ml_df['Player1_Wins'].mean()*100:.1f}% Player1 wins")
    print(f"Data spans: {ml_df['Date'].min().date()} to {ml_df['Date'].max().date()}")
    
    # Show sample of features
    print(f"\nSample features:")
    for i, feature in enumerate(feature_columns[:10]):
        print(f"  {i+1}. {feature}")
    if len(feature_columns) > 10:
        print(f"  ... and {len(feature_columns)-10} more")
    
    print("\n✅ ML preprocessing complete!")
    print("Next steps:")
    print("1. Update model files to use tennis_data_ml_ready.csv")
    print("2. Use Player1_Wins as target variable")
    print("3. Compare results to baseline accuracies:")
    print("   - ATP Ranking: 63.9%")
    print("   - Bookmaker: 67.9%")
    
    return ml_df, feature_columns

if __name__ == "__main__":
    # Run preprocessing
    ml_df, feature_columns = preprocess_tennis_data_for_ml()