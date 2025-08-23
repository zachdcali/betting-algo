#!/usr/bin/env python3
"""
Combine all JeffSackmann data sources (ATP, Futures, Challengers) into a single master file.
This script assumes you've already run the individual combination scripts in each subfolder.
"""

import pandas as pd
import os
from pathlib import Path

def combine_all_jeffsackmann():
    """Combine all JeffSackmann data sources into a master file."""
    
    # Get current directory (JeffSackmann folder)
    current_dir = Path(__file__).parent
    
    # Define the combined files to look for
    combined_files = [
        "ATP 1968-2024/atp_matches_combined_1968_2024.csv",
        "Futures 1991-2024/atp_matches_futures_combined_1991_2024.csv",
        "Qualifiers:Challengers 1978-2024/challengers_qualifiers_combined_1978_2024.csv"
    ]
    
    print("Combining all JeffSackmann data sources...")
    
    # List to store dataframes
    dataframes = []
    total_matches = 0
    
    for file_path in combined_files:
        full_path = current_dir / file_path
        
        if full_path.exists():
            source_name = file_path.split('/')[0]
            print(f"Loading {source_name}...")
            
            try:
                df = pd.read_csv(full_path)
                
                # Add source column to track data origin
                df['data_source'] = source_name
                
                print(f"  - Loaded {len(df):,} matches from {source_name}")
                dataframes.append(df)
                total_matches += len(df)
                
            except Exception as e:
                print(f"  - Error loading {full_path}: {e}")
                continue
        else:
            print(f"File not found: {full_path}")
            print("  Run the individual combination scripts first:")
            print(f"  python '{current_dir}/{file_path.split('/')[0]}/combine_*_files.py'")
    
    if not dataframes:
        print("No combined files found. Run individual combination scripts first.")
        return
    
    # Combine all dataframes
    print(f"\nCombining {len(dataframes)} data sources...")
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    # Sort by tournament date for chronological order
    print("Sorting by tournament date...")
    combined_df['tourney_date'] = pd.to_datetime(combined_df['tourney_date'], format='%Y%m%d')
    combined_df = combined_df.sort_values(['tourney_date', 'match_num'])
    
    # Convert date back to original format
    combined_df['tourney_date'] = combined_df['tourney_date'].dt.strftime('%Y%m%d')
    
    # Save master combined file
    output_file = current_dir / "jeffsackmann_master_combined.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nMaster combination complete!")
    print(f"Total matches: {len(combined_df):,}")
    print(f"Date range: {combined_df['tourney_date'].min()} to {combined_df['tourney_date'].max()}")
    print(f"Output file: {output_file}")
    print(f"Columns: {len(combined_df.columns)}")
    
    # Show breakdown by data source
    print(f"\nBreakdown by source:")
    source_counts = combined_df['data_source'].value_counts()
    for source, count in source_counts.items():
        print(f"  {source}: {count:,} matches")

if __name__ == "__main__":
    combine_all_jeffsackmann()