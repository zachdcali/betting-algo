#!/usr/bin/env python3
"""
Combine all Tennis-Data.co.uk Excel files from 2000-2025 into a single master CSV file.
Handles different Excel formats (.xls vs .xlsx) and missing columns gracefully.
"""

import pandas as pd
import glob
import os
from pathlib import Path

def combine_tennis_data_files():
    """Combine all Tennis-Data.co.uk Excel files into a single master CSV file."""
    
    # Get current directory (Tennis-Data.co.uk folder)
    current_dir = Path(__file__).parent
    
    # Find all Excel files (both .xls and .xlsx)
    excel_files = []
    excel_files.extend(glob.glob(str(current_dir / "*.xls")))
    excel_files.extend(glob.glob(str(current_dir / "*.xlsx")))
    excel_files.sort()  # Sort by filename (year order)
    
    print(f"Found {len(excel_files)} Tennis-Data.co.uk Excel files to combine")
    
    # List to store dataframes
    dataframes = []
    
    for excel_file in excel_files:
        year = os.path.basename(excel_file).split('.')[0]
        print(f"Processing {year}...")
        
        try:
            # Handle both .xls and .xlsx formats
            if excel_file.endswith('.xls'):
                df = pd.read_excel(excel_file, engine='xlrd')
            else:
                df = pd.read_excel(excel_file, engine='openpyxl')
            
            # Add year column for tracking
            df['file_year'] = year
            
            print(f"  - Loaded {len(df)} matches from {year}")
            print(f"  - Columns: {len(df.columns)}")
            
            # Show first few column names for debugging
            if len(df.columns) > 0:
                sample_cols = list(df.columns[:5])
                print(f"  - Sample columns: {sample_cols}")
            
            dataframes.append(df)
            
        except Exception as e:
            print(f"  - Error loading {excel_file}: {e}")
            continue
    
    if not dataframes:
        print("No data loaded. Exiting.")
        return
    
    # Combine all dataframes (using outer join to include all columns)
    print("\nCombining all dataframes...")
    combined_df = pd.concat(dataframes, ignore_index=True, sort=False)
    
    # Try to sort by date if a date column exists
    print("Attempting to sort by date...")
    date_columns = [col for col in combined_df.columns if 'date' in col.lower()]
    if date_columns:
        print(f"Found date columns: {date_columns}")
        try:
            # Try to use the first date column for sorting
            date_col = date_columns[0]
            combined_df[date_col] = pd.to_datetime(combined_df[date_col], errors='coerce')
            combined_df = combined_df.sort_values(date_col)
            print(f"Sorted by {date_col}")
        except Exception as e:
            print(f"Could not sort by date: {e}")
            print("Data will be sorted by file year instead")
            combined_df = combined_df.sort_values('file_year')
    else:
        print("No date column found, sorting by file year")
        combined_df = combined_df.sort_values('file_year')
    
    # Save combined file as CSV
    output_file = current_dir / "tennis_data_combined_2000_2025.csv"
    combined_df.to_csv(output_file, index=False)
    
    print(f"\nCombination complete!")
    print(f"Total matches: {len(combined_df):,}")
    print(f"Years covered: {combined_df['file_year'].min()} to {combined_df['file_year'].max()}")
    print(f"Output file: {output_file}")
    print(f"Total columns: {len(combined_df.columns)}")
    
    # Show column summary
    print(f"\nColumn summary:")
    print(f"All columns: {list(combined_df.columns)}")
    
    # Show breakdown by year
    print(f"\nBreakdown by year:")
    year_counts = combined_df['file_year'].value_counts().sort_index()
    for year, count in year_counts.items():
        print(f"  {year}: {count:,} matches")

if __name__ == "__main__":
    combine_tennis_data_files()