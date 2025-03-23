#!/usr/bin/env python3
# clean_and_run.py
# This script cleans data directories and runs the UTR scraper with the updated functions

import os
import shutil
from pathlib import Path
import argparse
from utr_scraper import UTRScraper, logger
import time
import random

def clean_data_folders(base_path=None):
    """Clean up all data folders to start fresh"""
    if base_path is None:
        base_dir = Path(__file__).parent.parent
        base_path = base_dir / "data"
    
    # Data directories
    players_dir = base_path / "players"
    matches_dir = base_path / "matches"
    ratings_dir = base_path / "ratings"
    
    # Remove screenshots
    for png_file in base_path.parent.glob("*.png"):
        try:
            os.remove(png_file)
            print(f"Removed screenshot: {png_file}")
        except Exception as e:
            print(f"Could not remove {png_file}: {e}")
    
    # Remove existing directories if they exist
    for dir_path in [players_dir, matches_dir, ratings_dir]:
        if dir_path.exists():
            shutil.rmtree(dir_path)
            print(f"Removed {dir_path}")
        
        # Create fresh directories
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"Created {dir_path}")
    
    return base_path

def run_full_collection(gender="men", limit=100, player_limit=None, years=None):
    """Run full data collection with the updated scraper"""
    print(f"Starting full data collection for {gender}'s players...")
    
    # Clean data folders
    clean_data_folders()
    
    # Setup credentials
    email = os.environ.get("UTR_EMAIL") or "zachdodson12@gmail.com"
    password = os.environ.get("UTR_PASSWORD") or "Thailand@123"
    
    # Set default years if none provided
    if years is None:
        years = ["2025", "2024", "2023", "2022"]
    
    # Initialize scraper
    with UTRScraper(email=email, password=password, headless=True) as scraper:
        if scraper.login():
            print(f"Successfully logged in, starting data collection for {gender}'s players...")
            
            # Collect top players data
            start_time = time.time()
            top_players = scraper.collect_top_players_data(
                gender=gender,
                limit=limit,
                player_limit=player_limit,
                include_matches=True,
                include_ratings=True,
                years=years,
                utr_threshold=14.0
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            # Print summary
            print(f"\nData collection completed!")
            print(f"Time taken: {duration/60:.2f} minutes")
            print(f"Players collected: {len(top_players)}")
            
            return True
    
    print("Data collection failed - could not log in.")
    return False

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run UTR data collection')
    parser.add_argument('--gender', choices=['men', 'women'], default='men',
                      help='Gender of players to collect (default: men)')
    parser.add_argument('--limit', type=int, default=100,
                      help='Number of top players to collect (default: 100)')
    parser.add_argument('--player-limit', type=int, default=None,
                      help='Limit to first N players for testing (default: None)')
    parser.add_argument('--years', nargs='+', default=["2025", "2024", "2023", "2022"],
                      help='Years to collect match data for (default: 2025 2024 2023 2022)')
    parser.add_argument('--clean-only', action='store_true',
                      help='Only clean data folders without running collection')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    if args.clean_only:
        print("Cleaning data folders only...")
        clean_data_folders()
        print("Done!")
    else:
        run_full_collection(
            gender=args.gender,
            limit=args.limit,
            player_limit=args.player_limit,
            years=args.years
        )