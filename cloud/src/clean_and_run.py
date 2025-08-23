#!/usr/bin/env python3
# clean_and_run.py
# Updated script with proper Playwright management and optimizations

import os
import shutil
from pathlib import Path
import argparse
import time
import random
import pandas as pd
import json  # Added import for json
from multiprocessing import Pool
from playwright.sync_api import sync_playwright
from utr_scraper import UTRScraper, logger

# Global Playwright instance
playwright = None

def initialize_playwright():
    """Initialize the global Playwright instance"""
    global playwright
    if playwright is None:
        playwright = sync_playwright().start()
        print("Initialized global Playwright instance")
    return playwright

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

def validate_collected_data(data_dir):
    """Validate the collected data and generate a report"""
    print("\n===== DATA VALIDATION REPORT =====\n")
    
    # Check directories exist
    players_dir = data_dir / "players"
    matches_dir = data_dir / "matches"
    ratings_dir = data_dir / "ratings"
    
    valid_dirs = all(d.exists() for d in [players_dir, matches_dir, ratings_dir])
    print(f"Data directories check: {'PASS' if valid_dirs else 'FAIL'}")
    
    # Count files
    player_files = list(players_dir.glob("*.json"))
    match_files = list(matches_dir.glob("*.csv"))
    rating_files = list(ratings_dir.glob("*.csv"))
    
    print(f"Player profiles: {len(player_files)}")
    print(f"Match history files: {len(match_files)}")
    print(f"Rating history files: {len(rating_files)}")
    
    # Check for empty files
    empty_player_files = [f for f in player_files if os.path.getsize(f) == 0]
    empty_match_files = [f for f in match_files if os.path.getsize(f) == 0]
    empty_rating_files = [f for f in rating_files if os.path.getsize(f) == 0]
    
    print(f"Empty player files: {len(empty_player_files)} {'WARN' if empty_player_files else 'PASS'}")
    print(f"Empty match files: {len(empty_match_files)} {'WARN' if empty_match_files else 'PASS'}")
    print(f"Empty rating files: {len(empty_rating_files)} {'WARN' if empty_rating_files else 'PASS'}")
    
    # Check for year-specific files
    year_specific_files = list(matches_dir.glob("*_20*.csv"))
    print(f"Year-specific match files: {len(year_specific_files)}")
    
    # Check enhanced match data
    enhanced_match_file = data_dir / "all_enhanced_matches.csv"
    valid_matches_file = data_dir / "valid_matches_for_model.csv"
    unrated_file = data_dir / "unrated_player_matches.csv"
    
    if enhanced_match_file.exists():
        try:
            enhanced_df = pd.read_csv(enhanced_match_file)
            print(f"Enhanced matches: {len(enhanced_df)}")
            
            # Check for required columns
            required_columns = ['player_id', 'player_name', 'opponent_id', 'opponent_name', 
                              'date', 'score', 'result', 'match_id', 'standardized_match_id']
            missing_columns = [col for col in required_columns if col not in enhanced_df.columns]
            print(f"Required columns check: {'PASS' if not missing_columns else f'FAIL - Missing: {missing_columns}'}")
            
            # Check for null values in critical columns
            if not missing_columns:
                critical_columns = ['player_id', 'opponent_id', 'date', 'score', 'result']
                null_counts = {col: enhanced_df[col].isna().sum() for col in critical_columns}
                print(f"Null values check: {'PASS' if sum(null_counts.values()) == 0 else 'WARN - ' + str(null_counts)}")
            
            # Check date distribution
            if 'date' in enhanced_df.columns:
                enhanced_df['date'] = pd.to_datetime(enhanced_df['date'], errors='coerce')
                year_counts = enhanced_df.groupby(enhanced_df['date'].dt.year).size().to_dict()
                print(f"Match year distribution: {year_counts}")
            
            # Check for duplicate matches
            if 'standardized_match_id' in enhanced_df.columns:
                duplicates = enhanced_df.duplicated(subset=['standardized_match_id'], keep=False)
                print(f"Duplicate matches check: {'PASS' if not duplicates.any() else f'WARN - {duplicates.sum()} duplicates'}")
                
                if duplicates.any():
                    duplicate_counts = enhanced_df[duplicates]['standardized_match_id'].value_counts()
                    excessive_duplicates = duplicate_counts[duplicate_counts > 2]
                    if not excessive_duplicates.empty:
                        print(f"WARN: {len(excessive_duplicates)} matches appear more than twice")
            
            # Check win-loss consistency
            if 'standardized_match_id' in enhanced_df.columns and 'result' in enhanced_df.columns:
                match_groups = enhanced_df[enhanced_df.duplicated(subset=['standardized_match_id'], keep=False)]
                match_groups = match_groups.groupby('standardized_match_id')
                
                inconsistent_matches = 0
                both_win_matches = 0
                both_lose_matches = 0
                for match_id, group in match_groups:
                    if len(group) == 2:
                        results = group['result'].tolist()
                        if results.count('W') == 2:
                            both_win_matches += 1
                            inconsistent_matches += 1
                        elif results.count('L') == 2:
                            both_lose_matches += 1
                            inconsistent_matches += 1
                        elif results.count('W') != 1 or results.count('L') != 1:
                            inconsistent_matches += 1
                
                print(f"Win-loss consistency check: {'PASS' if inconsistent_matches == 0 else f'FAIL - {inconsistent_matches} inconsistent'}")
                if inconsistent_matches > 0:
                    print(f"  Both win matches: {both_win_matches}")
                    print(f"  Both lose matches: {both_lose_matches}")
                    print(f"  Other inconsistencies: {inconsistent_matches - both_win_matches - both_lose_matches}")
            
            # Check valid matches
            if valid_matches_file.exists():
                valid_df = pd.read_csv(valid_matches_file)
                print(f"Valid matches with UTR: {len(valid_df)}")
                
                if 'player_utr_at_match' in valid_df.columns and 'opponent_utr_at_match' in valid_df.columns:
                    missing_utrs = (valid_df['player_utr_at_match'].isna() | valid_df['opponent_utr_at_match'].isna()).sum()
                    print(f"UTR at match time check: {'PASS' if missing_utrs == 0 else f'WARN - {missing_utrs} missing'}")
                
                if 'utr_diff' in valid_df.columns:
                    missing_diffs = valid_df['utr_diff'].isna().sum()
                    print(f"UTR difference check: {'PASS' if missing_diffs == 0 else f'WARN - {missing_diffs} missing'}")
                
                if 'standardized_match_id' in valid_df.columns:
                    valid_duplicates = valid_df.duplicated(subset=['standardized_match_id'], keep=False).sum()
                    print(f"Valid matches duplicates check: {'PASS' if valid_duplicates == 0 else f'FAIL - {valid_duplicates} duplicates'}")
                
                if 'tournament' in valid_df.columns:
                    tournament_counts = valid_df['tournament'].value_counts()
                    print(f"Top 5 tournaments: {tournament_counts.head(5).to_dict()}")
            
            if unrated_file.exists():
                unrated_df = pd.read_csv(unrated_file)
                print(f"Unrated player matches: {len(unrated_df)}")
                
        except Exception as e:
            print(f"Error analyzing match data: {e}")
            import traceback
            print(traceback.format_exc())
    else:
        print("Enhanced matches file not found - cross-reference may have failed")
    
    print("\nRating history validation:")
    try:
        sample_size = min(10, len(rating_files))
        sampled_rating_files = random.sample(rating_files, sample_size) if rating_files else []
        
        valid_structure = 0
        rating_problems = 0
        
        for rating_file in sampled_rating_files:
            try:
                rating_df = pd.read_csv(rating_file)
                
                if 'date' in rating_df.columns and 'utr' in rating_df.columns:
                    rating_df['date'] = pd.to_datetime(rating_df['date'], errors='coerce')
                    invalid_dates = rating_df['date'].isna().sum()
                    
                    if rating_df['utr'].dtype != object:
                        invalid_utrs = (rating_df['utr'] < 0).sum() + (rating_df['utr'] > 16.5).sum()
                    else:
                        invalid_utrs = 0
                        for utr in rating_df['utr']:
                            if utr != "UR" and not isinstance(utr, (int, float)):
                                invalid_utrs += 1
                    
                    if invalid_dates == 0 and invalid_utrs == 0:
                        valid_structure += 1
                    else:
                        rating_problems += 1
                else:
                    rating_problems += 1
            except Exception as e:
                rating_problems += 1
        
        print(f"Rating files structure check: {'PASS' if valid_structure == sample_size else f'WARN - {rating_problems}/{sample_size} have issues'}")
    except Exception as e:
        print(f"Error checking rating files: {e}")
    
    print("\n===== END OF REPORT =====\n")

def process_player(args):
    """Process a single player in a separate process"""
    player, scraper_args, years, include_matches, include_ratings, utr_threshold, dynamic_years = args
    player_id = player['id']
    name = player['name']
    
    # Each process needs its own scraper instance
    with UTRScraper(**scraper_args) as scraper:
        if not scraper.login():
            logger.error(f"Failed to login for player {name}")
            return None, None, None
        
        logger.info(f"Processing player: {name} (Rank: {player['rank']}, ID: {player_id})")
        
        # Check if profile already exists
        profile_file = scraper.players_dir / f"player_{player_id}.json"
        if profile_file.exists():
            logger.info(f"Profile already exists for {name}, skipping profile fetch")
            with open(profile_file, 'r') as f:
                profile = json.load(f)
        else:
            profile = scraper.get_player_profile(player_id)
        
        # Get years
        player_years = years
        if dynamic_years:
            player_years = scraper.get_available_years(player_id)
            if not player_years:
                player_years = years
        
        # Get ratings
        ratings = None
        if include_ratings:
            rating_file = scraper.ratings_dir / f"player_{player_id}_ratings.csv"
            if rating_file.exists():
                logger.info(f"Rating history already exists for {name}, skipping ratings fetch")
                ratings = pd.read_csv(rating_file)
            else:
                ratings = scraper.get_player_rating_history(player_id)
        
        # Get matches
        matches = []
        if include_matches:
            for year in player_years:
                match_file = scraper.matches_dir / f"player_{player_id}_matches_{year}.csv"
                if match_file.exists():
                    logger.info(f"Matches for {name} in {year} already exist, skipping")
                    match_df = pd.read_csv(match_file)
                    matches.append(match_df)
                else:
                    match_df = scraper.get_player_match_history(player_id, year=year, limit=100)
                    if not match_df.empty:
                        matches.append(match_df)
        
        matches_df = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
        
        # Extract opponent IDs
        opponent_ids = set()
        high_utr_opponents = set()
        if not matches_df.empty and 'opponent_id' in matches_df.columns:
            opponent_ids = set(matches_df['opponent_id'].dropna().astype(str).unique())
            high_utr_matches = matches_df[pd.to_numeric(matches_df['opponent_utr_displayed'], errors='coerce') >= utr_threshold]
            if not high_utr_matches.empty:
                high_utr_ids = high_utr_matches['opponent_id'].dropna().astype(str).unique()
                high_utr_opponents = set(high_utr_ids)
        
        return opponent_ids, high_utr_opponents, player_id

def process_opponent(args):
    """Process a single opponent in a separate process"""
    opponent_id, scraper_args, years, include_matches, utr_threshold, dynamic_years = args
    
    with UTRScraper(**scraper_args) as scraper:
        if not scraper.login():
            logger.error(f"Failed to login for opponent {opponent_id}")
            return None, None
        
        # Get profile
        profile_file = scraper.players_dir / f"player_{opponent_id}.json"
        if profile_file.exists():
            with open(profile_file, 'r') as f:
                profile = json.load(f)
        else:
            profile = scraper.get_player_profile(opponent_id)
        
        # Get ratings
        rating_file = scraper.ratings_dir / f"player_{opponent_id}_ratings.csv"
        if rating_file.exists():
            ratings = pd.read_csv(rating_file)
        else:
            ratings = scraper.get_player_rating_history(opponent_id)
        
        # Get matches if high UTR opponent
        matches = []
        if include_matches:
            player_years = years
            if dynamic_years:
                player_years = scraper.get_available_years(opponent_id)
                if not player_years:
                    player_years = years
            
            for year in player_years:
                match_file = scraper.matches_dir / f"player_{opponent_id}_matches_{year}.csv"
                if match_file.exists():
                    match_df = pd.read_csv(match_file)
                    matches.append(match_df)
                else:
                    match_df = scraper.get_player_match_history(opponent_id, year=year, limit=50)
                    if not match_df.empty:
                        matches.append(match_df)
        
        matches_df = pd.concat(matches, ignore_index=True) if matches else pd.DataFrame()
        
        # Extract second-degree opponent IDs
        second_degree_opponents = set()
        if not matches_df.empty and 'opponent_id' in matches_df.columns:
            second_ids = matches_df['opponent_id'].dropna().astype(str).unique()
            second_degree_opponents = set(second_ids)
        
        return second_degree_opponents, opponent_id

def process_second_degree_opponent(args):
    """Process a second-degree opponent (ratings only)"""
    opponent_id, scraper_args = args
    
    with UTRScraper(**scraper_args) as scraper:
        if not scraper.login():
            logger.error(f"Failed to login for second-degree opponent {opponent_id}")
            return
        
        # Get profile
        profile_file = scraper.players_dir / f"player_{opponent_id}.json"
        if profile_file.exists():
            return
        
        profile = scraper.get_player_profile(opponent_id)
        
        # Get ratings
        rating_file = scraper.ratings_dir / f"player_{opponent_id}_ratings.csv"
        if not rating_file.exists():
            scraper.get_player_rating_history(opponent_id)

def run_full_collection(gender="men", limit=100, player_limit=None, years=None, dynamic_years=True, resume=False):
    """Run full data collection with the updated scraper"""
    print(f"Starting full data collection for {gender}'s players...")
    
    # Create data directory reference
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    # Clean data folders only if not resuming
    if not resume:
        clean_data_folders(data_dir)
    
    # Setup credentials
    email = os.environ.get("UTR_EMAIL") or "zachdodson12@gmail.com"
    password = os.environ.get("UTR_PASSWORD") or "Thailand@123"
    
    # Set default years if none provided
    if years is None:
        years = ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018"]
    
    # Scraper args for each process
    scraper_args = {
        "email": email,
        "password": password,
        "headless": True,
        "data_dir": data_dir
    }
    
    # Step 1: Collect top players data in parallel
    start_time = time.time()
    print("Collecting top players data...")
    
    with UTRScraper(**scraper_args) as scraper:
        top_players = scraper.get_top_players(gender=gender, limit=limit)
        if top_players.empty:
            print("No top players found.")
            return False
    
    if player_limit:
        top_players = top_players.head(player_limit)
    
    # Process top players in parallel
    player_tasks = [(player, scraper_args, years, True, True, 14.0, dynamic_years) for _, player in top_players.iterrows()]
    all_opponent_ids = set()
    high_utr_opponents = set()
    processed_players = set()
    
    with Pool(processes=4) as pool:
        for batch_idx, batch in enumerate([player_tasks[i:i+4] for i in range(0, len(player_tasks), 4)]):
            print(f"Processing top player batch {batch_idx+1}/{len(player_tasks)//4 + 1}")
            results = pool.map(process_player, batch)
            for opponent_ids, high_utr_ids, player_id in results:
                if opponent_ids is not None:
                    all_opponent_ids.update(opponent_ids)
                    high_utr_opponents.update(high_utr_ids)
                    processed_players.add(player_id)
            time.sleep(5)  # Delay between batches
    
    # Step 2: Process direct opponents
    print(f"Found {len(all_opponent_ids)} direct opponents, {len(high_utr_opponents)} with high UTR")
    
    opponent_tasks = []
    for opponent_id in all_opponent_ids:
        # Skip if already processed (e.g., as a top player)
        if opponent_id in processed_players:
            continue
        # Check if already processed
        rating_file = data_dir / "ratings" / f"player_{opponent_id}_ratings.csv"
        if rating_file.exists():
            matches_exist = all((data_dir / "matches" / f"player_{opponent_id}_matches_{year}.csv").exists() for year in years)
            if opponent_id not in high_utr_opponents or matches_exist:
                continue
        opponent_tasks.append((opponent_id, scraper_args, years, opponent_id in high_utr_opponents, 14.0, dynamic_years))
    
    second_degree_opponents = set()
    processed_opponents = set()
    
    with Pool(processes=4) as pool:
        for batch_idx, batch in enumerate([opponent_tasks[i:i+4] for i in range(0, len(opponent_tasks), 4)]):
            print(f"Processing opponent batch {batch_idx+1}/{len(opponent_tasks)//4 + 1}")
            results = pool.map(process_opponent, batch)
            for second_ids, opponent_id in results:
                if second_ids is not None:
                    second_degree_opponents.update(second_ids)
                    processed_opponents.add(opponent_id)
            time.sleep(5)
    
    # Step 3: Process second-degree opponents (ratings only)
    print(f"Found {len(second_degree_opponents)} second-degree opponents")
    
    second_tasks = []
    for opponent_id in second_degree_opponents:
        if opponent_id in processed_players or opponent_id in processed_opponents:
            continue
        rating_file = data_dir / "ratings" / f"player_{opponent_id}_ratings.csv"
        if rating_file.exists():
            continue
        second_tasks.append((opponent_id, scraper_args))
    
    with Pool(processes=4) as pool:
        for batch_idx, batch in enumerate([second_tasks[i:i+4] for i in range(0, len(second_tasks), 4)]):
            print(f"Processing second-degree opponent batch {batch_idx+1}/{len(second_tasks)//4 + 1}")
            pool.map(process_second_degree_opponent, batch)
            time.sleep(5)
    
    # Step 4: Cross-reference
    print("\nRunning final cross-reference...")
    with UTRScraper(**scraper_args) as scraper:
        if scraper.login():
            valid_matches = scraper.cross_reference_matches_with_ratings()
            print(f"Cross-reference complete with {len(valid_matches)} valid matches")
    
    end_time = time.time()
    duration = (end_time - start_time) / 60
    print(f"\nData collection completed in {duration:.2f} minutes")
    
    # Run validation
    print("\nRunning data validation...")
    validate_collected_data(data_dir)
    
    return True

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run UTR data collection')
    parser.add_argument('--gender', choices=['men', 'women'], default='men',
                      help='Gender of players to collect (default: men)')
    parser.add_argument('--limit', type=int, default=100,
                      help='Number of top players to collect (default: 100)')
    parser.add_argument('--player-limit', type=int, default=None,
                      help='Limit to first N players for testing (default: None)')
    parser.add_argument('--years', nargs='+', default=["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018"],
                      help='Years to collect match data for')
    parser.add_argument('--clean-only', action='store_true',
                      help='Only clean data folders without running collection')
    parser.add_argument('--resume', action='store_true',
                      help='Resume processing without cleaning data folders')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    
    # Initialize Playwright
    initialize_playwright()
    
    try:
        if args.clean_only:
            print("Cleaning data folders only...")
            clean_data_folders()
            print("Done!")
        else:
            success = run_full_collection(
                gender=args.gender,
                limit=args.limit,
                player_limit=args.player_limit,
                years=args.years,
                dynamic_years=True,
                resume=args.resume
            )
    finally:
        # Clean up Playwright
        if playwright:
            playwright.stop()
            print("Closed Playwright instance")