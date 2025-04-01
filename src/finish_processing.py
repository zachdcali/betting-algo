#!/usr/bin/env python3
# fixed_processing.py
# Fixed script that properly handles Playwright instances

import os
import sys
from pathlib import Path
import time
import pandas as pd
import numpy as np
import re
import json
import random
from datetime import datetime
import traceback
from playwright.sync_api import sync_playwright

# Import UTRScraper but modify how we create the browser
from utr_scraper import UTRScraper, logger

# Create a global Playwright instance
playwright = None

def initialize_playwright():
    """Initialize the global Playwright instance"""
    global playwright
    if playwright is None:
        playwright = sync_playwright().start()
        print("Initialized global Playwright instance")
    return playwright

# Modify the UTRScraper to use our global playwright instance
class FixedUTRScraper(UTRScraper):
    def start_browser(self):
        """Initialize the browser using the shared playwright instance"""
        global playwright
        if playwright is None:
            initialize_playwright()
            
        self.browser = playwright.chromium.launch(headless=self.headless)
        self.context = self.browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
            viewport={"width": 1280, "height": 800}
        )
        self.page = self.context.new_page()
        
        # Add random delays to appear more human-like
        self.page.set_default_timeout(30000)  # 30 seconds default timeout
        return self

def collect_missing_opponent_data(data_dir, email, password, years=None, dynamic_years=True):
    """
    Find and collect data for direct opponents of top players that don't have match histories
    
    Returns:
    bool: Whether the collection was successful
    """
    # Identify top player IDs by looking for ranking files
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    
    if ranking_files:
        # Use the most recent ranking file
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        
        try:
            rankings_df = pd.read_csv(latest_ranking)
            if 'id' in rankings_df.columns:
                top_player_ids = set(rankings_df['id'].dropna().astype(str))
                print(f"Found {len(top_player_ids)} top players from ranking file: {latest_ranking}")
        except Exception as e:
            print(f"Error reading ranking file: {e}")
    
    # Find direct opponent IDs from match files of top players
    matches_dir = data_dir / "matches"
    direct_opponent_ids = set()
    
    for player_id in top_player_ids:
        # Look for match files for this top player
        for match_file in matches_dir.glob(f"player_{player_id}_matches*.csv"):
            try:
                df = pd.read_csv(match_file)
                if 'opponent_id' in df.columns:
                    file_opponent_ids = df['opponent_id'].dropna().astype(str).unique()
                    direct_opponent_ids.update([oid for oid in file_opponent_ids if oid and oid != 'nan'])
            except Exception as e:
                print(f"Error reading match file {match_file}: {e}")
    
    print(f"Found {len(direct_opponent_ids)} direct opponents of top players")
    
    # Check which direct opponents don't have match histories
    missing_matches = []
    for opponent_id in direct_opponent_ids:
        match_file = matches_dir / f"player_{opponent_id}_matches.csv"
        if not match_file.exists():
            missing_matches.append(opponent_id)
    
    print(f"Found {len(missing_matches)} direct opponents missing match histories")
    
    if not missing_matches:
        return True
    
    # Process these direct opponents in batches
    batch_size = 20
    batches = [missing_matches[i:i+batch_size] for i in range(0, len(missing_matches), batch_size)]
    
    # Initialize the global playwright instance
    initialize_playwright()
    
    for batch_idx, batch in enumerate(batches):
        print(f"Processing opponent batch {batch_idx+1}/{len(batches)}")
        
        # Create a fresh scraper for the batch
        scraper = FixedUTRScraper(email=email, password=password, headless=True)
        scraper.start_browser()
        
        try:
            if not scraper.login():
                print(f"Failed to log in for batch {batch_idx+1}. Skipping batch.")
                scraper.close_browser()
                continue
            
            # Process each opponent in the batch
            for idx, opponent_id in enumerate(batch):
                try:
                    print(f"Processing opponent {idx+1}/{len(batch)}: {opponent_id}")
                    
                    # Get player profile
                    profile = scraper.get_player_profile(opponent_id)
                    if profile:
                        player_name = profile['name']
                        print(f"Got profile for {player_name}")
                        
                        # Get rating history
                        ratings = scraper.get_player_rating_history(opponent_id)
                        if isinstance(ratings, pd.DataFrame) and not ratings.empty:
                            print(f"Got {len(ratings)} rating entries for {player_name}")
                        
                        # Get match history for each year
                        match_count = 0
                        player_years = years
                        
                        # Get available years if dynamic_years is enabled
                        if dynamic_years:
                            try:
                                available_years = scraper.get_available_years(opponent_id)
                                if available_years:
                                    player_years = available_years
                                    print(f"Using years for {player_name}: {', '.join(player_years)}")
                            except Exception as e:
                                print(f"Error getting available years: {e}")
                        
                        for year in player_years:
                            try:
                                print(f"Getting matches for {player_name} - Year {year}")
                                matches = scraper.get_player_match_history(opponent_id, year=year)
                                
                                if isinstance(matches, pd.DataFrame):
                                    year_count = len(matches)
                                    match_count += year_count
                                    print(f"Got {year_count} matches for {player_name} in {year}")
                                else:
                                    print(f"No matches found for {player_name} in {year}")
                                    
                                # Add delay between years
                                time.sleep(random.uniform(1.0, 2.0))
                            except Exception as e:
                                print(f"Error getting matches for year {year}: {e}")
                        
                        print(f"Retrieved total of {match_count} matches for {player_name}")
                    else:
                        print(f"Failed to get profile for opponent {opponent_id}")
                    
                    time.sleep(random.uniform(1.0, 2.0))
                    
                except Exception as e:
                    print(f"Error processing opponent {opponent_id}: {e}")
                    traceback.print_exc()
            
        except Exception as e:
            print(f"Error processing batch {batch_idx+1}: {e}")
            traceback.print_exc()
        finally:
            # Close just the browser, not the playwright instance
            scraper.browser.close()
            
        # Wait between batches
        if batch_idx < len(batches) - 1:
            print(f"Waiting 10 seconds before next batch...")
            time.sleep(10)
    
    # Don't close the global playwright, we'll reuse it for cross-reference
    return True

def run_cross_reference(data_dir, email, password):
    """Run the cross-reference process to generate enhanced files"""
    print("\nRunning cross-reference to generate enhanced files...")
    
    # Maximum retry attempts
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            print(f"Attempt {attempt+1}/{max_retries}: Running cross-reference process...")
            
            # Initialize the global playwright instance if not already done
            initialize_playwright()
            
            # Initialize fresh scraper instance
            scraper = FixedUTRScraper(email=email, password=password, headless=True)
            scraper.start_browser()
            
            if not scraper.login():
                print("Failed to log in to UTR website.")
                scraper.browser.close()
                if attempt < max_retries - 1:
                    print(f"Waiting 30 seconds before next attempt...")
                    time.sleep(30)
                continue
            
            print("Successfully logged in")
            
            start_time = time.time()
            
            # Run cross-reference directly
            valid_matches = scraper.cross_reference_matches_with_ratings()
            
            end_time = time.time()
            duration = end_time - start_time
            
            print(f"Cross-reference completed successfully in {duration/60:.2f} minutes")
            print(f"Generated {len(valid_matches)} valid matches")
            
            # Close the browser
            scraper.browser.close()
            
            # Verify the output files
            all_enhanced_file = data_dir / "all_enhanced_matches.csv"
            valid_matches_file = data_dir / "valid_matches_for_model.csv"
            
            if all_enhanced_file.exists() and valid_matches_file.exists():
                print("Successfully created enhanced matches and valid matches files")
                
                # Check year distribution in the enhanced file
                try:
                    enhanced_df = pd.read_csv(all_enhanced_file)
                    enhanced_df['date'] = pd.to_datetime(enhanced_df['date'], errors='coerce')
                    year_counts = enhanced_df.groupby(enhanced_df['date'].dt.year).size().to_dict()
                    print(f"Match year distribution in enhanced file: {year_counts}")
                    
                    # Check valid matches
                    valid_df = pd.read_csv(valid_matches_file)
                    valid_df['date'] = pd.to_datetime(valid_df['date'], errors='coerce')
                    valid_year_counts = valid_df.groupby(valid_df['date'].dt.year).size().to_dict()
                    print(f"Match year distribution in valid matches: {valid_year_counts}")
                except Exception as e:
                    print(f"Error analyzing output files: {e}")
                
                return True
            else:
                print("Failed to generate all output files.")
                if attempt < max_retries - 1:
                    print(f"Waiting 30 seconds before next attempt...")
                    time.sleep(30)
                
        except Exception as e:
            print(f"Error during cross-reference (attempt {attempt+1}): {e}")
            traceback.print_exc()
            
            if attempt < max_retries - 1:
                print(f"Waiting 30 seconds before next attempt...")
                time.sleep(30)
    
    print("Failed to complete cross-reference after maximum attempts")
    return False

def complete_processing():
    """Complete the data processing by checking what we have and filling in gaps"""
    print(f"Starting completion process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    # Create data directory reference
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    # Check if necessary directories exist
    players_dir = data_dir / "players"
    matches_dir = data_dir / "matches"
    ratings_dir = data_dir / "ratings"
    
    for directory in [players_dir, matches_dir, ratings_dir]:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Created directory: {directory}")
    
    # Setup credentials
    email = os.environ.get("UTR_EMAIL") or "zachdodson12@gmail.com"
    password = os.environ.get("UTR_PASSWORD") or "Thailand@123"
    
    # Default years to collect
    years = ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018"]
    
    # Step 4: Collect missing data for opponents
    print("\nChecking opponent data...")
    collect_missing_opponent_data(
        data_dir, 
        email, 
        password, 
        years=years, 
        dynamic_years=True
    )
    
    # Step 5: Check for missing rating histories
    # This will be handled as part of cross-reference
    
    # Step 6: Run cross-reference
    print("\nRunning final cross-reference to complete the process...")
    cross_ref_success = run_cross_reference(data_dir, email, password)
    
    # Finally, close the global playwright instance
    global playwright
    if playwright:
        playwright.stop()
    
    print("\nAll processing complete!")
    
    return cross_ref_success

if __name__ == "__main__":
    success = complete_processing()
    sys.exit(0 if success else 1)