#!/usr/bin/env python3
# finish_processing.py
# Optimized script to resume processing with parallelization and progress tracking

import os
import sys
from pathlib import Path
import pandas as pd
from datetime import datetime
import asyncio
import random
import time
from multiprocessing import Pool
from utr_scraper import UTRScraper, logger

def process_opponent_worker(args):
    """
    Worker function for multiprocessing to process a single opponent.
    Runs async code in its own event loop.
    """
    opponent_id, email, password, data_dir_str, years, dynamic_years = args
    data_dir = Path(data_dir_str)  # Convert string back to Path object

    async def inner():
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for opponent {opponent_id}")
                return None, None

            second_ids, processed_id = await process_opponent_simple(opponent_id, scraper, years, dynamic_years)
            return second_ids, processed_id
        except Exception as e:
            logger.error(f"Process ID {os.getpid()}: Error processing opponent {opponent_id}: {e}")
            return None, None
        finally:
            await scraper.close_browser()

    return asyncio.run(inner())

def process_second_degree_opponent_worker(args):
    """Worker function for multiprocessing to process a second-degree opponent."""
    opponent_id, email, password, data_dir_str = args
    data_dir = Path(data_dir_str)

    async def inner():
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for second-degree opponent {opponent_id}")
                return
            await process_second_degree_opponent_simple(opponent_id, scraper)
        except Exception as e:
            logger.error(f"Process ID {os.getpid()}: Error processing second-degree opponent {opponent_id}: {e}")
        finally:
            await scraper.close_browser()

    return asyncio.run(inner())

def process_top_player_worker(args):
    """Worker function for multiprocessing to process a top player (ratings and matches)."""
    player_id, email, password, data_dir_str, years, dynamic_years = args
    data_dir = Path(data_dir_str)

    # Skip if already processed
    match_file = data_dir / "matches" / f"player_{player_id}_matches.csv"
    if match_file.exists():
        logger.info(f"Process ID {os.getpid()}: Top player {player_id} already has match history, skipping")
        # Still need to return second-degree opponents if any
        try:
            df = pd.read_csv(match_file)
            second_ids = set(df['opponent_id'].dropna().astype(str).unique()) if 'opponent_id' in df.columns else set()
            return second_ids, player_id
        except Exception as e:
            logger.error(f"Process ID {os.getpid()}: Error reading match file for player {player_id}: {e}")
            return set(), player_id

    async def inner():
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for top player {player_id}")
                return None

            second_ids, processed_id = await process_opponent_simple(player_id, scraper, years, dynamic_years)
            return second_ids, processed_id
        except Exception as e:
            logger.error(f"Process ID {os.getpid()}: Error processing top player {player_id}: {e}")
            return None
        finally:
            await scraper.close_browser()

    return asyncio.run(inner())

def collect_top_players_data(data_dir, email, password, years=None, dynamic_years=True):
    """Collect data for top players in parallel using multiprocessing."""
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            top_player_ids = set(rankings_df['id'].dropna().astype(str))
            print(f"Found {len(top_player_ids)} top players from ranking file: {latest_ranking}")
    
    if not top_player_ids:
        print("No top players found to process")
        return False, set()

    num_processes = 8
    print(f"Starting parallel processing of top players with {num_processes} processes...")

    second_degree_opponents = set()
    processed_players = set()
    
    total_players = len(top_player_ids)
    players_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(pid, email, password, str(data_dir), years, dynamic_years) for pid in top_player_ids]
        for second_ids, processed_id in pool.imap_unordered(process_top_player_worker, worker_args):
            if second_ids is not None and processed_id is not None:
                second_degree_opponents.update(second_ids)
                processed_players.add(processed_id)
            
            players_processed += 1
            progress_percent = (players_processed / total_players) * 100
            print(f"Progress: {players_processed}/{total_players} top players processed ({progress_percent:.2f}% complete)")

    print(f"Completed processing {len(processed_players)} top players with {len(second_degree_opponents)} second-degree opponents found")
    return True, second_degree_opponents

def collect_missing_opponent_data(data_dir, email, password, years=None, dynamic_years=True):
    """Collect data for missing opponents in parallel using multiprocessing."""
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            top_player_ids = set(rankings_df['id'].dropna().astype(str))
            print(f"Found {len(top_player_ids)} top players from ranking file: {latest_ranking}")

    matches_dir = data_dir / "matches"
    direct_opponent_ids = set()
    for player_id in top_player_ids:
        match_file = matches_dir / f"player_{player_id}_matches.csv"
        if match_file.exists():
            df = pd.read_csv(match_file)
            if 'opponent_id' in df.columns:
                file_opponent_ids = df['opponent_id'].dropna().astype(str).unique()
                direct_opponent_ids.update(file_opponent_ids)

    print(f"Found {len(direct_opponent_ids)} direct opponents of top players")

    missing_matches = [oid for oid in direct_opponent_ids if not (matches_dir / f"player_{oid}_matches.csv").exists()]
    print(f"Found {len(missing_matches)} direct opponents missing match histories")

    if not missing_matches:
        return True, set()

    num_processes = 8
    print(f"Starting parallel processing with {num_processes} processes...")

    second_degree_opponents = set()
    processed_opponents = set()
    
    total_opponents = len(missing_matches)
    opponents_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(oid, email, password, str(data_dir), years, dynamic_years) for oid in missing_matches]
        for second_ids, processed_id in pool.imap_unordered(process_opponent_worker, worker_args):
            if second_ids is not None and processed_id is not None:
                second_degree_opponents.update(second_ids)
                processed_opponents.add(processed_id)
            
            opponents_processed += 1
            progress_percent = (opponents_processed / total_opponents) * 100
            print(f"Progress: {opponents_processed}/{total_opponents} opponents processed ({progress_percent:.2f}% complete)")

    print(f"Completed processing {len(processed_opponents)} opponents with {len(second_degree_opponents)} second-degree opponents found")
    return True, second_degree_opponents

async def process_opponent_simple(opponent_id, scraper, years, dynamic_years):
    """Process a single opponent using an existing scraper instance."""
    try:
        logger.info(f"Process ID {os.getpid()}: Processing opponent: {opponent_id}")
        
        profile = await scraper.get_player_profile(opponent_id)
        if not profile:
            logger.warning(f"Process ID {os.getpid()}: Failed to get profile for opponent {opponent_id}")
            return None, None
        
        player_name = profile.get('name', 'Unknown')
        logger.info(f"Process ID {os.getpid()}: Got profile for {player_name}")
        
        ratings = await scraper.get_player_rating_history(opponent_id)
        if isinstance(ratings, pd.DataFrame) and not ratings.empty:
            logger.info(f"Process ID {os.getpid()}: Got {len(ratings)} rating entries for {player_name}")
        
        player_years = years
        if dynamic_years:
            player_years = await scraper.get_available_years(opponent_id)
            if not player_years:
                player_years = years
        
        all_matches = []
        for year in player_years:
            logger.info(f"Process ID {os.getpid()}: Getting matches for {player_name} - Year {year}")
            try:
                matches = await scraper.get_player_match_history(opponent_id, year=year, limit=50)
                if isinstance(matches, pd.DataFrame) and not matches.empty:
                    year_count = len(matches)
                    logger.info(f"Process ID {os.getpid()}: Got {year_count} matches for {player_name} in {year}")
                    all_matches.append(matches)
                else:
                    logger.info(f"Process ID {os.getpid()}: No matches found for {player_name} in {year}")
                await asyncio.sleep(random.uniform(2.0, 4.0))
            except Exception as e:
                logger.error(f"Process ID {os.getpid()}: Error getting matches for year {year}: {e}")
        
        if all_matches:
            combined_matches = pd.concat([df for df in all_matches if not df.empty], ignore_index=True)
            if not combined_matches.empty:
                logger.info(f"Process ID {os.getpid()}: Combined {len(all_matches)} years of match data")
                logger.info(f"Process ID {os.getpid()}: Total matches before deduplication: {len(combined_matches)}")
                combined_matches.drop_duplicates(subset=['match_id'], keep='first', inplace=True)
                logger.info(f"Process ID {os.getpid()}: Removed {len(all_matches) - len(combined_matches)} duplicate matches")
                
                match_file = scraper.matches_dir / f"player_{opponent_id}_matches.csv"
                if match_file.exists():
                    try:
                        existing_matches = pd.read_csv(match_file)
                        combined_matches = pd.concat([existing_matches, combined_matches], ignore_index=True)
                        combined_matches.drop_duplicates(subset=['match_id'], keep='first', inplace=True)
                    except Exception as e:
                        logger.warning(f"Process ID {os.getpid()}: Error reading existing matches file {match_file}: {e}")
                
                logger.info(f"Process ID {os.getpid()}: Combined new and existing data for a total of {len(combined_matches)} matches")
                combined_matches.to_csv(match_file, index=False)
                logger.info(f"Process ID {os.getpid()}: Saved {len(combined_matches)} total matches to main file for player {opponent_id}")
            else:
                combined_matches = pd.DataFrame()
        else:
            combined_matches = pd.DataFrame()
        
        match_count = len(combined_matches) if not combined_matches.empty else 0
        logger.info(f"Process ID {os.getpid()}: Retrieved total of {match_count} matches for {player_name}")
        
        second_degree_opponents = set()
        if not combined_matches.empty and 'opponent_id' in combined_matches.columns:
            second_ids = combined_matches['opponent_id'].dropna().astype(str).unique()
            second_degree_opponents = set(second_ids)
        
        await asyncio.sleep(random.uniform(2.0, 4.0))
        
        return second_degree_opponents, opponent_id
    
    except Exception as e:
        logger.error(f"Process ID {os.getpid()}: Error processing opponent {opponent_id}: {e}")
        return None, None

def collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, processed_players):
    """Collect ratings for second-degree opponents in parallel using multiprocessing."""
    if not second_degree_opponents:
        print("No second-degree opponents to process")
        return True
    
    second_tasks = []
    for opponent_id in second_degree_opponents:
        if opponent_id in processed_players:
            continue
        rating_file = data_dir / "ratings" / f"player_{opponent_id}_ratings.csv"
        if rating_file.exists():
            continue
        second_tasks.append(opponent_id)
    
    print(f"Found {len(second_tasks)} second-degree opponents needing ratings")
    
    if not second_tasks:
        return True

    num_processes = 8
    print(f"Starting parallel processing with {num_processes} processes for second-degree opponents...")

    total_second_opponents = len(second_tasks)
    second_opponents_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(oid, email, password, str(data_dir)) for oid in second_tasks]
        for _ in pool.imap_unordered(process_second_degree_opponent_worker, worker_args):
            second_opponents_processed += 1
            progress_percent = (second_opponents_processed / total_second_opponents) * 100
            print(f"Progress: {second_opponents_processed}/{total_second_opponents} second-degree opponents processed ({progress_percent:.2f}% complete)")

    print(f"Completed processing {second_opponents_processed} second-degree opponents")
    return True

async def process_second_degree_opponent_simple(opponent_id, scraper):
    """Process a second-degree opponent (ratings only) using an existing scraper instance."""
    try:
        logger.info(f"Process ID {os.getpid()}: Processing second-degree opponent: {opponent_id}")
        
        profile_file = scraper.players_dir / f"player_{opponent_id}.json"
        if profile_file.exists():
            logger.info(f"Process ID {os.getpid()}: Profile for opponent {opponent_id} already exists, skipping")
        else:
            await scraper.get_player_profile(opponent_id)
        
        rating_file = scraper.ratings_dir / f"player_{opponent_id}_ratings.csv"
        if not rating_file.exists():
            await scraper.get_player_rating_history(opponent_id)
        
        await asyncio.sleep(random.uniform(2.0, 4.0))
        
    except Exception as e:
        logger.error(f"Process ID {os.getpid()}: Error processing second-degree opponent {opponent_id}: {e}")

async def run_cross_reference(data_dir, email, password):
    """Run the cross-reference process to generate enhanced files with retries."""
    print("\nRunning cross-reference to generate enhanced files...")
    
    max_retries = 3
    for attempt in range(max_retries):
        scraper = None
        try:
            print(f"Attempt {attempt + 1}/{max_retries}: Running cross-reference process...")
            
            scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
            await scraper.start_browser()
            
            login_success = await scraper.login()
            if not login_success:
                print("Failed to log in to UTR website.")
                await scraper.close_browser()
                if attempt < max_retries - 1:
                    print(f"Waiting 30 seconds before next attempt...")
                    await asyncio.sleep(30)
                continue
            
            print("Successfully logged in")
            start_time = time.time()
            
            valid_matches = await scraper.cross_reference_matches_with_ratings()
            
            end_time = time.time()
            duration = (end_time - start_time) / 60
            print(f"Cross-reference completed successfully in {duration:.2f} minutes")
            print(f"Generated {len(valid_matches)} valid matches")
            
            await scraper.close_browser()
            scraper = None
            
            all_enhanced_file = data_dir / "all_enhanced_matches.csv"
            valid_matches_file = data_dir / "valid_matches_for_model.csv"
            
            if all_enhanced_file.exists() and valid_matches_file.exists():
                print("Successfully created enhanced matches and valid matches files")
                return True
            else:
                print("Failed to generate all output files.")
                if attempt < max_retries - 1:
                    print(f"Waiting 30 seconds before next attempt...")
                    await asyncio.sleep(30)
                
        except Exception as e:
            print(f"Error during cross-reference (attempt {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                print(f"Waiting 30 seconds before next attempt...")
                await asyncio.sleep(30)
        finally:
            if scraper:
                try:
                    await scraper.close_browser()
                except Exception as close_err:
                    print(f"Error closing scraper browser: {close_err}")
    
    print("Failed to complete cross-reference after maximum attempts")
    return False

async def complete_processing():
    """Main async function to orchestrate the entire processing pipeline."""
    print(f"Starting completion process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    
    for directory in [data_dir / "players", data_dir / "matches", data_dir / "ratings"]:
        directory.mkdir(parents=True, exist_ok=True)
    
    email = os.environ.get("UTR_EMAIL", "zachdodson12@gmail.com")
    password = os.environ.get("UTR_PASSWORD", "Thailand@123")
    years = ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018"]
    
    print("\nCollecting data for top players...")
    success, second_degree_opponents = collect_top_players_data(data_dir, email, password, years, True)
    if not success:
        print("Failed to collect top players data. Exiting.")
        return False
    
    print("\nChecking opponent data...")
    success, more_second_degree_opponents = collect_missing_opponent_data(data_dir, email, password, years, True)
    if not success:
        print("Failed to collect opponent data. Exiting.")
        return False
    
    second_degree_opponents.update(more_second_degree_opponents)
    
    print("\nChecking second-degree opponent data...")
    processed_players = {player_file.stem.split('_')[1] for player_file in (data_dir / "players").glob("player_*.json")}
    success = collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, processed_players)
    if not success:
        print("Failed to collect second-degree opponent data. Exiting.")
        return False
    
    print("\nRunning final cross-reference...")
    cross_ref_success = await run_cross_reference(data_dir, email, password)
    
    print("\nAll processing complete!")
    return cross_ref_success

if __name__ == "__main__":
    success = asyncio.run(complete_processing())
    sys.exit(0 if success else 1)