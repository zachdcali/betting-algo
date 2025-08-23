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
import argparse
from multiprocessing import Pool
from utr_scraper_cloud import UTRScraper, logger

def process_opponent_batch_worker(args):
    """
    Worker function for multiprocessing to process a batch of opponents.
    Runs async code in its own event loop.
    """
    opponent_ids, email, password, data_dir_str, years, dynamic_years = args
    data_dir = Path(data_dir_str)

    async def inner():
        # Stagger login based on process ID to avoid overwhelming the server
        worker_delay = (os.getpid() % 8) * 5  # 0-35 second delay spread
        await asyncio.sleep(worker_delay)
        
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        results = []
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for batch of {len(opponent_ids)} opponents")
                return results

            for opponent_id in opponent_ids:
                try:
                    second_ids, processed_id = await process_opponent_simple(opponent_id, scraper, years, dynamic_years)
                    if second_ids is not None and processed_id is not None:
                        results.append((second_ids, processed_id))
                except Exception as e:
                    logger.error(f"Process ID {os.getpid()}: Error processing opponent {opponent_id}: {e}")
                    # Restart browser on error
                    await scraper.close_browser()
                    await scraper.start_browser()
                    login_success = await scraper.login()
                    if not login_success:
                        logger.error(f"Process ID {os.getpid()}: Failed to re-login after error")
                        break
        finally:
            await scraper.close_browser()
        return results

    return asyncio.run(inner())

def process_second_degree_opponent_batch_worker(args):
    """Worker function for multiprocessing to process a batch of second-degree opponents."""
    opponent_ids, email, password, data_dir_str = args
    data_dir = Path(data_dir_str)

    async def inner():
        # Stagger login based on process ID to avoid overwhelming the server
        worker_delay = (os.getpid() % 8) * 5  # 0-35 second delay spread
        await asyncio.sleep(worker_delay)
        
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        processed_ids = []
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for batch of {len(opponent_ids)} second-degree opponents")
                return processed_ids

            for opponent_id in opponent_ids:
                try:
                    await process_second_degree_opponent_simple(opponent_id, scraper)
                    processed_ids.append(opponent_id)
                except Exception as e:
                    logger.error(f"Process ID {os.getpid()}: Error processing second-degree opponent {opponent_id}: {e}")
                    # Restart browser on error
                    await scraper.close_browser()
                    await scraper.start_browser()
                    login_success = await scraper.login()
                    if not login_success:
                        logger.error(f"Process ID {os.getpid()}: Failed to re-login after error")
                        break
        finally:
            await scraper.close_browser()
        return processed_ids

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

def collect_top_players_data(data_dir, email, password, years=None, dynamic_years=True, num_processes=4):
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

def collect_missing_opponent_data(data_dir, email, password, years=None, dynamic_years=True, num_processes=4):
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

    # Split missing matches into batches
    BATCH_SIZE = 50
    missing_batches = [list(missing_matches)[i:i + BATCH_SIZE] for i in range(0, len(missing_matches), BATCH_SIZE)]

    print(f"Starting parallel processing with {num_processes} processes...")
    print(f"Processing {len(missing_batches)} batches of up to {BATCH_SIZE} opponents each")

    second_degree_opponents = set()
    processed_opponents = set()
    
    total_opponents = len(missing_matches)
    opponents_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(batch, email, password, str(data_dir), years, dynamic_years) for batch in missing_batches]
        for batch_results in pool.imap_unordered(process_opponent_batch_worker, worker_args):
            for second_ids, processed_id in batch_results:
                second_degree_opponents.update(second_ids)
                processed_opponents.add(processed_id)
            
            opponents_processed += len(batch_results)
            progress_percent = (opponents_processed / total_opponents) * 100
            batches_done = (opponents_processed // BATCH_SIZE) + (1 if opponents_processed % BATCH_SIZE else 0)
            print(f"Progress: {batches_done}/{len(missing_batches)} batches processed ({progress_percent:.2f}% complete) - {opponents_processed} opponents done")

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

def collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, players_to_exclude, num_processes=4):
    """Collect ratings for second-degree opponents in parallel using multiprocessing."""
    if not second_degree_opponents:
        print("No second-degree opponents to process")
        return True
    
    second_tasks = []
    for opponent_id in second_degree_opponents:
        if opponent_id in players_to_exclude:
            continue
        rating_file = data_dir / "ratings" / f"player_{opponent_id}_ratings.csv"
        if rating_file.exists():
            continue
        second_tasks.append(opponent_id)
    
    print(f"Found {len(second_tasks)} second-degree opponents needing ratings")
    
    if not second_tasks:
        return True

    # Split into batches of 50 opponents each
    BATCH_SIZE = 50
    batches = [second_tasks[i:i + BATCH_SIZE] for i in range(0, len(second_tasks), BATCH_SIZE)]
    
    print(f"Starting parallel processing with {num_processes} processes for second-degree opponents...")
    print(f"Processing {len(batches)} batches of up to {BATCH_SIZE} opponents each")

    batches_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(batch, email, password, str(data_dir)) for batch in batches]
        for batch_processed_ids in pool.imap_unordered(process_second_degree_opponent_batch_worker, worker_args):
            batches_processed += 1
            progress_percent = (batches_processed / len(batches)) * 100
            opponents_done = min(batches_processed * BATCH_SIZE, len(second_tasks))
            print(f"Progress: {batches_processed}/{len(batches)} batches processed ({progress_percent:.2f}% complete) - {opponents_done} opponents done")

    print(f"Completed processing {len(batches)} batches covering {len(second_tasks)} second-degree opponents")
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

async def complete_processing(num_processes=4):
    """Main async function to orchestrate the entire processing pipeline."""
    print(f"Starting completion process at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}...")
    print(f"Using {num_processes} parallel processes")
    
    base_dir = Path(__file__).parent.parent.parent  # Go to project root
    data_dir = base_dir / "data"
    
    for directory in [data_dir / "players", data_dir / "matches", data_dir / "ratings"]:
        directory.mkdir(parents=True, exist_ok=True)
    
    email = os.environ.get("UTR_EMAIL", "zachdodson12@gmail.com")
    password = os.environ.get("UTR_PASSWORD", "Thailand@123")
    years = ["2025", "2024", "2023", "2022", "2021", "2020", "2019", "2018"]
    
    print("\nCollecting data for top players...")
    success, second_degree_opponents = collect_top_players_data(data_dir, email, password, years, True, num_processes)
    if not success:
        print("Failed to collect top players data. Exiting.")
        return False
    
    print("\nChecking opponent data...")
    success, more_second_degree_opponents = collect_missing_opponent_data(data_dir, email, password, years, True, num_processes)
    if not success:
        print("Failed to collect opponent data. Exiting.")
        return False
    
    second_degree_opponents.update(more_second_degree_opponents)
    
    # Also collect second-degree opponents from ALL existing first-degree opponent match files
    print("\nCollecting second-degree opponents from all first-degree opponent match files...")
    top_player_ids_temp = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            top_player_ids_temp = set(rankings_df['id'].dropna().astype(str))
    
    # Get ALL first-degree opponents (from existing match files)
    all_first_degree_opponents = set()
    matches_dir = data_dir / "matches"
    for player_id in top_player_ids_temp:
        match_file = matches_dir / f"player_{player_id}_matches.csv"
        if match_file.exists():
            df = pd.read_csv(match_file)
            if 'opponent_id' in df.columns:
                file_opponent_ids = df['opponent_id'].dropna().astype(str).unique()
                all_first_degree_opponents.update(file_opponent_ids)
    
    # Now get second-degree opponents from ALL first-degree opponent match files
    print(f"Scanning {len(all_first_degree_opponents)} first-degree opponent match files for second-degree opponents...")
    for opponent_id in all_first_degree_opponents:
        match_file = matches_dir / f"player_{opponent_id}_matches.csv"
        if match_file.exists():
            try:
                df = pd.read_csv(match_file)
                if 'opponent_id' in df.columns:
                    file_second_ids = df['opponent_id'].dropna().astype(str).unique()
                    second_degree_opponents.update(file_second_ids)
            except Exception as e:
                print(f"Warning: Error reading {match_file}: {e}")
    
    print(f"Total second-degree opponents found: {len(second_degree_opponents)}")
    
    print("\nChecking second-degree opponent data...")
    # Get top players to exclude from second-degree opponents
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            top_player_ids = set(rankings_df['id'].dropna().astype(str))
    
    # Get first-degree opponents to exclude from second-degree opponents  
    first_degree_opponents = set()
    matches_dir = data_dir / "matches"
    for player_id in top_player_ids:
        match_file = matches_dir / f"player_{player_id}_matches.csv"
        if match_file.exists():
            df = pd.read_csv(match_file)
            if 'opponent_id' in df.columns:
                file_opponent_ids = df['opponent_id'].dropna().astype(str).unique()
                first_degree_opponents.update(file_opponent_ids)
    
    # Second-degree opponents should exclude top players AND first-degree opponents
    players_to_exclude = top_player_ids.union(first_degree_opponents)
    print(f"Excluding {len(top_player_ids)} top players and {len(first_degree_opponents)} first-degree opponents")
    
    success = collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, players_to_exclude, num_processes)
    if not success:
        print("Failed to collect second-degree opponent data. Exiting.")
        return False
    
    print("\nRunning final cross-reference...")
    cross_ref_success = await run_cross_reference(data_dir, email, password)
    
    print("\nAll processing complete!")
    return cross_ref_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Complete UTR data processing with parallel processing")
    parser.add_argument("--processes", type=int, default=4, help="Number of parallel processes to use (default: 4)")
    args = parser.parse_args()
    
    success = asyncio.run(complete_processing(num_processes=args.processes))
    sys.exit(0 if success else 1)