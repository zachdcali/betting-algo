#!/usr/bin/env python3
# finish_processing.py
# Optimized script to resume processing with parallelization and progress tracking

import os
import sys
import inspect
from pathlib import Path
import pandas as pd
from datetime import datetime
import asyncio
import random
import time
from multiprocessing import Pool

from utr_scraper import UTRScraper, logger

# Diagnostic function to check for the cross-reference method
def check_scraper_methods(scraper):
    print("\n=== SCRAPER METHOD DIAGNOSTICS ===")
    
    has_method = hasattr(scraper, 'cross_reference_matches_with_ratings')
    print(f"Has cross_reference_matches_with_ratings method: {has_method}")
    
    has_class_method = hasattr(UTRScraper, 'cross_reference_matches_with_ratings')
    print(f"Class has cross_reference_matches_with_ratings method: {has_class_method}")
    
    methods = [m for m in dir(scraper) if "cross" in m.lower()]
    print(f"Methods containing 'cross': {methods}")
    
    print(f"UTRScraper module location: {inspect.getfile(UTRScraper)}")
    
    return has_method

def process_opponent_batch_worker(args):
    """
    Worker function for multiprocessing to process a batch of opponents.
    Runs async code in its own event loop.
    """
    opponent_ids, email, password, data_dir_str, years, dynamic_years = args
    data_dir = Path(data_dir_str)

    async def inner():
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        results = []
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for batch of opponents")
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
        scraper = UTRScraper(email=email, password=password, headless=True, data_dir=data_dir)
        processed_ids = []
        try:
            await scraper.start_browser()
            login_success = await scraper.login()
            if not login_success:
                logger.error(f"Process ID {os.getpid()}: Failed to login for batch of second-degree opponents")
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

    # Skip if already processed (has master match file and rating file)
    match_file = data_dir / "matches" / f"player_{player_id}_matches.csv"
    ratings_file = data_dir / "ratings" / f"player_{player_id}_ratings.csv"
    
    if match_file.exists() and ratings_file.exists():
        logger.info(f"Process ID {os.getpid()}: Top player {player_id} already has match and rating history, skipping")
        try:
            df = pd.read_csv(match_file)
            if 'opponent_id' in df.columns:
                valid_ids = df['opponent_id'].dropna().astype(str)
                second_ids = set(oid for oid in valid_ids if oid.isdigit())
                return second_ids, player_id
            return set(), player_id
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
                return None, None

            second_ids, processed_id = await process_opponent_simple(player_id, scraper, years, dynamic_years)
            return second_ids, processed_id
        except Exception as e:
            logger.error(f"Process ID {os.getpid()}: Error processing top player {player_id}: {e}")
            return None, None
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
            valid_ids = rankings_df['id'].dropna().astype(str)
            top_player_ids = set(id for id in valid_ids if id.isdigit())
            print(f"Found {len(top_player_ids)} top players from ranking file: {latest_ranking}")

    if not top_player_ids:
        print("No top players found to process")
        return False, set()

    # Count how many are already processed (have master match and rating files)
    matches_dir = data_dir / "matches"
    ratings_dir = data_dir / "ratings"
    
    match_files = {f.stem.split('_')[1] for f in matches_dir.glob("player_*_matches.csv")} if matches_dir.exists() else set()
    rating_files = {f.stem.split('_')[1] for f in ratings_dir.glob("player_*_ratings.csv")} if ratings_dir.exists() else set()
    
    fully_processed = match_files.intersection(rating_files).intersection(top_player_ids)
    to_process = top_player_ids - fully_processed
    
    print(f"Status: {len(fully_processed)}/{len(top_player_ids)} top players already fully processed")
    
    if not to_process:
        print("All top players are already fully processed")
        
        # Collect direct opponents from existing master match files
        direct_opponent_ids = set()
        for player_id in top_player_ids:
            try:
                match_file = matches_dir / f"player_{player_id}_matches.csv"
                if match_file.exists():
                    df = pd.read_csv(match_file)
                    if 'opponent_id' in df.columns:
                        valid_ids = df['opponent_id'].dropna().astype(str)
                        second_ids = set(oid for oid in valid_ids if oid.isdigit())
                        direct_opponent_ids.update(second_ids)
            except Exception as e:
                print(f"Error reading match file for player {player_id}: {e}")
        
        print(f"Collected {len(direct_opponent_ids)} direct opponents (second-degree for next step) from existing match files")
        return True, direct_opponent_ids

    num_processes = 8
    print(f"Starting parallel processing of {len(to_process)} remaining top players with {num_processes} processes...")

    direct_opponent_ids = set()
    processed_players = set()
    
    total_players = len(to_process)
    players_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(pid, email, password, str(data_dir), years, dynamic_years) for pid in to_process]
        for second_ids, processed_id in pool.imap_unordered(process_top_player_worker, worker_args):
            if second_ids is not None and processed_id is not None:
                direct_opponent_ids.update(second_ids)
                processed_players.add(processed_id)
            
            players_processed += 1
            progress_percent = (players_processed / total_players) * 100
            print(f"Progress: {players_processed}/{total_players} top players processed ({progress_percent:.2f}% complete)")

    # Add direct opponents from already processed top players
    for player_id in fully_processed:
        try:
            match_file = matches_dir / f"player_{player_id}_matches.csv"
            if match_file.exists():
                df = pd.read_csv(match_file)
                if 'opponent_id' in df.columns:
                    valid_ids = df['opponent_id'].dropna().astype(str)
                    second_ids = set(oid for oid in valid_ids if oid.isdigit())
                    direct_opponent_ids.update(second_ids)
        except Exception as e:
            print(f"Error reading match file for player {player_id}: {e}")

    # Clean to ensure only valid IDs
    valid_direct_opponents = set(oid for oid in direct_opponent_ids if oid.isdigit())
    invalid_count = len(direct_opponent_ids) - len(valid_direct_opponents)
    if invalid_count > 0:
        print(f"Filtered out {invalid_count} invalid opponent IDs")
    
    print(f"Completed processing top players. Now have {len(processed_players)} newly processed players and {len(fully_processed)} previously processed players")
    print(f"Found {len(valid_direct_opponents)} direct opponents (second-degree for next step)")
    
    return True, valid_direct_opponents

def collect_missing_opponent_data(data_dir, email, password, years=None, dynamic_years=True):
    """Collect data for missing direct opponents in parallel using multiprocessing."""
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            valid_ids = rankings_df['id'].dropna().astype(str)
            top_player_ids = set(id for id in valid_ids if id.isdigit())
            print(f"Found {len(top_player_ids)} top players from ranking file: {latest_ranking}")

    matches_dir = data_dir / "matches"
    ratings_dir = data_dir / "ratings"
    
    if not matches_dir.exists():
        print("Matches directory doesn't exist, cannot identify direct opponents")
        return False, set()
    
    # Identify direct opponents from top players' master match files
    direct_opponent_ids = set()
    for player_id in top_player_ids:
        match_file = matches_dir / f"player_{player_id}_matches.csv"
        if match_file.exists():
            try:
                df = pd.read_csv(match_file)
                if 'opponent_id' in df.columns:
                    valid_ids = df['opponent_id'].dropna().astype(str)
                    opponent_ids = set(oid for oid in valid_ids if oid.isdigit())
                    direct_opponent_ids.update(opponent_ids)
            except Exception as e:
                print(f"Error reading match file for top player {player_id}: {e}")
    
    print(f"Found {len(direct_opponent_ids)} direct opponents of top players")

    # Check which direct opponents are already fully processed (have master match and rating files)
    processed_match_files = {f.stem.split('_')[1] for f in matches_dir.glob("player_*_matches.csv")}
    processed_rating_files = {f.stem.split('_')[1] for f in ratings_dir.glob("player_*_ratings.csv")} if ratings_dir.exists() else set()
    
    fully_processed = processed_match_files.intersection(processed_rating_files).intersection(direct_opponent_ids)
    missing_matches = direct_opponent_ids - processed_match_files
    
    print(f"Status: {len(fully_processed)}/{len(direct_opponent_ids)} direct opponents already fully processed")
    print(f"Found {len(missing_matches)} direct opponents missing match histories")

    if not missing_matches:
        print("All direct opponents are already processed")
        
        # Collect second-degree opponents from existing master match files
        second_degree_opponents = set()
        for opponent_id in direct_opponent_ids:
            try:
                match_file = matches_dir / f"player_{opponent_id}_matches.csv"
                if match_file.exists():
                    df = pd.read_csv(match_file)
                    if 'opponent_id' in df.columns:
                        valid_ids = df['opponent_id'].dropna().astype(str)
                        second_ids = set(oid for oid in valid_ids if oid.isdigit())
                        second_degree_opponents.update(second_ids)
            except Exception as e:
                print(f"Error reading match file for direct opponent {opponent_id}: {e}")
        
        # Remove top players and direct opponents from second-degree opponents
        second_degree_opponents -= top_player_ids
        second_degree_opponents -= direct_opponent_ids
        
        print(f"Collected {len(second_degree_opponents)} second-degree opponents from existing direct opponent match files")
        return True, second_degree_opponents

    num_processes = 8
    print(f"Starting parallel processing of {len(missing_matches)} missing direct opponents with {num_processes} processes...")

    # Split missing matches into batches
    batch_size = 50
    missing_batches = [list(missing_matches)[i:i + batch_size] for i in range(0, len(missing_matches), batch_size)]

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
            print(f"Progress: {opponents_processed}/{total_opponents} direct opponents processed ({progress_percent:.2f}% complete)")

    # Add second-degree opponents from already processed direct opponents
    for opponent_id in fully_processed:
        try:
            match_file = matches_dir / f"player_{opponent_id}_matches.csv"
            if match_file.exists():
                df = pd.read_csv(match_file)
                if 'opponent_id' in df.columns:
                    valid_ids = df['opponent_id'].dropna().astype(str)
                    second_ids = set(oid for oid in valid_ids if oid.isdigit())
                    second_degree_opponents.update(second_ids)
        except Exception as e:
            print(f"Error reading match file for direct opponent {opponent_id}: {e}")
    
    # Remove top players and direct opponents from second-degree opponents
    second_degree_opponents -= top_player_ids
    second_degree_opponents -= direct_opponent_ids
    
    # Clean to ensure only valid IDs
    valid_second_degree = set(oid for oid in second_degree_opponents if oid.isdigit())
    invalid_count = len(second_degree_opponents) - len(valid_second_degree)
    if invalid_count > 0:
        print(f"Filtered out {invalid_count} invalid opponent IDs")
    
    print(f"Completed processing direct opponents. Now have {len(processed_opponents)} newly processed opponents and {len(fully_processed)} previously processed opponents")
    print(f"Found {len(valid_second_degree)} second-degree opponents")
    
    return True, valid_second_degree

async def process_opponent_simple(opponent_id, scraper, years, dynamic_years):
    """Process a single opponent using an existing scraper instance."""
    try:
        logger.info(f"Process ID {os.getpid()}: Processing opponent: {opponent_id}")
        
        # Get profile (includes rating history)
        profile_file = scraper.players_dir / f"player_{opponent_id}.json"
        if not profile_file.exists():
            profile = await scraper.get_player_profile(opponent_id)
            if not profile:
                logger.warning(f"Process ID {os.getpid()}: Failed to get profile for opponent {opponent_id}")
                return None, None
        else:
            logger.info(f"Process ID {os.getpid()}: Profile for opponent {opponent_id} already exists, skipping fetch")
            with open(profile_file, 'r') as f:
                import json
                profile = json.load(f)
        
        player_name = profile.get('name', f"player_{opponent_id}")
        
        # Check if the player is unrated
        if 'rated' in profile and not profile['rated']:
            logger.info(f"Process ID {os.getpid()}: Player {player_name} (ID {opponent_id}) is unrated, skipping match fetching")
            return set(), opponent_id
        
        # Ensure rating history is fetched if not already present
        rating_file = scraper.ratings_dir / f"player_{opponent_id}_ratings.csv"
        if not rating_file.exists():
            ratings = await scraper.get_player_rating_history(opponent_id)
            if isinstance(ratings, pd.DataFrame) and not ratings.empty:
                logger.info(f"Process ID {os.getpid()}: Got {len(ratings)} rating entries for {player_name}")
        
        # Dynamically fetch all available years
        player_years = years
        if dynamic_years:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    player_years = await scraper.get_available_years(opponent_id)
                    if player_years:
                        logger.info(f"Process ID {os.getpid()}: Found available years for {player_name}: {player_years}")
                        break
                    else:
                        logger.warning(f"Process ID {os.getpid()}: No available years found for {player_name} on attempt {attempt + 1}")
                        if attempt < max_retries - 1:
                            await asyncio.sleep(random.uniform(2.0, 4.0))
                except Exception as e:
                    logger.error(f"Process ID {os.getpid()}: Error getting available years for {player_name} on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(random.uniform(2.0, 4.0))
            if not player_years:
                logger.warning(f"Process ID {os.getpid()}: Falling back to broader year range for {player_name}")
                player_years = [str(year) for year in range(2025, 1999, -1)]

        all_matches = []
        for year in player_years:
            yearly_match_file = scraper.matches_dir / f"player_{opponent_id}_matches_{year}.csv"
            if yearly_match_file.exists():
                try:
                    matches = pd.read_csv(yearly_match_file)
                    if not matches.empty:
                        logger.info(f"Process ID {os.getpid()}: Loaded {len(matches)} matches for {player_name} in {year} from file")
                        all_matches.append(matches)
                    continue
                except Exception as e:
                    logger.warning(f"Process ID {os.getpid()}: Error reading yearly match file {yearly_match_file}: {e}")
            
            logger.info(f"Process ID {os.getpid()}: Getting matches for {player_name} - Year {year}")
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    # Refresh the page to ensure a fresh session state
                    await scraper.page.reload(timeout=30000)
                    await asyncio.sleep(random.uniform(2.0, 4.0))
                    matches = await scraper.get_player_match_history(opponent_id, year=year, limit=50)
                    if isinstance(matches, pd.DataFrame) and not matches.empty:
                        year_count = len(matches)
                        logger.info(f"Process ID {os.getpid()}: Got {year_count} matches for {player_name} in {year}")
                        matches.to_csv(yearly_match_file, index=False)
                        logger.info(f"Process ID {os.getpid()}: Saved {year_count} matches to {yearly_match_file}")
                        all_matches.append(matches)
                    else:
                        logger.info(f"Process ID {os.getpid()}: No matches found for {player_name} in {year}")
                    break
                except Exception as e:
                    logger.error(f"Process ID {os.getpid()}: Error getting matches for year {year} on attempt {attempt + 1}: {e}")
                    if attempt < max_retries - 1:
                        logger.info("Retrying after refreshing the page...")
                        await scraper.page.reload(timeout=30000)
                        await asyncio.sleep(random.uniform(2.0, 4.0))
                    else:
                        logger.warning(f"Process ID {os.getpid()}: Failed to fetch matches for {player_name} in {year} after {max_retries} attempts")
        
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
            valid_ids = combined_matches['opponent_id'].dropna().astype(str)
            second_ids = set(oid for oid in valid_ids if oid.isdigit())
            second_degree_opponents = set(second_ids)
        
        await asyncio.sleep(random.uniform(2.0, 4.0))
        
        return second_degree_opponents, opponent_id
    
    except Exception as e:
        logger.error(f"Process ID {os.getpid()}: Error processing opponent {opponent_id}: {e}")
        return None, None

def collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, processed_players, top_player_ids, direct_opponent_ids):
    """Collect ratings for second-degree opponents in parallel using multiprocessing."""
    if not second_degree_opponents:
        print("No second-degree opponents to process")
        return True
    
    # Verify the ratings directory path
    ratings_dir = data_dir / "ratings"
    print(f"Checking ratings directory: {ratings_dir}")
    if not ratings_dir.exists():
        print(f"Ratings directory does not exist: {ratings_dir}")
        return False
    
    # Count existing rating files
    existing_rating_files = list(ratings_dir.glob("player_*_ratings.csv"))
    print(f"Found {len(existing_rating_files)} existing rating files in {ratings_dir}")
    
    # Remove any second-degree opponents that are top players, direct opponents, or already processed
    second_degree_opponents -= top_player_ids
    second_degree_opponents -= direct_opponent_ids
    second_degree_opponents -= processed_players
    existing_rating_ids = {f.stem.split('_')[1] for f in existing_rating_files}
    second_degree_opponents -= existing_rating_ids
    print(f"After removing top players, direct opponents, processed players, and those with existing ratings, {len(second_degree_opponents)} second-degree opponents remain")
    
    if not second_degree_opponents:
        print("All second-degree opponents have rating histories, proceeding")
        return True

    # Split into batches
    batch_size = 50
    opponent_batches = [list(second_degree_opponents)[i:i + batch_size] for i in range(0, len(second_degree_opponents), batch_size)]

    num_processes = 8
    print(f"Starting parallel processing of {len(second_degree_opponents)} second-degree opponents with {num_processes} processes...")

    total_opponents = len(second_degree_opponents)
    opponents_processed = 0

    with Pool(processes=num_processes) as pool:
        worker_args = [(batch, email, password, str(data_dir)) for batch in opponent_batches]
        for batch_processed_ids in pool.imap_unordered(process_second_degree_opponent_batch_worker, worker_args):
            opponents_processed += len(batch_processed_ids)
            progress_percent = (opponents_processed / total_opponents) * 100
            print(f"Progress: {opponents_processed}/{total_opponents} second-degree opponents processed ({progress_percent:.2f}% complete)")

    print(f"Completed processing {opponents_processed} second-degree opponents")
    
    # Verify all were processed
    current_rating_files = list(ratings_dir.glob("player_*_ratings.csv"))
    current_rating_ids = {f.stem.split('_')[1] for f in current_rating_files}
    unprocessed = set(second_degree_opponents) - current_rating_ids
    
    if unprocessed:
        print(f"Warning: {len(unprocessed)} second-degree opponents still don't have rating files")
        print(f"First few unprocessed IDs: {list(unprocessed)[:5]}")
    else:
        print("All targeted second-degree opponents now have rating files")
    
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
        raise

async def run_cross_reference(data_dir, email, password):
    """Run the cross-reference process to generate enhanced files with retries."""
    print("\nRunning cross-reference to generate enhanced files...")
    
    all_enhanced_file = data_dir / "all_enhanced_matches.csv"
    valid_matches_file = data_dir / "valid_matches_for_model.csv"
    
    if all_enhanced_file.exists() and valid_matches_file.exists():
        print("Enhanced match files already exist.")
        try:
            all_matches = pd.read_csv(all_enhanced_file)
            valid_matches = pd.read_csv(valid_matches_file)
            print(f"all_enhanced_matches.csv contains {len(all_matches)} matches")
            print(f"valid_matches_for_model.csv contains {len(valid_matches)} matches")
            
            user_input = input("Do you want to regenerate these files? (y/n): ").lower()
            if user_input != 'y':
                print("Skipping cross-reference step as files already exist.")
                return True
            else:
                print("Will regenerate enhanced match files...")
        except Exception as e:
            print(f"Error reading existing files: {e}")
            print("Will attempt to regenerate...")
    
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
            
            has_method = check_scraper_methods(scraper)
            
            cross_ref_method = 'cross_reference_matches_with_ratings'
            if not has_method:
                print("CRITICAL: cross_reference_matches_with_ratings method not found!")
                print("Looking for alternative methods...")
                
                method_names = dir(scraper)
                cross_methods = [m for m in method_names if "cross_reference" in m.lower()]
                print(f"Methods that might be alternatives: {cross_methods}")
                
                if cross_methods:
                    cross_ref_method = cross_methods[0]
                    print(f"Using alternative method: {cross_ref_method}")
                else:
                    print("No alternative cross-reference method found. Skipping cross-referencing.")
                    await scraper.close_browser()
                    print("WARNING: Cross-referencing skipped due to missing method. Proceeding to completion.")
                    return True
            
            start_time = time.time()
            
            print(f"Calling {cross_ref_method} method...")
            cross_ref_func = getattr(scraper, cross_ref_method)
            await cross_ref_func()
            print("Method call completed successfully")
            
            end_time = time.time()
            duration = (end_time - start_time) / 60
            print(f"Cross-reference completed in {duration:.2f} minutes")
            
            match_count = 0
            if all_enhanced_file.exists():
                try:
                    matches_df = pd.read_csv(all_enhanced_file)
                    match_count = len(matches_df)
                except Exception as e:
                    print(f"Error reading enhanced matches file: {e}")
            
            print(f"Generated {match_count} enhanced matches")
            
            if all_enhanced_file.exists() and valid_matches_file.exists():
                print("Successfully created enhanced matches file")
                await scraper.close_browser()
                return True
            else:
                print("Failed to generate proper output files.")
                if attempt < max_retries - 1:
                    print(f"Waiting 30 seconds before next attempt...")
                    await asyncio.sleep(30)
                
        except Exception as e:
            print(f"Error during cross-reference (attempt {attempt + 1}): {e}")
            import traceback
            traceback.print_exc()
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
    
    # Count existing files for status report
    match_files = list((data_dir / "matches").glob("player_*_matches.csv")) if (data_dir / "matches").exists() else []
    rating_files = list((data_dir / "ratings").glob("player_*_ratings.csv")) if (data_dir / "ratings").exists() else []
    player_files = list((data_dir / "players").glob("player_*.json")) if (data_dir / "players").exists() else []
    
    print(f"\nCurrent state of data directories:")
    print(f"- Master match files: {len(match_files)}")
    print(f"- Rating files: {len(rating_files)}")
    print(f"- Player profile files: {len(player_files)}")
    
    print("\nCollecting data for top players...")
    success, direct_opponent_ids = collect_top_players_data(data_dir, email, password, years, True)
    if not success:
        print("Failed to collect top players data. Exiting.")
        return False
    
    print("\nChecking direct opponent data...")
    success, second_degree_opponents = collect_missing_opponent_data(data_dir, email, password, years, True)
    if not success:
        print("Failed to collect direct opponent data. Exiting.")
        return False
    
    # Get top player IDs for exclusion in second-degree processing
    top_player_ids = set()
    ranking_files = list(data_dir.glob("*_utr_rankings_*.csv"))
    if ranking_files:
        ranking_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_ranking = ranking_files[0]
        rankings_df = pd.read_csv(latest_ranking)
        if 'id' in rankings_df.columns:
            valid_ids = rankings_df['id'].dropna().astype(str)
            top_player_ids = set(id for id in valid_ids if id.isdigit())
    
    print(f"\nCombined second-degree opponents before filtering: {len(second_degree_opponents)}")
    
    print("\nChecking second-degree opponent data...")
    processed_players = {player_file.stem.split('_')[1] for player_file in (data_dir / "players").glob("player_*.json")}
    success = collect_second_degree_opponents(data_dir, email, password, second_degree_opponents, processed_players, top_player_ids, direct_opponent_ids)
    if not success:
        print("Failed to collect second-degree opponent data. Exiting.")
        return False
    
    # Count files after processing for status report
    match_files_after = list((data_dir / "matches").glob("player_*_matches.csv")) if (data_dir / "matches").exists() else []
    rating_files_after = list((data_dir / "ratings").glob("player_*_ratings.csv")) if (data_dir / "ratings").exists() else []
    player_files_after = list((data_dir / "players").glob("player_*.json")) if (data_dir / "players").exists() else []
    
    print(f"\nCurrent state of data directories after processing:")
    print(f"- Master match files: {len(match_files_after)} (+{len(match_files_after) - len(match_files)})")
    print(f"- Rating files: {len(rating_files_after)} (+{len(rating_files_after) - len(rating_files)})")
    print(f"- Player profile files: {len(player_files_after)} (+{len(player_files_after) - len(player_files)})")
    
    print("\nRunning final cross-reference...")
    cross_ref_success = await run_cross_reference(data_dir, email, password)
    
    print("\nAll processing complete!")
    return cross_ref_success

if __name__ == "__main__":
    success = asyncio.run(complete_processing())
    sys.exit(0 if success else 1)
