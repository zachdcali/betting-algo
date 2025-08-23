# Replace the Playwright import at the top of utr_scraper.py
from playwright.async_api import async_playwright, Playwright, TimeoutError as PlaywrightTimeoutError
import asyncio
# Keep other imports as they are
import pandas as pd
import time
import json
import os
from pathlib import Path
from datetime import datetime
import re
import logging
import random
import numpy as np
import csv
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("utr_scraping.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class UTRScraper:
    def __init__(self, email, password, headless=True, data_dir=None):
        self.email = email
        self.password = password
        self.headless = headless
        
        if data_dir is None:
            base_dir = Path(__file__).parent.parent
            data_dir = base_dir / "data"
        
        self.data_dir = Path(data_dir)
        self.players_dir = self.data_dir / "players"
        self.matches_dir = self.data_dir / "matches"
        self.ratings_dir = self.data_dir / "ratings"
        
        for directory in [self.data_dir, self.players_dir, self.matches_dir, self.ratings_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
        self.logged_in = False
    
    async def __aenter__(self):
        await self.start_browser()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close_browser()
    
    async def start_browser(self):
        try:
            if self.browser is not None or self.playwright is not None:
                logger.info("Browser already initialized, closing existing instance first")
                await self.close_browser()
                
            logger.info("Starting new browser instance")
            self.playwright = await async_playwright().start()
            if not self.playwright:
                raise RuntimeError("Failed to initialize Playwright")
                
            self.browser = await self.playwright.chromium.launch(headless=self.headless)
            if not self.browser:
                raise RuntimeError("Failed to launch browser")
                
            self.context = await self.browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                viewport={"width": 1280, "height": 800}
            )
            if not self.context:
                raise RuntimeError("Failed to create browser context")
                
            self.page = await self.context.new_page()
            if not self.page:
                raise RuntimeError("Failed to create new page")
                
            self.page.set_default_timeout(30000)  # Synchronous call, no await
            logger.info("Browser initialized successfully")
            return self
        except Exception as e:
            logger.error(f"Error initializing browser: {e}")
            # Cleanup code...
            raise
    
    async def close_browser(self):
        """Close browser and clean up resources with proper error handling"""
        logger.info("Closing browser and cleaning up resources")
        
        # Clean up in reverse order of creation
        if self.page:
            try:
                await self.page.close()
                logger.info("Closed page")
            except Exception as e:
                logger.warning(f"Error closing page: {e}")
            self.page = None
            
        if self.context:
            try:
                await self.context.close()
                logger.info("Closed browser context")
            except Exception as e:
                logger.warning(f"Error closing browser context: {e}")
            self.context = None
            
        if self.browser:
            try:
                await self.browser.close()
                logger.info("Closed browser")
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
            self.browser = None
            
        if self.playwright:
            try:
                await self.playwright.stop()
                logger.info("Stopped playwright")
            except Exception as e:
                logger.warning(f"Error stopping playwright: {e}")
            self.playwright = None
            
        self.logged_in = False
        logger.info("Browser resources cleaned up")
    
    async def login(self):
        """Log in to UTR website"""
        if self.logged_in:
            return True
            
        try:
            logger.info("Logging in to UTR...")
            await self.page.goto("https://app.utrsports.net/login")
            
            # Wait a bit for the page to fully load
            await asyncio.sleep(3)
            
            # Take a screenshot of the login page to verify its structure
            await self.page.screenshot(path="login_page.png")
            logger.info("Login page loaded, checking for login form...")
            
            # Check for and dismiss cookie consent if present
            try:
                cookie_button = await self.page.query_selector('button[data-testid="onetrust-accept-btn-handler"]')
                if cookie_button:
                    logger.info("Accepting cookies...")
                    await cookie_button.click()
                    await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"No cookie banner found or error dismissing it: {e}")
            
            # Wait for the email input using ID selector
            logger.info("Waiting for email input field...")
            await self.page.wait_for_selector('#emailInput', state="visible", timeout=30000)
            
            # Fill in login credentials using IDs
            logger.info("Filling email field...")
            await self.page.fill('#emailInput', self.email)
            
            logger.info("Filling password field...")
            await self.page.fill('#passwordInput', self.password)
            
            logger.info("Clicking sign in button...")
            # Use a more specific selector for the sign in button
            try:
                # Try using the text content
                await self.page.click('button:has-text("SIGN IN")')
            except Exception as e:
                logger.warning(f"Could not click button by text, trying alternate selector: {e}")
                try:
                    # Try using a button that's a direct child of the form
                    await self.page.click('form > button[type="submit"]')
                except Exception as e2:
                    logger.warning(f"Could not click button with alternate selector, trying another: {e2}")
                    # Try a more generic selector
                    await self.page.click('button[type="submit"]:not([data-location="Menu"])')
            
            # Wait for login to complete - wait for URL change or navbar to appear
            logger.info("Waiting for successful login...")
            await self.page.wait_for_url(lambda url: "login" not in url, timeout=30000)
            
            self.logged_in = True
            logger.info("Successfully logged in to UTR")
            return True
            
        except PlaywrightTimeoutError as e:
            logger.error(f"Login failed: Timeout waiting for page to load: {e}")
            await self.page.screenshot(path="login_error.png")
            
            # Log current URL and page content for debugging
            logger.error(f"Current URL: {self.page.url}")
            with open("login_error_page.html", "w") as f:
                f.write(await self.page.content())
                
            return False
        except Exception as e:
            logger.error(f"Login failed: {e}")
            await self.page.screenshot(path="login_error.png")
            return False
    
    def handle_superscript(self, raw_score):
        """Convert superscript digits to standard tiebreak notation."""
        if not raw_score:
            return raw_score
            
        # Map for superscript characters
        SUPERSCRIPT_MAP = {
            '⁰': '0', '¹': '1', '²': '2', '³': '3', '⁴': '4',
            '⁵': '5', '⁶': '6', '⁷': '7', '⁸': '8', '⁹': '9'
        }
        
        # Get the main number first (before any superscript)
        main_num = re.search(r'^(\d+)', raw_score)
        main_digit = main_num.group(1) if main_num else ""
        
        # Find any superscript digit
        for sup, digit in SUPERSCRIPT_MAP.items():
            if sup in raw_score:
                return f"{main_digit}({digit})"
        
        # No superscript found, return the original number
        return main_digit if main_digit else raw_score

    async def search_player(self, name, wait_time=3):
        """
        Search for a player by name
        
        Parameters:
        name (str): Player name to search for
        wait_time (int): Seconds to wait for search results
        
        Returns:
        list: List of player dictionaries with id, name, utr
        """
        try:
            logger.info(f"Searching for player: {name}")
            
            # Fix the domain from .com to .net
            await self.page.goto("https://app.utrsports.net/search")
            
            # Click on the search input and fill
            await self.page.wait_for_selector('input[type="search"]', state="visible")
            await self.page.fill('input[type="search"]', name)
            
            # Wait for search results
            await asyncio.sleep(wait_time)
            
            # Extract player data from search results
            players = []
            player_elements = await self.page.query_selector_all('a[href*="/profiles/"]')
            
            for element in player_elements:
                try:
                    player_name_elem = await element.query_selector("div > div:nth-child(1)")
                    player_name = await player_name_elem.inner_text() if player_name_elem else ""
                    player_name = player_name.strip()
                    
                    player_utr_elem = await element.query_selector("div > div:nth-child(2)")
                    player_utr = await player_utr_elem.inner_text() if player_utr_elem else ""
                    player_utr = player_utr.strip()
                    
                    href = await element.get_attribute("href")
                    player_id = re.search(r"/profiles/(\d+)", href).group(1) if href else None
                    
                    # Extract just the UTR rating number
                    utr_value = re.search(r"([\d.]+)", player_utr)
                    utr_rating = float(utr_value.group(1)) if utr_value else None
                    
                    players.append({
                        "id": player_id,
                        "name": player_name,
                        "utr": utr_rating,
                        "url": f"https://app.utrsports.net/profiles/{player_id}"
                    })
                except Exception as e:
                    logger.error(f"Error extracting player data: {e}")
                    continue
            
            logger.info(f"Found {len(players)} matching players")
            return players
            
        except Exception as e:
            logger.error(f"Error searching for player: {e}")
            await self.page.screenshot(path=f"search_error_{name.replace(' ', '_')}.png")
            return []
    
    async def get_player_profile(self, player_id):
        """
        Get a player's profile information
        
        Parameters:
        player_id (str): Player's UTR ID
        
        Returns:
        dict: Player's profile information
        """
        logger.info(f"Getting profile for player ID: {player_id}")
        
        try:
            # Go directly to the profile page
            profile_url = f"https://app.utrsports.net/profiles/{player_id}"
            try:
                await self.page.goto(profile_url, timeout=20000)  # Reduced timeout
            except PlaywrightTimeoutError as e:
                logger.error(f"Timeout getting profile for player {player_id}: {e}")
                # Try to take a screenshot of the error
                try:
                    await self.page.screenshot(path=f"profileerror{player_id}.png")
                except Exception as screenshot_error:
                    logger.warning(f"Additionally failed to take error screenshot: {screenshot_error}")
                return None
            
            # Wait for the page to load with reduced timeout
            try:
                await self.page.wait_for_selector('h1', state="visible", timeout=20000)
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout waiting for profile page content for player {player_id}")
                await asyncio.sleep(3)  # Extra wait for content to load
            
            # Call our helper function to select singles
            await self.select_verified_singles()
            
            # Get basic player info
            try:
                player_name_elem = await self.page.query_selector('h1')
                player_name = await player_name_elem.inner_text() if player_name_elem else f"Unknown Player {player_id}"
                player_name = player_name.strip()
                
                # Look for UTR using multiple selectors - more precise and targeting exact elements
                current_utr = None
                utr_selectors = [
                    'div[class*="ratingDisplayV5__value"] [title]',  # Target the title attribute which has the exact UTR
                    'div[class*="value"][title]',
                    'div[title*="."]', # UTR values have decimal points
                    'div[class*="value"][noselect]',
                    'div[class*="ratingDisplayV5__value"]'
                ]
                
                for selector in utr_selectors:
                    utr_elems = await self.page.query_selector_all(selector)
                    for utr_elem in utr_elems:
                        # Try to get from title attribute first
                        utr_value = await utr_elem.get_attribute('title')
                        if not utr_value:
                            utr_value = await utr_elem.inner_text()
                        utr_value = utr_value.strip()
                        
                        if utr_value and "." in utr_value:  # Make sure it looks like a UTR
                            try:
                                current_utr = float(re.search(r'(\d+\.\d+)', utr_value).group(1))
                                logger.info(f"Found UTR value: {current_utr}")
                                break
                            except (ValueError, AttributeError):
                                pass
                    
                    if current_utr:
                        break
                
                # Use JavaScript to find the UTR as a fallback
                if not current_utr:
                    js_utr = """
                    () => {
                        // Look for elements with numeric content containing a decimal point (likely UTR ratings)
                        const elements = Array.from(document.querySelectorAll('div'));
                        for (const el of elements) {
                            if (el.textContent && el.textContent.match(/\\d+\\.\\d+/) && 
                                (el.getAttribute('title') || el.textContent.length < 10)) {
                                // Looks like a UTR value
                                const match = el.textContent.match(/(\\d+\\.\\d+)/);
                                return match ? match[1] : null;
                            }
                        }
                        return null;
                    }
                    """
                    try:
                        utr_result = await self.page.evaluate(js_utr)
                        if utr_result:
                            current_utr = float(utr_result)
                            logger.info(f"Found UTR using JavaScript: {current_utr}")
                    except Exception as js_error:
                        logger.warning(f"JavaScript UTR extraction failed: {js_error}")
                
                # Check if player is unrated
                if not current_utr:
                    logger.info(f"Player {player_name} appears to be unrated")
                    current_utr = "UR"
                
                # Look for reliability percentage - more precise selectors
                reliability = None
                reliability_date = None
                rating_outdated = False
                reliability_selectors = [
                    'div[class*="footer"] div:has-text("reliable")',
                    'div:has-text("% reliable")',
                    'div:has-text("reliable")',
                    'div:has-text("From ")',
                    'div[class*="ratingDisplayV5__footer"]'
                ]
                
                for selector in reliability_selectors:
                    reliability_elems = await self.page.query_selector_all(selector)
                    for rel_elem in reliability_elems:
                        reliability_text = await rel_elem.inner_text()
                        reliability_text = reliability_text.strip()
                        
                        # Check for percentage
                        reliability_match = re.search(r'(\d+)%', reliability_text)
                        if reliability_match:
                            reliability = int(reliability_match.group(1))
                            logger.info(f"Found reliability: {reliability}%")
                            break
                        
                        # Check for "From Month Year" format
                        from_match = re.search(r'From\s+([A-Za-z]+\s+\d{4})', reliability_text)
                        if from_match:
                            reliability_date = from_match.group(1)
                            reliability = 0  # Set to 0% for outdated ratings
                            rating_outdated = True
                            logger.info(f"Found outdated reliability: From {reliability_date}")
                            break
                        
                        # Check for "100% reliable" format
                        if "100% reliable" in reliability_text:
                            reliability = 100
                            logger.info("Found 100% reliability")
                            break
                    
                    if reliability is not None:
                        break
                
                # JavaScript fallback for reliability
                if reliability is None:
                    js_reliability = """
                    () => {
                        // Look for reliability percentage
                        const elements = Array.from(document.querySelectorAll('div'));
                        for (const el of elements) {
                            if (el.textContent && el.textContent.includes('reliable')) {
                                const match = el.textContent.match(/(\\d+)%/);
                                return match ? parseInt(match[1]) : null;
                            }
                            if (el.textContent && el.textContent.includes('From ')) {
                                return el.textContent.trim();
                            }
                        }
                        return null;
                    }
                    """
                    try:
                        rel_result = await self.page.evaluate(js_reliability)
                        if rel_result and isinstance(rel_result, int):
                            reliability = rel_result
                        elif rel_result and isinstance(rel_result, str) and "From " in rel_result:
                            from_match = re.search(r'From\s+([A-Za-z]+\s+\d{4})', rel_result)
                            if from_match:
                                reliability_date = from_match.group(1)
                                reliability = 0  # Set to 0% for outdated ratings
                                rating_outdated = True
                                logger.info(f"Found outdated reliability (JS): From {reliability_date}")
                        elif rel_result:
                            reliability = rel_result
                    except Exception as js_error:
                        logger.warning(f"JavaScript reliability extraction failed: {js_error}")
                
                # Basic info
                profile_data = {
                    "id": player_id,
                    "name": player_name,
                    "current_utr": current_utr,
                    "reliability": reliability,
                    "profile_url": profile_url,
                    "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                
                # Add outdated rating info if relevant
                if rating_outdated and reliability_date:
                    profile_data["reliability_date"] = reliability_date
                    profile_data["rating_outdated"] = True
                
                # Save profile data
                profile_file = self.players_dir / f"player_{player_id}_profile.json"
                with open(profile_file, 'w') as f:
                    # Convert NumPy types to Python native types
                    json_safe_data = {}
                    for k, v in profile_data.items():
                        if isinstance(v, np.integer):
                            json_safe_data[k] = int(v)
                        elif isinstance(v, np.floating):
                            json_safe_data[k] = float(v)
                        else:
                            json_safe_data[k] = v
                    
                    json.dump(json_safe_data, f, indent=2)
                
                logger.info(f"Saved profile for player {player_name} ({player_id})")
                return profile_data
            
            except Exception as e:
                logger.error(f"Error getting profile for player {player_id}: {e}")
                return None
        
        except Exception as e:
            logger.error(f"Error getting profile for player {player_id}: {e}")
            return None
    
    async def get_player_rating_history(self, player_id, max_retries=3):
        """
        Get a player's UTR rating history using direct URL navigation
        
        Parameters:
        player_id (str): Player's UTR ID
        max_retries (int): Maximum number of retry attempts
        
        Returns:
        DataFrame: Rating history with dates and UTR values
        """
        for retry in range(max_retries):
            try:
                # Check if we already have this player's rating history
                rating_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
                if rating_file.exists() and os.path.getsize(rating_file) > 0:
                    logger.info(f"Already have rating history for player {player_id}, loading from file")
                    try:
                        ratings_df = pd.read_csv(rating_file)
                        if not ratings_df.empty and 'date' in ratings_df.columns and 'utr' in ratings_df.columns:
                            logger.info(f"Successfully loaded {len(ratings_df)} rating entries from file")
                            return ratings_df
                        else:
                            logger.warning(f"Rating file for player {player_id} exists but has invalid structure")
                    except Exception as e:
                        logger.warning(f"Error loading existing rating file: {e}")
                        # Continue with fetching new data
                
                logger.info(f"Getting rating history for player ID: {player_id} (attempt {retry+1}/{max_retries})")
                
                # Go directly to player stats page using the URL parameter
                stats_url = f"https://app.utrsports.net/profiles/{player_id}?t=6"
                
                # Add timeout handling here - prevent the entire script from failing
                try:
                    await self.page.goto(stats_url, timeout=30000)  # Increased timeout for better reliability
                except PlaywrightTimeoutError as e:
                    logger.warning(f"Timeout navigating to stats page for player {player_id} (attempt {retry+1}): {e}")
                    # If not the last retry, try again
                    if retry < max_retries - 1:
                        logger.info(f"Retrying after timeout ({retry+1}/{max_retries})...")
                        await asyncio.sleep(random.uniform(3.0, 6.0))  # Longer delay after timeouts
                        continue
                    else:
                        # Create an empty DataFrame and save it to avoid retrying this player
                        empty_df = pd.DataFrame(columns=['date', 'utr'])
                        empty_df.to_csv(rating_file, index=False)
                        logger.info(f"Created empty ratings file for player {player_id} after max retries")
                        return empty_df
                
                # Wait for the page to load - using a more generic selector with sufficient timeout
                try:
                    await self.page.wait_for_selector('h1, .header, [class*="header"]', state="visible", timeout=20000)
                except PlaywrightTimeoutError:
                    logger.warning(f"Timeout waiting for page content for player {player_id} (attempt {retry+1})")
                    # Try to proceed anyway, in case some content is available
                
                await asyncio.sleep(4)  # Increased wait to ensure page is fully loaded
                
                # Take a screenshot for debugging - with error handling
                try:
                    await self.page.screenshot(path=f"stats_page_{player_id}.png")
                except Exception as e:
                    logger.warning(f"Failed to take screenshot: {e}")
                
                # Use our helper function to select singles
                await self.select_verified_singles()
                
                # Check for premium content blocker with multiple selectors
                premium_blocker = await self.page.query_selector('div:has-text("Subscribe to UTR Pro"), div:has-text("Upgrade to UTR Pro")')
                if premium_blocker:
                    logger.warning(f"Rating history for player {player_id} requires premium subscription")
                    
                    # If not the last retry, try the JavaScript extraction approach directly
                    if retry < max_retries - 1:
                        logger.info(f"Attempting JavaScript extraction on retry {retry+1}")
                        
                        # Try to extract UTR history from the raw HTML
                        html_content = await self.page.content()
                        
                        # Look for patterns like "date":"2023-05-15","rating":15.23
                        date_rating_matches = re.findall(r'"date":"(\d{4}-\d{2}-\d{2})","rating":([\d.]+)', html_content)
                        
                        if date_rating_matches:
                            logger.info(f"Extracted {len(date_rating_matches)} ratings from HTML despite premium block")
                            
                            rating_history = [
                                {"date": date, "utr": float(rating)}
                                for date, rating in date_rating_matches
                            ]
                            
                            df = pd.DataFrame(rating_history)
                            csv_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
                            df.to_csv(csv_file, index=False)
                            
                            logger.info(f"Saved rating history (HTML extraction) for player {player_id}: {len(rating_history)} entries")
                            return df
                        
                        # If HTML extraction failed, retry with next attempt
                        logger.info(f"JavaScript extraction failed, continuing to retry {retry+2}")
                        await asyncio.sleep(random.uniform(3.0, 5.0))
                        continue
                    else:
                        # Final attempt failed - create an empty DataFrame with a note
                        empty_df = pd.DataFrame(columns=['date', 'utr', 'note'])
                        empty_df.loc[0] = [datetime.now().strftime("%Y-%m-%d"), None, "Premium content"]
                        empty_df.to_csv(rating_file, index=False)
                        return empty_df

                # Check for "Show All" button for full rating history and click it if present
                try:
                    # Try different selector patterns for the "Show all" button
                    show_all_selectors = [
                        'a:has-text("Show all")',
                        'a.underline:has-text("Show all")',
                        'button:has-text("Show all")',
                        'div:has-text("Show all")'
                    ]
                    
                    for selector in show_all_selectors:
                        show_all = await self.page.query_selector(selector)
                        if show_all:
                            logger.info(f"Found 'Show all' button using selector: {selector}")
                            await show_all.click()
                            # Wait for additional history to load
                            await asyncio.sleep(4)
                            # Take another screenshot after clicking Show All
                            try:
                                await self.page.screenshot(path=f"stats_page_after_show_all_{player_id}.png")
                            except Exception as e:
                                logger.warning(f"Failed to take screenshot after Show All: {e}")
                            break
                            
                except Exception as e:
                    logger.warning(f"Could not find or click 'Show all' button: {e}")
                
                # Try multiple different selectors for rating history items
                history_selectors = [
                    '.newStatsTabContent__historyItem__INNPC',  # Original selector
                    'div[class*="historyItem"]',     # More generic selector targeting class containing "historyItem"
                    'div.history-item',              # Another possible class name
                    '.rating-history-item',          # Another possible class name
                    '[class*="rating-history"] div', # Any div inside rating history container
                    'div[class*="history"]'          # Any div with "history" in class name
                ]
                    
                history_items = []
                for selector in history_selectors:
                    items = await self.page.query_selector_all(selector)
                    if items and len(items) > 0:
                        logger.info(f"Found {len(items)} rating history items using selector: {selector}")
                        history_items = items
                        break
                    
                # Try a more brute force approach if the selectors don't work
                if not history_items:
                    logger.info("Trying alternative approach to extract rating history")
                    # Get all date-like text on the page
                    date_elements = await self.page.query_selector_all('div:text-matches("\\d{4}-\\d{2}-\\d{2}")')
                    if date_elements:
                        logger.info(f"Found {len(date_elements)} date elements on the page")
                        
                        # Extract dates and try to find nearby rating values
                        rating_history = []
                        for date_elem in date_elements:
                            try:
                                date_str = await date_elem.inner_text()
                                date_str = date_str.strip()
                                # Look for a nearby number that could be a UTR rating
                                parent = await date_elem.evaluate_handle('node => node.parentElement')
                                parent_text = await parent.inner_text()
                                
                                # Use regex to find a number like "15.23" in the parent element's text
                                utr_match = re.search(r'([\d.]{2,5})', parent_text)
                                if utr_match and date_str:
                                    rating_value = float(utr_match.group(1))
                                    rating_history.append({
                                        "date": date_str,
                                        "utr": rating_value
                                    })
                            except Exception as e:
                                logger.error(f"Error extracting alternative rating data: {e}")
                        
                        if rating_history:
                            df = pd.DataFrame(rating_history)
                            csv_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
                            df.to_csv(csv_file, index=False)
                            logger.info(f"Saved rating history (alternative method) for player {player_id}: {len(rating_history)} entries")
                            return df
                        
                # Process standard history items if found
                rating_history = []
                
                for item in history_items:
                    try:
                        # Try different selectors for date and rating within each history item
                        date_selectors = [
                            '.newStatsTabContent__historyItemDate__JFjy',
                            'div[class*="Date"]',
                            'div[class*="date"]',
                            'span[class*="date"]'
                        ]
                        
                        rating_selectors = [
                            '.newStatsTabContent__historyItemRating__GQUXX',
                            'div[class*="Rating"]',
                            'div[class*="rating"]',
                            'span[class*="rating"]'
                        ]
                        
                        # Try to find date using selectors
                        date_str = None
                        for date_selector in date_selectors:
                            date_element = await item.query_selector(date_selector)
                            if date_element:
                                date_str = await date_element.inner_text()
                                date_str = date_str.strip()
                                break
                        
                        # If date not found, try to extract it directly from the item text
                        if not date_str:
                            item_text = await item.inner_text()
                            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', item_text)
                            if date_match:
                                date_str = date_match.group(1)
                        
                        # Try to find rating using selectors
                        rating_str = None
                        for rating_selector in rating_selectors:
                            rating_element = await item.query_selector(rating_selector)
                            if rating_element:
                                rating_str = await rating_element.inner_text()
                                rating_str = rating_str.strip()
                                break
                        
                        # If rating not found, try to extract it directly from the item text
                        if not rating_str and date_str:
                            item_text = await item.inner_text()
                            # Look for a number after the date
                            rating_match = re.search(r'([\d.]{2,5})', item_text.replace(date_str, '', 1))
                            if rating_match:
                                rating_str = rating_match.group(1)
                        
                        if date_str and rating_str:
                            # Parse date and rating
                            try:
                                parsed_date = datetime.strptime(date_str, "%Y-%m-%d").strftime("%Y-%m-%d")
                                # Extract just the UTR rating number using regex
                                rating_match = re.search(r'([\d.]+)', rating_str)
                                if rating_match:
                                    rating_value = float(rating_match.group(1))
                                    rating_history.append({
                                        "date": parsed_date,
                                        "utr": rating_value
                                    })
                            except Exception as e:
                                logger.error(f"Error parsing date or rating: {e}")
                                
                    except Exception as e:
                        logger.error(f"Error extracting rating history item: {e}")
                        continue
                
                # Create DataFrame and save
                if rating_history:
                    df = pd.DataFrame(rating_history)
                    
                    # Save as CSV
                    csv_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
                    df.to_csv(csv_file, index=False)
                    
                    logger.info(f"Saved rating history for player {player_id}: {len(rating_history)} entries")
                    return df
                else:
                    # One last attempt - try to extract UTR history from the raw HTML
                    html_content = await self.page.content()
                    
                    # Look for patterns like "date":"2023-05-15","rating":15.23
                    date_rating_matches = re.findall(r'"date":"(\d{4}-\d{2}-\d{2})","rating":([\d.]+)', html_content)
                    
                    if date_rating_matches:
                        logger.info(f"Extracted {len(date_rating_matches)} ratings from HTML")
                        
                        rating_history = [
                            {"date": date, "utr": float(rating)}
                            for date, rating in date_rating_matches
                        ]
                        
                        df = pd.DataFrame(rating_history)
                        csv_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
                        df.to_csv(csv_file, index=False)
                        
                        logger.info(f"Saved rating history (HTML extraction) for player {player_id}: {len(rating_history)} entries")
                        return df
                    
                    # If this isn't the last retry, try again
                    if retry < max_retries - 1:
                        logger.warning(f"No rating history found for player {player_id} on attempt {retry+1}, retrying...")
                        await asyncio.sleep(random.uniform(3.0, 6.0))
                        continue
                    else:
                        # Create empty file to avoid retrying this player
                        logger.warning(f"No rating history found for player {player_id} after {max_retries} attempts")
                        empty_df = pd.DataFrame(columns=['date', 'utr'])
                        empty_df.to_csv(rating_file, index=False)
                        return empty_df
                        
            except Exception as e:
                logger.error(f"Error getting rating history for player {player_id} (attempt {retry+1}): {e}")
                try:
                    # Added proper error handling around the screenshot operation
                    await self.page.screenshot(path=f"rating_history_error_{player_id}_{retry+1}.png")
                except Exception as screenshot_error:
                    logger.error(f"Additionally failed to take error screenshot: {screenshot_error}")
                
                # If not the last retry, try again after resetting browser
                if retry < max_retries - 1:
                    logger.info(f"Retrying after error ({retry+1}/{max_retries})...")
                    await asyncio.sleep(random.uniform(3.0, 6.0))
                    
                    # Restart browser on major errors
                    try:
                        await self.close_browser()
                        await asyncio.sleep(2)
                        await self.start_browser()
                        await self.login()
                    except Exception as browser_error:
                        logger.error(f"Error restarting browser: {browser_error}")
                    
                    continue
                else:
                    # Create an empty file to prevent retrying this player
                    empty_df = pd.DataFrame(columns=['date', 'utr'])
                    empty_df.to_csv(self.ratings_dir / f"player_{player_id}_ratings.csv", index=False)
                    
                    return pd.DataFrame()
        
    async def select_year_from_dropdown(self, year):
        """
        Select a specific year from the dropdown
        
        Parameters:
        year (str): Year to select (e.g. "2025")
        
        Returns:
        str or bool: "year_not_found" if year not present, True if successful, False if failed for other reasons
        """
        try:
            logger.info(f"Attempting to select year {year} from dropdown")
            
            # Take a screenshot before clicking anything
            await self.page.screenshot(path="before_dropdown.png")
            
            # Look specifically for the LATEST button using more precise selectors
            # First try to find the LATEST text element
            latest_selectors = [
                'h6.text-uppercase.title:has-text("LATEST")',
                '.popovermenu-anchor:has-text("LATEST")',
                'span:has-text("LATEST")',
                'div.popovermenu-anchor span.text-uppercase.title'
            ]
            
            latest_button = None
            for selector in latest_selectors:
                latest_button = await self.page.query_selector(selector)
                if latest_button:
                    logger.info(f"Found LATEST button using selector: {selector}")
                    break
                    
            if not latest_button:
                # Try finding any element with "LATEST" text
                logger.warning("Could not find LATEST button with specific selectors, trying generic approach")
                latest_js = """
                () => {
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        if (el.textContent && el.textContent.trim() === 'LATEST') {
                            return el;
                        }
                    }
                    return null;
                }
                """
                latest_button = await self.page.evaluate_handle(latest_js)
                
            if not latest_button:
                logger.warning("Could not find LATEST dropdown button")
                return False
                
            # Click the LATEST button
            await latest_button.click()
            logger.info("Clicked LATEST dropdown button")
            await asyncio.sleep(2)  # Wait for dropdown to appear
            
            # Take a screenshot after dropdown should be visible
            await self.page.screenshot(path="dropdown_opened.png")
            
            # Try to locate the year item
            year_selectors = [
                f'div.menu-item:has-text("{year}")',
                f'div[id="item{year}"]',
                f'#item{year}',
                f'div.popovermenu-list div:has-text("{year}")'
            ]
            
            year_item = None
            for selector in year_selectors:
                try:
                    year_item = await self.page.query_selector(selector)
                    if year_item:
                        logger.info(f"Found year {year} using selector: {selector}")
                        break
                except Exception as e:
                    logger.warning(f"Error with selector {selector}: {e}")
            
            if not year_item:
                # Use JavaScript to find the year element more flexibly
                year_js = f"""
                () => {{
                    // Look through all divs for one with exactly the year text
                    const allDivs = Array.from(document.querySelectorAll('div'));
                    for (const div of allDivs) {{
                        if (div.textContent && div.textContent.trim() === '{year}') {{
                            return div;
                        }}
                    }}
                    
                    // Check for items with an ID containing the year
                    const yearItems = document.querySelectorAll(`div[id*="{year}"]`);
                    if (yearItems.length > 0) return yearItems[0];
                    
                    // If nothing found, return null explicitly
                    return null;
                }}
                """
                try:
                    year_item = await self.page.evaluate_handle(year_js)
                    # Check if the JavaScript returned a non-null value that's actually an element
                    if not year_item.as_element():
                        logger.info(f"Year {year} not found in dropdown - player likely has no matches for this year")
                        return "year_not_found"
                except Exception as e:
                    logger.warning(f"JavaScript evaluation error: {e}")
                    # If JavaScript evaluation fails, assume year not found
                    return "year_not_found"
            
            if not year_item:
                logger.info(f"Year {year} not found in dropdown - player likely has no matches for this year")
                return "year_not_found"
                
            # Click the year
            await year_item.click()
            logger.info(f"Clicked year {year}")
            
            # Wait for new data to load
            await asyncio.sleep(5)
            
            # Take screenshot after selection
            await self.page.screenshot(path=f"after_{year}_selection.png")
            
            return True
            
        except Exception as e:
            logger.error(f"Error selecting year: {e}")
            await self.page.screenshot(path=f"year_selection_error_{year}.png")
            return False
        
    async def select_singles_from_dropdown(self):
        """
        Select singles from dropdown menu
        
        Returns:
        bool: Whether singles was successfully selected
        """
        try:
            logger.info("Attempting to select Singles from dropdown")
            
            # Check if match cards are already visible - indicates we're on the correct view
            match_cards = await self.page.query_selector_all('.utr-card.score-card, div[class*="eventItem"], div[class*="scorecard"]')
            if match_cards and len(match_cards) > 0:
                logger.info("Match cards already visible, already on Singles view")
                return True
                
            # First check if we're already viewing singles by text
            current_view = await self.page.query_selector('h6.text-uppercase.title:has-text("SINGLES")')
            if current_view:
                # If we found the Singles title and there's content, we're likely already there
                logger.info("Already on Singles view, skipping navigation")
                return True
            
            # Take a screenshot before clicking anything
            try:
                await self.page.screenshot(path="before_singles_dropdown.png")
            except Exception as e:
                logger.warning(f"Failed to take before screenshot: {e}")
            
            # First, look for the singles/doubles button (which should show SINGLES)
            singles_button_selectors = [
                'h6.text-uppercase.title:has-text("SINGLES")',
                'h6:has-text("SINGLES")',
                'div.popovermenu-anchor:has-text("SINGLES")',
                'span:has-text("SINGLES")',
                '.popovermenu-anchor h6.title'
            ]
            
            singles_button = None
            for selector in singles_button_selectors:
                singles_button = await self.page.query_selector(selector)
                if singles_button:
                    logger.info(f"Found SINGLES button using selector: {selector}")
                    break
                    
            if not singles_button:
                # Try finding any element with "SINGLES" text
                logger.warning("Could not find SINGLES button with specific selectors, trying generic approach")
                singles_js = """
                () => {
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        if (el.textContent && (el.textContent.trim() === 'SINGLES' || 
                                            el.textContent.includes('SINGLES ▾'))) {
                            return el;
                        }
                    }
                    return null;
                }
                """
                singles_button = await self.page.evaluate_handle(singles_js)
                    
            if not singles_button:
                logger.warning("Could not find SINGLES dropdown button")
                return False
                    
            # Click the SINGLES button
            await singles_button.click()
            logger.info("Clicked SINGLES dropdown button")
            await asyncio.sleep(2)  # Wait for dropdown to appear
            
            # Take a screenshot after dropdown should be visible
            try:
                await self.page.screenshot(path="singles_dropdown_opened.png")
            except Exception as e:
                logger.warning(f"Failed to take dropdown screenshot: {e}")
            
            # Try to locate the Singles item
            singles_selectors = [
                'div.menu-item:has-text("Singles")',
                'div#itemsingles',
                '#itemsingles',
                'div.popovermenu-list div:has-text("Singles")',
                'div:has-text("Singles"):not(h6)'
            ]
            
            singles_item = None
            for selector in singles_selectors:
                try:
                    singles_item = await self.page.query_selector(selector)
                    if singles_item:
                        logger.info(f"Found Singles using selector: {selector}")
                        break
                except Exception as e:
                    logger.warning(f"Error with selector {selector}: {e}")
            
            if not singles_item:
                # Use JavaScript to find the Singles element more flexibly
                singles_js = """
                () => {
                    // Look through all divs for one with exactly the Singles text
                    const allDivs = Array.from(document.querySelectorAll('div'));
                    for (const div of allDivs) {
                        if (div.textContent && div.textContent.trim() === 'Singles') {
                            return div;
                        }
                    }
                    
                    // Check for menu items
                    const menuItems = document.querySelectorAll('div.menu-item');
                    for (const item of menuItems) {
                        if (item.textContent && item.textContent.includes('Singles')) {
                            return item;
                        }
                    }
                    
                    return null;
                }
                """
                singles_item = await self.page.evaluate_handle(singles_js)
            
            if not singles_item:
                logger.warning(f"Could not find Singles option in dropdown")
                return False
                    
            # Click the Singles option
            await singles_item.click()
            logger.info(f"Clicked Singles option")
            
            # Wait for new data to load
            await asyncio.sleep(3)
            
            # Take screenshot after selection
            try:
                await self.page.screenshot(path=f"after_singles_selection.png")
            except Exception as e:
                logger.warning(f"Failed to take after screenshot: {e}")
            
            return True
                
        except Exception as e:
            logger.error(f"Error selecting Singles: {e}")
            try:
                await self.page.screenshot(path=f"singles_selection_error.png")
            except Exception as screenshot_error:
                logger.error(f"Also failed to take error screenshot: {screenshot_error}")
            return False

    async def select_verified_singles(self):
        """
        Select verified singles on profile/stats pages
        
        Returns:
        bool: Whether verified singles was successfully selected
        """
        try:
            logger.info("Attempting to select Verified singles")
            
            # First check - maybe we're already on singles view
            # Check if singles UTR is visible already
            utr_visible = await self.page.query_selector('div[class*="singles-utr"], div[title*="16.41"], div[class*="verified14"]')
            if utr_visible:
                logger.info("Already on Verified Singles view")
                return True
                
            # Take a screenshot before clicking
            try:
                await self.page.screenshot(path="before_verified_singles.png")
            except Exception as e:
                logger.warning(f"Failed to take before screenshot: {e}")
            
            # Look for verified singles elements - more comprehensive selectors
            verified_singles_selectors = [
                'div:text("Verified Singles")',
                'div:has-text("Verified singles")',
                'div:has-text("Verified Singles")',
                'div.singles',
                'div[class*="singles"]',
                '[class*="verified14"]',
                'button:has-text("Singles")',
                '#singlesTab',
                'div.singles-tab'
            ]
            
            verified_singles_elem = None
            for selector in verified_singles_selectors:
                verified_singles_elem = await self.page.query_selector(selector)
                if verified_singles_elem:
                    logger.info(f"Found Verified singles using selector: {selector}")
                    break
                    
            if not verified_singles_elem:
                # Try JavaScript approach to find any singles-related element
                verified_singles_js = """
                () => {
                    // Look for any element with singles text
                    const elements = Array.from(document.querySelectorAll('*'));
                    for (const el of elements) {
                        const text = el.textContent || '';
                        if (text.includes('Verified Singles') || 
                            text.includes('Verified singles') || 
                            text.includes('Singles') && !text.includes('Doubles')) {
                            return el;
                        }
                    }
                    
                    // Look for elements with singles-related class names
                    const classElements = document.querySelectorAll('[class*="singles"], [class*="verified"]');
                    if (classElements.length > 0) return classElements[0];
                    
                    return null;
                }
                """
                try:
                    verified_singles_elem = await self.page.evaluate_handle(verified_singles_js)
                except Exception as js_error:
                    logger.warning(f"JavaScript approach failed: {js_error}")
            
            if not verified_singles_elem:
                # Maybe we're already on singles view, skip the click
                logger.info("Could not find Verified singles element, assuming already on singles view")
                return True
            
            # Click the Verified singles element
            try:
                await verified_singles_elem.click()
                logger.info("Selected Singles view")
                await asyncio.sleep(2)
            except Exception as click_error:
                logger.warning(f"Error clicking singles element: {click_error}")
                # Check if we're already on singles view
                if await self.page.query_selector('div[class*="singles-utr"], div[class*="verified14"]'):
                    logger.info("Already on Verified Singles view despite click error")
                    return True
                return False
            
            # Take screenshot after selection
            try:
                await self.page.screenshot(path="after_verified_singles.png")
            except Exception as e:
                logger.warning(f"Failed to take after screenshot: {e}")
                
            return True
                
        except Exception as e:
            logger.error(f"Error selecting Verified singles: {e}")
            try:
                await self.page.screenshot(path="verified_singles_error.png")
            except Exception as screenshot_error:
                logger.error(f"Also failed to take error screenshot: {screenshot_error}")
            return False
    
    async def get_player_match_history(self, player_id, year=None, limit=200):
        """
        Get a player's match history using direct URL navigation
        
        Parameters:
        player_id (str): Player's UTR ID
        year (str): Optional specific year to fetch (e.g., "2025")
        limit (int): Maximum number of matches to retrieve
        
        Returns:
        DataFrame: Match history with results and opponent info
        """
        try:
            logger.info(f"Getting match history for player ID: {player_id}")
            
            # Go directly to the results tab with ?t=2 parameter
            profile_url = f"https://app.utrsports.net/profiles/{player_id}?t=2"
            
            try:
                await self.page.goto(profile_url, timeout=20000)  # Reduced timeout
            except PlaywrightTimeoutError as e:
                logger.warning(f"Timeout navigating to match history page for player {player_id}: {e}")
                return pd.DataFrame()
            
            # Wait for the page to load with reduced timeout
            try:
                await self.page.wait_for_selector('h1', state="visible", timeout=15000)  # Reduced timeout
            except PlaywrightTimeoutError:
                logger.warning(f"Timeout waiting for page content for player {player_id}")
                # Try to proceed anyway
            
            await asyncio.sleep(3)  # Extra wait for content to load
            
            # Take a screenshot for debugging
            try:
                await self.page.screenshot(path=f"match_history_page_{player_id}.png")
            except Exception as e:
                logger.warning(f"Failed to take screenshot: {e}")
                
            # Ensure we're looking at singles matches using our dropdown helper
            await self.select_singles_from_dropdown()
            
            # If a specific year is requested, try to select it from the dropdown
            if year:
                logger.info(f"Selecting year: {year}")
                try:
                    year_selected = await self.select_year_from_dropdown(year)
                    if year_selected == "year_not_found":
                        # Create empty dataframe for this year and return it
                        empty_df = pd.DataFrame()
                        year_file = self.matches_dir / f"player_{player_id}_matches_{year}.csv"
                        empty_df.to_csv(year_file, index=False)
                        logger.info(f"No matches found for year {year} - year not in dropdown")
                        return empty_df
                    elif not year_selected:
                        # Check if any year dropdown is available at all
                        year_dropdown = await self.page.query_selector('div:has-text("LATEST")')
                        if not year_dropdown:
                            logger.warning(f"No year dropdown found, player may not have matches for any year")
                            return pd.DataFrame()  # Return empty if no dropdown at all
                        else:
                            logger.warning(f"Could not select year {year}, using default view")
                except Exception as e:
                    logger.error(f"Error selecting year {year}: {e}")
            
            # Find all match cards with more comprehensive selectors
            try:
                # Use a variety of selectors to find match cards
                match_card_selectors = [
                    '.utr-card.score-card',
                    '.scorecard__scorecard__3oNJK', 
                    '.eventItem__eventItem__2xpsd', 
                    'div[class*="eventItem"]',
                    'div[class*="match-card"]', 
                    'div[class*="scorecard"]',
                    'div[class*="event-card"]'
                ]
                
                match_cards = []
                for selector in match_card_selectors:
                    cards = await self.page.query_selector_all(selector)
                    if cards and len(cards) > 0:
                        match_cards = cards
                        logger.info(f"Found {len(cards)} match cards using selector: {selector}")
                        break
                
                if not match_cards:
                    # Try JavaScript approach to find match cards
                    js_cards = """
                    () => {
                        // Find div elements that might be match cards based on their content
                        const possibleCards = Array.from(document.querySelectorAll('div')).filter(div => {
                            const text = div.innerText;
                            // Look for content that suggests this is a match card (scores, vs, W/L)
                            return (text.includes(' vs ') || text.includes('W ') || text.includes('L ') || 
                                    text.includes('-') || text.includes('/'));
                        });
                        
                        // Try to identify actual match cards from candidates
                        return possibleCards.filter(div => {
                            // Match cards typically have multiple child divs
                            return div.children.length >= 3 && 
                                div.getBoundingClientRect().height > 50;
                        });
                    }
                    """
                    
                    cards_handle = await self.page.evaluate_handle(js_cards)
                    if cards_handle:
                        cards_array = await self.page.evaluate("cards => Array.from(cards)", cards_handle)
                        match_cards = cards_array
                        logger.info(f"Found {len(match_cards)} match cards using JavaScript approach")
                
                if not match_cards or len(match_cards) == 0:
                    logger.warning(f"No match cards found for player {player_id}")
                    # Create empty file to avoid retrying
                    empty_df = pd.DataFrame()
                    matches_file = self.matches_dir / f"player_{player_id}_matches.csv"
                    empty_df.to_csv(matches_file, index=False)
                    return pd.DataFrame()
                    
                logger.info(f"Found {len(match_cards)} match cards")
                
            except Exception as e:
                logger.error(f"Error finding match cards: {e}")
                return pd.DataFrame()
            
            # Get global page information that will apply to all matches
            player_name = None
            try:
                # Get the player name from the page header
                player_name_elem = await self.page.query_selector('h1')
                if player_name_elem:
                    player_name = await player_name_elem.inner_text()
                    player_name = player_name.strip()
            except Exception as e:
                logger.warning(f"Error getting player name from page: {e}")
            
            # Find all event headers (not match containers)
            event_headers = []
            try:
                # Use the correct selector for event headers based on the DOM
                event_header_selector = 'div.eventItem__eventHeaderContainer__3pg9m'
                headers = await self.page.query_selector_all(event_header_selector)
                if headers:
                    logger.info(f"Found {len(headers)} event headers using selector: {event_header_selector}")
                else:
                    logger.warning(f"No event headers found with selector: {event_header_selector}")

                if headers:
                    for idx, header in enumerate(headers):
                        try:
                            # Extract the main event name (e.g., "USTA JTT - Southern - North Carolina - District Championship 2016")
                            event_name_elem = await header.query_selector('div.eventItem__eventName__6hntZ > span')
                            event_name = await event_name_elem.inner_text() if event_name_elem else ""
                            event_name = event_name.strip()
                            if event_name:
                                logger.info(f"Found main event name for event header {idx+1}: {event_name}")
                            else:
                                logger.warning(f"No main event name found for event header {idx+1}, trying draw name")

                            # Extract the draw name (e.g., "LAKE SZILAGYI vs LAKE BIPPUS") for additional context or fallback
                            draw_name_elem = await header.query_selector('div.eventItem__drawName__29qpC')
                            draw_name = await draw_name_elem.inner_text() if draw_name_elem else ""
                            draw_name = draw_name.strip()
                            # Remove the middle dot character if present
                            draw_name = re.sub(r'^\s*•\s*', '', draw_name)
                            draw_name = draw_name.strip()
                            if draw_name:
                                logger.info(f"Found draw name for event header {idx+1}: {draw_name}")
                            else:
                                logger.warning(f"No draw name found for event header {idx+1}")

                            # If main event name is not available, fall back to the draw name
                            if not event_name and draw_name:
                                event_name = draw_name
                                logger.info(f"Using draw name as event name for event header {idx+1}: {event_name}")

                            # Handle empty or invalid event names
                            if not event_name or event_name.isspace():
                                event_name = "Unknown Tournament"
                                logger.info(f"Event header {idx+1} has no valid event or draw name - labeled as 'Unknown Tournament'")

                            # Clean the event name (remove only times and dates, preserve all other terms)
                            event_name = re.sub(r'\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)', '', event_name)  # Remove times
                            event_name = re.sub(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+\w+\s+\d{1,2}', '', event_name)  # Remove dates
                            event_name = event_name.strip()
                            logger.info(f"Processed event name: {event_name}")

                            # Get the position of this event header in the DOM
                            y_position = await self.page.evaluate("""
                                (element) => {
                                    const rect = element.getBoundingClientRect();
                                    return rect.top;
                                }
                            """, header)

                            # Store both the event name and draw name in the event_headers list
                            event_headers.append({
                                "name": event_name,
                                "draw": draw_name if draw_name else "Unknown Draw",  # Default to "Unknown Draw" if empty
                                "position": y_position,
                                "index": idx
                            })

                            logger.info(f"Event header {idx+1}: {event_name} (Draw: {draw_name if draw_name else 'Unknown Draw'}) at position {y_position}")
                        except Exception as e:
                            logger.warning(f"Error processing event header {idx+1}: {e}")

                    # Sort event headers by position
                    event_headers.sort(key=lambda x: x["position"])
                    logger.info(f"Total event headers found: {len(event_headers)}")
                else:
                    logger.warning("No event headers found, using fallback")
                    event_headers = [{"name": "Unknown Tournament", "draw": "Unknown Draw", "position": -float('inf'), "index": 0}]

            except Exception as e:
                logger.error(f"Error finding event headers: {e}")
                event_headers = [{"name": "Unknown Tournament", "draw": "Unknown Draw", "position": -float('inf'), "index": 0}]

            # Add a function to find the event header for a given position
            def get_event_for_position(y_position):
                """Find which event and draw a match at position y_position belongs to"""
                matching_event = "Unknown Tournament"
                matching_draw = "Unknown Draw"
                
                if not event_headers:
                    return matching_event, matching_draw
                
                # Find the last event header that's above this position
                best_match = None
                for header in event_headers:
                    header_pos = header["position"]
                    if header_pos <= y_position:
                        if best_match is None or header_pos > best_match["position"]:
                            best_match = header
                
                if best_match:
                    return best_match["name"], best_match["draw"]
                
                return matching_event, matching_draw
            
            matches = []
            
            # Process all matches
            for match_idx, card in enumerate(match_cards):
                try:
                    # Check if the match is retired, but don't skip it
                    retired = False
                    try:
                        match_text = await card.inner_text()
                        if "Retired" in match_text or "retired" in match_text.lower():
                            retired = True
                            logger.info(f"Match {match_idx+1} contains 'Retired' - marking as retired match")
                    except Exception as e:
                        logger.error(f"Error checking for retired status: {e}")

                    # Extract match details (date, round, tournament name) from the card header
                    header_text = ""
                    try:
                        # Try more specific selectors first to target the element that contains the time
                        header_selectors = [
                            'div[class*="scorecard_header_2iDdF"]',
                            '[class*="header_2iDdF"]',
                            'div[class*="header"] > div',
                            '.scorecard__header__2iDdF', 
                            '[class*="header"]',
                            'div.date'
                        ]
                        
                        for selector in header_selectors:
                            header_elem = await card.query_selector(selector)
                            if header_elem:
                                extracted_text = await header_elem.inner_text()
                                extracted_text = extracted_text.strip()
                                logger.info(f"Found header with selector '{selector}': '{extracted_text}'")
                                # If we found a header with pipe separators, use it and break
                                if '|' in extracted_text:
                                    header_text = extracted_text
                                    logger.info(f"Using header with pipe separators: '{header_text}'")
                                    break
                        
                        # If no header with pipe separator was found, try a JavaScript approach
                        if not header_text or '|' not in header_text:
                            try:
                                header_js = """
                                (card) => {
                                    const headers = Array.from(card.querySelectorAll('div[class*="header"]'));
                                    for (const header of headers) {
                                        const text = header.innerText;
                                        if (text.includes('|')) {
                                            return text.trim();
                                        }
                                    }
                                    
                                    // Try a more specific approach for the scorecard header
                                    const scorecard_header = card.querySelector('[class*="scorecard_header_2iDdF"]') || 
                                                            card.querySelector('[class*="header_2iDdF"]');
                                    if (scorecard_header) {
                                        return scorecard_header.innerText.trim();
                                    }
                                    
                                    return null;
                                }
                                """
                                js_header = await self.page.evaluate(header_js, card)
                                if js_header:
                                    header_text = js_header
                                    logger.info(f"Found header with JavaScript: '{header_text}'")
                            except Exception as e:
                                logger.warning(f"JavaScript header extraction failed: {e}")
                            
                            # If still no header text, use the original selector
                            if not header_text:
                                header_elem = await card.query_selector('.scorecard__header__2iDdF, [class*="header"], div.date')
                                if header_elem:
                                    header_text = await header_elem.inner_text()
                                    header_text = header_text.strip()
                                    logger.info(f"Using fallback header: '{header_text}'")
                    except Exception as e:
                        logger.warning(f"Error extracting header from match {match_idx+1}: {e}")

                    # Parse match time, date, and round from header
                    match_time = None
                    match_date = None
                    match_round = None

                    # If header contains pipe separators, parse it accordingly
                    if '|' in header_text:
                        # Format is typically "8:40 AM | Jan 26 | Final"
                        header_parts = header_text.split('|')
                        logger.info(f"Split header parts: {header_parts}")
                        
                        if len(header_parts) >= 1:
                            match_time = header_parts[0].strip()
                            logger.info(f"Extracted time from header part: {match_time}")
                        
                        if len(header_parts) >= 2:
                            date_part = header_parts[1].strip()
                            try:
                                year_val = year or datetime.now().year
                                date_obj = datetime.strptime(f"{date_part} {year_val}", "%b %d %Y")
                                match_date = date_obj.strftime("%Y-%m-%d")
                                logger.info(f"Parsed date: {match_date}")
                            except Exception as e:
                                logger.warning(f"Error parsing date '{date_part}': {e}")
                                match_date = date_part
                        
                        if len(header_parts) >= 3:
                            match_round = header_parts[2].strip()
                            logger.info(f"Extracted round: {match_round}")
                    else:
                        # No pipe separators, use simple regex approach
                        logger.info(f"No pipe separators in header, using regex: '{header_text}'")
                        
                        # Extract time if it exists (looking for patterns like "1:00 AM")
                        time_patterns = [
                            r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))',  # "1:00 AM"
                            r'(\d{1,2}[:\.]\d{2}(?:AM|PM|am|pm))',  # "1:00AM"
                            r'(\d{1,2}\s*(?:AM|PM|am|pm))'          # "1 AM"
                        ]
                        
                        for pattern in time_patterns:
                            time_match = re.search(pattern, header_text)
                            if time_match:
                                match_time = time_match.group(1).strip()
                                logger.info(f"Found time using pattern: {match_time}")
                                break
                        
                        # If no time found, use day of week pattern
                        if not match_time:
                            day_match = re.search(r'((?:Sun|Mon|Tue|Wed|Thu|Fri|Sat)\s+\w+\s+\d{1,2})', header_text)
                            if day_match:
                                match_time = day_match.group(1).strip()
                                logger.info(f"Using day pattern as time: {match_time}")
                        
                        # Extract date
                        date_match = re.search(r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})', header_text)
                        if date_match:
                            month = date_match.group(1)
                            day = date_match.group(2)
                            year_val = year or datetime.now().year
                            try:
                                date_obj = datetime.strptime(f"{month} {day} {year_val}", "%b %d %Y")
                                match_date = date_obj.strftime("%Y-%m-%d")
                                logger.info(f"Extracted date from regex: {match_date}")
                            except:
                                match_date = f"{month} {day}"

                    # Try to extract round information
                    try:
                        # Use the round from the header if available
                        if match_round:
                            # Clean up the round value (remove " • Singles" if present)
                            match_round = re.sub(r'\s*•\s*Singles', '', match_round, flags=re.IGNORECASE)
                            if match_round.lower() == "singles":
                                match_round = "Unknown Round"
                            logger.info(f"Using round from header: {match_round}")
                        else:
                            # Fallback to element if header didn't provide a round
                            round_elem = await card.query_selector('.round, [class*="round"], [class*="phase"]')
                            if round_elem:
                                match_round = await round_elem.inner_text()
                                match_round = match_round.strip()
                                # Clean up the round value (remove " • Singles" if present)
                                match_round = re.sub(r'\s*•\s*Singles', '', match_round, flags=re.IGNORECASE)
                                if match_round.lower() == "singles":
                                    match_round = "Unknown Round"
                                logger.info(f"Found round from element: {match_round}")
                            
                            # If still not found, try to extract from header text
                            if not match_round and header_text:
                                round_match = re.search(r'(Round of \d+|Final|Semi-?final|Quarter-?final|Qualifier)', header_text, re.IGNORECASE)
                                if round_match:
                                    match_round = round_match.group(1)
                                    logger.info(f"Found round from header: {match_round}")
                                
                            # Try to parse round from tournament info if present
                            if not match_round:
                                tournament_info = await card.inner_text()
                                round_match = re.search(r'(Round of \d+|Final|Semi-?final|Quarter-?final|Qualifier)', tournament_info, re.IGNORECASE)
                                if round_match:
                                    match_round = round_match.group(1)
                                    logger.info(f"Found round from tournament info: {match_round}")
                                
                            # If still not found, set to "Unknown Round"
                            if not match_round:
                                match_round = "Unknown Round"
                                logger.info(f"No round information found, defaulting to: {match_round}")
                    except Exception as e:
                        logger.warning(f"Error extracting round info: {e}")
                        match_round = "Unknown Round"

                    # If header parsing failed, try to extract date from element
                    if not match_date:
                        try:
                            date_elem = await card.query_selector('div.date, [class*="date"]')
                            if date_elem:
                                date_text = await date_elem.inner_text()
                                date_text = date_text.strip()
                                # Extract date using regex
                                date_match = re.search(r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+(\d{1,2})(,?\s+\d{4})?', date_text)
                                if date_match:
                                    month = date_match.group(1)
                                    day = date_match.group(2)
                                    year_val = year or datetime.now().year
                                    try:
                                        date_obj = datetime.strptime(f"{month} {day} {year_val}", "%b %d %Y")
                                        match_date = date_obj.strftime("%Y-%m-%d")
                                        logger.info(f"Extracted date from date element: {match_date}")
                                    except:
                                        match_date = f"{month} {day}, {year_val}"
                                        logger.info(f"Using raw date text: {match_date}")
                        except Exception as e:
                            logger.warning(f"Error extracting date from element: {e}")

                    # Determine which event this match belongs to
                    event_name = "Unknown Tournament"  # Default value
                    draw_name = "Unknown Draw"  # Default value
                    try:
                        match_y_position = await self.page.evaluate("""
                            (element) => {
                                const rect = element.getBoundingClientRect();
                                return rect.top;
                            }
                        """, card)
                        event_name, draw_name = get_event_for_position(match_y_position)
                        logger.info(f"Match card {match_idx+1} at position {match_y_position} belongs to event: {event_name} (Draw: {draw_name})")
                    except Exception as e:
                        logger.warning(f"Error determining event for match {match_idx+1}: {e}")
                        event_name = "Unknown Tournament"
                        draw_name = "Unknown Draw"

                    # Determine tournament type based on name
                    tournament_type = "Regular"
                    # Check for exhibition tournaments
                    exhibition_terms = ['exhibition', 'laver cup', 'show match', 'charity']
                    for term in exhibition_terms:
                        if term.lower() in event_name.lower():
                            tournament_type = "Exhibition"
                            break

                    # Determine if exhibition match
                    is_exhibition = tournament_type == "Exhibition"
                    if is_exhibition:
                        logger.info(f"Match {match_idx+1} is an exhibition match")

                    # Look for player and opponent elements in the card
                    teams = await card.query_selector_all('.team, .team.aic, [class*="team"], div.players')
                    
                    if not teams or len(teams) < 2:
                        # Try more generic approach to find player elements
                        try:
                            # Find all names in the card
                            name_elems = await card.query_selector_all('[class*="name"], .player-name, div.name')
                            if len(name_elems) >= 2:
                                # Create two team elements from the names
                                player_team = name_elems[0]
                                opponent_team = name_elems[1]
                                teams = [player_team, opponent_team]
                        except:
                            pass
                    
                    if not teams or len(teams) < 2:
                        logger.warning(f"Could not identify player and opponent elements in match {match_idx+1}")
                        continue
                    
                    # Determine which team is the player and which is the opponent
                    player_team = None
                    opponent_team = None
                    
                    for team in teams:
                        try:
                            player_link = await team.query_selector(f'a[href*="{player_id}"]')
                            if player_link:
                                player_team = team
                            else:
                                opponent_team = team
                        except:
                            continue
                    
                    # If we couldn't identify the teams, assume first team is player and second is opponent
                    if not player_team or not opponent_team:
                        if len(teams) >= 2:
                            player_team = teams[0]
                            opponent_team = teams[1]
                        else:
                            logger.warning(f"Could not identify player and opponent in match {match_idx+1}")
                            continue
                    
                    # Extract player info
                    extracted_player_name = None
                    try:
                        player_name_elem = await player_team.query_selector('.player-name, [class*="player-name"], [class*="name"], div.name')
                        if player_name_elem:
                            extracted_player_name = await player_name_elem.inner_text()
                            extracted_player_name = extracted_player_name.strip()
                        else:
                            # Try whole team text if no name element found
                            extracted_player_name = await player_team.inner_text()
                            extracted_player_name = extracted_player_name.strip().split('\n')[0]
                            
                        # If we still don't have a player name, use the global name
                        if not extracted_player_name and player_name:
                            extracted_player_name = player_name
                    except Exception as e:
                        logger.warning(f"Error extracting player name: {e}")
                        extracted_player_name = player_name or "Unknown"
                    
                    player_utr = None
                    try:
                        player_utr_elem = await player_team.query_selector('.utr, [class*="utr"], [class*="rating"]')
                        if player_utr_elem:
                            utr_text = await player_utr_elem.inner_text()
                            utr_text = utr_text.strip()
                            # Handle "UR" case
                            if utr_text == "UR":
                                player_utr = "UR"
                            else:
                                utr_match = re.search(r'([\d.]+)', utr_text)
                                if utr_match:
                                    player_utr = float(utr_match.group(1))
                    except Exception as e:
                        logger.warning(f"Error extracting player UTR: {e}")
                    
                    # Extract opponent info
                    opponent_id = None
                    try:
                        opponent_link_elem = await opponent_team.query_selector('a[href*="profiles"]')
                        if opponent_link_elem:
                            href = await opponent_link_elem.get_attribute('href')
                            id_match = re.search(r'/profiles/(\d+)', href)
                            if id_match:
                                opponent_id = id_match.group(1)
                    except Exception as e:
                        logger.warning(f"Error extracting opponent ID: {e}")
                    
                    opponent_name = None
                    try:
                        opponent_name_elem = await opponent_team.query_selector('.player-name, [class*="player-name"], [class*="name"], div.name')
                        if opponent_name_elem:
                            opponent_name = await opponent_name_elem.inner_text()
                            opponent_name = opponent_name.strip()
                        else:
                            # Try whole team text if no name element found
                            opponent_name = await opponent_team.inner_text()
                            opponent_name = opponent_name.strip().split('\n')[0]
                    except Exception as e:
                        logger.warning(f"Error extracting opponent name: {e}")
                        opponent_name = "Unknown"
                    
                    opponent_utr = None
                    try:
                        opponent_utr_elem = await opponent_team.query_selector('.utr, [class*="utr"], [class*="rating"]')
                        if opponent_utr_elem:
                            utr_text = await opponent_utr_elem.inner_text()
                            utr_text = utr_text.strip()
                            # Handle "UR" case
                            if utr_text == "UR":
                                opponent_utr = "UR"
                            else:
                                utr_match = re.search(r'([\d.]+)', utr_text)
                                if utr_match:
                                    opponent_utr = float(utr_match.group(1))
                    except Exception as e:
                        logger.warning(f"Error extracting opponent UTR: {e}")
                    
                    # Extract scores with improved tiebreak detection
                    player_scores = []
                    opponent_scores = []
                    raw_score = ""
                    formatted_score = ""

                    try:
                        # Find score elements
                        player_score_elems = await player_team.query_selector_all('.score-item, [class*="score-item"]')
                        opponent_score_elems = await opponent_team.query_selector_all('.score-item, [class*="score-item"]')
                        
                        logger.info(f"Found {len(player_score_elems)} player score elements and {len(opponent_score_elems)} opponent score elements")
                        
                        # Process each set score
                        for i in range(min(len(player_score_elems), len(opponent_score_elems))):
                            try:
                                # Get player score
                                p_raw = await player_score_elems[i].inner_text()
                                p_raw = p_raw.strip().replace('\n', '')
                                logger.debug(f"Raw player score text: '{p_raw}'")
                                p_score = self.handle_superscript(p_raw)
                                logger.debug(f"Processed player score: '{p_score}'")
                                
                                # Get opponent score
                                o_raw = await opponent_score_elems[i].inner_text()
                                o_raw = o_raw.strip().replace('\n', '')
                                logger.debug(f"Raw opponent score text: '{o_raw}'")
                                o_score = self.handle_superscript(o_raw)
                                logger.debug(f"Processed opponent score: '{o_score}'")
                                
                                # Create a single set score with both player and opponent
                                set_score = f"{p_score}-{o_score}"
                                
                                # Build the arrays for individual processing if needed
                                player_scores.append(p_score)
                                opponent_scores.append(o_score)
                                
                                logger.info(f"Set {i+1}: {p_score} vs {o_score}")
                            except Exception as e:
                                logger.warning(f"Error processing set {i+1}: {e}")
                        
                        # Combine into a raw score string
                        if player_scores and opponent_scores:
                            raw_score = " ".join([f"{p}-{o}" for p, o in zip(player_scores, opponent_scores)])
                            logger.debug(f"Initial raw_score: '{raw_score}'")
                            
                            # Apply our enhanced clean_tennis_score to format tiebreaks correctly
                            formatted_score = self.clean_tennis_score(raw_score)
                            logger.debug(f"Initial formatted_score: '{formatted_score}'")
                            
                            logger.info(f"Raw score: {raw_score}")
                            logger.info(f"Formatted score: {formatted_score}")
                        
                        # Special handling for score patterns that might not be numeric
                        if raw_score and re.search(r'[A-Za-z]', raw_score):
                            logger.warning(f"Score contains non-numeric characters: '{raw_score}' - attempting to fix")
                            
                            # Try extracting just the numeric parts of the score
                            numeric_parts = []
                            for part in raw_score.split():
                                # For each part, extract numbers separated by a dash
                                matches = re.findall(r'(\d+)[^0-9]+(\d+)', part)
                                if matches:
                                    for match in matches:
                                        numeric_parts.append(f"{match[0]}-{match[1]}")
                            
                            if numeric_parts:
                                fixed_score = " ".join(numeric_parts)
                                logger.info(f"Fixed score with numeric extraction: {fixed_score}")
                                raw_score = fixed_score
                                formatted_score = fixed_score
                                logger.debug(f"After fixing non-numeric: raw_score='{raw_score}', formatted_score='{formatted_score}'")
                        
                        # Special handling for retired matches
                        if retired:
                            logger.info(f"Processing score for retired match {match_idx+1}")
                            
                            # For retired matches, directly extract scores from elements
                            direct_extraction = False
                            original_raw_score = raw_score
                            original_formatted_score = formatted_score
                            
                            try:
                                # Dump all score element contents for debugging
                                logger.debug("DEBUG: All player score element contents:")
                                for i, elem in enumerate(player_score_elems):
                                    content = await elem.inner_text()
                                    logger.debug(f"  Player score elem {i}: '{content.strip()}'")
                                
                                logger.debug("DEBUG: All opponent score element contents:")
                                for i, elem in enumerate(opponent_score_elems):
                                    content = await elem.inner_text()
                                    logger.debug(f"  Opponent score elem {i}: '{content.strip()}'")
                                
                                # Extract direct numeric values from score elements
                                p_direct_scores = []
                                o_direct_scores = []
                                
                                for elem in player_score_elems:
                                    text = await elem.inner_text()
                                    text = text.strip()
                                    logger.debug(f"Checking player elem text: '{text}'")
                                    # Only add if it's a numeric value
                                    if re.match(r'^\d+$', text):
                                        p_direct_scores.append(text)
                                        logger.debug(f"Added numeric player score: {text}")
                                
                                for elem in opponent_score_elems:
                                    text = await elem.inner_text()
                                    text = text.strip()
                                    logger.debug(f"Checking opponent elem text: '{text}'")
                                    # Only add if it's a numeric value
                                    if re.match(r'^\d+$', text):
                                        o_direct_scores.append(text)
                                        logger.debug(f"Added numeric opponent score: {text}")
                                
                                logger.debug(f"Direct scores: player={p_direct_scores}, opponent={o_direct_scores}")
                                
                                # If we have direct scores, use them
                                if p_direct_scores and o_direct_scores:
                                    direct_score_parts = []
                                    for i in range(min(len(p_direct_scores), len(o_direct_scores))):
                                        direct_score_parts.append(f"{p_direct_scores[i]}-{o_direct_scores[i]}")
                                    
                                    if direct_score_parts:
                                        direct_score = " ".join(direct_score_parts)
                                        logger.info(f"Directly extracted retired match score: {direct_score}")
                                        raw_score = direct_score
                                        formatted_score = direct_score
                                        direct_extraction = True
                                        logger.debug(f"After direct extraction: raw_score='{raw_score}', formatted_score='{formatted_score}'")
                            except Exception as e:
                                logger.error(f"Error in direct score extraction for retired match: {e}")
                            
                            # If direct extraction failed, try fallback methods
                            if not direct_extraction or not raw_score:
                                logger.debug("Direct extraction failed or produced empty score, trying fallbacks")
                                
                                # Try to extract any score-like pattern from the card text
                                try:
                                    card_text = await card.inner_text()
                                    logger.debug(f"Card text for score extraction: '{card_text[:300]}...'")  # Log first 300 chars
                                    score_patterns = re.findall(r'(\d+)[-/](\d+)', card_text)
                                    logger.debug(f"Found score patterns: {score_patterns}")
                                    
                                    if score_patterns:
                                        fallback_scores = []
                                        for pattern in score_patterns:
                                            fallback_scores.append(f"{pattern[0]}-{pattern[1]}")
                                        
                                        fallback_score = " ".join(fallback_scores)
                                        logger.info(f"Extracted score from card text: {fallback_score}")
                                        raw_score = fallback_score
                                        formatted_score = fallback_score
                                        logger.debug(f"After fallback extraction: raw_score='{raw_score}', formatted_score='{formatted_score}'")
                                except Exception as e:
                                    logger.error(f"Error in fallback score extraction: {e}")
                            
                            # Log what happened to scores during retired match processing
                            logger.debug(f"Score before retired processing: '{original_raw_score}', after: '{raw_score}'")
                            logger.debug(f"Formatted score before retired processing: '{original_formatted_score}', after: '{formatted_score}'")
                            
                            # Super fallback for any remaining "Mar-00" pattern
                            if "Mar-00" in raw_score or "Mar-0" in raw_score:
                                logger.warning(f"CRITICAL: 'Mar-00' pattern still present in score '{raw_score}' after all processing")
                                # Try one last approach - forcibly convert month abbreviations to numbers
                                month_map = {"Jan": "1", "Feb": "2", "Mar": "3", "Apr": "4", "May": "5", "Jun": "6", 
                                            "Jul": "7", "Aug": "8", "Sep": "9", "Oct": "10", "Nov": "11", "Dec": "12"}
                                
                                for month, num in month_map.items():
                                    if month in raw_score:
                                        logger.warning(f"Forcibly replacing '{month}' with '{num}' in score")
                                        raw_score = raw_score.replace(f"{month}-", f"{num}-")
                                        formatted_score = formatted_score.replace(f"{month}-", f"{num}-")
                        
                        # Set the final cleaned score
                        cleaned_score = formatted_score or raw_score
                        logger.debug(f"Final cleaned_score: '{cleaned_score}'")
                        
                        # Special pre-storage treatment to prevent Excel date conversions
                        if cleaned_score:
                            # Escape any score that might be interpreted as a date by adding a quote prefix
                            month_abbrs = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                            if any(abbr in cleaned_score for abbr in month_abbrs):
                                original_cleaned = cleaned_score
                                # Convert month abbreviations to numbers
                                for i, month in enumerate(month_abbrs, 1):
                                    cleaned_score = re.sub(f"{month}-", f"{i}-", cleaned_score)
                                logger.warning(f"Pre-storage conversion: '{original_cleaned}' -> '{cleaned_score}'")
                    
                    except Exception as e:
                        logger.error(f"Error extracting scores: {e}")
                        cleaned_score = ""

                    # Final improved winner determination logic
                    is_winner = False
                    try:
                        # Try different selector variations to capture the winning-scores element
                        winning_scores_selectors = [
                            # Standard approach with class containing "winning-scores"
                            '[class*="winner-display-container"] [class*="winning-scores"]',
                            # Exact class name approach
                            '.winning-scores',
                            # Direct child approach
                            'div.d-flex.flex-column.jcc.winner-display-container > div.winning-scores',
                            # More general approach
                            'div[class*="winner-display"] div[class*="winning"]'
                        ]
                        
                        player_has_winning = False
                        opponent_has_winning = False
                        
                        # Try each selector until we find a match
                        for selector in winning_scores_selectors:
                            player_winning = await player_team.query_selector(selector)
                            opponent_winning = await opponent_team.query_selector(selector)
                            
                            if player_winning:
                                player_has_winning = True
                                logger.info(f"Match {match_idx+1}: Found player winning-scores with selector: {selector}")
                                break
                            elif opponent_winning:
                                opponent_has_winning = True
                                logger.info(f"Match {match_idx+1}: Found opponent winning-scores with selector: {selector}")
                                break
                        
                        # If none of the selectors worked, try JavaScript as a final approach
                        if not player_has_winning and not opponent_has_winning:
                            js_check = """
                            (element) => {
                                // Check all possible variations of winning-scores
                                const winnerContainer = element.querySelector('[class*="winner-display-container"]');
                                if (!winnerContainer) return false;
                                
                                // Look for any child with winning-scores in the class
                                const winningScores = winnerContainer.querySelector('[class*="winning-scores"]');
                                if (winningScores) return true;
                                
                                // Check for any child with "winning" in the class name
                                const winningElement = winnerContainer.querySelector('[class*="winning"]');
                                return winningElement !== null;
                            }
                            """
                            
                            try:
                                player_has_winning = await self.page.evaluate(js_check, player_team)
                                opponent_has_winning = await self.page.evaluate(js_check, opponent_team)
                                
                                if player_has_winning or opponent_has_winning:
                                    logger.info(f"Match {match_idx+1}: Found winning indicator using JavaScript approach")
                            except Exception as js_error:
                                logger.error(f"JavaScript evaluation error: {js_error}")
                        
                        # Determine winner based on our findings
                        if player_has_winning:
                            is_winner = True
                            logger.info(f"Match {match_idx+1}: Player won (found winning-scores element)")
                        elif opponent_has_winning:
                            is_winner = False
                            logger.info(f"Match {match_idx+1}: Opponent won (found winning-scores element)")
                        else:
                            # Fallback to set score analysis if no winning-scores found
                            logger.warning(f"Match {match_idx+1}: No winning-scores element found, falling back to set score analysis")
                            
                            player_sets_won = 0
                            opponent_sets_won = 0
                            
                            for i in range(min(len(player_scores), len(opponent_scores))):
                                p_match = re.search(r'(\d+)', player_scores[i])
                                o_match = re.search(r'(\d+)', opponent_scores[i])
                                
                                if p_match and o_match:
                                    p_val = int(p_match.group(1))
                                    o_val = int(o_match.group(1))
                                    
                                    if p_val > o_val:
                                        player_sets_won += 1
                                    elif o_val > p_val:
                                        opponent_sets_won += 1
                            
                            if player_sets_won > opponent_sets_won:
                                is_winner = True
                                logger.info(f"Match {match_idx+1}: Player won (based on set count - {player_sets_won} to {opponent_sets_won})")
                            elif opponent_sets_won > player_sets_won:
                                is_winner = False
                                logger.info(f"Match {match_idx+1}: Opponent won (based on set count - {opponent_sets_won} to {player_sets_won})")
                            else:
                                # If we still can't determine a winner, skip this match
                                logger.warning(f"Match {match_idx+1}: Could not determine winner - skipping this match")
                                continue
                    except Exception as e:
                        logger.error(f"Error determining winner for match {match_idx+1}: {e}")
                        # Skip this match if we couldn't determine the winner
                        continue

                    result = "W" if is_winner else "L"
                    
                    # Add this after determining the winner but before creating the match_info
                    if is_winner:
                        logger.info(f"Match {match_idx+1}: {extracted_player_name} WON against {opponent_name} with score {cleaned_score}")
                    else:
                        logger.info(f"Match {match_idx+1}: {extracted_player_name} LOST to {opponent_name} with score {cleaned_score}")
                    
                    # Create a unique match identifier
                    if match_date and opponent_id:
                        match_id = f"{match_date}_{player_id}_{opponent_id}"
                    else:
                        match_id = f"{player_id}_{opponent_id or 'unknown'}_{match_idx}"
                    
                    # Create match record
                    match_info = {
                        "match_id": match_id,
                        "player_id": player_id,
                        "player_name": extracted_player_name,
                        "player_utr_displayed": player_utr,
                        "date": match_date,
                        "time": match_time,
                        "tournament": event_name,
                        "draw": draw_name,  # New field for the draw name
                        "tournament_type": tournament_type,
                        "round": match_round,
                        "opponent_name": opponent_name,
                        "opponent_id": opponent_id,
                        "opponent_utr_displayed": opponent_utr,
                        "score": cleaned_score,
                        "raw_score": raw_score if raw_score else cleaned_score,
                        "result": result,
                        "retired": retired,  
                        "is_exhibition": is_exhibition
                    }
                    
                    matches.append(match_info)
                    logger.info(f"Processed match {match_idx+1}: {extracted_player_name} vs {opponent_name}")
                
                except Exception as e:
                    logger.error(f"Error processing match {match_idx+1}: {e}")
                    continue

            # Create DataFrame and save
            if matches:
                df = pd.DataFrame(matches)
                
                # Clean up any missing dates
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df = df.sort_values('date', na_position='last')  # Sort by date, putting NaN dates last
                
                # If a specific year was requested, filter for that year
                year_specific_df = pd.DataFrame()  # Initialize to avoid reference error
                if year and not df.empty and 'date' in df.columns:
                    year_int = int(year)
                    year_specific_df = df[df['date'].dt.year == year_int].copy()
                    logger.info(f"Filtered matches to year {year}: {len(year_specific_df)} matches")
                
                # Clean up scores - format them properly
                if 'score' in df.columns:
                    # First do the normal cleaning
                    df['score'] = df['score'].apply(lambda s: self.clean_tennis_score(s) if isinstance(s, str) else s)
                    
                    # Now add a specific fix for Excel's date interpretation
                    def prevent_excel_date_conversion(score):
                        if not isinstance(score, str):
                            return score
                        
                        # Add a single quote prefix to force Excel to treat as text
                        return "'" + score
                    
                    # Apply the fix to all scores
                    df['score'] = df['score'].apply(prevent_excel_date_conversion)
                    logger.info("Added Excel text format prefix to all scores to prevent date conversion")

                # Save the year-specific file if a year was requested
                if year and not year_specific_df.empty:
                    year_file = self.matches_dir / f"player_{player_id}_matches_{year}.csv"
                    year_specific_df.to_csv(year_file, index=False)
                    logger.info(f"Saved {len(year_specific_df)} matches for player {player_id} in year {year}")
                
                # Now append to the main player file (without overwriting existing data)
                main_file = self.matches_dir / f"player_{player_id}_matches.csv"
                
                # Check if main file exists and read it if it does
                if main_file.exists():
                    try:
                        # Read existing data
                        existing_df = pd.read_csv(main_file)
                        if not existing_df.empty:
                            # Log existing data for debugging
                            logger.info(f"Found existing file for player {player_id} with {len(existing_df)} matches")
                            
                            # Check for year distribution in existing data
                            if 'date' in existing_df.columns:
                                existing_df['date'] = pd.to_datetime(existing_df['date'], errors='coerce')
                                year_counts = existing_df.groupby(existing_df['date'].dt.year).size().to_dict()
                                logger.info(f"Year distribution in existing file: {year_counts}")
                            
                            # Combine with new data
                            combined_df = pd.concat([existing_df, df], ignore_index=True)
                            
                            # Remove duplicates based on match_id
                            if 'match_id' in combined_df.columns:
                                before_dedup = len(combined_df)
                                combined_df = combined_df.drop_duplicates(subset=['match_id'], keep='first')
                                logger.info(f"Removed {before_dedup - len(combined_df)} duplicate matches")
                            
                            df = combined_df
                            logger.info(f"Combined new and existing data for a total of {len(df)} matches")
                    except Exception as e:
                        logger.warning(f"Error reading existing match file: {e}")
                
                # Save combined data to main file
                df.to_csv(main_file, index=False)
                logger.info(f"Saved {len(df)} total matches to main file for player {player_id}")

            # Return the correct dataset based on what was requested
            if year:
                if not year_specific_df.empty:
                    logger.info(f"Returning {len(year_specific_df)} matches for year {year}")
                    return year_specific_df
                else:
                    logger.info(f"No matches found for year {year}")
                    return pd.DataFrame()
            else:
                return df
        except Exception as e:
            logger.error(f"Error getting match history for player {player_id}: {e}")
            try:
                await self.page.screenshot(path=f"match_history_error_{player_id}.png")
            except Exception as screenshot_error:
                logger.error(f"Failed to take error screenshot: {screenshot_error}")
            return pd.DataFrame()
                       

    async def search_and_get_player_by_name(self, name):
        """
        Search for a player by name and navigate to their profile
        
        Parameters:
        name (str): Player name to search for
        
        Returns:
        str: Player ID if found, None otherwise
        """
        try:
            logger.info(f"Searching for player: {name}")
            
            # Go to the app domain home page
            await self.page.goto("https://app.utrsports.net/home")
            
            # Check for and click "CONTINUE" button if present
            try:
                logger.info("Checking for CONTINUE button...")
                continue_button = await self.page.query_selector('button:has-text("CONTINUE")')
                if continue_button:
                    logger.info("Clicking CONTINUE button")
                    await continue_button.click()
                    await asyncio.sleep(2)
            except Exception as e:
                logger.warning(f"Error handling CONTINUE button: {e}")
            
            # Handle cookie consent if present
            try:
                cookie_button = await self.page.query_selector('button[data-testid="onetrust-accept-btn-handler"]')
                if cookie_button:
                    logger.info("Accepting cookies")
                    await cookie_button.click()
                    await asyncio.sleep(1)
            except Exception as e:
                logger.warning(f"Error handling cookie consent: {e}")
            
            # Take a screenshot to see current state
            await self.page.screenshot(path="home_page.png")
            
            # Click on the search icon/input
            logger.info("Clicking search icon")
            search_input = await self.page.query_selector('input[placeholder="Search"]')
            if search_input:
                await search_input.click()
            else:
                logger.error("Could not find search input")
                return None
            
            # Wait for dropdown to appear
            await asyncio.sleep(2)
            
            # Click on "Players" option
            players_option = await self.page.query_selector('div:has-text("Players")')
            if players_option:
                logger.info("Clicking Players option")
                await players_option.click()
            else:
                logger.error("Could not find Players option in dropdown")
                return None
            
            # Wait for search page to load
            await asyncio.sleep(2)
            
            # Fill player name in search
            search_input = await self.page.query_selector('input[placeholder="Search"]')
            if search_input:
                logger.info(f"Entering player name: {name}")
                await search_input.fill(name)
            else:
                logger.error("Could not find search input on search page")
                return None
            
            # Wait for search results
            await asyncio.sleep(3)
            
            # Click on player in results
            player_element = await self.page.query_selector(f'span:text("{name}")')
            if player_element:
                logger.info(f"Found player: {name}")
                await player_element.click()
                
                # Wait for profile to load
                await asyncio.sleep(3)
                
                # Extract player ID from URL
                current_url = self.page.url
                id_match = re.search(r"profiles/(\d+)", current_url)
                if id_match:
                    player_id = id_match.group(1)
                    logger.info(f"Navigated to player profile, ID: {player_id}")
                    return player_id
            
            logger.warning(f"Player not found in search results: {name}")
            return None
            
        except Exception as e:
            logger.error(f"Error searching for player {name}: {e}")
            await self.page.screenshot(path=f"search_error_{name.replace(' ', '_')}.png")
            return None
    
    async def collect_top_players_data(self, gender="men", limit=100, player_limit=None, include_matches=True, include_ratings=True, years=None, utr_threshold=1.0, dynamic_years=False):
        """
        Collect comprehensive data for top players and their opponents
        
        Parameters:
        gender (str): 'men' or 'women'
        limit (int): Maximum number of top players to retrieve
        player_limit (int): Optional limit of players to process (for testing)
        include_matches (bool): Whether to collect match history
        include_ratings (bool): Whether to collect rating history
        years (list): Optional list of years to collect match data for (e.g., ["2023", "2022"])
        utr_threshold (float): UTR threshold for collecting match history of opponents
        dynamic_years (bool): Whether to automatically detect all available years for each player
        
        Returns:
        DataFrame: Top players with IDs and UTR ratings
        """
        # Set default years if none provided
        if years is None:
            years = ["2025", "2024", "2023", "2022"]
        
        # Get top players
        top_players = self.get_top_players(gender=gender, limit=limit)
        
        if top_players.empty:
            logger.error(f"Failed to retrieve top {gender}'s players")
            return pd.DataFrame()
        
        # Limit to first N players if player_limit is specified
        if player_limit and len(top_players) > player_limit:
            top_players = top_players.head(player_limit)
            logger.info(f"Limited processing to first {player_limit} players")
        
        # Track all opponent IDs we encounter
        all_opponent_ids = set()
        high_utr_opponents = set()  # Opponents with UTR above threshold
        second_degree_opponents = set()  # To track opponents of opponents
        
        # Process each top player
        for _, player in top_players.iterrows():
            player_id = player['id']
            name = player['name']
            
            if not player_id:
                logger.warning(f"Skipping player {name} - no ID found")
                continue
            
            logger.info(f"Processing player: {name} (Rank: {player['rank']}, ID: {player_id})")
            
            try:
                # Get profile data
                profile = await self.get_player_profile(player_id)

                # If dynamic_years is enabled, try to get all available years for this player
                player_years = years
                if dynamic_years:
                    player_years = await self.get_available_years(player_id)
                    if not player_years:  # Fallback if no years found
                        player_years = years
                    logger.info(f"Using years for {name}: {', '.join(player_years)}")
                
                # Add random delay between players to avoid rate limiting
                await asyncio.sleep(random.uniform(1.0, 3.0))
                
                # Get rating history if requested
                if include_ratings:
                    try:
                        ratings = await self.get_player_rating_history(player_id)
                        logger.info(f"Retrieved {len(ratings)} rating history entries for {name}")
                    except Exception as e:
                        logger.error(f"Error getting rating history for {name}: {e}")
                    
                    # Add random delay
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Get match history if requested
                if include_matches:
                    # Process each year
                    for year in player_years:
                        try:
                            logger.info(f"Getting matches for {name} - Year {year}")
                            matches = await self.get_player_match_history(player_id, year=year, limit=100)
                            
                            # Collect opponent IDs from matches
                            if not matches.empty and 'opponent_id' in matches.columns:
                                # Add all opponents to the general set
                                opponent_ids = matches['opponent_id'].dropna().unique()
                                all_opponent_ids.update(opponent_ids)
                                
                                # Identify high UTR opponents
                                high_utr_matches = matches[pd.to_numeric(matches['opponent_utr_displayed'], errors='coerce') >= utr_threshold]
                                if not high_utr_matches.empty and 'opponent_id' in high_utr_matches.columns:
                                    high_utr_ids = high_utr_matches['opponent_id'].dropna().unique()
                                    high_utr_opponents.update(high_utr_ids)
                                
                                logger.info(f"Found {len(opponent_ids)} unique opponents in {year}")
                                logger.info(f"Found {len(high_utr_ids) if 'high_utr_ids' in locals() else 0} opponents with UTR >= {utr_threshold}")
                            
                            logger.info(f"Retrieved {len(matches)} matches for {name} in {year}")
                        except Exception as e:
                            logger.error(f"Error getting matches for {name} in {year}: {e}")
                        
                        # Add random delay between years
                        await asyncio.sleep(random.uniform(2.0, 3.0))
                    
                    # Add random delay between players
                    await asyncio.sleep(random.uniform(2.0, 4.0))
            except Exception as e:
                logger.error(f"Error processing player {name} ({player_id}): {e}")
        
        # Process opponent data after collecting all top players data
        logger.info(f"Found {len(all_opponent_ids)} unique opponents to process")
        logger.info(f"Found {len(high_utr_opponents)} unique opponents with UTR >= {utr_threshold}")
        
        # Process each opponent's data
        opponent_count = 0
        failed_opponents = 0
        
        for opponent_id in all_opponent_ids:
            try:
                # Check if we already have this player's data (might be in the top players list)
                rating_file = self.ratings_dir / f"player_{opponent_id}_ratings.csv"
                
                if rating_file.exists():
                    logger.info(f"Already have ratings for opponent {opponent_id}, skipping")
                    continue
                
                # Get opponent profile
                logger.info(f"Getting profile for opponent ID: {opponent_id} ({opponent_count+1}/{len(all_opponent_ids)})")
                
                try:
                    opponent_profile = await self.get_player_profile(opponent_id)
                    opponent_name = opponent_profile['name'] if opponent_profile else f"Opponent {opponent_id}"
                    opponent_utr = opponent_profile.get('current_utr') if opponent_profile else None
                except Exception as e:
                    logger.error(f"Error getting profile for opponent {opponent_id}: {e}")
                    opponent_name = f"Opponent {opponent_id}"
                    opponent_utr = None
                    # Create a blank profile to prevent retrying
                    blank_profile = {
                        "id": opponent_id,
                        "name": f"Opponent {opponent_id}",
                        "current_utr": None,
                        "profile_url": f"https://app.utrsports.net/profiles/{opponent_id}",
                        "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "error": str(e)
                    }
                    profile_file = self.players_dir / f"player_{opponent_id}_profile.json"
                    with open(profile_file, 'w') as f:
                        json.dump(blank_profile, f, indent=2)
                
                # Add random delay
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Get opponent rating history
                try:
                    logger.info(f"Getting rating history for opponent: {opponent_name} ({opponent_id})")
                    opponent_ratings = await self.get_player_rating_history(opponent_id)
                    logger.info(f"Retrieved {len(opponent_ratings)} rating history entries for {opponent_name}")
                except Exception as e:
                    logger.error(f"Error getting rating history for opponent {opponent_name}: {e}")
                    # Create blank rating file to prevent retrying
                    empty_df = pd.DataFrame(columns=['date', 'utr'])
                    empty_df.to_csv(self.ratings_dir / f"player_{opponent_id}_ratings.csv", index=False)
                    failed_opponents += 1
                
                # If this is a high-UTR opponent, get their match history too
                if opponent_id in high_utr_opponents:
                    logger.info(f"Getting match history for high-UTR opponent: {opponent_name} ({opponent_id})")
                    
                    # If dynamic_years, try to get all available years for this opponent
                    opponent_years = years
                    if dynamic_years:
                        opponent_years = await self.get_available_years(opponent_id)
                        if not opponent_years:  # Fallback if no years found
                            opponent_years = years
                        logger.info(f"Using years for opponent {opponent_name}: {', '.join(opponent_years)}")
                    
                    for year in opponent_years:
                        try:
                            logger.info(f"Getting matches for {opponent_name} - Year {year}")
                            opponent_matches = await self.get_player_match_history(opponent_id, year=year, limit=50)
                            
                            # Collect second-degree opponents
                            if not opponent_matches.empty and 'opponent_id' in opponent_matches.columns:
                                second_ids = opponent_matches['opponent_id'].dropna().unique()
                                # Add to second-degree set if not already in first-degree
                                for sid in second_ids:
                                    if sid not in all_opponent_ids and sid not in high_utr_opponents:
                                        second_degree_opponents.add(sid)
                            
                            logger.info(f"Retrieved {len(opponent_matches)} matches for {opponent_name} in {year}")
                        except Exception as e:
                            logger.error(f"Error getting matches for {opponent_name} in {year}: {e}")
                        
                        # Add random delay between years
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Add random delay between opponents
                await asyncio.sleep(random.uniform(2.0, 3.0))
                opponent_count += 1
                
                # Restart browser every 20 opponents to prevent memory issues
                if opponent_count % 20 == 0:
                    logger.info(f"Processed {opponent_count} opponents, browser restart needed")
                    # Don't try to restart here - just break the loop and let the outer context handle it
                    logger.info(f"Completed processing {opponent_count} opponents - browser will be restarted by parent process")
                    return top_players
                    
            except Exception as e:
                logger.error(f"Error processing opponent {opponent_id}: {e}")
                failed_opponents += 1
        
        # Process second-degree opponents (just get their rating histories)
        logger.info(f"Found {len(second_degree_opponents)} second-degree opponents to process")
        second_degree_count = 0
        
        for opponent_id in second_degree_opponents:
            try:
                # Check if we already have this player's data
                rating_file = self.ratings_dir / f"player_{opponent_id}_ratings.csv"
                
                if rating_file.exists():
                    logger.info(f"Already have ratings for second-degree opponent {opponent_id}, skipping")
                    continue
                
                # Get profile and rating history
                profile = await self.get_player_profile(opponent_id)
                name = profile['name'] if profile else f"Second-Degree Opponent {opponent_id}"
                logger.info(f"Getting rating history for second-degree opponent: {name} ({opponent_id})")
                
                ratings = await self.get_player_rating_history(opponent_id)
                logger.info(f"Retrieved {len(ratings)} rating history entries for {name}")
                
                second_degree_count += 1
                
                # Add delay
                await asyncio.sleep(random.uniform(1.0, 2.0))
                
                # Restart browser periodically
                if second_degree_count % 20 == 0:
                    logger.info(f"Processed {second_degree_count} second-degree opponents, restarting browser")
                    await self.close_browser()
                    await asyncio.sleep(2)
                    await self.start_browser()
                    await self.login()
                    await asyncio.sleep(2)
                    
            except Exception as e:
                logger.error(f"Error processing second-degree opponent {opponent_id}: {e}")
        
        logger.info(f"Completed data collection for {len(top_players)} {gender}'s players")
        logger.info(f"Processed {opponent_count} unique opponents with {failed_opponents} failures")
        logger.info(f"Processed {second_degree_count} second-degree opponents")
        
        # Run cross-reference to match UTRs with match dates
        logger.info("Running cross-reference to match UTRs with match dates")
        try:
            valid_matches = await self.cross_reference_matches_with_ratings()
            logger.info(f"Cross-reference complete with {len(valid_matches)} valid matches")
        except Exception as e:
            logger.error(f"Error during cross-reference: {e}")
            import traceback
            logger.error(f"Error trace: {traceback.format_exc()}")
        
        return top_players
    
    def clean_tennis_score(self, score_text):
        """
        Clean up and standardize tennis score notation
        
        Parameters:
        score_text (str): Raw score text from UTR
        
        Returns:
        str: Cleaned, standardized score
        """
        if not score_text:
            return ""
        
        if re.search(r'[A-Za-z]', score_text) and not re.search(r'ret', score_text, re.IGNORECASE):
            return score_text
        
        score_text = re.sub(r'[A-Za-z\.]+', '', score_text)
        
        tb_sets = []
        for set_text in score_text.split():
            tb_pattern = re.match(r'(\d)(\d+)-(\d)(\d+)', set_text)
            if tb_pattern:
                p1_game, p1_tb_digits, p2_game, p2_tb_digits = tb_pattern.groups()
                p1_game, p2_game = int(p1_game), int(p2_game)
                p1_tb = int(p1_tb_digits)
                p2_tb = int(p2_tb_digits)
                print(f"Matched pattern: p1_game={p1_game}, p2_game={p2_game}, p1_tb={p1_tb}, p2_tb={p2_tb}")
                
                if p1_game == 7 and p2_game == 6:
                    print(f"Tiebreak detected: 7-6, winner's tiebreak={p1_tb}, loser's tiebreak={p2_tb}")
                    tb_sets.append(f"7-6({p2_tb})")
                elif p1_game == 6 and p2_game == 7:
                    print(f"Tiebreak detected: 6-7, winner's tiebreak={p2_tb}, loser's tiebreak={p1_tb}")
                    tb_sets.append(f"6-7({p1_tb})")
                else:
                    tb_sets.append(f"{p1_game}-{p2_game}")
            else:
                tb_sets.append(set_text)
        
        if tb_sets:
            return " ".join(tb_sets)
        
        # Handle concatenated scores like "6366-7740" or "63710-6825"
        concat_match = re.search(r'(\d+)-(\d+)', score_text)
        if concat_match:
            p1_raw = concat_match.group(1)
            p2_raw = concat_match.group(2)
            min_length = min(len(p1_raw), len(p2_raw))
            new_sets = []
            i = 0
            while i < min_length:
                if i + 1 < min_length:
                    p1_game = p1_raw[i]
                    p2_game = p2_raw[i]
                    # Find the end of the tiebreak digits (next set score or end)
                    p1_tb_start = i + 1
                    p2_tb_start = i + 1
                    p1_tb_end = p1_tb_start
                    p2_tb_end = p2_tb_start
                    # Cap tiebreak length to the next set score or min length
                    while p1_tb_end < len(p1_raw) and p2_tb_end < len(p2_raw) and \
                        p1_tb_end < min_length and p2_tb_end < min_length and \
                        p1_raw[p1_tb_end].isdigit() and p2_raw[p2_tb_end].isdigit():
                        p1_tb_end += 1
                        p2_tb_end += 1
                    p1_tb = p1_raw[p1_tb_start:p1_tb_end] if p1_tb_start < p1_tb_end else '0'
                    p2_tb = p2_raw[p2_tb_start:p2_tb_end] if p2_tb_start < p2_tb_end else '0'
                    print(f"Concat match: p1_game={p1_game}, p2_game={p2_game}, p1_tb={p1_tb}, p2_tb={p2_tb}")
                    
                    if p1_game == '6' and p2_game == '7':
                        new_sets.append(f"6-7({p1_tb})")
                    elif p1_game == '7' and p2_game == '6':
                        new_sets.append(f"7-6({p2_tb})")
                    else:
                        new_sets.append(f"{p1_game}-{p2_game}")
                    i = p1_tb_end
                else:
                    break
            if new_sets:
                return " ".join(new_sets)
        
        # Look for standard set patterns like "6-4 7-6(5)"
        sets = []
        for set_match in re.finditer(r'(\d+)[-](\d+)(?:\((\d+)\))?', score_text):
            p1_score = set_match.group(1)
            p2_score = set_match.group(2)
            tb_score = set_match.group(3)
            
            if tb_score:
                sets.append(f"{p1_score}-{p2_score}({tb_score})")
            else:
                sets.append(f"{p1_score}-{p2_score}")
        
        if sets:
            return " ".join(sets)
        
        # Final attempt - just get any number-dash-number combinations
        simple_sets = re.findall(r'(\d+)[-](\d+)', score_text)
        if simple_sets:
            return " ".join([f"{p1}-{p2}" for p1, p2 in simple_sets])
        
        # If all else fails, return cleaned original
        cleaned = re.sub(r'[^0-9\-\(\) ]', ' ', score_text)
        return re.sub(r'\s+', ' ', cleaned).strip()
    
    async def get_available_years(self, player_id):
        """
        Determine all available years of match data for a player
        
        Parameters:
        player_id: Player's UTR ID
        
        Returns:
        list: All available years in descending order (newest first)
        """
        try:
            print(f"Checking available years for player {player_id}...")
            
            # Go to the player's results page
            profile_url = f"https://app.utrsports.net/profiles/{player_id}?t=2"
            
            try:
                await self.page.goto(profile_url, timeout=20000)
            except Exception as e:
                print(f"Error navigating to player page: {e}")
                return ["2025", "2024", "2023", "2022", "2021", "2020"]  # Fallback to default years
            
            # Wait for the page to load
            try:
                await self.page.wait_for_selector('h1', state="visible", timeout=15000)
                await asyncio.sleep(3)  # Extra wait for content to load
            except Exception as e:
                print(f"Error waiting for page content: {e}")
                return ["2025", "2024", "2023", "2022", "2021", "2020"]  # Fallback to default years
            
            # Click on the LATEST dropdown to see available years
            try:
                # Look for the LATEST button using the same selectors as in select_year_from_dropdown
                latest_selectors = [
                    'h6.text-uppercase.title:has-text("LATEST")',
                    '.popovermenu-anchor:has-text("LATEST")',
                    'span:has-text("LATEST")',
                    'div.popovermenu-anchor span.text-uppercase.title'
                ]
                
                latest_button = None
                for selector in latest_selectors:
                    latest_button = await self.page.query_selector(selector)
                    if latest_button:
                        break
                        
                if latest_button:
                    await latest_button.click()
                    await asyncio.sleep(2)  # Wait for dropdown to appear
                    
                    # Use JavaScript to extract all year options
                    years_js = """
                    () => {
                        const years = [];
                        const menuItems = document.querySelectorAll('.menu-item, [class*="menu-item"]');
                        for (const item of menuItems) {
                            const text = item.textContent.trim();
                            // Match 4-digit years only
                            if (/^20\d{2}$/.test(text)) {
                                years.push(text);
                            }
                        }
                        return years;
                    }
                    """
                    
                    available_years = await self.page.evaluate(years_js)
                    
                    if available_years and len(available_years) > 0:
                        print(f"Found {len(available_years)} available years: {', '.join(available_years)}")
                        return available_years
                
                # Click somewhere else to close the dropdown
                await self.page.click('h1')
                
            except Exception as e:
                print(f"Error getting available years: {e}")
            
            # Default years if we couldn't get available years
            return ["2025", "2024", "2023", "2022", "2021", "2020"]
            
        except Exception as e:
            print(f"Unexpected error in get_available_years: {e}")
            traceback.print_exc()
            return ["2025", "2024", "2023", "2022", "2021", "2020"]  # Fallback years


async def cross_reference_matches_with_ratings(self, max_days_diff=30):
    """
    Cross-reference match data with rating history to get UTR at match time
    
    Parameters:
    max_days_diff (int): Maximum number of days between rating date and match date
    
    Returns:
    DataFrame: Enhanced match data with UTR at match time
    """
    logger.info(f"Cross-referencing match data with rating history (max {max_days_diff} days difference)")
    
    # Wait briefly to ensure all files are fully written
    await asyncio.sleep(2)
    
    # Get all match files
    match_files = list(self.matches_dir.glob("player_*_matches.csv"))
    logger.info(f"Found {len(match_files)} match files to process")
    
    # Extract all unique player IDs (including both players and opponents)
    all_player_ids = set()
    opponent_ids = set()
    matches_by_player = {}  # To track which matches contain each player
    
    # First, collect all unique player IDs and track match importance
    for match_file in match_files:
        try:
            player_id_match = re.search(r"player_(\d+)_matches", str(match_file))
            if player_id_match:
                player_id = player_id_match.group(1)
                all_player_ids.add(player_id)
                
                # Read matches to get opponent IDs
                if os.path.getsize(match_file) > 0:  # Make sure file isn't empty
                    matches_df = pd.read_csv(match_file)
                    
                    # Track this player's matches
                    if player_id not in matches_by_player:
                        matches_by_player[player_id] = []
                    
                    # Add all matches for this player
                    if not matches_df.empty and 'match_id' in matches_df.columns:
                        matches_by_player[player_id].extend(matches_df['match_id'].dropna().tolist())
                    
                    if not matches_df.empty and 'opponent_id' in matches_df.columns:
                        file_opponent_ids = matches_df['opponent_id'].dropna().astype(str).unique()
                        opponent_ids.update(file_opponent_ids)
                        all_player_ids.update(file_opponent_ids)
                        
                        # Track matches by opponent as well
                        for opp_id in file_opponent_ids:
                            opp_matches = matches_df[matches_df['opponent_id'] == opp_id]
                            if not opp_matches.empty and 'match_id' in opp_matches.columns:
                                if opp_id not in matches_by_player:
                                    matches_by_player[opp_id] = []
                                matches_by_player[opp_id].extend(opp_matches['match_id'].dropna().tolist())
        except Exception as e:
            logger.error(f"Error processing match file {match_file}: {e}")
    
    logger.info(f"Found {len(all_player_ids)} unique players to process")
    logger.info(f"Including {len(opponent_ids)} unique opponents")
    
    # Check for missing rating histories and try to get them if needed
    missing_ratings = []
    for player_id in all_player_ids:
        if not player_id or player_id == 'nan':
            continue
            
        rating_file = self.ratings_dir / f"player_{player_id}_ratings.csv"
        if not rating_file.exists() or os.path.getsize(rating_file) == 0:
            missing_ratings.append(player_id)
    
    if missing_ratings:
        logger.info(f"Found {len(missing_ratings)} players with missing rating histories")
        
        # Sort missing ratings by match importance (higher matches count = more important)
        def get_match_count(player_id):
            return len(matches_by_player.get(player_id, []))
        
        # Sort by match count (descending) so more important players are processed first
        missing_ratings.sort(key=get_match_count, reverse=True)
        
        for idx, player_id in enumerate(missing_ratings):
            # Determine importance for retry strategy
            match_count = get_match_count(player_id)
            is_important = match_count > 2
            max_retries = 5 if is_important else 3
            
            logger.info(f"Getting missing rating history for {'important ' if is_important else ''}player {idx+1}/{len(missing_ratings)}: {player_id} (matches: {match_count})")
            
            try:
                # Get player profile first
                await self.get_player_profile(player_id)
                # Then get rating history with appropriate retries
                ratings = await self.get_player_rating_history(player_id, max_retries=max_retries)
                
                if not ratings.empty:
                    logger.info(f"Successfully retrieved {len(ratings)} ratings for player {player_id}")
                else:
                    logger.warning(f"Failed to get ratings for player {player_id} after multiple attempts")
                
                # Add a random delay to avoid rate limiting
                await asyncio.sleep(random.uniform(2.0, 4.0))
            except Exception as e:
                logger.error(f"Failed to get rating history for {player_id}: {e}")
                # Create empty file to avoid retrying this player
                empty_df = pd.DataFrame(columns=['date', 'utr'])
                empty_df.to_csv(self.ratings_dir / f"player_{player_id}_ratings.csv", index=False)
                
                # Restart browser every 10 players to avoid memory issues
                if idx % 10 == 9:
                    try:
                        logger.info(f"Restarting browser after processing {idx+1} players...")
                        await self.close_browser()
                        await asyncio.sleep(3)
                        await self.start_browser()
                        await self.login()
                        await asyncio.sleep(3)
                    except Exception as browser_error:
                        logger.error(f"Error restarting browser: {browser_error}")
    
    # Create a cache to store player ratings
    player_ratings_cache = {}
    
    # Load all existing rating histories into cache
    for rating_file in self.ratings_dir.glob("player_*_ratings.csv"):
        try:
            player_id_match = re.search(r"player_(\d+)_ratings", str(rating_file))
            if player_id_match and os.path.getsize(rating_file) > 0:  # Ensure file isn't empty
                player_id = player_id_match.group(1)
                
                # Only cache if we actually need this player's data
                if player_id in all_player_ids:
                    ratings_df = pd.read_csv(rating_file)
                    
                    # Skip empty rating files
                    if ratings_df.empty:
                        logger.warning(f"Rating file for player {player_id} exists but is empty")
                        continue
                        
                    # Handle the case where a player might be "UR" (unrated)
                    if 'utr' in ratings_df.columns:
                        # Convert "UR" to NaN for proper datetime operations
                        if ratings_df['utr'].dtype == object:  # Only for string columns
                            ratings_df.loc[ratings_df['utr'] == "UR", 'utr'] = np.nan
                        
                        # Ensure all UTR values are numeric
                        ratings_df['utr'] = pd.to_numeric(ratings_df['utr'], errors='coerce')
                    else:
                        logger.warning(f"Rating file for player {player_id} has no 'utr' column")
                        continue
                    
                    # Convert dates to datetime
                    if 'date' in ratings_df.columns:
                        ratings_df['date'] = pd.to_datetime(ratings_df['date'], errors='coerce')
                        ratings_df = ratings_df.sort_values('date')
                    else:
                        logger.warning(f"Rating file for player {player_id} has no 'date' column")
                        continue
                        
                    player_ratings_cache[player_id] = ratings_df
                    logger.info(f"Cached rating history for player {player_id}: {len(ratings_df)} entries")
        except Exception as e:
            logger.error(f"Error loading rating history from {rating_file}: {e}")
    
    # Helper function to get player rating at a specific date
    def get_player_rating_at_date(player_id, match_date):
        """Find the closest rating before match_date within max_days_diff"""
        if player_id is None or str(player_id) == 'nan':
            return None, None
        
        player_id = str(player_id)  # Ensure string type
        
        # Check if we have ratings for this player
        if player_id not in player_ratings_cache:
            logger.warning(f"No rating history available for player {player_id}")
            return None, None
        
        ratings_df = player_ratings_cache[player_id]
        
        # Ensure we have date and utr columns
        if 'date' not in ratings_df.columns or 'utr' not in ratings_df.columns:
            logger.warning(f"Invalid rating history format for player {player_id}")
            return None, None
        
        # Convert match_date to datetime if it's not already
        if not isinstance(match_date, pd.Timestamp) and not isinstance(match_date, datetime):
            try:
                match_date = pd.to_datetime(match_date)
            except:
                logger.warning(f"Invalid match date format: {match_date}")
                return None, None
        
        # Find the closest rating before the match date
        past_ratings = ratings_df[ratings_df['date'] <= match_date]
        
        if past_ratings.empty:
            return None, None
        
        # Make sure we don't select a row with null/NaN UTR
        past_ratings = past_ratings.dropna(subset=['utr'])
        
        if past_ratings.empty:
            return None, None
            
        closest_rating_idx = past_ratings.index[-1]
        closest_rating_date = past_ratings.loc[closest_rating_idx, 'date']
        
        # Calculate days between the rating and the match
        if pd.notnull(closest_rating_date) and pd.notnull(match_date):
            days_diff = (match_date - closest_rating_date).days
            
            if days_diff <= max_days_diff:
                return past_ratings.loc[closest_rating_idx, 'utr'], days_diff
        
        return None, None
    
    # Function to create standardized match IDs
    def get_standardized_match_id(row):
        """
        Create a standardized match ID by ordering player IDs
        so the same match will have the same ID regardless of perspective
        """
        if pd.isna(row['date']) or pd.isna(row['player_id']) or pd.isna(row['opponent_id']):
            return None
            
        date_str = str(row['date']).split()[0]  # Get just the date part
        player_id = str(row['player_id'])
        opponent_id = str(row['opponent_id'])
        
        # Sort the IDs so the smaller ID always comes first
        player_ids = sorted([player_id, opponent_id])
        
        # Create standardized match ID
        return f"{date_str}_{player_ids[0]}_{player_ids[1]}"
    
    # Process each match file to enhance with historical UTRs
    all_enhanced_matches = []
    valid_match_count = 0
    skipped_match_count = 0
    
    for match_file in match_files:
        try:
            # Extract player ID from filename
            player_id_match = re.search(r"player_(\d+)_matches", str(match_file))
            if not player_id_match:
                logger.warning(f"Could not extract player ID from {match_file}")
                continue
            
            player_id = player_id_match.group(1)
            
            # Load match data
            if os.path.getsize(match_file) == 0:  # Check if file is empty
                logger.warning(f"Match file for player {player_id} is empty")
                continue
                
            matches_df = pd.read_csv(match_file)
            
            if matches_df.empty:
                logger.warning(f"No matches found for player {player_id}")
                continue
            
            # Ensure date column is datetime
            matches_df['date'] = pd.to_datetime(matches_df['date'], errors='coerce')
            
            # Process each match to find UTRs at match time
            for idx, match in matches_df.iterrows():
                match_date = match['date']
                
                if pd.isna(match_date):
                    logger.warning(f"Skipping match with no date for player {player_id}")
                    skipped_match_count += 1
                    continue
                
                opponent_id = match.get('opponent_id')
                if pd.isna(opponent_id):
                    logger.warning(f"Skipping match with no opponent ID for player {player_id}")
                    skipped_match_count += 1
                    continue
                
                # Get player's UTR rating at match time
                player_rating, player_days_diff = get_player_rating_at_date(player_id, match_date)
                
                # Get opponent's UTR rating at match time
                opponent_rating, opponent_days_diff = get_player_rating_at_date(opponent_id, match_date)
                
                # Handle special cases like "UR" (Unrated) players
                player_displayed_utr = match.get('player_utr_displayed')
                opponent_displayed_utr = match.get('opponent_utr_displayed')
                
                # Special handling for "UR" (Unrated) players
                if str(player_displayed_utr) == "UR":
                    player_rating = "UR"
                    player_days_diff = 0
                
                if str(opponent_displayed_utr) == "UR":
                    opponent_rating = "UR"
                    opponent_days_diff = 0
                
                # Update match data with historical UTR ratings
                if player_rating is not None:
                    matches_df.at[idx, 'player_utr_at_match'] = player_rating
                    matches_df.at[idx, 'player_utr_days_diff'] = player_days_diff
                    
                    if opponent_rating is not None:
                        matches_df.at[idx, 'opponent_utr_at_match'] = opponent_rating
                        matches_df.at[idx, 'opponent_utr_days_diff'] = opponent_days_diff
                        
                        # Calculate UTR difference if both are numeric
                        if isinstance(player_rating, (int, float)) and isinstance(opponent_rating, (int, float)):
                            matches_df.at[idx, 'utr_diff'] = float(player_rating) - float(opponent_rating)
                            matches_df.at[idx, 'valid_for_model'] = True
                            valid_match_count += 1
                            logger.info(f"Valid match: {match['player_name']} ({player_rating}) vs {match['opponent_name']} ({opponent_rating}) on {match_date.strftime('%Y-%m-%d')}")
                        else:
                            # One of the players is "UR" - mark accordingly
                            matches_df.at[idx, 'utr_diff'] = None
                            matches_df.at[idx, 'valid_for_model'] = False
                            matches_df.at[idx, 'unrated_player'] = True
                            logger.info(f"Match with unrated player: {match['player_name']} ({player_rating}) vs {match['opponent_name']} ({opponent_rating})")
                    else:
                        matches_df.at[idx, 'valid_for_model'] = False
                        skipped_match_count += 1
                        logger.debug(f"No valid UTR for opponent {match['opponent_name']} (ID: {opponent_id}) within {max_days_diff} days of match")
                else:
                    matches_df.at[idx, 'valid_for_model'] = False
                    skipped_match_count += 1
                    logger.debug(f"No valid UTR for player {match['player_name']} within {max_days_diff} days of match")
            
            # Add to collection and save back to file
            all_enhanced_matches.append(matches_df)
            matches_df.to_csv(match_file, index=False)
            
            logger.info(f"Processed {len(matches_df)} matches for player {player_id}")
                
        except Exception as e:
            logger.error(f"Error processing matches from {match_file}: {e}")
    
    # Combine all enhanced matches
    if all_enhanced_matches:
        try:
            combined_df = pd.concat(all_enhanced_matches, ignore_index=True)
            
            # Add standardized match ID
            combined_df['standardized_match_id'] = combined_df.apply(get_standardized_match_id, axis=1)
            
            # Save all enhanced matches
            combined_file = self.data_dir / "all_enhanced_matches.csv"
            combined_df.to_csv(combined_file, index=False)
            
            # For the valid_matches output, we want to eliminate duplicates entirely
            # and only keep one version of each match
            valid_matches_df = combined_df[combined_df['valid_for_model'] == True].copy()
            
            # Use the standardized match ID to drop duplicates, keeping the first occurrence
            if 'standardized_match_id' in valid_matches_df.columns:
                original_count = len(valid_matches_df)
                valid_matches_df.drop_duplicates(subset=['standardized_match_id'], keep='first', inplace=True)
                dupe_count = original_count - len(valid_matches_df)
                if dupe_count > 0:
                    logger.info(f"Removed {dupe_count} duplicate matches from final valid matches output")
            
            # Save only valid matches with complete UTR data
            valid_file = self.data_dir / "valid_matches_for_model.csv"
            valid_matches_df.to_csv(valid_file, index=False)
            
            # Save matches with unrated players - FIX: Check if column exists first
            if 'unrated_player' in combined_df.columns:
                unrated_matches_df = combined_df[combined_df['unrated_player'] == True].copy()
                if not unrated_matches_df.empty:
                    unrated_file = self.data_dir / "unrated_player_matches.csv"
                    unrated_matches_df.to_csv(unrated_fileindex=False)
                    logger.info(f"Saved {len(unrated_matches_df)} matches with unrated players to {unrated_file}")
            
            logger.info(f"Saved {len(combined_df)} enhanced matches to {combined_file}")
            logger.info(f"Saved {len(valid_matches_df)} valid matches with complete UTR data to {valid_file}")
            logger.info(f"Valid matches: {valid_match_count}, Skipped matches: {skipped_match_count}")
            
            return valid_matches_df
        except Exception as e:
            logger.error(f"Error combining matches: {e}")
            # Log the stack trace for better debugging
            import traceback
            logger.error(f"Stack trace: {traceback.format_exc()}")
            return pd.DataFrame()
    else:
        logger.warning("No enhanced matches to combine")
        return pd.DataFrame()

# Example usage
if __name__ == "__main__":
    email = "zachdodson12@gmail.com"
    password = "Thailand@123"
    
    async def main():
        async with UTRScraper(email=email, password=password, headless=False) as scraper:
            if not await scraper.login():
                logger.error("Failed to log in to UTR website")
                return
            
            logger.info("Starting comprehensive test with Jannik Sinner")
            
            # 1. Get Sinner's profile
            player_id = "247320"  # Jannik Sinner's ID
            profile = await scraper.get_player_profile(player_id)
            logger.info(f"Got profile for {profile['name']}")
            
            # 2. Get Sinner's rating history
            ratings = await scraper.get_player_rating_history(player_id)
            logger.info(f"Got {len(ratings)} rating history entries for Sinner")
            
            # 3. Get Sinner's match history for each year
            all_opponent_ids = set()
            years = ["2025", "2024", "2023", "2022", "2021", "2020"]  # Added more years
            
            for year in years:
                logger.info(f"Getting matches for year {year}")
                matches = await scraper.get_player_match_history(player_id, year=year, limit=200)
                
                if not matches.empty and 'opponent_id' in matches.columns:
                    # Collect opponent IDs
                    year_opponent_ids = matches['opponent_id'].dropna().unique()
                    all_opponent_ids.update(year_opponent_ids)
                    logger.info(f"Found {len(matches)} matches and {len(year_opponent_ids)} unique opponents in {year}")
                else:
                    logger.info(f"No matches found for {year}")
            
            logger.info(f"Total unique opponents found: {len(all_opponent_ids)}")
            
            # 4. Process each opponent - get profile, rating history, and match history (all years)
            for idx, opponent_id in enumerate(all_opponent_ids):
                if opponent_id is None or opponent_id == "":
                    continue
                    
                logger.info(f"Processing opponent {idx+1}/{len(all_opponent_ids)}: ID {opponent_id}")
                
                # Get opponent profile
                opponent_profile = await scraper.get_player_profile(opponent_id)
                if opponent_profile:
                    opponent_name = opponent_profile['name']
                    logger.info(f"Got profile for {opponent_name}")
                    
                    # Get opponent rating history
                    opponent_ratings = await scraper.get_player_rating_history(opponent_id)
                    logger.info(f"Got {len(opponent_ratings)} rating history entries for {opponent_name}")
                    
                    # Get match history for all opponents, for all years
                    logger.info(f"Getting match history for opponent: {opponent_name}")
                    
                    # Get all years of matches for opponents too
                    for year in years:
                        logger.info(f"Getting {year} matches for {opponent_name}")
                        opponent_matches = await scraper.get_player_match_history(opponent_id, year=year, limit=200)
                        logger.info(f"Got {len(opponent_matches)} matches for {opponent_name} in {year}")
                        
                        # Add random delay between years
                        await asyncio.sleep(random.uniform(1.0, 2.0))
                else:
                    logger.warning(f"Could not get profile for opponent ID {opponent_id}")
                
                # Add delay between opponents
                await asyncio.sleep(random.uniform(1.0, 2.0))
            
            # 5. Run cross-reference to match UTRs with match dates
            logger.info("Running cross-reference to match UTRs with match dates")
            valid_matches = await scraper.cross_reference_matches_with_ratings(max_days_diff=30)
            
            logger.info(f"Cross-reference complete. Valid matches with UTR data: {len(valid_matches)}")
            
            # 6. Print some sample results for verification
            if not valid_matches.empty:
                logger.info("Sample of valid matches with historical UTR data:")
                sample = valid_matches.head(5)
                for _, match in sample.iterrows():
                    logger.info(f"Match: {match['player_name']} ({match['player_utr_at_match']}) vs {match['opponent_name']} ({match['opponent_utr_at_match']})")
                    logger.info(f"UTR Difference: {match['utr_diff']}, Match Date: {match['date']}")
                    logger.info(f"Score: {match['score']}")
                    logger.info("---")

    asyncio.run(main())