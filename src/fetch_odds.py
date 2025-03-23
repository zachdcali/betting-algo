from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import os
import re

def fetch_bovada_tennis_odds():
    url = "https://www.bovada.lv/sports/tennis"
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url)
        page.wait_for_selector('sp-coupon', timeout=15000)
        
        # Take a screenshot to help with debugging (optional)
        page.screenshot(path="bovada_tennis_page.png")
        
        content = page.content()
        browser.close()

    soup = BeautifulSoup(content, 'html.parser')
    upcoming_matches = []
    live_matches = []

    # Find all event groups
    events = soup.select('div.grouped-events')
    
    # Debug: Check if we're finding events
    print(f"Found {len(events)} event groups")

    for event in events:
        event_title_tag = event.select_one('h4.header-collapsible')
        event_title = event_title_tag.get_text(strip=True) if event_title_tag else "Unknown Event"
        
        games = event.select('sp-coupon')
        print(f"Found {len(games)} games in event: {event_title}")
        
        for game_index, game in enumerate(games):
            # Try multiple selectors for date and time
            # Check if match is live
            is_live = bool(game.select_one('.live-indicator'))
            
            # Try different selectors for match time
            match_time = "Unknown"
            
            # Try to find elements with specific text patterns like a date or time
            time_patterns = [
                # Sample date format: 3/20/25
                r"\d{1,2}/\d{1,2}/\d{2}",  
                # Sample time format: 8:00 AM, 12:30 PM, etc.
                r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)"
            ]
            
            # Extract all text from the game element
            all_text = game.get_text()
            
            # Check for date and time patterns in all text
            date_match = re.search(r"\d{1,2}/\d{1,2}/\d{2}", all_text)
            time_match = re.search(r"\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)", all_text)
            
            if date_match and time_match:
                match_time = f"{date_match.group(0)} {time_match.group(0)}"
            elif date_match:
                match_time = date_match.group(0)
            elif time_match:
                match_time = time_match.group(0)
            
            # Extract player names and odds
            players = game.select('span.name')
            odds = game.select('span.bet-price')
            
            if len(players) == 2 and len(odds) >= 2:
                match_info = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'event': event_title,
                    'player1': players[0].get_text(strip=True),
                    'player2': players[1].get_text(strip=True),
                    'player1_odds': odds[0].get_text(strip=True),
                    'player2_odds': odds[1].get_text(strip=True),
                    'match_time': match_time,
                    'is_live': is_live
                }
                
                # Debug specific matches
                if game_index < 2:  # Just print info for the first two matches in each event
                    print(f"Match: {players[0].get_text(strip=True)} vs {players[1].get_text(strip=True)}")
                    print(f"  Time found: {match_time}")
                    print(f"  Live: {is_live}")
                
                # Add to appropriate list
                if is_live:
                    live_matches.append(match_info)
                else:
                    upcoming_matches.append(match_info)

    # Save data to CSV files
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)
    odds_dir = os.path.join(base_dir, "data", "odds_data")
    os.makedirs(odds_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M')
    
    # Save upcoming matches
    if upcoming_matches:
        upcoming_df = pd.DataFrame(upcoming_matches)
        upcoming_filename = os.path.join(odds_dir, f"bovada_upcoming_tennis_{timestamp}.csv")
        upcoming_df.to_csv(upcoming_filename, index=False)
        print(f"Found {len(upcoming_matches)} upcoming matches:")
        print(upcoming_df.head())
        print(f"Upcoming matches saved to: {upcoming_filename}")
    else:
        print("No upcoming matches found")
    
    # Save live matches
    if live_matches:
        live_df = pd.DataFrame(live_matches)
        live_filename = os.path.join(odds_dir, f"bovada_live_tennis_{timestamp}.csv")
        live_df.to_csv(live_filename, index=False)
        print(f"Found {len(live_matches)} live matches (not for betting)")
        print(f"Live matches saved to: {live_filename}")
    
    # Also save all matches combined
    all_matches = upcoming_matches + live_matches
    if all_matches:
        all_df = pd.DataFrame(all_matches)
        all_filename = os.path.join(odds_dir, f"bovada_all_tennis_{timestamp}.csv")
        all_df.to_csv(all_filename, index=False)
        print(f"All {len(all_matches)} matches saved to: {all_filename}")
    
    return upcoming_df if upcoming_matches else None

if __name__ == "__main__":
    fetch_bovada_tennis_odds()