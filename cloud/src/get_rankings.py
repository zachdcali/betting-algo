#!/usr/bin/env python3
"""
Quick script to generate initial UTR rankings CSV file
Run this first to create the rankings file that finish_processing.py needs
"""

import asyncio
import os
from pathlib import Path
from utr_scraper import UTRScraper

async def generate_rankings():
    """Generate initial rankings CSV file"""
    print("Generating UTR rankings...")
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "data"
    data_dir.mkdir(exist_ok=True)
    
    email = "zachdodson12@gmail.com"
    password = "Thailand@123"
    
    async with UTRScraper(email=email, password=password, headless=True, data_dir=data_dir) as scraper:
        # Login first
        if not await scraper.login():
            print("Failed to login")
            return False
        
        # Get top 100 men's players
        top_players = await scraper.get_top_players(gender="men", limit=100)
        
        if top_players.empty:
            print("Failed to get top players")
            return False
        
        print(f"Successfully generated rankings with {len(top_players)} players")
        print("Sample players:")
        print(top_players.head())
        
        return True

if __name__ == "__main__":
    success = asyncio.run(generate_rankings())
    if success:
        print("\n✅ Rankings generated! Now you can run finish_processing.py")
    else:
        print("\n❌ Failed to generate rankings")