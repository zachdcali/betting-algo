# Tennis Data Scrapers

Automated data collection utilities for gathering tennis match and player information from professional tennis platforms.

## Overview

The scraping infrastructure supports systematic data collection to enhance match prediction models. Current focus is on Universal Tennis Rating (UTR) data integration, which preliminary analysis indicates provides superior predictive power compared to ATP rankings alone.

## UTR Scraper

Professional tennis player rating and match data collection from UTR Sports.

### Data Collection Strategy

The scraper implements a systematic approach to gather comprehensive rating and match data:

1. **Top-100 ATP Players**: Complete match histories and UTR rating progressions
2. **First-degree opponents**: All players who have faced top-100 players (match and rating histories)
3. **Second-degree opponents**: Rating histories only (for temporal rating interpolation)

This multi-degree collection ensures accurate historical rating reconstruction for temporal feature engineering.

### Technical Features
- Player profile scraping (ratings, biographical info)
- Match history collection with temporal ordering
- Rating timeline tracking and interpolation
- Tournament result aggregation
- Robust retry mechanisms with exponential backoff
- Comprehensive error handling and logging

### Configuration

Set credentials via environment variables:
```bash
export UTR_EMAIL="your-email@example.com"
export UTR_PASSWORD="your-password"
```

### Usage

```python
from utr_scraper import UTRScraper

async def main():
    async with UTRScraper(email=email, password=password) as scraper:
        # Login to UTR platform
        await scraper.login()
        
        # Get player profile
        profile = await scraper.get_player_profile(player_id)
        
        # Get match history
        matches = await scraper.get_player_matches(player_id)
        
        # Get rating history
        ratings = await scraper.get_rating_history(player_id)
```

### Data Output

The scraper generates structured CSV files:
- `player_mapping.csv`: Player ID to name mappings
- `tournament_locations.csv`: Tournament location data
- `players/`: Individual player profiles and statistics
- `matches/`: Match results and details
- `ratings/`: Historical rating progressions

### Technical Implementation

- **Async/await architecture** for concurrent data collection
- **Playwright browser automation** with stealth measures
- **Exponential backoff retry logic** for robust data collection
- **Rate limiting** to respect website terms
- **Comprehensive logging** for monitoring and debugging
- **Data validation** and error recovery mechanisms

### Requirements

```bash
pip install playwright pandas asyncio
playwright install chromium
```

### Rationale

UTR provides more granular and frequently updated player ratings compared to weekly ATP rankings. The lack of bulk public APIs necessitates systematic scraping for research purposes. Preliminary analysis suggests UTR-based features improve match prediction accuracy by approximately 1% over ATP rankings alone.

Note: This scraper requires valid UTR Sports credentials and should be used in accordance with the platform's terms of service.