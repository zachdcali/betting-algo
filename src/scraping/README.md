# UTR Tennis Data Scraping

> Production-ready Playwright scraper + cross-referencing pipeline for UTR.  
> **Sample run:** 378,401 matches · Overall accuracy 72.9% (95% CI via Wilson)

**Docs:** [Usage](./USAGE.md) · [Architecture](./ARCHITECTURE.md) · [Full Results](./RESULTS.md)  
**Sample Report:** [`analysis_output/utr_analysis_report.html`](./analysis_output/utr_analysis_report.html)

## Quick Start

```bash
# Setup credentials
cp .env.example .env
# Edit .env with your UTR_EMAIL and UTR_PASSWORD

# Install dependencies
pip install playwright pandas asyncio
playwright install chromium

# Run scraper (adjust processes as needed)
python finish_processing.py --processes 6

# View sample analysis (no private data required)
python ./scripts/demo_report.py
```

## Data Access

We **don't** commit scraped data. A **UTR Premium** account is required to reproduce the full pipeline.  
We ship a **static sample report** and plots so you can see expected outputs without data access.

## Core Files

- **`finish_processing.py`** - Complete pipeline orchestrator with multiprocessing
- **`utr_scraper_cloud.py`** - Main scraper class with Playwright automation
- **`scripts/demo_report.py`** - Generate sample report from bundled data
- **`archive/`** - Legacy scraper versions (deprecated)

## Three-Tier Collection Strategy

1. **ATP Rankings** - Downloads top 100 men's rankings from UTR
2. **Top Players** - Complete match + rating histories for all top 100  
3. **First-Degree Opponents** - Match + rating data for ~15,000 opponents
4. **Second-Degree Opponents** - Rating histories only for ~50,000+ players

Cross-references all matches with historical UTR ratings (±30 days tolerance) to ensure both players have valid ratings at match time.

## Key Features

- **Resume capability** - Automatically resumes where interrupted
- **Multi-processing** - Configurable process count for optimal performance  
- **Error recovery** - Browser restarts and retry logic on failures
- **Batch processing** - Groups requests to minimize server load
- **Data quality** - Enforces rating_date ≤ match_date to prevent leakage

## Requirements

- **UTR Premium subscription** required for full decimal ratings
- **Python 3.8+** with Playwright browser automation
- **Environment variables:** `UTR_EMAIL` and `UTR_PASSWORD`

## Known Limitations

- Requires premium UTR account access
- Rate-limited by UTR site capacity  
- Browser automation may occasionally fail (auto-restarts)
- Large datasets require significant processing time

## License & Disclaimer

For personal research use only. Respect UTR's terms of service. No redistribution of scraped data.