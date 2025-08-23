# UTR Tennis Data Scraping Infrastructure

Production-ready scraping system for collecting Universal Tennis Rating (UTR) data and cross-referencing with match histories.

## Overview

This scraping infrastructure systematically collects tennis match and rating data using a three-tier approach to maximize dataset completeness for predictive modeling. Successfully processed **378,401 matches** with UTR ratings at match time.

## Core Files

- **`utr_scraper_cloud.py`**: Main scraper class with Playwright browser automation
- **`finish_processing.py`**: Complete pipeline orchestrator with multiprocessing
- **`archive/`**: Legacy scraper versions (deprecated)

## How It Works

### Data Collection Strategy
1. **ATP Rankings Import**: Downloads top 100 men's ATP player rankings from UTR site
2. **Tier 1 - Top Players**: Collects complete match and rating histories for all top 100 ATP players  
3. **Tier 2 - First-Degree Opponents**: Collects match and rating histories for all opponents of top players (~15,000 players)
4. **Tier 3 - Second-Degree Opponents**: Collects rating histories for opponents of first-degree opponents (~50,000+ players)

### Cross-Referencing Engine
- Matches historical UTR ratings with match dates (±30 days tolerance)
- Ensures both players have valid UTR ratings at time of match
- Maximizes dataset completeness for accurate predictive modeling

### Resilience Features
- **Resume Capability**: Automatically detects existing files and resumes where interrupted
- **Multi-processing**: Configurable process count for optimal performance
- **Error Recovery**: Automatic browser restarts and retry logic on failures
- **Batch Processing**: Groups requests to minimize server load

## Requirements

- **UTR Premium subscription** required for full decimal ratings and complete rating histories
- Python 3.8+ with Playwright browser automation
- Environment variables: `UTR_EMAIL` and `UTR_PASSWORD`

## Usage

### Setup Environment
```bash
# Set your UTR credentials
export UTR_EMAIL=your_email@example.com
export UTR_PASSWORD=your_password

# Install dependencies
pip install playwright pandas asyncio
playwright install chromium
```

### Run Complete Pipeline
```bash
# Run with default 4 processes
python finish_processing.py

# Run with custom process count (recommended: 2-8)
python finish_processing.py --processes 6
```

### Generate Analysis Report
```bash
# Create comprehensive statistical analysis
python ../../utr_analysis_report.py
```

## Output Structure

```
data/
├── matches/           # Match histories: player_{id}_matches.csv
├── players/          # Player profiles: player_{id}.json
└── ratings/          # Rating histories: player_{id}_ratings.csv

analysis_output/
├── utr_analysis_report.html    # Statistical analysis report
├── analysis_results.json       # Raw analysis data
└── plots/                      # Generated visualizations
```

## Key Results

- **Overall UTR Prediction Accuracy**: 72.87%
- **Highest Accuracy at Lower Levels**: 75%+ for UTR <12
- **Elite Level Challenges**: ~65% accuracy for UTR 14+
- **Temporal Stability**: Consistent across multiple years
- **Publication-Ready**: Wilson confidence intervals, calibration metrics

## Configuration

### Command Line Options
- `--processes N`: Number of parallel processes (default: 4)

### Internal Parameters
- Cross-reference tolerance: ±30 days for rating-match date matching
- Batch sizes: 50 opponents per batch for optimal processing
- Browser: Headless Chromium with stealth configuration

## Monitoring

The scraper provides comprehensive logging:
- Progress tracking with percentage completion
- Error reporting with automatic recovery
- File existence checks to avoid duplicate work
- Cross-reference statistics and validation

## Architecture Notes

- **Asynchronous Processing**: Uses asyncio for efficient I/O operations
- **Multiprocessing**: Parallel workers for scalable data collection  
- **Error Handling**: Graceful browser restarts and retry logic
- **Data Integrity**: Validates file completeness before processing
- **Memory Efficient**: Streams large datasets without loading into memory