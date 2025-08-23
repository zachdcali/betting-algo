# UTR Scraping Usage Guide

Simple instructions for running the UTR scraping pipeline and generating analysis reports.

## Quick Start

### 1. Setup Credentials
```bash
# Copy the environment template
cp .env.example .env

# Edit .env with your UTR credentials
UTR_EMAIL=your_email@example.com
UTR_PASSWORD=your_password

# Or set environment variables directly
export UTR_EMAIL=your_email@example.com
export UTR_PASSWORD=your_password
```

### 2. Install Dependencies
```bash
pip install playwright pandas asyncio
playwright install chromium
```

### 3. Run the Pipeline
```bash
# Run with default 4 processes
python finish_processing.py

# Run with custom process count (recommended: 2-8)
python finish_processing.py --processes 6
```

### 4. View Results

#### Option A: View Existing Sample Report (No Scraping Required)
```bash
# Open the committed sample analysis in your browser
open ../../analysis_output/utr_analysis_report.html

# Or regenerate from JSON data
python ../../scripts/demo_report.py
```

#### Option B: Generate Fresh Analysis From Your Data
```bash
# Create new analysis from your scraped data
python ../../utr_analysis_report.py
```

## Output Structure

After running the pipeline, you'll have:

```
../../data/
├── matches/           # Match histories: player_{id}_matches.csv
├── players/          # Player profiles: player_{id}.json  
└── ratings/          # Rating histories: player_{id}_ratings.csv

../../analysis_output/
├── utr_analysis_report.html    # Statistical analysis report
├── analysis_results.json       # Raw analysis data
└── plots/                      # Generated visualizations
```

## Requirements

- **UTR Premium subscription** required for full decimal ratings and complete rating histories
- **Environment variables**: `UTR_EMAIL` and `UTR_PASSWORD` must be set
- **Python 3.8+** with Playwright browser automation
- **Recommended**: 2-8 parallel processes depending on your system

## Troubleshooting

- **Browser Issues**: Run `playwright install chromium` if browser fails to start
- **Rate Limiting**: Reduce `--processes` count if you get connection errors
- **Resume After Interruption**: Just run the same command again - it automatically resumes
- **Missing Credentials**: Make sure `UTR_EMAIL` and `UTR_PASSWORD` are set in your environment