# UTR Scraping Architecture

## Overview

The UTR scraping system is designed as a multi-tier, fault-tolerant pipeline that systematically collects tennis match and rating data while respecting server constraints and maximizing data completeness.

## System Architecture

### Core Components

```
finish_processing.py (Orchestrator)
├── UTRScraper (utr_scraper_cloud.py)
│   ├── Playwright Browser Automation
│   ├── Async HTTP Requests  
│   ├── Data Parsing & Validation
│   └── File I/O Management
├── Multiprocessing Pool
├── Cross-Reference Engine
└── Analysis Pipeline
```

### Data Flow

```
UTR Website → Rankings CSV → Top 100 ATP Players
                                    ↓
                            Player Match History
                                    ↓
                           First-Degree Opponents 
                                    ↓
                          Second-Degree Opponents
                                    ↓
                          Cross-Reference Engine
                                    ↓
                            Enhanced Dataset
                                    ↓
                          Statistical Analysis
```

## Three-Tier Collection Strategy

### Tier 1: Top ATP Players (N=100)
- **Input**: UTR rankings CSV (downloaded or existing)
- **Collection**: Full match histories + rating histories
- **Output**: `player_{id}_matches.csv`, `player_{id}.json`, `player_{id}_ratings.csv`
- **Purpose**: Core dataset of elite players

### Tier 2: First-Degree Opponents (N=~15,000)
- **Input**: Opponent IDs from Tier 1 match files
- **Collection**: Full match histories + rating histories  
- **Output**: Same file structure as Tier 1
- **Purpose**: Expand dataset with high-quality opponents

### Tier 3: Second-Degree Opponents (N=~50,000)
- **Input**: Opponent IDs from Tier 2 match files
- **Collection**: Rating histories only (no matches)
- **Output**: `player_{id}_ratings.csv` only
- **Purpose**: Enable cross-referencing for maximum dataset completeness

## Cross-Reference Engine

### Purpose
Match historical UTR ratings with match dates to ensure both players have valid ratings at match time.

### Algorithm
1. **Load all match files** from Tiers 1 & 2
2. **Extract unique player pairs** with match dates
3. **Load rating histories** for all players (including Tier 3)
4. **Time-based matching**: Find closest rating ≤ match_date within ±30 days
5. **Validation**: Ensure both players have valid ratings
6. **Output**: Enhanced match dataset with UTR at match time

### Temporal Constraints
- **Primary**: Use most recent rating ≤ match_date
- **Fallback**: Allow ratings up to 30 days before match_date
- **Never**: Use post-match ratings (prevents data leakage)

## Fault Tolerance & Resume Logic

### File-Based State Management
- **Existence Checks**: Skip processing if output files already exist
- **Partial Completion**: Resume from any interruption point
- **Atomic Writes**: Ensure file completeness before moving to next step

### Error Recovery
```python
try:
    process_player(player_id)
except Exception:
    restart_browser()
    retry_with_backoff()
finally:
    cleanup_resources()
```

### Browser Management
- **Headless Operation**: Chromium with stealth configuration
- **Auto-Restart**: Fresh browser instance on errors
- **Connection Pooling**: Reuse connections when possible
- **Timeout Handling**: Graceful degradation on network issues

## Multiprocessing Architecture

### Process Pool Design
- **Configurable Size**: 2-8 processes (default: 4)
- **Worker Isolation**: Each process has independent browser
- **Batch Processing**: 50 opponents per batch to minimize overhead
- **Load Balancing**: Dynamic task distribution via `imap_unordered`

### Memory Management
- **Streaming I/O**: Process data without loading entire datasets
- **Process Cleanup**: Explicit browser closure and resource deallocation
- **Garbage Collection**: Periodic cleanup of large objects

### Inter-Process Communication
- **Shared State**: Minimal (only progress tracking)
- **File Coordination**: Atomic file operations prevent conflicts
- **Progress Reporting**: Real-time updates with completion percentages

## Data Schema

### Match Files (`player_{id}_matches.csv`)
```
match_id, date, opponent_id, opponent_name, result, score, 
tournament, surface, round, player_utr, opponent_utr
```

### Rating Files (`player_{id}_ratings.csv`)
```
date, utr_rating, singles_rating, doubles_rating, 
reliability, verified_results
```

### Player Files (`player_{id}.json`)
```json
{
  "id": "player_id",
  "name": "Player Name", 
  "location": "City, Country",
  "rankings": {...},
  "career_stats": {...}
}
```

## Performance Considerations

### Rate Limiting
- **Staggered Logins**: 5-second delays between processes
- **Request Spacing**: Natural delays from page rendering
- **Batch Sizing**: Optimal 50-player batches

### Scalability
- **Horizontal**: Add more processes (up to ~8 effective)
- **Vertical**: Faster network/CPU improves individual worker speed
- **Resume**: Can scale up/down between runs

### Monitoring
- **Progress Tracking**: Real-time percentage completion
- **Error Logging**: Comprehensive failure reporting
- **Performance Metrics**: Processing rates and bottleneck identification

## Configuration

### Environment Variables
```bash
UTR_EMAIL=credentials           # Required
UTR_PASSWORD=credentials        # Required  
UTR_PROCESSES=4                # Optional
UTR_MAX_DAYS_DIFF=30           # Optional
```

### Runtime Parameters
```bash
python finish_processing.py --processes 6
```

### Internal Configuration
- Cross-reference tolerance: ±30 days
- Batch sizes: 50 opponents per batch
- Browser timeout: 30 seconds per page
- Retry attempts: 3 per failed request

## Security & Ethics

### Data Protection
- **No Credential Storage**: Environment variables only
- **Minimal Data Retention**: Only necessary fields stored
- **Access Control**: Premium subscription required

### Rate Limiting & Respect
- **Conservative Rates**: Batch processing with natural delays
- **Error Backoff**: Exponential delays on repeated failures
- **Resource Cleanup**: Proper browser closure and connection cleanup

### Terms of Service Compliance
- **Personal Use**: Individual account access only
- **No Redistribution**: Raw data not included in repository
- **Attribution**: Proper citation of data source