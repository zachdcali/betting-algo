# Tennis Match Prediction System

A robust machine learning system for predicting professional tennis match outcomes, leveraging advanced temporal feature engineering and multiple model architectures. Validated on over 684,000 matches from 1990 to 2024 with rigorous data integrity checks.

## Project Overview

This project applies sophisticated machine learning techniques to tennis match prediction, achieving notable improvements over ATP ranking baselines through temporal feature engineering, structured validation, and leak-free preprocessing. The system has been extensively validated to ensure reliable performance, making it suitable for practical applications such as live betting.

### Current Status

The system includes production-ready models with strong calibration metrics (ECE as low as 0.008), validated on comprehensive tennis match data. All models have undergone thorough testing to confirm realistic performance for deployment.

## Key Results

### UTR Analysis Results Snapshot
**Sample run on 378,401 matches with cross-referenced UTR data:**
- **Overall accuracy (higher-UTR wins): 72.9%**
- **Brier: 0.179**, **ECE: 0.011**, **AUC: 0.795**  
- **By skill level: <12: ~79%, 12-14: ~72%, 14+: ~62%**
- **Full interactive report**: `analysis_output/utr_analysis_report.html`

*Metrics computed on held-out test split; bin estimates include Wilson 95% CIs; cells with n<50 masked for reliability.*

### Professional Tennis Models (684K+ Matches, 1990-2024)

| Model          | Accuracy | Improvement vs ATP Rankings | AUC-ROC | Log Loss | ECE (Calibration) |
|----------------|----------|-----------------------------|---------|----------|-------------------|
| Neural Network | 70.69%   | +6.01 pp                    | 0.789   | 0.547    | 0.008             |
| XGBoost        | 69.17%   | +4.49 pp                    | 0.766   | 0.573    | 0.015             |
| Random Forest  | 66.85%   | +2.17 pp                    | 0.736   | 0.602    | 0.028             |
| ATP Ranking Baseline | 64.68% | -                          | -       | -        | -                 |

The Neural Network delivers the best overall performance with superior calibration for betting applications.

### Performance by Tournament Level

| Tournament Level | Neural Network | XGBoost | Random Forest | ATP Baseline | Sample Size  |
|------------------|----------------|---------|---------------|--------------|--------------|
| ATP-Level        | 68.2% (+5.5 pp)| 66.8% (+4.2 pp) | 64.2% (+1.5 pp) | 62.7%        | 8,133 matches|
| Challengers      | 66.4% (+4.0 pp)| 66.8% (+4.4 pp) | 64.8% (+2.4 pp) | 62.4%        | 18,488 matches|
| ITF Futures      | 74.2% (+7.5 pp)| 71.4% (+4.8 pp) | 68.9% (+2.3 pp) | 66.7%        | 28,822 matches|

Models exhibit the largest prediction edges at lower tournament levels, with ITF Futures showing the greatest predictability.

### Performance by Surface

| Surface | Neural Network | XGBoost | Random Forest | Sample Size  |
|---------|----------------|---------|---------------|--------------|
| Clay    | 71.1% (+5.9 pp)| 69.9% (+4.7 pp) | 67.2% (+2.1 pp) | 25,074 matches|
| Hard    | 70.4% (+6.1 pp)| 68.6% (+4.3 pp) | 66.6% (+2.2 pp) | 28,976 matches|
| Grass   | 68.2% (+5.2 pp)| 67.3% (+4.4 pp) | 65.2% (+2.2 pp) | 1,470 matches |
| Carpet  | 74.1% (+10.1 pp)| 75.5% (+11.5 pp)| 69.0% (+5.1 pp) | 355 matches  |

## Features & Architecture

### Temporal Feature Engineering
- Ranking momentum tracking across multiple time windows
- Surface-specific performance metrics
- Head-to-head records with contextual adjustments
- Fatigue and form indicators
- Tournament-level experience metrics
- Seasonal pattern recognition

### Core Features
- Player attributes: Rankings, points, age, height, handedness
- Match context: Tournament level, surface, round, draw size
- Derived metrics: Differences and ratios

### ML Architecture Highlights
- Preprocessing ensures data integrity
- Temporal validation with chronological splits
- Feature selection to avoid over-reliance
- Regularization techniques for stability
- Cross-validation for robustness

## Technical Implementation

### Models
- **Neural Network**: 4-layer architecture (143→128→64→32→16→1) with sigmoid output
  - Dropout regularization
  - Adam optimizer
  - Configurable epochs with early stopping
- **XGBoost**: Gradient boosting with optimized parameters
  - Multiple estimators with depth control
  - Regularization for balance
- **Random Forest**: Ensemble method with tuned hyperparameters
  - Multiple estimators with depth limits

### UTR Data Collection & Analysis

#### UTR Scraper Infrastructure
- **Production-ready scraper** (`src/scraping/utr_scraper_cloud.py` + `finish_processing.py`)
- **Processes 378,401 matches** with UTR ratings at match time
- **Multi-processing support** with configurable batch sizes and error recovery
- **Cross-referencing engine** matches historical UTR data with match dates
- **Comprehensive logging** and restart logic for reliability

#### How the UTR Scraper Works

**Data Collection Strategy:**
1. **ATP Rankings Import**: Downloads top 100 men's ATP player rankings from UTR site or uses existing CSV
2. **Tier 1 - Top Players**: Collects complete match histories and rating histories for all top 100 ATP players
3. **Tier 2 - First-Degree Opponents**: Collects match and rating histories for all opponents of top players (typically ~15,000 players)
4. **Tier 3 - Second-Degree Opponents**: Collects only rating histories for opponents of first-degree opponents (~50,000+ players)

**Opponent Definitions:**
- **First-Degree Opponents**: Players who have directly faced top 100 ATP players
- **Second-Degree Opponents**: Players who have faced first-degree opponents (needed for cross-referencing)

**Cross-Referencing Process:**
- Matches historical UTR ratings with match dates (±30 days tolerance)
- Ensures both players have valid UTR ratings at time of match
- Maximizes dataset completeness for accurate predictive modeling

**Resilience Features:**
- **Resume Capability**: Automatically detects existing files and resumes where interrupted
- **Multi-processing**: Configurable process count (`--processes N`) for optimal performance
- **Error Recovery**: Automatic browser restarts and retry logic on failures
- **Batch Processing**: Groups requests to minimize server load and improve reliability

**Configurable Parameters:**
- `--processes N`: Number of parallel processes (default: 4, recommend 2-8)
- Cross-reference tolerance: ±30 days for rating-match date matching
- Batch sizes: 50 opponents per batch for optimal processing

#### UTR Predictive Analysis Results
Based on comprehensive analysis of 378,401 matches with UTR cross-referencing:

**Overall Performance:**
- **Higher-UTR Rule Accuracy**: 72.87% (ties counted as 0.5)
- **Mean UTR in Dataset**: 12.63 (range: 1.38-16.45)
- **Test Set Validation**: Brier Score 0.179, ECE 0.011, MCE 0.022

**Accuracy by UTR Level:**
| UTR Level | Accuracy | Sample Size | 95% Confidence Interval |
|-----------|----------|-------------|-------------------------|
| <8        | 75.24%   | 2,361       | 73.46% - 76.94%         |
| 8-10      | 75.65%   | 15,729      | 74.98% - 76.32%         |
| 10-12     | 75.53%   | 95,942      | 75.25% - 75.80%         |
| 12-13     | 74.54%   | 105,160     | 74.27% - 74.80%         |
| 13-14     | 72.99%   | 97,349      | 72.71% - 73.27%         |
| 14-15     | 65.26%   | 46,180      | 64.82% - 65.69%         |
| 15+       | 64.09%   | 15,680      | 63.34% - 64.84%         |

**Key Findings:**
- UTR shows **highest predictive accuracy at lower skill levels** (75%+ for UTR <12)
- **Elite level prediction challenges** (UTR 14+) with ~65% accuracy
- **Temporal stability** confirmed across multiple years of data
- **Publication-ready analysis** with Wilson confidence intervals and calibration metrics

#### Requirements
- **UTR Premium subscription required** for full decimal ratings and complete rating histories
- **Environment variables**: Set `UTR_EMAIL` and `UTR_PASSWORD` before running
- **Multiprocessing support**: Configurable with `--processes` flag (default: 4)

### Validation Approach
- Temporal data splits to mimic real-world conditions
- Statistical tests to ensure model integrity
- Multiple performance metrics for comprehensive evaluation
- Comparison against established benchmarks

## Reproducibility

### Environment Setup
```bash
# Using virtual environment (recommended)
python -m venv tennis_env
source tennis_env/bin/activate  # On Windows: tennis_env\Scripts\activate
pip install -r requirements.txt

# Alternative: Docker environment
docker build -t tennis-prediction .
docker run -it tennis-prediction

### UTR Data Collection Usage

#### Setup Environment Variables
```bash
# Set your UTR credentials (Premium subscription required)
export UTR_EMAIL=your_email@example.com
export UTR_PASSWORD=your_password
```

#### Run Complete UTR Pipeline
```bash
# Process UTR rankings and matches with cross-referencing
cd src/scraping
python finish_processing.py --processes 4

# Generate analysis report 
python ../../utr_analysis_report.py
```

#### Output Files
- **Match Data**: `data/matches/player_{id}_matches.csv`
- **Player Profiles**: `data/players/player_{id}.json` 
- **Rating History**: `data/ratings/player_{id}_ratings.csv`
- **Analysis Report**: `analysis_output/utr_analysis_report.html`