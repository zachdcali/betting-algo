# Tennis Match Prediction System

A robust machine learning system for predicting professional tennis match outcomes, leveraging advanced temporal feature engineering and multiple model architectures. Validated on over 684,000 matches from 1990 to 2024 with rigorous data integrity checks.

## Project Overview

This project applies sophisticated machine learning techniques to tennis match prediction, achieving notable improvements over ATP ranking baselines through temporal feature engineering, structured validation, and leak-free preprocessing. The system has been extensively validated to ensure reliable performance, making it suitable for practical applications such as live betting.

### Current Status

The system includes production-ready models with strong calibration metrics (ECE as low as 0.008), validated on comprehensive tennis match data. All models have undergone thorough testing to confirm realistic performance for deployment.

## Key Results

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