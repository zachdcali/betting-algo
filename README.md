# Tennis Match Prediction System

A comprehensive system for predicting tennis match outcomes and analyzing betting opportunities using various machine learning models.

## Models

### Baseline Heuristic Models
- Rank difference logistic regression
- Points difference logistic regression
- Surface-specific models

### Machine Learning Models
- Random Forest
- XGBoost
- Neural Network (2 hidden layers: 32 and 16 neurons)

## Data Files

Due to GitHub file size limitations, only sample files are included in this repository. Full data files can be obtained separately. The samples include:

- `valid_matches_sample.csv`: Sample of matches used for training models
- `processed_atp_sample.csv`: Sample of processed ATP match data
- `master_jeffsackmann_sample.csv`: Sample of combined Jeff Sackmann tennis data

Contact the repository owner to obtain the full datasets.

## Model Features

The models use a variety of features including:
- Basic match data (rankings, points, surface)
- Head-to-head statistics
- Recent form metrics
- Fatigue indicators
- Tournament information

## Output

Models produce:
- Win probability predictions
- Feature importance analysis
- Performance metrics
- Betting simulation results

Each model writes its output to a dedicated directory under `/data/output/`.