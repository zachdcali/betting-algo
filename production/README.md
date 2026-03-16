# Production Pipeline

Internal documentation for the live betting pipeline. Not intended for public consumption.

## Files

```
main.py                  # End-to-end orchestrator — run this
odds/
  fetch_bovada.py        # Scrapes upcoming ATP/Challenger matches and moneyline odds
features/
  ta_feature_calculator.py  # Builds all 141 features from Tennis Abstract data
models/
  inference.py           # Loads NN model, runs predictions, exposes EXACT_141_FEATURES list
utils/
  stake_calculator.py    # Fractional Kelly criterion
  bet_tracker.py         # Session and bet logging
tournaments/
  resolve_tournament.py  # Maps Bovada event names → surface/level/draw_size/round
scraping/
  ta_scraper.py          # Tennis Abstract HTTP scraper with session caching and rate limiting
prediction_logger.py     # Appends every prediction to prediction_log.csv
settle_predictions.py    # Run after matches complete to record outcomes and compute accuracy
tests/
  test_system.py         # Integration smoke tests
```

## Running

```bash
cd production
python main.py
```

## Prediction Logging

Every pipeline run appends to `prediction_log.csv`. After matches complete:

```bash
python settle_predictions.py
```

This records actual winners and computes live model vs market accuracy over time.

## Model

Active model: `results/professional_tennis/Neural_Network/neural_network_model_UNBIASED_TEMPORAL.pth`
Scaler: `results/professional_tennis/Neural_Network/scaler_UNBIASED_TEMPORAL.pkl`
Features: 141 (defined in `models/inference.py` → `EXACT_141_FEATURES`)

Backups stored in `results/*/backups/` with date suffix.

## Known Limitations

- `Player1_Rank_Points` / `Player2_Rank_Points` hardcoded to 500 (ATP points not yet scraped)
- Height missing for some lower-ranked players on Tennis Abstract — those matches are skipped
- Career-accumulated features can overweight declined veterans vs their current form
