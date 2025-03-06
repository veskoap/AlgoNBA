# AlgoNBA: Enhanced NBA Game Prediction System

## Overview

AlgoNBA is a sophisticated machine learning system designed to predict the outcomes of NBA basketball games. It leverages multiple data sources and advanced algorithms to provide accurate win probabilities with confidence scores.

## Features

- **Comprehensive Data Processing**: Fetches historical game data from NBA API with advanced statistics
- **Advanced Feature Engineering**: Creates 100+ features including team performance metrics, head-to-head statistics, rest days, travel impact, and more
- **Ensemble Machine Learning**: Combines gradient boosting (XGBoost) and deep neural networks for robust predictions
- **Time-Series Validation**: Uses proper time-series cross-validation to prevent data leakage
- **Confidence Scoring**: Provides confidence metrics for each prediction
- **Multiple Models**: Supports ensemble, deep learning, and hybrid prediction approaches

## Project Structure

The project is organized into the following modules:

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── data_loader.py          # Data fetching from NBA API
├── features/
│   ├── __init__.py
│   └── feature_processor.py    # Feature engineering 
├── models/
│   ├── __init__.py
│   ├── deep_model.py           # Deep learning models
│   └── ensemble_model.py       # XGBoost ensemble models
├── utils/
│   ├── __init__.py
│   ├── constants.py            # Shared constants
│   └── helpers.py              # Utility functions
└── predictor.py                # Main predictor class
main.py                         # Entry point
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AlgoNBA.git
cd AlgoNBA
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script to train the model and make predictions:

```bash
python main.py
```

## Example Code

```python
from src.predictor import EnhancedNBAPredictor

# Initialize the predictor with seasons to include
seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
predictor = EnhancedNBAPredictor(seasons)

# Fetch and process data
predictor.fetch_and_process_data()

# Train models
predictor.train_models()

# Make a prediction for a specific game
prediction = predictor.predict_game(
    home_team_id=1610612738,  # BOS
    away_team_id=1610612749,  # MIL
    model_type='hybrid'  # Options: 'ensemble', 'deep', or 'hybrid'
)

# Print prediction results
print(f"Home win probability: {prediction['home_win_probability']:.2f}")
print(f"Confidence: {prediction['confidence']:.2f}")
```

## Model Details

### Features

The system generates features including:
- Win percentages over various time windows
- Offensive/defensive ratings and differentials
- Rest day advantages
- Travel distance and timezone changes
- Head-to-head historical performance
- Team consistency metrics
- Momentum indicators

### Models

1. **Ensemble Model**: Uses XGBoost with time-window specific models combined for optimal performance
2. **Deep Learning Model**: Multi-layer neural network with batch normalization and dropout
3. **Hybrid Model**: Combines predictions from both models

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- pytorch
- nba_api
- geopy
- pytz

## License

MIT License - see LICENSE file for details.

---

*Note: This project is for educational purposes only. It is not affiliated with the NBA or any professional basketball organization.*