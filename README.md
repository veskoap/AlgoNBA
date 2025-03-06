# AlgoNBA: Enhanced NBA Game Prediction System

## Overview

AlgoNBA is a sophisticated machine learning system designed to predict the outcomes of NBA basketball games. It leverages multiple data sources and advanced algorithms to provide accurate win probabilities with confidence scores.

## New High-Accuracy Models

The system now includes enhanced prediction models targeting **70%+ accuracy** and **0.8+ confidence scores**:

- **Player Availability Analysis**: Incorporates player impact and availability for more nuanced predictions
- **Enhanced Ensemble Models**: Combines XGBoost and LightGBM with model calibration for higher accuracy
- **Advanced Deep Learning**: Implements residual networks and self-attention mechanisms
- **Monte Carlo Dropout**: Provides uncertainty estimates for more reliable confidence scores
- **Sophisticated Model Integration**: Uses dynamic weighting between models based on prediction strength

## Features

- **Comprehensive Data Processing**: Fetches historical game data from NBA API with advanced statistics
- **Advanced Feature Engineering**: Creates 100+ features including team performance metrics, head-to-head statistics, rest days, player impact, and travel data
- **Enhanced Machine Learning**: Combines multiple algorithms (XGBoost, LightGBM, deep learning) with stacking and calibration
- **Time-Series Validation**: Uses proper time-series cross-validation to prevent data leakage
- **Improved Confidence Scoring**: Provides robust confidence metrics for each prediction
- **Multiple Models**: Supports ensemble, deep learning, and hybrid prediction approaches

## Project Structure

The project is organized into the following modules:

```
src/
├── __init__.py
├── data/
│   ├── __init__.py
│   └── data_loader.py             # Data fetching from NBA API
├── features/
│   ├── __init__.py
│   ├── feature_processor.py       # Feature engineering
│   └── advanced/
│       ├── __init__.py
│       └── player_availability.py # Player impact features
├── models/
│   ├── __init__.py
│   ├── deep_model.py              # Standard deep learning models
│   ├── enhanced_deep_model.py     # Enhanced neural networks
│   ├── ensemble_model.py          # Standard XGBoost ensemble models
│   ├── enhanced_ensemble.py       # Enhanced ensemble with multiple algorithms
│   └── hybrid_model.py            # Advanced model integration
├── utils/
│   ├── __init__.py
│   ├── constants.py               # Shared constants
│   ├── helpers.py                 # Utility functions
│   └── scaling/
│       └── enhanced_scaler.py     # Robust feature scaling
└── predictor.py                   # Main predictor class
main.py                            # Entry point
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
# Use enhanced models (default)
python main.py

# Use standard models
python main.py --standard

# Specify seasons to use for training
python main.py --seasons 2021-22 2022-23

# Use quick mode for faster testing with simplified models
python main.py --quick

# Save trained models to disk for later use
python main.py --save-models

# Load previously saved models (skips training)
python main.py --load-models saved_models/nba_model_20230401_120000
```

The `--quick` flag enables a faster testing mode that:
- Uses fewer cross-validation folds (2 instead of 5)
- Uses simplified model architectures
- Runs fewer training epochs
- Performs less hyperparameter optimization

This mode is useful for development and testing purposes, but for the highest accuracy, run without the quick flag.

### Model Persistence

You can save trained models to disk using the `--save-models` flag. Models will be saved to the `saved_models` directory with a timestamp, allowing you to load them later without retraining.

When running on resource-constrained environments like Google Colab, it's a good practice to:
1. Train models with the `--save-models` flag
2. Save the models to Google Drive to preserve them between sessions
3. Load the saved models with `--load-models` for making predictions

Example workflow:
```python
# In Google Colab
from google.colab import drive
drive.mount('/content/drive')

# Clone repository and install requirements
!git clone https://github.com/yourusername/AlgoNBA.git
%cd AlgoNBA
!pip install -r requirements.txt

# Train models and save them to Google Drive
!python main.py --quick --save-models
!mkdir -p /content/drive/MyDrive/AlgoNBA/models/
!cp -r saved_models/* /content/drive/MyDrive/AlgoNBA/models/

# In a later session, load models from Google Drive
!mkdir -p saved_models
!cp -r /content/drive/MyDrive/AlgoNBA/models/* saved_models/
!python main.py --load-models saved_models/nba_model_20230401_120000
```

## Example Code

```python
from src.predictor import EnhancedNBAPredictor

# Initialize the predictor with enhanced models
seasons = ['2022-23', '2023-24']  # Recent seasons for better predictions
predictor = EnhancedNBAPredictor(
    seasons=seasons,
    use_enhanced_models=True,  # Use enhanced models for higher accuracy
    quick_mode=False  # Set to True for faster testing/development
)

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
print(f"Home team: {prediction['home_team']} vs Away team: {prediction['away_team']}")
print(f"Home win probability: {prediction['home_win_probability']:.2f}")
print(f"Confidence: {prediction['confidence']:.2f}")

# Make another prediction with a different matchup
prediction2 = predictor.predict_game(
    home_team_id=1610612747,  # LAL
    away_team_id=1610612744,  # GSW
    model_type='hybrid'
)

print(f"Home team: {prediction2['home_team']} vs Away team: {prediction2['away_team']}")
print(f"Home win probability: {prediction2['home_win_probability']:.2f}")
print(f"Confidence: {prediction2['confidence']:.2f}")
```

## Model Details

### New Features

The enhanced system includes additional features:
- Player availability impact scores
- Team strength adjustments based on available players
- Player impact momentum features
- Feature stability scores for confidence calculation
- Model consensus metrics

### Enhanced Models

1. **Enhanced Ensemble Model**:
   - Uses both XGBoost and LightGBM with optimized hyperparameters
   - Implements model stacking and probability calibration
   - Features improved confidence scoring with uncertainty quantification
   - Includes feature stability analysis for more robust feature selection
   - Supports configurable cross-validation folds for speed/accuracy tradeoffs

2. **Enhanced Deep Learning Model**:
   - Implements residual connections for improved gradient flow
   - Uses self-attention mechanisms to capture feature relationships
   - Applies Monte Carlo dropout for uncertainty estimation
   - Features learning rate scheduling with cosine annealing
   - Supports configurable network architecture and training parameters
   - Includes early stopping and gradient clipping for stable training

3. **Advanced Hybrid Model**:
   - Uses dynamic weighting between models based on prediction strengths
   - Implements meta-learning to optimize model integration
   - Provides unified confidence scores that account for model uncertainty
   - Offers streamlined quick mode for rapid development and testing
   - Features improved robustness through model consensus metrics

4. **Performance Optimizations**:
   - Enhanced data scaling with robust handling of extreme values
   - Improved feature alignment with optimized memory usage
   - Quick mode for faster testing and development iterations
   - Configurable training parameters for different use cases
   - Comprehensive error handling and fallback mechanisms for production reliability

## Requirements

- Python 3.9+
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- pytorch
- nba_api
- geopy
- pytz
- joblib (for model serialization)
- tqdm (for progress tracking)

## License

MIT License - see LICENSE file for details.

---

*Note: This project is for educational purposes only. It is not affiliated with the NBA or any professional basketball organization.*