"""
Main entry point for the NBA Prediction System.
"""
import sys
import pandas as pd
from src.predictor import EnhancedNBAPredictor


def main():
    """
    Initialize and run the NBA prediction model.
    """
    print("Starting NBA prediction system...")
    
    # Use seasons from 2020-21 to 2023-24
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    
    # Initialize the predictor
    predictor = EnhancedNBAPredictor(seasons)
    
    try:
        # Fetch and process data
        print("Fetching and processing NBA game data...")
        predictor.fetch_and_process_data()
        
        # Train models
        print("Training prediction models...")
        predictor.train_models()
        
        # Print top features
        top_features = predictor.get_feature_importances(10)
        print("\nTop 10 most important features:")
        for feature, importance in top_features.items():
            print(f"- {feature}: {importance:.4f}")
        
        # Example prediction
        print("\nPrediction example:")
        
        # Boston Celtics vs Milwaukee Bucks
        prediction = predictor.predict_game(
            home_team_id=1610612738,  # BOS
            away_team_id=1610612749,  # MIL
            model_type='hybrid'
        )
        
        print(f"Boston Celtics vs Milwaukee Bucks:")
        print(f"Home win probability: {prediction['home_win_probability']:.2f}")
        print(f"Confidence: {prediction['confidence']:.2f}")
        
        print("\nNBA prediction system ready!")
        
    except Exception as e:
        print(f"Error during model training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
