"""
Main entry point for the NBA Prediction System.

This script provides a command-line interface to the NBA prediction system, 
allowing users to train models and generate predictions with various options:

- Enhanced vs standard models: Control accuracy vs complexity tradeoff
- Season selection: Choose which NBA seasons to use for training
- Quick mode: Faster execution for testing and development
- Save/load models: Save trained models to disk and load them for later use

Example usage:
    # Train with enhanced models (default)
    python main.py
    
    # Use standard models instead
    python main.py --standard
    
    # Specify specific seasons
    python main.py --seasons 2022-23 2023-24
    
    # Run in quick mode for faster testing
    python main.py --quick
    
    # Save trained models to disk
    python main.py --save-models
    
    # Load previously saved models
    python main.py --load-models saved_models/nba_model_20230401_120000
"""
import sys
import pandas as pd
import argparse
from src.predictor import EnhancedNBAPredictor


def main():
    """
    Initialize and run the NBA prediction model.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='NBA Game Prediction System')
    parser.add_argument('--standard', action='store_true', 
                      help='Use standard models instead of enhanced models. Standard models '
                           'are simpler but may be less accurate than enhanced models.')
    parser.add_argument('--seasons', nargs='+', default=['2022-23', '2023-24'],
                      help='NBA seasons to use for training (format: YYYY-YY). '
                           'More recent seasons provide better predictions. '
                           'Example: --seasons 2021-22 2022-23 2023-24')
    parser.add_argument('--quick', action='store_true', 
                      help='Run with simplified models for quick testing. '
                           'Uses fewer folds, simpler architectures, and fewer training '
                           'epochs. Useful for development and testing, not for '
                           'production-quality predictions.')
    parser.add_argument('--save-models', action='store_true',
                      help='Save trained models to disk for later use. Models will be saved '
                           'in the "saved_models" directory with a timestamp.')
    parser.add_argument('--load-models', metavar='MODEL_DIR', type=str,
                      help='Load previously saved models from specified directory instead of '
                           'training new ones. Example: --load-models saved_models/nba_model_20230401_120000')
    args = parser.parse_args()
    
    # Use enhanced models by default unless --standard flag is provided
    use_enhanced = not args.standard
    model_type = "enhanced" if use_enhanced else "standard"
    
    # Check if we're loading pre-trained models
    if args.load_models:
        print(f"Loading pre-trained models from {args.load_models}...")
        predictor = EnhancedNBAPredictor.load_models(args.load_models)
        print("Models loaded successfully!")
    else:
        # Initialize a new predictor with specified settings
        print(f"Starting NBA prediction system with {model_type} models...")
        predictor = EnhancedNBAPredictor(
            seasons=args.seasons,
            use_enhanced_models=use_enhanced,
            quick_mode=args.quick
        )
    
    try:
        # If not loading pre-trained models, fetch data and train
        if not args.load_models:
            # Fetch and process data
            print("Fetching and processing NBA game data...")
            predictor.fetch_and_process_data()
            
            # Train models
            print("Training prediction models...")
            predictor.train_models()
            
            # Save models if requested
            if args.save_models:
                save_dir = predictor.save_models()
                print(f"Models saved to {save_dir}")
        
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
        
        # Additional prediction with different matchup
        prediction2 = predictor.predict_game(
            home_team_id=1610612747,  # LAL
            away_team_id=1610612744,  # GSW
            model_type='hybrid'
        )
        
        print(f"\nLos Angeles Lakers vs Golden State Warriors:")
        print(f"Home win probability: {prediction2['home_win_probability']:.2f}")
        print(f"Confidence: {prediction2['confidence']:.2f}")
        
        print("\nNBA prediction system ready!")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()