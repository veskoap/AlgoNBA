"""
Main entry point for the NBA Prediction System.

This script provides a command-line interface to the NBA prediction system, 
allowing users to train models and generate predictions with various options:

- Enhanced vs standard models: Control accuracy vs complexity tradeoff
- Season selection: Choose which NBA seasons to use for training
- Quick mode: Faster execution for testing and development
- Save/load models: Save trained models to disk and load them for later use
- Cache system: Speed up subsequent runs by caching data, features, and model results

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
    
    # Disable the cache system to always fetch fresh data
    python main.py --no-cache
    
    # View cache statistics
    python main.py --cache-action status
    
    # Clear a specific cache type (games, features, models, predictions)
    python main.py --cache-action clear_type --cache-type features
    
    # Clear all cache entries
    python main.py --cache-action clear_all
"""
import sys
import os
import pandas as pd
import argparse
import warnings
from src.predictor import EnhancedNBAPredictor
from src.utils.helpers import suppress_sklearn_warnings

# Suppress all sklearn warnings before any imports
suppress_sklearn_warnings()

# Also add a broader warning filter for other FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)


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
    parser.add_argument('--no-cache', action='store_true',
                      help='Disable the cache system. By default, the system caches data, features, '
                           'and predictions to speed up subsequent runs. Use this flag to always '
                           'fetch and process fresh data.')
    parser.add_argument('--selective-cache', choices=['data', 'features', 'models', 'all'], default=None,
                      help='Selectively enable caching for specific components while ignoring others. '
                           'Useful with --no-cache to only refresh certain parts of the pipeline. '
                           'Example: --no-cache --selective-cache data (will use cached raw data but refresh features/models).')
    parser.add_argument('--cache-action', choices=['status', 'clear_type', 'clear_all'],
                      help='Perform cache management actions: get cache status, clear specific '
                           'cache type, or clear all cache.')
    parser.add_argument('--cache-type', choices=['games', 'features', 'models', 'predictions'],
                      help='Specify cache type for cache-action')
    parser.add_argument('--cache-dir', type=str,
                      help='Specify custom directory for cache storage')
    parser.add_argument('--no-hardware-optimization', action='store_true',
                      help='Disable hardware-specific optimizations for M1/CUDA/Colab environments')
    parser.add_argument('--colab-drive', action='store_true',
                      help='Use Google Drive for storage when in Colab environment')
    args = parser.parse_args()
    
    # Use enhanced models by default unless --standard flag is provided
    use_enhanced = not args.standard
    model_type = "enhanced" if use_enhanced else "standard"
    
    # Determine whether to use cache and hardware optimizations
    use_cache = not args.no_cache
    selective_cache = args.selective_cache
    
    # Handle selective caching (allows parts of the system to use cache while others don't)
    cache_components = {
        'data': use_cache,
        'features': use_cache,
        'models': use_cache,
        'predictions': use_cache
    }
    
    # If selective caching is enabled, modify the cache settings
    if args.no_cache and selective_cache:
        if selective_cache == 'all':
            # Enable all caches despite --no-cache
            cache_components = {k: True for k in cache_components}
        else:
            # Default to disabled for all components when --no-cache is used
            cache_components = {k: False for k in cache_components}
            # Enable only the selected component
            cache_components[selective_cache] = True
    
    # Status reporting
    if args.no_cache and selective_cache:
        cache_status = f"partially enabled (only {selective_cache})"
    else:
        cache_status = "disabled" if args.no_cache else "enabled"
    
    hw_optimization = not args.no_hardware_optimization
    hw_status = "disabled" if args.no_hardware_optimization else "enabled"
    cache_dir = args.cache_dir
    
    # Check if running in Google Colab
    is_colab = False
    try:
        import google.colab
        is_colab = True
        print("Google Colab environment detected")
        
        # Set up Drive integration if requested
        if args.colab_drive:
            try:
                from google.colab import drive
                if not os.path.exists('/content/drive'):
                    print("Mounting Google Drive for persistent storage...")
                    drive.mount('/content/drive')
                    
                # Double-check that Drive was mounted successfully
                if os.path.exists('/content/drive/MyDrive'):
                    # Create AlgoNBA directories in Drive if they don't exist
                    os.makedirs('/content/drive/MyDrive/AlgoNBA/cache', exist_ok=True)
                    os.makedirs('/content/drive/MyDrive/AlgoNBA/models', exist_ok=True)
                    print("Google Drive mounted and AlgoNBA directories created")
                    
                    # Use Drive for cache unless explicitly specified
                    if cache_dir is None:
                        cache_dir = '/content/drive/MyDrive/AlgoNBA/cache'
                        print(f"Using Google Drive for cache storage: {cache_dir}")
                else:
                    print("WARNING: Google Drive mount may have failed. Using session storage instead.")
            except Exception as e:
                print(f"WARNING: Error setting up Google Drive: {e}")
                print("Continuing without Drive integration.")
    except ImportError:
        pass  # Not in Colab environment
    
    # Check if we're just performing a cache management action
    if args.cache_action:
        # Create a temporary predictor just for cache management
        temp_predictor = EnhancedNBAPredictor(
            seasons=args.seasons,
            use_enhanced_models=use_enhanced,
            quick_mode=args.quick,
            use_cache=True,  # Must be enabled for cache management
            cache_dir=cache_dir,
            hardware_optimization=hw_optimization
        )
        
        # Perform the requested cache action
        result = temp_predictor.manage_cache(args.cache_action, args.cache_type)
        print(f"Cache {args.cache_action} result: {result['status']}")
        if 'statistics' in result:
            stats = result['statistics']
            print(f"Cache entries: {stats['total_entries']}")
            print(f"Cache size: {stats['total_size_mb']:.2f} MB")
            print(f"Cache directory: {temp_predictor.cache_manager.cache_dir}")
            print("Cache types:")
            for cache_type, count in stats['by_type'].items():
                print(f"  - {cache_type}: {count} entries")
        
        # Exit after performing cache action
        return
    
    # Check if we're loading pre-trained models
    if args.load_models:
        # For Colab with Drive integration, adjust model path if needed
        model_dir = args.load_models
        if is_colab and args.colab_drive and not model_dir.startswith('/content/drive'):
            # Check if model exists in Drive
            drive_model_path = f"/content/drive/MyDrive/AlgoNBA/models/{os.path.basename(model_dir)}"
            if os.path.exists(drive_model_path):
                model_dir = drive_model_path
                print(f"Using model from Google Drive: {model_dir}")
        
        print(f"Loading pre-trained models from {model_dir}...")
        predictor = EnhancedNBAPredictor.load_models(
            model_dir, 
            use_cache=use_cache,
            cache_dir=cache_dir,
            hardware_optimization=hw_optimization
        )
        print(f"Models loaded successfully! Cache {cache_status}, Hardware optimizations {hw_status}.")
    else:
        # Initialize a new predictor with specified settings
        print(f"Starting NBA prediction system with {model_type} models...")
        print(f"Cache system {cache_status}, Hardware optimizations {hw_status}.")
        
        # Pass selective cache components if enabled
        if args.no_cache and selective_cache:
            print(f"Using selective caching: {'data' if cache_components['data'] else 'no data'}, "
                 f"{'features' if cache_components['features'] else 'no features'}, "
                 f"{'models' if cache_components['models'] else 'no models'}")
            predictor = EnhancedNBAPredictor(
                seasons=args.seasons,
                use_enhanced_models=use_enhanced,
                quick_mode=args.quick,
                use_cache=use_cache,
                cache_dir=cache_dir,
                hardware_optimization=hw_optimization,
                selective_cache_components=cache_components
            )
        else:
            # Standard initialization without selective caching
            predictor = EnhancedNBAPredictor(
                seasons=args.seasons,
                use_enhanced_models=use_enhanced,
                quick_mode=args.quick,
                use_cache=use_cache,
                cache_dir=cache_dir,
                hardware_optimization=hw_optimization
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
                # Determine save directory
                if is_colab and args.colab_drive:
                    # Use Google Drive for persistence when in Colab
                    save_dir = predictor.save_models("/content/drive/MyDrive/AlgoNBA/models")
                    print(f"Models saved to Google Drive: {save_dir}")
                else:
                    # Use standard location
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