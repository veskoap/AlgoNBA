"""
Main predictor class that orchestrates the NBA prediction workflow.
"""
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
import datetime

from src.data.data_loader import NBADataLoader
from src.features.feature_processor import NBAFeatureProcessor
from src.features.advanced.player_availability import PlayerAvailabilityProcessor
from src.models.ensemble_model import NBAEnsembleModel
from src.models.enhanced_ensemble import NBAEnhancedEnsembleModel
from src.models.deep_model import DeepModelTrainer
from src.models.enhanced_deep_model import EnhancedDeepModelTrainer
from src.models.hybrid_model import HybridModel
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS, FEATURE_REGISTRY


class EnhancedNBAPredictor:
    """Main class for NBA prediction system."""
    
    def __init__(self, seasons: List[str], lookback_windows: List[int] = None, 
                use_enhanced_models: bool = True, quick_mode: bool = False):
        """
        Initialize the NBA prediction system.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY' (e.g., '2022-23')
                    More recent seasons will provide better predictions
            lookback_windows: List of day windows for rolling statistics 
                              (default: [7, 14, 30, 60])
            use_enhanced_models: Whether to use enhanced models or standard ones.
                                 Enhanced models offer higher accuracy but may take
                                 longer to train.
            quick_mode: Whether to run in quick test mode with simplified models.
                        When True:
                        - Uses fewer cross-validation folds (2 instead of 5)
                        - Uses simpler model architectures
                        - Runs fewer training epochs
                        - Performs less hyperparameter optimization
                        Useful for development and testing, but for highest accuracy,
                        set to False.
        """
        self.seasons = seasons
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        self.use_enhanced_models = use_enhanced_models
        self.quick_mode = quick_mode
        
        # Initialize components
        self.data_loader = NBADataLoader()
        self.feature_processor = NBAFeatureProcessor(self.lookback_windows)
        self.player_processor = PlayerAvailabilityProcessor()
        
        # Initialize appropriate models based on flag
        if use_enhanced_models:
            if self.quick_mode:
                # Use simplified models for quick testing
                self.ensemble_model = NBAEnhancedEnsembleModel(
                    use_calibration=False, 
                    use_stacking=False,
                    n_folds=2  # Use fewer folds for faster testing
                )
                self.deep_model_trainer = EnhancedDeepModelTrainer(
                    use_residual=False, 
                    use_attention=False, 
                    use_mc_dropout=False,
                    epochs=5,  # Very few epochs for quick testing
                    hidden_layers=[64, 32],  # Simplified architecture
                    n_folds=2  # Fewer folds for faster testing
                )
                self.hybrid_model = HybridModel(
                    ensemble_model=self.ensemble_model,
                    deep_model=self.deep_model_trainer,
                    quick_mode=True
                )
            else:
                # Use full models
                self.ensemble_model = NBAEnhancedEnsembleModel()
                self.deep_model_trainer = EnhancedDeepModelTrainer()
                self.hybrid_model = HybridModel()
        else:
            self.ensemble_model = NBAEnsembleModel()
            self.deep_model_trainer = DeepModelTrainer()
            self.hybrid_model = None
        
        # Storage for models and data
        self.games = None
        self.advanced_metrics = None
        self.stats_df = None
        self.features = None
        self.targets = None
        
    def fetch_and_process_data(self) -> None:
        """Fetch NBA data and process it into features."""
        # Load data
        print("Fetching NBA game data...")
        self.games, self.advanced_metrics = self.data_loader.fetch_games(self.seasons)
        
        # Calculate team statistics
        print("Calculating team statistics...")
        self.stats_df = self.feature_processor.calculate_team_stats(self.games)
        
        # Calculate player availability impact (new)
        print("Calculating player availability impact...")
        player_features = self.player_processor.calculate_player_impact_features(self.games)
        
        # Merge player features with team stats
        self.stats_df = self.stats_df.merge(
            player_features,
            on=['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'],
            how='left'
        )
        
        # Fill missing player impact with default values
        if 'PLAYER_IMPACT_HOME' not in self.stats_df.columns:
            self.stats_df['PLAYER_IMPACT_HOME'] = 1.0
        if 'PLAYER_IMPACT_AWAY' not in self.stats_df.columns:
            self.stats_df['PLAYER_IMPACT_AWAY'] = 1.0
        if 'PLAYER_IMPACT_DIFF' not in self.stats_df.columns:
            self.stats_df['PLAYER_IMPACT_DIFF'] = self.stats_df['PLAYER_IMPACT_HOME'] - self.stats_df['PLAYER_IMPACT_AWAY']
        
        # Prepare features
        print("Preparing features...")
        self.features, self.targets = self.feature_processor.prepare_features(self.stats_df)
        
        # Add game date and target together to avoid fragmentation
        additional_cols = {
            'GAME_DATE': self.stats_df['GAME_DATE'],
            'TARGET': self.targets
        }
        self.features = pd.concat([self.features, pd.DataFrame(additional_cols, index=self.features.index)], axis=1)
        
    def train_models(self) -> None:
        """Train all prediction models."""
        if self.features is None:
            raise ValueError("Features not available. Call fetch_and_process_data first.")
        
        if self.use_enhanced_models:
            # Train the hybrid model (which trains both ensemble and deep models)
            print("\nTraining advanced hybrid model...")
            self.hybrid_model.train(self.features)
        else:
            # Train models separately
            # Train ensemble model
            print("\nTraining ensemble model...")
            self.ensemble_model.train(self.features)
            
            # Train deep model
            print("\nTraining deep learning model...")
            self.deep_model_trainer.train_deep_model(self.features)
        
    def predict(self, 
                features: pd.DataFrame,
                model_type: str = 'hybrid') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            features: DataFrame containing features
            model_type: Type of model to use ('ensemble', 'deep', or 'hybrid')
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if model_type == 'ensemble':
            if self.use_enhanced_models:
                return self.ensemble_model.predict_with_confidence(features)
            else:
                return self.ensemble_model.predict_with_confidence(features)
        elif model_type == 'deep':
            if self.use_enhanced_models:
                preds, uncertainties = self.deep_model_trainer.predict_with_uncertainty(features)
                confidence_scores = self.deep_model_trainer.calculate_confidence_from_uncertainty(preds, uncertainties)
                return preds, confidence_scores
            else:
                predictions = self.deep_model_trainer.predict(features)
                confidence_scores = self.ensemble_model.calculate_confidence_score(predictions, features)
                return predictions, confidence_scores
        elif model_type == 'hybrid':
            if self.use_enhanced_models:
                return self.hybrid_model.predict_with_confidence(features)
            else:
                # Get predictions from both models
                ensemble_preds = self.ensemble_model.predict(features)
                deep_preds = self.deep_model_trainer.predict(features)
                
                # Average the predictions (weighted equally for now)
                hybrid_preds = (ensemble_preds + deep_preds) / 2
                
                # Use ensemble model for confidence scores
                confidence_scores = self.ensemble_model.calculate_confidence_score(hybrid_preds, features)
                return hybrid_preds, confidence_scores
        else:
            raise ValueError("Invalid model_type. Choose from 'ensemble', 'deep', or 'hybrid'.")
            
    def get_feature_importances(self, n: int = 20) -> Dict:
        """
        Get the top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            dict: Feature names and importance scores
        """
        if self.use_enhanced_models and self.hybrid_model:
            return self.hybrid_model.get_feature_importances(n)
        else:
            return self.ensemble_model.get_top_features(n)
    
    def prepare_game_prediction(self, 
                              home_team_id: int, 
                              away_team_id: int,
                              game_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare features for a specific game prediction using the standardized feature pipeline.
        
        Args:
            home_team_id: NBA API team ID for home team
            away_team_id: NBA API team ID for away team
            game_date: Game date in format 'YYYY-MM-DD' (default: latest available date)
            
        Returns:
            pd.DataFrame: Features for the game
        """
        if self.stats_df is None:
            raise ValueError("Stats not available. Call fetch_and_process_data first.")
            
        # If no date provided, use the latest available
        if game_date is None:
            game_date = self.stats_df['GAME_DATE'].max()
        else:
            game_date = pd.to_datetime(game_date)
            
        print(f"Preparing prediction for {home_team_id} (home) vs {away_team_id} (away) on {game_date}")
            
        # Find the most recent entry for both teams
        home_stats = self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['GAME_DATE'] <= game_date)
        ].sort_values('GAME_DATE', ascending=False).iloc[0].copy() if not self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['GAME_DATE'] <= game_date)
        ].empty else pd.Series()
        
        away_stats = self.stats_df[
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) & 
            (self.stats_df['GAME_DATE'] <= game_date)
        ].sort_values('GAME_DATE', ascending=False).iloc[0].copy() if not self.stats_df[
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) & 
            (self.stats_df['GAME_DATE'] <= game_date)
        ].empty else pd.Series()
        
        if home_stats.empty or away_stats.empty:
            raise ValueError("Not enough data available for one or both teams.")
            
        # Create a new game entry with these teams
        new_game = pd.Series({
            'GAME_DATE': game_date,
            'TEAM_ID_HOME': home_team_id,
            'TEAM_ID_AWAY': away_team_id
        })
        
        # For simplicity, copy all features from the most recent games
        for col in home_stats.index:
            if col not in new_game and '_HOME' in col:
                new_game[col] = home_stats[col]
                
        for col in away_stats.index:
            if col not in new_game and '_AWAY' in col:
                new_game[col] = away_stats[col]
                
        # Add head-to-head features
        h2h_stats = self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) &
            (self.stats_df['GAME_DATE'] <= game_date)
        ].sort_values('GAME_DATE', ascending=False)
        
        if not h2h_stats.empty:
            h2h_recent = h2h_stats.iloc[0]
            for col in ['H2H_GAMES', 'H2H_WIN_PCT', 'DAYS_SINCE_H2H', 'LAST_GAME_HOME', 
                       'H2H_AVG_MARGIN', 'H2H_STREAK', 'H2H_HOME_ADVANTAGE', 'H2H_MOMENTUM']:
                if col in h2h_recent:
                    new_game[col] = h2h_recent[col]
        else:
            # Default values if no head-to-head history
            h2h_defaults = {
                'H2H_GAMES': 0,
                'H2H_WIN_PCT': 0.5,
                'DAYS_SINCE_H2H': 365,
                'LAST_GAME_HOME': 0,
                'H2H_AVG_MARGIN': 0,
                'H2H_STREAK': 0,
                'H2H_HOME_ADVANTAGE': 0.5,
                'H2H_MOMENTUM': 0.5
            }
            for col, val in h2h_defaults.items():
                new_game[col] = val
                
        # Add player impact features (default values)
        if 'PLAYER_IMPACT_HOME' not in new_game:
            new_game['PLAYER_IMPACT_HOME'] = 1.0
        if 'PLAYER_IMPACT_AWAY' not in new_game:
            new_game['PLAYER_IMPACT_AWAY'] = 1.0
        if 'PLAYER_IMPACT_DIFF' not in new_game:
            new_game['PLAYER_IMPACT_DIFF'] = new_game['PLAYER_IMPACT_HOME'] - new_game['PLAYER_IMPACT_AWAY']
        if 'PLAYER_IMPACT_HOME_MOMENTUM' not in new_game:
            new_game['PLAYER_IMPACT_HOME_MOMENTUM'] = 1.0
        if 'PLAYER_IMPACT_AWAY_MOMENTUM' not in new_game:
            new_game['PLAYER_IMPACT_AWAY_MOMENTUM'] = 1.0
        if 'PLAYER_IMPACT_MOMENTUM_DIFF' not in new_game:
            new_game['PLAYER_IMPACT_MOMENTUM_DIFF'] = 0.0
            
        # Process features for this game
        game_df = pd.DataFrame([new_game])
        
        # Collect required features from all sources
        required_features = self._get_required_features()
        
        print(f"Identified {len(required_features)} required features for prediction")
        
        # Initialize feature transformer with required features
        self.feature_processor.feature_transformer.register_features(list(required_features))
        
        # Generate prediction features using the standardized pipeline
        print("Generating prediction features using standardized pipeline...")
        prediction_features = self.feature_processor.prepare_enhanced_features(game_df)
        
        # Final validation to ensure all required features are present
        missing_features = []
        missing_feature_values = {}
        
        for feature in required_features:
            if feature not in prediction_features.columns and feature not in ['GAME_DATE', 'TARGET']:
                missing_features.append(feature)
                # Add with default value (0 or 0.5 for probabilities)
                if any(prob_term in feature for prob_term in ['WIN_PCT', 'PROBABILITY', 'H2H_']):
                    missing_feature_values[feature] = 0.5
                else:
                    missing_feature_values[feature] = 0
        
        # Add all missing features at once to avoid fragmentation
        if missing_feature_values:
            # Create a DataFrame with missing features and concatenate
            missing_df = pd.DataFrame(missing_feature_values, index=prediction_features.index)
            prediction_features = pd.concat([prediction_features, missing_df], axis=1)
        
        if missing_features:
            print(f"Added {len(missing_features)} missing features with default values: {missing_features[:5]}...")
        
        return prediction_features
        
    def _get_required_features(self) -> Set[str]:
        """
        Get a comprehensive set of all required features from all components.
        
        Returns:
            Set of required feature names
        """
        required_features = set()
        
        # From ensemble model
        if hasattr(self.ensemble_model, 'training_features'):
            required_features.update(self.ensemble_model.training_features)
        
        # From deep model
        if hasattr(self.deep_model_trainer, 'training_features'):
            required_features.update(self.deep_model_trainer.training_features)
            
        # From feature registry (important derived features)
        for base_feature, info in FEATURE_REGISTRY.items():
            if info['type'] in ['derived', 'interaction']:
                # Add all window variants if applicable
                if info['windows']:
                    for window in info['windows']:
                        required_features.add(f"{base_feature}_{window}D")
                else:
                    required_features.add(base_feature)
                    
                # Also add dependencies
                if 'dependencies' in info:
                    for dep in info['dependencies']:
                        if info['windows']:
                            for window in info['windows']:
                                required_features.add(f"{dep}_{window}D")
                        else:
                            required_features.add(dep)
            
        # If we have a record of the training features
        if self.features is not None:
            feature_cols = [col for col in self.features.columns if col not in ['GAME_DATE', 'TARGET']]
            required_features.update(feature_cols)
            
        # Remove non-feature columns
        for col in ['GAME_DATE', 'TARGET', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']:
            if col in required_features:
                required_features.remove(col)
                
        return required_features
    
    def predict_game(self, 
                   home_team_id: int, 
                   away_team_id: int,
                   game_date: Optional[str] = None,
                   model_type: str = 'hybrid') -> Dict:
        """
        Predict the outcome of a specific game with improved feature compatibility.
        
        Args:
            home_team_id: NBA API team ID for home team
            away_team_id: NBA API team ID for away team
            game_date: Game date in format 'YYYY-MM-DD' (default: latest available date)
            model_type: Type of model to use ('ensemble', 'deep', or 'hybrid')
            
        Returns:
            dict: Prediction details
        """
        # Prepare features for the game - already ensures required features are present
        game_features = self.prepare_game_prediction(home_team_id, away_team_id, game_date)
        
        # Ensure model is trained
        if self.features is None:
            raise ValueError("Model not trained properly. Call train_models first.")
        
        # Make prediction with the prepared feature set
        # The models' predict methods handle feature alignment internally now
        probs, confidence = self.predict(game_features, model_type)
        
        # Get team abbreviations for better output
        from src.utils.constants import TEAM_ID_TO_ABBREV
        home_team_abbrev = TEAM_ID_TO_ABBREV.get(home_team_id, str(home_team_id))
        away_team_abbrev = TEAM_ID_TO_ABBREV.get(away_team_id, str(away_team_id))
        
        # Format output
        result = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team': home_team_abbrev,
            'away_team': away_team_abbrev,
            'game_date': game_features['GAME_DATE'].iloc[0],
            'home_win_probability': float(probs[0]),
            'confidence': float(confidence[0]),
            'model_type': model_type
        }
        
        return result
        
    def save_models(self, directory: str = 'saved_models') -> str:
        """
        Save all trained models to disk.
        
        Args:
            directory: Directory to save models in
            
        Returns:
            str: Path to the saved model directory
        """
        # Create save directory if it doesn't exist
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(directory, f"nba_model_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving models to {save_dir}...")
        
        # Save model configuration
        config = {
            'seasons': self.seasons,
            'lookback_windows': self.lookback_windows,
            'use_enhanced_models': self.use_enhanced_models,
            'quick_mode': self.quick_mode,
            'timestamp': timestamp
        }
        
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        # Save the feature processor state
        with open(os.path.join(save_dir, 'feature_processor.pkl'), 'wb') as f:
            pickle.dump(self.feature_processor, f)
        
        # Save the ensemble model
        if self.ensemble_model:
            self.ensemble_model.save_model(os.path.join(save_dir, 'ensemble_model'))
        
        # Save the deep model
        if self.deep_model_trainer:
            self.deep_model_trainer.save_model(os.path.join(save_dir, 'deep_model'))
        
        # Save the hybrid model if available
        if self.hybrid_model:
            self.hybrid_model.save_model(os.path.join(save_dir, 'hybrid_model'))
        
        # Save feature statistics
        if self.features is not None:
            with open(os.path.join(save_dir, 'feature_stats.pkl'), 'wb') as f:
                # Save only the feature names and statistics, not the full data
                feature_stats = {
                    'feature_names': list(self.features.columns),
                    'feature_means': self.features.mean().to_dict(),
                    'feature_stds': self.features.std().to_dict(),
                }
                pickle.dump(feature_stats, f)
        
        print(f"Models successfully saved to {save_dir}")
        
        return save_dir
    
    @classmethod
    def load_models(cls, model_dir: str) -> 'EnhancedNBAPredictor':
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory containing saved model files
            
        Returns:
            EnhancedNBAPredictor: Loaded predictor instance
        """
        print(f"Loading models from {model_dir}...")
        
        # Load configuration
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        # Create a new instance with the same configuration
        predictor = cls(
            seasons=config['seasons'],
            lookback_windows=config['lookback_windows'],
            use_enhanced_models=config['use_enhanced_models'],
            quick_mode=config['quick_mode']
        )
        
        # Load feature processor
        with open(os.path.join(model_dir, 'feature_processor.pkl'), 'rb') as f:
            predictor.feature_processor = pickle.load(f)
        
        # Load models
        # Ensemble model
        ensemble_dir = os.path.join(model_dir, 'ensemble_model')
        if os.path.exists(ensemble_dir):
            if config['use_enhanced_models']:
                predictor.ensemble_model = NBAEnhancedEnsembleModel.load_model(ensemble_dir)
            else:
                predictor.ensemble_model = NBAEnsembleModel.load_model(ensemble_dir)
        
        # Deep model
        deep_dir = os.path.join(model_dir, 'deep_model')
        if os.path.exists(deep_dir):
            if config['use_enhanced_models']:
                predictor.deep_model_trainer = EnhancedDeepModelTrainer.load_model(deep_dir)
            else:
                predictor.deep_model_trainer = DeepModelTrainer.load_model(deep_dir)
        
        # Hybrid model
        hybrid_dir = os.path.join(model_dir, 'hybrid_model')
        if os.path.exists(hybrid_dir):
            predictor.hybrid_model = HybridModel.load_model(hybrid_dir)
        
        # Load feature statistics if available
        stats_path = os.path.join(model_dir, 'feature_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                feature_stats = pickle.load(f)
                # Set as placeholder to enable prediction
                predictor.features = pd.DataFrame(columns=feature_stats['feature_names'])
        
        print(f"Models successfully loaded from {model_dir}")
        return predictor