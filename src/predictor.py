"""
Main predictor class that orchestrates the NBA prediction workflow.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set

from src.data.data_loader import NBADataLoader
from src.features.feature_processor import NBAFeatureProcessor
from src.models.ensemble_model import NBAEnsembleModel
from src.models.deep_model import DeepModelTrainer
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS, FEATURE_REGISTRY


class EnhancedNBAPredictor:
    """Main class for NBA prediction system."""
    
    def __init__(self, seasons: List[str], lookback_windows: List[int] = None):
        """
        Initialize the NBA prediction system.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY'
            lookback_windows: List of day windows for rolling statistics (default: [7, 14, 30, 60])
        """
        self.seasons = seasons
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        
        # Initialize components
        self.data_loader = NBADataLoader()
        self.feature_processor = NBAFeatureProcessor(self.lookback_windows)
        self.ensemble_model = NBAEnsembleModel()
        self.deep_model_trainer = DeepModelTrainer()
        
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
        """Train both ensemble and deep learning models."""
        if self.features is None:
            raise ValueError("Features not available. Call fetch_and_process_data first.")
        
        # Train ensemble model
        print("\nTraining ensemble model...")
        self.ensemble_model.train(self.features)
        
        # Train deep model
        print("\nTraining deep learning model...")
        self.deep_model_trainer.train_deep_model(self.features)
        
    def predict(self, 
                features: pd.DataFrame,
                model_type: str = 'ensemble') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            features: DataFrame containing features
            model_type: Type of model to use ('ensemble', 'deep', or 'hybrid')
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if model_type == 'ensemble':
            return self.ensemble_model.predict_with_confidence(features)
        elif model_type == 'deep':
            predictions = self.deep_model_trainer.predict(features)
            confidence_scores = self.ensemble_model.calculate_confidence_score(predictions, features)
            return predictions, confidence_scores
        elif model_type == 'hybrid':
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