"""
Main predictor class that orchestrates the NBA prediction workflow.
"""
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union

from src.data.data_loader import NBADataLoader
from src.features.feature_processor import NBAFeatureProcessor
from src.models.ensemble_model import NBAEnsembleModel
from src.models.deep_model import DeepModelTrainer
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS


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
        
        # Add game date back to features for time-based splits
        self.features['GAME_DATE'] = self.stats_df['GAME_DATE']
        self.features['TARGET'] = self.targets
        
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
        Prepare features for a specific game prediction with improved feature compatibility.
        
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
            for col in ['H2H_GAMES', 'H2H_WIN_PCT', 'DAYS_SINCE_H2H', 'LAST_GAME_HOME']:
                if col in h2h_recent:
                    new_game[col] = h2h_recent[col]
        else:
            # Default values if no head-to-head history
            new_game['H2H_GAMES'] = 0
            new_game['H2H_WIN_PCT'] = 0.5
            new_game['DAYS_SINCE_H2H'] = 365
            new_game['LAST_GAME_HOME'] = 0
            
        # Process features for this game
        game_df = pd.DataFrame([new_game])
        
        # Get features from the feature processor
        print("Generating prediction features...")
        
        # Get required features from the training models
        required_features = set()
        
        # From ensemble model
        if hasattr(self.ensemble_model, 'training_features'):
            required_features.update(self.ensemble_model.training_features)
        
        # From deep model
        if hasattr(self.deep_model_trainer, 'training_features'):
            required_features.update(self.deep_model_trainer.training_features)
            
        # If we have a record of the training features
        if self.features is not None:
            feature_cols = [col for col in self.features.columns if col not in ['GAME_DATE', 'TARGET']]
            required_features.update(feature_cols)
        
        # Create a new features DataFrame with the game date
        prediction_features = pd.DataFrame(index=[0])
        prediction_features['GAME_DATE'] = game_date
        
        # Generate base features using the feature processor
        base_features = self.feature_processor.prepare_enhanced_features(game_df)
        
        # Copy all features from base_features
        for col in base_features.columns:
            if col != 'TARGET':  # Skip target column
                prediction_features[col] = base_features[col].values
        
        # Add specifically identified missing features
        # PACE differentials
        for window in self.lookback_windows:
            window_str = f'_{window}D'
            pace_diff_col = f'PACE_DIFF{window_str}'
            
            # If PACE_DIFF is missing but required
            if pace_diff_col in required_features and pace_diff_col not in prediction_features.columns:
                home_pace_col = f'PACE_mean_HOME{window_str}'
                away_pace_col = f'PACE_mean_AWAY{window_str}'
                
                # If we have the component values
                if home_pace_col in prediction_features.columns and away_pace_col in prediction_features.columns:
                    prediction_features[pace_diff_col] = (
                        prediction_features[home_pace_col] - prediction_features[away_pace_col]
                    )
                else:
                    # Otherwise use a default
                    prediction_features[pace_diff_col] = 0
        
        # H2H_RECENCY_WEIGHT
        if 'H2H_RECENCY_WEIGHT' in required_features and 'H2H_RECENCY_WEIGHT' not in prediction_features.columns:
            if 'H2H_WIN_PCT' in prediction_features.columns and 'DAYS_SINCE_H2H' in prediction_features.columns:
                import numpy as np
                # Safely calculate the recency weight
                days = max(1, prediction_features['DAYS_SINCE_H2H'].iloc[0])
                prediction_features['H2H_RECENCY_WEIGHT'] = (
                    prediction_features['H2H_WIN_PCT'].iloc[0] / np.log1p(days)
                )
            else:
                prediction_features['H2H_RECENCY_WEIGHT'] = 0.5
                
        # Add all consistency features
        for window in self.lookback_windows:
            window_str = f'_{window}D'
            for location in ['HOME', 'AWAY']:
                # For each consistency feature
                consistency_feat = f'{location}_CONSISTENCY{window_str}'
                if consistency_feat in required_features and consistency_feat not in prediction_features.columns:
                    # Try to derive from PTS_mean and PTS_std if available
                    pts_mean_col = f'PTS_mean_{location}{window_str}'
                    pts_std_col = f'PTS_std_{location}{window_str}'
                    
                    if pts_mean_col in prediction_features.columns and pts_std_col in prediction_features.columns:
                        # Only calculate if mean is non-zero
                        if prediction_features[pts_mean_col].iloc[0] > 0:
                            prediction_features[consistency_feat] = (
                                prediction_features[pts_std_col].iloc[0] / 
                                prediction_features[pts_mean_col].iloc[0]
                            )
                        else:
                            prediction_features[consistency_feat] = 0.5
                    else:
                        prediction_features[consistency_feat] = 0.5  # Middle value for default
        
        # Add all FATIGUE_DIFF features
        for window in self.lookback_windows:
            window_str = f'_{window}D'
            fatigue_diff_col = f'FATIGUE_DIFF{window_str}'
            
            if fatigue_diff_col in required_features and fatigue_diff_col not in prediction_features.columns:
                home_fatigue_col = f'FATIGUE_HOME{window_str}'
                away_fatigue_col = f'FATIGUE_AWAY{window_str}'
                
                if home_fatigue_col in prediction_features.columns and away_fatigue_col in prediction_features.columns:
                    prediction_features[fatigue_diff_col] = (
                        prediction_features[home_fatigue_col] - prediction_features[away_fatigue_col]
                    )
                else:
                    prediction_features[fatigue_diff_col] = 0
        
        # Add all WIN_PCT_DIFF features
        for window in self.lookback_windows:
            window_str = f'_{window}D'
            win_pct_diff_col = f'WIN_PCT_DIFF{window_str}'
            
            if win_pct_diff_col in required_features and win_pct_diff_col not in prediction_features.columns:
                home_win_pct_col = f'WIN_PCT_HOME{window_str}'
                away_win_pct_col = f'WIN_PCT_AWAY{window_str}'
                
                if home_win_pct_col in prediction_features.columns and away_win_pct_col in prediction_features.columns:
                    prediction_features[win_pct_diff_col] = (
                        prediction_features[home_win_pct_col] - prediction_features[away_win_pct_col]
                    )
                else:
                    prediction_features[win_pct_diff_col] = 0
                    
        # Add all Rating (RTG) differential features
        for rtg_type in ['OFF_RTG', 'DEF_RTG', 'NET_RTG']:
            for window in self.lookback_windows:
                window_str = f'_{window}D'
                rtg_diff_col = f'{rtg_type}_DIFF{window_str}'
                
                if rtg_diff_col in required_features and rtg_diff_col not in prediction_features.columns:
                    home_rtg_col = f'{rtg_type}_mean_HOME{window_str}'
                    away_rtg_col = f'{rtg_type}_mean_AWAY{window_str}'
                    
                    if home_rtg_col in prediction_features.columns and away_rtg_col in prediction_features.columns:
                        prediction_features[rtg_diff_col] = (
                            prediction_features[home_rtg_col] - prediction_features[away_rtg_col]
                        )
                    else:
                        prediction_features[rtg_diff_col] = 0
        
        # Add any remaining missing required features with default values
        for col in required_features:
            if col not in prediction_features.columns and col not in ['GAME_DATE', 'TARGET']:
                prediction_features[col] = 0
        
        # Add head-to-head features if they're required but missing
        for h2h_feat in ['H2H_GAMES', 'H2H_WIN_PCT', 'DAYS_SINCE_H2H', 'H2H_MOMENTUM', 'H2H_HOME_ADVANTAGE']:
            if h2h_feat in required_features and h2h_feat not in prediction_features.columns:
                prediction_features[h2h_feat] = 0 if h2h_feat != 'H2H_WIN_PCT' else 0.5
        
        return prediction_features
    
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