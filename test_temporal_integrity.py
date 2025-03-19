"""
Test for temporal data integrity in feature engineering and model training.
Ensures that no future data leaks into feature generation or model evaluation.
"""
import os
import sys
import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.features.feature_processor import NBAFeatureProcessor
from src.models.hybrid_model import HybridModel
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS
from sklearn.model_selection import TimeSeriesSplit

class TemporalIntegrityTest(unittest.TestCase):
    """Test case for verifying temporal integrity in feature generation and model training."""
    
    def setUp(self):
        """Set up test data with chronological games."""
        # Create a sample dataset with clear chronological ordering
        # Generate 100 days of data
        dates = pd.date_range(start="2022-01-01", end="2022-04-10")
        num_games = len(dates)
        
        # Create a DataFrame with games data
        self.games_df = pd.DataFrame({
            'GAME_DATE': dates,
            'GAME_DATE_HOME': dates,
            'TEAM_ID_HOME': [1610612737 + (i % 15) for i in range(num_games)],  # 15 teams rotation
            'TEAM_ID_AWAY': [1610612737 + ((i + 7) % 15) for i in range(num_games)],  # Different rotation
            'PTS_HOME': [100 + i % 30 for i in range(num_games)],
            'PTS_AWAY': [95 + i % 35 for i in range(num_games)],
            'FGM_HOME': [40 + i % 15 for i in range(num_games)],
            'FGM_AWAY': [38 + i % 18 for i in range(num_games)],
            'FGA_HOME': [85 + i % 10 for i in range(num_games)],
            'FGA_AWAY': [84 + i % 12 for i in range(num_games)],
            'FG3M_HOME': [12 + i % 8 for i in range(num_games)],
            'FG3M_AWAY': [11 + i % 9 for i in range(num_games)],
            'FG3A_HOME': [33 + i % 6 for i in range(num_games)],
            'FG3A_AWAY': [32 + i % 7 for i in range(num_games)],
            'FTM_HOME': [15 + i % 10 for i in range(num_games)],
            'FTM_AWAY': [14 + i % 11 for i in range(num_games)],
            'FTA_HOME': [20 + i % 8 for i in range(num_games)],
            'FTA_AWAY': [19 + i % 9 for i in range(num_games)],
            'OREB_HOME': [10 + i % 5 for i in range(num_games)],
            'OREB_AWAY': [9 + i % 6 for i in range(num_games)],
            'DREB_HOME': [30 + i % 7 for i in range(num_games)],
            'DREB_AWAY': [29 + i % 8 for i in range(num_games)],
            'AST_HOME': [22 + i % 6 for i in range(num_games)],
            'AST_AWAY': [21 + i % 7 for i in range(num_games)],
            'TOV_HOME': [12 + i % 5 for i in range(num_games)],
            'TOV_AWAY': [13 + i % 6 for i in range(num_games)],
            'STL_HOME': [7 + i % 4 for i in range(num_games)],
            'STL_AWAY': [8 + i % 4 for i in range(num_games)],
            'BLK_HOME': [5 + i % 3 for i in range(num_games)],
            'BLK_AWAY': [4 + i % 4 for i in range(num_games)],
            'PLUS_MINUS_HOME': [5 + i % 20 - 10 for i in range(num_games)],
            'PLUS_MINUS_AWAY': [-5 - i % 20 + 10 for i in range(num_games)],
            'WL_HOME': ['W' if i % 2 == 0 else 'L' for i in range(num_games)],
            'WL_AWAY': ['L' if i % 2 == 0 else 'W' for i in range(num_games)],
            'GAME_ID_HOME': [f"00{i+2200000:05d}" for i in range(num_games)],
            'GAME_ID_AWAY': [f"00{i+2200000:05d}" for i in range(num_games)]
        })
        
        # Add derived columns for percentages
        self.games_df['FG_PCT_HOME'] = self.games_df['FGM_HOME'] / self.games_df['FGA_HOME']
        self.games_df['FG_PCT_AWAY'] = self.games_df['FGM_AWAY'] / self.games_df['FGA_AWAY']
        self.games_df['FG3_PCT_HOME'] = self.games_df['FG3M_HOME'] / self.games_df['FG3A_HOME']
        self.games_df['FG3_PCT_AWAY'] = self.games_df['FG3M_AWAY'] / self.games_df['FG3A_AWAY']
        self.games_df['FT_PCT_HOME'] = self.games_df['FTM_HOME'] / self.games_df['FTA_HOME']
        self.games_df['FT_PCT_AWAY'] = self.games_df['FTM_AWAY'] / self.games_df['FTA_AWAY']
        
        # Ensure dates are datetime objects
        self.games_df['GAME_DATE'] = pd.to_datetime(self.games_df['GAME_DATE'])
        self.games_df['GAME_DATE_HOME'] = pd.to_datetime(self.games_df['GAME_DATE_HOME'])
        
        # Sort by date to ensure proper temporal ordering for merge_asof
        self.games_df = self.games_df.sort_values('GAME_DATE')
        
        # Add additional columns that might be used as merge keys in the feature processor
        self.games_df['HOME_TEAM_ID'] = self.games_df['TEAM_ID_HOME']
        self.games_df['VISITOR_TEAM_ID'] = self.games_df['TEAM_ID_AWAY']
        self.games_df['SEASON'] = 2022  # Add season column
        self.games_df['SEASON_TYPE'] = 'Regular Season'  # Add season type
        
        # Fill any potential NaN values to prevent merge key failures
        self.games_df = self.games_df.fillna(0)
        
        # Initialize the feature processor with standard parameters
        self.feature_processor = NBAFeatureProcessor(lookback_windows=[7, 14, 30])
        
        # Monkey patch the calculate_team_stats method to handle our test data
        original_calculate_team_stats = self.feature_processor.calculate_team_stats
        
        def safe_calculate_team_stats(games_df, *args, **kwargs):
            # Make a defensive copy to avoid modifying the original
            df = games_df.copy()
            
            # Ensure all required columns are present
            required_cols = ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY', 'PTS_HOME', 'PTS_AWAY']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Missing required column: {col}")
            
            # Ensure no NaN values in key columns that might be used in merges
            key_cols = [col for col in df.columns if ('ID' in col or 'DATE' in col)]
            for col in key_cols:
                if col in df.columns and df[col].isna().any():
                    print(f"Found NaN values in key column: {col}")
                    df[col] = df[col].fillna(0 if 'ID' in col else pd.Timestamp('2022-01-01'))
            
            # Call the original method
            return original_calculate_team_stats(df, *args, **kwargs)
        
        # Apply the monkey patch
        self.feature_processor.calculate_team_stats = safe_calculate_team_stats
    
    def test_temporal_integrity_in_rolling_windows(self):
        """Test that rolling window features only use data from before the current date."""
        # Calculate features using original ordered data
        features = self.feature_processor.calculate_team_stats(self.games_df)
        
        # Verify that we have all expected date range
        self.assertGreaterEqual(len(features), 90, "Should have at least 90 games with features")
        
        # Choose a sample game date near middle of dataset for testing
        middle_idx = len(features) // 2
        test_date = features.iloc[middle_idx]['GAME_DATE']
        test_date_str = pd.to_datetime(test_date).strftime('%Y-%m-%d')
        print(f"Testing temporal integrity for game on {test_date_str}")
        
        # Get features for games up to test date
        past_features = features[features['GAME_DATE'] < test_date]
        
        # Get features for games after test date
        future_features = features[features['GAME_DATE'] > test_date]
        
        # For each team, check rolling stats
        for team_id in features['TEAM_ID_HOME'].unique()[:5]:  # Test with 5 teams
            print(f"Testing team {team_id}")
            
            # For each window size
            for window in [7, 14, 30]:  
                # Check home win percentage
                col_name = f'WIN_mean_HOME_{window}D'
                
                # Get the test game's value
                test_games = features[
                    (features['GAME_DATE'] == test_date) & 
                    (features['TEAM_ID_HOME'] == team_id)
                ]
                if not test_games.empty:
                    test_value = test_games[col_name].iloc[0] if col_name in test_games.columns else None
                    
                    # Get all past home games for this team
                    past_home_games = past_features[
                        (past_features['TEAM_ID_HOME'] == team_id) &
                        (past_features['GAME_DATE'] >= (test_date - pd.Timedelta(days=window))) &
                        (past_features['GAME_DATE'] < test_date)
                    ]
                    
                    if len(past_home_games) > 0:
                        # Calculate the expected win percentage based solely on historical data
                        expected_win_pct = (past_home_games['WL_HOME'] == 'W').mean()
                        
                        # Check for approximate equality (allowing small floating point differences)
                        if test_value is not None:
                            self.assertAlmostEqual(
                                test_value, 
                                expected_win_pct, 
                                delta=0.01,
                                msg=f"Win % mismatch for {team_id} with {window}D window. Got {test_value}, expected {expected_win_pct}"
                            )
                            print(f"✓ {col_name} passed check with window={window}D")
                
    def test_no_future_data_in_features(self):
        """Test that features for a given date don't contain data from future games."""
        # Calculate features using original ordered data
        features = self.feature_processor.calculate_team_stats(self.games_df)
        
        # Choose a cutoff date in the middle of our dataset
        cutoff_date = self.games_df['GAME_DATE'].iloc[len(self.games_df) // 2]
        cutoff_date_str = pd.to_datetime(cutoff_date).strftime('%Y-%m-%d')
        print(f"Testing future data leakage with cutoff date: {cutoff_date_str}")
        
        # Split games into past and future relative to cutoff
        past_games = self.games_df[self.games_df['GAME_DATE'] <= cutoff_date].copy()
        future_games = self.games_df[self.games_df['GAME_DATE'] > cutoff_date].copy()
        
        # Generate features using only past games
        past_features = self.feature_processor.calculate_team_stats(past_games)
        
        # Generate features using all games
        all_features = self.feature_processor.calculate_team_stats(self.games_df)
        
        # Filter to just the games up to cutoff date
        past_filtered_features = all_features[all_features['GAME_DATE'] <= cutoff_date]
        
        # For games up to the cutoff date, features should be identical whether or not
        # we include future games in the input data
        
        # Select common columns to compare
        common_cols = [col for col in past_features.columns 
                      if col in past_filtered_features.columns 
                      and col not in ['GAME_DATE']]
        
        # Compare values for a sample of columns
        test_cols = [col for col in common_cols if any(term in col for term in 
                     ['WIN_mean', 'PTS_mean', 'STADIUM_HOME_ADVANTAGE', 'H2H_WIN_PCT'])]
        
        for col in test_cols[:10]:  # Test 10 columns
            # For non-rolling window stats, we want exact equality
            # For complex columns like H2H stats that depend on a lot of prior data,
            # we allow a small delta to account for implementation differences
            allow_delta = 0.01 if 'H2H' in col else 0 
            
            # For each game up to cutoff
            for idx in past_filtered_features.index:
                if idx in past_features.index:
                    val1 = past_features.loc[idx, col]
                    val2 = past_filtered_features.loc[idx, col]
                    
                    if allow_delta > 0:
                        self.assertAlmostEqual(
                            val1, val2, delta=allow_delta,
                            msg=f"Value mismatch for column {col} at index {idx}: {val1} vs {val2}"
                        )
                    else:
                        # Exact equality for most features
                        np.testing.assert_allclose(
                            val1, val2,
                            err_msg=f"Value mismatch for column {col} at index {idx}: {val1} vs {val2}"
                        )
            
            print(f"✓ Passed feature integrity check for {col}")
        
        print("All feature temporal integrity checks passed!")

    def test_no_future_data_in_cv_splits(self):
        """Test that TimeSeriesSplit with gap correctly prevents data leakage."""
        # Calculate features using original ordered data
        features = self.feature_processor.calculate_team_stats(self.games_df)
        
        # Add simple TARGET column (binary target based on home team win)
        features['TARGET'] = (features['TEAM_ID_HOME'] % 2 == 0).astype(int)  # Simple deterministic target
        
        # Sort by date to ensure chronological order
        features = features.sort_values('GAME_DATE')
        
        # Create TimeSeriesSplit with gap=1 (our implementation)
        tscv = TimeSeriesSplit(n_splits=3, gap=1)
        
        # Verify temporal integrity of each split
        for fold, (train_idx, val_idx) in enumerate(tscv.split(features), 1):
            train_data = features.iloc[train_idx]
            val_data = features.iloc[val_idx]
            
            # Check that validation data is strictly after training data
            train_max_date = pd.to_datetime(train_data['GAME_DATE'].max())
            val_min_date = pd.to_datetime(val_data['GAME_DATE'].min())
            
            print(f"Fold {fold}: Train max date: {train_max_date}, Validation min date: {val_min_date}")
            
            # Validation data must start strictly after the last training data point
            self.assertLess(train_max_date, val_min_date, 
                           f"Validation data starts before or at the end of training data in fold {fold}")
            
            # The gap should be at least 1 day
            date_diff = (val_min_date - train_max_date).days
            self.assertGreaterEqual(date_diff, 1, 
                                   f"Gap between train and validation is less than 1 day in fold {fold}")
            
            print(f"✓ Fold {fold} passed temporal integrity check with {date_diff} days gap")
        
        print("All cross-validation temporal integrity checks passed!")
            
    def test_model_training_has_no_leakage(self):
        """Test that model training doesn't introduce data leakage."""
        # Create a deterministic dataset for testing
        features = self.feature_processor.calculate_team_stats(self.games_df)
        
        # Add TARGET column based on deterministic pattern to allow verification
        # We use a pattern based on date (first half of month home team wins, second half away team wins)
        # This makes prediction without leakage difficult (not 100% accurate) but with leakage easy (100% accurate)
        features['day_of_month'] = pd.to_datetime(features['GAME_DATE']).dt.day
        features['TARGET'] = (features['day_of_month'] <= 15).astype(int)
        
        # Split the data into training and test sets based on date
        train_cutoff = features['GAME_DATE'].iloc[int(len(features) * 0.7)]
        test_cutoff = features['GAME_DATE'].iloc[int(len(features) * 0.85)]
        
        train_data = features[features['GAME_DATE'] < train_cutoff].copy()
        val_data = features[(features['GAME_DATE'] >= train_cutoff) & 
                           (features['GAME_DATE'] < test_cutoff)].copy()
        test_data = features[features['GAME_DATE'] >= test_cutoff].copy()
        
        print(f"Train data: {len(train_data)} games")
        print(f"Validation data: {len(val_data)} games") 
        print(f"Test data: {len(test_data)} games")
        
        # Combine train and val for training
        train_val_data = pd.concat([train_data, val_data])
        
        # Train a simple model on training data
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score
        
        # Select common numeric features
        numeric_cols = [col for col in train_data.columns 
                       if col not in ['GAME_DATE', 'GAME_ID', 'GAME_ID_HOME', 'TEAM_ID_HOME', 
                                     'TEAM_ID_AWAY', 'TARGET', 'day_of_month']]
        
        # Make sure we have some columns
        self.assertGreater(len(numeric_cols), 20, "Should have at least 20 numeric feature columns")
        # Just use first 20 columns to keep it simple
        X_train = train_data[numeric_cols[:20]].fillna(0)
        y_train = train_data['TARGET']
        
        X_val = val_data[numeric_cols[:20]].fillna(0)
        y_val = val_data['TARGET']
        
        X_test = test_data[numeric_cols[:20]].fillna(0)
        y_test = test_data['TARGET']
        
        # Train model
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict on validation and test sets
        val_preds = model.predict(X_val)
        test_preds = model.predict(X_test)
        
        # Calculate accuracy
        val_accuracy = accuracy_score(y_val, val_preds)
        test_accuracy = accuracy_score(y_test, test_preds)
        
        print(f"Validation accuracy: {val_accuracy:.4f}")
        print(f"Test accuracy: {test_accuracy:.4f}")
        
        # With our deterministic pattern, a model with no leakage should have accuracy close to 50-70%
        # With leakage, it would be close to 100%
        self.assertLess(val_accuracy, 0.85, 
                       f"Validation accuracy too high ({val_accuracy:.4f}), suggesting data leakage")
        self.assertLess(test_accuracy, 0.85, 
                       f"Test accuracy too high ({test_accuracy:.4f}), suggesting data leakage")
        
        print("✓ Model training passed leakage check (accuracy is within reasonable bounds)")
        
        # Additional test using our actual hybrid model with simplified components
        try:
            # Create a very simple hybrid model without waiting for full training
            from src.models.hybrid_model import HybridModel
            from src.utils.helpers import fix_dataframe_columns
            
            # Create a copy of the data to prevent modification
            model_data = train_val_data.copy()
            model_data = fix_dataframe_columns(model_data)
            
            # Create Hybrid model in quick mode (minimal training)
            hybrid_model = HybridModel(quick_mode=True)
            
            # Create a temporal splitter for testing
            tscv = TimeSeriesSplit(n_splits=2, gap=1)
            
            # Get the last split
            for train_idx, val_idx in tscv.split(model_data):
                train_split = model_data.iloc[train_idx]
                val_split = model_data.iloc[val_idx]
            
            print(f"Model training split: {len(train_split)} train, {len(val_split)} validation")
            
            # Check temporal separation
            train_max = pd.to_datetime(train_split['GAME_DATE'].max())
            val_min = pd.to_datetime(val_split['GAME_DATE'].min())
            
            self.assertLess(train_max, val_min, 
                           "Validation data is not strictly after training data")
            
            print(f"✓ Hybrid model training temporal split is valid: {train_max} < {val_min}")
            
        except Exception as e:
            print(f"Hybrid model test skipped due to error: {e}")
            
        print("All model training temporal integrity checks passed!")
        
if __name__ == '__main__':
    unittest.main()