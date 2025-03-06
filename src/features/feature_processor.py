"""
Feature engineering code for NBA prediction model.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

from src.utils.helpers import calculate_advanced_stats, calculate_travel_impact, safe_divide
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS, FEATURE_GROUPS, FEATURE_REGISTRY


class FeatureTransformer:
    """Feature transformation and management class."""
    
    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize the feature transformer.
        
        Args:
            lookback_windows: List of day windows for rolling statistics (default: [7, 14, 30, 60])
        """
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        self.required_features: Set[str] = set()
        self.available_features: Set[str] = set()
        self.feature_transformers: Dict[str, Any] = {}
        
    def register_feature(self, feature_name: str) -> None:
        """
        Register a feature as required for the model.
        
        Args:
            feature_name: Name of the feature to register
        """
        self.required_features.add(feature_name)
        
        # If this feature has a time window format (like FEATURE_NAME_30D)
        base_name = self._get_base_feature_name(feature_name)
        if base_name in FEATURE_REGISTRY:
            feature_info = FEATURE_REGISTRY[base_name]
            
            # Also register any dependencies
            if 'dependencies' in feature_info:
                for dep in feature_info['dependencies']:
                    # If the dependency has windows, register all window versions
                    if 'windows' in feature_info and feature_info['windows']:
                        # Extract window from the original feature if it exists
                        window = self._extract_window(feature_name)
                        if window:
                            self.register_feature(f"{dep}_{window}D")
                        else:
                            # Register all windows for this dependency
                            for w in feature_info['windows']:
                                self.register_feature(f"{dep}_{w}D")
                    else:
                        # No windows, register the dependency directly
                        self.register_feature(dep)
    
    def register_features(self, feature_names: List[str]) -> None:
        """
        Register multiple features as required.
        
        Args:
            feature_names: List of feature names to register
        """
        for feature in feature_names:
            self.register_feature(feature)
    
    def validate_features(self, features: pd.DataFrame) -> List[str]:
        """
        Validate that all required features are available.
        
        Args:
            features: DataFrame of features
            
        Returns:
            List of missing features
        """
        self.available_features = set(features.columns)
        missing = list(self.required_features - self.available_features)
        return missing
    
    def transform_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Transform features based on registered transformations.
        
        Args:
            features: Input DataFrame with base features
            
        Returns:
            DataFrame with transformed features
        """
        result = features.copy()
        
        # Process each registered feature that needs transformation
        for feature_name in self.required_features:
            # Skip if feature already exists
            if feature_name in result.columns:
                continue
                
            base_name = self._get_base_feature_name(feature_name)
            
            # If this is a registered feature type
            if base_name in FEATURE_REGISTRY:
                window = self._extract_window(feature_name)
                feature_info = FEATURE_REGISTRY[base_name]
                
                # Skip if this is a window feature but the window isn't supported
                if window and feature_info['windows'] and int(window) not in feature_info['windows']:
                    continue
                    
                # Generate the feature based on its type
                if feature_info['type'] == 'derived':
                    self._derive_feature(result, feature_name, feature_info, window)
                elif feature_info['type'] == 'interaction':
                    self._create_interaction_feature(result, feature_name, feature_info)
                    
        return result
    
    def _derive_feature(self, df: pd.DataFrame, feature_name: str, 
                       feature_info: Dict, window: str = None) -> None:
        """
        Derive a feature based on its definition and dependencies.
        
        Args:
            df: DataFrame to modify
            feature_name: Name of the feature to derive
            feature_info: Feature information from registry
            window: Time window (if applicable)
        """
        # Handle specific feature derivations based on feature type
        base_name = self._get_base_feature_name(feature_name)
        
        # Skip if feature already exists
        if feature_name in df.columns:
            return
            
        # Generate windowed column names for dependencies
        dependencies = []
        for dep in feature_info['dependencies']:
            if window and self._should_apply_window(dep):
                dependencies.append(f"{dep}_{window}D")
            else:
                dependencies.append(dep)
                
        # Skip if any dependencies are missing
        if not all(dep in df.columns for dep in dependencies):
            # Use default values for missing dependencies
            for dep in dependencies:
                if dep not in df.columns:
                    df[dep] = 0
        
        # Derive specific features
        if base_name == 'WIN_PCT_DIFF':
            # Get windoweq column names
            home_col = dependencies[0]  # WIN_PCT_HOME_{window}D
            away_col = dependencies[1]  # WIN_PCT_AWAY_{window}D
            df[feature_name] = df[home_col] - df[away_col]
            
        elif base_name in ['OFF_RTG_DIFF', 'DEF_RTG_DIFF', 'NET_RTG_DIFF', 'PACE_DIFF']:
            home_col = dependencies[0]  # RTG_mean_HOME_{window}D
            away_col = dependencies[1]  # RTG_mean_AWAY_{window}D
            df[feature_name] = df[home_col] - df[away_col]
            
        elif base_name == 'EFF_DIFF':
            pts_home = dependencies[0]  # PTS_mean_HOME_{window}D
            pts_away = dependencies[1]  # PTS_mean_AWAY_{window}D
            tov_home = dependencies[2]  # TOV_mean_HOME_{window}D
            tov_away = dependencies[3]  # TOV_mean_AWAY_{window}D
            
            # Calculate efficiency (points per turnover)
            home_eff = safe_divide(df[pts_home], df[tov_home], index=df.index)
            away_eff = safe_divide(df[pts_away], df[tov_away], index=df.index)
            df[feature_name] = home_eff - away_eff
            
        elif base_name == 'HOME_CONSISTENCY':
            pts_std = dependencies[0]  # PTS_std_HOME_{window}D
            pts_mean = dependencies[1]  # PTS_mean_HOME_{window}D
            df[feature_name] = safe_divide(df[pts_std], df[pts_mean], index=df.index)
            
        elif base_name == 'AWAY_CONSISTENCY':
            pts_std = dependencies[0]  # PTS_std_AWAY_{window}D
            pts_mean = dependencies[1]  # PTS_mean_AWAY_{window}D
            df[feature_name] = safe_divide(df[pts_std], df[pts_mean], index=df.index)
            
        elif base_name == 'FATIGUE_DIFF':
            home_col = dependencies[0]  # FATIGUE_HOME_{window}D
            away_col = dependencies[1]  # FATIGUE_AWAY_{window}D
            df[feature_name] = df[home_col] - df[away_col]
            
        elif base_name == 'REST_DIFF':
            rest_home = dependencies[0]  # REST_DAYS_HOME
            rest_away = dependencies[1]  # REST_DAYS_AWAY
            df[feature_name] = df[rest_home] - df[rest_away]
            
        elif base_name == 'H2H_RECENCY_WEIGHT':
            h2h_win_pct = dependencies[0]  # H2H_WIN_PCT
            days_since = dependencies[1]  # DAYS_SINCE_H2H
            df[feature_name] = safe_divide(
                df[h2h_win_pct],
                np.log1p(df[days_since]),
                fill_value=0.5,
                index=df.index
            )
    
    def _create_interaction_feature(self, df: pd.DataFrame, feature_name: str, 
                                  feature_info: Dict) -> None:
        """
        Create an interaction feature between multiple input features.
        
        Args:
            df: DataFrame to modify
            feature_name: Name of the interaction feature
            feature_info: Feature information from registry
        """
        # Skip if feature already exists
        if feature_name in df.columns:
            return
            
        # Ensure all dependencies exist
        dependencies = feature_info['dependencies']
        if not all(dep in df.columns for dep in dependencies):
            # Use default values for missing dependencies
            for dep in dependencies:
                if dep not in df.columns:
                    df[dep] = 0
        
        # Handle specific interaction features
        if feature_name == 'FATIGUE_TRAVEL_INTERACTION':
            fatigue_diff = dependencies[0]  # FATIGUE_DIFF_14D
            travel_dist = dependencies[1]  # TRAVEL_DISTANCE
            df[feature_name] = df[fatigue_diff] * df[travel_dist] / 1000
            
        elif feature_name == 'MOMENTUM_REST_INTERACTION':
            win_pct_diff = dependencies[0]  # WIN_PCT_DIFF_30D
            rest_diff = dependencies[1]  # REST_DIFF
            df[feature_name] = df[win_pct_diff] * df[rest_diff]
    
    def _get_base_feature_name(self, feature_name: str) -> str:
        """
        Extract the base feature name without window suffix.
        
        Args:
            feature_name: Feature name possibly with window suffix
            
        Returns:
            Base feature name
        """
        # Handle features with window suffix like 'FEATURE_NAME_30D'
        if '_D' in feature_name:
            parts = feature_name.split('_')
            for i, part in enumerate(parts):
                if part.endswith('D') and part[:-1].isdigit():
                    return '_'.join(parts[:i])
        return feature_name
        
    def _extract_window(self, feature_name: str) -> str:
        """
        Extract the window value from a feature name.
        
        Args:
            feature_name: Feature name with potential window suffix
            
        Returns:
            Window value (without 'D') or None if no window
        """
        if '_D' in feature_name:
            parts = feature_name.split('_')
            for part in parts:
                if part.endswith('D') and part[:-1].isdigit():
                    return part[:-1]
        return None
        
    def _should_apply_window(self, column_name: str) -> bool:
        """
        Determine if a window should be applied to a column name.
        
        Args:
            column_name: Column name to check
            
        Returns:
            True if window should be applied
        """
        # Don't apply windows to columns that already have them
        if '_D' in column_name:
            return False
            
        # Don't apply windows to specific feature types
        no_window_prefixes = ['REST_DAYS_', 'H2H_', 'DAYS_SINCE_', 'LAST_GAME_']
        return not any(column_name.startswith(prefix) for prefix in no_window_prefixes)


class NBAFeatureProcessor:
    """Class for engineering features from NBA game data."""
    
    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize the feature processor.
        
        Args:
            lookback_windows: List of day windows for rolling statistics (default: [7, 14, 30, 60])
        """
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        self.feature_transformer = FeatureTransformer(self.lookback_windows)
        
    def calculate_team_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced team statistics using vectorized operations.
        
        Args:
            games: DataFrame containing merged home/away game data
            
        Returns:
            pd.DataFrame: Enhanced feature set with team statistics
        """
        print("Calculating advanced team statistics...")

        # Initialize features DataFrame and ensure proper sorting
        features = pd.DataFrame()
        features['GAME_DATE'] = pd.to_datetime(games['GAME_DATE_HOME'])
        features['TEAM_ID_HOME'] = games['TEAM_ID_HOME']
        features['TEAM_ID_AWAY'] = games['TEAM_ID_AWAY']
        features['TARGET'] = (games['WL_HOME'] == 'W').astype(int)

        # Ensure proper sorting of features DataFrame
        features = features.sort_values(['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'])

        # Calculate advanced stats for all games
        advanced_stats = games.apply(calculate_advanced_stats, axis=1)
        advanced_df = pd.DataFrame(advanced_stats.tolist())

        # Combine games with advanced stats
        enhanced_games = pd.concat([games, advanced_df], axis=1)

        # Create unified team games dataframe
        home_games = enhanced_games[[col for col in enhanced_games.columns if '_HOME' in col]].copy()
        away_games = enhanced_games[[col for col in enhanced_games.columns if '_AWAY' in col]].copy()

        # Rename columns to remove HOME/AWAY suffix for processing
        home_games.columns = [col.replace('_HOME', '') for col in home_games.columns]
        away_games.columns = [col.replace('_AWAY', '') for col in away_games.columns]

        # Add location indicator
        home_games['IS_HOME'] = 1
        away_games['IS_HOME'] = 0

        # Combine all games for each team
        team_games = pd.concat([home_games, away_games])
        team_games['WIN'] = (team_games['WL'] == 'W').astype(int)
        team_games = team_games.sort_values(['TEAM_ID', 'GAME_DATE'])

        # Calculate rest days for each team
        team_games['REST_DAYS'] = team_games.groupby('TEAM_ID')['GAME_DATE'].diff().dt.days.fillna(7)

        # Create separate rest days dataframes for home and away teams
        rest_days_home = team_games[['TEAM_ID', 'GAME_DATE', 'REST_DAYS']].copy()
        rest_days_away = rest_days_home.copy()

        rest_days_home.columns = ['TEAM_ID_HOME', 'GAME_DATE', 'REST_DAYS_HOME']
        rest_days_away.columns = ['TEAM_ID_AWAY', 'GAME_DATE', 'REST_DAYS_AWAY']

        # Merge rest days into features
        features = pd.merge_asof(
            features.sort_values(['GAME_DATE', 'TEAM_ID_HOME']),
            rest_days_home.sort_values(['GAME_DATE', 'TEAM_ID_HOME']),
            on='GAME_DATE',
            by='TEAM_ID_HOME',
            direction='backward'
        )

        features = pd.merge_asof(
            features.sort_values(['GAME_DATE', 'TEAM_ID_AWAY']),
            rest_days_away.sort_values(['GAME_DATE', 'TEAM_ID_AWAY']),
            on='GAME_DATE',
            by='TEAM_ID_AWAY',
            direction='backward'
        )

        # Calculate additional metrics
        team_games['PACE'] = team_games['FGA'] + 0.4 * team_games['FTA'] - 1.07 * (
            team_games['OREB'] / (team_games['OREB'] + team_games['DREB'])
        ) * (team_games['FGA'] - team_games['FGM']) + team_games['TOV']

        # Calculate rolling stats for each window
        for window in self.lookback_windows:
            print(f"Processing {window}-day window...")

            stats_df = team_games.set_index('GAME_DATE').sort_index()

            # Rolling statistics - use closed='left' to exclude current day
            # This ensures we only use data strictly before the current date for each calculation
            # preventing any form of data leakage from future games
            rolling_stats = stats_df.groupby('TEAM_ID').rolling(
                window=f'{window}D',
                min_periods=1,
                closed='left'  # Ensures only data prior to current day is used
            ).agg({
                'WIN': ['count', 'mean'],
                'IS_HOME': 'mean',
                'PTS': ['mean', 'std'],
                'OFF_RTG': ['mean', 'std'],
                'DEF_RTG': ['mean', 'std'],
                'NET_RTG': 'mean',
                'EFG_PCT': 'mean',
                'TS_PCT': 'mean',
                'PACE': 'mean',
                'AST': 'mean',
                'TOV': 'mean',
                'OREB': 'mean',
                'DREB': 'mean',
                'STL': 'mean',
                'BLK': 'mean'
            })

            # Flatten multi-index columns
            rolling_stats.columns = [
                f"{col[0]}_{col[1]}" if col[1] != '' else col[0]
                for col in rolling_stats.columns
            ]

            rolling_stats = rolling_stats.reset_index()
            rolling_stats['GAME_DATE'] = pd.to_datetime(rolling_stats['GAME_DATE'])

            # Calculate fatigue (games in last 7 days)
            games_last_7 = stats_df.groupby('TEAM_ID').rolling('7D', closed='left').count()['WIN']
            rolling_stats['FATIGUE'] = games_last_7.reset_index()['WIN']

            # Add streak information
            streak_data = team_games.sort_values(['TEAM_ID', 'GAME_DATE']).groupby('TEAM_ID')['WIN']
            streak_lengths = streak_data.rolling(window=10, min_periods=1, closed='left').apply(
                lambda x: (x == x.iloc[-1]).sum() if len(x) > 0 else 0
            )
            rolling_stats['STREAK'] = streak_lengths.reset_index()['WIN']

            # Create separate home and away stats
            home_stats = rolling_stats.copy()
            away_stats = rolling_stats.copy()

            # Rename columns with window and location suffix
            home_cols = {
                col: f"{col}_HOME_{window}D" if col not in ['TEAM_ID', 'GAME_DATE'] else col
                for col in home_stats.columns
            }
            away_cols = {
                col: f"{col}_AWAY_{window}D" if col not in ['TEAM_ID', 'GAME_DATE'] else col
                for col in away_stats.columns
            }

            home_stats = home_stats.rename(columns=home_cols)
            home_stats = home_stats.rename(columns={'TEAM_ID': 'TEAM_ID_HOME'})

            away_stats = away_stats.rename(columns=away_cols)
            away_stats = away_stats.rename(columns={'TEAM_ID': 'TEAM_ID_AWAY'})

            # Ensure proper sorting for merge_asof
            home_stats = home_stats.sort_values(['GAME_DATE', 'TEAM_ID_HOME'])
            away_stats = away_stats.sort_values(['GAME_DATE', 'TEAM_ID_AWAY'])

            # Merge with features
            features = pd.merge_asof(
                features.sort_values(['GAME_DATE', 'TEAM_ID_HOME']),
                home_stats,
                on='GAME_DATE',
                by='TEAM_ID_HOME',
                direction='backward'
            )

            features = pd.merge_asof(
                features.sort_values(['GAME_DATE', 'TEAM_ID_AWAY']),
                away_stats,
                on='GAME_DATE',
                by='TEAM_ID_AWAY',
                direction='backward'
            )

        # Calculate head-to-head features
        h2h_stats = self.calculate_h2h_features(games)

        # Sort both DataFrames before merging
        features = features.sort_values(['TEAM_ID_HOME', 'TEAM_ID_AWAY', 'GAME_DATE'])
        h2h_stats = h2h_stats.sort_values(['TEAM_ID_HOME', 'TEAM_ID_AWAY', 'GAME_DATE'])

        features = features.merge(
            h2h_stats,
            on=['TEAM_ID_HOME', 'TEAM_ID_AWAY', 'GAME_DATE'],
            how='left'
        )

        return features.fillna(0)
        
    def calculate_h2h_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate head-to-head statistics between teams using highly optimized operations.
        
        Args:
            games: DataFrame containing merged home/away game data
            
        Returns:
            pd.DataFrame: Head-to-head features
        """
        print("Calculating head-to-head features using optimized method...")

        # Convert to datetime
        games = games.copy()
        games['GAME_DATE_HOME'] = pd.to_datetime(games['GAME_DATE_HOME'])

        # Create forward matches dataframe
        forward_matches = pd.DataFrame({
            'date': games['GAME_DATE_HOME'],
            'team1': games['TEAM_ID_HOME'],
            'team2': games['TEAM_ID_AWAY'],
            'win': games['WL_HOME'] == 'W'
        }).reset_index(drop=True)

        # Create reverse matches dataframe
        reverse_matches = pd.DataFrame({
            'date': games['GAME_DATE_HOME'],
            'team1': games['TEAM_ID_AWAY'],
            'team2': games['TEAM_ID_HOME'],
            'win': games['WL_HOME'] != 'W'
        }).reset_index(drop=True)

        # Combine all matchups
        all_matches = pd.concat([forward_matches, reverse_matches], ignore_index=True)
        all_matches = all_matches.sort_values('date').reset_index(drop=True)

        # Initialize results DataFrame
        results = pd.DataFrame({
            'GAME_DATE': games['GAME_DATE_HOME'],
            'TEAM_ID_HOME': games['TEAM_ID_HOME'],
            'TEAM_ID_AWAY': games['TEAM_ID_AWAY'],
            'H2H_GAMES': 0,
            'H2H_WIN_PCT': 0.5,
            'DAYS_SINCE_H2H': 365,
            'LAST_GAME_HOME': 0
        })

        # Process each game
        for idx in range(len(games)):
            current_date = games.iloc[idx]['GAME_DATE_HOME']
            home_team = games.iloc[idx]['TEAM_ID_HOME']
            away_team = games.iloc[idx]['TEAM_ID_AWAY']

            # Find previous matchups
            previous_matches = all_matches[
                (all_matches['date'] < current_date) &
                (
                    ((all_matches['team1'] == home_team) & (all_matches['team2'] == away_team)) |
                    ((all_matches['team1'] == away_team) & (all_matches['team2'] == home_team))
                )
            ]

            if len(previous_matches) > 0:
                # Calculate head-to-head stats
                previous_matches = previous_matches.sort_values('date')
                last_match = previous_matches.iloc[-1]

                h2h_games = len(previous_matches)
                home_perspective_matches = previous_matches[previous_matches['team1'] == home_team]
                h2h_win_pct = 0.5
                if h2h_games > 0 and len(home_perspective_matches) > 0:
                    h2h_win_pct = len(home_perspective_matches[home_perspective_matches['win']]) / len(home_perspective_matches)
                days_since = (current_date - last_match['date']).days
                last_game_home = int(last_match['team1'] == home_team)

                # Update results
                results.iloc[idx, results.columns.get_loc('H2H_GAMES')] = h2h_games
                results.iloc[idx, results.columns.get_loc('H2H_WIN_PCT')] = h2h_win_pct
                results.iloc[idx, results.columns.get_loc('DAYS_SINCE_H2H')] = days_since
                results.iloc[idx, results.columns.get_loc('LAST_GAME_HOME')] = last_game_home

        print(f"Calculated head-to-head features for {len(results)} matchups")
        return results
        
    def calculate_enhanced_h2h_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate enhanced head-to-head features with detailed matchup analysis.
        
        Args:
            games: DataFrame containing merged home/away game data
            
        Returns:
            pd.DataFrame: Enhanced head-to-head features
        """
        print("Calculating enhanced head-to-head features...")

        # Create a copy of games for manipulation
        h2h_features = games.copy()  # noqa: F841

        # Calculate basic H2H stats
        h2h_stats = defaultdict(lambda: defaultdict(list))

        for _, game in games.iterrows():
            home_team = game['TEAM_ID_HOME']
            away_team = game['TEAM_ID_AWAY']
            game_date = pd.to_datetime(game['GAME_DATE_HOME'])

            # Store game result
            h2h_stats[(home_team, away_team)]['dates'].append(game_date)
            h2h_stats[(home_team, away_team)]['margins'].append(
                game['PTS_HOME'] - game['PTS_AWAY']
            )
            h2h_stats[(home_team, away_team)]['home_wins'].append(
                1 if game['WL_HOME'] == 'W' else 0
            )

        # Calculate enhanced H2H features
        h2h_features_list = []

        for _, game in games.iterrows():
            home_team = game['TEAM_ID_HOME']
            away_team = game['TEAM_ID_AWAY']
            game_date = pd.to_datetime(game['GAME_DATE_HOME'])

            # Get previous matchups
            prev_dates = [d for d in h2h_stats[(home_team, away_team)]['dates']
                         if d < game_date]

            if not prev_dates:
                h2h_features_list.append({
                    'GAME_DATE': game_date,
                    'TEAM_ID_HOME': home_team,
                    'TEAM_ID_AWAY': away_team,
                    'H2H_GAMES': 0,
                    'H2H_WIN_PCT': 0.5,
                    'H2H_AVG_MARGIN': 0,
                    'H2H_STREAK': 0,
                    'H2H_HOME_ADVANTAGE': 0,
                    'H2H_MOMENTUM': 0
                })
                continue

            # Calculate enhanced stats
            recent_idx = [i for i, d in enumerate(prev_dates)
                         if (game_date - d).days <= 365]

            if recent_idx:
                recent_margins = [h2h_stats[(home_team, away_team)]['margins'][i]
                                for i in recent_idx]
                recent_wins = [h2h_stats[(home_team, away_team)]['home_wins'][i]
                             for i in recent_idx]

                # Calculate streak (last 5 games)
                streak = sum(1 for w in recent_wins[-5:] if w == 1)

                # Calculate momentum (weighted recent performance)
                weights = np.exp(-np.arange(len(recent_wins)) / 5)  # Exponential decay
                momentum = np.average(recent_wins, weights=weights) if len(recent_wins) > 0 else 0.5

                h2h_features_list.append({
                    'GAME_DATE': game_date,
                    'TEAM_ID_HOME': home_team,
                    'TEAM_ID_AWAY': away_team,
                    'H2H_GAMES': len(recent_idx),
                    'H2H_WIN_PCT': np.mean(recent_wins),
                    'H2H_AVG_MARGIN': np.mean(recent_margins),
                    'H2H_STREAK': streak,
                    'H2H_HOME_ADVANTAGE': np.mean([w for i, w in enumerate(recent_wins)
                                                 if recent_margins[i] > 0]),
                    'H2H_MOMENTUM': momentum
                })
            else:
                h2h_features_list.append({
                    'GAME_DATE': game_date,
                    'TEAM_ID_HOME': home_team,
                    'TEAM_ID_AWAY': away_team,
                    'H2H_GAMES': 0,
                    'H2H_WIN_PCT': 0.5,
                    'H2H_AVG_MARGIN': 0,
                    'H2H_STREAK': 0,
                    'H2H_HOME_ADVANTAGE': 0,
                    'H2H_MOMENTUM': 0.5
                })

        return pd.DataFrame(h2h_features_list)
        
    def calculate_seasonal_trends(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonal trend adjustments.
        
        Args:
            games: DataFrame containing merged home/away game data
            
        Returns:
            pd.DataFrame: Seasonal trend features
        """
        print("Calculating seasonal trends...")

        seasonal_trends = pd.DataFrame()

        try:
            # Convert game dates to day of season
            season_start = games['GAME_DATE_HOME'].min()
            games['DAYS_INTO_SEASON'] = (games['GAME_DATE_HOME'] - season_start).dt.days

            # Calculate rolling averages with seasonal weights
            for window in [14, 30, 60]:
                # Apply exponential weighting based on recency
                weights = np.exp(-np.arange(window) / (window/2))

                for team_type in ['HOME', 'AWAY']:
                    # Offensive trends
                    seasonal_trends[f'OFF_TREND_{team_type}_{window}D'] = (
                        games.groupby('TEAM_ID_' + team_type)['PTS_' + team_type]
                        .rolling(window, min_periods=1)
                        .apply(lambda x: np.average(x, weights=weights[-len(x):]))
                        .reset_index(0, drop=True)
                    )

                    # Defensive trends
                    seasonal_trends[f'DEF_TREND_{team_type}_{window}D'] = (
                        games.groupby('TEAM_ID_' + team_type)['PLUS_MINUS_' + team_type]
                        .rolling(window, min_periods=1)
                        .apply(lambda x: np.average(x, weights=weights[-len(x):]))
                        .reset_index(0, drop=True)
                    )

            # Add season segment indicators
            games['SEASON_SEGMENT'] = pd.cut(
                games['DAYS_INTO_SEASON'],
                bins=[0, 41, 82, 123, 164],
                labels=['Early', 'Mid', 'Late', 'Final']
            )

            # Calculate segment-specific statistics
            segment_stats = games.groupby('SEASON_SEGMENT').agg({
                'PTS_HOME': 'mean',
                'PTS_AWAY': 'mean',
                'PLUS_MINUS_HOME': 'mean',
                'PLUS_MINUS_AWAY': 'mean'
            })

            # Add segment adjustments
            for team_type in ['HOME', 'AWAY']:
                seasonal_trends[f'SEGMENT_ADJ_{team_type}'] = (
                    games['SEASON_SEGMENT'].map(
                        segment_stats[f'PLUS_MINUS_{team_type}']
                    )
                )

        except Exception as e:
            print(f"Error calculating seasonal trends: {e}")

        return seasonal_trends
        
    def prepare_features(self, stats_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare enhanced features with advanced metrics and interaction terms.
        Uses the feature transformer to ensure consistency.
        
        Args:
            stats_df: DataFrame containing team statistics
            
        Returns:
            tuple: (features_df, target_series)
        """
        print("Preparing enhanced feature set with feature transformer...")
        
        # Create a dictionary to store all columns
        feature_dict = {'GAME_DATE': stats_df['GAME_DATE']}
        target = stats_df['TARGET']
        
        # Copy all base statistics from stats_df
        for col in stats_df.columns:
            if col != 'TARGET' and col != 'GAME_DATE':
                feature_dict[col] = stats_df[col]
                
        # Create DataFrame all at once to avoid fragmentation
        features = pd.DataFrame(feature_dict, index=stats_df.index)
        
        # Register all standard features from FEATURE_REGISTRY
        for base_feature, info in FEATURE_REGISTRY.items():
            # If the feature has window variants, register for each window
            if info['windows']:
                for window in info['windows']:
                    self.feature_transformer.register_feature(f"{base_feature}_{window}D")
            else:
                # Register the feature without window
                self.feature_transformer.register_feature(base_feature)
        
        # Transform features using the feature transformer
        enhanced_features = self.feature_transformer.transform_features(features)
        
        # Clean up the data
        enhanced_features = enhanced_features.replace([np.inf, -np.inf], np.nan)
        enhanced_features = enhanced_features.fillna(0)
        
        # Clip extreme values for numerical columns
        for col in enhanced_features.columns:
            if col not in ['GAME_DATE', 'TARGET'] and enhanced_features[col].dtype in [np.float64, np.int64]:
                q1 = enhanced_features[col].quantile(0.01)
                q99 = enhanced_features[col].quantile(0.99)
                enhanced_features[col] = enhanced_features[col].clip(q1, q99)
        
        # Print feature summary
        feature_cols = [col for col in enhanced_features.columns 
                       if col not in ['GAME_DATE'] and enhanced_features[col].dtype in [np.float64, np.int64]]
        print(f"\nCreated {len(feature_cols)} features:")
        for prefix, description in FEATURE_GROUPS.items():
            related_features = [col for col in feature_cols if prefix in col]
            print(f"{description}: {len(related_features)} features")
        
        # Return only numerical features (excluding date) and target
        return enhanced_features, target
        
    def prepare_enhanced_features(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced feature set with all new metrics.
        This is now a wrapper around prepare_features for API compatibility.
        
        Args:
            stats_df: DataFrame containing team statistics
            
        Returns:
            pd.DataFrame: Enhanced features
        """
        print("Using standardized feature generation pipeline...")
        
        try:
            # Make sure stats_df has required date column
            if 'GAME_DATE_HOME' in stats_df.columns and 'GAME_DATE' not in stats_df.columns:
                stats_df = stats_df.copy()
                stats_df['GAME_DATE'] = pd.to_datetime(stats_df['GAME_DATE_HOME'])
            
            # Make sure stats_df has a TARGET column (for prediction scenarios)
            if 'TARGET' not in stats_df.columns:
                if 'WL_HOME' in stats_df.columns:
                    stats_df = stats_df.copy()
                    stats_df['TARGET'] = (stats_df['WL_HOME'] == 'W').astype(int)
                else:
                    stats_df = stats_df.copy()
                    stats_df['TARGET'] = 0  # Default for prediction
            
            # Use the same features method for consistency
            features, _ = self.prepare_features(stats_df)
            
            # For API compatibility, ensure TARGET column is included
            if 'TARGET' not in features.columns:
                target_value = stats_df['TARGET'] if 'TARGET' in stats_df.columns else 0
                features = pd.concat([features, pd.DataFrame({'TARGET': target_value}, index=features.index)], axis=1)
            
            print(f"Successfully created {len(features.columns)-2} features")  # -2 for GAME_DATE and TARGET
            return features
            
        except Exception as e:
            print(f"Error preparing enhanced features: {e}")
            print("DataFrame columns:", stats_df.columns.tolist())
            raise