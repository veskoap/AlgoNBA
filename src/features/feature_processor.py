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
            
        elif feature_name == 'MATCHUP_COMPATIBILITY':
            pace_home = dependencies[0]  # PACE_mean_HOME_30D
            pace_away = dependencies[1]  # PACE_mean_AWAY_30D
            fg3a_home = dependencies[2]  # FG3A_mean_HOME_30D
            fg3a_away = dependencies[3]  # FG3A_mean_AWAY_30D
            
            # Calculate pace similarity (opposite of difference)
            pace_similarity = 1.0 / (1.0 + abs(df[pace_home] - df[pace_away]))
            
            # Calculate 3-point tendency similarity (opposite of difference)
            fg3_similarity = 1.0 / (1.0 + abs(df[fg3a_home] - df[fg3a_away]))
            
            # Combine into a single metric (higher = more similar styles)
            df[feature_name] = (pace_similarity + fg3_similarity) / 2.0
            
        elif feature_name == 'STYLE_ADVANTAGE':
            pace_home = dependencies[0]  # PACE_mean_HOME_30D
            pace_away = dependencies[1]  # PACE_mean_AWAY_30D
            fg3_pct_home = dependencies[2]  # FG3_PCT_mean_HOME_30D
            fg3_pct_away = dependencies[3]  # FG3_PCT_mean_AWAY_30D
            
            # Fast team vs slow team advantage
            pace_advantage = (df[pace_home] - df[pace_away]) / (df[pace_home] + df[pace_away] + 0.001)
            
            # 3-point shooting advantage
            shooting_advantage = (df[fg3_pct_home] - df[fg3_pct_away]) / (df[fg3_pct_home] + df[fg3_pct_away] + 0.001)
            
            # Combine into a style advantage metric
            df[feature_name] = pace_advantage + shooting_advantage
            
        elif feature_name == 'MATCHUP_HISTORY_SCORE':
            h2h_win_pct = dependencies[0]  # H2H_WIN_PCT
            h2h_margin = dependencies[1]  # H2H_AVG_MARGIN
            h2h_momentum = dependencies[2]  # H2H_MOMENTUM
            
            # Normalize margin to -1 to 1 range
            normalized_margin = df[h2h_margin] / 30.0  # Typical max margin
            normalized_margin = normalized_margin.clip(-1, 1)
            
            # Combine into a weighted matchup score
            df[feature_name] = (
                0.5 * df[h2h_win_pct] +  # 50% weight on win percentage
                0.3 * normalized_margin +  # 30% weight on scoring margin
                0.2 * df[h2h_momentum]    # 20% weight on recent momentum
            )
            
        elif feature_name == 'TRAVEL_FATIGUE':
            travel_distance = dependencies[0]  # TRAVEL_DISTANCE
            timezone_diff = dependencies[1]  # TIMEZONE_DIFF
            
            # Normalize distance (typical max domestic flight ~3000 miles)
            normalized_distance = df[travel_distance] / 3000.0
            
            # Combine distance and timezone effect
            # Timezone changes have more impact than pure distance
            df[feature_name] = normalized_distance + (abs(df[timezone_diff]) * 0.5)
            df[feature_name] = df[feature_name].clip(0, 1)  # Scale 0-1
            
        elif feature_name == 'LINEUP_IMPACT_DIFF':
            home_impact = dependencies[0]  # LINEUP_IMPACT_HOME
            away_impact = dependencies[1]  # LINEUP_IMPACT_AWAY
            
            # Simple difference in lineup impact scores
            df[feature_name] = df[home_impact] - df[away_impact]
    
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
        # Add GAME_ID for merging with other data sources
        features['GAME_ID'] = games['GAME_ID_HOME']
        features['GAME_ID_HOME'] = games['GAME_ID_HOME']
        
        # Add contextual features
        # Weekend game flag
        features['WEEKEND_GAME'] = (pd.to_datetime(games['GAME_DATE_HOME']).dt.dayofweek >= 5).astype(int)
        
        # Add nationally televised game flag (placeholder - would need actual TV schedule data)
        features['NATIONAL_TV'] = 0
        
        # Add rivalry matchup flag (placeholder - would need predefined rivalry pairs)
        features['RIVALRY_MATCHUP'] = 0

        # Ensure proper sorting of features DataFrame
        features = features.sort_values(['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'])
        
        # Add stadium-specific home advantage (would ideally be based on historical home win % by arena)
        # For now, use a simplified placeholder that could be replaced with actual data
        home_teams = games['TEAM_ID_HOME'].unique()
        stadium_advantage = {}
        
        for team_id in home_teams:
            team_home_games = games[games['TEAM_ID_HOME'] == team_id]
            if len(team_home_games) > 0:
                win_pct = (team_home_games['WL_HOME'] == 'W').mean()
                # Normalize around average home advantage of ~60%
                stadium_advantage[team_id] = (win_pct - 0.5) / 0.1  # Scale to make 60% → 1.0
        
        # Add stadium advantage to features
        features['STADIUM_HOME_ADVANTAGE'] = features['TEAM_ID_HOME'].map(
            stadium_advantage).fillna(1.0).clip(-2, 2)  # Clip to reasonable range

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
        
        # Add day of week feature for weekend games
        team_games['GAME_DAY'] = pd.to_datetime(team_games['GAME_DATE']).dt.dayofweek
        team_games['IS_WEEKEND'] = (team_games['GAME_DAY'] >= 5).astype(int)  # 5=Saturday, 6=Sunday

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
                'BLK': 'mean',
                'FGA': 'mean',
                'FGM': 'mean',
                'FG_PCT': 'mean',
                'FG3A': 'mean',
                'FG3M': 'mean',
                'FG3_PCT': 'mean',
                'IS_WEEKEND': 'mean'
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
            
            # Add weekend performance metrics
            weekend_games = team_games[team_games['IS_WEEKEND'] == 1]
            if not weekend_games.empty:
                weekend_data = weekend_games.sort_values(['TEAM_ID', 'GAME_DATE']).groupby('TEAM_ID')
                weekend_win_pct = weekend_data['WIN'].mean().reset_index()
                weekend_win_pct.columns = ['TEAM_ID', 'WEEKEND_WIN_PCT']
                
                # Check if we have any weekend data after grouping
                if not weekend_win_pct.empty:
                    # Merge with rolling_stats
                    rolling_stats = pd.merge(
                        rolling_stats,
                        weekend_win_pct,
                        on='TEAM_ID',
                        how='left'
                    )
                    
                    # Fill NaN values with overall win percentage
                    rolling_stats['WEEKEND_WIN_PCT'].fillna(rolling_stats['WIN_mean'], inplace=True)
                else:
                    # If no weekend data available, use overall win percentage
                    rolling_stats['WEEKEND_WIN_PCT'] = rolling_stats['WIN_mean']
            else:
                # If no weekend data available, use overall win percentage
                rolling_stats['WEEKEND_WIN_PCT'] = rolling_stats['WIN_mean']

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

        # Get player availability and injury features
        try:
            # Get seasons from games dataframe
            seasons = pd.to_datetime(games['GAME_DATE_HOME']).dt.year.unique()
            formatted_seasons = [f"{year}-{str(year+1)[-2:]}" for year in seasons]
            
            print(f"Getting player availability for season(s): {formatted_seasons}")
            
            # Create data loader instance to get player data
            from src.data.data_loader import NBADataLoader
            data_loader = NBADataLoader()
            
            player_avail_data = None
            
            # For each detected season, get player availability data
            for season in formatted_seasons:
                season_avail = data_loader.fetch_player_availability(season)
                
                if not season_avail.empty:
                    print(f"Successfully loaded player data for {season} with {len(season_avail)} records")
                    
                    if player_avail_data is None:
                        player_avail_data = season_avail
                    else:
                        player_avail_data = pd.concat([player_avail_data, season_avail], ignore_index=True)
            
            # If we have player data, merge it with features
            if player_avail_data is not None and not player_avail_data.empty:
                print(f"Merging {len(player_avail_data)} player availability records with features")
                
                # Split into home and away data
                home_player_data = player_avail_data[player_avail_data['IS_HOME'] == 1].copy()
                away_player_data = player_avail_data[player_avail_data['IS_HOME'] == 0].copy()
                
                # Rename columns for merging
                home_cols = {col: f"{col}_HOME" for col in home_player_data.columns 
                             if col not in ['GAME_ID', 'TEAM_ID', 'IS_HOME']}
                away_cols = {col: f"{col}_AWAY" for col in away_player_data.columns
                             if col not in ['GAME_ID', 'TEAM_ID', 'IS_HOME']}
                
                home_player_data = home_player_data.rename(columns=home_cols)
                away_player_data = away_player_data.rename(columns=away_cols)
                
                # Verify column existence before merge
                print(f"Features columns before merge: {features.columns[:5]}...")
                print(f"Home player data columns: {home_player_data.columns[:5]}...")
                
                # Check if we have the required columns for merging
                required_cols = {'GAME_ID', 'TEAM_ID'}
                if set(required_cols).issubset(home_player_data.columns):
                    # Identify correct game ID column in features
                    game_id_col = 'GAME_ID_HOME' if 'GAME_ID_HOME' in features.columns else 'GAME_ID'
                    
                    if game_id_col not in features.columns:
                        print(f"Warning: {game_id_col} not found in features. Available columns: {features.columns[:10]}")
                        
                        # If GAME_ID_HOME isn't available but GAME_ID is, use that
                        if 'GAME_ID' not in features.columns and 'GAME_DATE' in features.columns:
                            print("Adding GAME_ID column for merging")
                            # Create a temporary game ID for merging (not ideal but allows the process to continue)
                            features['GAME_ID'] = ['G' + str(i).zfill(10) for i in range(len(features))]
                            game_id_col = 'GAME_ID'
                    
                    # Merge home player data if we have a valid game ID column
                    if game_id_col in features.columns:
                        features = pd.merge(
                            features,
                            home_player_data,
                            left_on=[game_id_col, 'TEAM_ID_HOME'],
                            right_on=['GAME_ID', 'TEAM_ID'],
                            how='left'
                        )
                        
                        # Merge away player data
                        features = pd.merge(
                            features,
                            away_player_data,
                            left_on=[game_id_col, 'TEAM_ID_AWAY'],
                            right_on=['GAME_ID', 'TEAM_ID'],
                            how='left'
                        )
                    else:
                        print("Skipping player data merge due to missing key columns")
                else:
                    print(f"Skipping player data merge due to missing columns in player data. Found: {home_player_data.columns}")
                
                # Drop duplicate columns
                drop_cols = [col for col in features.columns if col in ['GAME_ID', 'TEAM_ID', 'IS_HOME']]
                features = features.drop(columns=drop_cols, errors='ignore')
                
                print(f"Successfully merged player data with {len(features)} features")
            else:
                print("No player availability data found to merge")
            
            # Add injury data
            print("Adding player injury features...")
            
            try:
                # Import the injury tracker
                from src.data.injury.injury_tracker import PlayerInjuryTracker
                
                # Initialize player impact scores dict
                player_impact_scores = {}
                
                # Extract player impact scores from our features if possible
                # This would be a more detailed implementation in production
                injury_tracker = PlayerInjuryTracker()
                
                # Generate injury features directly from the tracker
                injury_features = injury_tracker.generate_injury_features(
                    games, player_impact_scores
                )
                
                # If we have injury features, merge them into main features
                if not injury_features.empty:
                    print(f"Adding {len(injury_features.columns)} injury features")
                    features = pd.concat([features, injury_features], axis=1)
                
            except Exception as e:
                print(f"Error processing injury data: {e}")
                import traceback
                traceback.print_exc()
                
        except Exception as e:
            print(f"Error loading player availability data: {e}")
            import traceback
            traceback.print_exc()

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
            
            # Store additional style metrics for matchup analysis
            if all(col in game for col in ['PACE_HOME', 'PACE_AWAY']):
                h2h_stats[(home_team, away_team)]['pace_diff'].append(
                    game['PACE_HOME'] - game['PACE_AWAY']
                )
            
            if all(col in game for col in ['FG3_PCT_HOME', 'FG3_PCT_AWAY']):
                h2h_stats[(home_team, away_team)]['three_pt_diff'].append(
                    game['FG3_PCT_HOME'] - game['FG3_PCT_AWAY']
                )
                
            # Track weekend performance in matchups
            if 'GAME_DATE_HOME' in game:
                game_day = pd.to_datetime(game['GAME_DATE_HOME']).dayofweek
                is_weekend = 1 if game_day >= 5 else 0  # 5=Saturday, 6=Sunday
                h2h_stats[(home_team, away_team)]['weekend_games'].append(is_weekend)
                h2h_stats[(home_team, away_team)]['weekend_wins'].append(
                    1 if is_weekend == 1 and game['WL_HOME'] == 'W' else 0
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
                    'H2H_MOMENTUM': 0,
                    'H2H_STYLE_DIFF': 0,
                    'H2H_WEEKEND_WIN_PCT': 0.5,
                    'H2H_PACE_ADVANTAGE': 0,
                    'H2H_SHOOTING_ADVANTAGE': 0
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

                # Calculate style differences and advantages
                style_diff = 0
                pace_advantage = 0
                shooting_advantage = 0
                weekend_win_pct = 0.5
                
                if 'pace_diff' in h2h_stats[(home_team, away_team)] and recent_idx:
                    recent_pace_diffs = [h2h_stats[(home_team, away_team)]['pace_diff'][i] for i in recent_idx 
                                         if i < len(h2h_stats[(home_team, away_team)]['pace_diff'])]
                    if recent_pace_diffs:
                        pace_advantage = np.mean(recent_pace_diffs)
                
                if 'three_pt_diff' in h2h_stats[(home_team, away_team)] and recent_idx:
                    recent_three_pt_diffs = [h2h_stats[(home_team, away_team)]['three_pt_diff'][i] for i in recent_idx
                                            if i < len(h2h_stats[(home_team, away_team)]['three_pt_diff'])]
                    if recent_three_pt_diffs:
                        shooting_advantage = np.mean(recent_three_pt_diffs)
                
                # Combined style difference metric
                style_diff = abs(pace_advantage) + abs(shooting_advantage)
                
                # Weekend performance
                if 'weekend_games' in h2h_stats[(home_team, away_team)] and 'weekend_wins' in h2h_stats[(home_team, away_team)]:
                    recent_weekend_games = [h2h_stats[(home_team, away_team)]['weekend_games'][i] for i in recent_idx
                                           if i < len(h2h_stats[(home_team, away_team)]['weekend_games'])]
                    recent_weekend_wins = [h2h_stats[(home_team, away_team)]['weekend_wins'][i] for i in recent_idx
                                          if i < len(h2h_stats[(home_team, away_team)]['weekend_wins'])]
                    
                    weekend_games_count = sum(recent_weekend_games)
                    if weekend_games_count > 0:
                        weekend_win_pct = sum(recent_weekend_wins) / weekend_games_count
                    else:
                        weekend_win_pct = np.mean(recent_wins)  # default to overall win pct
                
                h2h_features_list.append({
                    'GAME_DATE': game_date,
                    'TEAM_ID_HOME': home_team,
                    'TEAM_ID_AWAY': away_team,
                    'H2H_GAMES': len(recent_idx),
                    'H2H_WIN_PCT': np.mean(recent_wins),
                    'H2H_AVG_MARGIN': np.mean(recent_margins),
                    'H2H_STREAK': streak,
                    'H2H_HOME_ADVANTAGE': np.mean([w for i, w in enumerate(recent_wins)
                                                 if i < len(recent_margins) and recent_margins[i] > 0]),
                    'H2H_MOMENTUM': momentum,
                    'H2H_STYLE_DIFF': style_diff,
                    'H2H_WEEKEND_WIN_PCT': weekend_win_pct,
                    'H2H_PACE_ADVANTAGE': pace_advantage,
                    'H2H_SHOOTING_ADVANTAGE': shooting_advantage
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
                    'H2H_MOMENTUM': 0.5,
                    'H2H_STYLE_DIFF': 0,
                    'H2H_WEEKEND_WIN_PCT': 0.5,
                    'H2H_PACE_ADVANTAGE': 0,
                    'H2H_SHOOTING_ADVANTAGE': 0
                })

        return pd.DataFrame(h2h_features_list)
        
    def calculate_seasonal_trends(self, games: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate seasonal trend adjustments with enhanced exponential decay weighting.
        
        Args:
            games: DataFrame containing merged home/away game data
            
        Returns:
            pd.DataFrame: Seasonal trend features
        """
        print("Calculating enhanced seasonal trends with exponential decay...")

        seasonal_trends = pd.DataFrame()

        try:
            # Convert game dates to day of season
            season_start = games['GAME_DATE_HOME'].min()
            games['DAYS_INTO_SEASON'] = (games['GAME_DATE_HOME'] - season_start).dt.days

            # Calculate rolling averages with seasonal weights for each window
            for window in DEFAULT_LOOKBACK_WINDOWS:
                # Apply exponential weighting based on recency
                # More aggressive decay for more recent games
                decay_factor = window / 10  # Adjust decay rate based on window size
                weights = np.exp(-np.arange(window) / decay_factor)
                
                # Normalize weights to sum to 1
                weights = weights / weights.sum()

                for team_type in ['HOME', 'AWAY']:
                    # Offensive trends with exponential decay
                    seasonal_trends[f'TREND_SCORE_{team_type}_{window}D'] = (
                        games.groupby('TEAM_ID_' + team_type)['PTS_' + team_type]
                        .rolling(window, min_periods=max(1, window//5))
                        .apply(lambda x: np.average(x, weights=weights[-len(x):]) if len(x) > 0 else np.nan)
                        .reset_index(0, drop=True)
                    )

                    # Win trends with exponential decay (based on PLUS_MINUS as a proxy for dominance)
                    seasonal_trends[f'TREND_WIN_{team_type}_{window}D'] = (
                        games.groupby('TEAM_ID_' + team_type)['PLUS_MINUS_' + team_type]
                        .rolling(window, min_periods=max(1, window//5))
                        .apply(lambda x: np.average(x, weights=weights[-len(x):]) if len(x) > 0 else np.nan)
                        .reset_index(0, drop=True)
                    )
                    
                    # Efficiency trends (points per possession proxy)
                    if all(col in games.columns for col in [f'PTS_{team_type}', f'TOV_{team_type}', f'FGA_{team_type}']):
                        # Calculate points per possession for each game
                        ppp = games[f'PTS_{team_type}'] / (games[f'FGA_{team_type}'] - games[f'TOV_{team_type}'] + 0.001)
                        
                        seasonal_trends[f'TREND_EFF_{team_type}_{window}D'] = (
                            games.groupby('TEAM_ID_' + team_type).apply(
                                lambda group: pd.Series(
                                    [
                                        np.average(
                                            ppp.loc[group.index][-window:], 
                                            weights=weights[-len(ppp.loc[group.index][-window:]):],
                                        ) if len(ppp.loc[group.index][-window:]) > 0 else np.nan
                                        for _ in range(len(group))
                                    ],
                                    index=group.index
                                )
                            )
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
                'PLUS_MINUS_AWAY': 'mean',
                'FG3_PCT_HOME': 'mean',
                'FG3_PCT_AWAY': 'mean',
                'FG_PCT_HOME': 'mean',
                'FG_PCT_AWAY': 'mean'
            })

            # Add segment adjustments
            for team_type in ['HOME', 'AWAY']:
                seasonal_trends[f'SEGMENT_ADJ_{team_type}'] = (
                    games['SEASON_SEGMENT'].map(
                        segment_stats[f'PLUS_MINUS_{team_type}']
                    )
                )
                
                # Add new segment shooting adjustments
                seasonal_trends[f'SEGMENT_SHOOTING_{team_type}'] = (
                    games['SEASON_SEGMENT'].map(
                        segment_stats[f'FG_PCT_{team_type}']
                    )
                )
                
                seasonal_trends[f'SEGMENT_3PT_{team_type}'] = (
                    games['SEASON_SEGMENT'].map(
                        segment_stats[f'FG3_PCT_{team_type}']
                    )
                )

        except Exception as e:
            print(f"Error calculating seasonal trends: {e}")
            print(f"Error details: {str(e)}")
            import traceback
            traceback.print_exc()

        return seasonal_trends.fillna(0)
        
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