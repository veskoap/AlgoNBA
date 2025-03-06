"""
Feature engineering code for NBA prediction model.
"""
import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from src.utils.helpers import calculate_advanced_stats, calculate_travel_impact, safe_divide
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS, FEATURE_GROUPS


class NBAFeatureProcessor:
    """Class for engineering features from NBA game data."""
    
    def __init__(self, lookback_windows: List[int] = None):
        """
        Initialize the feature processor.
        
        Args:
            lookback_windows: List of day windows for rolling statistics (default: [7, 14, 30, 60])
        """
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        
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

            # Rolling statistics
            rolling_stats = stats_df.groupby('TEAM_ID').rolling(
                window=f'{window}D',
                min_periods=1,
                closed='left'
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
        
        Args:
            stats_df: DataFrame containing team statistics
            
        Returns:
            tuple: (features_df, target_series)
        """
        print("Preparing enhanced feature set...")

        feature_dict = {}

        # Process each time window
        for window in self.lookback_windows:
            print(f"Processing {window}-day window features...")

            window_str = f'_{window}D'

            # Helper function to get stats safely with window parameter
            def get_stat(stat_name, side, window_size=None, default=0):
                if window_size is None:
                    window_size = window
                col_name = f"{stat_name}_{side}_{window_size}D"
                return stats_df[col_name].fillna(default) if col_name in stats_df else pd.Series(default, index=stats_df.index)

            # Basic win percentage features
            home_win_pct = get_stat('WIN_mean', 'HOME')
            away_win_pct = get_stat('WIN_mean', 'AWAY')

            feature_dict.update({
                f'WIN_PCT_DIFF{window_str}': home_win_pct - away_win_pct,
                f'WIN_PCT_HOME{window_str}': home_win_pct,
                f'WIN_PCT_AWAY{window_str}': away_win_pct,
            })

            # Advanced metric differentials
            for metric in ['OFF_RTG', 'DEF_RTG', 'NET_RTG', 'PACE']:
                home_stat = get_stat(f'{metric}_mean', 'HOME')
                away_stat = get_stat(f'{metric}_mean', 'AWAY')
                feature_dict[f'{metric}_DIFF{window_str}'] = home_stat - away_stat

            # Efficiency and consistency metrics
            home_pts = get_stat('PTS_mean', 'HOME')
            away_pts = get_stat('PTS_mean', 'AWAY')
            home_tov = get_stat('TOV_mean', 'HOME', default=1)
            away_tov = get_stat('TOV_mean', 'AWAY', default=1)

            feature_dict.update({
                f'EFF_DIFF{window_str}': safe_divide(home_pts, home_tov, index=stats_df.index) - 
                                         safe_divide(away_pts, away_tov, index=stats_df.index),
                f'HOME_CONSISTENCY{window_str}': safe_divide(
                    get_stat('PTS_std', 'HOME'),
                    home_pts,
                    fill_value=1,
                    index=stats_df.index
                ),
                f'AWAY_CONSISTENCY{window_str}': safe_divide(
                    get_stat('PTS_std', 'AWAY'),
                    away_pts,
                    fill_value=1,
                    index=stats_df.index
                )
            })

            # Fatigue features
            feature_dict[f'FATIGUE_DIFF{window_str}'] = (
                get_stat('FATIGUE', 'HOME') - get_stat('FATIGUE', 'AWAY')
            )

            # Calculate momentum features if not shortest window
            if window != min(self.lookback_windows):
                # Get base window stats
                home_win_base = get_stat('WIN_mean', 'HOME', window_size=min(self.lookback_windows))
                away_win_base = get_stat('WIN_mean', 'AWAY', window_size=min(self.lookback_windows))
                home_pts_base = get_stat('PTS_mean', 'HOME', window_size=min(self.lookback_windows))
                away_pts_base = get_stat('PTS_mean', 'AWAY', window_size=min(self.lookback_windows))

                feature_dict.update({
                    f'WIN_MOMENTUM{window_str}': (
                        safe_divide(home_win_pct, home_win_base, index=stats_df.index) -
                        safe_divide(away_win_pct, away_win_base, index=stats_df.index)
                    ),
                    f'SCORING_MOMENTUM{window_str}': (
                        safe_divide(home_pts, home_pts_base, index=stats_df.index) -
                        safe_divide(away_pts, away_pts_base, index=stats_df.index)
                    )
                })

        # Rest features (not window dependent)
        rest_days_home = stats_df['REST_DAYS_HOME'].fillna(0)
        rest_days_away = stats_df['REST_DAYS_AWAY'].fillna(0)
        feature_dict['REST_DIFF'] = rest_days_home - rest_days_away

        # Head-to-head features
        h2h_win_pct = stats_df['H2H_WIN_PCT'].fillna(0.5)
        h2h_games = stats_df['H2H_GAMES'].fillna(0)
        days_since_h2h = stats_df['DAYS_SINCE_H2H'].fillna(365)
        last_game_home = stats_df['LAST_GAME_HOME'].fillna(0)

        feature_dict.update({
            'H2H_WIN_PCT': h2h_win_pct,
            'H2H_GAMES': h2h_games,
            'DAYS_SINCE_H2H': days_since_h2h,
            'LAST_GAME_HOME_ADVANTAGE': last_game_home
        })

        # Composite features
        min_window = min(self.lookback_windows)
        max_window = max(self.lookback_windows)

        feature_dict.update({
            'RECENT_VS_LONG_TERM_HOME': safe_divide(
                get_stat('WIN_mean', 'HOME', window_size=min_window),
                get_stat('WIN_mean', 'HOME', window_size=max_window),
                index=stats_df.index
            ),
            'RECENT_VS_LONG_TERM_AWAY': safe_divide(
                get_stat('WIN_mean', 'AWAY', window_size=min_window),
                get_stat('WIN_mean', 'AWAY', window_size=max_window),
                index=stats_df.index
            ),
            'HOME_AWAY_CONSISTENCY_30D': np.abs(
                get_stat('WIN_mean', 'HOME', window_size=30) -
                get_stat('WIN_mean', 'AWAY', window_size=30)
            )
        })

        # H2H recency interaction
        feature_dict['H2H_RECENCY_WEIGHT'] = safe_divide(
            h2h_win_pct,
            np.log1p(days_since_h2h),
            fill_value=0.5,
            index=stats_df.index
        )

        # Create features DataFrame
        features = pd.DataFrame(feature_dict)

        # Clean up the data
        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.fillna(0)

        # Clip extreme values
        for col in features.columns:
            q1 = features[col].quantile(0.01)
            q99 = features[col].quantile(0.99)
            features[col] = features[col].clip(q1, q99)

        # Print feature summary
        print(f"\nCreated {len(features.columns)} features:")
        for prefix, description in FEATURE_GROUPS.items():
            related_features = [col for col in features.columns if prefix in col]
            print(f"{description}: {len(related_features)} features")

        return features, stats_df['TARGET']
        
    def prepare_enhanced_features(self, stats_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare enhanced feature set with all new metrics.
        
        Args:
            stats_df: DataFrame containing team statistics
            
        Returns:
            pd.DataFrame: Enhanced features
        """
        print("Preparing enhanced feature set...")

        try:
            # Initialize features DataFrame
            features = pd.DataFrame()

            # Get the game date column - check both possible names
            if 'GAME_DATE_HOME' in stats_df.columns:
                date_col = 'GAME_DATE_HOME'
            elif 'GAME_DATE' in stats_df.columns:
                date_col = 'GAME_DATE'
            else:
                raise KeyError("No game date column found in stats DataFrame")

            features['GAME_DATE'] = pd.to_datetime(stats_df[date_col])

            # Get target variable - check various possible names/scenarios
            if 'WL_HOME' in stats_df.columns:
                features['TARGET'] = (stats_df['WL_HOME'] == 'W').astype(int)
            elif 'TARGET' in stats_df.columns:
                features['TARGET'] = stats_df['TARGET']
            else:
                # For prediction scenarios where no target exists yet
                print("No target column found - this is likely a prediction scenario")
                features['TARGET'] = 0  # Default placeholder value for prediction cases

            # Process time window features
            for window in self.lookback_windows:
                window_str = f'_{window}D'

                # Win percentage features
                for team_type in ['HOME', 'AWAY']:
                    win_col = f'WIN_mean_{team_type}{window_str}'
                    if win_col in stats_df.columns:
                        features[f'WIN_PCT_{team_type}{window_str}'] = stats_df[win_col]
                    else:
                        print(f"Warning: {win_col} not found in stats DataFrame")
                        features[f'WIN_PCT_{team_type}{window_str}'] = 0

                # Calculate win percentage differential if both home and away exist
                if all(f'WIN_PCT_{t}{window_str}' in features.columns for t in ['HOME', 'AWAY']):
                    features[f'WIN_PCT_DIFF{window_str}'] = (
                        features[f'WIN_PCT_HOME{window_str}'] -
                        features[f'WIN_PCT_AWAY{window_str}']
                    )

                # Rating metrics
                for metric in ['OFF_RTG', 'DEF_RTG', 'NET_RTG', 'PACE']:
                    home_col = f'{metric}_mean_HOME{window_str}'
                    away_col = f'{metric}_mean_AWAY{window_str}'

                    if home_col in stats_df.columns and away_col in stats_df.columns:
                        home_stat = stats_df[home_col].fillna(0)
                        away_stat = stats_df[away_col].fillna(0)
                        features[f'{metric}_DIFF{window_str}'] = home_stat - away_stat
                    else:
                        print(f"Warning: {metric} columns not found for {window_str} window")

                # Points and turnover features
                pts_home = f'PTS_mean_HOME{window_str}'
                pts_away = f'PTS_mean_AWAY{window_str}'
                tov_home = f'TOV_mean_HOME{window_str}'
                tov_away = f'TOV_mean_AWAY{window_str}'

                if all(col in stats_df.columns for col in [pts_home, pts_away, tov_home, tov_away]):
                    features[f'EFF_DIFF{window_str}'] = (
                        (stats_df[pts_home].fillna(0) / stats_df[tov_home].fillna(1)) -
                        (stats_df[pts_away].fillna(0) / stats_df[tov_away].fillna(1))
                    )

                # Consistency metrics
                for team_type in ['HOME', 'AWAY']:
                    pts_mean = f'PTS_mean_{team_type}{window_str}'
                    pts_std = f'PTS_std_{team_type}{window_str}'

                    if pts_mean in stats_df.columns and pts_std in stats_df.columns:
                        features[f'{team_type}_CONSISTENCY{window_str}'] = (
                            stats_df[pts_std].fillna(0) / stats_df[pts_mean].fillna(1)
                        )

            # Add travel impact features
            print("Calculating travel impacts...")
            travel_impacts = stats_df.apply(calculate_travel_impact, axis=1)
            features['TRAVEL_DISTANCE'] = [x['distance'] for x in travel_impacts]
            features['TIMEZONE_DIFF'] = [x['timezone_diff'] for x in travel_impacts]

            # Add rest and fatigue features
            print("Adding rest and fatigue features...")
            if 'REST_DAYS_HOME' in stats_df.columns and 'REST_DAYS_AWAY' in stats_df.columns:
                features['REST_DIFF'] = (
                    stats_df['REST_DAYS_HOME'].fillna(0) -
                    stats_df['REST_DAYS_AWAY'].fillna(0)
                )

            for window in self.lookback_windows:
                fatigue_home = f'FATIGUE_HOME_{window}D'
                fatigue_away = f'FATIGUE_AWAY_{window}D'

                if fatigue_home in stats_df.columns and fatigue_away in stats_df.columns:
                    features[f'FATIGUE_DIFF_{window}D'] = (
                        stats_df[fatigue_home].fillna(0) -
                        stats_df[fatigue_away].fillna(0)
                    )

            # Add head-to-head features
            print("Adding head-to-head features...")
            h2h_cols = ['H2H_GAMES', 'H2H_WIN_PCT', 'H2H_AVG_MARGIN',
                      'H2H_STREAK', 'H2H_HOME_ADVANTAGE', 'H2H_MOMENTUM']
            for col in h2h_cols:
                if col in stats_df.columns:
                    features[col] = stats_df[col].fillna(0)

            # Create interaction features
            print("Creating interaction features...")
            if all(col in features.columns for col in ['FATIGUE_DIFF_14D', 'TRAVEL_DISTANCE']):
                features['FATIGUE_TRAVEL_INTERACTION'] = (
                    features['FATIGUE_DIFF_14D'] * features['TRAVEL_DISTANCE'] / 1000
                )

            if all(col in features.columns for col in ['WIN_PCT_DIFF_30D', 'REST_DIFF']):
                features['MOMENTUM_REST_INTERACTION'] = (
                    features['WIN_PCT_DIFF_30D'] * features['REST_DIFF']
                )

            # Clean up the data
            print("Cleaning and validating data...")
            features = features.replace([np.inf, -np.inf], np.nan)
            features = features.fillna(0)

            # Clip extreme values for numerical columns
            for col in features.columns:
                if col not in ['GAME_DATE', 'TARGET']:
                    q1 = features[col].quantile(0.01)
                    q99 = features[col].quantile(0.99)
                    features[col] = features[col].clip(q1, q99)

            print(f"Successfully created {len(features.columns)} features")
            return features

        except Exception as e:
            print(f"Error preparing enhanced features: {e}")
            print("DataFrame columns:", stats_df.columns.tolist())
            raise