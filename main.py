import pandas as pd
import numpy as np
from collections import defaultdict
from nba_api.stats.endpoints import leaguegamefinder, commonallplayers, playerprofilev2
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from typing import List, Dict, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
import pytz
from geopy.distance import geodesic
import time

class DeepNBAPredictor(nn.Module):
    def __init__(self, input_size):
        super(DeepNBAPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        return self.network(x)

class EnhancedNBAPredictor:
    def __init__(self, seasons: List[str]):
        self.seasons = seasons
        self.lookback_windows = [7, 14, 30, 60]
        self.team_ratings_cache = {}
        self.feature_importances = {}
        self.selected_features = {}
        self.feature_importance_summary = {}
        self.team_locations = self._initialize_team_locations()
        self.timezone_map = self._initialize_timezone_map()

    def _initialize_team_locations(self) -> Dict:
        """Initialize NBA team locations with coordinates."""
        return {
            'ATL': {'coords': (33.7573, -84.3963), 'timezone': 'America/New_York'},
            'BOS': {'coords': (42.3662, -71.0621), 'timezone': 'America/New_York'},
            'BKN': {'coords': (40.6828, -73.9758), 'timezone': 'America/New_York'},
            'CHA': {'coords': (35.2251, -80.8392), 'timezone': 'America/New_York'},
            'CHI': {'coords': (41.8807, -87.6742), 'timezone': 'America/Chicago'},
            'CLE': {'coords': (41.4965, -81.6882), 'timezone': 'America/New_York'},
            'DAL': {'coords': (32.7905, -96.8103), 'timezone': 'America/Chicago'},
            'DEN': {'coords': (39.7487, -105.0077), 'timezone': 'America/Denver'},
            'DET': {'coords': (42.3410, -83.0550), 'timezone': 'America/New_York'},
            'GSW': {'coords': (37.7679, -122.3874), 'timezone': 'America/Los_Angeles'},
            'HOU': {'coords': (29.7508, -95.3621), 'timezone': 'America/Chicago'},
            'IND': {'coords': (39.7640, -86.1555), 'timezone': 'America/Indiana/Indianapolis'},
            'LAC': {'coords': (34.0430, -118.2673), 'timezone': 'America/Los_Angeles'},
            'LAL': {'coords': (34.0430, -118.2673), 'timezone': 'America/Los_Angeles'},
            'MEM': {'coords': (35.1382, -90.0505), 'timezone': 'America/Chicago'},
            'MIA': {'coords': (25.7814, -80.1870), 'timezone': 'America/New_York'},
            'MIL': {'coords': (43.0436, -87.9172), 'timezone': 'America/Chicago'},
            'MIN': {'coords': (44.9795, -93.2762), 'timezone': 'America/Chicago'},
            'NOP': {'coords': (29.9511, -90.0821), 'timezone': 'America/Chicago'},
            'NYK': {'coords': (40.7505, -73.9934), 'timezone': 'America/New_York'},
            'OKC': {'coords': (35.4634, -97.5151), 'timezone': 'America/Chicago'},
            'ORL': {'coords': (28.5392, -81.3839), 'timezone': 'America/New_York'},
            'PHI': {'coords': (39.9012, -75.1720), 'timezone': 'America/New_York'},
            'PHX': {'coords': (33.4457, -112.0712), 'timezone': 'America/Phoenix'},
            'POR': {'coords': (45.5316, -122.6668), 'timezone': 'America/Los_Angeles'},
            'SAC': {'coords': (38.5806, -121.4996), 'timezone': 'America/Los_Angeles'},
            'SAS': {'coords': (29.4271, -98.4375), 'timezone': 'America/Chicago'},
            'TOR': {'coords': (43.6435, -79.3791), 'timezone': 'America/Toronto'},
            'UTA': {'coords': (40.7683, -111.9011), 'timezone': 'America/Denver'},
            'WAS': {'coords': (38.8981, -77.0209), 'timezone': 'America/New_York'}
        }

    def _initialize_timezone_map(self) -> Dict:
        """Initialize timezone mapping for calculations."""
        return {tz: pytz.timezone(tz) for team in self.team_locations.values()
                for tz in [team['timezone']]}

    def fetch_games(self) -> Tuple[pd.DataFrame, Dict]:
        """Fetch NBA games data with detailed statistics and advanced metrics."""
        print("Fetching basic game data...")
        all_games = []
        advanced_metrics = {}

        for season in self.seasons:
            print(f"Fetching {season} data...")
            # Basic game data
            games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
            all_games.append(games)

            # Advanced metrics
            metrics = self.fetch_advanced_metrics(season)
            if not metrics.empty:
                advanced_metrics[season] = metrics

            time.sleep(1)

        df = pd.concat(all_games, ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Split into home/away
        home = df[df['MATCHUP'].str.contains('vs')].copy()
        away = df[df['MATCHUP'].str.contains('@')].copy()

        # Enhanced columns list including advanced stats
        basic_cols = [
            'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_NAME', 'WL',
            'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV',
            'PLUS_MINUS', 'FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM',
            'OREB', 'DREB', 'STL', 'BLK', 'PF'
        ]

        # Merge games data
        games = pd.merge(
            home[basic_cols].add_suffix('_HOME'),
            away[basic_cols].add_suffix('_AWAY'),
            left_on=['GAME_ID_HOME'],
            right_on=['GAME_ID_AWAY']
        )

        games = games.sort_values('GAME_DATE_HOME')
        print(f"Retrieved {len(games)} games with enhanced statistics")

        return games, advanced_metrics

    def fetch_advanced_metrics(self, season: str) -> pd.DataFrame:
        """Fetch advanced team metrics for a given season.

        Args:
            season (str): NBA season in format 'YYYY-YY'

        Returns:
            pd.DataFrame: Advanced team metrics or empty DataFrame if fetch fails
        """
        try:
            from nba_api.stats.endpoints import teamestimatedmetrics

            print(f"Fetching advanced metrics for {season}...")
            metrics = teamestimatedmetrics.TeamEstimatedMetrics(
                season=season,
                season_type='Regular Season'
            ).get_data_frames()[0]

            print(f"Successfully fetched advanced metrics for {season}")
            time.sleep(1)  # Respect API rate limits
            return metrics

        except ImportError:
            print(f"Warning: teamestimatedmetrics endpoint not available - skipping advanced metrics for {season}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Warning: Could not fetch advanced metrics for {season} - {str(e)}")
            return pd.DataFrame()

    def calculate_team_stats(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced team statistics using vectorized operations."""
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
        advanced_stats = games.apply(self.calculate_advanced_stats, axis=1)
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

    def calculate_advanced_stats(self, row: pd.Series) -> Dict:
        """Calculate advanced basketball statistics for a single game."""
        stats = {}

        # Offensive Rating (Points per 100 possessions)
        possessions = (
            0.5 * ((row['FGA_HOME'] + 0.4 * row['FTA_HOME'] - 1.07 *
                    (row['OREB_HOME'] / (row['OREB_HOME'] + row['DREB_AWAY'])) *
                    (row['FGA_HOME'] - row['FGM_HOME']) + row['TOV_HOME']) +
                   (row['FGA_AWAY'] + 0.4 * row['FTA_AWAY'] - 1.07 *
                    (row['OREB_AWAY'] / (row['OREB_AWAY'] + row['DREB_HOME'])) *
                    (row['FGA_AWAY'] - row['FGM_AWAY']) + row['TOV_AWAY'])))

        stats['OFF_RTG_HOME'] = 100 * row['PTS_HOME'] / possessions if possessions > 0 else 0
        stats['OFF_RTG_AWAY'] = 100 * row['PTS_AWAY'] / possessions if possessions > 0 else 0

        # Effective Field Goal Percentage
        stats['EFG_PCT_HOME'] = (row['FGM_HOME'] + 0.5 * row['FG3M_HOME']) / row['FGA_HOME'] if row['FGA_HOME'] > 0 else 0
        stats['EFG_PCT_AWAY'] = (row['FGM_AWAY'] + 0.5 * row['FG3M_AWAY']) / row['FGA_AWAY'] if row['FGA_AWAY'] > 0 else 0

        # True Shooting Percentage
        ts_attempts_home = row['FGA_HOME'] + 0.44 * row['FTA_HOME']
        ts_attempts_away = row['FGA_AWAY'] + 0.44 * row['FTA_AWAY']
        stats['TS_PCT_HOME'] = (row['PTS_HOME'] / (2 * ts_attempts_home)) if ts_attempts_home > 0 else 0
        stats['TS_PCT_AWAY'] = (row['PTS_AWAY'] / (2 * ts_attempts_away)) if ts_attempts_away > 0 else 0

        # Defensive Rating (Points allowed per 100 possessions)
        stats['DEF_RTG_HOME'] = stats['OFF_RTG_AWAY']
        stats['DEF_RTG_AWAY'] = stats['OFF_RTG_HOME']

        # Net Rating
        stats['NET_RTG_HOME'] = stats['OFF_RTG_HOME'] - stats['DEF_RTG_HOME']
        stats['NET_RTG_AWAY'] = stats['OFF_RTG_AWAY'] - stats['DEF_RTG_AWAY']

        return stats

    def calculate_h2h_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head statistics between teams using highly optimized operations."""
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
                h2h_win_pct = len(home_perspective_matches[home_perspective_matches['win']]) / h2h_games
                days_since = (current_date - last_match['date']).days
                last_game_home = int(last_match['team1'] == home_team)

                # Update results
                results.iloc[idx, results.columns.get_loc('H2H_GAMES')] = h2h_games
                results.iloc[idx, results.columns.get_loc('H2H_WIN_PCT')] = h2h_win_pct
                results.iloc[idx, results.columns.get_loc('DAYS_SINCE_H2H')] = days_since
                results.iloc[idx, results.columns.get_loc('LAST_GAME_HOME')] = last_game_home

        print(f"Calculated head-to-head features for {len(results)} matchups")
        return results

    def prepare_features(self, stats_df: pd.DataFrame) -> tuple:
        """Prepare enhanced features with advanced metrics and interaction terms."""
        print("Preparing enhanced feature set...")

        # Helper function to safely divide
        def safe_divide(a, b, fill_value=0):
            # Convert inputs to pandas Series if they aren't already
            if not isinstance(a, pd.Series):
                a = pd.Series(a, index=stats_df.index)
            if not isinstance(b, pd.Series):
                b = pd.Series(b, index=stats_df.index)

            # Fill NA values
            a = a.fillna(0)
            b = b.fillna(1)  # Use 1 for denominator to avoid division by zero

            # Convert to numpy arrays
            a_array = np.asarray(a.astype(float))
            b_array = np.asarray(b.astype(float))

            # Perform division
            with np.errstate(divide='ignore', invalid='ignore'):
                result = np.divide(a_array, b_array)
                mask = ~np.isfinite(result)
                result[mask] = fill_value

            return pd.Series(result, index=stats_df.index)

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
                f'EFF_DIFF{window_str}': safe_divide(home_pts, home_tov) - safe_divide(away_pts, away_tov),
                f'HOME_CONSISTENCY{window_str}': safe_divide(
                    get_stat('PTS_std', 'HOME'),
                    home_pts,
                    fill_value=1
                ),
                f'AWAY_CONSISTENCY{window_str}': safe_divide(
                    get_stat('PTS_std', 'AWAY'),
                    away_pts,
                    fill_value=1
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
                        safe_divide(home_win_pct, home_win_base) -
                        safe_divide(away_win_pct, away_win_base)
                    ),
                    f'SCORING_MOMENTUM{window_str}': (
                        safe_divide(home_pts, home_pts_base) -
                        safe_divide(away_pts, away_pts_base)
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
                get_stat('WIN_mean', 'HOME', window_size=max_window)
            ),
            'RECENT_VS_LONG_TERM_AWAY': safe_divide(
                get_stat('WIN_mean', 'AWAY', window_size=min_window),
                get_stat('WIN_mean', 'AWAY', window_size=max_window)
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
            fill_value=0.5
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
        feature_groups = {
            'WIN_PCT': 'Win percentage and momentum',
            'RTG': 'Rating metrics (OFF/DEF/NET)',
            'EFF': 'Efficiency metrics',
            'MOMENTUM': 'Momentum and consistency',
            'REST': 'Rest and fatigue',
            'H2H': 'Head-to-head matchups'
        }

        for prefix, description in feature_groups.items():
            related_features = [col for col in features.columns if prefix in col]
            print(f"{description}: {len(related_features)} features")

        return features, stats_df['TARGET']

    def calculate_travel_impact(self, row: pd.Series) -> Dict:
            """Calculate travel distance and timezone changes between games."""
            try:
                # Extract team abbreviations - handle both full names and abbreviations
                def get_team_abbrev(team_id):
                    # Map of common team IDs to abbreviations
                    team_map = {
                        1610612737: 'ATL', 1610612738: 'BOS', 1610612751: 'BKN',
                        1610612766: 'CHA', 1610612741: 'CHI', 1610612739: 'CLE',
                        1610612742: 'DAL', 1610612743: 'DEN', 1610612765: 'DET',
                        1610612744: 'GSW', 1610612745: 'HOU', 1610612754: 'IND',
                        1610612746: 'LAC', 1610612747: 'LAL', 1610612763: 'MEM',
                        1610612748: 'MIA', 1610612749: 'MIL', 1610612750: 'MIN',
                        1610612740: 'NOP', 1610612752: 'NYK', 1610612760: 'OKC',
                        1610612753: 'ORL', 1610612755: 'PHI', 1610612756: 'PHX',
                        1610612757: 'POR', 1610612758: 'SAC', 1610612759: 'SAS',
                        1610612761: 'TOR', 1610612762: 'UTA', 1610612764: 'WAS'
                    }
                    return team_map.get(team_id, 'UNK')

                home_team = get_team_abbrev(row['TEAM_ID_HOME'])
                away_team = get_team_abbrev(row['TEAM_ID_AWAY'])

                if home_team == 'UNK' or away_team == 'UNK':
                    return {'distance': 0, 'timezone_diff': 0}

                if home_team not in self.team_locations or away_team not in self.team_locations:
                    return {'distance': 0, 'timezone_diff': 0}

                # Calculate distance
                home_coords = self.team_locations[home_team]['coords']
                away_coords = self.team_locations[away_team]['coords']
                distance = geodesic(home_coords, away_coords).miles

                # Calculate timezone difference
                home_tz = self.timezone_map[self.team_locations[home_team]['timezone']]
                away_tz = self.timezone_map[self.team_locations[away_team]['timezone']]

                if 'GAME_DATE' in row:
                    game_date = pd.to_datetime(row['GAME_DATE'])
                elif 'GAME_DATE_HOME' in row:
                    game_date = pd.to_datetime(row['GAME_DATE_HOME'])
                else:
                    game_date = pd.Timestamp.now()

                # Account for daylight savings
                home_dt = home_tz.localize(game_date)
                away_dt = away_tz.localize(game_date)
                timezone_diff = (home_dt.utcoffset() - away_dt.utcoffset()).total_seconds() / 3600

                return {'distance': distance, 'timezone_diff': timezone_diff}

            except Exception as e:
                print(f"Error calculating travel impact: {e}")
                return {'distance': 0, 'timezone_diff': 0}

    def fetch_player_availability(self, season: str) -> pd.DataFrame:
        """Fetch player availability data for a season."""
        try:
            # Typically connects to the NBA API's injury reports
            # For demonstration, I created a simplified structure
            return pd.DataFrame({
                'GAME_ID': [],
                'TEAM_ID': [],
                'PLAYERS_AVAILABLE': [],
                'STARTERS_AVAILABLE': [],
                'ROTATION_STRENGTH': []  # Calculated based on available players' stats
            })
        except Exception as e:
            print(f"Error fetching player availability: {e}")
            return pd.DataFrame()

    def calculate_enhanced_h2h_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate enhanced head-to-head features with detailed matchup analysis."""
        print("Calculating enhanced head-to-head features...")

        # Create a copy of games for manipulation
        h2h_features = games.copy()

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

    def train_deep_model(self, X: pd.DataFrame) -> Tuple[List, List]:
            """Train deep neural network model with enhanced architecture."""
            print("\nTraining deep neural network model...")

            # Extract target variable
            y = X['TARGET']
            X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')

            print(f"Training deep model with {len(X)} samples and {len(X.columns)} features")

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")

            models = []
            scalers = []
            fold_metrics = []
            tscv = TimeSeriesSplit(n_splits=5)

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                print(f"\nTraining deep model fold {fold}...")

                # Prepare data
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

                # Scale features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                # Convert to PyTorch tensors
                X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
                y_train_tensor = torch.LongTensor(y_train.values).to(device)
                X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
                y_val_tensor = torch.LongTensor(y_val.values).to(device)

                # Initialize model
                model = DeepNBAPredictor(X_train.shape[1]).to(device)
                criterion = nn.CrossEntropyLoss()
                optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

                # Training loop
                best_val_loss = float('inf')
                patience = 10
                patience_counter = 0
                best_metrics = None

                for epoch in range(100):
                    # Training
                    model.train()
                    optimizer.zero_grad()
                    outputs = model(X_train_tensor)
                    loss = criterion(outputs, y_train_tensor)
                    loss.backward()
                    optimizer.step()

                    # Validation
                    model.eval()
                    with torch.no_grad():
                        val_outputs = model(X_val_tensor)
                        val_loss = criterion(val_outputs, y_val_tensor)
                        val_preds = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                        val_pred_binary = (val_preds > 0.5).astype(int)

                        # Calculate metrics
                        acc = accuracy_score(y_val, val_pred_binary)
                        brier = brier_score_loss(y_val, val_preds)
                        auc = roc_auc_score(y_val, val_preds)

                        # Store best metrics
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_model_state = model.state_dict().copy()
                            best_metrics = {
                                'accuracy': acc,
                                'brier_score': brier,
                                'auc': auc
                            }
                        else:
                            patience_counter += 1

                    # Update learning rate
                    scheduler.step(val_loss)

                    # Early stopping check
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch}")
                        break

                # Load best model
                model.load_state_dict(best_model_state)
                models.append(model)
                scalers.append(scaler)
                fold_metrics.append(best_metrics)

                print(f"Fold {fold} Best Metrics:")
                print(f"Accuracy: {best_metrics['accuracy']:.3f}")
                print(f"Brier Score: {best_metrics['brier_score']:.3f}")
                print(f"AUC-ROC: {best_metrics['auc']:.3f}")

            # Print overall performance
            print("\nOverall Deep Model Performance:")
            metrics_df = pd.DataFrame(fold_metrics)
            for metric in metrics_df.columns:
                mean_val = metrics_df[metric].mean()
                std_val = metrics_df[metric].std()
                print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

            return models, scalers

    def calculate_position_metrics(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate position-specific performance metrics."""
        print("Calculating position-specific metrics...")

        position_metrics = pd.DataFrame()

        try:
            # Group players by primary position
            positions = {
                'Guards': ['PG', 'SG'],
                'Wings': ['SF', 'SG'],
                'Bigs': ['PF', 'C']
            }

            # Initialize metrics for each position group
            for team_type in ['HOME', 'AWAY']:
                for pos_group in positions:
                    position_metrics[f'{pos_group}_EFF_{team_type}'] = 0
                    position_metrics[f'{pos_group}_USAGE_{team_type}'] = 0
                    position_metrics[f'{pos_group}_DEF_RTG_{team_type}'] = 0

            # Calculate metrics based on available boxscore data
            for _, game in games.iterrows():
                for team_type in ['HOME', 'AWAY']:
                    team_metrics = self._calculate_team_position_metrics(game, team_type)
                    for metric, value in team_metrics.items():
                        position_metrics.at[game.name, metric] = value

        except Exception as e:
            print(f"Error calculating position metrics: {e}")

        return position_metrics

    def calculate_seasonal_trends(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate seasonal trend adjustments."""
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

    def calculate_confidence_score(self, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        confidence_scores = np.zeros(len(predictions))

        try:
            # Factors affecting confidence
            factors = {
                'prediction_margin': 0.3,  # Weight for prediction probability margin
                'sample_size': 0.2,        # Weight for number of previous matches
                'recent_consistency': 0.2,  # Weight for consistency in recent games
                'h2h_history': 0.15,       # Weight for head-to-head history
                'rest_advantage': 0.15     # Weight for rest day advantage
            }

            for i, pred in enumerate(predictions):
                score = 0

                # Prediction margin confidence
                prob_margin = abs(pred - 0.5) * 2  # Scale to [0, 1]
                score += prob_margin * factors['prediction_margin']

                # Sample size confidence
                games_played = features.iloc[i]['WIN_count_HOME_60D']
                sample_size_conf = min(games_played / 20, 1)  # Scale to [0, 1]
                score += sample_size_conf * factors['sample_size']

                # Recent consistency confidence
                consistency = 1 - features.iloc[i]['HOME_CONSISTENCY_30D']  # Lower variance is better
                score += consistency * factors['recent_consistency']

                # Head-to-head confidence
                h2h_games = features.iloc[i]['H2H_GAMES']
                h2h_conf = min(h2h_games / 10, 1)  # Scale to [0, 1]
                score += h2h_conf * factors['h2h_history']

                # Rest advantage confidence
                rest_diff = abs(features.iloc[i]['REST_DIFF'])
                rest_conf = min(rest_diff / 3, 1)  # Scale to [0, 1]
                score += rest_conf * factors['rest_advantage']

                confidence_scores[i] = score

            # Normalize to [0, 1]
            confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())

        except Exception as e:
            print(f"Error calculating confidence scores: {e}")
            confidence_scores = np.full(len(predictions), 0.5)

        return confidence_scores

    def prepare_enhanced_features(self, stats_df: pd.DataFrame) -> pd.DataFrame:
            """Prepare enhanced feature set with all new metrics."""
            print("Preparing enhanced feature set...")

            try:
                # First, let's debug what columns we actually have
                print("Available columns:", stats_df.columns.tolist())

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

                # Get target variable - check both possible names
                if 'WL_HOME' in stats_df.columns:
                    features['TARGET'] = (stats_df['WL_HOME'] == 'W').astype(int)
                elif 'TARGET' in stats_df.columns:
                    features['TARGET'] = stats_df['TARGET']
                else:
                    raise KeyError("No target column found in stats DataFrame")

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
                travel_impacts = stats_df.apply(self.calculate_travel_impact, axis=1)
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

    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Make predictions with confidence scores."""
        predictions = super().predict(X)
        confidence_scores = self.calculate_confidence_score(predictions, X)

        return predictions, confidence_scores

    def train(self, X: pd.DataFrame) -> None:
        """Train enhanced ensemble of models with improved stability."""
        print("Training model ensemble...")

        # Extract target variable
        y = X['TARGET']
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')

        print(f"Training with {len(X)} samples and {len(X.columns)} features")

        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize tracking
        self.models = []
        self.scalers = []
        self.feature_selectors = []
        fold_metrics = []
        feature_importance_dict = defaultdict(list)

        # First pass: identify consistently important features
        print("Performing initial feature stability analysis...")
        feature_stability = defaultdict(int)
        feature_selector_list = []

        for fold, (train_idx, _) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            # Initial feature selection
            selector = SelectFromModel(
                xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42 + fold
                ),
                threshold='mean'
            )

            selector.fit(X_train, y_train)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_selector_list.append(selected_features)

            for feat in selected_features:
                feature_stability[feat] += 1

        # Identify stable features (selected in majority of folds)
        stable_features = [feat for feat, count in feature_stability.items() if count >= 3]
        print(f"\nIdentified {len(stable_features)} stable features")

        # Main training loop with enhanced monitoring
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nTraining fold {fold}...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Use stable features
            feature_mask = X.columns.isin(stable_features)
            X_train_selected = X_train_scaled[:, feature_mask]
            X_val_selected = X_val_scaled[:, feature_mask]

            # Train window-specific models
            window_models = []

            for window in self.lookback_windows:
                # Get window-specific features
                window_features = [feat for feat in stable_features if f'_{window}D' in feat]
                base_features = [feat for feat in stable_features if '_D' not in feat]
                combined_features = window_features + base_features

                if not combined_features:
                    continue

                feature_indices = [stable_features.index(feat) for feat in combined_features]

                # Window-specific model with early stopping
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.005,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=4,
                    gamma=0.5,
                    reg_alpha=0.3,
                    reg_lambda=1.5,
                    scale_pos_weight=1,
                    random_state=42 + window,
                    eval_metric=['logloss', 'auc']
                )

                X_train_window = X_train_selected[:, feature_indices]
                X_val_window = X_val_selected[:, feature_indices]

                # Train with early stopping
                model.fit(
                    X_train_window, y_train,
                    eval_set=[(X_val_window, y_val)],
                    verbose=0
                )

                window_models.append((f'{window}d', model, combined_features))

                # Store feature importance
                importances = model.feature_importances_
                for feat, imp in zip(combined_features, importances):
                    feature_importance_dict[feat].append(imp)

            # Store models and scalers
            self.models.append((window_models, scaler))

            # Evaluate performance
            y_preds = []
            for _, model, feats in window_models:
                feature_indices = [stable_features.index(f) for f in feats]
                y_pred = model.predict_proba(X_val_selected[:, feature_indices])[:, 1]
                y_preds.append(y_pred)

            # Average predictions from all window models
            y_pred_avg = np.mean(y_preds, axis=0)
            y_pred_binary = (y_pred_avg > 0.5).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_val, y_pred_binary)
            brier = brier_score_loss(y_val, y_pred_avg)
            auc = roc_auc_score(y_val, y_pred_avg)

            fold_metrics.append({
                'accuracy': acc,
                'brier_score': brier,
                'auc': auc
            })

            print(f"Fold {fold} Metrics:")
            print(f"Accuracy: {acc:.3f}")
            print(f"Brier Score: {brier:.3f}")
            print(f"AUC-ROC: {auc:.3f}")

        # Print overall performance
        print("\nOverall Model Performance:")
        metrics_df = pd.DataFrame(fold_metrics)
        for metric in metrics_df.columns:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

def main():
    seasons = ['2020-21', '2021-22', '2022-23', '2023-24']
    predictor = EnhancedNBAPredictor(seasons)

    try:
        print("Starting enhanced NBA prediction model training...")
        games, advanced_metrics = predictor.fetch_games()
        print(f"Retrieved {len(games)} games with advanced metrics")

        # Output sample fetched games data
        print("\nSample fetched games data:")
        print(games.head(2))

        stats_df = predictor.calculate_team_stats(games)

        # Output sample calculated team stats data
        print("\nSample calculated team stats data:")
        print(stats_df.head(2))

        enhanced_features = predictor.prepare_enhanced_features(stats_df)

        # Output sample enhanced features data
        print("\nSample enhanced features data:")
        print(enhanced_features.head(2))

        # Train traditional model
        predictor.train(enhanced_features)

        # Output sample data after training
        print("\nSample data after training the traditional model:")
        print(enhanced_features.head(2))

        # Train deep model
        deep_models, deep_scalers = predictor.train_deep_model(enhanced_features)

        # Output sample data after training deep model
        print("\nSample data after training the deep neural network model:")
        print(enhanced_features.head(2))

        print("Model training completed successfully")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()