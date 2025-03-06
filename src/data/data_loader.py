"""
Data loading functions for the NBA prediction model.
"""
import pandas as pd
import time
from typing import Dict, List, Tuple
from nba_api.stats.endpoints import leaguegamefinder

from src.utils.constants import BASIC_STATS_COLUMNS


class NBADataLoader:
    """Class for loading and preprocessing NBA game data."""
    
    def __init__(self):
        """Initialize the NBA data loader."""
        pass
    
    def fetch_games(self, seasons: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch NBA games data with detailed statistics and advanced metrics.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY'
            
        Returns:
            tuple: (games_df, advanced_metrics_dict)
        """
        print("Fetching basic game data...")
        all_games = []
        advanced_metrics = {}

        for season in seasons:
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

            time.sleep(1)  # Respect API rate limits

        df = pd.concat(all_games, ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Split into home/away
        home = df[df['MATCHUP'].str.contains('vs')].copy()
        away = df[df['MATCHUP'].str.contains('@')].copy()

        # Merge games data - ensuring we only merge exact game matches
        games = pd.merge(
            home[BASIC_STATS_COLUMNS].add_suffix('_HOME'),
            away[BASIC_STATS_COLUMNS].add_suffix('_AWAY'),
            left_on=['GAME_ID_HOME'],
            right_on=['GAME_ID_AWAY'],
            how='inner'  # Only keep exact matches
        )

        # Validate that game dates match between home and away records
        date_mismatch = (games['GAME_DATE_HOME'] != games['GAME_DATE_AWAY']).sum()
        if date_mismatch > 0:
            print(f"Warning: {date_mismatch} games have mismatched dates between home and away records")
            
        # Sort chronologically to prevent any data leakage from future games
        games = games.sort_values('GAME_DATE_HOME')
        print(f"Retrieved {len(games)} games with enhanced statistics")

        return games, advanced_metrics

    def fetch_advanced_metrics(self, season: str) -> pd.DataFrame:
        """
        Fetch advanced team metrics for a given season.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
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
            
    def fetch_player_availability(self, season: str) -> pd.DataFrame:
        """
        Fetch player availability data for a season.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Player availability data
        """
        try:
            # This would typically connect to the NBA API's injury reports
            # For demonstration, a simplified structure is returned
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