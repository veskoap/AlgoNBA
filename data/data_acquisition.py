import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, teamestimatedmetrics
import time

class DataAcquirer:
    def __init__(self, seasons):
        self.seasons = seasons
        
    def fetch_games(self):
        """Fetch NBA games data with detailed statistics and advanced metrics."""
        print("Fetching basic game data...")
        all_games = []
        advanced_metrics = {}

        for season in self.seasons:
            print(f"Fetching {season} data...")
            games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
            all_games.append(games)

            metrics = self.fetch_advanced_metrics(season)
            if not metrics.empty:
                advanced_metrics[season] = metrics

            time.sleep(1)

        df = pd.concat(all_games, ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
        return df, advanced_metrics

    def fetch_advanced_metrics(self, season: str) -> pd.DataFrame:
        """Fetch advanced team metrics for a given season."""
        try:
            print(f"Fetching advanced metrics for {season}...")
            metrics = teamestimatedmetrics.TeamEstimatedMetrics(
                season=season,
                season_type='Regular Season'
            ).get_data_frames()[0]
            time.sleep(1)
            return metrics
        except Exception as e:
            print(f"Warning: Could not fetch advanced metrics - {str(e)}")
            return pd.DataFrame()
