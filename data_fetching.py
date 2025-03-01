import pandas as pd
from nba_api.stats.endpoints import leaguegamefinder, teamestimatedmetrics
import time
from typing import Tuple, Dict, List

def fetch_games_data(seasons: List[str]) -> Tuple[pd.DataFrame, Dict]:
    """Fetch NBA games data with detailed statistics and advanced metrics."""
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
        metrics = fetch_advanced_metrics(season)
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

def fetch_advanced_metrics(season: str) -> pd.DataFrame:
    """Fetch advanced team metrics for a given season.

    Args:
        season (str): NBA season in format 'YYYY-YY'

    Returns:
        pd.DataFrame: Advanced team metrics or empty DataFrame if fetch fails
    """
    try:
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

def fetch_player_availability(season: str) -> pd.DataFrame:
    """Fetch player availability data for a season."""
    try:
        # Placeholder - Replace with actual NBA API call if available
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