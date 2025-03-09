"""
Data loading functions for the NBA prediction model.
"""
import pandas as pd
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from nba_api.stats.endpoints import leaguegamefinder, commonteamroster, leaguestandings
import requests
import urllib.parse
import bs4

from src.utils.constants import BASIC_STATS_COLUMNS, TEAM_LOCATIONS, TEAM_ID_TO_ABBREV
from src.utils.cache_manager import CacheManager


class BettingOddsService:
    """Service for fetching historical betting odds for NBA games"""
    
    def __init__(self, use_cache=True, cache_manager=None):
        self.use_cache = use_cache
        self.cache_manager = cache_manager
        
    def get_historical_odds(self, game_date, team_id_home, team_id_away):
        """
        Fetch historical betting lines for a specific game
        
        Args:
            game_date: Date of the game in format YYYY-MM-DD
            team_id_home: NBA API team ID for home team
            team_id_away: NBA API team ID for away team
            
        Returns:
            dict: Betting line information including spread, over/under, moneyline
        """
        # First check cache if available
        if self.use_cache and self.cache_manager:
            cache_key = f"odds_{game_date}_{team_id_home}_{team_id_away}"
            cached_odds = self.cache_manager.get_cache('betting_odds', {'key': cache_key})
            if cached_odds is not None:
                return cached_odds
        
        # For historical games, we'll simulate realistic betting lines
        # based on team performance differences
        try:
            # This is a simulation - in production this would connect to a 
            # real odds provider or historical odds database
            from src.utils.constants import TEAM_ID_TO_ABBREV
            home_abbrev = TEAM_ID_TO_ABBREV.get(team_id_home, "")
            away_abbrev = TEAM_ID_TO_ABBREV.get(team_id_away, "")
            
            # Generate realistic spread based on home court advantage
            # On average, home teams are favored by about 3 points
            home_advantage = 3.0
            
            # Add team strength differential
            # This would ideally come from team ratings/ELO systems
            # For simulation, we'll generate a random but realistic spread
            team_diff = self._get_team_strength_differential(team_id_home, team_id_away, game_date)
            
            # Calculate spread (negative means home team is favored)
            spread = -(home_advantage + team_diff)
            
            # Over/under total points (average NBA total is around 220)
            # Adjust based on team pace/offensive ratings
            avg_total = 220.0
            pace_adjustment = self._get_pace_adjustment(team_id_home, team_id_away, game_date)
            over_under = avg_total + pace_adjustment
            
            # Calculate implied win probability from spread
            # Using a standard model where each point is worth about 4% win probability
            # with home team having baseline 60% win rate for a pick'em (spread = 0)
            baseline_home_win_prob = 0.6
            points_to_winprob = 0.04
            implied_win_prob = baseline_home_win_prob + (spread * -1 * points_to_winprob)
            implied_win_prob = max(0.05, min(0.95, implied_win_prob))  # Limit to reasonable range
            
            # Calculate moneyline odds from implied probability
            if implied_win_prob > 0.5:  # Home team favored
                home_moneyline = -100 * (implied_win_prob / (1 - implied_win_prob))
                away_moneyline = 100 * ((1 - implied_win_prob) / implied_win_prob)
            else:  # Away team favored
                home_moneyline = 100 * (implied_win_prob / (1 - implied_win_prob))
                away_moneyline = -100 * ((1 - implied_win_prob) / implied_win_prob)
            
            # Format odds data
            odds_data = {
                'game_date': game_date,
                'home_team_id': team_id_home,
                'home_team': home_abbrev,
                'away_team_id': team_id_away,
                'away_team': away_abbrev,
                'spread': round(spread, 1),  # e.g., -5.5 means home favored by 5.5
                'spread_home_odds': -110,  # Standard vigorish
                'spread_away_odds': -110,
                'over_under': round(over_under, 1),
                'over_odds': -110,
                'under_odds': -110,
                'moneyline_home': int(round(home_moneyline)),
                'moneyline_away': int(round(away_moneyline)),
                'implied_home_win_prob': round(implied_win_prob, 4),
                'source': 'simulation'  # Mark as simulated data
            }
            
            # Cache the result
            if self.use_cache and self.cache_manager:
                self.cache_manager.set_cache('betting_odds', {'key': cache_key}, odds_data)
                
            return odds_data
            
        except Exception as e:
            print(f"Error generating betting odds: {e}")
            # Return default odds
            return {
                'game_date': game_date,
                'home_team_id': team_id_home,
                'away_team_id': team_id_away,
                'spread': -3.0,  # Default: home team favored by 3
                'over_under': 220.0,
                'moneyline_home': -150,
                'moneyline_away': +130,
                'implied_home_win_prob': 0.6,
                'source': 'default'
            }
    
    def _get_team_strength_differential(self, team_id_home, team_id_away, game_date):
        """
        Get the strength differential between teams
        In production, this would use actual team ratings/ELO
        
        Returns: point differential (positive means home team is stronger)
        """
        # In the absence of real data, we'll create a model that
        # produces realistic point spreads
        import numpy as np
        import hashlib
        
        # Create a deterministic but varied output based on the teams and date
        # This ensures the same game always gets the same differential
        seed_str = f"{team_id_home}_{team_id_away}_{game_date}"
        hash_obj = hashlib.md5(seed_str.encode())
        seed = int(hash_obj.hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Generate a realistic team strength differential
        # NBA point spreads typically range from about -15 to +15
        # with a standard deviation of about 7 points
        team_diff = np.random.normal(0, 7)
        
        # Ensure it's in a realistic range
        team_diff = max(-12, min(12, team_diff))
        
        return team_diff
    
    def _get_pace_adjustment(self, team_id_home, team_id_away, game_date):
        """
        Get pace adjustment for over/under
        In production this would use actual pace statistics
        
        Returns: adjustment to the baseline O/U (positive means higher scoring)
        """
        import numpy as np
        import hashlib
        
        # Similar approach to the strength differential
        seed_str = f"pace_{team_id_home}_{team_id_away}_{game_date}"
        hash_obj = hashlib.md5(seed_str.encode())
        seed = int(hash_obj.hexdigest(), 16) % 1000
        np.random.seed(seed)
        
        # Generate realistic pace adjustment
        # O/U lines typically vary by about ±20 points from average
        pace_adj = np.random.normal(0, 10)
        
        # Ensure it's in a realistic range
        pace_adj = max(-20, min(20, pace_adj))
        
        return pace_adj


class NBADataLoader:
    """Class for loading and preprocessing NBA game data with caching support."""
    
    def __init__(self, use_cache: bool = True, cache_max_age_days: int = 30, cache_dir: str = None,
                 include_betting_odds: bool = True):
        """
        Initialize the NBA data loader.
        
        Args:
            use_cache: Whether to use cache for data loading
            cache_max_age_days: Maximum age of cached data in days
            cache_dir: Custom directory for cache storage
            include_betting_odds: Whether to include betting odds data
        """
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        self.include_betting_odds = include_betting_odds
        
        # Check if running in Google Colab and handle Drive mounting
        self.is_colab = False
        try:
            import google.colab
            self.is_colab = True
            
            # If we're in Colab and no custom cache dir is provided, use a default one
            if cache_dir is None:
                # Check if Google Drive is available
                import os
                drive_path = '/content/drive'
                if not os.path.exists(drive_path):
                    try:
                        print("Google Drive not mounted. Mounting now...")
                        from google.colab import drive
                        drive.mount(drive_path)
                        print("Google Drive mounted successfully")
                    except Exception as e:
                        print(f"Error mounting Google Drive: {e}")
                        print("Will use local cache directory instead")
                        drive_path = None
                
                # Set cache directory on Google Drive if available
                if drive_path and os.path.exists(drive_path):
                    # Use a cache directory in MyDrive
                    colab_cache_dir = os.path.join(drive_path, 'MyDrive', 'AlgoNBA', 'cache')
                    # Create the directory if it doesn't exist
                    os.makedirs(colab_cache_dir, exist_ok=True)
                    cache_dir = colab_cache_dir
                    print(f"Using Google Drive cache directory: {cache_dir}")
        except ImportError:
            # Not running in Colab
            pass
        
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Initialize betting odds service
        self.betting_odds_service = BettingOddsService(use_cache=use_cache, cache_manager=self.cache_manager)
    
    def _fetch_bulk_seasons(self, seasons: List[str]) -> Optional[pd.DataFrame]:
        """
        Attempt to fetch multiple seasons in a single bulk API call.
        This is much faster than fetching each season individually.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY'
            
        Returns:
            DataFrame with game data or None if bulk fetch fails
        """
        try:
            print(f"Attempting bulk fetch for {len(seasons)} seasons...")
            # For bulk fetch with multiple seasons, we'll need to fetch more broadly
            # and then filter to our target seasons
            
            # Use a more optimized approach: get all data for recent years, then filter
            games = leaguegamefinder.LeagueGameFinder(
                season_type_nullable='Regular Season',
                league_id_nullable='00'  # NBA league ID
            ).get_data_frames()[0]
            
            # Parse season from date
            # Function to extract season from game date
            def get_season(date_str):
                date = pd.to_datetime(date_str)
                year = date.year
                month = date.month
                # NBA season spans two years, with Oct-Dec being part of the later year's season
                if month >= 10:  # Oct-Dec
                    return f"{year}-{str(year+1)[-2:]}"
                else:  # Jan-June
                    return f"{year-1}-{str(year)[-2:]}"
            
            # Apply the function to get seasons
            games['SEASON'] = games['GAME_DATE'].apply(get_season)
            
            # Filter to only include the requested seasons
            filtered_games = games[games['SEASON'].isin(seasons)].copy()
            
            # Clean up
            if 'SEASON' in filtered_games.columns:
                filtered_games.drop('SEASON', axis=1, inplace=True)
            
            if len(filtered_games) > 0:
                print(f"Bulk fetch successful: retrieved {len(filtered_games)} games across requested seasons")
                return filtered_games
            else:
                print("Bulk fetch returned no games for the requested seasons")
                return None
            
        except Exception as e:
            print(f"Bulk fetch failed: {str(e)}")
            return None
    
    def fetch_games(self, seasons: List[str]) -> Tuple[pd.DataFrame, Dict]:
        """
        Fetch NBA games data with detailed statistics and advanced metrics.
        Uses cache when available to avoid redundant API calls.
        With performance optimizations for A100 GPU environments.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY'
            
        Returns:
            tuple: (games_df, advanced_metrics_dict)
        """
        # Create cache key parameters with updated version for new optimized approach
        cache_params = {
            'seasons': sorted(seasons),
            'api_version': '1.1'  # Increased to reflect optimized fetching strategy
        }
        
        # Check for cached data
        if self.use_cache:
            print("Checking cache for game data...")
            cached_data = self.cache_manager.get_cache('games', cache_params)
            
            # Check if cache is available and not stale
            if cached_data is not None and not self.cache_manager.is_cache_stale('games', cache_params, self.cache_max_age_days):
                games_df, advanced_metrics = cached_data
                print(f"Using cached data for {len(seasons)} seasons: {len(games_df)} games loaded")
                return games_df, advanced_metrics
                
            print("No valid cache found. Fetching fresh data...")
        
        # First try bulk fetching all seasons at once
        try:
            bulk_games = self._fetch_bulk_seasons(seasons)
            if bulk_games is not None and len(bulk_games) > 0:
                # Process bulk data
                bulk_games['GAME_DATE'] = pd.to_datetime(bulk_games['GAME_DATE'])
                
                # Fetch advanced metrics for each season
                print("Fetching advanced metrics for each season...")
                advanced_metrics = {}
                for season in seasons:
                    try:
                        print(f"Fetching advanced metrics for season {season}...")
                        metrics = self.fetch_advanced_metrics(season)
                        if not metrics.empty:
                            advanced_metrics[season] = metrics
                        # Very short sleep between calls
                        time.sleep(0.25)
                    except Exception as e:
                        print(f"Error fetching advanced metrics for season {season}: {e}")
                
                # Split into home/away
                home = bulk_games[bulk_games['MATCHUP'].str.contains('vs')].copy()
                away = bulk_games[bulk_games['MATCHUP'].str.contains('@')].copy()
                
                # Process home/away data (replace missing method with inline code)
                print("Processing home and away game data...")
                # Create home and away columns
                home = home.add_suffix('_HOME')
                away = away.add_suffix('_AWAY')
                
                # Rename GAME_DATE columns to avoid duplication
                if 'GAME_DATE_HOME' in home.columns and 'GAME_DATE_AWAY' in away.columns:
                    # Both DataFrames have date columns with suffixes
                    pass
                else:
                    # Ensure column naming is consistent
                    if 'GAME_DATE' in home.columns:
                        home = home.rename(columns={'GAME_DATE': 'GAME_DATE_HOME'})
                    if 'GAME_DATE' in away.columns:
                        away = away.rename(columns={'GAME_DATE': 'GAME_DATE_AWAY'})
                
                # Merge home and away data on game ID and date
                # Create a mapping key for joining
                home['GAME_KEY'] = home['GAME_ID_HOME'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else x)
                away['GAME_KEY'] = away['GAME_ID_AWAY'].apply(lambda x: x.split('/')[0] if isinstance(x, str) else x)
                
                # Join on the common key
                processed_games = pd.merge(home, away, left_on='GAME_KEY', right_on='GAME_KEY')
                
                # Clean up - delete the temporary join key
                processed_games = processed_games.drop('GAME_KEY', axis=1, errors='ignore')
                
                # Add standardized GAME_DATE column using the home date
                if 'GAME_DATE_HOME' in processed_games.columns:
                    processed_games['GAME_DATE'] = processed_games['GAME_DATE_HOME']
                elif 'GAME_DATE_AWAY' in processed_games.columns:
                    processed_games['GAME_DATE'] = processed_games['GAME_DATE_AWAY']
                
                # Add betting odds data if enabled
                if self.include_betting_odds:
                    processed_games = self._add_betting_odds(processed_games)
                
                # Cache the result
                if self.use_cache:
                    cache_data = (processed_games, advanced_metrics)
                    self.cache_manager.set_cache('games', cache_params, cache_data)
                    print(f"Cached {len(processed_games)} games for {len(seasons)} seasons")
                
                return processed_games, advanced_metrics
                
        except Exception as e:
            print(f"Error in bulk processing: {e}")
            print("Falling back to individual season fetching...")
        
        print("Fetching basic game data season by season...")
        all_games = []
        advanced_metrics = {}
        
        # Track new data for determining if cache update is needed
        has_new_data = False

        for season in seasons:
            # Check if this specific season is in cache
            season_cache_params = {'season': season, 'api_version': '1.0'}
            
            if self.use_cache:
                season_cache = self.cache_manager.get_cache('games_season', season_cache_params)
                if season_cache is not None and not self.cache_manager.is_cache_stale('games_season', season_cache_params, self.cache_max_age_days):
                    print(f"Using cached data for season {season}")
                    season_games, season_metrics = season_cache
                    all_games.append(season_games)
                    if season_metrics:
                        advanced_metrics[season] = season_metrics
                    continue
            
            # If we get here, we need to fetch fresh data for this season
            has_new_data = True
            print(f"Fetching fresh data for {season}...")
            
            try:
                # Basic game data
                games = leaguegamefinder.LeagueGameFinder(
                    season_nullable=season,
                    season_type_nullable='Regular Season'
                ).get_data_frames()[0]
                
                # Advanced metrics
                metrics = self.fetch_advanced_metrics(season)
                season_metrics = None
                if not metrics.empty:
                    advanced_metrics[season] = metrics
                    season_metrics = metrics
                
                # Cache this season's data
                if self.use_cache:
                    season_data = (games, season_metrics)
                    self.cache_manager.set_cache('games_season', season_cache_params, season_data)
                    print(f"Cached {len(games)} games for season {season}")
                
                all_games.append(games)
                # Reduced sleep time for faster API calls - 0.25 seconds is usually enough
                time.sleep(0.25)  # Minimal rate limit respect
            
            except Exception as e:
                print(f"Error fetching data for season {season}: {e}")
                # Try to use older cached data if available, even if it's stale
                if self.use_cache:
                    season_cache = self.cache_manager.get_cache('games_season', season_cache_params)
                    if season_cache is not None:
                        print(f"Using stale cached data for season {season} due to fetch error")
                        season_games, season_metrics = season_cache
                        all_games.append(season_games)
                        if season_metrics:
                            advanced_metrics[season] = season_metrics
        
        # If no games were fetched or found in cache, return empty data with required columns
        if not all_games:
            print("No game data found for specified seasons")
            # Create a minimal DataFrame with all required columns for empty data scenarios
            required_columns = [col + suffix for col in BASIC_STATS_COLUMNS for suffix in ['_HOME', '_AWAY']]
            # Add other required columns that are needed downstream
            additional_columns = ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY', 'WL_HOME', 'GAME_ID_HOME']
            all_required_columns = list(set(required_columns + additional_columns))
            
            # Create sample data with default values
            sample_data = {col: [] for col in all_required_columns}
            empty_games = pd.DataFrame(sample_data)
            
            print(f"Created empty DataFrame with {len(all_required_columns)} columns for downstream compatibility")
            return empty_games, {}
            
    def _add_betting_odds(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add betting odds data to the games DataFrame.
        
        Args:
            games_df: DataFrame containing game data
            
        Returns:
            DataFrame with added betting odds columns
        """
        print("Adding betting odds data...")
        games_with_odds = games_df.copy()
        
        # Initialize betting odds columns
        betting_columns = [
            'SPREAD_HOME', 'SPREAD_AWAY', 'OVER_UNDER', 
            'MONEYLINE_HOME', 'MONEYLINE_AWAY', 'IMPLIED_WIN_PROB'
        ]
        
        for col in betting_columns:
            games_with_odds[col] = None
        
        odds_count = 0
        
        # Process each game
        for idx, game in games_with_odds.iterrows():
            try:
                game_date = game['GAME_DATE'].strftime('%Y-%m-%d')
                team_id_home = game['TEAM_ID_HOME']
                team_id_away = game['TEAM_ID_AWAY']
                
                # Get odds for this game
                odds = self.betting_odds_service.get_historical_odds(
                    game_date, team_id_home, team_id_away
                )
                
                # Add odds data to the DataFrame
                games_with_odds.at[idx, 'SPREAD_HOME'] = odds['spread']
                games_with_odds.at[idx, 'SPREAD_AWAY'] = -odds['spread']  # Away spread is inverse of home
                games_with_odds.at[idx, 'OVER_UNDER'] = odds['over_under']
                games_with_odds.at[idx, 'MONEYLINE_HOME'] = odds['moneyline_home']
                games_with_odds.at[idx, 'MONEYLINE_AWAY'] = odds['moneyline_away']
                games_with_odds.at[idx, 'IMPLIED_WIN_PROB'] = odds['implied_home_win_prob']
                
                odds_count += 1
                
                # Print progress every 500 games
                if odds_count % 500 == 0:
                    print(f"Added odds for {odds_count} games...")
                
            except Exception as e:
                # Use default values if odds retrieval fails
                games_with_odds.at[idx, 'SPREAD_HOME'] = -3.0  # Default home advantage
                games_with_odds.at[idx, 'SPREAD_AWAY'] = 3.0
                games_with_odds.at[idx, 'OVER_UNDER'] = 220.0
                games_with_odds.at[idx, 'MONEYLINE_HOME'] = -150
                games_with_odds.at[idx, 'MONEYLINE_AWAY'] = 130
                games_with_odds.at[idx, 'IMPLIED_WIN_PROB'] = 0.6
        
        print(f"Added betting odds for {odds_count} games")
        return games_with_odds
        

    def fetch_advanced_metrics(self, season: str) -> pd.DataFrame:
        """
        Fetch advanced team metrics for a given season.
        Uses cache when available to avoid redundant API calls.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Advanced team metrics or empty DataFrame if fetch fails
        """
        # Create cache parameters
        cache_params = {
            'season': season,
            'type': 'advanced_metrics',
            'api_version': '1.0'
        }
        
        # Check cache first
        if self.use_cache:
            cached_metrics = self.cache_manager.get_cache('metrics', cache_params)
            if cached_metrics is not None and not self.cache_manager.is_cache_stale('metrics', cache_params, self.cache_max_age_days):
                print(f"Using cached advanced metrics for {season}")
                return cached_metrics
        
        try:
            from nba_api.stats.endpoints import teamestimatedmetrics

            print(f"Fetching fresh advanced metrics for {season}...")
            metrics = teamestimatedmetrics.TeamEstimatedMetrics(
                season=season,
                season_type='Regular Season'
            ).get_data_frames()[0]

            print(f"Successfully fetched advanced metrics for {season}")
            
            # Cache the results
            if self.use_cache and not metrics.empty:
                self.cache_manager.set_cache('metrics', cache_params, metrics)
                print(f"Cached advanced metrics for {season}")
                
            time.sleep(0.25)  # Reduced sleep time for faster execution
            return metrics

        except ImportError:
            print(f"Warning: teamestimatedmetrics endpoint not available - skipping advanced metrics for {season}")
            return pd.DataFrame()

        except Exception as e:
            print(f"Warning: Could not fetch advanced metrics for {season} - {str(e)}")
            
            # Try to use stale cache data in case of an error
            if self.use_cache:
                cached_metrics = self.cache_manager.get_cache('metrics', cache_params)
                if cached_metrics is not None:
                    print(f"Using stale cached metrics for {season} due to fetch error")
                    return cached_metrics
                    
            return pd.DataFrame()
            
    def fetch_player_availability(self, season: str) -> pd.DataFrame:
        """
        Fetch player availability data for a season, including detailed lineup-based impact scores
        and star player matchup information.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Enhanced player availability data
        """
        # Create cache parameters for player availability data
        avail_cache_params = {
            'season': season,
            'type': 'player_availability',
            'api_version': '1.1'  # Updated version to fix merging issues
        }
        
        # Check for cached availability data
        if self.use_cache:
            cached_availability = self.cache_manager.get_cache('player_availability', avail_cache_params)
            if cached_availability is not None and not self.cache_manager.is_cache_stale('player_availability', avail_cache_params, self.cache_max_age_days):
                print(f"Using cached player availability data for {season}")
                return cached_availability
                
        # Create cache parameters for player impact data (used in availability processing)
        player_cache_params = {
            'season': season,
            'type': 'player_impact',
            'api_version': '1.1'  # Updated version to fix data structure issues
        }
        
        # Check for cached player impact data
        player_data = None
        if self.use_cache:
            player_data = self.cache_manager.get_cache('player_impact', player_cache_params)
            if player_data is not None and not self.cache_manager.is_cache_stale('player_impact', player_cache_params, self.cache_max_age_days):
                print(f"Using cached player impact data for {season}")
                
        # Initialize empty player_data with proper structure if it's None
        if player_data is None:
            player_data = {
                'player_impact': {}, 
                'team_lineup_strength': {}, 
                'player_stats_by_id': {}, 
                'player_position_map': {}
            }
                
        # Fallback to old cache system for backward compatibility
        # Define old cache paths
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        player_cache_file = os.path.join(cache_dir, f'player_data_{season.replace("-", "_")}.pkl')
        game_cache_file = os.path.join(cache_dir, f'game_availability_{season.replace("-", "_")}.pkl')
        
        # Try to load from old cache for availability data
        try:
            if os.path.exists(game_cache_file):
                print(f"Loading player availability data from legacy cache for {season}...")
                availability_data = pd.read_pickle(game_cache_file)
                if not availability_data.empty:
                    print(f"Successfully loaded {len(availability_data)} player availability records from legacy cache")
                    
                    # Verify data structure - add any missing columns
                    required_columns = [
                        'GAME_ID', 'GAME_ID_HOME', 'TEAM_ID', 'IS_HOME', 
                        'PLAYERS_AVAILABLE', 'STARTERS_AVAILABLE', 'LINEUP_IMPACT'
                    ]
                    
                    for column in required_columns:
                        if column not in availability_data.columns:
                            print(f"Adding missing column {column} to availability data")
                            availability_data[column] = 0
                    
                    # Migrate to new cache system
                    if self.use_cache:
                        self.cache_manager.set_cache('player_availability', avail_cache_params, availability_data)
                        print(f"Migrated legacy cache to new cache system")
                        
                    return availability_data
        except Exception as e:
            print(f"Error loading legacy availability cache, will fetch fresh data: {e}")
            
        # Try to load player impact data from old cache if player_data is empty
        if not player_data.get('player_impact'):
            try:
                if os.path.exists(player_cache_file):
                    print(f"Loading player data from legacy cache for {season}...")
                    loaded_data = pd.read_pickle(player_cache_file)
                    
                    # Ensure we have a dictionary with the expected structure
                    if isinstance(loaded_data, dict):
                        player_data = loaded_data
                    elif isinstance(loaded_data, pd.DataFrame):
                        # Convert DataFrame to expected structure if needed
                        player_data = {
                            'player_impact': {}, 
                            'team_lineup_strength': {}, 
                            'player_stats_by_id': {}, 
                            'player_position_map': {}
                        }
                    
                    # Migrate to new cache system
                    if self.use_cache:
                        self.cache_manager.set_cache('player_impact', player_cache_params, player_data)
                        print(f"Migrated legacy player data to new cache system")
            except Exception as e:
                print(f"Error loading legacy player cache: {e}")
                # Clear/reset player_data structure if there was an error loading
                player_data = {
                    'player_impact': {}, 
                    'team_lineup_strength': {}, 
                    'player_stats_by_id': {}, 
                    'player_position_map': {}
                }
                
        try:
            from nba_api.stats.endpoints import teamplayerdashboard, boxscoreadvancedv2, playergamelogs
            from nba_api.stats.endpoints import leaguedashplayerstats
            
            print(f"Fetching player availability data for {season}...")
            
            # Get all teams for the season with their player impact data
            # We need to fetch PIE data for each team's players
            try:
                teams = self.fetch_teams(season)
                unique_teams = teams['TEAM_ID'].unique() if not teams.empty else []
                print(f"Calculating player impact metrics for {len(unique_teams)} teams using parallel processing...")
            except Exception as e:
                print(f"Error getting teams: {e}")
                teams = pd.DataFrame({'TEAM_ID': [], 'TEAM_NAME': []})
                unique_teams = []
                print("Using fallback team data (empty)")
                
            # Set up player impact dictionaries
            player_impact = {}
            team_lineup_strength = {}
            
            # Skip if no teams found
            if len(unique_teams) > 0:
                # Prepare for parallel team processing
                import concurrent.futures
                
                # Function to fetch PIE data for a single player with rate limiting
                def fetch_player_pie(player_id, season, team_abbrev):
                    try:
                        from nba_api.stats.endpoints import playerdashboardbygeneralsplits
                        
                        print(f"Fetching PIE data for player {player_id} on team {team_abbrev}")
                        
                        # This endpoint requires player_id and gives PIE data
                        player_data = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                            player_id=player_id,
                            season=season,
                            measure_type_detailed='Advanced',
                            per_mode_detailed='PerGame'
                        ).get_data_frames()[0]
                        
                        # If we have PIE data, return it
                        if 'PIE' in player_data.columns and not player_data.empty:
                            pie_value = player_data['PIE'].iloc[0]
                            print(f"Successfully fetched PIE value {pie_value:.4f} for player {player_id}")
                            return {'PLAYER_ID': player_id, 'PIE': pie_value}
                        return None
                    except Exception as player_err:
                        print(f"Error fetching PIE for player {player_id}: {str(player_err)[:100]}")
                        return None
                
                # Function to process a single team's player impact data
                def process_team_impact(team_id, season):
                    team_abbrev = self.get_team_abbrev(team_id)
                    print(f"Processing team {team_abbrev} impact data...")
                    
                    try:
                        # Fetch team player dashboard
                        try:
                            from nba_api.stats.endpoints import teamplayerdashboard
                            
                            # Get the team's player data
                            team_players = teamplayerdashboard.TeamPlayerDashboard(
                                team_id=team_id,
                                season=season,
                                per_mode_simple='PerGame'
                            ).get_data_frames()[1]  # [1] contains individual player data
                        except Exception as e:
                            print(f"Error fetching team dashboard for {team_abbrev}: {str(e)[:100]}")
                            # Create mock player data as fallback
                            mock_players = []
                            for i in range(15):  # Typical NBA roster size
                                mock_players.append({
                                    'PLAYER_ID': int(f"{team_id}{i:02d}"),
                                    'PLAYER_NAME': f"Player {i+1}",
                                    'MIN': max(10, 25 - i * 1.5),
                                    'PTS': max(5, 20 - i * 1.5),
                                    'AST': max(1, 6 - i * 0.5),
                                    'REB': max(1, 8 - i * 0.5),
                                    'STL': max(0.5, 1.5 - i * 0.1),
                                    'BLK': max(0.2, 1.0 - i * 0.1)
                                })
                            team_players = pd.DataFrame(mock_players)
                        
                        # Check if PIE is missing and fetch it using parallel processing
                        if not team_players.empty and 'PIE' not in team_players.columns:
                            print(f"PIE not found for team {team_abbrev}, fetching with parallel processing")
                            
                            # Get player IDs for this team (up to 5 key players to avoid excessive API calls)
                            if 'PLAYER_ID' in team_players.columns:
                                player_ids = team_players['PLAYER_ID'].tolist()[:5]
                                
                                if player_ids:
                                    # Use ThreadPoolExecutor for parallel API requests
                                    pie_data_list = []
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                                        # Submit all player requests with staggered timing
                                        futures = []
                                        for player_id in player_ids:
                                            futures.append(executor.submit(
                                                fetch_player_pie, player_id, season, team_abbrev
                                            ))
                                            time.sleep(0.5)  # Stagger requests to avoid rate limiting
                                        
                                        # Process results as they complete
                                        for future in concurrent.futures.as_completed(futures):
                                            result = future.result()
                                            if result:
                                                pie_data_list.append(result)
                                    
                                    # If we collected PIE data, merge it with team_players
                                    if pie_data_list:
                                        pie_subset = pd.DataFrame(pie_data_list)
                                        print(f"Adding PIE data for {len(pie_subset)} players on {team_abbrev}")
                                        
                                        # Merge with team_players
                                        team_players = pd.merge(
                                            team_players,
                                            pie_subset,
                                            on='PLAYER_ID',
                                            how='left'
                                        )
                        
                        # If PIE is still missing, calculate it
                        if 'PIE' not in team_players.columns:
                            print(f"Calculating PIE for team {team_abbrev}")
                            # Add a synthetic PIE based on other stats
                            if 'PTS' in team_players.columns and 'MIN' in team_players.columns:
                                team_players['PIE'] = (
                                    (team_players['PTS'] + 
                                     team_players.get('REB', 0) * 1.2 + 
                                     team_players.get('AST', 0) * 1.5 + 
                                     team_players.get('STL', 0) * 2 + 
                                     team_players.get('BLK', 0) * 2) / 
                                    (team_players['MIN'].replace(0, 10))
                                ) / 15.0  # Scale to realistic PIE range
                            else:
                                # Default value if no stats available
                                team_players['PIE'] = 0.1
                        
                        # Add a default position if missing
                        if 'POSITION' not in team_players.columns:
                            team_players['POSITION'] = 'F'  # Default all to forwards
                        
                        # Calculate player impact scores
                        team_players['IMPACT_SCORE'] = team_players['PIE'] * 100  # Simple conversion
                        
                        # Get only relevant columns
                        if 'PLAYER_NAME' not in team_players.columns:
                            team_players['PLAYER_NAME'] = [f"Player {i+1}" for i in range(len(team_players))]
                            
                        result = {
                            'players': team_players[['PLAYER_ID', 'PLAYER_NAME', 'PIE', 'POSITION', 'IMPACT_SCORE']].to_dict('records'),
                            'team_id': team_id,
                            'team_abbrev': team_abbrev
                        }
                        
                        return result
                    except Exception as e:
                        print(f"Error processing team {team_abbrev}: {str(e)[:100]}")
                        return {
                            'players': [],
                            'team_id': team_id,
                            'team_abbrev': team_abbrev,
                            'error': str(e)
                        }
                
                # Process teams in parallel
                print("Starting parallel processing of team data...")
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit team processing tasks
                    futures = []
                    for team_id in unique_teams:
                        futures.append(executor.submit(process_team_impact, team_id, season))
                        time.sleep(0.5)  # Stagger starts to avoid rate limiting
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            team_id = result['team_id']
                            team_abbrev = result['team_abbrev']
                            
                            if 'error' not in result and result['players']:
                                print(f"Successfully processed team {team_abbrev}")
                                # Convert player data to DataFrame
                                player_impact[team_id] = pd.DataFrame(result['players'])
                                
                                # Calculate team strength metrics
                                players_df = pd.DataFrame(result['players'])
                                if not players_df.empty:
                                    # Calculate positional strengths
                                    guards = players_df[players_df['POSITION'].isin(['G', 'PG', 'SG'])]
                                    forwards = players_df[players_df['POSITION'].isin(['F', 'SF', 'PF'])]
                                    centers = players_df[players_df['POSITION'].isin(['C'])]
                                    
                                    team_lineup_strength[team_id] = {
                                        'TOTAL': players_df.nlargest(8, 'IMPACT_SCORE')['IMPACT_SCORE'].sum(),
                                        'GUARDS': guards.nlargest(3, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not guards.empty else 0,
                                        'FORWARDS': forwards.nlargest(3, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not forwards.empty else 0,
                                        'CENTERS': centers.nlargest(2, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not centers.empty else 0
                                    }
                            else:
                                print(f"Failed to process team {team_abbrev}")
                        except Exception as e:
                            print(f"Error processing result: {str(e)[:100]}")
            
            # Now get all games for the season
            print("Fetching game data for player availability analysis...")
            all_games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
            
            # Generate availability data for each game
            availability_data = []
            sample_size = min(100, len(all_games['GAME_ID'].unique()))
            
            # Process games in parallel batches for faster data generation
            import numpy as np
            unique_games = all_games['GAME_ID'].unique()
            sampled_games = np.random.choice(unique_games, size=sample_size, replace=False)
            
            print(f"Creating availability data for {sample_size} games...")
            
            # Function to process a batch of games in parallel
            def process_game_batch(game_ids):
                batch_results = []
                for game_id in game_ids:
                    try:
                        # Get teams for this game
                        game_data = all_games[all_games['GAME_ID'] == game_id]
                        if game_data.empty:
                            continue
                            
                        # Determine home and away teams
                        home_team = None
                        away_team = None
                        
                        if 'MATCHUP' in game_data.columns:
                            home_games = game_data[game_data['MATCHUP'].str.contains('vs', na=False)]
                            away_games = game_data[game_data['MATCHUP'].str.contains('@', na=False)]
                            
                            if not home_games.empty:
                                home_team = home_games['TEAM_ID'].iloc[0]
                            if not away_games.empty:
                                away_team = away_games['TEAM_ID'].iloc[0]
                        
                        # Fallback if we couldn't determine teams
                        if home_team is None or away_team is None:
                            teams = game_data['TEAM_ID'].unique()
                            if len(teams) >= 2:
                                home_team, away_team = teams[:2]
                            elif len(teams) == 1:
                                home_team = away_team = teams[0]
                            else:
                                # Skip if no teams found
                                continue
                        
                        # Use team strength data if available
                        home_strength = team_lineup_strength.get(home_team, {}).get('TOTAL', 50.0)
                        away_strength = team_lineup_strength.get(away_team, {}).get('TOTAL', 50.0)
                        
                        # Add home team data
                        batch_results.append({
                            'GAME_ID': game_id,
                            'GAME_ID_HOME': game_id,
                            'TEAM_ID': home_team,
                            'IS_HOME': 1,
                            'PLAYERS_AVAILABLE': 12,  # Default full roster
                            'STARTERS_AVAILABLE': 5,
                            'LINEUP_IMPACT': home_strength,
                            'GUARD_STRENGTH': team_lineup_strength.get(home_team, {}).get('GUARDS', 20.0),
                            'FORWARD_STRENGTH': team_lineup_strength.get(home_team, {}).get('FORWARDS', 20.0),
                            'CENTER_STRENGTH': team_lineup_strength.get(home_team, {}).get('CENTERS', 10.0),
                            'GUARD_ADVANTAGE': 0,
                            'FORWARD_ADVANTAGE': 0,
                            'CENTER_ADVANTAGE': 0,
                            'STAR_MATCHUP_ADVANTAGE': 0
                        })
                        
                        # Calculate advantages
                        guard_adv = (team_lineup_strength.get(home_team, {}).get('GUARDS', 20.0) - 
                                     team_lineup_strength.get(away_team, {}).get('GUARDS', 20.0))
                        forward_adv = (team_lineup_strength.get(home_team, {}).get('FORWARDS', 20.0) - 
                                      team_lineup_strength.get(away_team, {}).get('FORWARDS', 20.0))
                        center_adv = (team_lineup_strength.get(home_team, {}).get('CENTERS', 10.0) - 
                                     team_lineup_strength.get(away_team, {}).get('CENTERS', 10.0))
                        star_adv = home_strength - away_strength
                        
                        # Update home team advantages
                        batch_results[-1]['GUARD_ADVANTAGE'] = guard_adv
                        batch_results[-1]['FORWARD_ADVANTAGE'] = forward_adv
                        batch_results[-1]['CENTER_ADVANTAGE'] = center_adv
                        batch_results[-1]['STAR_MATCHUP_ADVANTAGE'] = star_adv
                        
                        # Add away team data
                        batch_results.append({
                            'GAME_ID': game_id,
                            'GAME_ID_HOME': game_id,
                            'TEAM_ID': away_team,
                            'IS_HOME': 0,
                            'PLAYERS_AVAILABLE': 12,  # Default full roster
                            'STARTERS_AVAILABLE': 5,
                            'LINEUP_IMPACT': away_strength,
                            'GUARD_STRENGTH': team_lineup_strength.get(away_team, {}).get('GUARDS', 20.0),
                            'FORWARD_STRENGTH': team_lineup_strength.get(away_team, {}).get('FORWARDS', 20.0),
                            'CENTER_STRENGTH': team_lineup_strength.get(away_team, {}).get('CENTERS', 10.0),
                            'GUARD_ADVANTAGE': -guard_adv,
                            'FORWARD_ADVANTAGE': -forward_adv,
                            'CENTER_ADVANTAGE': -center_adv,
                            'STAR_MATCHUP_ADVANTAGE': -star_adv
                        })
                    except Exception as e:
                        print(f"Error processing game {game_id}: {str(e)[:100]}")
                
                return batch_results
            
            # Split games into batches for parallel processing
            batch_size = 10
            game_batches = [sampled_games[i:i+batch_size] for i in range(0, len(sampled_games), batch_size)]
            
            # Process batches in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                futures = [executor.submit(process_game_batch, batch) for batch in game_batches]
                
                # Collect results
                for future in concurrent.futures.as_completed(futures):
                    try:
                        batch_results = future.result()
                        availability_data.extend(batch_results)
                    except Exception as e:
                        print(f"Error processing batch: {str(e)[:100]}")
            
            # Create final DataFrame
            result_df = pd.DataFrame(availability_data) if availability_data else pd.DataFrame({
                'GAME_ID': [],
                'GAME_ID_HOME': [],
                'TEAM_ID': [],
                'IS_HOME': [],
                'PLAYERS_AVAILABLE': [],
                'STARTERS_AVAILABLE': [],
                'LINEUP_IMPACT': [],
                'GUARD_STRENGTH': [],
                'FORWARD_STRENGTH': [],
                'CENTER_STRENGTH': [],
                'GUARD_ADVANTAGE': [],
                'FORWARD_ADVANTAGE': [],
                'CENTER_ADVANTAGE': [],
                'STAR_MATCHUP_ADVANTAGE': []
            })
            
            # Cache the result
            if self.use_cache and not result_df.empty:
                self.cache_manager.set_cache('player_availability', avail_cache_params, result_df)
                print(f"Cached player availability data with {len(result_df)} records")
            
            return result_df
            
        except Exception as e:
            print(f"Error fetching player availability: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty DataFrame with expected structure for graceful failure
            return pd.DataFrame({
                'GAME_ID': [],
                'GAME_ID_HOME': [],  # Add this for easier merging
                'TEAM_ID': [],
                'IS_HOME': [],
                'PLAYERS_AVAILABLE': [],
                'STARTERS_AVAILABLE': [],
                'LINEUP_IMPACT': [],
                'GUARD_STRENGTH': [],
                'FORWARD_STRENGTH': [],
                'CENTER_STRENGTH': [],
                'GUARD_ADVANTAGE': [],
                'FORWARD_ADVANTAGE': [],
                'CENTER_ADVANTAGE': [],
                'STAR_MATCHUP_ADVANTAGE': []
            })
        
        except Exception as e:
            print(f"Error fetching player availability: {e}")
            import traceback
            traceback.print_exc()
            
            # Return empty DataFrame with expected structure for graceful failure
            return pd.DataFrame({
                'GAME_ID': [],
                'GAME_ID_HOME': [],  # Add this for easier merging
                'TEAM_ID': [],
                'IS_HOME': [],
                'PLAYERS_AVAILABLE': [],
                'STARTERS_AVAILABLE': [],
                'LINEUP_IMPACT': [],
                'GUARD_STRENGTH': [],
                'FORWARD_STRENGTH': [],
                'CENTER_STRENGTH': [],
                'GUARD_ADVANTAGE': [],
                'FORWARD_ADVANTAGE': [],
                'CENTER_ADVANTAGE': [],
                'STAR_MATCHUP_ADVANTAGE': []
            })
    
    def get_team_abbrev(self, team_id):
        """
        Get team abbreviation from team ID.
        
        Args:
            team_id: NBA API team ID
            
        Returns:
            str: Team abbreviation or 'UNK' if not found
        """
        return TEAM_ID_TO_ABBREV.get(team_id, 'UNK')
            
    def fetch_teams(self, season: str) -> pd.DataFrame:
        """
        Fetch all NBA teams active during a given season.
        Uses cache when available to avoid redundant API calls.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Teams data
        """
        # Create cache parameters
        cache_params = {
            'season': season,
            'type': 'teams',
            'api_version': '1.0'
        }
        
        # Check cache first
        if self.use_cache:
            cached_teams = self.cache_manager.get_cache('teams', cache_params)
            if cached_teams is not None and not self.cache_manager.is_cache_stale('teams', cache_params, self.cache_max_age_days):
                print(f"Using cached teams data for {season}")
                return cached_teams
                
        try:
            from nba_api.stats.endpoints import commonteamyears, teaminfocommon
            
            print(f"Fetching fresh teams data for {season}...")
            
            # Get teams for the specified season
            # Get all teams data first
            teams_data = commonteamyears.CommonTeamYears().get_data_frames()[0]
            
            # Debug available seasons
            if 'SEASON_ID' in teams_data.columns:
                available_seasons = teams_data['SEASON_ID'].unique()
                print(f"Available seasons in API: {available_seasons[:5]}...")
            else:
                print("Warning: SEASON_ID column not found in API response")
                # Create a fallback DataFrame with NBA team IDs
                return self._create_fallback_teams_data()
            
            # Convert from '2022-23' format to expected format like '22022' or '22023'
            # Different NBA API endpoints use different formats
            season_year = season.split('-')[0]
            possible_formats = [
                f"{season_year}-{str(int(season_year)+1)[-2:]}",  # 2022-23
                f"2{season_year}",                                # 22022
                f"2{str(int(season_year)+1)}",                    # 22023
                season                                            # Original format
            ]
            
            # Try each format until we find a match
            season_teams = pd.DataFrame()
            for format in possible_formats:
                temp = teams_data[teams_data['SEASON_ID'] == format]
                if not temp.empty:
                    print(f"Found matching season format: {format}")
                    season_teams = temp
                    break
                    
            # If no match, use all teams as fallback
            if season_teams.empty:
                print(f"No exact season match found, using all teams as fallback")
                season_teams = teams_data
            
            team_details = []
            
            # Get detailed info for each team
            for team_id in season_teams['TEAM_ID'].unique():
                try:
                    # Get team info with correct parameter name
                    team_info = teaminfocommon.TeamInfoCommon(
                        team_id=team_id,
                        season_nullable=season
                    ).get_data_frames()[0]
                    
                    if not team_info.empty:
                        team_details.append(team_info)
                        
                    time.sleep(1)  # Respect API rate limits
                    
                except Exception as e:
                    print(f"Error fetching info for team {team_id}: {e}")
                    continue
            
            teams_df = pd.DataFrame(columns=['TEAM_ID', 'TEAM_NAME'])
            if team_details:
                teams_df = pd.concat(team_details, ignore_index=True)
                
            # Cache the results
            if self.use_cache and not teams_df.empty:
                self.cache_manager.set_cache('teams', cache_params, teams_df)
                print(f"Cached teams data for {season}")
                
            return teams_df
                
        except Exception as e:
            print(f"Error fetching teams: {e}")
            
            # Try to use stale cache in case of error
            if self.use_cache:
                cached_teams = self.cache_manager.get_cache('teams', cache_params)
                if cached_teams is not None:
                    print(f"Using stale cached teams data for {season} due to fetch error")
                    return cached_teams
                    
            # Return fallback teams data
            return self._create_fallback_teams_data()
            
    def _create_fallback_teams_data(self) -> pd.DataFrame:
        """
        Create a fallback DataFrame with all NBA team IDs when API fails.
        
        Returns:
            DataFrame with team IDs and names
        """
        # Get team data from constants
        team_data = []
        for team_id, abbrev in TEAM_ID_TO_ABBREV.items():
            team_name = TEAM_LOCATIONS.get(team_id, "Unknown") + " " + abbrev
            team_data.append({
                'TEAM_ID': team_id,
                'TEAM_NAME': team_name,
                'TEAM_ABBREVIATION': abbrev
            })
        
        teams_df = pd.DataFrame(team_data)
        print(f"Created fallback teams data with {len(teams_df)} teams")
        return teams_df
        
    def _create_mock_player_stats(self) -> pd.DataFrame:
        """
        Create mock player statistics when the API call fails.
        
        Returns:
            DataFrame with mock player statistics
        """
        # Generate mock player data
        mock_players = []
        
        # Use team IDs from constants to create realistic mock data
        for team_id, team_abbrev in TEAM_ID_TO_ABBREV.items():
            # Create 15 players for each team (typical NBA roster size)
            for i in range(15):
                # Create unique player ID
                player_id = int(f"{team_id}{i:02d}")
                
                # Determine mock position based on player number
                if i < 4:  # First 4 players are guards
                    position = 'G'
                elif i < 9:  # Next 5 are forwards
                    position = 'F'
                else:  # Rest are centers
                    position = 'C'
                
                # Create player stats with realistic values
                mock_players.append({
                    'PLAYER_ID': player_id,
                    'PLAYER_NAME': f"{team_abbrev} Player {i+1}",
                    'TEAM_ID': team_id,
                    'TEAM_ABBREVIATION': team_abbrev,
                    'PLAYER_POSITION': position,
                    'MIN': max(10, 25 - i * 1.5),  # Minutes decrease as player number increases
                    'PTS': max(5, 20 - i * 1.5),   # Points decrease as player number increases
                    'AST': max(1, 6 - i * 0.5),    # Assists decrease as player number increases
                    'REB': max(1, 8 - i * 0.5),    # Rebounds decrease as player number increases
                    'STL': max(0.5, 1.5 - i * 0.1),  # Steals decrease as player number increases
                    'BLK': max(0.2, 1.0 - i * 0.1),  # Blocks decrease as player number increases
                    'PIE': max(0.05, 0.15 - i * 0.01),  # Player Impact Estimate
                    'PLUS_MINUS': max(-5, 5 - i * 0.7),  # Plus/minus decreases as player number increases
                    'USG_PCT': max(10, 30 - i * 2.0)  # Usage percentage decreases as player number increases
                })
        
        # Create DataFrame from mock data
        mock_df = pd.DataFrame(mock_players)
        print(f"Created mock player stats for {len(mock_df)} players")
        return mock_df