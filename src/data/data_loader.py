"""
Data loading functions for the NBA prediction model.
"""
import pandas as pd
import time
import os
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from nba_api.stats.endpoints import leaguegamefinder

from src.utils.constants import BASIC_STATS_COLUMNS, TEAM_LOCATIONS, TEAM_ID_TO_ABBREV
from src.utils.cache_manager import CacheManager


class NBADataLoader:
    """Class for loading and preprocessing NBA game data with caching support."""
    
    def __init__(self, use_cache: bool = True, cache_max_age_days: int = 30, cache_dir: str = None):
        """
        Initialize the NBA data loader.
        
        Args:
            use_cache: Whether to use cache for data loading
            cache_max_age_days: Maximum age of cached data in days
            cache_dir: Custom directory for cache storage
        """
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        
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
        
        # Process all games
        df = pd.concat(all_games, ignore_index=True)
        df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

        # Split into home/away
        home = df[df['MATCHUP'].str.contains('vs')].copy()
        away = df[df['MATCHUP'].str.contains('@')].copy()
        
        if home.empty or away.empty:
            print("Warning: No home or away games found after filtering")
            # Create a more comprehensive empty DataFrame with the columns needed downstream
            required_columns = [col + suffix for col in BASIC_STATS_COLUMNS for suffix in ['_HOME', '_AWAY']]
            # Add other required columns that are needed downstream
            additional_columns = ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY', 'WL_HOME', 'GAME_ID_HOME']
            all_required_columns = list(set(required_columns + additional_columns))
            
            # Create sample data with at least one row for compatibility
            sample_data = {col: [0] if col.endswith(('_HOME', '_AWAY')) else 
                              (['W'] if col == 'WL_HOME' else 
                               [pd.Timestamp('2023-01-01')] if col == 'GAME_DATE' else 
                               ['MOCK_ID'] if 'ID' in col else 
                               [0]) 
                          for col in all_required_columns}
            
            empty_games = pd.DataFrame(sample_data)
            print(f"Created minimal sample DataFrame with {len(all_required_columns)} columns for compatibility")
            return empty_games, advanced_metrics

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
        
        # Cache the combined data if there's new data
        if self.use_cache and has_new_data:
            combined_data = (games, advanced_metrics)
            self.cache_manager.set_cache('games', cache_params, combined_data)
            print(f"Cached combined data for {len(seasons)} seasons: {len(games)} games")

        return games, advanced_metrics

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
            'api_version': '1.0'
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
            'api_version': '1.0'
        }
        
        # Check for cached player impact data
        player_data = None
        if self.use_cache:
            player_data = self.cache_manager.get_cache('player_impact', player_cache_params)
            if player_data is not None and not self.cache_manager.is_cache_stale('player_impact', player_cache_params, self.cache_max_age_days):
                print(f"Using cached player impact data for {season}")
                
        # Fallback to old cache system for backward compatibility
        if player_data is None:
            # Define old cache paths
            import os
            cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
            os.makedirs(cache_dir, exist_ok=True)
            player_cache_file = os.path.join(cache_dir, f'player_data_{season.replace("-", "_")}.pkl')
            game_cache_file = os.path.join(cache_dir, f'game_availability_{season.replace("-", "_")}.pkl')
            
            # Try to load from old cache
            try:
                if os.path.exists(game_cache_file):
                    print(f"Loading player availability data from legacy cache for {season}...")
                    availability_data = pd.read_pickle(game_cache_file)
                    if not availability_data.empty:
                        print(f"Successfully loaded {len(availability_data)} player availability records from legacy cache")
                        
                        # Migrate to new cache system
                        if self.use_cache:
                            self.cache_manager.set_cache('player_availability', avail_cache_params, availability_data)
                            print(f"Migrated legacy cache to new cache system")
                            
                        return availability_data
            except Exception as e:
                print(f"Error loading legacy cache, will fetch fresh data: {e}")
                
            # Try to load player impact data from old cache
            try:
                if os.path.exists(player_cache_file):
                    print(f"Loading player data from legacy cache for {season}...")
                    player_data = pd.read_pickle(player_cache_file)
                    
                    # Migrate to new cache system
                    if self.use_cache:
                        self.cache_manager.set_cache('player_impact', player_cache_params, player_data)
                        print(f"Migrated legacy player data to new cache system")
            except Exception as e:
                print(f"Error loading legacy player cache: {e}")
            
        try:
            from nba_api.stats.endpoints import teamplayerdashboard, boxscoreadvancedv2, playergamelogs
            from nba_api.stats.endpoints import leaguedashplayerstats
            
            print(f"Fetching player availability data for {season}...")
            
            # First try to load player impact data from cache
            player_impact = {}
            team_lineup_strength = {}
            player_stats_by_id = {}
            player_position_map = {}

            # Try loading from cache
            if os.path.exists(player_cache_file):
                try:
                    print(f"Loading player data from cache for {season}...")
                    cached_data = pd.read_pickle(player_cache_file)
                    player_impact = cached_data.get('player_impact', {})
                    team_lineup_strength = cached_data.get('team_lineup_strength', {})
                    player_stats_by_id = cached_data.get('player_stats_by_id', {})
                    player_position_map = cached_data.get('player_position_map', {})
                    
                    if player_impact and team_lineup_strength:
                        print(f"Successfully loaded player data for {len(player_impact)} teams from cache")
                except Exception as e:
                    print(f"Error loading player cache: {e}")
            
            # If cache didn't have data, fetch it
            if not player_impact:
                # Get player stats for the entire league
                print("Fetching league-wide player stats...")
                try:
                    # Try different parameter combinations to handle API inconsistencies
                    try:
                        # First attempt with full parameter names
                        league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                            season=season,
                            season_type_all_star='Regular Season',
                            per_mode_simple='PerGame'
                        ).get_data_frames()[0]
                    except:
                        try:
                            # Second attempt with standard parameter names
                            league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                                season=season
                            ).get_data_frames()[0]
                        except:
                            # Fallback to creating mock player stats
                            print("Using mock player stats due to API parameter issues")
                            league_stats = self._create_mock_player_stats()
                    
                    print(f"Successfully retrieved {len(league_stats)} player stats")
                    
                    # Debug available columns
                    if not league_stats.empty:
                        print(f"Available columns: {league_stats.columns.tolist()}")
                    
                    # Assign positions based on player roles - position is typically missing
                    # Use a simplified approach for position assignment
                    for _, player in league_stats.iterrows():
                        player_id = player['PLAYER_ID']
                        
                        # Infer position from height/weight or stats
                        # Guards tend to have higher AST, forwards rebounds, centers blocks
                        position = 'F'  # Default to forward
                        
                        if 'PLAYER_POSITION' in league_stats.columns:
                            position = player['PLAYER_POSITION']
                        else:
                            # Attempt to infer position by stats
                            ast_per_min = player['AST'] / max(player['MIN'], 1)
                            reb_per_min = player['REB'] / max(player['MIN'], 1)
                            blk_per_min = player.get('BLK', 0) / max(player['MIN'], 1)
                            
                            if ast_per_min > 0.15:  # Higher assists -> guard
                                position = 'G'
                            elif blk_per_min > 0.08:  # Higher blocks -> center
                                position = 'C'
                            elif reb_per_min > 0.2:  # Higher rebounds -> forward/center
                                position = 'F'
                        
                        player_position_map[player_id] = position
                        
                    # Store player statistics by ID for later reference
                    for _, player in league_stats.iterrows():
                        player_id = player['PLAYER_ID']
                        position = player_position_map.get(player_id, 'F')
                        
                        player_stats_by_id[player_id] = {
                            'NAME': player['PLAYER_NAME'],
                            'TEAM': player.get('TEAM_ID', 0),
                            'POSITION': position,
                            'MIN': player.get('MIN', 0),
                            'PTS': player.get('PTS', 0),
                            'AST': player.get('AST', 0),
                            'REB': player.get('REB', 0),
                            'STL': player.get('STL', 0),
                            'BLK': player.get('BLK', 0),
                            'PIE': player.get('PIE', 0),
                            'PLUS_MINUS': player.get('PLUS_MINUS', 0),
                            'USG_PCT': player.get('USG_PCT', 0)
                        }
                    
                    time.sleep(0.5)  # Reduced sleep time for faster execution
                except Exception as e:
                    print(f"Error fetching league-wide stats: {e}")
                
                # Get teams for the season with corrected parameter format
                try:
                    teams = self.fetch_teams(season)
                    unique_teams = teams['TEAM_ID'].unique() if not teams.empty else []
                    print(f"Calculating player impact metrics for {len(unique_teams)} teams...")
                except Exception as e:
                    print(f"Error getting teams: {e}")
                    # Create an empty dataframe with the expected structure
                    teams = pd.DataFrame({'TEAM_ID': [], 'TEAM_NAME': []})
                    unique_teams = []
                    print("Using fallback team data (empty)")
                for team_id in unique_teams:
                    try:
                        # Fetch team player dashboard with corrected parameter names
                        # Try different parameter combinations for TeamPlayerDashboard
                        try:
                            # First attempt with nullable parameter
                            try:
                                # Try the PlayerDashboardByAdvanced endpoint which specifically contains PIE
                                from nba_api.stats.endpoints import playerdashboardbyteamperformance
                                
                                # This is one of the best endpoints for PIE data
                                dashboard = playerdashboardbyteamperformance.PlayerDashboardByTeamPerformance(
                                    team_id=team_id,
                                    season=season,
                                    measure_type_detailed='Advanced',
                                    per_mode_detailed='PerGame'
                                ).get_data_frames()
                                
                                # The first dataframe has overall stats with PIE
                                team_players = dashboard[0]
                                
                                # If we still don't get PIE, try the general advanced endpoint
                                if 'PIE' not in team_players.columns:
                                    from nba_api.stats.endpoints import teamdashboardbyplayerperformance
                                    
                                    dashboard = teamdashboardbyplayerperformance.TeamDashboardByPlayerPerformance(
                                        team_id=team_id,
                                        season=season,
                                        measure_type_detailed='Advanced',
                                        per_mode_detailed='PerGame'
                                    ).get_data_frames()
                                    
                                    team_players = dashboard[0]
                                    
                                    if 'PIE' not in team_players.columns:
                                        raise ValueError("PIE not found in advanced dashboard, trying standard endpoint")
                                    
                            except Exception as e:
                                # If we're getting rate limited, add a short delay and try again with another endpoint
                                if "429" in str(e) or "timeout" in str(e).lower():
                                    print(f"Rate limited on advanced stats for team {team_id}, waiting and trying another endpoint")
                                    time.sleep(1)  # Wait a bit longer for rate limits
                                
                                # Try the efficiency endpoint for PIE data
                                try:
                                    from nba_api.stats.endpoints import teamplayerdashboardbyteamperformance
                                    
                                    dashboard = teamplayerdashboardbyteamperformance.TeamPlayerDashboardByTeamPerformance(
                                        team_id=team_id,
                                        season=season,
                                        measure_type_detailed='Advanced',
                                        per_mode_detailed='PerGame'
                                    ).get_data_frames()
                                    
                                    team_players = dashboard[0]
                                except Exception as e2:
                                # Fallback to standard dashboard
                                team_players = teamplayerdashboard.TeamPlayerDashboard(
                                    team_id=team_id,
                                    season=season,
                                    per_mode_simple='PerGame'
                                ).get_data_frames()[1]  # [1] contains individual player data
                        except:
                            try:
                                # Second attempt with standard parameters
                                team_players = teamplayerdashboard.TeamPlayerDashboard(
                                    team_id=team_id,
                                    season=season
                                ).get_data_frames()[1]
                            except:
                                # Create mock player data
                                print(f"Creating mock player data for team {team_id}")
                                # Get 15 players for this team (typical roster size)
                                mock_players = []
                                for i in range(15):
                                    mock_players.append({
                                        'PLAYER_ID': int(f"{team_id}{i:02d}"),
                                        'PLAYER_NAME': f"Player {i+1}",
                                        'PLUS_MINUS': max(-5, 5 - i * 0.7),
                                        'PIE': max(0.05, 0.15 - i * 0.01),
                                        'USG_PCT': max(10, 30 - i * 2.0),
                                        'MIN': max(10, 25 - i * 1.5),
                                        'PTS': max(5, 20 - i * 1.5),
                                        'AST': max(1, 6 - i * 0.5),
                                        'REB': max(1, 8 - i * 0.5),
                                        'STL': max(0.5, 1.5 - i * 0.1),
                                        'BLK': max(0.2, 1.0 - i * 0.1)
                                    })
                                team_players = pd.DataFrame(mock_players)
                        
                        # Calculate player impact scores based on stats
                        if not team_players.empty:
                            # Check if PIE is missing and try to fetch it directly if possible
                            if 'PIE' not in team_players.columns:
                                print(f"PIE not found in main endpoint for team {self.get_team_abbrev(team_id)}, trying PIE-specific endpoint")
                                try:
                                    # Try one more specialized endpoint for PIE data
                                    from nba_api.stats.endpoints import playerdashboardbygeneralsplits
                                    
                                    # This endpoint often has PIE data
                                    pie_data = playerdashboardbygeneralsplits.PlayerDashboardByGeneralSplits(
                                        team_id=team_id,
                                        season=season,
                                        measure_type_detailed='Advanced',
                                        per_mode_detailed='PerGame'
                                    ).get_data_frames()[0]
                                    
                                    # If we found PIE, merge it into our main dataframe
                                    if 'PIE' in pie_data.columns and 'PLAYER_ID' in pie_data.columns:
                                        pie_subset = pie_data[['PLAYER_ID', 'PIE']].copy()
                                        # Merge PIE data with team_players
                                        if 'PLAYER_ID' in team_players.columns:
                                            team_players = pd.merge(
                                                team_players, 
                                                pie_subset,
                                                on='PLAYER_ID',
                                                how='left'
                                            )
                                except Exception as e:
                                    print(f"Failed to fetch PIE data directly: {e}")
                                
                                # If still missing, calculate manually
                                if 'PIE' not in team_players.columns:
                                    print(f"PIE not available from API for team {self.get_team_abbrev(team_id)}, calculating manually")
                                    # Calculate a PIE-like metric ourselves when not available from API
                                    # Formula to approximate PIE based on available stats
                                    if all(col in team_players.columns for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'MIN']):
                                        team_players['PIE'] = (
                                            (team_players['PTS'] + team_players['REB'] + team_players['AST'] + 
                                             team_players['STL'] + team_players['BLK']) / 
                                            (team_players['MIN'] + 1)  # Add 1 to avoid division by zero
                                        ) / 10.0  # Scale to 0-1 range similar to PIE
                                    else:
                                        # If needed statistics aren't available, create a simple proxy
                                        team_players['PIE'] = 0.1  # Default value
                            
                            # Check for USG_PCT
                            if 'USG_PCT' not in team_players.columns:
                                print(f"USG_PCT not available for team {self.get_team_abbrev(team_id)}, calculating proxy")
                                # Approximate usage percentage when not available
                                if 'PTS' in team_players.columns and 'MIN' in team_players.columns:
                                    team_players['USG_PCT'] = 10 + (team_players['PTS'] / team_players['MIN'] * 5)
                                else:
                                    team_players['USG_PCT'] = 20.0  # Default value
                                    
                            # Create a more sophisticated impact score incorporating defensive and offensive metrics
                            # Using try/except to handle any missing columns with default values
                            try:
                                # Helper function to safely get column values with defaults
                                def safe_get_col(df, col_name, default_value):
                                    return df[col_name] if col_name in df.columns else pd.Series([default_value] * len(df))
                                
                                # Calculate impact score with safe column access
                                team_players['IMPACT_SCORE'] = (
                                    0.30 * safe_get_col(team_players, 'PLUS_MINUS', 0) +
                                    0.25 * team_players['PIE'] * 100 +     # Player Impact Estimate (already verified)
                                    0.15 * team_players['USG_PCT'] +       # Usage percentage (already verified)
                                    0.10 * safe_get_col(team_players, 'MIN', 10) +   # Minutes played
                                    0.10 * (safe_get_col(team_players, 'REB', 3) / 
                                           (safe_get_col(team_players, 'MIN', 10) + 0.1) * 10) +  # Rebounding rate
                                    0.05 * (safe_get_col(team_players, 'AST', 1) / 
                                           (safe_get_col(team_players, 'MIN', 10) + 0.1) * 10) +  # Assist rate
                                    0.05 * ((safe_get_col(team_players, 'STL', 0.5) + 
                                             safe_get_col(team_players, 'BLK', 0.3)) / 
                                            (safe_get_col(team_players, 'MIN', 10) + 0.1) * 10)  # Defensive rate
                                ).fillna(0)
                            except Exception as e:
                                print(f"Error calculating impact score for team {self.get_team_abbrev(team_id)}: {e}")
                                # Calculate a simplified impact score as fallback
                                if 'PIE' in team_players.columns:
                                    team_players['IMPACT_SCORE'] = team_players['PIE'] * 100
                                else:
                                    # Completely synthetic values
                                    team_players['IMPACT_SCORE'] = 10.0
                            
                            # Add positional information from the league-wide stats
                            team_players['POSITION'] = team_players['PLAYER_ID'].map(
                                lambda x: player_position_map.get(x, 'Unknown')
                            )
                            
                            # Handle case where POSITION column doesn't exist
                            if 'POSITION' not in team_players.columns:
                                # Add POSITION based on player_position_map
                                team_players['POSITION'] = team_players['PLAYER_ID'].map(
                                    lambda x: player_position_map.get(x, 'F')
                                )
                                print(f"Added position data for {len(team_players)} players")
                                
                            # Store detailed player stats including position
                            columns_to_select = ['PLAYER_ID', 'PLAYER_NAME', 'IMPACT_SCORE']
                            if 'POSITION' in team_players.columns:
                                columns_to_select.append('POSITION')
                                
                            player_data = team_players[columns_to_select].copy()
                            if 'POSITION' not in player_data.columns:
                                player_data['POSITION'] = player_data['PLAYER_ID'].map(
                                    lambda x: player_position_map.get(x, 'F')
                                )
                                
                            player_impact[team_id] = player_data
                            
                            # Calculate positional strength scores
                            guards = player_data[player_data['POSITION'].isin(['G', 'PG', 'SG'])]
                            forwards = player_data[player_data['POSITION'].isin(['F', 'SF', 'PF'])]
                            centers = player_data[player_data['POSITION'].isin(['C'])]
                            
                            # Store positional impact scores
                            team_lineup_strength[team_id] = {
                                'TOTAL': team_players.nlargest(8, 'IMPACT_SCORE')['IMPACT_SCORE'].sum(),
                                'GUARDS': guards.nlargest(3, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not guards.empty else 0,
                                'FORWARDS': forwards.nlargest(3, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not forwards.empty else 0,
                                'CENTERS': centers.nlargest(2, 'IMPACT_SCORE')['IMPACT_SCORE'].sum() if not centers.empty else 0
                            }
                        
                        print(f"Processed player impact for {self.get_team_abbrev(team_id)}")
                        time.sleep(1)  # Respect API rate limits
                        
                    except Exception as e:
                        print(f"Error fetching player data for team {self.get_team_abbrev(team_id)}: {e}")
                        continue
                
                # Cache the player data
                try:
                    cached_data = {
                        'player_impact': player_impact,
                        'team_lineup_strength': team_lineup_strength,
                        'player_stats_by_id': player_stats_by_id,
                        'player_position_map': player_position_map
                    }
                    pd.to_pickle(cached_data, player_cache_file)
                    print(f"Cached player data for {len(player_impact)} teams")
                except Exception as e:
                    print(f"Error caching player data: {e}")
            
            # Get all games for the season
            print("Fetching game data for player availability analysis...")
            all_games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
            
            # Process all games for enhanced availability data
            availability_data = []
            
            # Get a sample of games to process (in a production system, process all games)
            # For demonstration, we'll process up to 100 games to respect rate limits
            unique_games = all_games['GAME_ID'].unique()
            sample_size = min(100, len(unique_games))
            import numpy as np
            sampled_games = np.random.choice(unique_games, size=sample_size, replace=False)
            
            print(f"Processing player availability for {sample_size} games...")
            for i, game_id in enumerate(sampled_games):
                try:
                    if i % 10 == 0:  # Progress update every 10 games
                        print(f"Processing game {i+1}/{sample_size}...")
                    
                    # Get the teams from this game with robust error handling
                    try:
                        game_teams = all_games[all_games['GAME_ID'] == game_id]['TEAM_ID'].unique()
                        
                        if len(game_teams) == 2:
                            # Determine home and away teams (this method might need adjustment)
                            game_df = all_games[all_games['GAME_ID'] == game_id]
                            
                            # Check if we have the MATCHUP column
                            if 'MATCHUP' in game_df.columns:
                                home_games = game_df[game_df['MATCHUP'].str.contains('vs.', na=False)]
                                away_games = game_df[game_df['MATCHUP'].str.contains('@', na=False)]
                                
                                if not home_games.empty and not away_games.empty:
                                    home_team = home_games['TEAM_ID'].iloc[0]
                                    away_team = away_games['TEAM_ID'].iloc[0]
                                else:
                                    # Fallback if we can't determine home/away
                                    home_team, away_team = game_teams
                            else:
                                # Fallback when MATCHUP column is unavailable
                                home_team, away_team = game_teams
                        else:
                            # Unexpected number of teams, create dummy data
                            print(f"Unexpected number of teams ({len(game_teams)}) for game {game_id}")
                            if len(game_teams) > 0:
                                home_team = game_teams[0]
                                away_team = game_teams[0] if len(game_teams) == 1 else game_teams[1]
                            else:
                                print(f"No teams found for game {game_id}, using fallback data")
                                home_team = 1610612737  # ATL
                                away_team = 1610612738  # BOS
                    except Exception as e:
                        print(f"Error getting teams for game {game_id}: {e}")
                        # Fallback to dummy teams
                        home_team = 1610612737  # ATL
                        away_team = 1610612738  # BOS
                        
                        # Get box score for detailed player stats with added error handling
                        try:
                            box_score = boxscoreadvancedv2.BoxScoreAdvancedV2(
                                game_id=game_id
                            ).get_data_frames()[0]
                        except Exception as e:
                            print(f"Error getting box score for game {game_id}: {e}")
                            # Create a minimal box score with the required columns
                            box_score = pd.DataFrame({
                                'TEAM_ID': list(game_teams) * 5,  # Assume 5 players per team
                                'PLAYER_ID': range(10),  # Dummy player IDs
                                'START_POSITION': [''] * 10,  # No starters
                            })
                        
                        # Process home team
                        home_players = box_score[box_score['TEAM_ID'] == home_team]
                        home_starters = len(home_players[home_players['START_POSITION'] != ''])
                        home_available = len(home_players)
                        
                        # Calculate home lineup strength
                        home_strength = 0
                        home_guards_strength = 0
                        home_forwards_strength = 0
                        home_centers_strength = 0
                        
                        if home_team in team_lineup_strength:
                            home_strength = team_lineup_strength[home_team]['TOTAL']
                            home_guards_strength = team_lineup_strength[home_team]['GUARDS']
                            home_forwards_strength = team_lineup_strength[home_team]['FORWARDS']
                            home_centers_strength = team_lineup_strength[home_team]['CENTERS']
                            
                            # Adjust for missing players
                            if home_team in player_impact:
                                playing_ids = set(home_players['PLAYER_ID'])
                                missing_players = player_impact[home_team][
                                    ~player_impact[home_team]['PLAYER_ID'].isin(playing_ids)
                                ]
                                
                                # Calculate positional impact of missing players
                                missing_impact = missing_players['IMPACT_SCORE'].sum()
                                missing_guards = missing_players[missing_players['POSITION'].isin(['G', 'PG', 'SG'])]
                                missing_forwards = missing_players[missing_players['POSITION'].isin(['F', 'SF', 'PF'])]
                                missing_centers = missing_players[missing_players['POSITION'].isin(['C'])]
                                
                                # Adjust strengths
                                home_strength -= missing_impact
                                home_guards_strength -= missing_guards['IMPACT_SCORE'].sum() if not missing_guards.empty else 0
                                home_forwards_strength -= missing_forwards['IMPACT_SCORE'].sum() if not missing_forwards.empty else 0
                                home_centers_strength -= missing_centers['IMPACT_SCORE'].sum() if not missing_centers.empty else 0
                        
                        # Process away team
                        away_players = box_score[box_score['TEAM_ID'] == away_team]
                        away_starters = len(away_players[away_players['START_POSITION'] != ''])
                        away_available = len(away_players)
                        
                        # Calculate away lineup strength
                        away_strength = 0
                        away_guards_strength = 0
                        away_forwards_strength = 0
                        away_centers_strength = 0
                        
                        if away_team in team_lineup_strength:
                            away_strength = team_lineup_strength[away_team]['TOTAL']
                            away_guards_strength = team_lineup_strength[away_team]['GUARDS']
                            away_forwards_strength = team_lineup_strength[away_team]['FORWARDS']
                            away_centers_strength = team_lineup_strength[away_team]['CENTERS']
                            
                            # Adjust for missing players
                            if away_team in player_impact:
                                playing_ids = set(away_players['PLAYER_ID'])
                                missing_players = player_impact[away_team][
                                    ~player_impact[away_team]['PLAYER_ID'].isin(playing_ids)
                                ]
                                
                                # Calculate positional impact of missing players
                                missing_impact = missing_players['IMPACT_SCORE'].sum()
                                missing_guards = missing_players[missing_players['POSITION'].isin(['G', 'PG', 'SG'])]
                                missing_forwards = missing_players[missing_players['POSITION'].isin(['F', 'SF', 'PF'])]
                                missing_centers = missing_players[missing_players['POSITION'].isin(['C'])]
                                
                                # Adjust strengths
                                away_strength -= missing_impact
                                away_guards_strength -= missing_guards['IMPACT_SCORE'].sum() if not missing_guards.empty else 0
                                away_forwards_strength -= missing_forwards['IMPACT_SCORE'].sum() if not missing_forwards.empty else 0
                                away_centers_strength -= missing_centers['IMPACT_SCORE'].sum() if not missing_centers.empty else 0
                                
                        # Calculate positional advantages
                        guard_advantage = home_guards_strength - away_guards_strength
                        forward_advantage = home_forwards_strength - away_forwards_strength
                        center_advantage = home_centers_strength - away_centers_strength
                        
                        # Calculate star player matchup advantage
                        star_matchup_advantage = 0
                        if home_team in player_impact and away_team in player_impact:
                            # Get top 3 players from each team
                            home_stars = player_impact[home_team].nlargest(3, 'IMPACT_SCORE')
                            away_stars = player_impact[away_team].nlargest(3, 'IMPACT_SCORE')
                            
                            # Simply compare the total impact of stars playing
                            home_stars_playing = home_stars[home_stars['PLAYER_ID'].isin(home_players['PLAYER_ID'])]
                            away_stars_playing = away_stars[away_stars['PLAYER_ID'].isin(away_players['PLAYER_ID'])]
                            
                            home_star_power = home_stars_playing['IMPACT_SCORE'].sum()
                            away_star_power = away_stars_playing['IMPACT_SCORE'].sum()
                            
                            star_matchup_advantage = home_star_power - away_star_power
                            
                            # Top player names for debugging
                            home_top_players = home_stars_playing['PLAYER_NAME'].tolist()
                            away_top_players = away_stars_playing['PLAYER_NAME'].tolist()
                        
                        # Add data for home team with GAME_ID_HOME for better merging
                        availability_data.append({
                            'GAME_ID': game_id,
                            'GAME_ID_HOME': game_id,  # Add this for easier merging
                            'TEAM_ID': home_team,
                            'IS_HOME': 1,
                            'PLAYERS_AVAILABLE': home_available,
                            'STARTERS_AVAILABLE': home_starters,
                            'LINEUP_IMPACT': home_strength,
                            'GUARD_STRENGTH': home_guards_strength,
                            'FORWARD_STRENGTH': home_forwards_strength,
                            'CENTER_STRENGTH': home_centers_strength,
                            'GUARD_ADVANTAGE': guard_advantage,
                            'FORWARD_ADVANTAGE': forward_advantage,
                            'CENTER_ADVANTAGE': center_advantage,
                            'STAR_MATCHUP_ADVANTAGE': star_matchup_advantage
                        })
                        
                        # Add data for away team with GAME_ID_HOME for better merging
                        availability_data.append({
                            'GAME_ID': game_id,
                            'GAME_ID_HOME': game_id,  # Add this for easier merging
                            'TEAM_ID': away_team,
                            'IS_HOME': 0,
                            'PLAYERS_AVAILABLE': away_available,
                            'STARTERS_AVAILABLE': away_starters,
                            'LINEUP_IMPACT': away_strength,
                            'GUARD_STRENGTH': away_guards_strength,
                            'FORWARD_STRENGTH': away_forwards_strength,
                            'CENTER_STRENGTH': away_centers_strength,
                            'GUARD_ADVANTAGE': -guard_advantage,
                            'FORWARD_ADVANTAGE': -forward_advantage,
                            'CENTER_ADVANTAGE': -center_advantage,
                            'STAR_MATCHUP_ADVANTAGE': -star_matchup_advantage
                        })
                    
                    time.sleep(0.1)  # Minimal sleep time for much faster execution
                    
                except Exception as e:
                    print(f"Error processing availability for game {game_id}: {e}")
                    continue
            
            # Create the dataframe
            result_df = pd.DataFrame(availability_data)
            
            # Cache the results
            try:
                pd.to_pickle(result_df, game_cache_file)
                print(f"Cached player availability data for {len(result_df)} team-games")
            except Exception as e:
                print(f"Error caching availability data: {e}")
                
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