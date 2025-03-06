"""
Data loading functions for the NBA prediction model.
"""
import pandas as pd
import time
from typing import Dict, List, Tuple
from nba_api.stats.endpoints import leaguegamefinder

from src.utils.constants import BASIC_STATS_COLUMNS, TEAM_LOCATIONS, TEAM_ID_TO_ABBREV


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
        Fetch player availability data for a season, including detailed lineup-based impact scores
        and star player matchup information.
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Enhanced player availability data
        """
        # Define a cache directory to store data between runs
        import os
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
        os.makedirs(cache_dir, exist_ok=True)
        
        # Define cache paths
        player_cache_file = os.path.join(cache_dir, f'player_data_{season.replace("-", "_")}.pkl')
        game_cache_file = os.path.join(cache_dir, f'game_availability_{season.replace("-", "_")}.pkl')
        
        # Try to load from cache first
        try:
            if os.path.exists(game_cache_file):
                print(f"Loading player availability data from cache for {season}...")
                availability_data = pd.read_pickle(game_cache_file)
                if not availability_data.empty:
                    print(f"Successfully loaded {len(availability_data)} player availability records from cache")
                    return availability_data
        except Exception as e:
            print(f"Error loading cache, will fetch fresh data: {e}")
            
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
                    league_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                        season=season,
                        season_type_all_star='Regular Season',
                        measure_type_detailed_defense='Advanced',
                        per_mode_detailed='PerGame'
                    ).get_data_frames()[0]
                    
                    # Map player positions
                    for _, player in league_stats.iterrows():
                        player_position_map[player['PLAYER_ID']] = player['POSITION']
                        
                    # Store player statistics by ID for later reference
                    for _, player in league_stats.iterrows():
                        player_stats_by_id[player['PLAYER_ID']] = {
                            'NAME': player['PLAYER_NAME'],
                            'TEAM': player['TEAM_ID'],
                            'POSITION': player['POSITION'],
                            'MIN': player['MIN'],
                            'PTS': player['PTS'],
                            'AST': player['AST'],
                            'REB': player['REB'],
                            'STL': player['STL'],
                            'BLK': player['BLK'],
                            'PIE': player.get('PIE', 0),
                            'PLUS_MINUS': player.get('PLUS_MINUS', 0),
                            'USG_PCT': player.get('USG_PCT', 0)
                        }
                    
                    time.sleep(2)  # Respect API rate limits
                except Exception as e:
                    print(f"Error fetching league-wide stats: {e}")
                
                # Get teams for the season
                teams = self.fetch_teams(season)
                
                print(f"Calculating player impact metrics for {len(teams['TEAM_ID'].unique())} teams...")
                for team_id in teams['TEAM_ID'].unique():
                    try:
                        # Fetch team player dashboard
                        team_players = teamplayerdashboard.TeamPlayerDashboard(
                            team_id=team_id,
                            season=season,
                            season_type_all_star='Regular Season'
                        ).get_data_frames()[1]  # [1] contains individual player data
                        
                        # Calculate player impact scores based on stats
                        if not team_players.empty:
                            # Create a more sophisticated impact score incorporating defensive and offensive metrics
                            team_players['IMPACT_SCORE'] = (
                                0.30 * team_players['PLUS_MINUS'] +
                                0.25 * team_players['PIE'] * 100 +     # Player Impact Estimate
                                0.15 * team_players['USG_PCT'] +       # Usage percentage 
                                0.10 * team_players['MIN'] +           # Minutes played
                                0.10 * (team_players['REB'] / team_players['MIN'] * 10) +  # Rebounding rate
                                0.05 * (team_players['AST'] / team_players['MIN'] * 10) +  # Assist rate
                                0.05 * ((team_players['STL'] + team_players['BLK']) / team_players['MIN'] * 10)  # Defensive rate
                            ).fillna(0)
                            
                            # Add positional information from the league-wide stats
                            team_players['POSITION'] = team_players['PLAYER_ID'].map(
                                lambda x: player_position_map.get(x, 'Unknown')
                            )
                            
                            # Store detailed player stats including position
                            player_impact[team_id] = team_players[['PLAYER_ID', 'PLAYER_NAME', 'POSITION', 'IMPACT_SCORE']]
                            
                            # Calculate positional strength scores
                            guards = team_players[team_players['POSITION'].isin(['G', 'PG', 'SG'])]
                            forwards = team_players[team_players['POSITION'].isin(['F', 'SF', 'PF'])]
                            centers = team_players[team_players['POSITION'].isin(['C'])]
                            
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
                    
                    # Get the teams from this game
                    game_teams = all_games[all_games['GAME_ID'] == game_id]['TEAM_ID'].unique()
                    
                    if len(game_teams) == 2:
                        # Determine home and away teams (this method might need adjustment)
                        game_df = all_games[all_games['GAME_ID'] == game_id]
                        home_games = game_df[game_df['MATCHUP'].str.contains('vs.')]
                        away_games = game_df[game_df['MATCHUP'].str.contains('@')]
                        
                        if not home_games.empty and not away_games.empty:
                            home_team = home_games['TEAM_ID'].iloc[0]
                            away_team = away_games['TEAM_ID'].iloc[0]
                        else:
                            # Fallback if we can't determine home/away
                            home_team, away_team = game_teams
                        
                        # Get box score for detailed player stats
                        box_score = boxscoreadvancedv2.BoxScoreAdvancedV2(
                            game_id=game_id
                        ).get_data_frames()[0]
                        
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
                        
                        # Add data for home team
                        availability_data.append({
                            'GAME_ID': game_id,
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
                        
                        # Add data for away team
                        availability_data.append({
                            'GAME_ID': game_id,
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
                    
                    time.sleep(0.5)  # Reduced sleep time to process more games
                    
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
        
        Args:
            season: NBA season in format 'YYYY-YY'
            
        Returns:
            pd.DataFrame: Teams data
        """
        try:
            from nba_api.stats.endpoints import commonteamyears, teaminfocommon
            
            # Get teams for the specified season
            teams = commonteamyears.CommonTeamYears().get_data_frames()[0]
            season_teams = teams[teams['SEASON_ID'] == f"2{season.split('-')[0]}"]
            
            team_details = []
            
            # Get detailed info for each team
            for team_id in season_teams['TEAM_ID'].unique():
                try:
                    team_info = teaminfocommon.TeamInfoCommon(
                        team_id=team_id,
                        season=season
                    ).get_data_frames()[0]
                    
                    if not team_info.empty:
                        team_details.append(team_info)
                        
                    time.sleep(1)  # Respect API rate limits
                    
                except Exception as e:
                    print(f"Error fetching info for team {team_id}: {e}")
                    continue
            
            if team_details:
                return pd.concat(team_details, ignore_index=True)
            else:
                return pd.DataFrame(columns=['TEAM_ID', 'TEAM_NAME'])
                
        except Exception as e:
            print(f"Error fetching teams: {e}")
            return pd.DataFrame(columns=['TEAM_ID', 'TEAM_NAME'])