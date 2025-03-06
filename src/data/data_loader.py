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
        try:
            from nba_api.stats.endpoints import teamplayerdashboard, boxscoreadvancedv2
            
            print(f"Fetching player availability data for {season}...")
            
            # Dictionary to store player impact scores by team
            player_impact = {}
            team_lineup_strength = {}
            
            # Get teams for the season
            teams = self.fetch_teams(season)
            
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
                        team_players['IMPACT_SCORE'] = (
                            0.4 * team_players['PLUS_MINUS'] +
                            0.3 * team_players['PIE'] * 100 +  # Player Impact Estimate
                            0.2 * team_players['USG_PCT'] +     # Usage percentage
                            0.1 * team_players['MIN']           # Minutes played
                        )
                        
                        # Store top players and their impact scores
                        player_impact[team_id] = team_players[['PLAYER_ID', 'PLAYER_NAME', 'IMPACT_SCORE']]
                        
                        # Calculate team lineup strength based on top 8 players
                        top_players = team_players.nlargest(8, 'IMPACT_SCORE')
                        team_lineup_strength[team_id] = top_players['IMPACT_SCORE'].sum()
                    
                    time.sleep(1)  # Respect API rate limits
                    
                except Exception as e:
                    print(f"Error fetching player data for team {team_id}: {e}")
                    continue
            
            # For demonstration - create player availability dataframe with placeholder data
            # In a production system, this would use real injury data from ESPN, NBA.com, or other sources
            availability_data = []
            
            # Get all games for the season
            all_games = leaguegamefinder.LeagueGameFinder(
                season_nullable=season,
                season_type_nullable='Regular Season'
            ).get_data_frames()[0]
            
            # Process all games for enhanced availability data
            for game_id in all_games['GAME_ID'].unique():
                try:
                    # Get the teams from this game
                    game_teams = all_games[all_games['GAME_ID'] == game_id]['TEAM_ID'].unique()
                    
                    if len(game_teams) == 2:
                        home_team = game_teams[0]
                        away_team = game_teams[1]
                        
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
                        if home_team in team_lineup_strength:
                            home_strength = team_lineup_strength[home_team]
                            
                            # Adjust for missing players
                            if home_team in player_impact:
                                playing_ids = set(home_players['PLAYER_ID'])
                                missing_players = player_impact[home_team][
                                    ~player_impact[home_team]['PLAYER_ID'].isin(playing_ids)
                                ]
                                missing_impact = missing_players['IMPACT_SCORE'].sum()
                                home_strength -= missing_impact
                        
                        # Process away team
                        away_players = box_score[box_score['TEAM_ID'] == away_team]
                        away_starters = len(away_players[away_players['START_POSITION'] != ''])
                        away_available = len(away_players)
                        
                        # Calculate away lineup strength
                        away_strength = 0
                        if away_team in team_lineup_strength:
                            away_strength = team_lineup_strength[away_team]
                            
                            # Adjust for missing players
                            if away_team in player_impact:
                                playing_ids = set(away_players['PLAYER_ID'])
                                missing_players = player_impact[away_team][
                                    ~player_impact[away_team]['PLAYER_ID'].isin(playing_ids)
                                ]
                                missing_impact = missing_players['IMPACT_SCORE'].sum()
                                away_strength -= missing_impact
                                
                        # Calculate star player matchup advantage
                        # In a production system, this would use more detailed positional matchup data
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
                        
                        # Add data for both teams
                        availability_data.append({
                            'GAME_ID': game_id,
                            'TEAM_ID': home_team,
                            'IS_HOME': 1,
                            'PLAYERS_AVAILABLE': home_available,
                            'STARTERS_AVAILABLE': home_starters,
                            'LINEUP_IMPACT': home_strength,
                            'STAR_MATCHUP_ADVANTAGE': star_matchup_advantage
                        })
                        
                        availability_data.append({
                            'GAME_ID': game_id,
                            'TEAM_ID': away_team,
                            'IS_HOME': 0,
                            'PLAYERS_AVAILABLE': away_available,
                            'STARTERS_AVAILABLE': away_starters,
                            'LINEUP_IMPACT': away_strength,
                            'STAR_MATCHUP_ADVANTAGE': -star_matchup_advantage
                        })
                        
                    time.sleep(1)  # Respect API rate limits
                    
                except Exception as e:
                    print(f"Error processing availability for game {game_id}: {e}")
                    continue
            
            return pd.DataFrame(availability_data)
        
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
                'STAR_MATCHUP_ADVANTAGE': []
            })
            
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