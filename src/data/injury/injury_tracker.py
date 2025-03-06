"""
NBA player injury tracking and impact calculation.
"""
import os
import sqlite3
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import json
import concurrent.futures
from typing import Dict, List, Tuple, Any, Optional

from src.utils.constants import TEAM_ID_TO_ABBREV


class PlayerInjuryTracker:
    """Class for tracking and analyzing NBA player injuries."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the injury tracker with caching support.
        
        Args:
            cache_dir: Directory for caching injury data (default to project cache dir)
        """
        # Set up cache directory
        if cache_dir is None:
            self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
                            os.path.abspath(__file__)))), 'cache')
        else:
            self.cache_dir = cache_dir
            
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Set up database paths
        self.db_path = os.path.join(self.cache_dir, 'injury_data.db')
        self.current_injuries_path = os.path.join(self.cache_dir, 'current_injuries.pkl')
        
        # Initialize database
        self._setup_database()
        
        # Injury severity mapping for impact calculations
        self.severity_map = {
            'day-to-day': 'minor',
            'questionable': 'minor',
            'probable': 'minor',
            'doubtful': 'medium',
            'out': 'medium',
            'out (personal)': 'medium',
            'out (rest)': 'minor',
            'out for season': 'major',
            'acl': 'major',
            'broken': 'major',
            'fractured': 'major',
            'surgery': 'major',
            'injured reserve': 'major',
            None: 'medium'
        }
        
    def _setup_database(self) -> None:
        """
        Set up the SQLite database for storing injury data.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create injuries table if it doesn't exist
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS injuries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            player_id INTEGER NOT NULL,
            player_name TEXT NOT NULL,
            team_id INTEGER NOT NULL,
            injury_date TEXT NOT NULL,
            return_date TEXT,
            injury_type TEXT,
            status TEXT,
            body_part TEXT,
            severity TEXT,
            description TEXT,
            source TEXT,
            update_date TEXT NOT NULL
        )
        ''')
        
        # Create index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_player_date ON injuries (player_id, injury_date)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_team_date ON injuries (team_id, injury_date)')
        
        conn.commit()
        conn.close()
        
    def fetch_current_injuries(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Fetch current NBA injuries with caching to minimize API requests.
        
        Args:
            force_refresh: Force refresh data even if recently cached
            
        Returns:
            pd.DataFrame: Current NBA injuries data
        """
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Check if we have recent cached data (within last 12 hours)
        if not force_refresh and os.path.exists(self.current_injuries_path):
            try:
                cache_data = pd.read_pickle(self.current_injuries_path)
                cache_date = cache_data['update_date'].iloc[0] if not cache_data.empty else None
                
                # If cache is from today, use it
                if cache_date and cache_date == today:
                    print(f"Using cached injury data from {cache_date}")
                    return cache_data
            except Exception as e:
                print(f"Error reading injury cache: {e}")
        
        print("Fetching fresh injury data from multiple sources...")
        
        # Try multiple sources with fallbacks
        all_injuries = pd.DataFrame()
        
        # Try primary source first (basketball-reference in this case)
        try:
            injuries = self._fetch_from_basketball_reference()
            if not injuries.empty:
                all_injuries = pd.concat([all_injuries, injuries], ignore_index=True)
        except Exception as e:
            print(f"Error fetching from basketball-reference: {e}")
        
        # Try ESPN as backup if needed
        if all_injuries.empty:
            try:
                injuries = self._fetch_from_espn()
                if not injuries.empty:
                    all_injuries = pd.concat([all_injuries, injuries], ignore_index=True)
            except Exception as e:
                print(f"Error fetching from ESPN: {e}")
        
        # Try CBS Sports as final backup
        if all_injuries.empty:
            try:
                injuries = self._fetch_from_cbssports()
                if not injuries.empty:
                    all_injuries = pd.concat([all_injuries, injuries], ignore_index=True)
            except Exception as e:
                print(f"Error fetching from CBS Sports: {e}")
        
        # If we have injury data, save it to cache and database
        if not all_injuries.empty:
            all_injuries['update_date'] = today
            
            # Add severity based on status/description
            all_injuries['severity'] = all_injuries['status'].str.lower().map(self.severity_map)
            all_injuries.loc[all_injuries['severity'].isna(), 'severity'] = 'medium'
            
            # Save to cache
            all_injuries.to_pickle(self.current_injuries_path)
            
            # Update database
            self._update_injury_database(all_injuries)
            
            print(f"Successfully fetched {len(all_injuries)} current injuries")
            return all_injuries
        
        # If all sources failed, try to use older cache as fallback
        if os.path.exists(self.current_injuries_path):
            try:
                cache_data = pd.read_pickle(self.current_injuries_path)
                print(f"Using older cached injury data from {cache_data['update_date'].iloc[0]}")
                return cache_data
            except:
                pass
        
        # Return empty DataFrame if everything fails
        print("No injury data available")
        return pd.DataFrame({
            'player_id': [],
            'player_name': [],
            'team_id': [],
            'injury_date': [],
            'return_date': [],
            'injury_type': [],
            'status': [],
            'body_part': [],
            'severity': [],
            'description': [],
            'source': [],
            'update_date': []
        })
    
    def _fetch_from_basketball_reference(self) -> pd.DataFrame:
        """
        Fetch injury data from Basketball Reference.
        
        Returns:
            pd.DataFrame: Basketball Reference injury data
        """
        # This would be implemented with requests and BeautifulSoup
        # For demonstration, return dummy data
        try:
            url = "https://www.basketball-reference.com/friv/injuries.html"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            
            if response.status_code != 200:
                print(f"Failed to fetch data: Status code {response.status_code}")
                return pd.DataFrame()
                
            soup = BeautifulSoup(response.content, 'html.parser')
            injury_table = soup.find('table', id='injuries')
            
            if not injury_table:
                print("No injury table found")
                return pd.DataFrame()
                
            injuries = []
            today = datetime.now().strftime('%Y-%m-%d')
            
            # Parse each injury row
            for row in injury_table.find('tbody').find_all('tr'):
                try:
                    cols = row.find_all('td')
                    if len(cols) < 4:
                        continue
                        
                    player_link = row.find('th').find('a')
                    if not player_link:
                        continue
                        
                    # Extract player ID from URL
                    player_url = player_link['href']
                    player_id = int(player_url.split('/')[-1].split('.')[0][1:]) if player_url else None
                    
                    player_name = player_link.text
                    team_abbrev = cols[0].text.strip()
                    
                    # Map team abbreviation to team ID
                    team_id = None
                    for t_id, abbrev in TEAM_ID_TO_ABBREV.items():
                        if abbrev == team_abbrev:
                            team_id = t_id
                            break
                            
                    if not team_id:
                        continue
                        
                    update = cols[1].text.strip()
                    description = cols[2].text.strip()
                    
                    # Parse injury details from description
                    injury_type = None
                    body_part = None
                    status = cols[3].text.strip() if len(cols) > 3 else None
                    
                    injury_words = ['sprain', 'strain', 'fracture', 'tear', 'contusion', 'injury', 
                                  'surgery', 'concussion', 'illness', 'infection']
                    body_parts = ['ankle', 'knee', 'shoulder', 'hip', 'elbow', 'wrist', 'hand', 
                                'foot', 'hamstring', 'quad', 'back', 'heel', 'achilles', 'calf',
                                'groin', 'thigh', 'arm', 'leg', 'finger', 'toe', 'head']
                    
                    for word in injury_words:
                        if word in description.lower():
                            injury_type = word
                            break
                            
                    for part in body_parts:
                        if part in description.lower():
                            body_part = part
                            break
                            
                    injuries.append({
                        'player_id': player_id,
                        'player_name': player_name,
                        'team_id': team_id,
                        'injury_date': today,  # We don't know exact date, use today
                        'return_date': None,  # Return date usually not provided
                        'injury_type': injury_type,
                        'status': status,
                        'body_part': body_part,
                        'description': description,
                        'source': 'basketball-reference',
                        'update_date': today
                    })
                except Exception as e:
                    print(f"Error parsing injury row: {e}")
                    continue
                    
            return pd.DataFrame(injuries)
            
        except Exception as e:
            print(f"Error in basketball-reference scraping: {e}")
            return pd.DataFrame()
    
    def _fetch_from_espn(self) -> pd.DataFrame:
        """
        Fetch injury data from ESPN.
        
        Returns:
            pd.DataFrame: ESPN injury data
        """
        # This would be implemented with ESPN API or scraping
        # For now, return empty DataFrame
        print("ESPN injury source not yet implemented")
        return pd.DataFrame()
    
    def _fetch_from_cbssports(self) -> pd.DataFrame:
        """
        Fetch injury data from CBS Sports.
        
        Returns:
            pd.DataFrame: CBS Sports injury data
        """
        # This would be implemented with CBS Sports API or scraping
        # For now, return empty DataFrame
        print("CBS Sports injury source not yet implemented")
        return pd.DataFrame()
    
    def _update_injury_database(self, injuries: pd.DataFrame) -> None:
        """
        Update the injury database with new data.
        
        Args:
            injuries: DataFrame of current injuries
        """
        if injuries.empty:
            return
            
        conn = sqlite3.connect(self.db_path)
        
        try:
            # Insert new injuries
            for _, injury in injuries.iterrows():
                # Check if this injury already exists
                cursor = conn.cursor()
                cursor.execute(
                    "SELECT id FROM injuries WHERE player_id = ? AND injury_date = ?",
                    (injury['player_id'], injury['injury_date'])
                )
                existing = cursor.fetchone()
                
                if existing:
                    # Update existing injury
                    cursor.execute("""
                        UPDATE injuries
                        SET return_date = ?, injury_type = ?, status = ?, 
                            body_part = ?, severity = ?, description = ?,
                            source = ?, update_date = ?
                        WHERE id = ?
                    """, (
                        injury.get('return_date'),
                        injury.get('injury_type'),
                        injury.get('status'),
                        injury.get('body_part'),
                        injury.get('severity'),
                        injury.get('description'),
                        injury.get('source'),
                        injury.get('update_date'),
                        existing[0]
                    ))
                else:
                    # Insert new injury
                    cursor.execute("""
                        INSERT INTO injuries 
                        (player_id, player_name, team_id, injury_date, return_date, 
                         injury_type, status, body_part, severity, description, 
                         source, update_date)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        injury.get('player_id'),
                        injury.get('player_name'),
                        injury.get('team_id'),
                        injury.get('injury_date'),
                        injury.get('return_date'),
                        injury.get('injury_type'),
                        injury.get('status'),
                        injury.get('body_part'),
                        injury.get('severity'),
                        injury.get('description'),
                        injury.get('source'),
                        injury.get('update_date')
                    ))
            
            conn.commit()
            print(f"Updated injury database with {len(injuries)} records")
            
        except Exception as e:
            conn.rollback()
            print(f"Error updating injury database: {e}")
            
        finally:
            conn.close()
    
    def get_team_injuries(self, team_id: int, game_date: str) -> pd.DataFrame:
        """
        Get all injuries for a team on a specific date.
        
        Args:
            team_id: Team ID
            game_date: Game date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: Team injury data
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"""
        SELECT * FROM injuries 
        WHERE team_id = {team_id} 
        AND injury_date <= '{game_date}' 
        AND (return_date IS NULL OR return_date > '{game_date}')
        """
        
        try:
            injuries = pd.read_sql(query, conn)
            conn.close()
            return injuries
        except Exception as e:
            print(f"Error querying team injuries: {e}")
            conn.close()
            return pd.DataFrame()
    
    def calculate_injury_impact(self, team_id: int, game_date: str, 
                               player_impact_scores: Dict[int, Dict]) -> Dict:
        """
        Calculate impact of injuries on team performance.
        
        Args:
            team_id: Team ID
            game_date: Game date in YYYY-MM-DD format
            player_impact_scores: Dictionary of player impact scores by player ID
            
        Returns:
            dict: Injury impact metrics
        """
        # Get team injuries for the date
        injuries = self.get_team_injuries(team_id, game_date)
        
        if injuries.empty:
            return {
                'total_impact': 0.0,
                'guard_impact': 0.0,
                'forward_impact': 0.0,
                'center_impact': 0.0,
                'missing_players': 0,
                'missing_starters': 0,
                'description': "No injuries"
            }
        
        # Calculate impact
        total_impact = 0.0
        guard_impact = 0.0
        forward_impact = 0.0
        center_impact = 0.0
        missing_starters = 0
        player_details = []
        
        for _, injury in injuries.iterrows():
            player_id = injury['player_id']
            player_name = injury['player_name']
            severity = injury['severity']
            
            # Get player impact if available
            impact = 0.0
            position = "Unknown"
            is_starter = False
            
            if player_impact_scores and player_id in player_impact_scores:
                player_data = player_impact_scores[player_id]
                impact = player_data.get('impact_score', 0.0)
                position = player_data.get('position', 'Unknown')
                is_starter = player_data.get('starter', False)
                
                # Adjust impact based on injury severity
                severity_factor = {
                    'minor': 0.5,   # Player might return soon or play through it
                    'medium': 0.8,  # Typical injury impact
                    'major': 1.2    # Season-ending or severe injuries have outsized impact
                }
                
                impact_factor = severity_factor.get(severity, 0.8)
                adjusted_impact = impact * impact_factor
                
                total_impact += adjusted_impact
                
                # Track positional impact
                if position and position[0] == 'G':  # Guard
                    guard_impact += adjusted_impact
                elif position and position[0] == 'F':  # Forward
                    forward_impact += adjusted_impact
                elif position and position[0] == 'C':  # Center
                    center_impact += adjusted_impact
                    
                if is_starter:
                    missing_starters += 1
                    
                player_details.append({
                    'name': player_name,
                    'position': position,
                    'impact': adjusted_impact,
                    'status': injury['status']
                })
        
        # Create detailed description for visualization
        description = ", ".join([
            f"{p['name']} ({p['position']}): {p['status']}" 
            for p in sorted(player_details, key=lambda x: x['impact'], reverse=True)
        ])
        
        return {
            'total_impact': total_impact,
            'guard_impact': guard_impact,
            'forward_impact': forward_impact,
            'center_impact': center_impact,
            'missing_players': len(injuries),
            'missing_starters': missing_starters,
            'description': description
        }
    
    def get_injury_history(self, player_id: int, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Get injury history for a specific player.
        
        Args:
            player_id: Player ID
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            pd.DataFrame: Player injury history
        """
        conn = sqlite3.connect(self.db_path)
        
        query = f"SELECT * FROM injuries WHERE player_id = {player_id}"
        
        if start_date:
            query += f" AND injury_date >= '{start_date}'"
        
        if end_date:
            query += f" AND injury_date <= '{end_date}'"
            
        query += " ORDER BY injury_date DESC"
        
        try:
            history = pd.read_sql(query, conn)
            conn.close()
            return history
        except Exception as e:
            print(f"Error querying player injury history: {e}")
            conn.close()
            return pd.DataFrame()
    
    def estimate_recovery_time(self, injury_type: str, body_part: str) -> Tuple[int, int]:
        """
        Estimate recovery time range for given injury.
        
        Args:
            injury_type: Type of injury
            body_part: Affected body part
            
        Returns:
            tuple: (min_days, max_days) for recovery
        """
        # This would ideally use a more sophisticated model
        # For now, use basic lookup table
        recovery_times = {
            'sprain_ankle': (7, 21),
            'sprain_knee': (14, 42),
            'strain_hamstring': (14, 28),
            'fracture_foot': (42, 84),
            'concussion': (7, 21),
            'tear_acl': (270, 365),
            'contusion': (3, 10),
            'illness': (3, 10),
            'covid': (10, 21)
        }
        
        key = f"{injury_type}_{body_part}".lower()
        
        if key in recovery_times:
            return recovery_times[key]
        elif injury_type and injury_type.lower() in ['fracture', 'broken']:
            return (42, 84)  # Generic fracture timeline
        elif injury_type and injury_type.lower() in ['sprain']:
            return (7, 28)  # Generic sprain timeline
        elif injury_type and injury_type.lower() in ['strain']:
            return (7, 21)  # Generic strain timeline
        else:
            return (7, 21)  # Default
    
    def generate_injury_features(self, games_df: pd.DataFrame,
                               player_impact_scores: Dict[int, Dict]) -> pd.DataFrame:
        """
        Generate comprehensive injury features for games DataFrame.
        
        Args:
            games_df: Games DataFrame with dates and team IDs
            player_impact_scores: Dictionary of player impact scores
            
        Returns:
            pd.DataFrame: Injury features
        """
        print("Generating injury features for games...")
        
        # Create empty DataFrame to store results
        injury_features = pd.DataFrame(index=games_df.index)
        
        # Process each game
        for idx, row in games_df.iterrows():
            game_date = pd.to_datetime(row['GAME_DATE']).strftime('%Y-%m-%d')
            home_team = row['TEAM_ID_HOME']
            away_team = row['TEAM_ID_AWAY']
            
            # Get injury impact for home team
            home_impact = self.calculate_injury_impact(
                home_team, game_date, player_impact_scores
            )
            
            # Get injury impact for away team
            away_impact = self.calculate_injury_impact(
                away_team, game_date, player_impact_scores
            )
            
            # Add features
            injury_features.loc[idx, 'HOME_INJURY_IMPACT'] = home_impact['total_impact']
            injury_features.loc[idx, 'AWAY_INJURY_IMPACT'] = away_impact['total_impact']
            injury_features.loc[idx, 'INJURY_IMPACT_DIFF'] = home_impact['total_impact'] - away_impact['total_impact']
            
            injury_features.loc[idx, 'HOME_GUARD_INJURIES'] = home_impact['guard_impact']
            injury_features.loc[idx, 'AWAY_GUARD_INJURIES'] = away_impact['guard_impact']
            injury_features.loc[idx, 'GUARD_INJURY_DIFF'] = home_impact['guard_impact'] - away_impact['guard_impact']
            
            injury_features.loc[idx, 'HOME_FORWARD_INJURIES'] = home_impact['forward_impact']
            injury_features.loc[idx, 'AWAY_FORWARD_INJURIES'] = away_impact['forward_impact']
            injury_features.loc[idx, 'FORWARD_INJURY_DIFF'] = home_impact['forward_impact'] - away_impact['forward_impact']
            
            injury_features.loc[idx, 'HOME_CENTER_INJURIES'] = home_impact['center_impact']
            injury_features.loc[idx, 'AWAY_CENTER_INJURIES'] = away_impact['center_impact']
            injury_features.loc[idx, 'CENTER_INJURY_DIFF'] = home_impact['center_impact'] - away_impact['center_impact']
            
            injury_features.loc[idx, 'HOME_MISSING_PLAYERS'] = home_impact['missing_players']
            injury_features.loc[idx, 'AWAY_MISSING_PLAYERS'] = away_impact['missing_players']
            injury_features.loc[idx, 'MISSING_PLAYERS_DIFF'] = home_impact['missing_players'] - away_impact['missing_players']
            
            injury_features.loc[idx, 'HOME_MISSING_STARTERS'] = home_impact['missing_starters']
            injury_features.loc[idx, 'AWAY_MISSING_STARTERS'] = away_impact['missing_starters']
            injury_features.loc[idx, 'MISSING_STARTERS_DIFF'] = home_impact['missing_starters'] - away_impact['missing_starters']
            
        print(f"Generated {len(injury_features.columns)} injury features for {len(games_df)} games")
        return injury_features