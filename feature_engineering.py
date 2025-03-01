import pandas as pd
import numpy as np
from typing import Dict, List
from geopy.distance import geodesic
import pytz

TEAM_LOCATIONS = {
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

TIMEZONE_MAP = {tz: pytz.timezone(tz) for team in TEAM_LOCATIONS.values() for tz in [team['timezone']]}

def calculate_advanced_stats(row: pd.Series) -> Dict:
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

def calculate_travel_impact(row: pd.Series) -> Dict:
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

        if home_team not in TEAM_LOCATIONS or away_team not in TEAM_LOCATIONS:
            return {'distance': 0, 'timezone_diff': 0}

        # Calculate distance
        home_coords = TEAM_LOCATIONS[home_team]['coords']
        away_coords = TEAM_LOCATIONS[away_team]['coords']
        distance = geodesic(home_coords, away_coords).miles

        # Calculate timezone difference
        home_tz = TIMEZONE_MAP[TEAM_LOCATIONS[home_team]['timezone']]
        away_tz = TIMEZONE_MAP[TEAM_LOCATIONS[away_team]['timezone']]

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