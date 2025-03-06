"""
Constants used throughout the NBA prediction model.
"""
from typing import Dict, List

# Default lookback windows for feature engineering
DEFAULT_LOOKBACK_WINDOWS = [7, 14, 30, 60]

# NBA team locations with coordinates and timezone information
TEAM_LOCATIONS: Dict = {
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

# Team ID to abbreviation mapping
TEAM_ID_TO_ABBREV = {
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

# Basic statistics columns to extract from NBA API
BASIC_STATS_COLUMNS: List[str] = [
    'GAME_ID', 'GAME_DATE', 'TEAM_ID', 'TEAM_NAME', 'WL',
    'PTS', 'FG_PCT', 'FT_PCT', 'FG3_PCT', 'AST', 'REB', 'TOV',
    'PLUS_MINUS', 'FG3A', 'FG3M', 'FGA', 'FGM', 'FTA', 'FTM',
    'OREB', 'DREB', 'STL', 'BLK', 'PF'
]

# Feature groups for organizing and reporting
FEATURE_GROUPS = {
    'WIN_PCT': 'Win percentage and momentum',
    'RTG': 'Rating metrics (OFF/DEF/NET)',
    'EFF': 'Efficiency metrics',
    'MOMENTUM': 'Momentum and consistency',
    'REST': 'Rest and fatigue',
    'H2H': 'Head-to-head matchups'
}

# Confidence score weights
CONFIDENCE_WEIGHTS = {
    'prediction_margin': 0.3,  # Weight for prediction probability margin
    'sample_size': 0.2,        # Weight for number of previous matches
    'recent_consistency': 0.2,  # Weight for consistency in recent games
    'h2h_history': 0.15,       # Weight for head-to-head history
    'rest_advantage': 0.15     # Weight for rest day advantage
}