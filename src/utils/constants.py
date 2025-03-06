"""
Constants used throughout the NBA prediction model.
"""
from typing import Dict, List

# Default lookback windows for feature engineering
DEFAULT_LOOKBACK_WINDOWS = [3, 5, 7, 10, 15, 20, 30, 60]

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
    'H2H': 'Head-to-head matchups',
    'MATCHUP': 'Team matchup specialization',
    'PLAYER': 'Player impact modeling',
    'CONTEXT': 'Contextual features',
    'TREND': 'Historical trend features'
}

# Create a new cache directory in the project
import os
try:
    cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cache')
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    pass

# Comprehensive feature registry for the NBA prediction system
FEATURE_REGISTRY = {
    # Win percentage features
    'WIN_PCT_HOME': {
        'type': 'base',
        'description': 'Home team win percentage',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
    },
    'WIN_PCT_AWAY': {
        'type': 'base',
        'description': 'Away team win percentage',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
    },
    'WIN_PCT_DIFF': {
        'type': 'derived',
        'description': 'Difference in win percentage (home - away)',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['WIN_PCT_HOME', 'WIN_PCT_AWAY'],
    },
    
    # Rating metrics
    'OFF_RTG_DIFF': {
        'type': 'derived',
        'description': 'Difference in offensive rating',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['OFF_RTG_mean_HOME', 'OFF_RTG_mean_AWAY'],
    },
    'DEF_RTG_DIFF': {
        'type': 'derived',
        'description': 'Difference in defensive rating',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['DEF_RTG_mean_HOME', 'DEF_RTG_mean_AWAY'],
    },
    'NET_RTG_DIFF': {
        'type': 'derived',
        'description': 'Difference in net rating',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['NET_RTG_mean_HOME', 'NET_RTG_mean_AWAY'],
    },
    'PACE_DIFF': {
        'type': 'derived',
        'description': 'Difference in pace',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PACE_mean_HOME', 'PACE_mean_AWAY'],
    },
    
    # Efficiency metrics
    'EFF_DIFF': {
        'type': 'derived',
        'description': 'Difference in points to turnover ratio',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PTS_mean_HOME', 'PTS_mean_AWAY', 'TOV_mean_HOME', 'TOV_mean_AWAY'],
    },
    'HOME_CONSISTENCY': {
        'type': 'derived',
        'description': 'Home team scoring consistency (std/mean)',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PTS_std_HOME', 'PTS_mean_HOME'],
    },
    'AWAY_CONSISTENCY': {
        'type': 'derived',
        'description': 'Away team scoring consistency (std/mean)',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PTS_std_AWAY', 'PTS_mean_AWAY'],
    },
    
    # Fatigue and rest features
    'FATIGUE_DIFF': {
        'type': 'derived',
        'description': 'Difference in team fatigue levels',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['FATIGUE_HOME', 'FATIGUE_AWAY'],
    },
    'REST_DIFF': {
        'type': 'derived',
        'description': 'Difference in rest days',
        'windows': None,
        'dependencies': ['REST_DAYS_HOME', 'REST_DAYS_AWAY'],
    },
    
    # Momentum features
    'WIN_MOMENTUM': {
        'type': 'derived',
        'description': 'Ratio of recent to longer-term win percentage',
        'windows': [5, 10, 15, 20, 30, 60],  # Excludes smallest window
        'dependencies': ['WIN_PCT_HOME', 'WIN_PCT_AWAY'],
    },
    'SCORING_MOMENTUM': {
        'type': 'derived',
        'description': 'Ratio of recent to longer-term scoring',
        'windows': [5, 10, 15, 20, 30, 60],  # Excludes smallest window
        'dependencies': ['PTS_mean_HOME', 'PTS_mean_AWAY'],
    },
    
    # Head-to-head features
    'H2H_WIN_PCT': {
        'type': 'h2h',
        'description': 'Win percentage in head-to-head matchups',
        'windows': None,
        'dependencies': [],
    },
    'H2H_GAMES': {
        'type': 'h2h',
        'description': 'Number of head-to-head games',
        'windows': None,
        'dependencies': [],
    },
    'DAYS_SINCE_H2H': {
        'type': 'h2h',
        'description': 'Days since last head-to-head matchup',
        'windows': None,
        'dependencies': [],
    },
    'LAST_GAME_HOME_ADVANTAGE': {
        'type': 'h2h',
        'description': 'Whether the last h2h game was at home',
        'windows': None,
        'dependencies': ['LAST_GAME_HOME'],
    },
    'H2H_RECENCY_WEIGHT': {
        'type': 'derived',
        'description': 'Head-to-head win percentage weighted by recency',
        'windows': None,
        'dependencies': ['H2H_WIN_PCT', 'DAYS_SINCE_H2H'],
    },
    
    # Enhanced head-to-head features
    'H2H_AVG_MARGIN': {
        'type': 'h2h',
        'description': 'Average margin in head-to-head games',
        'windows': None,
        'dependencies': [],
    },
    'H2H_STREAK': {
        'type': 'h2h',
        'description': 'Current streak in head-to-head games',
        'windows': None,
        'dependencies': [],
    },
    'H2H_HOME_ADVANTAGE': {
        'type': 'h2h',
        'description': 'Home advantage in head-to-head games',
        'windows': None,
        'dependencies': [],
    },
    'H2H_MOMENTUM': {
        'type': 'h2h',
        'description': 'Momentum in head-to-head matchups',
        'windows': None,
        'dependencies': [],
    },
    
    # Composite features
    'RECENT_VS_LONG_TERM_HOME': {
        'type': 'composite',
        'description': 'Ratio of recent to long-term home performance',
        'windows': None,
        'dependencies': ['WIN_mean_HOME'],
    },
    'RECENT_VS_LONG_TERM_AWAY': {
        'type': 'composite',
        'description': 'Ratio of recent to long-term away performance',
        'windows': None,
        'dependencies': ['WIN_mean_AWAY'],
    },
    'HOME_AWAY_CONSISTENCY_30D': {
        'type': 'composite',
        'description': 'Consistency between home and away performance',
        'windows': None,
        'dependencies': ['WIN_mean_HOME_30D', 'WIN_mean_AWAY_30D'],
    },
    
    # Travel impact features
    'TRAVEL_DISTANCE': {
        'type': 'travel',
        'description': 'Distance traveled',
        'windows': None,
        'dependencies': [],
    },
    'TIMEZONE_DIFF': {
        'type': 'travel',
        'description': 'Timezone difference',
        'windows': None,
        'dependencies': [],
    },
    
    # Interaction features
    'FATIGUE_TRAVEL_INTERACTION': {
        'type': 'interaction',
        'description': 'Interaction between fatigue and travel distance',
        'windows': None,
        'dependencies': ['FATIGUE_DIFF_14D', 'TRAVEL_DISTANCE'],
    },
    'MOMENTUM_REST_INTERACTION': {
        'type': 'interaction',
        'description': 'Interaction between momentum and rest advantage',
        'windows': None,
        'dependencies': ['WIN_PCT_DIFF_30D', 'REST_DIFF'],
    },
    
    # Historical trend features with exponential decay
    'TREND_WIN_HOME': {
        'type': 'trend',
        'description': 'Home team win trend with exponential decay weights',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['WIN_mean_HOME'],
    },
    'TREND_WIN_AWAY': {
        'type': 'trend',
        'description': 'Away team win trend with exponential decay weights',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['WIN_mean_AWAY'],
    },
    'TREND_SCORE_HOME': {
        'type': 'trend',
        'description': 'Home team scoring trend with exponential decay weights',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PTS_mean_HOME'],
    },
    'TREND_SCORE_AWAY': {
        'type': 'trend',
        'description': 'Away team scoring trend with exponential decay weights',
        'windows': DEFAULT_LOOKBACK_WINDOWS,
        'dependencies': ['PTS_mean_AWAY'],
    },
    
    # Team matchup specialization features
    'MATCHUP_COMPATIBILITY': {
        'type': 'matchup',
        'description': 'Style-based compatibility between teams',
        'windows': None,
        'dependencies': ['PACE_mean_HOME_30D', 'PACE_mean_AWAY_30D', 'FG3A_mean_HOME_30D', 'FG3A_mean_AWAY_30D'],
    },
    'MATCHUP_HISTORY_SCORE': {
        'type': 'matchup',
        'description': 'Team-specific historical matchup score',
        'windows': None,
        'dependencies': ['H2H_WIN_PCT', 'H2H_AVG_MARGIN', 'H2H_MOMENTUM'],
    },
    'STYLE_ADVANTAGE': {
        'type': 'matchup',
        'description': 'Playing style advantage metric',
        'windows': None,
        'dependencies': ['PACE_mean_HOME_30D', 'PACE_mean_AWAY_30D', 'FG3_PCT_mean_HOME_30D', 'FG3_PCT_mean_AWAY_30D'],
    },
    
    # Player impact modeling features
    'LINEUP_IMPACT_HOME': {
        'type': 'player',
        'description': 'Home team lineup impact score',
        'windows': None,
        'dependencies': [],
    },
    'LINEUP_IMPACT_AWAY': {
        'type': 'player',
        'description': 'Away team lineup impact score',
        'windows': None,
        'dependencies': [],
    },
    'STAR_PLAYER_MATCHUP': {
        'type': 'player',
        'description': 'Star player matchup advantage',
        'windows': None,
        'dependencies': [],
    },
    'LINEUP_IMPACT_DIFF': {
        'type': 'player',
        'description': 'Difference in lineup impact scores',
        'windows': None,
        'dependencies': ['LINEUP_IMPACT_HOME', 'LINEUP_IMPACT_AWAY'],
    },
    'GUARD_STRENGTH_HOME': {
        'type': 'player',
        'description': 'Home team guard strength',
        'windows': None,
        'dependencies': [],
    },
    'GUARD_STRENGTH_AWAY': {
        'type': 'player',
        'description': 'Away team guard strength',
        'windows': None,
        'dependencies': [],
    },
    'FORWARD_STRENGTH_HOME': {
        'type': 'player',
        'description': 'Home team forward strength',
        'windows': None,
        'dependencies': [],
    },
    'FORWARD_STRENGTH_AWAY': {
        'type': 'player',
        'description': 'Away team forward strength',
        'windows': None,
        'dependencies': [],
    },
    'CENTER_STRENGTH_HOME': {
        'type': 'player',
        'description': 'Home team center strength',
        'windows': None,
        'dependencies': [],
    },
    'CENTER_STRENGTH_AWAY': {
        'type': 'player',
        'description': 'Away team center strength',
        'windows': None,
        'dependencies': [],
    },
    'GUARD_ADVANTAGE': {
        'type': 'player',
        'description': 'Home team guard advantage',
        'windows': None,
        'dependencies': ['GUARD_STRENGTH_HOME', 'GUARD_STRENGTH_AWAY'],
    },
    'FORWARD_ADVANTAGE': {
        'type': 'player',
        'description': 'Home team forward advantage',
        'windows': None,
        'dependencies': ['FORWARD_STRENGTH_HOME', 'FORWARD_STRENGTH_AWAY'],
    },
    'CENTER_ADVANTAGE': {
        'type': 'player',
        'description': 'Home team center advantage',
        'windows': None,
        'dependencies': ['CENTER_STRENGTH_HOME', 'CENTER_STRENGTH_AWAY'],
    },
    
    # Advanced contextual features
    'STADIUM_HOME_ADVANTAGE': {
        'type': 'context',
        'description': 'Stadium-specific home advantage factor',
        'windows': None,
        'dependencies': [],
    },
    'TRAVEL_FATIGUE': {
        'type': 'context',
        'description': 'Travel fatigue with time zone adjustments',
        'windows': None,
        'dependencies': ['TRAVEL_DISTANCE', 'TIMEZONE_DIFF'],
    },
    'WEEKEND_GAME': {
        'type': 'context',
        'description': 'Flag for weekend vs. weekday games',
        'windows': None,
        'dependencies': [],
    },
    'NATIONAL_TV': {
        'type': 'context',
        'description': 'Flag for nationally televised games',
        'windows': None,
        'dependencies': [],
    },
    'RIVALRY_MATCHUP': {
        'type': 'context',
        'description': 'Flag for rivalry matchups',
        'windows': None,
        'dependencies': [],
    }
}

# Confidence score weights
CONFIDENCE_WEIGHTS = {
    'prediction_margin': 0.3,  # Weight for prediction probability margin
    'sample_size': 0.2,        # Weight for number of previous matches
    'recent_consistency': 0.2,  # Weight for consistency in recent games
    'h2h_history': 0.15,       # Weight for head-to-head history
    'rest_advantage': 0.15     # Weight for rest day advantage
}