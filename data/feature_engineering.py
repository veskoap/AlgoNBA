import pandas as pd
import numpy as np
from geopy.distance import geodesic
import pytz
from typing import Dict

class FeatureEngineer:
    def __init__(self, lookback_windows=[7, 14, 30, 60]):
        self.lookback_windows = lookback_windows
        self.team_locations = self._initialize_team_locations()
        self.timezone_map = self._initialize_timezone_map()

    def _initialize_team_locations(self) -> Dict:
        """Initialize NBA team locations with coordinates."""
        return {
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

    def _initialize_timezone_map(self) -> Dict:
        """Initialize timezone mapping for calculations."""
        return {tz: pytz.timezone(tz) for team in self.team_locations.values()
                for tz in [team['timezone']]}

    def calculate_travel_impact(self, row: pd.Series) -> Dict:
        """Calculate travel distance and timezone changes between games."""
        # ... (keep existing calculate_travel_impact method)

    def calculate_advanced_stats(self, row: pd.Series) -> Dict:
        """Calculate advanced basketball statistics for a single game."""
        # ... (keep existing calculate_advanced_stats method)

    def calculate_h2h_features(self, games: pd.DataFrame) -> pd.DataFrame:
        """Calculate head-to-head statistics between teams."""
        # ... (keep existing calculate_h2h_features method)
