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
            # ... (keep the existing team locations dictionary)
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
