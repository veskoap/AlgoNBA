"""
Helper functions used throughout the NBA prediction model.
"""
import pandas as pd
import numpy as np
import pytz
import warnings
from typing import Dict, Tuple
from datetime import datetime
from geopy.distance import geodesic

from src.utils.constants import TEAM_LOCATIONS, TEAM_ID_TO_ABBREV


def suppress_sklearn_warnings():
    """
    Suppress common sklearn deprecation warnings.
    Call this function at the beginning of scripts to avoid warning spam.
    """
    # Suppress force_all_finite deprecation warnings
    warnings.filterwarnings('ignore', message='.*force_all_finite.*', 
                          category=FutureWarning, module='sklearn.*')
    
    # Suppress other common sklearn warnings
    warnings.filterwarnings('ignore', message='.*valid_leaf_size.*', 
                          category=FutureWarning, module='sklearn.*')
    warnings.filterwarnings('ignore', message='.*n_features_in_.*', 
                          category=FutureWarning, module='sklearn.*')
    
    # Add more patterns as needed


def safe_divide(a, b, fill_value=0, index=None):
    """
    Safely divide two arrays or pandas Series, handling division by zero.
    
    Args:
        a: Numerator (array or Series)
        b: Denominator (array or Series)
        fill_value: Value to use when division is undefined
        index: Index for the output Series (if input is not already a Series)
        
    Returns:
        pandas.Series: Result of division with undefined values replaced by fill_value
    """
    # Convert inputs to pandas Series if they aren't already
    if not isinstance(a, pd.Series):
        a = pd.Series(a, index=index)
    if not isinstance(b, pd.Series):
        b = pd.Series(b, index=index)

    # Fill NA values
    a = a.fillna(0)
    b = b.fillna(1)  # Use 1 for denominator to avoid division by zero

    # Convert to numpy arrays
    a_array = np.asarray(a.astype(float))
    b_array = np.asarray(b.astype(float))

    # Perform division
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a_array, b_array)
        mask = ~np.isfinite(result)
        result[mask] = fill_value

    return pd.Series(result, index=a.index)


def get_team_abbrev(team_id):
    """
    Get team abbreviation from team ID.
    
    Args:
        team_id: NBA API team ID
        
    Returns:
        str: Team abbreviation or 'UNK' if not found
    """
    return TEAM_ID_TO_ABBREV.get(team_id, 'UNK')


def calculate_travel_impact(row: pd.Series) -> Dict:
    """
    Calculate travel distance and timezone changes between games.
    
    Args:
        row: DataFrame row containing team IDs and game date
        
    Returns:
        dict: Dictionary with distance and timezone_diff values
    """
    try:
        # Extract team abbreviations
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
        timezone_map = {
            tz: pytz.timezone(tz) 
            for team in TEAM_LOCATIONS.values() 
            for tz in [team['timezone']]
        }
        
        home_tz = timezone_map[TEAM_LOCATIONS[home_team]['timezone']]
        away_tz = timezone_map[TEAM_LOCATIONS[away_team]['timezone']]

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


def calculate_advanced_stats(row: pd.Series) -> Dict:
    """
    Calculate advanced basketball statistics for a single game.
    
    Args:
        row: DataFrame row containing basic game stats
        
    Returns:
        dict: Dictionary with advanced statistics
    """
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