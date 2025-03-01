import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import List
from models.deep_models import DeepNBAPredictor
from data.data_acquisition import DataAcquirer
from data.feature_engineering import FeatureEngineer

class EnhancedNBAPredictor:
    def __init__(self, seasons: List[str]):
        self.seasons = seasons
        self.data_acquirer = DataAcquirer(seasons)
        self.feature_engineer = FeatureEngineer()
        self.models = []
        self.scalers = []

    def run_pipeline(self):
        """Run complete prediction pipeline"""
        games, metrics = self.data_acquirer.fetch_games()
        features = self.feature_engineer.calculate_team_stats(games)
        self.train(features)

    def train(self, X: pd.DataFrame) -> None:
        """Train model ensemble"""
        # ... (keep existing train method)

    def calculate_confidence_score(self, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """Calculate confidence scores for predictions."""
        # ... (keep existing calculate_confidence_score method)

    def train_deep_model(self, X: pd.DataFrame) -> tuple:
        """Train deep neural network model"""
        # ... (keep existing train_deep_model method)
