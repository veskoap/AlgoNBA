"""
Ensemble model for NBA prediction.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import Dict, List, Tuple


class NBAEnsembleModel:
    """Ensemble model for NBA game prediction."""
    
    def __init__(self):
        """Initialize the ensemble model."""
        self.models = []
        self.scalers = []
        self.feature_selectors = []
        self.feature_importances = {}
        self.selected_features = {}
        self.feature_importance_summary = {}
        
    def train(self, X: pd.DataFrame) -> None:
        """
        Train enhanced ensemble of models with improved stability.
        
        Args:
            X: DataFrame containing features and target variable
        """
        print("Training model ensemble...")

        # Extract target variable
        y = X['TARGET']
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')

        print(f"Training with {len(X)} samples and {len(X.columns)} features")

        tscv = TimeSeriesSplit(n_splits=5)

        # Initialize tracking
        self.models = []
        self.scalers = []
        self.feature_selectors = []
        fold_metrics = []
        feature_importance_dict = defaultdict(list)

        # First pass: identify consistently important features
        print("Performing initial feature stability analysis...")
        feature_stability = defaultdict(int)
        feature_selector_list = []

        for fold, (train_idx, _) in enumerate(tscv.split(X), 1):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]

            # Initial feature selection
            selector = SelectFromModel(
                xgb.XGBClassifier(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42 + fold
                ),
                threshold='mean'
            )

            selector.fit(X_train, y_train)
            selected_features = X.columns[selector.get_support()].tolist()
            feature_selector_list.append(selected_features)

            for feat in selected_features:
                feature_stability[feat] += 1

        # Identify stable features (selected in majority of folds)
        stable_features = [feat for feat, count in feature_stability.items() if count >= 3]
        print(f"\nIdentified {len(stable_features)} stable features")

        # Main training loop with enhanced monitoring
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nTraining fold {fold}...")

            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            # Use stable features
            feature_mask = X.columns.isin(stable_features)
            X_train_selected = X_train_scaled[:, feature_mask]
            X_val_selected = X_val_scaled[:, feature_mask]

            # Train window-specific models
            window_models = []

            # Extract time windows from features (assuming format like 'FEATURE_NAME_7D')
            windows = set()
            for feat in stable_features:
                if '_D' in feat:
                    parts = feat.split('_')
                    for part in parts:
                        if part.endswith('D') and part[:-1].isdigit():
                            windows.add(int(part[:-1]))
                            
            if not windows:
                windows = [7, 14, 30, 60]  # Default if no window-specific features found
                
            windows = sorted(list(windows))

            for window in windows:
                # Get window-specific features
                window_features = [feat for feat in stable_features if f'_{window}D' in feat]
                base_features = [feat for feat in stable_features if '_D' not in feat]
                combined_features = window_features + base_features

                if not combined_features:
                    continue

                feature_indices = [stable_features.index(feat) for feat in combined_features]

                # Window-specific model with early stopping
                model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=5,
                    learning_rate=0.005,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    min_child_weight=4,
                    gamma=0.5,
                    reg_alpha=0.3,
                    reg_lambda=1.5,
                    scale_pos_weight=1,
                    random_state=42 + window,
                    eval_metric=['logloss', 'auc']
                )

                X_train_window = X_train_selected[:, feature_indices]
                X_val_window = X_val_selected[:, feature_indices]

                # Train with early stopping
                model.fit(
                    X_train_window, y_train,
                    eval_set=[(X_val_window, y_val)],
                    verbose=0
                )

                window_models.append((f'{window}d', model, combined_features))

                # Store feature importance
                importances = model.feature_importances_
                for feat, imp in zip(combined_features, importances):
                    feature_importance_dict[feat].append(imp)

            # Store models and scalers
            self.models.append((window_models, scaler, stable_features))

            # Evaluate performance
            y_preds = []
            for _, model, feats in window_models:
                feature_indices = [stable_features.index(f) for f in feats]
                y_pred = model.predict_proba(X_val_selected[:, feature_indices])[:, 1]
                y_preds.append(y_pred)

            # Average predictions from all window models
            y_pred_avg = np.mean(y_preds, axis=0)
            y_pred_binary = (y_pred_avg > 0.5).astype(int)

            # Calculate metrics
            acc = accuracy_score(y_val, y_pred_binary)
            brier = brier_score_loss(y_val, y_pred_avg)
            auc = roc_auc_score(y_val, y_pred_avg)

            fold_metrics.append({
                'accuracy': acc,
                'brier_score': brier,
                'auc': auc
            })

            print(f"Fold {fold} Metrics:")
            print(f"Accuracy: {acc:.3f}")
            print(f"Brier Score: {brier:.3f}")
            print(f"AUC-ROC: {auc:.3f}")

        # Print overall performance
        print("\nOverall Model Performance:")
        metrics_df = pd.DataFrame(fold_metrics)
        for metric in metrics_df.columns:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")
            
        # Calculate mean feature importance
        for feat, values in feature_importance_dict.items():
            self.feature_importances[feat] = np.mean(values)
            
        # Summarize the top features
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_features[:20]
        
        print("\nTop 20 most important features:")
        for feat, imp in top_features:
            print(f"{feat}: {imp:.4f}")
            
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble of models.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.models:
            raise ValueError("Models not trained yet. Call train first.")
        
        # Create a backup of original input
        X_original = X.copy()
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Get predictions from each fold's ensemble
        all_fold_preds = []
        
        try:
            for fold_idx, (window_models, scaler, stable_features) in enumerate(self.models):
                print(f"Processing fold {fold_idx+1} prediction...")
                
                # In case of feature mismatch, try direct prediction with manual alignment
                try:
                    # Create a new DataFrame with the exact features needed
                    X_aligned = pd.DataFrame(index=X.index)
                    
                    # Add each expected feature, with a default of 0 if missing
                    for feature in stable_features:
                        if feature in X.columns:
                            X_aligned[feature] = X[feature].values
                        else:
                            X_aligned[feature] = np.zeros(len(X))
                    
                    # Scale the aligned features
                    try:
                        X_scaled = scaler.transform(X_aligned)
                    except Exception as e:
                        print(f"Warning: Scaling error: {e}, trying alternative approach")
                        # If scaler fails, just normalize the data
                        X_scaled = (X_aligned - X_aligned.mean()) / X_aligned.std().replace(0, 1)
                        X_scaled = X_scaled.fillna(0).to_numpy()
                    
                    # Get predictions from each window model
                    window_preds = []
                    for window_info, model, features in window_models:
                        print(f"  - Using {window_info} window model")
                        # Get indices of features used by this model
                        feature_indices = [i for i, f in enumerate(stable_features) if f in features]
                        
                        if feature_indices:
                            # Extract the appropriate feature subset
                            X_model_input = X_scaled[:, feature_indices]
                            
                            # Make predictions
                            try:
                                # Try direct predict_proba
                                preds = model.predict_proba(X_model_input)[:, 1]
                            except Exception as e1:
                                try:
                                    # Fallback to xgboost DMatrix approach
                                    import xgboost as xgb
                                    dmatrix = xgb.DMatrix(X_model_input)
                                    preds = model.predict(dmatrix)
                                except Exception as e2:
                                    print(f"Warning: Model prediction error: {e1}, {e2}")
                                    # Last resort default prediction
                                    preds = np.full(len(X), 0.5)
                            
                            window_preds.append(preds)
                    
                    # Average predictions across window models
                    if window_preds:
                        fold_preds = np.mean(window_preds, axis=0)
                        all_fold_preds.append(fold_preds)
                
                except Exception as e:
                    print(f"Warning: Error in fold {fold_idx+1} prediction: {e}")
                    # Add a backup default prediction
                    all_fold_preds.append(np.full(len(X), 0.5))
        except Exception as e:
            print(f"Error in ensemble prediction: {e}")
        
        # Average predictions across folds
        if all_fold_preds:
            ensemble_preds = np.mean(all_fold_preds, axis=0)
        else:
            # Default prediction if no models could be used
            ensemble_preds = np.full(len(X), 0.5)
            print("Warning: Using default predictions (0.5) as no models could make valid predictions")
        
        return ensemble_preds
    
    def calculate_confidence_score(self, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """
        Calculate confidence scores for predictions.
        
        Args:
            predictions: Prediction probabilities
            features: DataFrame containing features
            
        Returns:
            np.ndarray: Confidence scores
        """
        confidence_scores = np.zeros(len(predictions))

        try:
            # Factors affecting confidence
            factors = {
                'prediction_margin': 0.3,  # Weight for prediction probability margin
                'sample_size': 0.2,        # Weight for number of previous matches
                'recent_consistency': 0.2,  # Weight for consistency in recent games
                'h2h_history': 0.15,       # Weight for head-to-head history
                'rest_advantage': 0.15     # Weight for rest day advantage
            }

            for i, pred in enumerate(predictions):
                score = 0

                # Prediction margin confidence
                prob_margin = abs(pred - 0.5) * 2  # Scale to [0, 1]
                score += prob_margin * factors['prediction_margin']

                # Sample size confidence
                if 'WIN_count_HOME_60D' in features.columns:
                    games_played = features.iloc[i]['WIN_count_HOME_60D']
                    sample_size_conf = min(games_played / 20, 1)  # Scale to [0, 1]
                    score += sample_size_conf * factors['sample_size']

                # Recent consistency confidence
                if 'HOME_CONSISTENCY_30D' in features.columns:
                    consistency = 1 - features.iloc[i]['HOME_CONSISTENCY_30D']  # Lower variance is better
                    score += consistency * factors['recent_consistency']

                # Head-to-head confidence
                if 'H2H_GAMES' in features.columns:
                    h2h_games = features.iloc[i]['H2H_GAMES']
                    h2h_conf = min(h2h_games / 10, 1)  # Scale to [0, 1]
                    score += h2h_conf * factors['h2h_history']

                # Rest advantage confidence
                if 'REST_DIFF' in features.columns:
                    rest_diff = abs(features.iloc[i]['REST_DIFF'])
                    rest_conf = min(rest_diff / 3, 1)  # Scale to [0, 1]
                    score += rest_conf * factors['rest_advantage']

                confidence_scores[i] = score

            # Normalize to [0, 1]
            if len(confidence_scores) > 1:
                conf_min = confidence_scores.min()
                conf_range = confidence_scores.max() - conf_min
                if conf_range > 0:
                    confidence_scores = (confidence_scores - conf_min) / conf_range

        except Exception as e:
            print(f"Error calculating confidence scores: {e}")
            confidence_scores = np.full(len(predictions), 0.5)

        return confidence_scores
            
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        predictions = self.predict(X)
        confidence_scores = self.calculate_confidence_score(predictions, X)
        
        return predictions, confidence_scores
            
    def get_top_features(self, n: int = 20) -> Dict:
        """
        Get the top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            dict: Feature names and importance scores
        """
        if not self.feature_importances:
            raise ValueError("Feature importances not available. Train the model first.")
            
        sorted_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:n])