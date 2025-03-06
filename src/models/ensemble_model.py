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
from typing import Dict, List, Tuple, Set

from src.utils.constants import FEATURE_REGISTRY
from src.utils.scaling.enhanced_scaler import EnhancedScaler


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
        self.training_features = []  # Store the original feature names from training
        
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
        
        # Store the original feature names for later prediction use
        self.training_features = X.columns.tolist()

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

            # Scale features using enhanced scaler for robustness
            scaler = EnhancedScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Store feature names in the scaler for easier debugging
            if not hasattr(scaler, 'feature_names_in_'):
                setattr(scaler, 'feature_names_in_', np.array(X_train.columns))

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
        Make predictions using the trained ensemble of models with enhanced feature handling.
        
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
        
        # Ensure we have valid training features
        if not hasattr(self, 'training_features') or not self.training_features:
            # If training_features wasn't saved, try to infer from the first fold's scaler
            if self.models and self.models[0] and self.models[0][1]:
                # Try to get feature names from the scaler
                if hasattr(self.models[0][1], 'feature_names_in_'):
                    self.training_features = self.models[0][1].feature_names_in_.tolist()
                else:
                    print("Warning: Unable to determine original training features. Prediction may be inaccurate.")
        
        # Get predictions from each fold's ensemble
        all_fold_preds = []
        
        for fold_idx, (window_models, scaler, stable_features) in enumerate(self.models):
            # Only print for the first fold to reduce verbosity
            if fold_idx == 0:
                print(f"Processing ensemble model predictions...")
            
            try:
                # Create a dictionary to collect all features
                X_aligned_dict = {}
                
                # Add each expected feature, with a default of 0 if missing
                for feature in stable_features:
                    # If feature exists in input, use it
                    if feature in X.columns:
                        X_aligned_dict[feature] = X[feature].values
                    else:
                        # Check if we can derive this feature from others (for certain feature types)
                        feature_derived = False
                        
                        # Try to get the base feature name without window suffix
                        base_feature = feature
                        window = None
                        if '_D' in feature:
                            parts = feature.split('_')
                            for i, part in enumerate(parts):
                                if part.endswith('D') and part[:-1].isdigit():
                                    base_feature = '_'.join(parts[:i])
                                    window = part[:-1]
                                    break
                        
                        # Check if this is a registered feature type we can derive
                        if base_feature in FEATURE_REGISTRY:
                            feature_info = FEATURE_REGISTRY[base_feature]
                            
                            # Only try to derive if it's a derived feature and we have all dependencies
                            if feature_info['type'] in ['derived', 'interaction'] and 'dependencies' in feature_info:
                                # Get dependency column names, applying window if needed
                                dependencies = []
                                for dep in feature_info['dependencies']:
                                    if window and self._should_apply_window(dep):
                                        dependencies.append(f"{dep}_{window}D")
                                    else:
                                        dependencies.append(dep)
                                
                                # Check if all dependencies are available
                                if all(dep in X.columns for dep in dependencies):
                                    # Derive the feature based on its type
                                    if base_feature == 'WIN_PCT_DIFF':
                                        X_aligned_dict[feature] = X[dependencies[0]] - X[dependencies[1]]
                                        feature_derived = True
                                    elif base_feature in ['OFF_RTG_DIFF', 'DEF_RTG_DIFF', 'NET_RTG_DIFF', 'PACE_DIFF']:
                                        X_aligned_dict[feature] = X[dependencies[0]] - X[dependencies[1]]
                                        feature_derived = True
                                    # Add other derivation rules as needed...
                                
                                # More derivation cases can be added here
                        
                        # If we couldn't derive it, use a default value
                        if not feature_derived:
                            X_aligned_dict[feature] = np.zeros(len(X))
                
                # Create DataFrame all at once to avoid fragmentation
                X_aligned = pd.DataFrame(X_aligned_dict, index=X.index)
                
                # Scale the aligned features
                try:
                    if isinstance(scaler, EnhancedScaler):
                        # Use enhanced scaler directly
                        X_scaled = scaler.transform(X_aligned)
                    else:
                        # For backward compatibility with old models using StandardScaler
                        X_scaled = scaler.transform(X_aligned)
                except Exception as e:
                    # Create a more detailed error message with missing feature information
                    missing_features = []
                    if hasattr(scaler, 'feature_names_in_'):
                        expected = set(scaler.feature_names_in_)
                        actual = set(X_aligned.columns)
                        missing_features = list(expected - actual)
                    
                    print(f"Warning: Scaling error: {e}")
                    if missing_features:
                        print(f"Missing features: {missing_features[:5]}...")
                    
                    # Use enhanced scaler as fallback
                    fallback_scaler = EnhancedScaler()
                    X_scaled = fallback_scaler.fit_transform(X_aligned)
                
                # Get predictions from each window model
                window_preds = []
                for window_info, model, features in window_models:
                    # Get indices of features used by this model
                    feature_indices = [i for i, f in enumerate(stable_features) if f in features]
                    
                    if feature_indices:
                        # Extract the appropriate feature subset
                        X_model_input = X_scaled[:, feature_indices]
                        
                        # Make predictions
                        try:
                            # Try direct predict_proba
                            preds = model.predict_proba(X_model_input)[:, 1]
                            window_preds.append(preds)
                        except Exception as e1:
                            try:
                                # Create prediction directly using booster
                                import xgboost as xgb
                                feature_subset = [stable_features[i] for i in feature_indices]
                                
                                # Ensure we're passing a proper numpy array to DMatrix
                                if isinstance(X_model_input, np.ndarray):
                                    X_input_array = X_model_input
                                else:
                                    X_input_array = np.array(X_model_input)
                                
                                # Try using the model's booster directly
                                try:
                                    # Extract the model's booster for direct prediction
                                    if hasattr(model, 'get_booster'):
                                        booster = model.get_booster()
                                        dmatrix = xgb.DMatrix(X_input_array)
                                        preds = booster.predict(dmatrix)
                                        window_preds.append(preds)
                                    else:
                                        # Fallback to direct numpy prediction
                                        preds = model.predict(X_input_array)
                                        window_preds.append(preds)
                                except Exception as e3:
                                    if fold_idx == 0:  # Only print for first fold
                                        print(f"Warning: XGBoost prediction error: {e3}")
                                    window_preds.append(np.full(len(X), 0.5))
                            except Exception as e2:
                                if fold_idx == 0:  # Only print for first fold
                                    print(f"Warning: Model prediction error: {e2}")
                                # Last resort default prediction
                                window_preds.append(np.full(len(X), 0.5))
                
                # Average predictions across window models
                if window_preds:
                    fold_preds = np.mean(window_preds, axis=0)
                    all_fold_preds.append(fold_preds)
                else:
                    # If no window model worked, use a default prediction
                    all_fold_preds.append(np.full(len(X), 0.5))
            
            except Exception as e:
                # Only print for first fold to reduce verbosity
                if fold_idx == 0:
                    print(f"Warning: Error in ensemble model prediction: {e}")
                # Add a backup default prediction
                all_fold_preds.append(np.full(len(X), 0.5))
        
        # Average predictions across folds
        if all_fold_preds:
            ensemble_preds = np.mean(all_fold_preds, axis=0)
        else:
            # Default prediction if no models could be used
            ensemble_preds = np.full(len(X), 0.5)
            print("Warning: Using default predictions (0.5) as no models could make valid predictions")
        
        return ensemble_preds
        
    def _should_apply_window(self, column_name: str) -> bool:
        """
        Determine if a window should be applied to a column name.
        
        Args:
            column_name: Column name to check
            
        Returns:
            True if window should be applied
        """
        # Don't apply windows to columns that already have them
        if '_D' in column_name:
            return False
            
        # Don't apply windows to specific feature types
        no_window_prefixes = ['REST_DAYS_', 'H2H_', 'DAYS_SINCE_', 'LAST_GAME_']
        return not any(column_name.startswith(prefix) for prefix in no_window_prefixes)
    
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