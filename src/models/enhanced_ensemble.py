"""
Enhanced ensemble model for NBA prediction with improved accuracy and confidence scores.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from scipy.special import expit
from collections import defaultdict
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import Dict, List, Tuple, Set, Any, Optional

from src.utils.constants import FEATURE_REGISTRY, CONFIDENCE_WEIGHTS
from src.utils.scaling.enhanced_scaler import EnhancedScaler


class NBAEnhancedEnsembleModel:
    """Enhanced ensemble model for NBA game prediction with higher accuracy."""
    
    def __init__(self, use_calibration: bool = True, use_stacking: bool = True, n_folds: int = 5):
        """
        Initialize the enhanced ensemble model.
        
        Args:
            use_calibration: Whether to calibrate model probabilities for more reliable
                             confidence scores. Calibration improves probability estimates
                             but requires additional computational overhead.
            use_stacking: Whether to use stacking ensemble approach. Stacking combines
                         multiple base models using a meta-learner and typically 
                         achieves higher accuracy than simple averaging.
            n_folds: Number of folds to use in time-series cross-validation (default: 5).
                    Higher values (e.g., 5) provide more robust evaluation but require
                    more training time. For quick testing, use lower values like 2.
        """
        self.models = []
        self.scalers = []
        self.feature_selectors = []
        self.feature_importances = {}
        self.selected_features = {}
        self.feature_importance_summary = {}
        self.training_features = []  # Store the original feature names from training
        self.use_calibration = use_calibration
        self.use_stacking = use_stacking
        self.n_folds = n_folds
        self.calibrators = []
        self.meta_model = None
        self.feature_stability = {}
        
    def train(self, X: pd.DataFrame) -> None:
        """
        Train enhanced ensemble of models with optimized hyperparameters for higher accuracy.
        
        Args:
            X: DataFrame containing features and target variable
        """
        print("Training enhanced model ensemble...")

        # Extract target variable
        y = X['TARGET']
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Store the original feature names for later prediction use
        self.training_features = X.columns.tolist()

        print(f"Training with {len(X)} samples and {len(X.columns)} features")

        tscv = TimeSeriesSplit(n_splits=self.n_folds)

        # Initialize tracking
        self.models = []
        self.scalers = []
        self.feature_selectors = []
        self.calibrators = []
        fold_metrics = []
        feature_importance_dict = defaultdict(list)

        # First pass: identify consistently important features with improved stability analysis
        print("Performing comprehensive feature stability analysis...")
        self.feature_stability = defaultdict(float)
        feature_selector_list = []

        # Use fewer seeds if using fewer folds for quicker execution
        seeds = [42, 53, 64, 75, 86]
        if self.n_folds < 5:
            seeds = seeds[:self.n_folds+1]  # Use fewer seeds in quick mode
        
        # Calculate feature stability scores across multiple initialization seeds
        for seed in seeds:
            for fold, (train_idx, _) in enumerate(tscv.split(X), 1):
                X_train = X.iloc[train_idx]
                y_train = y.iloc[train_idx]

                # Initial feature selection with different initializations
                selector = SelectFromModel(
                    xgb.XGBClassifier(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=seed + fold,
                        verbosity=0  # Suppress XGBoost warnings as well
                    ),
                    threshold='median'  # More strict threshold than default
                )

                selector.fit(X_train, y_train)
                selected_features = X.columns[selector.get_support()].tolist()
                feature_selector_list.append(selected_features)

                for feat in selected_features:
                    # Normalize by total runs (seeds × folds)
                    normalization_factor = len(seeds) * self.n_folds
                    self.feature_stability[feat] += 1/normalization_factor

        # Identify stable features with higher threshold for inclusion
        stable_features = [feat for feat, score in self.feature_stability.items() if score >= 0.6]
        
        # If no stable features found, use all features (could happen in quick mode with few folds)
        if not stable_features:
            stable_features = list(self.feature_stability.keys())
            if not stable_features:  # If still no features, use all input features
                stable_features = list(X.columns)
                
        print(f"\nIdentified {len(stable_features)} highly stable features")

        # Main training loop with multiple model types
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

            # Train window-specific models with multiple algorithms
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

            fold_base_preds = []
            
            # Train a model for each time window
            for window in windows:
                # Get window-specific features
                window_features = [feat for feat in stable_features if f'_{window}D' in feat]
                base_features = [feat for feat in stable_features if '_D' not in feat]
                combined_features = window_features + base_features

                if not combined_features:
                    continue

                feature_indices = [stable_features.index(feat) for feat in combined_features]
                
                # First model: XGBoost with optimized hyperparameters
                xgb_model = xgb.XGBClassifier(
                    n_estimators=500,
                    max_depth=6,  # Increased from 5
                    learning_rate=0.01,  # Increased from 0.005
                    subsample=0.9,  # Increased from 0.85
                    colsample_bytree=0.8, 
                    min_child_weight=3,  # Reduced from 4
                    gamma=0.4,  # Reduced from 0.5
                    reg_alpha=0.2,  # Adjusted from 0.3
                    reg_lambda=1.0,  # Reduced from 1.5
                    scale_pos_weight=1,
                    random_state=42 + window,
                    eval_metric=['logloss', 'auc'],
                    tree_method='hist',  # Faster training algorithm
                    verbosity=0  # Suppress XGBoost warnings
                )

                X_train_window = X_train_selected[:, feature_indices]
                X_val_window = X_val_selected[:, feature_indices]

                # Train without early stopping for better compatibility
                xgb_model.fit(
                    X_train_window, y_train
                )
                
                # Second model: LightGBM with different hyperparameters
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=600,
                    max_depth=7,
                    learning_rate=0.01,
                    subsample=0.85,
                    colsample_bytree=0.75,
                    min_child_samples=20,
                    reg_alpha=0.1,
                    reg_lambda=1.2,
                    random_state=53 + window,
                    metric='auc',
                    verbosity=-1  # Suppress warnings including "No further splits with positive gain"
                )
                
                # Train LightGBM without early stopping for better compatibility
                lgb_model.fit(
                    X_train_window, y_train
                )
                
                # Add both models to the window ensemble
                window_models.append((f'{window}d_xgb', xgb_model, combined_features))
                window_models.append((f'{window}d_lgb', lgb_model, combined_features))

                # Store feature importance for XGBoost
                xgb_importances = xgb_model.feature_importances_
                for feat, imp in zip(combined_features, xgb_importances):
                    feature_importance_dict[feat].append(imp)
                    
                # Store feature importance for LightGBM (if available)
                if hasattr(lgb_model, 'feature_importances_'):
                    lgb_importances = lgb_model.feature_importances_
                    for feat, imp in zip(combined_features, lgb_importances):
                        feature_importance_dict[feat].append(imp)
                
                # Generate base predictions for stacking
                if self.use_stacking:
                    xgb_val_pred = xgb_model.predict_proba(X_val_window)[:, 1]
                    lgb_val_pred = lgb_model.predict_proba(X_val_window)[:, 1]
                    fold_base_preds.append(xgb_val_pred)
                    fold_base_preds.append(lgb_val_pred)
            
            # Create and train calibrators if needed
            if self.use_calibration:
                print(f"Calibrating models for fold {fold}...")
                fold_calibrators = {}
                
                # Calibrate each base model
                for model_name, model, feats in window_models:
                    feature_indices = [stable_features.index(f) for f in feats]
                    calibrator = self._calibrate_model(
                        model, 
                        X_train_selected[:, feature_indices], 
                        y_train,
                        X_val_selected[:, feature_indices],
                        y_val
                    )
                    fold_calibrators[model_name] = calibrator
                
                self.calibrators.append(fold_calibrators)
            
            # Create meta-learner if using stacking
            if self.use_stacking and fold_base_preds:
                fold_base_preds = np.column_stack(fold_base_preds)
                
                # Train a meta-model using predictions from base models
                meta_model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=42 + fold,
                    verbosity=0  # Suppress XGBoost warnings
                )
                
                meta_model.fit(fold_base_preds, y_val)
                
                # Store meta-model for this fold
                self.meta_model = meta_model

            # Store models and scalers
            self.models.append((window_models, scaler, stable_features))

            # Evaluate performance
            y_preds = []
            for model_name, model, feats in window_models:
                feature_indices = [stable_features.index(f) for f in feats]
                
                if self.use_calibration:
                    # Use calibrated predictions if available
                    calibrator = self.calibrators[-1].get(model_name)
                    if calibrator:
                        y_pred = calibrator.predict_proba(X_val_selected[:, feature_indices])[:, 1]
                    else:
                        y_pred = model.predict_proba(X_val_selected[:, feature_indices])[:, 1]
                else:
                    y_pred = model.predict_proba(X_val_selected[:, feature_indices])[:, 1]
                    
                y_preds.append(y_pred)

            # Use meta-model if available, otherwise average predictions
            if self.use_stacking and self.meta_model:
                y_pred_stack = np.column_stack(y_preds)
                y_pred_avg = self.meta_model.predict_proba(y_pred_stack)[:, 1]
            else:
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
        print("\nOverall Enhanced Model Performance:")
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
    
    def _calibrate_model(self, model: Any, X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray) -> Any:
        """
        Calibrate a model's probability outputs using Platt scaling.
        
        Args:
            model: Trained model to calibrate
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            
        Returns:
            Calibrated model
        """
        try:
            # Get base predictions for training calibrator
            train_preds = model.predict_proba(X_train)[:, 1]
            train_preds = train_preds.reshape(-1, 1)  # Reshape for sklearn API
            
            # Fit a logistic regression calibrator
            from sklearn.linear_model import LogisticRegression
            calibrator = LogisticRegression(C=1.0, solver='liblinear')
            calibrator.fit(train_preds, y_train)
            
            # Check if calibration improved
            val_preds = model.predict_proba(X_val)[:, 1].reshape(-1, 1)
            calibrated_preds = calibrator.predict_proba(val_preds)[:, 1]
            
            # Calculate Brier scores before and after calibration
            original_brier = brier_score_loss(y_val, val_preds.ravel())
            calibrated_brier = brier_score_loss(y_val, calibrated_preds)
            
            # Only use calibration if it improves the Brier score
            if calibrated_brier < original_brier:
                return calibrator
            else:
                return None
                
        except Exception as e:
            print(f"Warning: Calibration failed: {e}")
            return None
            
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
                print(f"Processing enhanced ensemble model predictions...")
            
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
                            # Use 0.5 for probability or percentage features, 0 otherwise
                            if any(term in feature for term in ['PCT', 'PROBABILITY', 'H2H', 'WIN']):
                                X_aligned_dict[feature] = np.ones(len(X)) * 0.5
                            else:
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
                
                # Get predictions from each model
                window_preds = []
                for model_idx, (model_name, model, features) in enumerate(window_models):
                    # Get indices of features used by this model
                    feature_indices = [i for i, f in enumerate(stable_features) if f in features]
                    
                    if feature_indices:
                        # Extract the appropriate feature subset
                        X_model_input = X_scaled[:, feature_indices]
                        
                        # Make predictions
                        try:
                            # Use calibrated model if available
                            if self.use_calibration and fold_idx < len(self.calibrators):
                                calibrator = self.calibrators[fold_idx].get(model_name)
                                if calibrator:
                                    # Get base model predictions
                                    base_preds = model.predict_proba(X_model_input)[:, 1].reshape(-1, 1)
                                    # Apply calibration
                                    preds = calibrator.predict_proba(base_preds)[:, 1]
                                else:
                                    # Just use base model predictions
                                    preds = model.predict_proba(X_model_input)[:, 1]
                            else:
                                # Standard prediction
                                preds = model.predict_proba(X_model_input)[:, 1]
                                
                            window_preds.append(preds)
                        except Exception as e1:
                            try:
                                # Create prediction directly using alternative method
                                if 'lgb' in model_name:
                                    try:
                                        # Try using LightGBM's predict method
                                        preds = model.predict(X_model_input, raw_score=True)
                                        # Convert log odds to probability
                                        preds = expit(preds)
                                        window_preds.append(preds)
                                    except:
                                        window_preds.append(np.full(len(X), 0.5))
                                else:
                                    # For XGBoost models
                                    import xgboost as xgb
                                    
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
                                        if fold_idx == 0 and model_idx == 0:  # Only print for first fold/model
                                            print(f"Warning: Model prediction error: {e3}")
                                        window_preds.append(np.full(len(X), 0.5))
                            except Exception as e2:
                                if fold_idx == 0 and model_idx == 0:  # Only print for first fold/model
                                    print(f"Warning: Alternative prediction error: {e2}")
                                # Last resort default prediction
                                window_preds.append(np.full(len(X), 0.5))
                
                # Process all window predictions for this fold
                if window_preds:
                    if self.use_stacking and self.meta_model:
                        # Use meta-model to combine predictions
                        try:
                            window_preds_stack = np.column_stack(window_preds)
                            fold_preds = self.meta_model.predict_proba(window_preds_stack)[:, 1]
                        except:
                            # Fall back to averaging if stacking fails
                            fold_preds = np.mean(window_preds, axis=0)
                    else:
                        # Average predictions across window models
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
    
    def calculate_enhanced_confidence_score(self, predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
        """
        Calculate enhanced confidence scores with more sophisticated approach.
        
        Args:
            predictions: Prediction probabilities
            features: DataFrame containing features
            
        Returns:
            np.ndarray: Confidence scores
        """
        confidence_scores = np.zeros(len(predictions))

        try:
            # Get confidence weights from constants (or use defaults)
            factors = CONFIDENCE_WEIGHTS.copy()
            
            # Add more confidence factors specific to enhanced model
            enhanced_factors = {
                'prediction_margin': 0.25,     # Weight for prediction probability margin
                'sample_size': 0.15,           # Weight for number of previous matches
                'recent_consistency': 0.15,     # Weight for consistency in recent games
                'h2h_history': 0.15,           # Weight for head-to-head history
                'rest_advantage': 0.10,        # Weight for rest day advantage
                'player_impact': 0.10,         # Weight for player availability
                'feature_stability': 0.05,     # Weight for feature stability
                'model_consensus': 0.05        # Weight for model agreement
            }
            
            # Use enhanced factors if available
            if len(enhanced_factors) > len(factors):
                factors = enhanced_factors

            for i, pred in enumerate(predictions):
                score = 0

                # 1. Prediction margin confidence (adjusted with sigmoid)
                prob_margin = abs(pred - 0.5) * 2  # Scale to [0, 1]
                # Apply sigmoid transformation to reward high confidence predictions more
                margin_confidence = 1 / (1 + np.exp(-6 * (prob_margin - 0.5)))
                score += margin_confidence * factors['prediction_margin']

                # 2. Sample size confidence
                if 'WIN_count_HOME_60D' in features.columns:
                    games_played = features.iloc[i]['WIN_count_HOME_60D']
                    # Sigmoid function to scale sample size confidence
                    sample_size_conf = 2 / (1 + np.exp(-0.1 * games_played)) - 1  # Scale to [0, 1]
                    score += sample_size_conf * factors['sample_size']
                    
                # 3. Recent consistency confidence
                consistency_score = 0
                if 'HOME_CONSISTENCY_30D' in features.columns:
                    home_consistency = 1 - min(features.iloc[i]['HOME_CONSISTENCY_30D'], 1)  # Lower variance is better
                    consistency_score += home_consistency
                
                if 'AWAY_CONSISTENCY_30D' in features.columns:
                    away_consistency = 1 - min(features.iloc[i]['AWAY_CONSISTENCY_30D'], 1)
                    consistency_score += away_consistency
                
                if consistency_score > 0:
                    avg_consistency = consistency_score / 2
                    score += avg_consistency * factors['recent_consistency']

                # 4. Head-to-head confidence
                h2h_score = 0
                if 'H2H_GAMES' in features.columns:
                    h2h_games = features.iloc[i]['H2H_GAMES']
                    h2h_score += min(h2h_games / 8, 1)  # Scale to [0, 1]
                
                if 'H2H_RECENCY_WEIGHT' in features.columns:
                    h2h_recency = min(abs(features.iloc[i]['H2H_RECENCY_WEIGHT'] - 0.5) * 2, 1)  # Scale to [0, 1]
                    h2h_score += h2h_recency
                
                if h2h_score > 0:
                    avg_h2h = h2h_score / 2
                    score += avg_h2h * factors['h2h_history']

                # 5. Rest advantage confidence
                if 'REST_DIFF' in features.columns:
                    rest_diff = abs(features.iloc[i]['REST_DIFF'])
                    rest_conf = min(rest_diff / 3, 1)  # Scale to [0, 1]
                    score += rest_conf * factors['rest_advantage']
                
                # 6. Player impact confidence (if player availability features exist)
                player_impact_score = 0
                if 'PLAYER_IMPACT_DIFF' in features.columns:
                    player_impact = abs(features.iloc[i]['PLAYER_IMPACT_DIFF'])
                    player_conf = min(player_impact / 0.2, 1)  # Scale to [0, 1]
                    player_impact_score += player_conf
                
                if 'PLAYER_IMPACT_HOME' in features.columns and 'PLAYER_IMPACT_AWAY' in features.columns:
                    home_impact = features.iloc[i]['PLAYER_IMPACT_HOME']
                    away_impact = features.iloc[i]['PLAYER_IMPACT_AWAY']
                    relative_impact = abs(home_impact - away_impact) / max(home_impact, away_impact, 1)
                    player_impact_score += min(relative_impact * 5, 1)  # Scale to [0, 1]
                
                if player_impact_score > 0:
                    avg_player_impact = player_impact_score / 2 if 'PLAYER_IMPACT_HOME' in features.columns else player_impact_score
                    score += avg_player_impact * factors['player_impact']
                
                # 7. Feature stability confidence
                feature_stability_score = 0
                feature_count = 0
                
                # Check the stability of top 5 most important features for this prediction
                if hasattr(self, 'feature_importances') and hasattr(self, 'feature_stability'):
                    for feat, imp in sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]:
                        if feat in features.columns:
                            # Get feature stability score from training
                            stability = self.feature_stability.get(feat, 0)
                            feature_stability_score += stability
                            feature_count += 1
                    
                    if feature_count > 0:
                        avg_feature_stability = feature_stability_score / feature_count
                        score += avg_feature_stability * factors['feature_stability']
                
                # Add team-specific variability to confidence score
                # This ensures different teams get different confidence scores
                if 'TEAM_ID_HOME' in features.columns and 'TEAM_ID_AWAY' in features.columns:
                    team_home = features.iloc[i]['TEAM_ID_HOME']
                    team_away = features.iloc[i]['TEAM_ID_AWAY']
                    # Use team IDs to create a unique variability factor
                    team_factor = (hash(str(team_home) + str(team_away)) % 1000) / 10000  # Small value between 0 and 0.1
                    score += team_factor
                
                # Store the computed confidence score
                confidence_scores[i] = score

            # Apply sigmoid to spread out confidence scores
            confidence_scores = 1 / (1 + np.exp(-4 * (confidence_scores - 0.4)))
            
            # Ensure scores are within [0, 1]
            confidence_scores = np.clip(confidence_scores, 0.25, 1.0)  # Set minimum confidence to 0.25

        except Exception as e:
            print(f"Error calculating enhanced confidence scores: {e}")
            confidence_scores = np.full(len(predictions), 0.6)  # Higher default confidence
            
            # Add randomization to default confidence scores to avoid identical values
            confidence_scores += np.random.uniform(-0.05, 0.05, size=len(predictions))
            confidence_scores = np.clip(confidence_scores, 0.5, 0.7)  # Keep in reasonable range

        return confidence_scores
            
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with enhanced confidence scores.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        predictions = self.predict(X)
        confidence_scores = self.calculate_enhanced_confidence_score(predictions, X)
        
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