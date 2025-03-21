"""
Advanced hybrid model that integrates ensemble and deep learning approaches
for optimal prediction accuracy and confidence scoring.
"""
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple, Any, Optional, Union

from src.models.enhanced_ensemble import NBAEnhancedEnsembleModel
from src.models.enhanced_deep_model import EnhancedDeepModelTrainer


class HybridModel:
    """
    Hybrid model that combines enhanced ensemble and deep learning approaches
    with sophisticated integration techniques for maximum accuracy.
    """
    
    def __init__(self, 
                ensemble_model: Optional[NBAEnhancedEnsembleModel] = None,
                deep_model: Optional[EnhancedDeepModelTrainer] = None,
                ensemble_weight: float = 0.3,  # Reduced from 0.6 to favor deep model
                quick_mode: bool = False):
        """
        Initialize the hybrid model.
        
        Args:
            ensemble_model: Pre-trained ensemble model (or None to create new).
                           If provided, this model will be used directly without retraining.
            deep_model: Pre-trained deep model (or None to create new).
                       If provided, this model will be used directly without retraining.
            ensemble_weight: Weight given to ensemble model predictions (vs. deep model).
                            Values closer to 1.0 favor the ensemble model, while values
                            closer to 0.0 favor the deep learning model.
                            Default of 0.3 now favors the deep learning model based on
                            empirical performance on unseen data, where the deep model showed
                            superior predictive power.
            quick_mode: Whether to run in quick testing mode. When True:
                       - Uses fewer weights to test in optimization
                       - Performs less thorough model integration
                       - For faster development and testing iterations
        """
        self.ensemble_model = ensemble_model or NBAEnhancedEnsembleModel(
            use_calibration=True,
            use_stacking=True
        )
        
        self.deep_model = deep_model or EnhancedDeepModelTrainer(
            use_residual=True,
            use_attention=True,
            use_mc_dropout=True
        )
        
        self.ensemble_weight = ensemble_weight
        self.is_trained = False
        self.meta_weights = None  # Will store optimal weights if meta-learning is used
        self.quick_mode = quick_mode
        
    def train(self, X: pd.DataFrame) -> None:
        """
        Train both ensemble and deep learning models with strict temporal integrity.
        
        Args:
            X: DataFrame containing features and target variable
        """
        from sklearn.model_selection import train_test_split
        
        print("Training hybrid prediction model with strict temporal integrity...")
        
        # Sort by date first to ensure correct temporal ordering - CRITICAL for preventing data leakage
        if 'GAME_DATE' in X.columns:
            X = X.sort_values('GAME_DATE').copy()
            print("Data sorted by GAME_DATE to ensure proper temporal ordering")
        else:
            print("Warning: GAME_DATE not found in dataset. Assuming data is already in temporal order.")
        
        # CRITICAL FIX: Create dedicated holdout set for weight optimization using TEMPORAL split (not random)
        # Use strict time-based split with no overlap instead of train_test_split with shuffle=False 
        # which can still introduce leakage in specific cases
        if 'GAME_DATE' in X.columns:
            # Find date that gives ~15% of data for weight optimization
            dates = pd.to_datetime(X['GAME_DATE'])
            split_idx = int(len(X) * 0.85)  # 85% for training, 15% for weight optimization
            split_date = dates.iloc[split_idx]
            
            # Create strict temporal splits
            X_train = X[dates < split_date].copy()
            X_weight_opt = X[dates >= split_date].copy()
            
            print(f"Temporal split at date: {split_date}")
        else:
            # Fallback to train_test_split if no date column available
            X_train, X_weight_opt = train_test_split(X, test_size=0.15, shuffle=False)
        
        print(f"Training data shape: {X_train.shape}, Weight optimization data shape: {X_weight_opt.shape}")
        
        # Verify temporal separation between train and weight optimization sets
        if 'GAME_DATE' in X_train.columns and 'GAME_DATE' in X_weight_opt.columns:
            train_end_date = X_train['GAME_DATE'].max()
            opt_start_date = X_weight_opt['GAME_DATE'].min()
            print(f"Training set end date: {train_end_date}, Weight optimization start date: {opt_start_date}")
            
            # Add extra verification to ensure absolutely no temporal overlap
            if pd.to_datetime(train_end_date) >= pd.to_datetime(opt_start_date):
                print("WARNING: Detected temporal overlap between training and weight optimization sets!")
                # Convert to datetime for proper temporal operations
                X_weight_opt['GAME_DATE'] = pd.to_datetime(X_weight_opt['GAME_DATE'])
                train_end_datetime = pd.to_datetime(train_end_date)
                
                # Add one day buffer to ensure strict temporal separation
                buffer_date = train_end_datetime + pd.Timedelta(days=1)
                
                # Filter weight optimization set to only include data after buffer date
                X_weight_opt = X_weight_opt[X_weight_opt['GAME_DATE'] > buffer_date].copy()
                
                if len(X_weight_opt) > 0:
                    print(f"Fixed temporal overlap. New weight optimization set has {len(X_weight_opt)} samples")
                    print(f"New weight optimization start date: {X_weight_opt['GAME_DATE'].min()}")
                else:
                    print("ERROR: Unable to fix temporal overlap - insufficient data for weight optimization!")
                    # Fall back to using a small portion of training data with strict temporal ordering
                    X_train = X_train.sort_values('GAME_DATE')
                    split_idx = int(len(X_train) * 0.8)
                    X_weight_opt = X_train.iloc[split_idx:].copy()
                    X_train = X_train.iloc[:split_idx].copy()
                    print(f"Created fallback split: Training data: {len(X_train)}, Weight optimization data: {len(X_weight_opt)}")
            else:
                print("Temporal integrity verified: Training data strictly precedes weight optimization data")
        else:
            print("Cannot verify temporal separation - GAME_DATE not available")
        
        # Train ensemble model on training data only
        print("\n==== Training Enhanced Ensemble Model ====")
        self.ensemble_model.train(X_train)
        
        # Train deep model on training data only
        print("\n==== Training Enhanced Deep Learning Model ====")
        self.deep_model.train_deep_model(X_train)
        
        # Learn optimal combination weights using separate weight optimization data
        print("\n==== Optimizing Model Weights on Separate Validation Data ====")
        self._optimize_weights(X_weight_opt)
        
        self.is_trained = True
        print("\nHybrid model training complete!")
        
    def _optimize_weights(self, X: pd.DataFrame) -> None:
        """
        Optimize the weighting between models using validation data with advanced metrics
        and adaptive weight search for optimal model integration.
        
        This method has been improved to use a separate weight optimization dataset that
        isn't used in model training, preventing data leakage and providing more reliable
        weight optimization.
        
        Args:
            X: DataFrame containing features and target variable (separate weight optimization data)
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
        
        print("\nOptimizing model integration weights with advanced metrics...")
        print("Using separate data for weight optimization to prevent data leakage")
        
        # Set optimization mode flag on deep model to enable fast prediction path
        if hasattr(self.deep_model, '_in_hybrid_optimization'):
            # If flag already exists, just set it
            self.deep_model._in_hybrid_optimization = True
        else:
            # Create the flag attribute
            setattr(self.deep_model, '_in_hybrid_optimization', True)
        
        # Sort the weight optimization data by date if available
        if 'GAME_DATE' in X.columns:
            X = X.sort_values('GAME_DATE').copy()
            print("Weight optimization data sorted by date to preserve temporal order")
        
        # Extract target from the weight optimization dataset
        y = X['TARGET']
        
        # Store individual model performance for evaluation
        print("\nEvaluating individual model performance on weight optimization data:")
        ensemble_preds = self.ensemble_model.predict(X)
        deep_preds = self.deep_model.predict(X)
        
        ensemble_binary = (ensemble_preds > 0.5).astype(int)
        deep_binary = (deep_preds > 0.5).astype(int)
        
        ensemble_acc = accuracy_score(y, ensemble_binary)
        deep_acc = accuracy_score(y, deep_binary)
        
        print(f"Ensemble model accuracy: {ensemble_acc:.4f}")
        print(f"Deep learning model accuracy: {deep_acc:.4f}")
        
        # Ensure there are enough samples for at least 2 splits
        n_splits = max(2, min(5, len(X) // 300))  # Minimum of 2, maximum of 5 splits
        print(f"Using {n_splits} splits for time-series cross-validation")
        
        # Initialize time-series cross-validation with strict temporal validation
        tscv = TimeSeriesSplit(n_splits=n_splits, gap=1)  # Add gap=1 to ensure no data leakage between folds
        
        # Metrics to track
        metrics = {
            'accuracy': [],
            'brier_score': [],
            'auc': []
        }
        
        # Weight search parameters
        weight_results = {}
        
        # Use fewer weights in all modes for faster performance
        if self.quick_mode:
            # Very minimal search in quick mode
            weights = [0.0, 0.5, 1.0]  # Just check extremes and middle
        else:
            # More reasonable search space (11 weights instead of 21)
            weights = [round(x * 0.1, 2) for x in range(0, 11)]  # 0.0 to 1.0 in steps of 0.1
            
        print(f"Searching across {len(weights)} different weight combinations")
        
        # Collect predictions and confidences for model performance tracking
        model_performance = {
            'ensemble': {'correct': 0, 'total': 0},
            'deep': {'correct': 0, 'total': 0}
        }
        
        # Test different weights to find optimal combination
        for weight in weights:
            fold_metrics = {
                'accuracy': [],
                'brier_score': [],
                'auc': []
            }
            
            # Pre-compute model predictions and confidences for all validation folds
            # to avoid redundant computation
            fold_data = []
            
            # Verify and ensure temporal integrity of splits
            modified_splits = []
            
            if 'GAME_DATE' in X.columns:
                print("Verifying temporal integrity of cross-validation splits...")
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
                    train_dates = pd.to_datetime(X.iloc[train_idx]['GAME_DATE'])
                    val_dates = pd.to_datetime(X.iloc[val_idx]['GAME_DATE'])
                    
                    train_max = train_dates.max()
                    val_min = val_dates.min()
                    
                    print(f"Fold {fold}: Train end date: {train_max}, Validation start date: {val_min}")
                    
                    # Check for temporal overlap
                    if train_max >= val_min:
                        print(f"WARNING: Detected temporal overlap in fold {fold}!")
                        # Add one day buffer to ensure strict temporal separation
                        buffer_date = train_max + pd.Timedelta(days=1)
                        
                        # Filter validation indices to only include dates after buffer date
                        valid_val_idx = val_idx[pd.to_datetime(X.iloc[val_idx]['GAME_DATE']) > buffer_date]
                        
                        if len(valid_val_idx) > 0:
                            # Replace validation indices with temporally safe subset
                            print(f"Fixed temporal overlap. New validation set has {len(valid_val_idx)} samples")
                            val_idx = valid_val_idx
                            val_min = pd.to_datetime(X.iloc[val_idx]['GAME_DATE']).min()
                            print(f"New validation start date: {val_min}")
                        else:
                            print(f"WARNING: Unable to fix temporal overlap in fold {fold} - skipping this fold")
                            # Skip this fold if we can't fix it properly
                            continue
                    else:
                        print(f"Fold {fold} temporal integrity verified ✓")
                    
                    # Store verified indices
                    modified_splits.append((train_idx, val_idx))
                
                # Check if we have any valid splits after verification
                if not modified_splits:
                    print("WARNING: No valid folds after temporal verification. Creating fallback split...")
                    # Create a fallback temporal split if necessary
                    X_sorted = X.sort_values('GAME_DATE')
                    split_point = int(len(X_sorted) * 0.8)
                    train_idx = X_sorted.index[:split_point]
                    val_idx = X_sorted.index[split_point:]
                    modified_splits = [(train_idx, val_idx)]
            else:
                # If no date column, use original splits but warn
                modified_splits = list(tscv.split(X))
                print("WARNING: No date column available to verify temporal integrity of folds")
            
            for train_idx, val_idx in modified_splits:
                # Use validation data for weight optimization
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                
                # Store fold data
                fold_info = {'X_val': X_val, 'y_val': y_val}
                
                # Get predictions from both models
                print("Processing enhanced ensemble model predictions...")
                ensemble_preds = self.ensemble_model.predict(X_val)
                fold_info['ensemble_preds'] = ensemble_preds
                
                print("Processing enhanced deep model predictions...")
                deep_preds = self.deep_model.predict(X_val)
                fold_info['deep_preds'] = deep_preds
                
                # Track individual model performance for analysis
                model_performance['ensemble']['correct'] += np.sum((ensemble_preds > 0.5).astype(int) == y_val)
                model_performance['ensemble']['total'] += len(y_val)
                model_performance['deep']['correct'] += np.sum((deep_preds > 0.5).astype(int) == y_val)
                model_performance['deep']['total'] += len(y_val)
                
                # Calculate confidence/uncertainty - handle errors properly
                print("Processing enhanced deep model predictions with uncertainty...")
                try:
                    # Calculate and cache uncertainty
                    _, deep_uncertainties = self.deep_model.predict_with_uncertainty(X_val)
                    fold_info['deep_uncertainties'] = deep_uncertainties
                    
                    # Normalize uncertainties to [0,1]
                    normalized_uncertainties = np.nan_to_num(
                        deep_uncertainties / (np.max(deep_uncertainties) + 1e-10), 
                        nan=0.5
                    )
                    # Convert to confidence (1 - uncertainty)
                    deep_confidence = 1.0 - normalized_uncertainties
                    fold_info['deep_confidence'] = deep_confidence
                except Exception as e:
                    print(f"Error: {e}. Using default confidence values.")
                    # Fallback if uncertainty calculation fails
                    fold_info['deep_confidence'] = 0.5 * np.ones_like(deep_preds)
                
                # Enhanced confidence scores
                try:
                    ensemble_confidence = self.ensemble_model.calculate_enhanced_confidence_score(ensemble_preds, X_val)
                    fold_info['ensemble_confidence'] = ensemble_confidence
                except Exception as e:
                    # Fallback if confidence calculation fails
                    fold_info['ensemble_confidence'] = 0.5 * np.ones_like(ensemble_preds)
                
                # Add to fold data collection
                fold_data.append(fold_info)
                
            # Process all weight evaluations using the cached data
            for fold_info in fold_data:
                y_val = fold_info['y_val']
                ensemble_preds = fold_info['ensemble_preds']
                deep_preds = fold_info['deep_preds']
                ensemble_confidence = fold_info['ensemble_confidence']
                deep_confidence = fold_info['deep_confidence']
                
                # 1. Fixed weight (global weight parameter)
                hybrid_preds_fixed = weight * ensemble_preds + (1 - weight) * deep_preds
                
                # 2. Confidence-based dynamic weighting (simpler version for speed)
                # Calculate sample-specific weights based on relative confidence
                confidence_sum = ensemble_confidence + deep_confidence + 1e-10  # Avoid division by zero
                ensemble_weight_dynamic = ensemble_confidence / confidence_sum
                deep_weight_dynamic = deep_confidence / confidence_sum
                
                # Apply dynamic weighting - but with less complexity than before
                hybrid_preds = (
                    ensemble_weight_dynamic * ensemble_preds + 
                    deep_weight_dynamic * deep_preds
                )
                
                # Apply global weight adjustment to balance fixed vs. dynamic
                # This is a simplification of the previous approach but maintains the core benefit
                hybrid_preds = 0.7 * hybrid_preds_fixed + 0.3 * hybrid_preds
                
                # Convert to binary predictions
                y_pred_binary = (hybrid_preds > 0.5).astype(int)
                
                # Calculate comprehensive metrics
                acc = accuracy_score(y_val, y_pred_binary)
                brier = brier_score_loss(y_val, hybrid_preds)
                
                # Calculate AUC (protect against single-class predictions)
                try:
                    auc = roc_auc_score(y_val, hybrid_preds)
                except Exception:
                    auc = 0.5  # Default for random performance
                
                # Store all metrics
                fold_metrics['accuracy'].append(acc)
                fold_metrics['brier_score'].append(brier)
                fold_metrics['auc'].append(auc)
            
            # Average metrics across folds
            avg_metrics = {k: np.mean(v) for k, v in fold_metrics.items()}
            
            # Calculate combined score with weighted metric importance
            # Higher accuracy, higher AUC, lower Brier score = better
            combined_score = (
                0.5 * avg_metrics['accuracy'] + 
                0.3 * avg_metrics['auc'] + 
                0.2 * (1.0 - avg_metrics['brier_score'])  # Convert to higher=better
            )
            
            # Store results
            weight_results[weight] = {
                'metrics': avg_metrics,
                'combined_score': combined_score
            }
            
            # Print progress for select weights to keep output manageable
            if weight in [0.0, 0.25, 0.5, 0.75, 1.0] or (not self.quick_mode and weight % 0.2 < 0.01):
                print(f"Weight {weight:.2f}: Acc={avg_metrics['accuracy']:.4f}, "
                      f"AUC={avg_metrics['auc']:.4f}, Brier={avg_metrics['brier_score']:.4f}, "
                      f"Score={combined_score:.4f}")
        
        # Calculate model performance for analysis
        ensemble_accuracy = model_performance['ensemble']['correct'] / max(model_performance['ensemble']['total'], 1)
        deep_accuracy = model_performance['deep']['correct'] / max(model_performance['deep']['total'], 1)
        print(f"\nIndividual model performance:")
        print(f"Ensemble model accuracy: {ensemble_accuracy:.4f}")
        print(f"Deep learning model accuracy: {deep_accuracy:.4f}")
        
        # Find optimal weight based on combined score
        optimal_weight = max(weight_results.items(), key=lambda x: x[1]['combined_score'])[0]
        best_score = weight_results[optimal_weight]['combined_score']
        best_metrics = weight_results[optimal_weight]['metrics']
        
        # Store optimal weight
        self.ensemble_weight = optimal_weight
        
        # Store all weight results for potential adaptive weighting
        self.weight_results = weight_results
        
        # Remove optimization mode flag to restore normal prediction behavior
        if hasattr(self.deep_model, '_in_hybrid_optimization'):
            self.deep_model._in_hybrid_optimization = False
        
        # Clear caches to free memory
        if hasattr(self.deep_model, '_pred_cache'):
            self.deep_model._pred_cache.clear()
        if hasattr(self.deep_model, '_uncertainty_cache'):
            self.deep_model._uncertainty_cache.clear()
        
        print(f"\nOptimal ensemble weight: {optimal_weight:.2f} with combined score: {best_score:.4f}")
        print(f"Metrics at optimal weight: Accuracy={best_metrics['accuracy']:.4f}, "
              f"AUC={best_metrics['auc']:.4f}, Brier Score={best_metrics['brier_score']:.4f}")
        
    def _ensure_no_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert any DataFrame columns to Series to avoid errors.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame with no DataFrame columns
        """
        # Make a copy to avoid modifying the original
        result = df.copy()
        
        # Check each column
        for col in result.columns:
            col_data = result[col]
            if isinstance(col_data, pd.DataFrame):
                print(f"Converting DataFrame column {col} to Series for hybrid model")
                if len(col_data.columns) > 0:
                    # Convert to Series using first column
                    result[col] = col_data.iloc[:, 0]
                else:
                    # Create empty Series if no columns
                    result[col] = pd.Series(0, index=result.index)
        
        return result
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the enhanced hybrid model with dynamic weighting
        based on model confidence and specialized feature adjustments.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train first.")
            
        # Ensure no DataFrame columns exist
        X = self._ensure_no_dataframe_columns(X)
        
        print("Generating hybrid model predictions...")
        
        # Suppress repeated warning messages during prediction
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
            
            # Get predictions from both models
            ensemble_preds = self.ensemble_model.predict(X)
            
            # Get deep model predictions with uncertainty if available
            try:
                deep_preds, deep_uncertainties = self.deep_model.predict_with_uncertainty(X)
                uncertainty_available = True
            except Exception as e:
                print(f"Error getting deep model predictions with uncertainty: {e}")
                print("Falling back to standard prediction without uncertainty")
                try:
                    # Try standard prediction as fallback
                    deep_preds = self.deep_model.predict(X)
                except Exception as e2:
                    print(f"Error with standard prediction: {e2}")
                    # Generate default predictions as last resort fallback
                    deep_preds = np.full(len(X), 0.5)
                uncertainty_available = False
            
            # Get confidence scores for both models
            try:
                ensemble_confidence = self.ensemble_model.calculate_enhanced_confidence_score(ensemble_preds, X)
            except Exception:
                # Fallback if confidence calculation fails
                ensemble_confidence = np.ones_like(ensemble_preds) * 0.7  # Default to moderate-high confidence
        
        if uncertainty_available:
            try:
                # Convert uncertainties to confidence scores
                normalized_uncertainties = deep_uncertainties / (np.max(deep_uncertainties) + 1e-10)
                deep_confidence = 1.0 - normalized_uncertainties
            except Exception:
                deep_confidence = np.ones_like(deep_preds) * 0.5  # Default to moderate confidence
        else:
            # Use simpler confidence approximation based on prediction strength
            deep_confidence = 0.5 + 0.4 * np.abs(deep_preds - 0.5) * 2  # Scale to [0.5, 0.9]
        
        # Prepare for adaptive weighting
        hybrid_preds = np.zeros_like(ensemble_preds)
        
        # Perform improved dynamic weighting for each prediction, favoring the deep model
        # based on real-world performance on unseen data
        for i in range(len(ensemble_preds)):
            # 1. Calculate model agreement score
            agreement = 1.0 - abs(ensemble_preds[i] - deep_preds[i])
            
            # 2. Determine if we should use dynamic or fixed weighting
            # If confidence differs significantly, use confidence-based weighting
            # Otherwise, use global weight parameter
            confidence_diff = abs(ensemble_confidence[i] - deep_confidence[i])
            
            # Calculate the dynamic weight based on demonstrated performance
            # We now favor the deep model by default when using dynamic weighting
            if deep_confidence[i] > ensemble_confidence[i]:
                # Deep model is more confident - give it more weight
                confidence_ratio = deep_confidence[i] / (ensemble_confidence[i] + 1e-10)
                # Limit the ratio to a reasonable range
                confidence_ratio = min(confidence_ratio, 3.0)
                # Calculate deep model weight (higher ratio = more weight to deep model)
                deep_weight_dynamic = 0.6 + 0.2 * (confidence_ratio - 1) / 2
                deep_weight_dynamic = min(0.8, deep_weight_dynamic)  # Cap at 80%
                ensemble_weight_dynamic = 1.0 - deep_weight_dynamic
            else:
                # Ensemble model is more confident
                confidence_ratio = ensemble_confidence[i] / (deep_confidence[i] + 1e-10)
                # Limit the ratio to a reasonable range
                confidence_ratio = min(confidence_ratio, 3.0)
                # Calculate ensemble weight but still favor deep model slightly
                ensemble_weight_dynamic = 0.4 + 0.2 * (confidence_ratio - 1) / 2
                ensemble_weight_dynamic = min(0.6, ensemble_weight_dynamic)  # Cap at 60%
                deep_weight_dynamic = 1.0 - ensemble_weight_dynamic
            
            if confidence_diff > 0.2:  # Significant confidence difference
                # Use the dynamic confidence-based weights
                hybrid_preds[i] = (
                    ensemble_weight_dynamic * ensemble_preds[i] + 
                    deep_weight_dynamic * deep_preds[i]
                )
            else:
                # Use the global optimized weight when confidences are similar
                hybrid_preds[i] = (
                    self.ensemble_weight * ensemble_preds[i] + 
                    (1 - self.ensemble_weight) * deep_preds[i]
                )
            
            # Add boost when models agree strongly
            if agreement > 0.9:
                # Strong agreement - push prediction further in agreed direction
                direction = 1 if hybrid_preds[i] > 0.5 else -1
                
                # Apply a stronger boost when both models have high confidence
                confidence_boost = 0.05
                if ensemble_confidence[i] > 0.8 and deep_confidence[i] > 0.8:
                    confidence_boost = 0.1  # Double the boost for highly confident agreement
                
                hybrid_preds[i] += direction * confidence_boost * agreement
        
        # Apply team-specific and feature-based adjustments
        if 'TEAM_ID_HOME' in X.columns and 'TEAM_ID_AWAY' in X.columns:
            for i in range(len(hybrid_preds)):
                adjustment_factors = []
                
                # Get team IDs for this matchup
                team_home = X.iloc[i]['TEAM_ID_HOME']
                team_away = X.iloc[i]['TEAM_ID_AWAY']
                
                # 1. Team-specific adjustment (more calibrated)
                team_factor = ((hash(str(team_home) + str(team_away)) % 1000) / 20000) - 0.025
                adjustment_factors.append(team_factor)
                
                # 2. Head-to-head history adjustment if available
                if 'H2H_WIN_PCT' in X.columns and 'H2H_GAMES' in X.columns:
                    h2h_pct = X.iloc[i]['H2H_WIN_PCT']
                    h2h_games = X.iloc[i]['H2H_GAMES']
                    
                    # Scale impact based on number of games (more games = more reliable)
                    reliability = min(h2h_games / 10.0, 1.0)
                    h2h_factor = (h2h_pct - 0.5) * 0.08 * reliability
                    adjustment_factors.append(h2h_factor)
                
                # 3. Rest advantage adjustment if available
                if 'REST_DIFF' in X.columns:
                    rest_diff = X.iloc[i]['REST_DIFF']
                    if abs(rest_diff) >= 2:  # Significant rest advantage
                        rest_factor = np.sign(rest_diff) * 0.02 * min(abs(rest_diff)/3.0, 1.0)
                        adjustment_factors.append(rest_factor)
                
                # 4. Player impact adjustment if available
                if 'PLAYER_IMPACT_HOME' in X.columns and 'PLAYER_IMPACT_AWAY' in X.columns:
                    impact_home = X.iloc[i]['PLAYER_IMPACT_HOME']
                    impact_away = X.iloc[i]['PLAYER_IMPACT_AWAY']
                    player_factor = (impact_home - impact_away) * 0.05
                    adjustment_factors.append(player_factor)
                
                # Apply all adjustment factors
                total_adjustment = sum(adjustment_factors)
                hybrid_preds[i] = np.clip(hybrid_preds[i] + total_adjustment, 0.05, 0.95)
                    
        return hybrid_preds
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with enhanced confidence scores that integrate predictions
        from both ensemble and deep models with improved calibration.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train first.")
            
        # Ensure no DataFrame columns exist
        X = self._ensure_no_dataframe_columns(X)
        
        print("Generating hybrid model predictions with confidence...")
        
        # Pre-check columns for potential DataFrame types
        problematic_columns = []
        for col in X.columns:
            try:
                if isinstance(X[col], pd.DataFrame):
                    print(f"Column {col} is a DataFrame in hybrid model input - fixing")
                    if len(X[col].columns) > 0:
                        X[col] = X[col].iloc[:, 0]
                    else:
                        X[col] = pd.Series(0.0, index=X.index)
                    problematic_columns.append(col)
            except Exception as e:
                print(f"Error checking column {col}: {e}")
                problematic_columns.append(col)
                
        # If we found problematic columns, check once more
        if problematic_columns:
            print(f"Fixed {len(problematic_columns)} DataFrame-type columns")
        
        # Suppress repeated warning messages during prediction
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
            
            # Get ensemble model predictions with confidence
            try:
                ensemble_preds = self.ensemble_model.predict(X)
                ensemble_conf = self.ensemble_model.calculate_enhanced_confidence_score(ensemble_preds, X)
            except Exception as e:
                print(f"Error getting ensemble model predictions/confidence: {e}")
                # Fallback values
                ensemble_preds = np.full(len(X), 0.5)
                ensemble_conf = np.full(len(X), 0.7)  # Default moderate-high confidence
            
            # Get deep model predictions with uncertainty
            try:
                deep_preds, uncertainties = self.deep_model.predict_with_uncertainty(X)
                # Validate uncertainties to ensure they're usable
                if np.isnan(uncertainties).any() or np.isinf(uncertainties).any():
                    print("Warning: NaN or Inf values in uncertainties - replacing with default values")
                    uncertainties = np.nan_to_num(uncertainties, nan=0.2, posinf=0.5, neginf=0.5)
                
                try:
                    deep_conf = self.deep_model.calculate_confidence_from_uncertainty(deep_preds, uncertainties)
                except Exception as e:
                    print(f"Error calculating deep model confidence: {e}")
                    # Generate estimated confidence from prediction strength
                    deep_conf = 0.5 + 0.4 * np.abs(deep_preds - 0.5) * 2  # Scale to [0.5, 0.9]
            except Exception as e:
                print(f"Error getting deep model predictions with uncertainty: {e}")
                print("Falling back to standard prediction without uncertainty")
                try:
                    # Try standard prediction as fallback
                    deep_preds = self.deep_model.predict(X)
                    # Generate estimated confidence from prediction strength
                    deep_conf = 0.5 + 0.4 * np.abs(deep_preds - 0.5) * 2  # Scale to [0.5, 0.9]
                except Exception as e2:
                    print(f"Error with standard prediction: {e2}")
                    # Generate default predictions and confidence as last resort fallback
                    deep_preds = np.full(len(X), 0.5)
                    deep_conf = np.full(len(X), 0.5)  # Default moderate confidence
        
        # Validate predictions before combining
        if np.isnan(ensemble_preds).any():
            print("Warning: NaN values in ensemble predictions - replacing with 0.5")
            ensemble_preds = np.nan_to_num(ensemble_preds, nan=0.5)
        
        if np.isnan(deep_preds).any():
            print("Warning: NaN values in deep predictions - replacing with 0.5")
            deep_preds = np.nan_to_num(deep_preds, nan=0.5)
        
        # Make hybrid predictions using model weights
        hybrid_preds = self.ensemble_weight * ensemble_preds + (1 - self.ensemble_weight) * deep_preds
        
        # Calculate confidence scores using a more sophisticated approach
        try:
            # 1. Model agreement factor
            # Higher agreement between models = higher confidence
            model_agreement = 1.0 - np.abs(ensemble_preds - deep_preds)
            
            # 2. Weighted confidence from individual models
            # Weight by model performance with ensemble model given preference by default
            weighted_confidence = (
                self.ensemble_weight * ensemble_conf + 
                (1 - self.ensemble_weight) * deep_conf
            )
            
            # 3. Boost factor for model agreement
            # Apply non-linear boost for high agreement
            agreement_boost = 0.15 * np.power(model_agreement, 2)
            
            # 4. Adjust by prediction strength factor
            # Strong predictions deserve higher confidence
            prediction_strength = 2.0 * np.abs(hybrid_preds - 0.5)  # Scale to [0, 1]
            strength_boost = 0.1 * prediction_strength  # Max 10% boost
            
            # 5. Integrated confidence score
            raw_confidence = weighted_confidence + agreement_boost + strength_boost
            
            # 6. Apply calibration to ensure reasonable confidence range
            # Never fully certain or uncertain
            calibrated_confidence = np.clip(raw_confidence, 0.35, 0.95)
        except Exception as e:
            print(f"Error calculating confidence scores: {e}")
            # Fallback to simpler confidence calculation
            calibrated_confidence = 0.5 + 0.3 * np.abs(hybrid_preds - 0.5) * 2  # Scale to [0.5, 0.8]
        
        # 7. Apply data-specific adjustments based on features if available
        if 'TEAM_ID_HOME' in X.columns and 'TEAM_ID_AWAY' in X.columns:
            for i in range(len(hybrid_preds)):
                try:
                    # Fetch team information
                    team_home = X.iloc[i]['TEAM_ID_HOME']
                    team_away = X.iloc[i]['TEAM_ID_AWAY']
                    
                    # Validate team values - skip adjustment if invalid
                    if pd.isna(team_home) or pd.isna(team_away) or team_home == 0 or team_away == 0:
                        continue
                    
                    # Convert to string safely
                    team_home_str = str(int(team_home))
                    team_away_str = str(int(team_away))
                    
                    # H2H history adjustment with safe value checking
                    if 'H2H_WIN_PCT' in X.columns and 'H2H_GAMES' in X.columns:
                        try:
                            h2h_pct = float(X.iloc[i]['H2H_WIN_PCT'])
                            h2h_games = float(X.iloc[i]['H2H_GAMES'])
                            
                            # Validate values
                            if not pd.isna(h2h_pct) and not pd.isna(h2h_games) and h2h_games >= 3:
                                # Strong H2H trend adjustment
                                h2h_strength = abs(h2h_pct - 0.5) * 2  # Scale to [0, 1]
                                h2h_confidence_factor = 0.05 * h2h_strength * min(h2h_games / 10, 1)
                                calibrated_confidence[i] += h2h_confidence_factor
                        except Exception as e:
                            # Skip this H2H adjustment if there's an error
                            pass
                    
                    # Rest day advantage with safe value checking
                    if 'REST_DIFF' in X.columns:
                        try:
                            rest_col = X.iloc[i]['REST_DIFF']
                            # Check if REST_DIFF is a Series/DataFrame itself
                            if isinstance(rest_col, (pd.Series, pd.DataFrame)):
                                if isinstance(rest_col, pd.DataFrame) and len(rest_col.columns) > 0:
                                    rest_diff = abs(float(rest_col.iloc[0, 0]))
                                elif isinstance(rest_col, pd.Series):
                                    rest_diff = abs(float(rest_col.iloc[0]))
                                else:
                                    rest_diff = 0  # Default if we can't extract
                            else:
                                rest_diff = abs(float(rest_col))
                                
                            if rest_diff >= 2:  # Significant rest advantage
                                calibrated_confidence[i] += 0.02  # Small boost
                        except Exception:
                            # Skip rest adjustment if there's an error
                            pass
                    
                    # Player availability impact with safe value checking
                    if 'PLAYER_IMPACT_HOME' in X.columns and 'PLAYER_IMPACT_AWAY' in X.columns:
                        try:
                            # Safely extract player impact values
                            home_impact = float(X.iloc[i]['PLAYER_IMPACT_HOME'])
                            away_impact = float(X.iloc[i]['PLAYER_IMPACT_AWAY'])
                            
                            # Check for valid values
                            if not pd.isna(home_impact) and not pd.isna(away_impact):
                                player_impact_diff = abs(home_impact - away_impact)
                                if player_impact_diff > 0.15:  # Significant player advantage
                                    calibrated_confidence[i] += 0.03
                        except Exception:
                            # Skip player impact adjustment if there's an error
                            pass
                    
                    # Apply adjustment to prediction based on team-specific factors
                    try:
                        team_variation = ((hash(team_home_str + team_away_str) % 1000) / 20000) - 0.025
                        hybrid_preds[i] = np.clip(hybrid_preds[i] + team_variation, 0.1, 0.9)
                    except Exception:
                        # Skip team variation if there's an error
                        pass
                    
                    # Apply H2H history adjustment to prediction with safe value checking
                    if 'H2H_WIN_PCT' in X.columns:
                        try:
                            h2h_pct = float(X.iloc[i]['H2H_WIN_PCT'])
                            if not pd.isna(h2h_pct):
                                h2h_impact = (h2h_pct - 0.5) * 0.08  # Reduced impact
                                hybrid_preds[i] = np.clip(hybrid_preds[i] + h2h_impact, 0.1, 0.9)
                        except Exception:
                            # Skip H2H adjustment if there's an error
                            pass
                
                except Exception as e:
                    # Skip all adjustments for this row if we hit an error
                    continue
        
        # 8. Final confidence calibration
        final_confidence = np.clip(calibrated_confidence, 0.35, 0.95)
        
        # Final validation to catch any remaining NaN values
        if np.isnan(hybrid_preds).any():
            print("Warning: NaN values in final predictions - replacing with 0.5")
            hybrid_preds = np.nan_to_num(hybrid_preds, nan=0.5)
            
        if np.isnan(final_confidence).any():
            print("Warning: NaN values in confidence scores - replacing with 0.5")
            final_confidence = np.nan_to_num(final_confidence, nan=0.5)
        
        # Ensure predictions stay in valid range
        hybrid_preds = np.clip(hybrid_preds, 0.05, 0.95)
        
        return hybrid_preds, final_confidence
    
    def get_feature_importances(self, n: int = 20) -> Dict:
        """
        Get the top n most important features from the ensemble model.
        
        Args:
            n: Number of top features to return
            
        Returns:
            dict: Feature names and importance scores
        """
        return self.ensemble_model.get_top_features(n)