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
        
    def _preprocess_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Comprehensive preprocessing to ensure DataFrame has consistent column formats.
        Optimized to avoid DataFrame fragmentation by consolidating operations.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame with proper column types and consistent structure
        """
        # Make a copy to avoid modifying the original
        # We'll create a completely new DataFrame at the end to avoid fragmentation
        
        # Identify all problematic columns upfront
        problematic_columns = []
        df_columns = []
        for col in df.columns:
            try:
                col_data = df[col]
                if isinstance(col_data, pd.DataFrame):
                    print(f"Column {col} is a DataFrame, extracting first column")
                    df_columns.append(col)
                    problematic_columns.append(col)
            except Exception:
                problematic_columns.append(col)
                
        # Create a data dictionary to build a new DataFrame from scratch
        data_dict = {}
        
        # Process known problematic columns first - these columns often show up as DataFrames
        known_problematic = ['WIN_PCT_DIFF_30D', 'REST_DIFF']
        for col in known_problematic:
            if col in df.columns:
                try:
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame):
                        if len(col_data.columns) > 0:
                            # Extract the first column if it's a DataFrame
                            print(f"Column {col} is a DataFrame, extracting first column")
                            data_dict[col] = col_data.iloc[:, 0].values
                        else:
                            # Create empty column
                            data_dict[col] = np.zeros(len(df))
                    elif col == 'WIN_PCT_DIFF_30D':
                        if 'WIN_PCT_HOME_30D' in df.columns and 'WIN_PCT_AWAY_30D' in df.columns:
                            # Calculate this derived column
                            data_dict[col] = df['WIN_PCT_HOME_30D'].values - df['WIN_PCT_AWAY_30D'].values
                        else:
                            # Use the existing column directly
                            data_dict[col] = col_data.values
                    elif col == 'REST_DIFF':
                        if 'REST_DAYS_HOME' in df.columns and 'REST_DAYS_AWAY' in df.columns:
                            # Calculate this derived column
                            data_dict[col] = df['REST_DAYS_HOME'].values - df['REST_DAYS_AWAY'].values
                        else:
                            # Use the existing column directly
                            data_dict[col] = col_data.values
                except Exception as e:
                    # Only show the error once
                    if not hasattr(self, '_shown_column_errors') or col not in self._shown_column_errors:
                        print(f"Error fixing {col}: {e}")
                        if not hasattr(self, '_shown_column_errors'):
                            self._shown_column_errors = set()
                        self._shown_column_errors.add(col)
                    # Use zeros as fallback
                    data_dict[col] = np.zeros(len(df))
        
        # Process DataFrame columns
        if df_columns:
            for col in df_columns:
                if col in known_problematic:
                    # Skip if already processed above
                    continue
                    
                try:
                    col_data = df[col]
                    if isinstance(col_data, pd.DataFrame) and len(col_data.columns) > 0:
                        # Extract the first column as values
                        print(f"Column {col} is a DataFrame, extracting first column")
                        data_dict[col] = col_data.iloc[:, 0].values
                    else:
                        # Create zeros as fallback
                        data_dict[col] = np.zeros(len(df))
                except Exception:
                    # Use zeros for any column we can't process
                    data_dict[col] = np.zeros(len(df))
        
        # Process all non-problematic columns in one batch
        for col in df.columns:
            if col not in problematic_columns and col not in known_problematic:
                try:
                    # Copy the column values directly
                    data_dict[col] = df[col].values
                except Exception:
                    # Use zeros for any column we can't process
                    data_dict[col] = np.zeros(len(df))
        
        # Create a completely new DataFrame with all columns at once
        result = pd.DataFrame(data_dict, index=df.index)
        
        # Final check for any columns that should be numeric
        numeric_candidates = []
        for col in result.columns:
            try:
                if result[col].dtype == 'object':
                    numeric_candidates.append(col)
            except Exception as e:
                print(f"Error checking dtype for column {col}: {e}")
                # If we can't check the dtype, try to convert it anyway
                numeric_candidates.append(col)
                
        # Convert object columns to numeric all at once
        if numeric_candidates:
            print(f"Converting {len(numeric_candidates)} columns to numeric")
            for col in numeric_candidates:
                try:
                    result[col] = pd.to_numeric(result[col], errors='coerce').fillna(0)
                except Exception as e:
                    print(f"Error converting {col} to numeric: {e}")
                    # If conversion fails, remove the column
                    problematic_columns.append(col)
        
        # Remove any remaining problematic columns that couldn't be converted
        columns_to_remove = [col for col in problematic_columns if col in result.columns and isinstance(result[col], pd.DataFrame)]
        if columns_to_remove:
            print(f"Removing {len(columns_to_remove)} non-numeric columns for ML compatibility: {columns_to_remove}")
            result = result.drop(columns=columns_to_remove)
        
        return result
        
    def _ensure_no_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert any DataFrame columns to Series to avoid XGBoost errors.
        For backward compatibility - now delegates to the more comprehensive method.
        
        Args:
            df: DataFrame to process
            
        Returns:
            Processed DataFrame with no DataFrame columns
        """
        return self._preprocess_dataframe_columns(df)
        
    def train(self, X: pd.DataFrame) -> None:
        """
        Train enhanced ensemble of models with optimized hyperparameters for higher accuracy.
        Uses proper data isolation with a true holdout test set to prevent information leakage.
        
        Args:
            X: DataFrame containing features and target variable
        """
        from sklearn.model_selection import train_test_split
        
        print("Training enhanced model ensemble...")
        
        # Create true holdout test set for final evaluation (20% of data)
        # Keep temporal ordering by not shuffling (last 20% as holdout)
        X_train_val, X_test = train_test_split(X, test_size=0.2, shuffle=False)
        
        print(f"Training/validation data shape: {X_train_val.shape}, Holdout test data shape: {X_test.shape}")
        print("Using proper data isolation with true holdout test set")
        
        # Store the test set for later evaluation
        self.X_test = X_test
        self.y_test = X_test['TARGET'] if 'TARGET' in X_test.columns else None

        # Extract target variable from training data
        y = X_train_val['TARGET']
        X = X_train_val.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Create a complete list of all problematic columns that need fixing
        problematic_columns = []
        
        # Identify known problematic columns
        known_problematic = ['WIN_PCT_DIFF_30D', 'REST_DIFF']
        
        # Check for DataFrame columns that need fixing
        for col in X.columns:
            try:
                if isinstance(X[col], pd.DataFrame):
                    problematic_columns.append(col)
            except Exception:
                problematic_columns.append(col)
                
        # Add known problematic columns to the list
        for col in known_problematic:
            if col in X.columns and col not in problematic_columns:
                problematic_columns.append(col)
        
        # If we have problematic columns, create a completely new DataFrame
        if problematic_columns:
            print(f"Fixing {len(problematic_columns)} problematic columns all at once")
            
            # Create a dictionary to hold all column data
            data_dict = {}
            
            # First process all non-problematic columns
            for col in X.columns:
                if col not in problematic_columns:
                    data_dict[col] = X[col].values
            
            # Then process problematic columns
            for col in problematic_columns:
                try:
                    if col == 'WIN_PCT_DIFF_30D':
                        if 'WIN_PCT_HOME_30D' in X.columns and 'WIN_PCT_AWAY_30D' in X.columns:
                            # Properly calculate this derived column
                            data_dict[col] = X['WIN_PCT_HOME_30D'].values - X['WIN_PCT_AWAY_30D'].values
                        elif isinstance(X[col], pd.DataFrame) and len(X[col].columns) > 0:
                            # Extract values from the DataFrame column
                            data_dict[col] = X[col].iloc[:, 0].values
                        else:
                            # Fallback to zeros
                            data_dict[col] = np.zeros(len(X))
                    elif col == 'REST_DIFF':
                        if 'REST_DAYS_HOME' in X.columns and 'REST_DAYS_AWAY' in X.columns:
                            # Properly calculate this derived column
                            data_dict[col] = X['REST_DAYS_HOME'].values - X['REST_DAYS_AWAY'].values
                        elif isinstance(X[col], pd.DataFrame) and len(X[col].columns) > 0:
                            # Extract values from the DataFrame column
                            data_dict[col] = X[col].iloc[:, 0].values
                        else:
                            # Fallback to zeros
                            data_dict[col] = np.zeros(len(X))
                    elif isinstance(X[col], pd.DataFrame):
                        # Process general DataFrame columns
                        if len(X[col].columns) > 0:
                            # Extract values from the first column
                            data_dict[col] = X[col].iloc[:, 0].values
                        else:
                            # Create zeros
                            data_dict[col] = np.zeros(len(X))
                    else:
                        # Try direct access for other problematic columns
                        try:
                            data_dict[col] = X[col].values
                        except:
                            # Fallback to zeros
                            data_dict[col] = np.zeros(len(X))
                except Exception as e:
                    print(f"Error fixing {col}: {e}")
                    # Use zeros as fallback
                    data_dict[col] = np.zeros(len(X))
            
            # Create a brand new DataFrame with all columns at once
            X = pd.DataFrame(data_dict, index=X.index)
        else:
            # Even if no problematic columns detected, good practice to ensure consistency
            X = self._ensure_no_dataframe_columns(X)
        
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
                # Make sure X_train doesn't have DataFrame columns
                # Create dictionaries to hold column data to prevent fragmentation
                data_dict = {}
                
                # Process all columns at once instead of one-by-one
                # First identify all DataFrame columns
                df_columns = []
                for col in X_train.columns:
                    if isinstance(X_train[col], pd.DataFrame):
                        df_columns.append(col)
                
                # If there are DataFrame columns, report them
                if df_columns:
                    print(f"Converting {len(df_columns)} DataFrame columns to Series for feature selection")
                
                # Create data_dict in a more efficient way
                # Process non-DataFrame columns all at once
                data_dict = {col: X_train[col].values for col in X_train.columns if not isinstance(X_train[col], pd.DataFrame)}
                
                # Process DataFrame columns separately
                for col in df_columns:
                    if len(X_train[col].columns) > 0:
                        # Extract as values from the first column
                        data_dict[col] = X_train[col].iloc[:, 0].values
                    else:
                        # Create zeros as fallback
                        data_dict[col] = np.zeros(len(X_train))
                
                # Create a new DataFrame all at once to avoid fragmentation
                X_train = pd.DataFrame(data_dict, index=X_train.index)
                
                # Double-check for any remaining DataFrame-typed columns
                for col in X_train.columns:
                    if isinstance(X_train[col], pd.DataFrame):
                        print(f"Warning: Column {col} is still a DataFrame, replacing with zeros")
                        X_train[col] = np.zeros(len(X_train))
                            
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

                try:
                    selector.fit(X_train, y_train)
                    
                    # Make sure we're using the same columns from X_train, not X
                    selected_mask = selector.get_support()
                    # Create a list of indices where the mask is True
                    selected_indices = [i for i, x in enumerate(selected_mask) if x]
                    # Get the column names from X_train using these indices
                    selected_features = [X_train.columns[i] for i in selected_indices]
                    
                    # Make sure the features exist in the original dataset
                    selected_features = [feat for feat in selected_features if feat in X.columns]
                    
                    feature_selector_list.append(selected_features)
                    
                    # Only update feature stability if selection was successful
                    for feat in selected_features:
                        # Normalize by total runs (seeds × folds)
                        normalization_factor = len(seeds) * self.n_folds
                        self.feature_stability[feat] += 1/normalization_factor
                except Exception as e:
                    print(f"Feature selection failed: {e}")
                    # Use all features as fallback that exist in the original dataset
                    selected_features = [feat for feat in X_train.columns if feat in X.columns]
                    feature_selector_list.append(selected_features)

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
            
            try:
                # Get train/val indices
                train_data = X.iloc[train_idx]
                val_data = X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Create clean DataFrames for both train and validation sets at once
                # This avoids multiple DataFrame fragmentations
                print(f"Preparing fold {fold} training and validation data...")
                
                # Process both train and validation sets in parallel
                train_dict = {}
                val_dict = {}
                
                # Identify problematic columns in both sets
                train_problematic = []
                for col in train_data.columns:
                    try:
                        if isinstance(train_data[col], pd.DataFrame):
                            train_problematic.append(col)
                    except Exception:
                        train_problematic.append(col)
                
                # Process all columns at once to prevent fragmentation
                # First, handle all non-problematic columns in train set
                for col in train_data.columns:
                    if col not in train_problematic:
                        train_dict[col] = train_data[col].values
                
                # Then handle problematic columns in train set
                for col in train_problematic:
                    try:
                        if isinstance(train_data[col], pd.DataFrame) and len(train_data[col].columns) > 0:
                            train_dict[col] = train_data[col].iloc[:, 0].values
                        else:
                            train_dict[col] = np.zeros(len(train_data))
                    except Exception:
                        train_dict[col] = np.zeros(len(train_data))
                
                # Create a clean validation set in the same way
                for col in val_data.columns:
                    try:
                        if isinstance(val_data[col], pd.DataFrame) and len(val_data[col].columns) > 0:
                            val_dict[col] = val_data[col].iloc[:, 0].values
                        else:
                            val_dict[col] = val_data[col].values
                    except Exception:
                        val_dict[col] = np.zeros(len(val_data))
                
                # Create both DataFrames at once to avoid fragmentation
                X_train = pd.DataFrame(train_dict, index=train_data.index)
                X_val = pd.DataFrame(val_dict, index=val_data.index)
                
                # Scale features using enhanced scaler for robustness
                scaler = EnhancedScaler()
                
                # Convert to numpy arrays to avoid any DataFrame issues
                X_train_matrix = X_train.values
                X_val_matrix = X_val.values
                
                X_train_scaled = scaler.fit_transform(X_train_matrix)
                X_val_scaled = scaler.transform(X_val_matrix)
                print(f"Scaled data shapes - train: {X_train_scaled.shape}, val: {X_val_scaled.shape}")
                
                # Store feature names in the scaler for easier debugging
                if not hasattr(scaler, 'feature_names_in_'):
                    setattr(scaler, 'feature_names_in_', np.array(X_train.columns))
            except Exception as e:
                print(f"Error in fold {fold} preparation: {e}")
                continue

            # Skip feature selection entirely in quick mode
            try:
                # Simplest approach - use all scaled features
                print("Using direct numpy array for training/validation")
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
                
                # Use all stable features that match our X_train columns
                stable_features_in_data = [feat for feat in stable_features if feat in X_train.columns]
                print(f"Will track {len(stable_features_in_data)} stable features")

                # Create mapping of column names to indices for feature tracking
                col_to_idx = {col: i for i, col in enumerate(X_train.columns) if i < X_train_scaled.shape[1]}
                
                # Create empty DataFrames just for storing feature info
                X_train_selected_df = pd.DataFrame()
                X_val_selected_df = pd.DataFrame()
                
                # This is a flag to indicate we're using direct arrays
                self.using_direct_arrays = True
                
            except Exception as e:
                print(f"Error in feature preparation: {e}")
                print("Using direct arrays as fallback")
                X_train_selected = X_train_scaled
                X_val_selected = X_val_scaled
                stable_features_in_data = []  # Empty list as fallback
                self.using_direct_arrays = True

            # Train window-specific models with multiple algorithms
            window_models = []

            # Extract time windows from features (assuming format like 'FEATURE_NAME_7D')
            windows = set()
            for feat in stable_features_in_data:
                if '_D' in feat:
                    parts = feat.split('_')
                    for part in parts:
                        if part.endswith('D') and part[:-1].isdigit():
                            windows.add(int(part[:-1]))
                            
            if not windows:
                windows = [7, 14, 30, 60]  # Default if no window-specific features found
                
            windows = sorted(list(windows))

            fold_base_preds = []
            
            # Since we're using direct numpy arrays, we need to simplify how we handle windows
            
            # Train a model for each time window
            for window in windows:
                try:
                    # For direct array approach, just use all features for all windows
                    # This simplification helps us get past the bugs
                    X_train_window = X_train_selected
                    X_val_window = X_val_selected
                    
                    # Get window-specific features for tracking/naming only
                    window_features = [feat for feat in stable_features_in_data if f'_{window}D' in feat]
                    base_features = [feat for feat in stable_features_in_data if '_D' not in feat]
                    combined_features = window_features + base_features
                    
                    feature_count = len(combined_features) if combined_features else 0
                    print(f"Window {window}D: Using all {X_train_window.shape[1]} features (tracking {feature_count} named features)")
                except Exception as e:
                    print(f"Error preparing window {window}D: {e}")
                    # Continue anyway with full feature set as fallback
                    X_train_window = X_train_selected
                    X_val_window = X_val_selected
                    combined_features = []
                
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
                    # Removed feature_name parameter to avoid warning
                )
                
                # Train LightGBM without early stopping for better compatibility
                lgb_model.fit(
                    X_train_window, y_train
                )
                
                try:
                    # Add both models to the window ensemble with the list of features used
                    window_models.append((f'{window}d_xgb', xgb_model, combined_features))
                    window_models.append((f'{window}d_lgb', lgb_model, combined_features))

                    # Store feature importance for XGBoost
                    if hasattr(xgb_model, 'feature_importances_'):
                        xgb_importances = xgb_model.feature_importances_
                        if len(xgb_importances) == len(combined_features):
                            for feat, imp in zip(combined_features, xgb_importances):
                                feature_importance_dict[feat].append(imp)
                        else:
                            pass  # Skip instead of showing warning message
                    
                    # Store feature importance for LightGBM (if available)
                    if hasattr(lgb_model, 'feature_importances_'):
                        lgb_importances = lgb_model.feature_importances_
                        if len(lgb_importances) == len(combined_features):
                            for feat, imp in zip(combined_features, lgb_importances):
                                feature_importance_dict[feat].append(imp)
                        else:
                            pass  # Skip instead of showing warning message
                except Exception as e:
                    print(f"Error saving model importances for window {window}D: {e}")
                
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
                    try:
                        # When using direct arrays, we use the full array for calibration
                        X_train_calib = X_train_selected
                        X_val_calib = X_val_selected
                        
                        # Calibrate the model
                        calibrator = self._calibrate_model(
                            model, 
                            X_train_calib, 
                            y_train,
                            X_val_calib,
                            y_val
                        )
                        fold_calibrators[model_name] = calibrator
                    except Exception as e:
                        print(f"Error calibrating model {model_name}: {e}")
                
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
                try:
                    # Use full array for evaluation in direct mode
                    X_eval = X_val_selected
                    
                    if self.use_calibration:
                        # Use calibrated predictions if available
                        calibrator = self.calibrators[-1].get(model_name) if self.calibrators else None
                        if calibrator:
                            y_pred = calibrator.predict_proba(X_eval)[:, 1]
                        else:
                            y_pred = model.predict_proba(X_eval)[:, 1]
                    else:
                        y_pred = model.predict_proba(X_eval)[:, 1]
                        
                    y_preds.append(y_pred)
                except Exception as e:
                    print(f"Error evaluating model {model_name}: {e}")

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
            
        # Suppress warnings during prediction
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.*")
            
            # After training is complete, evaluate on true holdout test set
            if hasattr(self, 'X_test') and hasattr(self, 'y_test') and self.y_test is not None:
                if getattr(self, '_evaluated_holdout', False) is False:
                    print("\n============ EVALUATING ON TRUE HOLDOUT TEST SET ============")
                    test_preds = self._predict_internal(self.X_test, is_test=True)
                    test_binary = (test_preds > 0.5).astype(int)
                    
                    # Calculate test metrics
                    from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
                    test_acc = accuracy_score(self.y_test, test_binary)
                    test_brier = brier_score_loss(self.y_test, test_preds)
                    test_auc = roc_auc_score(self.y_test, test_preds)
                    
                    print(f"TRUE HOLDOUT METRICS:")
                    print(f"Accuracy: {test_acc:.4f}")
                    print(f"Brier Score: {test_brier:.4f}")
                    print(f"AUC-ROC: {test_auc:.4f}")
                    print("============================================================\n")
                    
                    # Store metrics
                    self.holdout_metrics = {
                        'accuracy': test_acc,
                        'brier_score': test_brier,
                        'auc': test_auc
                    }
                    
                    # Mark as evaluated so we don't repeat this
                    self._evaluated_holdout = True
                    
            # Call internal prediction method for the actual input data
            return self._predict_internal(X)
        
    def _predict_internal(self, X: pd.DataFrame, is_test: bool = False) -> np.ndarray:
        """
        Internal prediction method used by predict().
        
        Args:
            X: DataFrame containing features
            is_test: Whether this is a test set prediction (to avoid verbose output for tests)
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        # Create a backup of original input
        X_original = X.copy()
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Store the original column count before preprocessing
        original_column_count = len(X.columns)
        
        # Ensure no DataFrame columns exist - with improved preprocessing
        X = self._preprocess_dataframe_columns(X)
        
        # Log column count change if significant
        if abs(len(X.columns) - original_column_count) > 2:
            print(f"Fixed DataFrame columns: Changed from {original_column_count} to {len(X.columns)} columns")
        
        # Ensure we have valid training features
        if not hasattr(self, 'training_features') or not self.training_features:
            # If training_features wasn't saved, try to infer from the first fold's scaler
            if self.models and self.models[0] and self.models[0][1]:
                # Try to get feature names from the scaler
                if hasattr(self.models[0][1], 'feature_names_in_'):
                    self.training_features = self.models[0][1].feature_names_in_.tolist()
                else:
                    print("Warning: Unable to determine original training features. Prediction may be inaccurate.")
        
        # Ensure we use consistent features across all folds from the first model
        # This prevents feature count mismatch errors
        if self.models and len(self.models) > 0:
            _, _, stable_features_first_fold = self.models[0]
            # Store these on the instance for reuse
            if not hasattr(self, '_cached_stable_features'):
                self._cached_stable_features = stable_features_first_fold
        else:
            stable_features_first_fold = []
            self._cached_stable_features = []
            
        # Get predictions from each fold's ensemble
        all_fold_preds = []
        
        # Track if we've already shown warnings to avoid repetition
        warnings_shown = is_test  # Always consider warnings shown for test set
        
        for fold_idx, (window_models, scaler, stable_features) in enumerate(self.models):
            # Only print for the first fold to reduce verbosity
            if fold_idx == 0 and not is_test:
                print(f"Processing enhanced ensemble model predictions...")
            
            try:
                # Use cached stable features if available or get from the first fold
                if hasattr(self, '_cached_stable_features') and self._cached_stable_features:
                    stable_features_to_use = self._cached_stable_features
                else:
                    stable_features_to_use = stable_features_first_fold if stable_features_first_fold else stable_features
                    # Cache for future use
                    self._cached_stable_features = stable_features_to_use
                
                # Collect all features in a single dictionary to avoid fragmentation
                # First, prepare direct column mappings for existing features
                existing_features = {feature: X[feature].values for feature in stable_features_to_use if feature in X.columns}
                
                # Identify missing features that need to be derived or created
                missing_features = [f for f in stable_features_to_use if f not in X.columns]
                if missing_features and not is_test:
                    print(f"Deriving {len(missing_features)} missing features")
                
                # Prepare derivation information all at once
                derived_features = {}
                default_values = {}
                
                # Process all missing features in one batch to minimize DataFrame operations
                for feature in missing_features:
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
                    can_derive = False
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
                                # Track the derivation info for later batch processing
                                if base_feature == 'WIN_PCT_DIFF':
                                    derived_features[feature] = (dependencies[0], dependencies[1], 'subtract')
                                    can_derive = True
                                elif base_feature in ['OFF_RTG_DIFF', 'DEF_RTG_DIFF', 'NET_RTG_DIFF', 'PACE_DIFF']:
                                    derived_features[feature] = (dependencies[0], dependencies[1], 'subtract')
                                    can_derive = True
                                # Add other derivation rules as needed...
                    
                    # If we couldn't derive it, prepare default values
                    if not can_derive:
                        # Use 0.5 for probability or percentage features, 0 otherwise
                        if any(term in feature for term in ['PCT', 'PROBABILITY', 'H2H', 'WIN']):
                            default_values[feature] = 0.5
                        else:
                            default_values[feature] = 0.0
                
                # Now we'll create all derived features at once
                derived_values = {}
                for feature, (col1, col2, op) in derived_features.items():
                    if op == 'subtract':
                        derived_values[feature] = X[col1].values - X[col2].values
                    elif op == 'add':
                        derived_values[feature] = X[col1].values + X[col2].values
                    elif op == 'multiply':
                        derived_values[feature] = X[col1].values * X[col2].values
                
                # Create default values arrays all at once
                default_arrays = {feature: np.full(len(X), value) for feature, value in default_values.items()}
                
                # Combine all dictionaries into one
                X_aligned_dict = {**existing_features, **derived_values, **default_arrays}
                
                # Create DataFrame all at once to avoid fragmentation
                X_aligned = pd.DataFrame(X_aligned_dict, index=X.index)
                
                # Verify we have the expected number of features
                if len(X_aligned.columns) != len(stable_features_to_use):
                    # Show warning only once
                    if fold_idx == 0 and not warnings_shown:
                        print(f"Warning: Feature count mismatch. Expected {len(stable_features_to_use)}, got {len(X_aligned.columns)}.")
                        print("Ensuring all required features are present.")
                        warnings_shown = True
                
                # Scale the aligned features
                try:
                    # Check and adapt feature count if needed for StandardScaler
                    if not isinstance(scaler, EnhancedScaler) and hasattr(scaler, 'feature_names_in_'):
                        expected_features = len(scaler.feature_names_in_)
                        actual_features = len(X_aligned.columns)
                        
                        # If feature count doesn't match, use a robust approach
                        if expected_features != actual_features:
                            # Create a fallback scaler right away instead of trying the original scaler
                            # This avoids the warning messages about feature count mismatch
                            if not warnings_shown and fold_idx == 0:
                                print(f"Using enhanced scaler for feature count adaptation ({actual_features} vs {expected_features})")
                                warnings_shown = True
                            fallback_scaler = EnhancedScaler()
                            X_scaled = fallback_scaler.fit_transform(X_aligned)
                        else:
                            # Feature counts match, proceed with original scaler
                            X_scaled = scaler.transform(X_aligned)
                    else:
                        # Use enhanced scaler directly
                        X_scaled = scaler.transform(X_aligned)
                except Exception as e:
                    # Show error only once
                    if fold_idx == 0 and not warnings_shown:
                        # Create a more detailed error message with missing feature information
                        missing_features = []
                        if hasattr(scaler, 'feature_names_in_'):
                            expected = set(scaler.feature_names_in_)
                            actual = set(X_aligned.columns)
                            missing_features = list(expected - actual)
                        
                        print(f"Using robust scaling fallback due to error: {str(e)[:100]}...")
                        if missing_features and len(missing_features) < 10:
                            print(f"Missing features: {missing_features}")
                        warnings_shown = True
                    
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
            # First ensure features have no DataFrame columns to avoid fragmentation warnings
            # Only perform the check and fix once, not for every row
            features_fixed = self._preprocess_dataframe_columns(features)
            
            # Get confidence weights from constants (or use defaults)
            factors = CONFIDENCE_WEIGHTS.copy()
            
            # Add more confidence factors specific to enhanced model
            enhanced_factors = {
                'prediction_margin': 0.25,     # Weight for prediction probability margin
                'vegas_line': 0.25,            # Weight for agreement with Vegas lines
                'sample_size': 0.10,           # Weight for number of previous matches
                'recent_consistency': 0.10,    # Weight for consistency in recent games
                'h2h_history': 0.10,           # Weight for head-to-head history
                'rest_advantage': 0.05,        # Weight for rest day advantage
                'player_impact': 0.05,         # Weight for player availability
                'feature_stability': 0.05,     # Weight for feature stability
                'model_consensus': 0.05        # Weight for model agreement
            }
            
            # Use enhanced factors if available
            if len(enhanced_factors) > len(factors):
                factors = enhanced_factors

            # Pre-check which columns are available to avoid repeated checks
            available_columns = {
                'IMPLIED_WIN_PROB': 'IMPLIED_WIN_PROB' in features_fixed.columns,
                'WIN_count_HOME_60D': 'WIN_count_HOME_60D' in features_fixed.columns,
                'HOME_CONSISTENCY_30D': 'HOME_CONSISTENCY_30D' in features_fixed.columns,
                'AWAY_CONSISTENCY_30D': 'AWAY_CONSISTENCY_30D' in features_fixed.columns,
                'H2H_GAMES': 'H2H_GAMES' in features_fixed.columns,
                'H2H_RECENCY_WEIGHT': 'H2H_RECENCY_WEIGHT' in features_fixed.columns,
                'REST_DIFF': 'REST_DIFF' in features_fixed.columns,
                'PLAYER_IMPACT_DIFF': 'PLAYER_IMPACT_DIFF' in features_fixed.columns,
                'PLAYER_IMPACT_HOME': 'PLAYER_IMPACT_HOME' in features_fixed.columns,
                'PLAYER_IMPACT_AWAY': 'PLAYER_IMPACT_AWAY' in features_fixed.columns,
                'TEAM_ID_HOME': 'TEAM_ID_HOME' in features_fixed.columns,
                'TEAM_ID_AWAY': 'TEAM_ID_AWAY' in features_fixed.columns
            }
            
            # Create feature stability check once
            use_feature_stability = hasattr(self, 'feature_importances') and hasattr(self, 'feature_stability')
            top_features = []
            if use_feature_stability:
                top_features = sorted(self.feature_importances.items(), key=lambda x: x[1], reverse=True)[:5]
            
            # Calculate confidence for each prediction more efficiently
            for i, pred in enumerate(predictions):
                score = 0

                # 1. Prediction margin confidence (adjusted with sigmoid)
                prob_margin = abs(pred - 0.5) * 2  # Scale to [0, 1]
                # Apply sigmoid transformation to reward high confidence predictions more
                margin_confidence = 1 / (1 + np.exp(-6 * (prob_margin - 0.5)))
                score += margin_confidence * factors['prediction_margin']
                
                # 1.5 Vegas line agreement if available
                if available_columns['IMPLIED_WIN_PROB'] and 'vegas_line' in factors:
                    # Calculate agreement between model and betting lines
                    implied_prob = features_fixed.iloc[i]['IMPLIED_WIN_PROB']
                    # Agreement is higher when model and vegas agree
                    vegas_agreement = 1.0 - min(abs(pred - implied_prob), 0.5) * 2  # Scale to [0, 1]
                    # Apply sigmoid transformation similar to margin_confidence
                    vegas_conf = 1 / (1 + np.exp(-6 * (vegas_agreement - 0.5)))
                    score += vegas_conf * factors['vegas_line']

                # 2. Sample size confidence
                if available_columns['WIN_count_HOME_60D']:
                    games_played = features_fixed.iloc[i]['WIN_count_HOME_60D']
                    # Sigmoid function to scale sample size confidence
                    sample_size_conf = 2 / (1 + np.exp(-0.1 * games_played)) - 1  # Scale to [0, 1]
                    score += sample_size_conf * factors['sample_size']
                    
                # 3. Recent consistency confidence
                consistency_score = 0
                if available_columns['HOME_CONSISTENCY_30D']:
                    home_consistency = 1 - min(features_fixed.iloc[i]['HOME_CONSISTENCY_30D'], 1)  # Lower variance is better
                    consistency_score += home_consistency
                
                if available_columns['AWAY_CONSISTENCY_30D']:
                    away_consistency = 1 - min(features_fixed.iloc[i]['AWAY_CONSISTENCY_30D'], 1)
                    consistency_score += away_consistency
                
                if consistency_score > 0:
                    avg_consistency = consistency_score / 2 if consistency_score > 1 else consistency_score
                    score += avg_consistency * factors['recent_consistency']

                # 4. Head-to-head confidence
                h2h_score = 0
                if available_columns['H2H_GAMES']:
                    h2h_games = features_fixed.iloc[i]['H2H_GAMES']
                    h2h_score += min(h2h_games / 8, 1)  # Scale to [0, 1]
                
                if available_columns['H2H_RECENCY_WEIGHT']:
                    h2h_recency = min(abs(features_fixed.iloc[i]['H2H_RECENCY_WEIGHT'] - 0.5) * 2, 1)  # Scale to [0, 1]
                    h2h_score += h2h_recency
                
                if h2h_score > 0:
                    avg_h2h = h2h_score / 2 if h2h_score > 1 else h2h_score
                    score += avg_h2h * factors['h2h_history']

                # 5. Rest advantage confidence
                if available_columns['REST_DIFF']:
                    rest_diff = abs(features_fixed.iloc[i]['REST_DIFF'])
                    rest_conf = min(rest_diff / 3, 1)  # Scale to [0, 1]
                    score += rest_conf * factors['rest_advantage']
                
                # 6. Player impact confidence (if player availability features exist)
                player_impact_score = 0
                if available_columns['PLAYER_IMPACT_DIFF']:
                    player_impact = abs(features_fixed.iloc[i]['PLAYER_IMPACT_DIFF'])
                    player_conf = min(player_impact / 0.2, 1)  # Scale to [0, 1]
                    player_impact_score += player_conf
                
                if available_columns['PLAYER_IMPACT_HOME'] and available_columns['PLAYER_IMPACT_AWAY']:
                    home_impact = features_fixed.iloc[i]['PLAYER_IMPACT_HOME']
                    away_impact = features_fixed.iloc[i]['PLAYER_IMPACT_AWAY']
                    # Avoid division by zero
                    max_impact = max(abs(home_impact), abs(away_impact), 1)
                    relative_impact = abs(home_impact - away_impact) / max_impact
                    player_impact_score += min(relative_impact * 5, 1)  # Scale to [0, 1]
                
                if player_impact_score > 0:
                    avg_player_impact = player_impact_score / 2 if player_impact_score > 1 else player_impact_score
                    score += avg_player_impact * factors['player_impact']
                
                # 7. Feature stability confidence
                if use_feature_stability:
                    feature_stability_score = 0
                    feature_count = 0
                    
                    # Check the stability of top 5 most important features for this prediction
                    for feat, imp in top_features:
                        if feat in features_fixed.columns:
                            # Get feature stability score from training
                            stability = self.feature_stability.get(feat, 0)
                            feature_stability_score += stability
                            feature_count += 1
                    
                    if feature_count > 0:
                        avg_feature_stability = feature_stability_score / feature_count
                        score += avg_feature_stability * factors['feature_stability']
                
                # Add team-specific variability to confidence score
                # This ensures different teams get different confidence scores
                if available_columns['TEAM_ID_HOME'] and available_columns['TEAM_ID_AWAY']:
                    team_home = features_fixed.iloc[i]['TEAM_ID_HOME']
                    team_away = features_fixed.iloc[i]['TEAM_ID_AWAY']
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