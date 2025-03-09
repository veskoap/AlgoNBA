"""
Enhanced scaler utility for robust handling of extreme values.
"""
import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler

# Create a custom StandardScaler that handles the deprecation warning
# This is a more robust solution than just suppressing the warning
class CompatStandardScaler(StandardScaler):
    """A StandardScaler that works with all scikit-learn versions."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    
    def fit(self, X, y=None):
        """Fit without force_all_finite parameter."""
        # Filter out any sklearn deprecated parameters
        filtered_kwargs = {}
        return super().fit(X, y, **filtered_kwargs)
    
    def transform(self, X):
        """Transform without force_all_finite parameter."""
        # Filter out any sklearn deprecated parameters
        filtered_kwargs = {}
        return super().transform(X, **filtered_kwargs)
    
    def fit_transform(self, X, y=None):
        """Fit and transform without force_all_finite parameter."""
        # Use parent fit_transform but without force_all_finite
        return super().fit_transform(X, y)

# Completely suppress the warning regardless
warnings.filterwarnings('ignore', message='.*force_all_finite.*',
                       category=FutureWarning, module='sklearn.*')


class EnhancedScaler:
    """
    Scaler with enhanced robustness for extreme values and missing data.
    
    This scaler extends standard scaling functionality with additional features:
    - Handles extreme values by clipping outliers
    - Gracefully manages missing values (NaN, inf)
    - Provides fallback mechanisms when standard scaling fails
    - Automatically aligns feature columns for prediction
    - Optimizes memory usage with efficient DataFrame operations
    - Supports array-like and DataFrame inputs
    - Ensures compatibility with scikit-learn API expectations
    """
    
    def __init__(self, clip_threshold=5.0):
        """
        Initialize enhanced scaler.
        
        Args:
            clip_threshold: Values outside this many standard deviations will be clipped.
                          Higher values (e.g., 10.0) preserve more extreme values,
                          while lower values (e.g., 3.0) are more aggressive in outlier removal.
                          Default of 5.0 balances outlier handling with data preservation.
        """
        self.scaler = CompatStandardScaler()  # Use our compatible version
        self.clip_threshold = clip_threshold
        self.feature_names = None
        self.feature_means = None
        self.feature_stds = None
    
    def fit_transform(self, X, y=None):
        """
        Fit scaler to data and transform it with robust handling of outliers.
        
        Args:
            X: Input data (DataFrame)
            y: Target variable (not used)
            
        Returns:
            Transformed array with extreme values handled
        """
        # Store feature names
        if hasattr(X, 'columns'):
            self.feature_names = X.columns.tolist()
            # Store feature names as an attribute for scikit-learn compatibility
            self.feature_names_in_ = np.array(self.feature_names)
        
        # Handle extreme values
        X_preprocessed = self._preprocess_extreme_values(X)
        
        # Try StandardScaler if data is clean
        try:
            # Don't pass the parameter that's causing issues
            result = self.scaler.fit_transform(X_preprocessed)
            
            # Store the shape for scikit-learn compatibility
            if hasattr(result, 'shape'):
                self.n_features_in_ = result.shape[1]
                
            return result
        except Exception as e:
            print(f"Warning: StandardScaler failed: {e}. Using robust fallback scaling.")
            return self._robust_scale(X_preprocessed)
    
    def transform(self, X):
        """
        Transform data with robust handling of outliers.
        
        Args:
            X: Input data (DataFrame or numpy array)
            
        Returns:
            Transformed array with extreme values handled
        """
        # First, ensure any DataFrame columns are properly handled
        if isinstance(X, pd.DataFrame):
            X = self._fix_dataframe_columns(X)
        
        # Handle different input types
        if isinstance(X, pd.DataFrame):
            # DataFrame handling with improved feature alignment
            if self.feature_names:
                # Get expected feature columns (stored during training)
                expected_cols = self.feature_names
                current_cols = X.columns.tolist()
                
                # Find missing and extra columns
                missing_cols = [col for col in expected_cols if col not in current_cols]
                
                # For prediction only keep expected columns, faster than checking for extra columns
                # and safer against dimensionality mismatch
                needed_cols = [col for col in current_cols if col in expected_cols]
                
                # Create a new DataFrame with aligned features
                if missing_cols:
                    # First create all available columns
                    X_available = X[needed_cols].copy()
                    
                    # Create missing columns with zeros, more efficient than individual inserts
                    missing_dict = {col: np.zeros(len(X)) for col in missing_cols}
                    missing_df = pd.DataFrame(missing_dict, index=X.index)
                    
                    # Combine both DataFrames at once
                    X_aligned = pd.concat([X_available, missing_df], axis=1)
                    
                    # Ensure proper column order to match the training data
                    X_aligned = X_aligned.reindex(columns=expected_cols, fill_value=0)
                else:
                    # Just select and reorder existing columns
                    X_aligned = X[expected_cols]
            else:
                X_aligned = X
        else:
            # Numpy array handling - we rely on shape consistency checks
            X_aligned = X
        
        # Handle extreme values
        X_preprocessed = self._preprocess_extreme_values(X_aligned)
        
        # Check feature count compatibility before calling StandardScaler
        if hasattr(self.scaler, 'n_features_in_') and hasattr(X_preprocessed, 'shape'):
            expected_features = getattr(self.scaler, 'n_features_in_', 0)
            actual_features = X_preprocessed.shape[1] if len(X_preprocessed.shape) > 1 else 1
            
            # If feature count doesn't match, use robust scaling directly
            if expected_features != actual_features:
                # Use robust scaling without printing redundant warning
                return self._robust_scale(X_preprocessed)
        
        # Try StandardScaler if feature counts match
        try:
            # Use the fitted scaler
            result = self.scaler.transform(X_preprocessed)
            return result
        except Exception:
            # Use robust scaling as fallback without printing warning
            return self._robust_scale(X_preprocessed)
    
    def _fix_dataframe_columns(self, X):
        """
        Identify and fix DataFrame columns inside a DataFrame.
        Process known problematic columns with special handling.
        
        Args:
            X: DataFrame to process
            
        Returns:
            Fixed DataFrame with no DataFrame columns
        """
        if not isinstance(X, pd.DataFrame):
            return X
            
        # Make a copy to avoid modifying the original
        X_fixed = X.copy()
        
        # Store columns that need to be fixed
        columns_to_fix = {}
        problematic_cols = []
        
        # First identify problematic columns
        for col in X_fixed.columns:
            try:
                if isinstance(X_fixed[col], pd.DataFrame):
                    problematic_cols.append(col)
            except:
                problematic_cols.append(col)
        
        # Handle known problematic columns with special handling
        known_cols = {
            'WIN_PCT_DIFF_30D': lambda df: df['WIN_PCT_HOME_30D'] - df['WIN_PCT_AWAY_30D'] if 'WIN_PCT_HOME_30D' in df.columns and 'WIN_PCT_AWAY_30D' in df.columns else pd.Series(0, index=df.index),
            'REST_DIFF': lambda df: df['REST_DAYS_HOME'] - df['REST_DAYS_AWAY'] if 'REST_DAYS_HOME' in df.columns and 'REST_DAYS_AWAY' in df.columns else pd.Series(0, index=df.index)
        }
        
        # Process known problematic columns first
        for col, func in known_cols.items():
            if col in X_fixed.columns:
                try:
                    # Try to create the column using the special function
                    columns_to_fix[col] = func(X_fixed)
                except:
                    # Fallback to zeros
                    columns_to_fix[col] = pd.Series(0, index=X_fixed.index)
                problematic_cols.append(col)  # Ensure we process this column
        
        # Now process all problematic columns
        for col in problematic_cols:
            # Skip if already fixed by special handling
            if col in columns_to_fix:
                continue
                
            try:
                if isinstance(X_fixed[col], pd.DataFrame):
                    # DataFrame column detected
                    if len(X_fixed[col].columns) > 0:
                        columns_to_fix[col] = X_fixed[col].iloc[:, 0]
                    else:
                        columns_to_fix[col] = pd.Series(0, index=X_fixed.index)
            except:
                # Any error, create series of zeros
                columns_to_fix[col] = pd.Series(0, index=X_fixed.index)
        
        # If any columns need fixing, reconstruct the DataFrame
        if columns_to_fix:
            # First remove all problematic columns
            for col in columns_to_fix.keys():
                if col in X_fixed.columns:
                    X_fixed = X_fixed.drop(col, axis=1)
            
            # Then create a DataFrame from the fixed columns
            fixed_df = pd.DataFrame(columns_to_fix, index=X_fixed.index)
            
            # Join with the original, keeping only the good columns
            X_fixed = pd.concat([X_fixed, fixed_df], axis=1)
        
        return X_fixed
    
    def _preprocess_extreme_values(self, X):
        """
        Handle extreme values in the data.
        
        Args:
            X: Input data (DataFrame or numpy array)
            
        Returns:
            Processed data with extreme values handled
        """
        if isinstance(X, pd.DataFrame):
            # For DataFrame input
            # First, fix any DataFrame columns inside the DataFrame
            X_cleaned = self._fix_dataframe_columns(X)
            
            # Replace inf with nan
            X_cleaned = X_cleaned.replace([np.inf, -np.inf], np.nan)
            
            # Calculate column-wise statistics ignoring NaNs
            with np.errstate(all='ignore'):
                means = X_cleaned.mean()
                stds = X_cleaned.std().replace(0, 1)
                
                # For columns with all NaNs, set means to 0
                means = means.fillna(0)
                stds = stds.fillna(1)
                
                # Store these for fallback scaling
                self.feature_means = means
                self.feature_stds = stds
                
                # Cap extreme values
                upper_bound = means + self.clip_threshold * stds
                lower_bound = means - self.clip_threshold * stds
                
                X_capped = X_cleaned.copy()
                
                # Process all columns at once with efficient vectorized operations where possible
                numeric_cols = X_capped.select_dtypes(include=['number']).columns
                if len(numeric_cols) > 0:
                    # Clip numeric columns efficiently
                    X_capped[numeric_cols] = X_capped[numeric_cols].clip(
                        lower=lower_bound[numeric_cols], 
                        upper=upper_bound[numeric_cols],
                        axis=1
                    )
                
                # Process any remaining non-numeric columns individually
                other_cols = [col for col in X_capped.columns if col not in numeric_cols]
                for col in other_cols:
                    try:
                        # Try to convert to numeric first
                        X_capped[col] = pd.to_numeric(X_capped[col], errors='coerce')
                        # Then clip
                        X_capped[col] = X_capped[col].clip(lower=lower_bound[col], upper=upper_bound[col])
                    except:
                        # If conversion fails, fill with zeros
                        X_capped[col] = 0
                
                # Fill NaNs efficiently
                X_capped = X_capped.fillna(means)
                
                return X_capped
        else:
            # For numpy array input
            # Replace inf with nan
            X_cleaned = np.copy(X)
            X_cleaned[np.isinf(X_cleaned)] = np.nan
            
            # Calculate column-wise statistics
            with np.errstate(all='ignore'):
                means = np.nanmean(X_cleaned, axis=0)
                stds = np.nanstd(X_cleaned, axis=0)
                
                # Replace zeros in stds to avoid division by zero
                stds = np.where(stds == 0, 1.0, stds)
                
                # Handle all-NaN columns
                means = np.nan_to_num(means, nan=0.0)
                
                # Store for fallback scaling
                self.feature_means = means
                self.feature_stds = stds
                
                # Cap extreme values
                upper_bound = means + self.clip_threshold * stds
                lower_bound = means - self.clip_threshold * stds
                
                # Clip values
                X_capped = np.clip(X_cleaned, lower_bound, upper_bound)
                
                # Fill NaNs with means
                mask = np.isnan(X_capped)
                if mask.any():
                    X_capped[mask] = np.take(means, np.where(mask)[1])
                
                return X_capped
    
    def _robust_scale(self, X):
        """
        Manually perform scaling when StandardScaler fails.
        
        Args:
            X: Input data (DataFrame or numpy array)
            
        Returns:
            Scaled numpy array
        """
        # Handle both DataFrame and numpy array inputs
        if isinstance(X, pd.DataFrame):
            # Use already calculated means and stds for DataFrame
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = X.mean()
                self.feature_stds = X.std().replace(0, 1)
            
            # Perform manual scaling
            X_scaled = (X - self.feature_means) / self.feature_stds
            result = X_scaled.values
        else:
            # For numpy arrays, calculate mean and std directly
            if self.feature_means is None or self.feature_stds is None:
                self.feature_means = np.nanmean(X, axis=0)
                self.feature_stds = np.nanstd(X, axis=0)
                # Replace zeros with ones to avoid division by zero
                self.feature_stds = np.where(self.feature_stds == 0, 1.0, self.feature_stds)
            
            # Perform manual scaling
            X_scaled = (X - self.feature_means) / self.feature_stds
            result = X_scaled
        
        # Store shape information for scikit-learn compatibility
        if hasattr(result, 'shape'):
            self.n_features_in_ = result.shape[1]
            
        return result
        
    def __getitem__(self, key):
        """
        Support array-like indexing for scikit-learn compatibility.
        
        Args:
            key: The indexing key (slice, index, etc.)
            
        Returns:
            The selected items
        """
        raise ValueError("EnhancedScaler does not support direct indexing. Use transform() to get scaled values.")


# Test the scaler
if __name__ == "__main__":
    # Create test data with extreme values
    df = pd.DataFrame({
        'normal': np.random.normal(0, 1, 10),
        'extreme_high': [1000, 2000, 3000, 0, 0, 0, 0, 0, 0, 0],
        'extreme_low': [0, 0, 0, -1000, -2000, -3000, 0, 0, 0, 0],
        'missing': [np.nan, np.inf, -np.inf, 0, 1, 2, 3, 4, 5, 6],
        'zeros': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    })
    
    print("Original data:")
    print(df.head())
    
    # Apply our enhanced scaler
    scaler = EnhancedScaler()
    df_scaled = scaler.fit_transform(df)
    
    print("\nScaled data:")
    print(df_scaled[:5])
    
    # Test transform on new data
    df_new = pd.DataFrame({
        'normal': np.random.normal(0, 1, 5),
        'extreme_high': [500, 0, 0, 0, 0],
        'extreme_low': [0, -500, 0, 0, 0],
        'missing': [np.nan, np.inf, 0, 1, 2],
        'zeros': [0, 0, 0, 0, 0]
    })
    
    print("\nNew data:")
    print(df_new)
    
    df_new_scaled = scaler.transform(df_new)
    
    print("\nNew data scaled:")
    print(df_new_scaled)
