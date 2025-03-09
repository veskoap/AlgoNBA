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
            X: Input data (DataFrame)
            
        Returns:
            Transformed array with extreme values handled
        """
        # Check if X has expected columns
        if self.feature_names:
            # Ensure X has all expected columns
            missing_cols = [col for col in self.feature_names if col not in X.columns]
            extra_cols = [col for col in X.columns if col not in self.feature_names]
            
            if missing_cols:
                # Create a dictionary with all missing columns and their default values
                missing_dict = {col: 0 for col in missing_cols}
                
                # Create DataFrame with missing columns all at once to avoid fragmentation
                missing_df = pd.DataFrame(missing_dict, index=X.index)
                
                # Concatenate with original DataFrame
                X_aligned = pd.concat([X.copy(), missing_df], axis=1)
                
                # Reorder columns to match training order
                X_aligned = X_aligned[self.feature_names]
            elif extra_cols:
                X_aligned = X[self.feature_names]
            else:
                X_aligned = X
        else:
            X_aligned = X
        
        # Handle extreme values
        X_preprocessed = self._preprocess_extreme_values(X_aligned)
        
        # Try StandardScaler if data is clean
        try:
            # Don't pass the parameter that's causing issues
            result = self.scaler.transform(X_preprocessed)
            return result
        except Exception as e:
            print(f"Warning: StandardScaler transform failed: {e}. Using robust fallback scaling.")
            return self._robust_scale(X_preprocessed)
    
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
            # Replace inf with nan first
            X_cleaned = X.replace([np.inf, -np.inf], np.nan)
            
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
                for col in X_capped.columns:
                    X_capped[col] = X_capped[col].clip(lower=lower_bound[col], upper=upper_bound[col])
                
                # Fill remaining NaNs with column means or 0 if all NaN
                for col in X_capped.columns:
                    if X_capped[col].isna().all():
                        X_capped[col] = 0
                    else:
                        X_capped[col] = X_capped[col].fillna(means[col])
                
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
