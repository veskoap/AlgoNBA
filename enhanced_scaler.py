"""
Enhanced scaler utility for robust handling of extreme values.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


class EnhancedScaler:
    """Scaler with enhanced robustness for extreme values and missing data."""
    
    def __init__(self, clip_threshold=5.0):
        """
        Initialize enhanced scaler.
        
        Args:
            clip_threshold: Values outside this many standard deviations will be clipped
        """
        self.scaler = StandardScaler()
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
        self.feature_names = X.columns.tolist()
        
        # Handle extreme values
        X_preprocessed = self._preprocess_extreme_values(X)
        
        # Try StandardScaler if data is clean
        try:
            result = self.scaler.fit_transform(X_preprocessed)
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
                X_aligned = X.copy()
                for col in missing_cols:
                    X_aligned[col] = 0  # Fill missing columns with zeros
                
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
            result = self.scaler.transform(X_preprocessed)
            return result
        except Exception as e:
            print(f"Warning: StandardScaler transform failed: {e}. Using robust fallback scaling.")
            return self._robust_scale(X_preprocessed)
    
    def _preprocess_extreme_values(self, X):
        """
        Handle extreme values in the data.
        
        Args:
            X: Input data (DataFrame)
            
        Returns:
            DataFrame with extreme values handled
        """
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
    
    def _robust_scale(self, X):
        """
        Manually perform scaling when StandardScaler fails.
        
        Args:
            X: Input data (DataFrame)
            
        Returns:
            Scaled numpy array
        """
        # Use already calculated means and stds
        if self.feature_means is None or self.feature_stds is None:
            self.feature_means = X.mean()
            self.feature_stds = X.std().replace(0, 1)
        
        # Perform manual scaling
        X_scaled = (X - self.feature_means) / self.feature_stds
        
        return X_scaled.values


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
