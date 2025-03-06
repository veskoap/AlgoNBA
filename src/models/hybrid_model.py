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
                ensemble_weight: float = 0.6,
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
                            Default of 0.6 slightly favors ensemble models as they typically
                            provide more stable predictions.
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
        Train both ensemble and deep learning models.
        
        Args:
            X: DataFrame containing features and target variable
        """
        print("Training hybrid prediction model...")
        
        # Train ensemble model
        print("\n==== Training Enhanced Ensemble Model ====")
        self.ensemble_model.train(X)
        
        # Train deep model
        print("\n==== Training Enhanced Deep Learning Model ====")
        self.deep_model.train_deep_model(X)
        
        # Learn optimal combination weights using validation data
        self._optimize_weights(X)
        
        self.is_trained = True
        print("\nHybrid model training complete!")
        
    def _optimize_weights(self, X: pd.DataFrame) -> None:
        """
        Optimize the weighting between models using validation data.
        
        Args:
            X: DataFrame containing features and target variable
        """
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.metrics import accuracy_score
        
        print("\nOptimizing model integration weights...")
        
        # Extract target
        y = X['TARGET']
        
        # Initialize time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)  # Fewer splits to save computation
        
        best_accuracy = 0
        optimal_weight = 0.5
        
        # Use fewer weights in quick mode
        weights = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        if self.quick_mode:
            weights = [0.3, 0.5, 0.7]  # Just test a few weights in quick mode
            
        # Test different weights to find optimal combination
        for weight in weights:
            fold_accuracies = []
            
            for _, val_idx in tscv.split(X):
                X_val = X.iloc[val_idx]
                y_val = y.iloc[val_idx]
                
                # Get predictions from both models
                ensemble_preds = self.ensemble_model.predict(X_val)
                deep_preds = self.deep_model.predict(X_val)
                
                # Combine predictions using current weight
                hybrid_preds = weight * ensemble_preds + (1 - weight) * deep_preds
                y_pred_binary = (hybrid_preds > 0.5).astype(int)
                
                # Calculate accuracy
                acc = accuracy_score(y_val, y_pred_binary)
                fold_accuracies.append(acc)
            
            # Average accuracy across folds
            avg_accuracy = np.mean(fold_accuracies)
            print(f"Ensemble weight {weight:.1f}: Accuracy {avg_accuracy:.4f}")
            
            # Update optimal weight if this is better
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                optimal_weight = weight
        
        # Store optimal weight
        self.ensemble_weight = optimal_weight
        print(f"Optimal ensemble weight: {optimal_weight:.1f} (Accuracy: {best_accuracy:.4f})")
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the hybrid model.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train first.")
        
        print("Generating hybrid model predictions...")
        
        # Get predictions from both models
        ensemble_preds = self.ensemble_model.predict(X)
        deep_preds = self.deep_model.predict(X)
        
        # Dynamic weighting based on prediction confidence
        hybrid_preds = self.ensemble_weight * ensemble_preds + (1 - self.ensemble_weight) * deep_preds
        
        return hybrid_preds
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train first.")
        
        # Get predictions from both models
        ensemble_preds = self.ensemble_model.predict(X)
        ensemble_conf = self.ensemble_model.calculate_enhanced_confidence_score(ensemble_preds, X)
        
        # Get deep model predictions with uncertainty
        deep_preds, uncertainties = self.deep_model.predict_with_uncertainty(X)
        deep_conf = self.deep_model.calculate_confidence_from_uncertainty(deep_preds, uncertainties)
        
        # Make hybrid predictions using optimal weights
        hybrid_preds = self.ensemble_weight * ensemble_preds + (1 - self.ensemble_weight) * deep_preds
        
        # Calculate weighted confidence scores with preference for higher confidence
        hybrid_conf = np.maximum(
            self.ensemble_weight * ensemble_conf,
            (1 - self.ensemble_weight) * deep_conf
        )
        
        # Apply additional boost to confidence for very strong predictions
        prediction_strength = np.abs(hybrid_preds - 0.5) * 2  # Scale to [0, 1]
        confidence_boost = np.clip(prediction_strength * 0.1, 0, 0.1)  # Max 10% boost
        
        # Apply boost with scaling to maintain [0, 1] range
        final_confidence = np.minimum(hybrid_conf + confidence_boost, 1.0)
        
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