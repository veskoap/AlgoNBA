"""
Deep learning models for NBA prediction.
"""
import os
import pickle
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import List, Tuple, Set, Dict, Any, Optional

from src.utils.constants import FEATURE_REGISTRY
from src.utils.scaling.enhanced_scaler import EnhancedScaler


class DeepNBAPredictor(nn.Module):
    """Deep neural network model for NBA game prediction."""
    
    def __init__(self, input_size: int):
        """
        Initialize the deep neural network model.
        
        Args:
            input_size: Number of input features
        """
        super(DeepNBAPredictor, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        return self.network(x)


class DeepModelTrainer:
    """Class for training and managing deep learning models."""
    
    def __init__(self):
        """Initialize the deep model trainer."""
        self.models = []
        self.scalers = []
        self.training_features = []  # Store original feature names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def train_deep_model(self, X: pd.DataFrame) -> Tuple[List, List]:
        """
        Train deep neural network model with enhanced architecture.
        
        Args:
            X: DataFrame containing features and target variable
            
        Returns:
            tuple: (models_list, scalers_list)
        """
        print("\nTraining deep neural network model...")

        # Extract target variable
        y = X['TARGET']
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Store original feature names for later prediction
        self.training_features = X.columns.tolist()

        print(f"Training deep model with {len(X)} samples and {len(X.columns)} features")
        print(f"Using device: {self.device}")

        models = []
        scalers = []
        fold_metrics = []
        tscv = TimeSeriesSplit(n_splits=5)

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
            print(f"\nTraining deep model fold {fold}...")

            # Prepare data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Scale features using enhanced scaler for robustness
            scaler = EnhancedScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Store feature names in the scaler for easier debugging
            if not hasattr(scaler, 'feature_names'):
                setattr(scaler, 'feature_names_in_', np.array(X_train.columns))

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.LongTensor(y_val.values).to(self.device)

            # Initialize model
            model = DeepNBAPredictor(X_train.shape[1]).to(self.device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)

            # Training loop
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            best_metrics = None

            for epoch in range(100):
                # Training
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                optimizer.step()

                # Validation
                model.eval()
                with torch.no_grad():
                    val_outputs = model(X_val_tensor)
                    val_loss = criterion(val_outputs, y_val_tensor)
                    val_preds = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
                    val_pred_binary = (val_preds > 0.5).astype(int)

                    # Calculate metrics
                    acc = accuracy_score(y_val, val_pred_binary)
                    brier = brier_score_loss(y_val, val_preds)
                    auc = roc_auc_score(y_val, val_preds)

                    # Store best metrics
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_model_state = model.state_dict().copy()
                        best_metrics = {
                            'accuracy': acc,
                            'brier_score': brier,
                            'auc': auc
                        }
                    else:
                        patience_counter += 1

                # Update learning rate
                scheduler.step(val_loss)

                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

            # Load best model
            model.load_state_dict(best_model_state)
            models.append(model)
            scalers.append(scaler)
            fold_metrics.append(best_metrics)

            print(f"Fold {fold} Best Metrics:")
            print(f"Accuracy: {best_metrics['accuracy']:.3f}")
            print(f"Brier Score: {best_metrics['brier_score']:.3f}")
            print(f"AUC-ROC: {best_metrics['auc']:.3f}")

        # Print overall performance
        print("\nOverall Deep Model Performance:")
        metrics_df = pd.DataFrame(fold_metrics)
        for metric in metrics_df.columns:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

        self.models = models
        self.scalers = scalers
        return models, scalers
        
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the trained ensemble of deep models with enhanced feature handling.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.models or not self.scalers:
            raise ValueError("Models not trained yet. Call train_deep_model first.")
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Ensure we have the training features to align with
        if not hasattr(self, 'training_features') or not self.training_features:
            # If training_features wasn't stored, try to infer from first scaler
            if self.scalers and hasattr(self.scalers[0], 'feature_names_in_'):
                self.training_features = self.scalers[0].feature_names_in_.tolist()
            else:
                print("Warning: No training features available. Prediction may be inaccurate.")
                
        # Get predictions from each model in the ensemble
        all_preds = []
        
        for fold_idx, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Only print for first fold to reduce verbosity
            if fold_idx == 0:
                print("Processing deep model predictions...")
                
            try:
                # Determine expected columns for this fold's scaler
                expected_cols = None
                if hasattr(scaler, 'feature_names_in_'):
                    expected_cols = scaler.feature_names_in_
                elif self.training_features:
                    expected_cols = self.training_features
                
                if expected_cols is not None:
                    # Create a dictionary to collect all columns
                    X_aligned_dict = {}
                    
                    for col in expected_cols:
                        # If feature exists in input, use it
                        if col in X.columns:
                            X_aligned_dict[col] = X[col].values
                        else:
                            # Check if we can derive this feature from others
                            feature_derived = False
                            
                            # Try to get base feature name and window
                            base_feature = col
                            window = None
                            if '_D' in col:
                                parts = col.split('_')
                                for i, part in enumerate(parts):
                                    if part.endswith('D') and part[:-1].isdigit():
                                        base_feature = '_'.join(parts[:i])
                                        window = part[:-1]
                                        break
                            
                            # Check if this is a registered feature type we can derive
                            if base_feature in FEATURE_REGISTRY:
                                feature_info = FEATURE_REGISTRY[base_feature]
                                
                                # Only derive if it's a derivable feature and dependencies are available
                                if feature_info['type'] in ['derived', 'interaction'] and 'dependencies' in feature_info:
                                    # Get dependency column names with appropriate windows
                                    dependencies = []
                                    for dep in feature_info['dependencies']:
                                        if window and self._should_apply_window(dep):
                                            dependencies.append(f"{dep}_{window}D")
                                        else:
                                            dependencies.append(dep)
                                    
                                    # Check if all dependencies are available
                                    if all(dep in X.columns for dep in dependencies):
                                        # Derive the feature based on its type
                                        if base_feature in ['WIN_PCT_DIFF', 'OFF_RTG_DIFF', 'DEF_RTG_DIFF', 
                                                          'NET_RTG_DIFF', 'PACE_DIFF', 'FATIGUE_DIFF']:
                                            # Simple difference features
                                            X_aligned_dict[col] = X[dependencies[0]] - X[dependencies[1]]
                                            feature_derived = True
                                        elif base_feature in ['HOME_CONSISTENCY', 'AWAY_CONSISTENCY']:
                                            # Consistency features (std/mean)
                                            if X[dependencies[1]].iloc[0] > 0:  # Avoid division by zero
                                                X_aligned_dict[col] = X[dependencies[0]] / X[dependencies[1]]
                                            else:
                                                X_aligned_dict[col] = 0.5  # Default if mean is zero
                                            feature_derived = True
                                        elif base_feature == 'H2H_RECENCY_WEIGHT':
                                            # H2H recency weight
                                            days = max(1, X[dependencies[1]].iloc[0])
                                            X_aligned_dict[col] = X[dependencies[0]].iloc[0] / np.log1p(days)
                                            feature_derived = True
                            
                            # If we couldn't derive it, use a default value
                            if not feature_derived:
                                # Use 0.5 for probability features, 0 for others
                                if any(prob_term in col for prob_term in ['WIN_PCT', 'PROBABILITY', 'H2H_']):
                                    X_aligned_dict[col] = 0.5
                                else:
                                    X_aligned_dict[col] = 0
                    
                    # Create DataFrame all at once to avoid fragmentation
                    X_aligned = pd.DataFrame(X_aligned_dict, index=X.index)
                    
                    # Scale features with enhanced scaler
                    try:
                        if isinstance(scaler, EnhancedScaler):
                            # Use enhanced scaler directly
                            X_scaled = scaler.transform(X_aligned)
                        else:
                            # For backward compatibility with old models using StandardScaler
                            X_scaled = scaler.transform(X_aligned)
                    except Exception as e:
                        if fold_idx == 0:  # Only print for first fold
                            print(f"Warning: Scaling error in deep model: {e}")
                        # Create an enhanced scaler and use it as fallback
                        fallback_scaler = EnhancedScaler()
                        X_scaled = fallback_scaler.fit_transform(X_aligned)
                else:
                    # If we can't determine expected columns, try direct transform
                    try:
                        if isinstance(scaler, EnhancedScaler):
                            X_scaled = scaler.transform(X)
                        else:
                            X_scaled = scaler.transform(X)
                    except Exception as e:
                        if fold_idx == 0:  # Only print for first fold
                            print(f"Warning: Direct scaling error in deep model: {e}")
                        # Use enhanced scaler as fallback
                        fallback_scaler = EnhancedScaler()
                        X_scaled = fallback_scaler.fit_transform(X)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                # Make predictions
                model.eval()
                with torch.no_grad():
                    outputs = model(X_tensor)
                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    all_preds.append(probs)
            except Exception as e:
                if fold_idx == 0:  # Only print for first fold
                    print(f"Error in deep model prediction: {e}")
                # Add default predictions in case of error
                all_preds.append(np.full(len(X), 0.5))
                
        # Average predictions from all models
        if all_preds:
            ensemble_preds = np.mean(all_preds, axis=0)
        else:
            # Default prediction if no models could be used
            ensemble_preds = np.full(len(X), 0.5)
            print("Warning: Using default predictions (0.5) as all deep models failed")
        
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
        
    def save_model(self, save_dir: str) -> None:
        """
        Save the deep model to disk.
        
        Args:
            save_dir: Directory to save the model in
        """
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save metadata
        metadata = {
            'training_features': self.training_features,
            'device': str(self.device)
        }
        with open(os.path.join(save_dir, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        # Save each model and scaler
        for fold_idx, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Create fold directory
            fold_dir = os.path.join(save_dir, f'fold_{fold_idx}')
            os.makedirs(fold_dir, exist_ok=True)
            
            # Save PyTorch model
            torch.save(model.state_dict(), os.path.join(fold_dir, 'model.pt'))
            
            # Save model architecture info
            with open(os.path.join(fold_dir, 'architecture.pkl'), 'wb') as f:
                pickle.dump({
                    'input_size': model.network[0].in_features,
                    'hidden_layers': [
                        model.network[0].out_features,  # First hidden layer size
                        model.network[4].out_features,  # Second hidden layer size
                        model.network[8].out_features,  # Third hidden layer size
                        model.network[12].out_features  # Fourth hidden layer size
                    ]
                }, f)
            
            # Save scaler
            joblib.dump(scaler, os.path.join(fold_dir, 'scaler.joblib'))
    
    @classmethod
    def load_model(cls, model_dir: str) -> 'DeepModelTrainer':
        """
        Load a deep model from disk.
        
        Args:
            model_dir: Directory containing the saved model
            
        Returns:
            DeepModelTrainer: Loaded model trainer
        """
        # Create new instance
        trainer = cls()
        
        # Load metadata
        with open(os.path.join(model_dir, 'metadata.pkl'), 'rb') as f:
            metadata = pickle.load(f)
            trainer.training_features = metadata.get('training_features', [])
            # Note: We use the current device rather than the saved one for flexibility
            trainer.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load models and scalers
        trainer.models = []
        trainer.scalers = []
        
        fold_idx = 0
        while True:
            fold_dir = os.path.join(model_dir, f'fold_{fold_idx}')
            if not os.path.exists(fold_dir):
                break
                
            # Load architecture info
            with open(os.path.join(fold_dir, 'architecture.pkl'), 'rb') as f:
                architecture = pickle.load(f)
                
            # Create model with the same architecture
            model = DeepNBAPredictor(architecture['input_size']).to(trainer.device)
            
            # Load weights
            model.load_state_dict(torch.load(
                os.path.join(fold_dir, 'model.pt'), 
                map_location=trainer.device
            ))
            
            # Load scaler
            scaler = joblib.load(os.path.join(fold_dir, 'scaler.joblib'))
            
            # Add to trainer
            trainer.models.append(model)
            trainer.scalers.append(scaler)
            
            fold_idx += 1
        
        print(f"Loaded deep model with {fold_idx} folds")
        return trainer