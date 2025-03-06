"""
Enhanced deep learning model for NBA prediction with residual connections
and advanced architecture for improved accuracy.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import List, Tuple, Dict, Any, Optional, Union

from src.utils.constants import FEATURE_REGISTRY
from src.utils.scaling.enhanced_scaler import EnhancedScaler


class ResidualBlock(nn.Module):
    """Residual block for deep neural network."""
    
    def __init__(self, input_dim: int, hidden_dim: int, dropout_rate: float = 0.2):
        """
        Initialize a residual block.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            dropout_rate: Dropout rate for regularization
        """
        super(ResidualBlock, self).__init__()
        
        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        return F.relu(x + self.block(x))


class SelfAttention(nn.Module):
    """Self-attention mechanism for capturing feature relationships."""
    
    def __init__(self, input_dim: int, attention_dim: int = 64):
        """
        Initialize self-attention module.
        
        Args:
            input_dim: Dimension of input features
            attention_dim: Dimension of attention space
        """
        super(SelfAttention, self).__init__()
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the self-attention module.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Attention-weighted output tensor
        """
        # Create attention vectors
        q = self.query(x).unsqueeze(1)  # [batch_size, 1, attention_dim]
        k = self.key(x).unsqueeze(2)    # [batch_size, attention_dim, 1]
        v = self.value(x)               # [batch_size, input_dim]
        
        # Calculate attention scores
        attention = torch.bmm(q, k).squeeze()  # [batch_size]
        attention = torch.sigmoid(attention).unsqueeze(1)  # [batch_size, 1]
        
        # Apply attention weights
        out = attention * v  # [batch_size, input_dim]
        
        return out + x  # Residual connection


class EnhancedNBAPredictor(nn.Module):
    """Enhanced deep neural network model for NBA game prediction."""
    
    def __init__(self, 
                input_size: int, 
                dropout_rates: List[float] = [0.3, 0.3, 0.2, 0.1],
                use_attention: bool = True,
                use_residual: bool = True,
                hidden_dims: List[int] = None):
        """
        Initialize the enhanced deep neural network model.
        
        Args:
            input_size: Number of input features
            dropout_rates: List of dropout rates for each layer
            use_attention: Whether to use self-attention mechanisms
            use_residual: Whether to use residual connections
        """
        super(EnhancedNBAPredictor, self).__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        
        # Use default hidden dimensions if none provided
        if hidden_dims is None:
            self.hidden_dims = [256, 128, 64, 32]
        else:
            self.hidden_dims = hidden_dims
            
        # Make sure we have at least 2 hidden dimensions
        if len(self.hidden_dims) < 2:
            self.hidden_dims = [256, 128]
            
        # Input layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_size, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout_rates[0])
        )
        
        # Residual blocks
        if use_residual:
            res_hidden = self.hidden_dims[0] // 2
            self.res_block1 = ResidualBlock(self.hidden_dims[0], res_hidden, dropout_rates[1])
            self.res_block2 = ResidualBlock(self.hidden_dims[0], res_hidden, dropout_rates[1])
        
        # Attention mechanism
        if use_attention:
            self.attention = SelfAttention(self.hidden_dims[0])
        
        # Middle layers
        middle_layers = []
        for i in range(len(self.hidden_dims) - 2):
            middle_layers.extend([
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                nn.BatchNorm1d(self.hidden_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(dropout_rates[min(i+2, len(dropout_rates)-1)])
            ])
        
        self.middle_layers = nn.Sequential(*middle_layers) if middle_layers else nn.Identity()
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.hidden_dims[-2], self.hidden_dims[-1]),
            nn.ReLU(),
            nn.Linear(self.hidden_dims[-1], 2)
        )
        
        # Monte Carlo dropout mode flag
        self.mc_dropout_enabled = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor
        """
        # Input processing
        x = self.input_layer(x)
        
        # Apply residual blocks if enabled
        if self.use_residual:
            x = self.res_block1(x)
            x = self.res_block2(x)
        
        # Apply attention if enabled
        if self.use_attention:
            x = self.attention(x)
        
        # Middle and output layers
        x = self.middle_layers(x)
        x = self.output_layers(x)
        
        return x
    
    def enable_mc_dropout(self, enable: bool = True) -> None:
        """
        Enable or disable Monte Carlo dropout mode.
        When enabled, dropout remains active during inference for uncertainty estimation.
        
        Args:
            enable: Whether to enable MC dropout
        """
        self.mc_dropout_enabled = enable
        
        # Manually set dropout layers to training mode regardless of model training status
        def set_dropout_mode(m):
            if type(m) == nn.Dropout:
                m.train(enable)
                
        if enable:
            self.apply(set_dropout_mode)


class EnhancedDeepModelTrainer:
    """Class for training and managing enhanced deep learning models."""
    
    def __init__(self, 
                use_residual: bool = True, 
                use_attention: bool = True,
                use_mc_dropout: bool = True,
                learning_rate: float = 0.001,
                weight_decay: float = 1e-5,
                epochs: int = 150,
                hidden_layers: List[int] = None,
                n_folds: int = 5):
        """
        Initialize the enhanced deep model trainer.
        
        Args:
            use_residual: Whether to use residual connections in the neural network.
                         Residual connections help with gradient flow in deeper networks
                         and typically improve performance, especially for complex data.
            use_attention: Whether to use self-attention mechanism in the model.
                          Attention allows the model to focus on important feature
                          relationships and can improve prediction accuracy.
            use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation.
                           When enabled, dropout remains active during inference, allowing
                           multiple predictions to estimate uncertainty.
            learning_rate: Initial learning rate for the optimizer.
                          Lower values (e.g., 0.0001-0.001) lead to more stable but slower learning.
            weight_decay: L2 regularization strength to prevent overfitting.
            epochs: Maximum number of training epochs.
                   Higher values allow more thorough training but increase time.
                   Early stopping will prevent unnecessary epochs.
            hidden_layers: List specifying the size of each hidden layer.
                         If None, defaults to [256, 128, 64, 32].
                         Example: [64, 32] creates a smaller network with two hidden layers.
            n_folds: Number of folds to use in time-series cross-validation.
                    Higher values (e.g., 5) provide more robust evaluation but require
                    more training time. For quick testing, use lower values like 2-3.
        """
        self.models = []
        self.scalers = []
        self.training_features = []  # Store original feature names
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_mc_dropout = use_mc_dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.hidden_layers = hidden_layers if hidden_layers is not None else [256, 128, 64, 32]
        self.n_folds = n_folds
        
    def train_deep_model(self, X: pd.DataFrame) -> Tuple[List, List]:
        """
        Train enhanced deep neural network model with advanced architecture.
        
        Args:
            X: DataFrame containing features and target variable
            
        Returns:
            tuple: (models_list, scalers_list)
        """
        print("\nTraining enhanced deep neural network model...")

        # Extract target variable
        y = X['TARGET']
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Store original feature names for later prediction
        self.training_features = X.columns.tolist()

        print(f"Training deep model with {len(X)} samples and {len(X.columns)} features")
        print(f"Using device: {self.device}")
        print(f"Architecture: Residual={self.use_residual}, Attention={self.use_attention}, MC Dropout={self.use_mc_dropout}")

        models = []
        scalers = []
        fold_metrics = []
        tscv = TimeSeriesSplit(n_splits=self.n_folds)

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
            if not hasattr(scaler, 'feature_names_in_'):
                setattr(scaler, 'feature_names_in_', np.array(X_train.columns))

            # Convert to PyTorch tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled).to(self.device)
            y_train_tensor = torch.LongTensor(y_train.values).to(self.device)
            X_val_tensor = torch.FloatTensor(X_val_scaled).to(self.device)
            y_val_tensor = torch.LongTensor(y_val.values).to(self.device)

            # Initialize enhanced model with configurable architecture
            model = EnhancedNBAPredictor(
                input_size=X_train.shape[1],
                use_residual=self.use_residual,
                use_attention=self.use_attention,
                hidden_dims=self.hidden_layers  # Use configurable hidden dimensions
            ).to(self.device)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.AdamW(model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
            
            # Learning rate scheduler with cosine annealing and warm restarts
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=10,  # Restart every 10 epochs
                T_mult=2  # Double the period after each restart
            )

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = 15  # Increased patience to allow learning rate scheduler to work
            patience_counter = 0
            best_metrics = None

            for epoch in range(self.epochs):  # Use configurable epoch limit
                # Training
                model.train()
                optimizer.zero_grad()
                outputs = model(X_train_tensor)
                loss = criterion(outputs, y_train_tensor)
                loss.backward()
                
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                scheduler.step(epoch)  # Update learning rate

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
                        
                        # Print progress for best epoch
                        print(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Acc: {acc:.3f}, AUC: {auc:.3f}")
                    else:
                        patience_counter += 1

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
        print("\nOverall Enhanced Deep Model Performance:")
        metrics_df = pd.DataFrame(fold_metrics)
        for metric in metrics_df.columns:
            mean_val = metrics_df[metric].mean()
            std_val = metrics_df[metric].std()
            print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

        self.models = models
        self.scalers = scalers
        return models, scalers
        
    def predict(self, X: pd.DataFrame, mc_samples: int = 10) -> np.ndarray:
        """
        Make predictions using ensemble of enhanced deep models with uncertainty estimation.
        
        Args:
            X: DataFrame containing features
            mc_samples: Number of Monte Carlo samples for uncertainty (if enabled)
            
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
                print("Processing enhanced deep model predictions...")
                
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
                
                # If using Monte Carlo dropout, run multiple predictions
                if self.use_mc_dropout:
                    model.eval()  # Set model to evaluation mode
                    model.enable_mc_dropout(True)  # Enable MC dropout
                    
                    # Run multiple predictions with dropout enabled
                    mc_preds = []
                    with torch.no_grad():
                        for _ in range(mc_samples):
                            outputs = model(X_tensor)
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                            mc_preds.append(probs)
                    
                    # Calculate mean prediction across MC samples
                    model_preds = np.mean(mc_preds, axis=0)
                    
                    # Reset model settings
                    model.enable_mc_dropout(False)
                else:
                    # Standard prediction without MC dropout
                    model.eval()
                    with torch.no_grad():
                        outputs = model(X_tensor)
                        model_preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                
                # Add model predictions to ensemble
                all_preds.append(model_preds)
            except Exception as e:
                if fold_idx == 0:  # Only print for first fold
                    print(f"Error in enhanced deep model prediction: {e}")
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
    
    def predict_with_uncertainty(self, X: pd.DataFrame, mc_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates.
        
        Args:
            X: DataFrame containing features
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            tuple: (predictions, uncertainties)
        """
        if not self.use_mc_dropout:
            print("Warning: MC dropout not enabled. Enabling it for uncertainty estimation.")
            self.use_mc_dropout = True
            
        # Track predictions from each model and each MC sample
        all_model_preds = []
        
        # Process each model in the ensemble
        for fold_idx, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            # Only print for first fold to reduce verbosity
            if fold_idx == 0:
                print("Generating uncertainty estimates...")
                
            try:
                # Similar data preparation as in predict method
                X_prepared = self._prepare_data_for_prediction(X, scaler)
                
                if X_prepared is not None:
                    # Convert to tensor
                    X_tensor = torch.FloatTensor(X_prepared).to(self.device)
                    
                    # Enable MC dropout mode
                    model.eval()
                    model.enable_mc_dropout(True)
                    
                    # Run multiple predictions with dropout enabled
                    model_mc_preds = []
                    with torch.no_grad():
                        for _ in range(mc_samples):
                            outputs = model(X_tensor)
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                            model_mc_preds.append(probs)
                    
                    # Add all MC predictions for this model
                    all_model_preds.extend(model_mc_preds)
                    
                    # Reset model settings
                    model.enable_mc_dropout(False)
            except Exception as e:
                if fold_idx == 0:
                    print(f"Error in uncertainty estimation: {e}")
        
        # Calculate mean and standard deviation across all predictions
        if all_model_preds:
            # Convert to numpy array with shape [n_samples, n_observations]
            all_preds_array = np.array(all_model_preds)
            
            # Mean prediction across all samples
            mean_preds = np.mean(all_preds_array, axis=0)
            
            # Standard deviation as uncertainty measure
            uncertainties = np.std(all_preds_array, axis=0)
        else:
            # Default values if no predictions
            mean_preds = np.full(len(X), 0.5)
            uncertainties = np.full(len(X), 0.2)  # Default moderate uncertainty
        
        return mean_preds, uncertainties
    
    def _prepare_data_for_prediction(self, X: pd.DataFrame, scaler: Any) -> Optional[np.ndarray]:
        """
        Prepare data for prediction, handling feature alignment and scaling.
        
        Args:
            X: Input feature DataFrame
            scaler: Scaler to use for feature scaling
            
        Returns:
            numpy.ndarray: Scaled features ready for prediction, or None if preparation fails
        """
        try:
            # Drop non-feature columns
            X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
            
            # Determine expected columns
            expected_cols = None
            if hasattr(scaler, 'feature_names_in_'):
                expected_cols = scaler.feature_names_in_
            elif hasattr(self, 'training_features') and self.training_features:
                expected_cols = self.training_features
            
            if expected_cols is not None:
                # Align features as needed (simplified for brevity)
                X_aligned = self._align_features(X, expected_cols)
                
                # Scale features
                if isinstance(scaler, EnhancedScaler):
                    X_scaled = scaler.transform(X_aligned)
                else:
                    X_scaled = scaler.transform(X_aligned)
                    
                return X_scaled
            else:
                return None
                
        except Exception as e:
            print(f"Error preparing data: {e}")
            return None
    
    def _align_features(self, X: pd.DataFrame, expected_cols: List[str]) -> pd.DataFrame:
        """
        Align features with expected columns, deriving or creating missing features.
        
        Args:
            X: Input feature DataFrame
            expected_cols: List of expected column names
            
        Returns:
            pd.DataFrame: DataFrame with aligned features
        """
        # Create a dictionary to collect all columns
        X_aligned_dict = {}
        
        for col in expected_cols:
            # If feature exists in input, use it
            if col in X.columns:
                X_aligned_dict[col] = X[col].values
            else:
                # Use default values for missing features
                if any(prob_term in col for prob_term in ['WIN_PCT', 'PROBABILITY', 'H2H_']):
                    X_aligned_dict[col] = 0.5
                else:
                    X_aligned_dict[col] = 0
                    
        # Create DataFrame all at once to avoid fragmentation
        return pd.DataFrame(X_aligned_dict, index=X.index)
        
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
    
    def calculate_confidence_from_uncertainty(self, predictions: np.ndarray, 
                                           uncertainties: np.ndarray) -> np.ndarray:
        """
        Calculate confidence scores based on prediction values and uncertainties.
        
        Args:
            predictions: Prediction probabilities
            uncertainties: Uncertainty estimates (standard deviations)
            
        Returns:
            np.ndarray: Confidence scores
        """
        # Calculate confidence based on prediction strength and uncertainty
        # Strong predictions with low uncertainty = high confidence
        prediction_strength = 4 * np.abs(predictions - 0.5)  # 0.5 → 0, 0.0/1.0 → 2.0
        uncertainty_penalty = 5 * uncertainties  # Scale uncertainties to have stronger effect
        
        # Combine into confidence score using sigmoid to map to [0, 1]
        confidence_scores = 1 / (1 + np.exp(-(prediction_strength - uncertainty_penalty)))
        
        # Further transform to push values higher (client expects 0.7+ confidence)
        confidence_scores = 0.5 + 0.5 * confidence_scores
        
        return confidence_scores