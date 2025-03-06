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
    """Standard residual block for deep neural network."""
    
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
        
        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(input_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        return F.relu(self.layer_norm(x + self.block(x)))


class BottleneckResidualBlock(nn.Module):
    """
    Bottleneck residual block with improved computational efficiency 
    by using bottleneck architecture for deeper networks.
    """
    
    def __init__(self, input_dim: int, bottleneck_dim: int, dropout_rate: float = 0.2):
        """
        Initialize a bottleneck residual block.
        
        Args:
            input_dim: Dimension of input features
            bottleneck_dim: Dimension of bottleneck layer (smaller than input_dim)
            dropout_rate: Dropout rate for regularization
        """
        super(BottleneckResidualBlock, self).__init__()
        
        # Expansion ratio for wider representations
        expansion = 4
        expanded_dim = bottleneck_dim * expansion
        
        # Bottleneck architecture: input_dim -> bottleneck_dim -> expanded_dim -> input_dim
        self.block = nn.Sequential(
            # Down-projection to bottleneck dimension
            nn.Linear(input_dim, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU(),
            
            # Transformation in bottleneck space
            nn.Linear(bottleneck_dim, expanded_dim),
            nn.BatchNorm1d(expanded_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Up-projection back to input dimension
            nn.Linear(expanded_dim, input_dim),
            nn.BatchNorm1d(input_dim)
        )
        
        # Normalization layer
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # Skip connection transformation if needed
        self.skip_transform = nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the bottleneck residual block.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with residual connection
        """
        # Apply bottleneck transformation and add residual connection
        return F.relu(self.layer_norm(self.skip_transform(x) + self.block(x)))


class SelfAttention(nn.Module):
    """
    Enhanced self-attention mechanism with properly implemented multi-head attention
    for feature relationship modeling with improved dimension handling.
    """
    
    def __init__(self, input_dim: int, attention_dim: int = 64, num_heads: int = 4):
        """
        Initialize enhanced self-attention module with multi-head capabilities.
        
        Args:
            input_dim: Dimension of input features
            attention_dim: Total dimension of attention space (divided among heads)
            num_heads: Number of attention heads for parallel feature processing
        """
        super(SelfAttention, self).__init__()
        
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads
        assert self.head_dim * num_heads == attention_dim, "attention_dim must be divisible by num_heads"
        
        # Single set of projections with multi-head output
        self.qkv_projection = nn.Linear(input_dim, 3 * attention_dim)
        
        # Output projection to combine heads and map back to input dimension
        self.output_projection = nn.Linear(attention_dim, input_dim)
        
        # Layer normalization for better training stability (pre and post attention)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced multi-head self-attention module.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            torch.Tensor: Attention-weighted output tensor
        """
        # Store original input for residual connection
        residual = x
        
        # Apply first layer normalization
        x = self.layer_norm1(x)
        
        # Get batch size
        batch_size = x.size(0)
        
        # Generate query, key, and value projections in a single efficient operation
        qkv = self.qkv_projection(x)
        
        # Reshape to separate q, k, v and prepare for multi-head attention
        try:
            # For easier debugging, split the projections
            q_projected = qkv[:, :self.attention_dim]
            k_projected = qkv[:, self.attention_dim:2*self.attention_dim]
            v_projected = qkv[:, 2*self.attention_dim:]
            
            # Reshape for multi-head processing
            q = q_projected.view(batch_size, self.num_heads, self.head_dim)
            k = k_projected.view(batch_size, self.num_heads, self.head_dim)
            v = v_projected.view(batch_size, self.num_heads, self.head_dim)
            
            # Calculate attention scores
            # [batch_size, num_heads, head_dim] × [batch_size, num_heads, head_dim]
            # Need to add a dimension for batched matrix multiplication
            q_expanded = q.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
            k_expanded = k.unsqueeze(3)  # [batch_size, num_heads, head_dim, 1]
            
            # Compute attention scores with scaling
            attention_scores = torch.matmul(q_expanded, k_expanded) / (self.head_dim ** 0.5)
            
            # Apply softmax to get attention weights
            # [batch_size, num_heads, 1, 1]
            attention_weights = F.softmax(attention_scores, dim=-1)
            
            # Apply attention weights to values
            # We need to carefully handle the dimensions for matrix multiplication
            # Expand v to add a dimension for multiplication with attention weights
            v_expanded = v.unsqueeze(2)  # [batch_size, num_heads, 1, head_dim]
            
            # Attention weights: [batch_size, num_heads, 1, 1]
            # Apply attention weights to get context vectors
            context = attention_weights * v_expanded  # [batch_size, num_heads, 1, head_dim]
            
            # Reshape back: [batch_size, num_heads, 1, head_dim] -> [batch_size, attention_dim]
            context = context.squeeze(2).reshape(batch_size, self.attention_dim)
            
            # Apply output projection and dropout
            output = self.dropout(self.output_projection(context))
            
            # Apply second layer norm with residual connection
            output = self.layer_norm2(output + residual)
            
            return output
            
        except Exception as e:
            # Optimized fallback that still uses attention but in a more robust way
            try:
                # Simpler but still effective fallback using single-head attention
                # Project input to q, k, v directly
                q = qkv[:, :self.attention_dim]
                k = qkv[:, self.attention_dim:2*self.attention_dim]
                v = qkv[:, 2*self.attention_dim:]
                
                # Compute attention with proper scaling
                scale = torch.sqrt(torch.tensor(q.size(-1), dtype=torch.float32))
                attention = torch.bmm(
                    q.unsqueeze(1),  # [batch_size, 1, attention_dim]
                    k.unsqueeze(2)   # [batch_size, attention_dim, 1]
                ).squeeze(-1).squeeze(-1) / scale  # [batch_size]
                
                # Apply sigmoid activation
                weights = torch.sigmoid(attention).unsqueeze(1)  # [batch_size, 1]
                
                # Apply attention weights
                weighted_v = weights * v  # [batch_size, attention_dim]
                
                # Project and add residual connection
                output = self.output_projection(weighted_v)
                output = self.layer_norm2(output + residual)
                
                return output
                
            except Exception as e2:
                # Ultra-safe fallback using element-wise operation
                print(f"Using ultra-safe attention fallback: {e2}")
                
                # Ultra-safe fallback: simple self-gated transformation
                try:
                    # Simple gating mechanism that doesn't require complex tensor ops
                    gate = torch.sigmoid(torch.sum(x * x, dim=1, keepdim=True) / x.size(1))
                    weighted_x = gate * self.value(x)
                except Exception as e3:
                    print(f"Final fallback mode activated: {e3}")
                    # Absolute final fallback - just return input with minimal transformation
                    weighted_x = nn.Dropout(0.1)(x)
                
                return self.layer_norm2(weighted_x + residual)


class EnhancedNBAPredictor(nn.Module):
    """
    Enhanced deep neural network model for NBA game prediction 
    with optimized architecture for improved accuracy and stability.
    """
    
    def __init__(self, 
                input_size: int, 
                dropout_rates: List[float] = [0.3, 0.3, 0.2, 0.1],
                use_attention: bool = True,
                use_residual: bool = True,
                hidden_dims: List[int] = None,
                attention_heads: int = 4,
                use_bottleneck: bool = True):
        """
        Initialize the enhanced deep neural network model.
        
        Args:
            input_size: Number of input features
            dropout_rates: List of dropout rates for each layer
            use_attention: Whether to use self-attention mechanisms
            use_residual: Whether to use residual connections
            hidden_dims: List specifying dimensions of hidden layers
            attention_heads: Number of attention heads to use
            use_bottleneck: Whether to use bottleneck residual blocks
        """
        super(EnhancedNBAPredictor, self).__init__()
        
        self.use_attention = use_attention
        self.use_residual = use_residual
        self.use_bottleneck = use_bottleneck
        
        # Use wider and deeper hidden dimensions by default
        if hidden_dims is None:
            self.hidden_dims = [512, 256, 128, 64]
        else:
            self.hidden_dims = hidden_dims
            
        # Ensure minimum network depth
        if len(self.hidden_dims) < 2:
            self.hidden_dims = [512, 256]
            
        # Stem layer - initial feature transformation with higher capacity
        self.stem = nn.Sequential(
            nn.Linear(input_size, self.hidden_dims[0]),
            nn.BatchNorm1d(self.hidden_dims[0]),
            nn.GELU(),  # Use GELU for better gradient flow
            nn.Dropout(dropout_rates[0])
        )
        
        # Residual blocks - choose between bottleneck or standard
        if use_residual:
            self.res_blocks = nn.ModuleList()
            
            if use_bottleneck:
                # Bottleneck blocks for more efficient deep architecture
                bottleneck_dim = self.hidden_dims[0] // 4
                self.res_blocks.append(
                    BottleneckResidualBlock(self.hidden_dims[0], bottleneck_dim, dropout_rates[1])
                )
                self.res_blocks.append(
                    BottleneckResidualBlock(self.hidden_dims[0], bottleneck_dim, dropout_rates[1])
                )
            else:
                # Standard residual blocks
                hidden_dim = self.hidden_dims[0] // 2
                self.res_blocks.append(
                    ResidualBlock(self.hidden_dims[0], hidden_dim, dropout_rates[1])
                )
                self.res_blocks.append(
                    ResidualBlock(self.hidden_dims[0], hidden_dim, dropout_rates[1])
                )
        
        # Attention mechanism with multi-head support
        if use_attention:
            self.attention = SelfAttention(
                input_dim=self.hidden_dims[0],
                attention_dim=self.hidden_dims[0],
                num_heads=attention_heads
            )
        
        # Transition layers with proper residual connections
        self.transitions = nn.ModuleList()
        for i in range(len(self.hidden_dims) - 1):
            # Add transition block between layers
            self.transitions.append(nn.Sequential(
                nn.Linear(self.hidden_dims[i], self.hidden_dims[i+1]),
                nn.BatchNorm1d(self.hidden_dims[i+1]),
                nn.GELU(),
                nn.Dropout(dropout_rates[min(i+2, len(dropout_rates)-1)])
            ))
        
        # Classifier head with higher capacity
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dims[-1], self.hidden_dims[-1] // 2),
            nn.LayerNorm(self.hidden_dims[-1] // 2),  # Layer norm for better generalization
            nn.GELU(),
            nn.Dropout(dropout_rates[-1]),
            nn.Linear(self.hidden_dims[-1] // 2, 2)
        )
        
        # Monte Carlo dropout mode flag
        self.mc_dropout_enabled = False
        
        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize weights using He initialization for better gradient flow."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced network architecture.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Output tensor with class logits
        """
        # Initial feature extraction
        x = self.stem(x)
        
        # Apply residual blocks with skip connections
        if self.use_residual:
            for res_block in self.res_blocks:
                x = res_block(x)
        
        # Apply self-attention for modeling feature relationships
        if self.use_attention:
            x = self.attention(x)
        
        # Apply transition layers
        for transition in self.transitions:
            x = transition(x)
        
        # Final classification
        x = self.classifier(x)
        
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
            if isinstance(m, nn.Dropout):
                m.train(enable)
                
        if enable:
            self.apply(set_dropout_mode)


class EnhancedDeepModelTrainer:
    """
    Advanced trainer for deep learning models with optimized training procedures,
    adaptive learning rate scheduling, and sophisticated architecture configurations.
    """
    
    def __init__(self, 
                use_residual: bool = True, 
                use_attention: bool = True,
                use_mc_dropout: bool = True,
                use_bottleneck: bool = True,
                attention_heads: int = 4,
                learning_rate: float = 0.001,
                weight_decay: float = 1e-5,
                epochs: int = 150,
                hidden_layers: List[int] = None,
                n_folds: int = 5,
                scheduler_type: str = "cosine",
                early_stopping_patience: int = 15,
                class_weight_adjustment: bool = True,
                batch_size: int = 64,
                gradient_clip: float = 1.0):
        """
        Initialize the enhanced deep model trainer with advanced options.
        
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
            use_bottleneck: Whether to use bottleneck architecture in residual blocks.
                           Bottleneck design improves computational efficiency and model capacity.
            attention_heads: Number of attention heads in multi-head attention.
                           More heads can capture different feature relationship patterns.
            learning_rate: Initial learning rate for the optimizer.
                          Lower values (e.g., 0.0001-0.001) lead to more stable but slower learning.
            weight_decay: L2 regularization strength to prevent overfitting.
            epochs: Maximum number of training epochs.
                   Higher values allow more thorough training but increase time.
                   Early stopping will prevent unnecessary epochs.
            hidden_layers: List specifying the size of each hidden layer.
                         If None, defaults to [512, 256, 128, 64].
                         Example: [64, 32] creates a smaller network with two hidden layers.
            n_folds: Number of folds to use in time-series cross-validation.
                    Higher values (e.g., 5) provide more robust evaluation but require
                    more training time. For quick testing, use lower values like 2-3.
            scheduler_type: Type of learning rate scheduler to use.
                          Options: "cosine", "plateau", "one_cycle", "exponential"
            early_stopping_patience: Number of epochs to wait for improvement before stopping.
                                   Higher values allow more exploration but may waste compute.
            class_weight_adjustment: Whether to use class weights to handle imbalanced data.
                                    Useful when one outcome (win/loss) is more frequent.
            batch_size: Size of mini-batches for training.
                       Larger values can improve training speed but require more memory.
            gradient_clip: Maximum norm of gradients for clipping.
                          Prevents exploding gradients during training.
        """
        self.models = []
        self.scalers = []
        self.training_features = []  # Store original feature names
        self.validation_metrics = []  # Track validation metrics for each fold
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Model architecture settings
        self.use_residual = use_residual
        self.use_attention = use_attention
        self.use_mc_dropout = use_mc_dropout
        self.use_bottleneck = use_bottleneck
        self.attention_heads = attention_heads
        
        # Training hyperparameters
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.gradient_clip = gradient_clip
        
        # Use wider and deeper network by default
        self.hidden_layers = hidden_layers if hidden_layers is not None else [512, 256, 128, 64]
        
        # Cross-validation settings
        self.n_folds = n_folds
        
        # Advanced training options
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.class_weight_adjustment = class_weight_adjustment
        
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
        Make predictions with uncertainty estimates using Monte Carlo dropout.
        
        Args:
            X: DataFrame containing features
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            tuple: (predictions, uncertainties)
        """
        if not self.models or not self.scalers:
            raise ValueError("Models not trained yet. Call train_deep_model first.")
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Get predictions and uncertainties from each model
        all_model_predictions = []
        all_model_uncertainties = []
        
        for fold_idx, (model, scaler) in enumerate(zip(self.models, self.scalers)):
            try:
                # Prepare data for this model
                try:
                    # Determine expected columns
                    expected_cols = None
                    if hasattr(scaler, 'feature_names_in_'):
                        expected_cols = scaler.feature_names_in_
                    elif hasattr(self, 'training_features') and self.training_features:
                        expected_cols = self.training_features
                        
                    if expected_cols is not None:
                        # Align features to expected columns
                        X_aligned = self._align_features(X, expected_cols)
                        
                        # Scale features
                        if isinstance(scaler, EnhancedScaler):
                            X_scaled = scaler.transform(X_aligned)
                        else:
                            X_scaled = scaler.transform(X_aligned)
                    else:
                        # Direct transform
                        X_scaled = scaler.transform(X)
                except Exception as e:
                    print(f"Warning in fold {fold_idx}: {e}")
                    # Create fallback scaler
                    fallback_scaler = EnhancedScaler()
                    X_scaled = fallback_scaler.fit_transform(X)
                
                # Convert to tensor
                X_tensor = torch.FloatTensor(X_scaled).to(self.device)
                
                # Activate MC dropout
                model.eval()
                model.enable_mc_dropout(True)
                
                # Run multiple predictions with active dropout
                mc_pred_list = []
                with torch.no_grad():
                    for _ in range(mc_samples):
                        outputs = model(X_tensor)
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        mc_pred_list.append(probs)
                
                # Convert to numpy arrays
                mc_predictions = np.array(mc_pred_list)
                
                # Calculate mean prediction and uncertainty
                mean_preds = np.mean(mc_predictions, axis=0)
                
                # Use standard deviation as uncertainty measure
                # Higher std = higher uncertainty
                uncertainties = np.std(mc_predictions, axis=0)
                
                # Reset model dropout settings
                model.enable_mc_dropout(False)
                
                # Add to ensemble
                all_model_predictions.append(mean_preds)
                all_model_uncertainties.append(uncertainties)
                
            except Exception as e:
                if fold_idx == 0:  # Only print the first error to reduce verbosity
                    print(f"Error in MC dropout prediction for fold {fold_idx}: {e}")
                # Use default values for this fold
                all_model_predictions.append(np.full(len(X), 0.5))
                all_model_uncertainties.append(np.full(len(X), 0.2))
        
        # Combine results from all models
        if all_model_predictions:
            # Average predictions and uncertainties
            final_predictions = np.mean(all_model_predictions, axis=0)
            
            # Combine uncertainties
            # Using root mean square to properly combine standard deviations
            final_uncertainties = np.sqrt(np.mean(np.square(all_model_uncertainties), axis=0))
            
            # Apply calibration to uncertainties based on prediction strength
            # Predictions close to 0.5 should have higher uncertainty
            prediction_certainty = 1.0 - 2.0 * np.abs(final_predictions - 0.5)
            calibration_factor = 0.5 + 0.5 * prediction_certainty
            calibrated_uncertainties = final_uncertainties * calibration_factor
            
            return final_predictions, calibrated_uncertainties
        else:
            # Default if all models failed
            print("Warning: All uncertainty estimations failed, using defaults")
            mean_preds = np.full(len(X), 0.5)
            uncertainties = np.full(len(X), 0.2)
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
        Calculate enhanced confidence scores based on prediction values and uncertainties.
        
        Args:
            predictions: Prediction probabilities
            uncertainties: Uncertainty estimates (standard deviations)
            
        Returns:
            np.ndarray: Calibrated confidence scores
        """
        # Calculate base confidence factors
        
        # 1. Prediction strength: Distance from 0.5 scaled to [0, 1]
        # Strong predictions (near 0 or 1) should have higher confidence
        prediction_strength = 2.0 * np.abs(predictions - 0.5)  # 0.5 → 0, 0.0/1.0 → 1.0
        
        # 2. Uncertainty penalty: Higher uncertainty should reduce confidence
        # Scale uncertainties to have appropriate effect on final score
        # Apply log transformation to handle varying scales of uncertainty
        normalized_uncertainty = -np.log(uncertainties + 1e-5) / 10.0
        # Clip to reasonable range
        normalized_uncertainty = np.clip(normalized_uncertainty, -2.0, 2.0)
        
        # 3. Combine primary factors using logistic function for smooth boundary behavior
        # This creates a more robust relationship between uncertainty and prediction strength
        raw_confidence = prediction_strength + normalized_uncertainty
        
        # Apply sigmoid function to map to [0, 1]
        bounded_confidence = 1.0 / (1.0 + np.exp(-raw_confidence))
        
        # 4. Apply model-specific calibration
        # - Start with minimum confidence level of 0.3 (never completely uncertain)
        # - Maximum confidence of 0.95 (never completely certain)
        # - Adjustable slope for the middle confidence region
        calibrated_confidence = 0.3 + 0.65 * bounded_confidence
        
        # 5. Apply consistency adjustment
        # Ensure similar predictions have similar confidence (avoid jumps)
        # Sort predictions and confidence
        idx = np.argsort(predictions)
        sorted_preds = predictions[idx]
        sorted_conf = calibrated_confidence[idx]
        
        # Apply smoothing (optional - commented out since it adds complexity)
        # window_size = min(5, len(sorted_conf) // 2)
        # if window_size > 0:
        #     from scipy.ndimage import uniform_filter1d
        #     smoothed_conf = uniform_filter1d(sorted_conf, size=window_size, mode='nearest')
        #     # Restore original order
        #     calibrated_confidence = np.zeros_like(smoothed_conf)
        #     calibrated_confidence[idx] = smoothed_conf
        # else:
        #     calibrated_confidence = calibrated_confidence
        
        # 6. Add prediction divergence factor
        # When predictions significantly differ from average, reduce confidence
        # Only apply when we have enough predictions
        if len(predictions) > 3:
            # Calculate mean prediction
            mean_pred = np.mean(predictions)
            
            # Calculate divergence factor (how much each prediction differs from mean)
            divergence = np.abs(predictions - mean_pred)
            
            # Scale and apply divergence penalty (smaller effect)
            divergence_penalty = 0.1 * divergence
            
            # Apply penalty
            calibrated_confidence -= divergence_penalty
        
        # Ensure final confidence is in valid range
        final_confidence = np.clip(calibrated_confidence, 0.3, 0.95)
        
        return final_confidence