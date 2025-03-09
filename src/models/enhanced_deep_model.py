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
from torch.utils.data import DataLoader, TensorDataset
# Import PyTorch AMP components
import torch

# Create a compatible autocast function
if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'autocast'):
    # Older version with autocast in torch.cuda.amp
    from torch.cuda.amp import GradScaler
    _autocast = torch.cuda.amp.autocast
else:
    # Newer version with autocast in torch.amp
    from torch.amp import GradScaler
    _autocast = torch.amp.autocast

# Create a wrapper function for autocast that handles version differences
def compatible_autocast():
    """
    Creates an autocast context manager that's compatible with different PyTorch versions.
    """
    try:
        # Try PyTorch 1.10+ style with device_type parameter
        return _autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu')
    except TypeError:
        # Fallback to older versions without device_type
        return _autocast()
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import List, Tuple, Dict, Any, Optional, Union
import math
import gc

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
            context = context.squeeze(2).view(batch_size, self.attention_dim)
            
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
            
        # Print model architecture sizing
        print(f"Model architecture: {self.hidden_dims}")
        print(f"Input size: {input_size}")
        # Calculate approximate number of parameters
        # Ensure calculations match actual dimensions used in forward pass
        params = input_size * self.hidden_dims[0]  # Input layer
        for i in range(len(self.hidden_dims) - 1):
            params += self.hidden_dims[i] * self.hidden_dims[i+1]  # Hidden layers
        params += self.hidden_dims[-1] * 2  # Output layer (binary classification)
        print(f"Approximate parameter count: {params:,}")
            
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
        # Validate input dimensions to debug matrix multiplication errors
        input_size = x.shape[1]
        expected_size = self.stem[0].in_features
        
        if input_size != expected_size:
            print(f"Dimension mismatch! Input tensor has {input_size} features, but model expects {expected_size}.")
            # Reshape input tensor if needed to match expected dimensions
            if input_size > expected_size:
                # Truncate extra features
                print(f"Warning: Truncating input tensor from {input_size} to {expected_size} features")
                x = x[:, :expected_size]
            elif input_size < expected_size:
                # Pad with zeros
                print(f"Warning: Padding input tensor from {input_size} to {expected_size} features")
                padding = torch.zeros(x.shape[0], expected_size - input_size, device=x.device)
                x = torch.cat([x, padding], dim=1)
        
        # Initial feature extraction
        x = self.stem(x)
        
        # Apply residual blocks with skip connections
        if self.use_residual:
            for i, res_block in enumerate(self.res_blocks):
                try:
                    x = res_block(x)
                except RuntimeError as e:
                    print(f"Error in residual block {i}: {e}")
                    raise
        
        # Apply self-attention for modeling feature relationships
        if self.use_attention:
            try:
                x = self.attention(x)
            except RuntimeError as e:
                print(f"Error in attention layer: {e}")
                raise
        
        # Apply transition layers
        for i, transition in enumerate(self.transitions):
            try:
                x = transition(x)
            except RuntimeError as e:
                print(f"Error in transition layer {i}: Input shape {x.shape}, expected shape: {self.hidden_dims[i]}→{self.hidden_dims[i+1]}")
                raise
        
        # Final classification
        try:
            x = self.classifier(x)
        except RuntimeError as e:
            print(f"Error in classifier: {e}")
            raise
        
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
                gradient_clip: float = 1.0,
                use_amp: bool = True,
                prefetch_factor: int = 2,
                num_workers: int = 2):
        """
        Initialize the enhanced deep model trainer with advanced options.
        
        Args:
            use_residual: Whether to use residual connections in the neural network.
            use_attention: Whether to use self-attention mechanism in the model.
            use_mc_dropout: Whether to use Monte Carlo dropout for uncertainty estimation.
            use_bottleneck: Whether to use bottleneck architecture in residual blocks.
            attention_heads: Number of attention heads in multi-head attention.
            learning_rate: Initial learning rate for the optimizer.
            weight_decay: L2 regularization strength to prevent overfitting.
            epochs: Maximum number of training epochs.
            hidden_layers: List specifying the size of each hidden layer.
            n_folds: Number of folds to use in time-series cross-validation.
            scheduler_type: Type of learning rate scheduler to use.
            early_stopping_patience: Number of epochs to wait for improvement before stopping.
            class_weight_adjustment: Whether to use class weights to handle imbalanced data.
            batch_size: Size of mini-batches for training.
            gradient_clip: Maximum norm of gradients for clipping.
            use_amp: Whether to use automatic mixed precision for faster training on GPUs.
            prefetch_factor: Number of batches to prefetch in DataLoader.
            num_workers: Number of worker processes for data loading.
        """
        # Check for TPU
        self.is_tpu = False
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            
            devices = xm.get_xla_supported_devices()
            if devices and 'TPU' in devices[0]:
                self.device = xm.xla_device()
                self.is_tpu = True
                print(f"TPU detected: {self.device}")
                
                # TPU-specific optimizations
                original_batch = batch_size
                
                # Much larger batch size for TPU v2-8
                batch_size = max(1024, batch_size * 16)  # 16x larger batches for TPU
                
                print(f"TPU detected: Significantly increasing batch size from {original_batch} to {batch_size}")
                
                # Force AMP for TPU
                use_amp = True
                
                # TPU prefers different scheduler types
                if scheduler_type == "cosine":
                    print("Adjusting scheduler for TPU: using one_cycle instead of cosine")
                    scheduler_type = "one_cycle"
                
                # Adjust worker threads for TPU
                num_workers = 4  # TPU works better with fewer workers
            else:
                # Not a TPU environment
                self.is_tpu = False
                
        except ImportError:
            # torch_xla not available
            self.is_tpu = False
            
        # Enhanced batch size settings for A100 GPU if not on TPU
        if not self.is_tpu and torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            if "A100" in device_name:
                # A100-specific optimizations
                original_batch = batch_size
                
                # Much more aggressive batch size for A100
                if n_folds > 2:  # Full model
                    batch_size = max(256, batch_size * 4)  # 4x larger batches for full models
                else:  # Quick mode
                    batch_size = max(128, batch_size * 2)  # 2x larger for quick mode
                    
                print(f"A100 detected: Significantly increasing batch size from {original_batch} to {batch_size}")
                
                # Additional A100 performance tweaks
                if not use_amp:
                    print("Enabling automatic mixed precision for A100 GPU")
                    use_amp = True
        
        # Store modified batch size        
        self.batch_size = batch_size
        self.models = []
        self.scalers = []
        self.training_features = []  # Store original feature names
        self.validation_metrics = []  # Track validation metrics for each fold
        
        # AMP optimization
        self.use_amp = use_amp  # Automatic mixed precision
        
        # Device configuration - TPU checked first, then GPU, then CPU
        if not self.is_tpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if torch.cuda.is_available():
                # Set device to the fastest available GPU (Colab gives access to exactly one)
                device_count = torch.cuda.device_count()
                if device_count > 0:
                    torch.cuda.set_device(0)  # Use first GPU
                    # Check if we have A100
                    device_name = torch.cuda.get_device_name(0)
                    print(f"Using GPU: {device_name}")
                    
                    # Force some tensor operations to GPU to ensure GPU memory usage
                    dummy_tensor = torch.ones(1, device=self.device)
                    del dummy_tensor  # Immediately delete it
                    
                    # Report GPU memory usage
                    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
                    allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                    print(f"GPU Memory: Total={total_mem:.2f}GB, Reserved={reserved_mem:.2f}GB, Allocated={allocated_mem:.2f}GB")
                    
                    if "A100" in device_name:
                        print("A100 GPU detected! Optimizing for maximum performance.")
                        # A100-specific optimizations
                        torch.backends.cudnn.benchmark = True
                        if torch.cuda.get_device_capability(0)[0] >= 8:  # A100 is compute capability 8.0
                            print("Enabling TF32 precision for faster training")
                            torch.backends.cuda.matmul.allow_tf32 = True
                            torch.backends.cudnn.allow_tf32 = True
            else:
                print("Using CPU. GPU is recommended for faster training.")
        
        # DataLoader settings
        self.batch_size = batch_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        
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
        self.gradient_clip = gradient_clip
        
        # Use wider and deeper network by default
        self.hidden_layers = hidden_layers if hidden_layers is not None else [512, 256, 128, 64]
        
        # Cross-validation settings
        self.n_folds = n_folds
        
        # Advanced training options
        self.scheduler_type = scheduler_type
        self.early_stopping_patience = early_stopping_patience
        self.class_weight_adjustment = class_weight_adjustment
        
    def _ensure_no_dataframe_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert any DataFrame columns to Series to avoid PyTorch errors.
        
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
                print(f"Converting DataFrame column {col} to Series for deep model")
                if len(col_data.columns) > 0:
                    # Convert to Series using first column
                    result[col] = col_data.iloc[:, 0]
                else:
                    # Create empty Series if no columns
                    result[col] = pd.Series(0, index=result.index)
        
        return result
        
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
        
        # Ensure no DataFrame columns exist
        X = self._ensure_no_dataframe_columns(X)
        
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
            feature_scaler = EnhancedScaler()
            print(f"Input feature dimensions - X_train: {X_train.shape}, X_val: {X_val.shape}")
            X_train_scaled = feature_scaler.fit_transform(X_train)
            X_val_scaled = feature_scaler.transform(X_val)
            
            # Store feature names in the scaler for easier debugging
            if not hasattr(feature_scaler, 'feature_names_in_'):
                setattr(feature_scaler, 'feature_names_in_', np.array(X_train.columns))
                
            # Store the input feature size to ensure model dimensions match
            input_feature_size = X_train_scaled.shape[1]
            print(f"Scaled feature dimensions - Input size for model: {input_feature_size}")
            
            # Update shared hidden_layers attribute to match feature dimensions if needed
            if hasattr(self, 'hidden_layers') and len(self.hidden_layers) > 0:
                # Print the current dimensions for debugging
                print(f"Current hidden layer dimensions: {self.hidden_layers}")

            # Create PyTorch datasets and dataloaders for batch processing
            # Create tensors
            X_train_tensor = torch.FloatTensor(X_train_scaled)
            y_train_tensor = torch.LongTensor(y_train.values)
            X_val_tensor = torch.FloatTensor(X_val_scaled)
            y_val_tensor = torch.LongTensor(y_val.values)
            
            # Log memory usage after tensor creation
            if torch.cuda.is_available() and not self.is_tpu:
                # Move smaller tensors to GPU to verify GPU memory usage
                dummy = torch.zeros(1, X_train_scaled.shape[1], device=self.device)
                print(f"Dummy tensor on GPU: {dummy.device}")
                del dummy
                allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                print(f"GPU Memory after tensor creation: {allocated_mem:.2f}GB")
            
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
            
            # Create dataloaders with appropriate optimization for device
            # We need different handling for TPU vs GPU
            if self.is_tpu:
                import torch_xla.core.xla_model as xm
                import torch_xla.distributed.parallel_loader as pl
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    drop_last=True,  # TPU performs better with fixed size batches
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    drop_last=False
                )
                
                # Wrap loaders with TPU ParallelLoader for better performance
                train_loader = pl.ParallelLoader(train_loader, [self.device]).per_device_loader(self.device)
                val_loader = pl.ParallelLoader(val_loader, [self.device]).per_device_loader(self.device)
                
                print(f"Created TPU-optimized data loaders with batch size {self.batch_size}")
            else:
                # Standard GPU/CPU data loaders
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    pin_memory=torch.cuda.is_available(),  # Speed up CPU->GPU transfers
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                    persistent_workers=self.num_workers > 0
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    pin_memory=torch.cuda.is_available(),
                    num_workers=self.num_workers,
                    prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
                    persistent_workers=self.num_workers > 0
                )
            
            # Initialize enhanced model with configurable architecture
            # Use the scaled data shape to ensure dimensions match
            model = EnhancedNBAPredictor(
                input_size=input_feature_size,  # Use the actual scaled feature dimension
                use_residual=self.use_residual,
                use_attention=self.use_attention,
                use_bottleneck=self.use_bottleneck,
                hidden_dims=self.hidden_layers,
                attention_heads=self.attention_heads
            ).to(self.device)
            
            # Initialize class weights if enabled
            if self.class_weight_adjustment:
                class_counts = np.bincount(y_train.values)
                weights = class_counts.sum() / (class_counts * len(class_counts))
                class_weights = torch.FloatTensor(weights).to(self.device)
                criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                criterion = nn.CrossEntropyLoss()
                
            # Optimizer with hardware-specific defaults
            if self.is_tpu:
                # TPU performs better with Adam than AdamW
                optimizer = optim.Adam(
                    model.parameters(), 
                    lr=self.learning_rate,
                    eps=1e-8  # TPU performs better with a larger epsilon
                )
            else:
                # AdamW for GPU/CPU
                optimizer = optim.AdamW(
                    model.parameters(), 
                    lr=self.learning_rate, 
                    weight_decay=self.weight_decay,
                    eps=1e-7  # More stable epsilon for A100
                )
            
            # Configure learning rate scheduler based on selection
            if self.scheduler_type == "cosine":
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=10,  # Restart every 10 epochs
                    T_mult=2  # Double the period after each restart
                )
            elif self.scheduler_type == "one_cycle":
                # One cycle learning rate (better for TPU and A100 with fewer epochs)
                scheduler = optim.lr_scheduler.OneCycleLR(
                    optimizer,
                    max_lr=self.learning_rate * 10,
                    total_steps=self.epochs * len(train_loader),
                    pct_start=0.3,
                    div_factor=25,
                    final_div_factor=1000
                )
            elif self.scheduler_type == "plateau":
                scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, 
                    mode='min', 
                    factor=0.5, 
                    patience=5,
                    verbose=True
                )
            else:
                # Default to cosine annealing
                scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                    optimizer, 
                    T_0=10,
                    T_mult=2
                )

            # Initialize mixed precision training (if enabled)
            # For TPU we don't need a scaler - it uses bfloat16 automatically
            if self.use_amp and torch.cuda.is_available() and not self.is_tpu:
                # Create GradScaler without any parameters to ensure compatibility
                print("Initializing GradScaler for mixed precision training")
                scaler = GradScaler()
            else:
                scaler = None
            
            # Log memory before training
            if torch.cuda.is_available() and not self.is_tpu:
                allocated_mem_before = torch.cuda.memory_allocated(0) / 1024**3
                print(f"GPU Memory allocated before training: {allocated_mem_before:.2f}GB")

            # Training loop with early stopping
            best_val_loss = float('inf')
            patience = self.early_stopping_patience
            patience_counter = 0
            best_metrics = None
            best_model_state = None

            for epoch in range(self.epochs):
                # Force memory allocation at the beginning of epoch for performance
                if epoch == 0:
                    if self.is_tpu:
                        print("Optimizing initial TPU memory allocation...")
                        # Create a large batch for TPU pre-allocation
                        large_batch_size = max(self.batch_size * 4, 2048)
                        print(f"TPU: Using large batch of {large_batch_size} for initialization")
                        
                        # Create a larger batch for pre-allocation
                        import torch_xla.core.xla_model as xm
                        dummy_input = torch.zeros(large_batch_size, X_train.shape[1], device=self.device)
                        dummy_out = model(dummy_input)
                        xm.mark_step()  # Force TPU execution
                        
                        # Clean up
                        del dummy_input, dummy_out
                        
                    elif torch.cuda.is_available():
                        print("Optimizing initial GPU memory allocation...")
                        
                        # Detect A100 GPU and set specialized parameters
                        is_a100 = torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0)
                        
                        # For A100, use much larger batches and more aggressive memory pre-allocation
                        if is_a100:
                            # A100 has more memory, use very large batches for pre-allocation
                            large_batch_size = max(self.batch_size * 8, 512)
                            print(f"A100 detected: Using extra-large batch of {large_batch_size} for optimized initialization")
                            
                            # Create multiple large batches to properly utilize A100 memory
                            dummy_input = torch.zeros(large_batch_size, X_train.shape[1], device=self.device)
                            with torch.cuda.amp.autocast(enabled=self.use_amp):
                                dummy_out = model(dummy_input)
                            
                            # Force more memory allocation with multiple batches
                            print("Priming GPU memory for optimal performance...")
                            for i in range(5):  # More batches for A100
                                dummy_input2 = torch.randn(large_batch_size, X_train.shape[1], device=self.device)
                                with torch.cuda.amp.autocast(enabled=self.use_amp):
                                    model(dummy_input2)
                        else:
                            # For other GPUs, use smaller batch sizes
                            large_batch_size = max(self.batch_size * 4, 128)
                            print(f"Using large batch of {large_batch_size} for initialization")
                            
                            # Create a larger batch and move it to GPU
                            dummy_input = torch.zeros(large_batch_size, X_train.shape[1], device=self.device)
                            dummy_out = model(dummy_input)
                            
                            # Create multiple batches to force memory usage
                            for i in range(3):  # Fewer batches for non-A100
                                dummy_input2 = torch.randn(large_batch_size, X_train.shape[1], device=self.device)
                                model(dummy_input2)
                        
                        # Clean up
                        del dummy_input, dummy_out, dummy_input2
                        torch.cuda.empty_cache()
                        
                        # Log memory usage
                        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                        reserved_mem = torch.cuda.memory_reserved(0) / 1024**3
                        print(f"Epoch start GPU memory: Allocated={allocated_mem:.2f}GB, Reserved={reserved_mem:.2f}GB")
                        
                        # Report model size on device
                        model_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
                        print(f"Model size on GPU: {model_size:.4f}GB")
                
                # ----- Training phase -----
                model.train()
                train_loss = 0.0
                train_steps = 0
                
                # Process mini-batches
                for batch_idx, (inputs, targets) in enumerate(train_loader):
                    # Move batch to device if needed (for TPU, the data is already on device)
                    if not self.is_tpu:
                        inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # Zero gradients
                    optimizer.zero_grad()
                    
                    # TPU-specific training path
                    if self.is_tpu:
                        import torch_xla.core.xla_model as xm
                        
                        # TPU uses bfloat16 automatically if XLA_USE_BF16=1
                        # Check input dimensions before forward pass
                        if inputs.shape[1] != model.stem[0].in_features:
                            print(f"TPU input shape mismatch: got {inputs.shape[1]}, expected {model.stem[0].in_features}")
                            # Adjust as needed
                            if inputs.shape[1] > model.stem[0].in_features:
                                inputs = inputs[:, :model.stem[0].in_features]
                            else:
                                padding = torch.zeros(inputs.shape[0], model.stem[0].in_features - inputs.shape[1], 
                                                     device=inputs.device)
                                inputs = torch.cat([inputs, padding], dim=1)
                                
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        
                        # Gradient clipping if needed
                        if self.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                        
                        # Optimizer step - for TPU we need to use xm.optimizer_step
                        xm.optimizer_step(optimizer)
                        
                        # Mark step for TPU execution
                        xm.mark_step()
                        
                    # GPU with mixed precision path
                    elif scaler is not None:
                        with compatible_autocast():
                            # Check input dimensions before forward pass
                            if inputs.shape[1] != model.stem[0].in_features:
                                print(f"Mixed precision input shape mismatch: got {inputs.shape[1]}, expected {model.stem[0].in_features}")
                                # Adjust as needed
                                if inputs.shape[1] > model.stem[0].in_features:
                                    inputs = inputs[:, :model.stem[0].in_features]
                                else:
                                    padding = torch.zeros(inputs.shape[0], model.stem[0].in_features - inputs.shape[1], 
                                                         device=inputs.device)
                                    inputs = torch.cat([inputs, padding], dim=1)
                            
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                        
                        # Backward pass with gradient scaling
                        scaler.scale(loss).backward()
                        
                        # Gradient clipping (with scaling)
                        if self.gradient_clip > 0:
                            scaler.unscale_(optimizer)
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                        
                        # Step optimizer and update scaler
                        scaler.step(optimizer)
                        scaler.update()
                    
                    # CPU or GPU without mixed precision path
                    else:
                        # Standard precision training
                        # Check input dimensions before forward pass
                        if inputs.shape[1] != model.stem[0].in_features:
                            print(f"Standard training input shape mismatch: got {inputs.shape[1]}, expected {model.stem[0].in_features}")
                            # Adjust dimensions for forward pass
                            if inputs.shape[1] > model.stem[0].in_features:
                                inputs = inputs[:, :model.stem[0].in_features]
                            else:
                                padding = torch.zeros(inputs.shape[0], model.stem[0].in_features - inputs.shape[1], 
                                                     device=inputs.device)
                                inputs = torch.cat([inputs, padding], dim=1)
                                
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        loss.backward()
                        
                        # Gradient clipping
                        if self.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip)
                        
                        optimizer.step()
                    
                    # Update learning rate for batch-based schedulers
                    if self.scheduler_type == "one_cycle":
                        scheduler.step()
                    
                    # Track loss
                    train_loss += loss.item()
                    train_steps += 1

                # ----- Validation phase -----
                model.eval()
                val_loss = 0.0
                val_steps = 0
                val_preds_all = []
                val_targets_all = []
                
                with torch.no_grad():
                    for inputs, targets in val_loader:
                        # Move data to device if needed (for TPU, already on device)
                        if not self.is_tpu:
                            inputs, targets = inputs.to(self.device), targets.to(self.device)
                        
                        # Forward pass with dimension check
                        try:
                            # Check that input dimensions match model expectations
                            if inputs.shape[1] != model.stem[0].in_features:
                                print(f"Validation input shape mismatch: got {inputs.shape[1]}, expected {model.stem[0].in_features}")
                                # Adjust dimensions to match
                                if inputs.shape[1] > model.stem[0].in_features:
                                    # Truncate extra features
                                    inputs = inputs[:, :model.stem[0].in_features]
                                else:
                                    # Pad with zeros
                                    padding = torch.zeros(inputs.shape[0], model.stem[0].in_features - inputs.shape[1], 
                                                         device=inputs.device)
                                    inputs = torch.cat([inputs, padding], dim=1)
                                    
                            outputs = model(inputs)
                            loss = criterion(outputs, targets)
                            probs = torch.softmax(outputs, dim=1)[:, 1]
                        except Exception as e:
                            print(f"Error in validation forward pass: {e}")
                            # Handle the error gracefully with default values
                            outputs = torch.zeros(inputs.shape[0], 2, device=inputs.device)
                            loss = torch.tensor(0.0, device=inputs.device)
                            probs = torch.zeros(inputs.shape[0], device=inputs.device) + 0.5
                        
                        # Track validation metrics
                        val_loss += loss.item()
                        val_steps += 1
                        
                        # Store predictions and targets for computing metrics
                        # For TPU, we need to transfer results back to CPU first
                        if self.is_tpu:
                            import torch_xla.core.xla_model as xm
                            val_preds_all.append(probs.cpu().numpy())
                            val_targets_all.append(targets.cpu().numpy())
                            # Mark step for TPU execution
                            xm.mark_step()
                        else:
                            val_preds_all.append(probs.cpu().numpy())
                            val_targets_all.append(targets.cpu().numpy())
                
                # Combine predictions
                val_preds = np.concatenate(val_preds_all)
                val_targets = np.concatenate(val_targets_all)
                val_pred_binary = (val_preds > 0.5).astype(int)
                
                # Calculate validation metrics
                avg_val_loss = val_loss / max(1, val_steps)
                acc = accuracy_score(val_targets, val_pred_binary)
                brier = brier_score_loss(val_targets, val_preds)
                auc = roc_auc_score(val_targets, val_preds)
                
                # Update epoch-based schedulers
                if self.scheduler_type in ["cosine", "exponential"]:
                    scheduler.step()
                elif self.scheduler_type == "plateau":
                    scheduler.step(avg_val_loss)
                
                # Check for best model
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    
                    # For TPU, we need special handling to move model state to CPU
                    if self.is_tpu:
                        import torch_xla.core.xla_model as xm
                        # Clone model parameters to CPU for saving
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        # Mark step to ensure TPU operations complete
                        xm.mark_step()
                    else:
                        # For GPU, move to CPU to save memory
                        best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
                        
                    best_metrics = {
                        'accuracy': acc,
                        'brier_score': brier,
                        'auc': auc
                    }
                    
                    # Print progress for best epoch
                    print(f"Epoch {epoch}: Val Loss: {avg_val_loss:.4f}, Acc: {acc:.3f}, AUC: {auc:.3f}")
                    
                    # Monitor memory usage for best epochs
                    if torch.cuda.is_available() and not self.is_tpu:
                        allocated_mem = torch.cuda.memory_allocated(0) / 1024**3
                        print(f"Epoch {epoch} GPU Memory: {allocated_mem:.2f}GB")
                else:
                    patience_counter += 1
                
                # Early stopping check
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                
                # Clear cache periodically to avoid fragmentation
                if epoch % 10 == 0:
                    if torch.cuda.is_available() and not self.is_tpu:
                        torch.cuda.empty_cache()
                    gc.collect()

            # Load best model back to device
            if self.is_tpu:
                import torch_xla.core.xla_model as xm
                best_model_state_device = {k: v.to(self.device) for k, v in best_model_state.items()}
                model.load_state_dict(best_model_state_device)
                # Mark step to ensure TPU operations complete
                xm.mark_step()
            else:
                best_model_state_device = {k: v.to(self.device) for k, v in best_model_state.items()}
                model.load_state_dict(best_model_state_device)
                
            models.append(model)
            scalers.append(feature_scaler)  # Store the feature scaler for later predictions
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
        GPU-optimized for faster inference, with caching for hybrid model optimization.
        
        Args:
            X: DataFrame containing features
            mc_samples: Number of Monte Carlo samples for uncertainty (if enabled)
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.models or not self.scalers:
            raise ValueError("Models not trained yet. Call train_deep_model first.")
            
        # Ensure no DataFrame columns exist
        X = self._ensure_no_dataframe_columns(X)
            
        # Optimization: Use prediction cache for identical inputs
        # This greatly speeds up the hybrid model weight optimization
        if hasattr(self, '_pred_cache'):
            # Hash the dataframe to use as cache key
            X_hash = hash(X.shape[0]) + hash(tuple(X.iloc[0].values)) + hash(tuple(X.iloc[-1].values))
            if X_hash in self._pred_cache:
                return self._pred_cache[X_hash]
        else:
            # Initialize cache dict
            self._pred_cache = {}
            X_hash = hash(X.shape[0]) + hash(tuple(X.iloc[0].values)) + hash(tuple(X.iloc[-1].values))
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Ensure we have the training features to align with
        if not hasattr(self, 'training_features') or not self.training_features:
            # If training_features wasn't stored, try to infer from first scaler
            if self.scalers and hasattr(self.scalers[0], 'feature_names_in_'):
                self.training_features = self.scalers[0].feature_names_in_.tolist()
            else:
                print("Warning: No training features available. Prediction may be inaccurate.")
        
        # Only print once for verbosity control
        print("Processing enhanced deep model predictions...")
        
        # Speed optimization: If we have multiple models but are in optimization mode,
        # just use the first model during hybrid model weight optimization
        num_models = len(self.models)
        if hasattr(self, '_in_hybrid_optimization') and self._in_hybrid_optimization:
            # Just use one model for speed
            models_to_use = 1
        else:
            models_to_use = num_models
        
        # Larger batch size for A100
        if torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0):
            batch_size = min(256, len(X))  # Much larger batch for A100
        else:
            batch_size = min(64, len(X))  # Standard batch size
            
        # Get predictions from each model in the ensemble
        all_preds = []
        
        # Process only the requested number of models
        for fold_idx, (model, scaler) in enumerate(zip(self.models[:models_to_use], self.scalers[:models_to_use])):
            try:
                # Align features with model's expected input
                X_aligned = self._prepare_aligned_features(X, scaler, fold_idx)
                
                # Scale features 
                X_scaled = self._scale_features(X_aligned, scaler, fold_idx)
                
                # Optimization: Disable MC dropout during hybrid model optimization
                use_mc = self.use_mc_dropout and not hasattr(self, '_in_hybrid_optimization')
                
                # If using Monte Carlo dropout, run multiple predictions (but fewer samples when optimizing)
                if use_mc:
                    # Use fewer samples in optimization mode
                    actual_samples = mc_samples if not hasattr(self, '_in_hybrid_optimization') else 3
                    model_preds = self._run_mc_dropout_prediction(model, X_scaled, actual_samples, batch_size)
                else:
                    # Standard batch prediction without MC dropout
                    model_preds = self._run_standard_prediction(model, X_scaled, batch_size)
                
                # Add model predictions to ensemble
                all_preds.append(model_preds)
                
                # If we only need one model for optimization, stop after the first one
                if hasattr(self, '_in_hybrid_optimization') and fold_idx == 0:
                    break
                    
            except Exception as e:
                if fold_idx == 0:  # Only print for first fold
                    print(f"Error in enhanced deep model prediction: {e}")
                # Add default predictions in case of error
                all_preds.append(np.full(len(X), 0.5))
        
        # Free memory on GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
                
        # Average predictions from all models
        if all_preds:
            ensemble_preds = np.mean(all_preds, axis=0)
        else:
            # Default prediction if no models could be used
            ensemble_preds = np.full(len(X), 0.5)
            print("Warning: Using default predictions (0.5) as all deep models failed")
        
        # Final NaN check
        if np.isnan(ensemble_preds).any():
            print("Error: Input contains NaN. Replacing with default values.")
            # Replace NaNs with default probability of 0.5
            ensemble_preds = np.nan_to_num(ensemble_preds, nan=0.5)
        
        # Store in cache for repeated calls with the same data
        self._pred_cache[X_hash] = ensemble_preds
        
        # Limit cache size to prevent memory issues
        if len(self._pred_cache) > 10:
            # Remove oldest entries (first ones added)
            keys_to_remove = list(self._pred_cache.keys())[:-10]
            for key in keys_to_remove:
                del self._pred_cache[key]
                
        return ensemble_preds
        
    def _prepare_aligned_features(self, X: pd.DataFrame, scaler, fold_idx: int) -> pd.DataFrame:
        """
        Prepare and align features for prediction, handling missing and derived features.
        Handles NaN values and ensures feature validity.
        
        Args:
            X: Input DataFrame
            scaler: Feature scaler to use
            fold_idx: Index of current fold (for logging)
            
        Returns:
            pd.DataFrame: Aligned features ready for scaling
        """
        # Determine expected columns for this fold's scaler
        expected_cols = None
        if hasattr(scaler, 'feature_names_in_'):
            expected_cols = scaler.feature_names_in_
        elif self.training_features:
            expected_cols = self.training_features
        
        if expected_cols is None:
            # No expected columns, return original dataframe
            return X
            
        # Create a dictionary to efficiently build the aligned dataframe
        X_aligned_dict = {}
        
        for col in expected_cols:
            # If feature exists in input, use it
            if col in X.columns:
                X_aligned_dict[col] = X[col].values
            else:
                # Try to derive the feature from existing ones
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
                    
                    # Only derive if it's a derivable feature and dependencies available
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
        return pd.DataFrame(X_aligned_dict, index=X.index)
    
    def _scale_features(self, X: pd.DataFrame, scaler, fold_idx: int) -> np.ndarray:
        """
        Scale features using the appropriate scaler with error handling.
        Includes NaN checking and handling.
        
        Args:
            X: DataFrame with aligned features
            scaler: Feature scaler to use
            fold_idx: Index of current fold (for logging)
            
        Returns:
            np.ndarray: Scaled features ready for model input
        """
        # Check for NaN values before scaling
        if X.isna().any().any():
            if fold_idx == 0:  # Only print for first fold
                print(f"Warning: NaN values detected in input data. Filling with appropriate values.")
            
            # Fill NaNs with appropriate values based on column type
            for col in X.columns:
                if 'PCT' in col or 'PROBABILITY' in col or 'H2H_' in col:
                    # Probability columns get filled with 0.5
                    X[col] = X[col].fillna(0.5)
                else:
                    # Other columns get filled with 0
                    X[col] = X[col].fillna(0)
        
        try:
            if isinstance(scaler, EnhancedScaler):
                # Use enhanced scaler directly
                scaled_data = scaler.transform(X)
            else:
                # For backward compatibility
                scaled_data = scaler.transform(X)
                
            # Check for NaN values after scaling
            if np.isnan(scaled_data).any():
                if fold_idx == 0:  # Only print for first fold
                    print(f"Warning: NaN values found after scaling. Replacing with zeros.")
                # Replace NaNs with zeros
                scaled_data = np.nan_to_num(scaled_data, nan=0.0)
                
            return scaled_data
            
        except Exception as e:
            if fold_idx == 0:  # Only print for first fold
                print(f"Warning: Scaling error in deep model: {e}")
            # Create an enhanced scaler and use it as fallback
            fallback_scaler = EnhancedScaler()
            scaled_data = fallback_scaler.fit_transform(X)
            
            # Check for NaNs in fallback result
            if np.isnan(scaled_data).any():
                scaled_data = np.nan_to_num(scaled_data, nan=0.0)
                
            return scaled_data
    
    def _run_mc_dropout_prediction(self, model, X_scaled: np.ndarray, mc_samples: int, batch_size: int) -> np.ndarray:
        """
        Run prediction with Monte Carlo dropout for uncertainty estimation, optimized for GPU.
        
        Args:
            model: PyTorch model to use
            X_scaled: Scaled features
            mc_samples: Number of MC samples to take
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Mean prediction across MC samples
        """
        # Set model to evaluation mode with MC dropout enabled
        model.eval()
        model.enable_mc_dropout(True)
        
        # Convert to tensor with GPU support if available
        X_tensor = torch.FloatTensor(X_scaled)
        
        # Create tensor dataset and dataloader for batch processing
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Run multiple predictions with dropout enabled
        all_mc_preds = []
        
        with torch.no_grad():
            for mc_run in range(mc_samples):
                batch_preds = []
                
                for batch_idx, (inputs,) in enumerate(dataloader):
                    # Move to device
                    inputs = inputs.to(self.device)
                    
                    # Forward pass with mixed precision if available
                    if self.use_amp and torch.cuda.is_available():
                        with compatible_autocast():
                            outputs = model(inputs)
                    else:
                        outputs = model(inputs)
                    
                    # Get probabilities with error handling
                    try:
                        # Check if outputs is a valid tensor with correct shape
                        if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                            # Proper softmax over class dimension
                            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                        else:
                            # Handle unexpected output format
                            print(f"Warning: Unexpected MC model output shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")
                            # Create default predictions
                            probs = np.ones(inputs.size(0)) * 0.5
                        
                        batch_preds.append(probs)
                    except Exception as e:
                        print(f"Error processing MC model output: {e}")
                        # Fallback to default predictions
                        probs = np.ones(inputs.size(0)) * 0.5
                        batch_preds.append(probs)
                
                # Combine batch predictions
                mc_preds = np.concatenate(batch_preds)
                all_mc_preds.append(mc_preds)
        
        # Calculate mean prediction across MC samples
        model_preds = np.mean(all_mc_preds, axis=0)
        
        # Reset model settings
        model.enable_mc_dropout(False)
        
        return model_preds
    
    def _run_standard_prediction(self, model, X_scaled: np.ndarray, batch_size: int) -> np.ndarray:
        """
        Run standard prediction without MC dropout, optimized for GPU.
        
        Args:
            model: PyTorch model to use
            X_scaled: Scaled features
            batch_size: Batch size for processing
            
        Returns:
            np.ndarray: Model predictions
        """
        # Set model to evaluation mode
        model.eval()
        
        # Convert to tensor dataset for batching
        X_tensor = torch.FloatTensor(X_scaled)
        dataset = torch.utils.data.TensorDataset(X_tensor)
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size,
            shuffle=False,
            pin_memory=torch.cuda.is_available()
        )
        
        # Make predictions in batches
        batch_preds = []
        
        with torch.no_grad():
            for batch_idx, (inputs,) in enumerate(dataloader):
                # Move to device
                inputs = inputs.to(self.device)
                
                # Forward pass with mixed precision if available
                if self.use_amp and torch.cuda.is_available():
                    with compatible_autocast():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
                
                # Get probabilities with error handling
                try:
                    # Check if outputs is a valid tensor with correct shape
                    if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                        # Proper softmax over class dimension
                        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                    else:
                        # Handle unexpected output format
                        print(f"Warning: Unexpected model output shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")
                        # Create default predictions
                        probs = np.ones(inputs.size(0)) * 0.5
                    
                    batch_preds.append(probs)
                except Exception as e:
                    print(f"Error processing model output: {e}")
                    # Fallback to default predictions
                    probs = np.ones(inputs.size(0)) * 0.5
                    batch_preds.append(probs)
        
        # Combine batch predictions
        return np.concatenate(batch_preds)
    
    def predict_with_uncertainty(self, X: pd.DataFrame, mc_samples: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using Monte Carlo dropout.
        GPU-optimized for faster inference with caching for hybrid model optimization.
        
        Args:
            X: DataFrame containing features
            mc_samples: Number of Monte Carlo samples for uncertainty estimation
            
        Returns:
            tuple: (predictions, uncertainties)
        """
        if not self.models or not self.scalers:
            raise ValueError("Models not trained yet. Call train_deep_model first.")
        
        # Optimization: Use prediction cache for identical inputs during optimization
        if hasattr(self, '_uncertainty_cache'):
            # Hash the dataframe to use as cache key
            X_hash = hash(X.shape[0]) + hash(tuple(X.iloc[0].values)) + hash(tuple(X.iloc[-1].values))
            if X_hash in self._uncertainty_cache:
                return self._uncertainty_cache[X_hash]
        else:
            # Initialize cache dict
            self._uncertainty_cache = {}
            X_hash = hash(X.shape[0]) + hash(tuple(X.iloc[0].values)) + hash(tuple(X.iloc[-1].values))
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Only print once for verbosity control
        print("Processing enhanced deep model predictions with uncertainty...")
        
        # Speed optimization: If in hybrid model optimization, use fewer models and samples
        if hasattr(self, '_in_hybrid_optimization') and self._in_hybrid_optimization:
            # Use fewer MC samples and only the first model during optimization
            actual_samples = min(5, mc_samples)  # Much smaller sample count
            models_to_use = 1  # Just use one model
        else:
            actual_samples = mc_samples
            models_to_use = len(self.models)
            
        # Get predictions and uncertainties from each model
        all_model_predictions = []
        all_model_uncertainties = []
        
        # Batch processing for predictions with larger batch size for A100
        if torch.cuda.is_available() and "A100" in torch.cuda.get_device_name(0):
            batch_size = min(256, len(X))  # Much larger batch for A100
        else:
            batch_size = min(64, len(X))  # Standard batch size
        
        # Only process the number of models we need
        for fold_idx, (model, scaler) in enumerate(zip(self.models[:models_to_use], self.scalers[:models_to_use])):
            try:
                # Prepare and align features
                X_aligned = self._prepare_aligned_features(X, scaler, fold_idx)
                
                # Scale features
                X_scaled = self._scale_features(X_aligned, scaler, fold_idx)
                
                # Generate MC dropout samples - fewer samples in optimization mode
                mc_samples_list = self._run_mc_dropout_samples(model, X_scaled, actual_samples, batch_size)
                
                # Calculate mean and uncertainty from samples
                if mc_samples_list:
                    try:
                        # Convert sample list to array with error checking
                        mc_predictions = np.array(mc_samples_list)
                        
                        # Ensure consistent dimensions for all samples
                        if mc_predictions.ndim != 2:
                            print(f"Warning: Inconsistent MC sample dimensions: {mc_predictions.shape}")
                            # Try to fix by reshaping or padding
                            if mc_predictions.ndim == 1:
                                # Single sample - reshape to 2D
                                mc_predictions = mc_predictions.reshape(1, -1)
                            elif mc_predictions.ndim > 2:
                                # Too many dimensions - take first slice
                                mc_predictions = mc_predictions[0].reshape(1, -1)
                        
                        # Mean prediction across samples
                        mean_preds = np.mean(mc_predictions, axis=0)
                        
                        # Standard deviation as uncertainty measure
                        uncertainties = np.std(mc_predictions, axis=0)
                        
                        # Add to ensemble
                        all_model_predictions.append(mean_preds)
                        all_model_uncertainties.append(uncertainties)
                    except Exception as e:
                        print(f"Error processing MC samples: {e}")
                        # Fallback if processing failed - use standard prediction
                        try:
                            fallback_preds = self._run_standard_prediction(model, X_scaled, batch_size)
                            # Use default uncertainty based on prediction strength
                            fallback_uncertainty = 0.2 * (1.0 - np.abs(fallback_preds - 0.5) * 2)
                            all_model_predictions.append(fallback_preds)
                            all_model_uncertainties.append(fallback_uncertainty)
                        except Exception as e2:
                            print(f"Fallback prediction also failed: {e2}")
                            # Last resort default values
                            default_preds = np.full(len(X), 0.5)
                            default_uncertainty = np.full(len(X), 0.2)
                            all_model_predictions.append(default_preds)
                            all_model_uncertainties.append(default_uncertainty)
                else:
                    # Fallback if MC sampling failed - use standard prediction
                    print("MC sampling failed to produce valid results. Using standard prediction as fallback.")
                    try:
                        fallback_preds = self._run_standard_prediction(model, X_scaled, batch_size)
                        # Use default uncertainty based on prediction strength
                        fallback_uncertainty = 0.2 * (1.0 - np.abs(fallback_preds - 0.5) * 2)
                        all_model_predictions.append(fallback_preds)
                        all_model_uncertainties.append(fallback_uncertainty)
                    except Exception as e:
                        print(f"Fallback prediction failed: {e}")
                        # Last resort default values
                        default_preds = np.full(len(X), 0.5)
                        default_uncertainty = np.full(len(X), 0.2)
                        all_model_predictions.append(default_preds)
                        all_model_uncertainties.append(default_uncertainty)
                    
                # If we're in optimization mode, one model is enough
                if hasattr(self, '_in_hybrid_optimization') and fold_idx == 0:
                    break
                
            except Exception as e:
                if fold_idx == 0:  # Only print first error to reduce verbosity
                    print(f"Error in uncertainty estimation for fold {fold_idx}: {e}")
                
                # Use standard prediction as fallback
                try:
                    # Standard prediction without MC dropout - with large batch size
                    model_preds = self._run_standard_prediction(model, X_scaled, batch_size)
                    
                    # Estimated uncertainty based on prediction strength
                    est_uncertainties = 0.2 * (1.0 - np.abs(model_preds - 0.5) * 2)
                    
                    all_model_predictions.append(model_preds)
                    all_model_uncertainties.append(est_uncertainties)
                except Exception:
                    # Last resort default values
                    all_model_predictions.append(np.full(len(X), 0.5))
                    all_model_uncertainties.append(np.full(len(X), 0.2))
        
        # Free GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Combine results from all models
        if all_model_predictions:
            # Average predictions
            final_predictions = np.mean(all_model_predictions, axis=0)
            
            # Simplified uncertainty calculation for optimization mode
            if hasattr(self, '_in_hybrid_optimization') and self._in_hybrid_optimization:
                # Simple standard deviation estimate - faster
                final_uncertainties = np.mean(all_model_uncertainties, axis=0)
                # Simple clipping
                calibrated_uncertainties = np.clip(final_uncertainties, 0.05, 0.5)
            else:
                # Full uncertainty calculation for normal mode
                # Combine uncertainties using root mean square
                # (proper way to combine standard deviations)
                final_uncertainties = np.sqrt(np.mean(np.square(all_model_uncertainties), axis=0))
                
                # Apply calibration to uncertainties based on prediction strength
                # Predictions close to 0.5 should have higher uncertainty
                prediction_certainty = 1.0 - 2.0 * np.abs(final_predictions - 0.5)
                calibration_factor = 0.5 + 0.5 * prediction_certainty
                calibrated_uncertainties = final_uncertainties * calibration_factor
                
                # Ensure uncertainties are in reasonable range
                calibrated_uncertainties = np.clip(calibrated_uncertainties, 0.05, 0.5)
            
            # Final NaN check
            if np.isnan(final_predictions).any() or np.isnan(calibrated_uncertainties).any():
                print("Error: Input contains NaN. Replacing with default values.")
                # Replace NaNs with defaults
                final_predictions = np.nan_to_num(final_predictions, nan=0.5)
                calibrated_uncertainties = np.nan_to_num(calibrated_uncertainties, nan=0.2)
            
            # Store in cache for repeated calls with the same data
            self._uncertainty_cache[X_hash] = (final_predictions, calibrated_uncertainties)
            
            # Limit cache size to prevent memory issues
            if len(self._uncertainty_cache) > 10:
                # Remove oldest entries (first ones added)
                keys_to_remove = list(self._uncertainty_cache.keys())[:-10]
                for key in keys_to_remove:
                    del self._uncertainty_cache[key]
            
            return final_predictions, calibrated_uncertainties
        else:
            # Default if all models failed
            print("Warning: All uncertainty estimations failed, using defaults")
            mean_preds = np.full(len(X), 0.5)
            uncertainties = np.full(len(X), 0.2)
            
            # Cache default values too
            self._uncertainty_cache[X_hash] = (mean_preds, uncertainties)
            
            return mean_preds, uncertainties
            
    def _run_mc_dropout_samples(self, model, X_scaled: np.ndarray, mc_samples: int, batch_size: int) -> List[np.ndarray]:
        """
        Run Monte Carlo dropout to generate samples for uncertainty estimation.
        Optimized for GPU batch processing with improved error handling.
        
        Args:
            model: PyTorch model
            X_scaled: Scaled features
            mc_samples: Number of samples to generate
            batch_size: Batch size for processing
            
        Returns:
            List of numpy arrays containing predictions for each MC sample
        """
        # Set model to evaluation mode with MC dropout enabled
        model.eval()
        model.enable_mc_dropout(True)
        
        # Input validation to ensure X_scaled is valid
        expected_features = None
        try:
            # Ensure input array is valid and has correct dimensions
            if not isinstance(X_scaled, np.ndarray):
                print(f"Warning: Input is not numpy array, but {type(X_scaled)}")
                # Try to convert if possible
                X_scaled = np.array(X_scaled)
                
            # Ensure 2D input shape
            if X_scaled.ndim == 1:
                print(f"Warning: Input is 1D, reshaping to 2D")
                X_scaled = X_scaled.reshape(1, -1)
            elif X_scaled.ndim > 2:
                print(f"Warning: Input has {X_scaled.ndim} dimensions, reshaping to 2D")
                X_scaled = X_scaled.reshape(X_scaled.shape[0], -1)
                
            # Check for input dimensions mismatch
            expected_features = model.stem[0].in_features if hasattr(model, 'stem') else None
            if expected_features and X_scaled.shape[1] != expected_features:
                print(f"Warning: Input has {X_scaled.shape[1]} features, but model expects {expected_features}")
                # Adjust dimensions
                if X_scaled.shape[1] > expected_features:
                    print(f"Truncating input from {X_scaled.shape[1]} to {expected_features} features")
                    X_scaled = X_scaled[:, :expected_features]
                else:
                    # Pad with zeros
                    print(f"Padding input from {X_scaled.shape[1]} to {expected_features} features")
                    padding = np.zeros((X_scaled.shape[0], expected_features - X_scaled.shape[1]))
                    X_scaled = np.concatenate([X_scaled, padding], axis=1)
                
                # Double-check the adjusted dimensions
                if X_scaled.shape[1] != expected_features:
                    print(f"ERROR: Failed to adjust input dimensions. Got {X_scaled.shape[1]}, expected {expected_features}")
                    # Force the correct size as a last resort
                    X_scaled = np.zeros((X_scaled.shape[0], expected_features))
        except Exception as ex:
            print(f"Error preparing input for MC dropout: {ex}")
            # Create default input as emergency fallback
            if expected_features:
                X_scaled = np.zeros((1, expected_features))
            else:
                # Try to infer expected features from model
                try:
                    for module in model.modules():
                        if isinstance(module, nn.Linear):
                            expected_features = module.in_features
                            print(f"Inferred expected features: {expected_features}")
                            break
                except Exception:
                    # If all else fails
                    expected_features = 512  # Typical default size for this model
                    print(f"Using default expected features: {expected_features}")
                
                X_scaled = np.zeros((1, expected_features))  # Safe fallback size
        
        # Final validation to ensure X_scaled has correct dimensions
        if expected_features and X_scaled.shape[1] != expected_features:
            print(f"CRITICAL: Input dimensions still don't match after adjustment!")
            print(f"Got {X_scaled.shape[1]}, expected {expected_features}")
            # Force correct dimensions as last resort
            X_scaled = np.zeros((X_scaled.shape[0], expected_features))
        
        # Create dataset and dataloader for batch processing
        try:
            X_tensor = torch.FloatTensor(X_scaled)
            dataset = torch.utils.data.TensorDataset(X_tensor)
            dataloader = DataLoader(
                dataset, 
                batch_size=batch_size,
                shuffle=False,
                pin_memory=torch.cuda.is_available()
            )
        except Exception as e:
            print(f"Error creating dataloader: {e}")
            # Emergency fallback - create a minimal dataset with a single zero sample
            if expected_features:
                X_tensor = torch.zeros((1, expected_features), device=self.device)
            else:
                X_tensor = torch.zeros((1, 512), device=self.device)  # Safe fallback size
            dataset = torch.utils.data.TensorDataset(X_tensor)
            dataloader = DataLoader(dataset, batch_size=1)
        
        # Storage for samples
        mc_sample_list = []
        
        try:
            with torch.no_grad():
                for mc_idx in range(mc_samples):
                    batch_preds = []
                    
                    for inputs, in dataloader:
                        try:
                            # Move to device
                            inputs = inputs.to(self.device)
                            
                            # Final verification of tensor shape before model forward pass
                            if hasattr(model, 'stem') and hasattr(model.stem[0], 'in_features'):
                                expected_input_size = model.stem[0].in_features
                                if inputs.shape[1] != expected_input_size:
                                    print(f"Mismatch before forward pass: {inputs.shape[1]} vs expected {expected_input_size}")
                                    # Adjust dimensions one final time
                                    if inputs.shape[1] > expected_input_size:
                                        inputs = inputs[:, :expected_input_size]
                                    else:
                                        padding = torch.zeros(inputs.shape[0], expected_input_size - inputs.shape[1], 
                                                            device=inputs.device)
                                        inputs = torch.cat([inputs, padding], dim=1)
                            
                            # Forward pass with mixed precision if available
                            try:
                                if self.use_amp and torch.cuda.is_available():
                                    with compatible_autocast():
                                        outputs = model(inputs)
                                else:
                                    outputs = model(inputs)
                            except RuntimeError as e:
                                if "mat1 and mat2 shapes cannot be multiplied" in str(e):
                                    # Handle dimension mismatch by trying one more approach
                                    print(f"Matrix dimension mismatch: {str(e)}")
                                    print("Attempting to recover with correct dimensions...")
                                    
                                    # Try to extract dimension information from the error
                                    import re
                                    match = re.search(r"(\d+)x(\d+) and (\d+)x(\d+)", str(e))
                                    if match:
                                        a, b, c, d = map(int, match.groups())
                                        print(f"Mismatch: ({a}x{b}) x ({c}x{d})")
                                        
                                        if b != c and c > 0:
                                            # We need input dimension c
                                            expected_input_size = c
                                            
                                            if inputs.shape[1] != expected_input_size:
                                                print(f"Correcting dimensions to {expected_input_size}")
                                                if inputs.shape[1] > expected_input_size:
                                                    inputs = inputs[:, :expected_input_size]
                                                else:
                                                    padding = torch.zeros(inputs.shape[0], expected_input_size - inputs.shape[1], 
                                                                        device=inputs.device)
                                                    inputs = torch.cat([inputs, padding], dim=1)
                                                
                                                # Try again with corrected dimensions
                                                outputs = model(inputs)
                                            else:
                                                # If dimensions already match but still fails, use default values
                                                raise
                                    else:
                                        # Couldn't parse dimensions from error, use default values
                                        raise
                                else:
                                    # Some other error occurred
                                    raise
                            
                            # Get probabilities with error handling
                            try:
                                # Check if outputs is a valid tensor with correct shape
                                if isinstance(outputs, torch.Tensor) and outputs.dim() > 1 and outputs.size(1) > 1:
                                    # Proper softmax over class dimension
                                    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                                else:
                                    # Handle unexpected output format
                                    print(f"Warning: Unexpected MC sample output shape: {outputs.shape if hasattr(outputs, 'shape') else 'unknown'}")
                                    # Create default predictions
                                    probs = np.ones(inputs.size(0)) * 0.5
                                
                                batch_preds.append(probs)
                            except Exception as e:
                                print(f"Error processing MC sample output: {e}")
                                # Fallback to default predictions
                                probs = np.ones(inputs.size(0)) * 0.5
                                batch_preds.append(probs)
                        except Exception as e:
                            print(f"Error in MC batch inference: {e}")
                            # Add default predictions for this batch
                            probs = np.ones(inputs.size(0)) * 0.5
                            batch_preds.append(probs)
                    
                    # Combine batch predictions for this MC sample with error handling
                    try:
                        if batch_preds:
                            full_sample = np.concatenate(batch_preds)
                            mc_sample_list.append(full_sample)
                        else:
                            # No valid predictions - add a default sample
                            print(f"No valid predictions for MC sample {mc_idx}")
                            default_sample = np.ones(len(X_scaled)) * 0.5
                            mc_sample_list.append(default_sample)
                    except Exception as e:
                        print(f"Error combining MC batch predictions: {e}")
                        # Add a default sample as fallback
                        default_sample = np.ones(len(X_scaled)) * 0.5
                        mc_sample_list.append(default_sample)
            
            # Reset model dropout settings
            model.enable_mc_dropout(False)
            
            # Check if we have valid samples
            if mc_sample_list:
                # Verify all samples have the same shape
                sample_shapes = [sample.shape for sample in mc_sample_list]
                if len(set(sample_shapes)) > 1:
                    print(f"Warning: Inconsistent sample shapes: {sample_shapes}")
                    # Fix inconsistent shapes
                    target_shape = max(len(sample) for sample in mc_sample_list)
                    for i, sample in enumerate(mc_sample_list):
                        if len(sample) != target_shape:
                            print(f"Fixing sample {i} from shape {sample.shape} to length {target_shape}")
                            # Pad or truncate to match target shape
                            if len(sample) < target_shape:
                                # Pad with default values
                                mc_sample_list[i] = np.concatenate([sample, np.ones(target_shape - len(sample)) * 0.5])
                            else:
                                # Truncate
                                mc_sample_list[i] = sample[:target_shape]
                
                return mc_sample_list
            else:
                # Create default samples if none were valid
                print("No valid MC samples were created. Using default samples.")
                default_samples = [np.ones(len(X_scaled)) * 0.5 for _ in range(mc_samples)]
                return default_samples
            
        except Exception as e:
            print(f"Error in MC sampling: {e}")
            model.enable_mc_dropout(False)
            
            # Create and return default samples
            print("Creating default MC samples due to error")
            default_samples = [np.ones(len(X_scaled)) * 0.5 for _ in range(mc_samples)]
            return default_samples
    
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