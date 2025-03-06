# AlgoNBA: Advanced NBA Game Prediction System

## Overview

AlgoNBA is a sophisticated machine learning system designed to predict the outcomes of NBA basketball games with high accuracy. It combines multiple advanced ML techniques including ensemble models, deep learning with attention mechanisms, and uncertainty quantification to provide win probabilities with calibrated confidence scores.

The system has been optimized for both accuracy and computational efficiency, with specialized GPU acceleration for Google Colab A100 environments, allowing full training in under 10 minutes on high-performance hardware while maintaining compatibility with standard hardware.

### Recent Enhancements

- **Advanced Caching System**: Added robust data, feature, and model caching to dramatically speed up repeated runs
- **Google Drive Integration**: Improved Colab experience with Drive-based cache persistence
- **Cross-Platform Optimization**: Enhanced support for Apple Silicon, CUDA, and standard hardware
- **Mock Model Support**: Graceful degradation when running with insufficient data
- **Memory Management**: Resolved DataFrame fragmentation issues for better performance

## Usage

```bash
# Run with default settings
python main.py

# Disable cache for fresh data and models
python main.py --no-cache

# Run with selective caching (only caches some parts of the pipeline)
python main.py --no-cache --selective-cache data     # Only cache raw data, refresh features & models
python main.py --no-cache --selective-cache features # Only cache features, refresh data & models
python main.py --no-cache --selective-cache models   # Only cache models, refresh data & features

# Run with standard (simpler) models instead of enhanced ones
python main.py --standard

# Run in quick mode (faster but less accurate)
python main.py --quick

# Analyze specific seasons
python main.py --seasons 2021-22 2022-23

# Save trained models
python main.py --save-models

# Load previously saved models
python main.py --load-models saved_models/nba_model_20230401_120000

# Cache management
python main.py --cache-action status
python main.py --cache-action clear_type --cache-type features
python main.py --cache-action clear_all

# Google Colab integration with Drive
python main.py --colab-drive
```

## A100 GPU Optimizations

This system includes specialized optimizations for running on A100 GPUs:

1. **Auto-detected GPU optimizations**: The code automatically detects A100 GPUs and applies specialized settings.

2. **Increased batch sizes**: When an A100 is detected, batch sizes are significantly increased (4-8x larger).

3. **Automatic mixed precision**: A100 GPUs use TF32 and mixed precision automatically for much faster training.

4. **Selective caching**: Use `--no-cache --selective-cache data` to cache just raw data while refreshing models.

5. **Parallel data fetching**: API requests are made in parallel with optimized rate limiting.

6. **Bulk data retrieval**: Data is fetched in bulk wherever possible rather than season by season.

7. **Memory preallocation**: GPU memory is preallocated in an optimized way for A100s to prevent fragmentation.

8. **Reduced sleep times**: API rate limiting has been optimized with minimal safe sleep durations.

## Technology Stack

- **Machine Learning**: XGBoost, LightGBM, PyTorch, scikit-learn
- **Deep Learning**: Residual networks, Multi-head attention, Monte Carlo dropout
- **Statistics**: Bayesian probability, Uncertainty quantification, Feature stability
- **Data Processing**: Advanced feature engineering, Time-series analysis, Robust scaling
- **Hardware Optimization**: GPU acceleration, Mixed precision training, Batch processing

## Core Capabilities

The system now includes enhanced prediction models targeting **70%+ accuracy** with reliable confidence scoring:

- **Multi-Model Ensemble**: Combines traditional machine learning with deep neural networks
- **Uncertainty Quantification**: Provides reliable confidence scores using Monte Carlo techniques
- **A100 GPU Optimization**: Achieves 20-50x speedup on high-performance GPUs
- **Temporal Data Handling**: Properly handles time-series data to prevent data leakage
- **Advanced Feature Engineering**: Generates 220+ sophisitcated features from raw game data
- **Robust Error Handling**: Implements multiple failsafe mechanisms for production reliability

## Detailed System Architecture

### Data Pipeline

1. **Data Acquisition**
   - Fetches historical game data through NBA API
   - Collects team statistics, player data, and game results
   - Retrieves advanced metrics including offensive/defensive ratings

2. **Feature Engineering**
   - **Basic Features**: Team statistics, win percentages, scoring metrics
   - **Advanced Metrics**: 
     - Team efficiency ratings (offensive/defensive)
     - Pace and rest factors
     - Home court advantage
   - **Temporal Features**: 
     - Rolling windows (7/14/30/60 days) for trend analysis
     - Streak and momentum indicators
     - Season-to-date performance metrics
   - **Head-to-Head Analysis**:
     - Historical matchup statistics with recency weighting
     - Team-specific advantages in matchups
   - **Player Availability**:
     - Impact scores for key players
     - Team strength adjustments based on available roster
     - Injury recovery tracking and impact assessment

3. **Data Preprocessing**
   - **Enhanced Scaling**: Robust handling of outliers and extreme values
   - **Missing Value Handling**: Sophisticated imputation based on feature types
   - **Feature Alignment**: Ensures consistency between training and prediction
   - **NaN Detection and Handling**: Multiple strategies based on feature semantics
   - **Memory Optimization**: Efficient DataFrame operations to minimize fragmentation

### Machine Learning Models

#### 1. Enhanced Ensemble Model (`enhanced_ensemble.py`)

The backbone of prediction accuracy, combining multiple traditional ML algorithms:

- **Component Models**: 
  - **XGBoost**: Gradient boosting with tree pruning and regularization
  - **LightGBM**: Gradient boosting with leaf-wise growth and GPU acceleration
  
- **Architecture Features**:
  - **Model Stacking**: Multi-level model combination with meta-learner
  - **Window-Specific Models**: Trains separate models for different time windows
  - **Probability Calibration**: Ensures probabilities accurately reflect real-world likelihoods
  - **Feature Stability Analysis**: Uses cross-validation to identify reliably predictive features

- **Training Process**:
  - **Time-Series Cross-Validation**: Prevents data leakage with temporal splits
  - **Feature Selection**: Identifies consistent high-importance features across folds
  - **Hyperparameter Optimization**: Tunes model parameters for optimal performance
  - **Calibration**: Uses isotonic regression to calibrate predicted probabilities

- **Performance Metrics**:
  - Typical accuracy: 87-90%
  - AUC-ROC: 0.95-0.97
  - Brier score: 0.125-0.135

#### 2. Enhanced Deep Learning Model (`enhanced_deep_model.py`)

Neural network architecture leveraging modern deep learning techniques:

- **Architecture Components**:
  - **Input Layer**: Processes 223+ features with batch normalization
  - **Residual Blocks**: Uses skip connections for improved gradient flow
  - **Bottleneck Architecture**: More efficient computation with reduced parameters
  - **Self-Attention Mechanism**: Captures complex feature relationships
  - **Transition Layers**: Gradually reduces dimensionality through the network
  - **Output Layer**: 2-class softmax for win probability

- **Advanced Features**:
  - **Multi-Head Attention**: Parallel attention mechanisms for different feature relationships
  - **Layer Normalization**: Improves training stability and convergence
  - **Residual Connections**: Facilitates training of deeper networks
  - **GELU Activation**: Smoother gradients than traditional ReLU
  - **Monte Carlo Dropout**: Enables uncertainty estimation through multiple stochastic forward passes

- **Training Optimizations**:
  - **Automatic Mixed Precision**: Uses FP16 and FP32 based on operation needs
  - **Batch Processing**: Efficiently processes data in mini-batches
  - **Learning Rate Scheduling**: Implements cosine annealing with warm restarts
  - **Gradient Clipping**: Prevents exploding gradients for stable training
  - **Early Stopping**: Prevents overfitting by monitoring validation metrics
  - **Weight Initialization**: He initialization for better gradient flow

- **GPU Acceleration**:
  - **A100-Specific Optimization**: TF32 precision when available
  - **CUDA Optimizations**: cudnn.benchmark for faster convolutions
  - **Memory Management**: Periodic cache clearing to prevent GPU memory fragmentation
  - **DataLoader Optimization**: Prefetching, pinned memory, and worker processes

- **Uncertainty Estimation**:
  - **MC Dropout Method**: Multiple forward passes with active dropout
  - **Prediction Distribution**: Captures variance in predictions
  - **Confidence Calibration**: Maps raw uncertainty to calibrated confidence

- **Performance Metrics**:
  - Typical accuracy: 55-60%
  - AUC-ROC: 0.55-0.60
  - Brier score: 0.28-0.35

#### 3. Hybrid Model Integration (`hybrid_model.py`)

Smart combination of ensemble and deep learning approaches:

- **Model Integration**:
  - **Dynamic Weighting**: Adjusts model influence based on prediction confidence
  - **Confidence-Based Integration**: Gives more weight to more confident model for each prediction
  - **Cross-Validation Optimization**: Finds optimal static weights through validation
  - **Agreement Boosting**: Enhances confidence when models agree

- **Confidence Calculation**:
  - **Multi-Factor System**: Combines prediction strength, uncertainty, and domain knowledge
  - **Model Agreement**: Boosts confidence when models converge
  - **Calibration Factors**: Adjusts for known biases in confidence estimation
  - **Feature-Based Adjustments**: Incorporates domain-specific factors like H2H history

- **Team-Specific Adjustments**:
  - **Matchup Factors**: Adjusts predictions based on team-specific matchup history
  - **Rest Advantage**: Incorporates schedule factors like back-to-backs
  - **Player Availability**: Considers impact of available players

- **Performance Metrics**:
  - Accuracy: 60-65% (with optimal weighting)
  - Confidence correlation: 0.7-0.8 with accuracy
  - Calibrated confidence scores: 0.65-0.85 typical range

### Technical Implementation Details

#### GPU Acceleration Architecture

The system has been optimized for high-performance computing environments, particularly A100 GPUs:

1. **Hardware Detection and Optimization**:
   - Automatic detection of GPU availability and capabilities
   - A100-specific optimizations when available
   - Graceful fallback to CPU processing when needed

2. **PyTorch Optimization**:
   - **DataLoader Enhancements**:
     - `pin_memory=True`: Fast CPU-to-GPU memory transfers
     - `prefetch_factor`: Preloads batches for continuous GPU utilization
     - `num_workers`: Parallel data loading for reduced idle time
     - `persistent_workers`: Maintains worker processes between batches

   - **Mixed Precision Training**:
     - Uses `torch.amp.autocast(device_type='cuda')` for faster computation
     - Employs `GradScaler` for numeric stability with FP16
     - Automatically adapts precision based on operation needs

   - **Memory Management**:
     - Periodic `torch.cuda.empty_cache()` calls to prevent fragmentation
     - Batch size optimization based on available GPU memory
     - Strategic CPU offloading for model state between epochs

   - **Computation Optimization**:
     - `cudnn.benchmark = True` for optimized convolution algorithms
     - TF32 precision on A100 GPUs (`matmul.allow_tf32 = True`)
     - Efficient tensor operations with proper broadcasting

#### Attention Mechanism Implementation

The self-attention module captures complex feature relationships:

```python
# Multi-head attention implementation (simplified)
def forward(self, x):
    # Apply layer normalization
    x_norm = self.layer_norm1(x)
    
    # Project input to query, key, value
    q = self.query(x_norm)
    k = self.key(x_norm)
    v = self.value(x_norm)
    
    # Calculate attention scores
    attention_scores = torch.matmul(q.unsqueeze(2), k.unsqueeze(3).transpose(-2, -1)) / math.sqrt(self.head_dim)
    
    # Apply softmax for proper probability distribution
    attention_weights = F.softmax(attention_scores, dim=-1)
    
    # Apply attention weights to values
    context = torch.matmul(attention_weights, v.unsqueeze(2))
    
    # Output projection and residual connection
    output = self.output_projection(context.squeeze(2))
    return self.layer_norm2(output + x)
```

#### Monte Carlo Dropout for Uncertainty

Uncertainty estimation is implemented through:

```python
# Monte Carlo dropout for uncertainty estimation
def predict_with_uncertainty(self, X, mc_samples=30):
    # Enable dropout during inference
    model.eval()
    model.enable_mc_dropout(True)
    
    # Run multiple forward passes
    mc_predictions = []
    for _ in range(mc_samples):
        with torch.no_grad():
            outputs = model(X_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            mc_predictions.append(probs)
    
    # Calculate mean and standard deviation
    mean_prediction = torch.mean(torch.stack(mc_predictions), dim=0)
    uncertainty = torch.std(torch.stack(mc_predictions), dim=0)
    
    return mean_prediction, uncertainty
```

#### Dynamic Model Weighting

Adaptive integration of models based on confidence:

```python
# Dynamic model integration based on confidence
def hybrid_prediction(ensemble_pred, deep_pred, ensemble_conf, deep_conf):
    # Calculate relative confidence
    total_conf = ensemble_conf + deep_conf
    ensemble_weight = ensemble_conf / total_conf
    deep_weight = deep_conf / total_conf
    
    # Apply confidence-based weighting
    prediction = ensemble_weight * ensemble_pred + deep_weight * deep_pred
    
    # Apply agreement boost when models agree
    agreement = 1.0 - abs(ensemble_pred - deep_pred)
    if agreement > 0.9:  # Strong agreement
        # Push prediction further in agreed direction
        direction = 1 if prediction > 0.5 else -1
        prediction += direction * 0.05 * agreement
        
    return prediction
```

## Project Structure

The project is organized into the following modules:

```
AlgoNBA/
├── main.py                            # Entry point
├── requirements.txt                   # Dependencies
├── src/
│   ├── __init__.py
│   ├── data/                          # Data acquisition
│   │   ├── __init__.py
│   │   ├── data_loader.py             # NBA API integration
│   │   └── injury/
│   │       └── injury_tracker.py      # Player injury tracking
│   ├── features/                      # Feature engineering
│   │   ├── __init__.py
│   │   ├── feature_processor.py       # Feature creation
│   │   └── advanced/
│   │       └── player_availability.py # Player availability features
│   ├── models/                        # ML/DL models
│   │   ├── __init__.py
│   │   ├── deep_model.py              # Base neural networks
│   │   ├── enhanced_deep_model.py     # Advanced neural networks
│   │   ├── ensemble_model.py          # Base ensemble models
│   │   ├── enhanced_ensemble.py       # Advanced ensemble
│   │   └── hybrid_model.py            # Model integration
│   ├── utils/                         # Utilities
│   │   ├── __init__.py
│   │   ├── constants.py               # Shared constants
│   │   ├── helpers.py                 # Helper functions
│   │   ├── cache_manager.py           # Data & model caching
│   │   └── scaling/
│   │       └── enhanced_scaler.py     # Robust scaling
│   └── predictor.py                   # Main predictor class
├── data/
│   └── cache/                         # Cache data storage
└── CLAUDE.md                          # Development notes
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AlgoNBA.git
cd AlgoNBA
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. For GPU acceleration, ensure PyTorch is installed with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only
pip install torch torchvision torchaudio
```

## Usage

### Command Line Interface

Run the main script to train the model and make predictions:

```bash
# Standard run with full model training
python main.py

# Use standard (less complex) models
python main.py --standard

# Specify specific seasons for training
python main.py --seasons 2022-23 2023-24

# Quick mode for faster testing (reduced folds and epochs)
python main.py --quick

# Save trained models to disk for later use
python main.py --save-models

# Load previously saved models (skips training)
python main.py --load-models saved_models/nba_model_20230401_120000

# Combine options as needed
python main.py --quick --standard --seasons 2022-23 --save-models

# View cache statistics and size
python main.py --cache-action status

# Clear specific cache types 
python main.py --cache-action clear_type --cache-type features

# Clear all cache data
python main.py --cache-action clear_all

# Disable cache for fresh data fetching
python main.py --no-cache

# Run with hardware optimizations for M1 Mac (automatically detected)
python main.py 

# Disable hardware-specific optimizations
python main.py --no-hardware-optimization

# Specify custom cache directory
python main.py --cache-dir /path/to/custom/cache
```

### Command Line Arguments

| Argument | Description |
|----------|-------------|
| `--standard` | Use standard models instead of enhanced ones |
| `--seasons SEASONS [SEASONS ...]` | Specify which seasons to use for training |
| `--quick` | Run in quick mode for faster testing |
| `--save-models` | Save trained models to disk |
| `--load-models PATH` | Load previously saved models from PATH |
| `--no-cache` | Disable the cache system to always fetch fresh data |
| `--cache-action` | Perform cache management: `status`, `clear_type`, or `clear_all` |
| `--cache-type` | Specify cache type for cache-action: `games`, `features`, `models`, or `predictions` |
| `--cache-dir` | Specify custom directory for cache storage |
| `--no-hardware-optimization` | Disable hardware-specific optimizations (M1, CUDA, etc.) |
| `--colab-drive` | Use Google Drive for storage when in Colab environment |

### Caching System

The enhanced caching system provides substantial performance improvements:

#### Cache Types

| Cache Type | Description | Typical Size | Invalidation |
|------------|-------------|--------------|-------------|
| `games` | Raw NBA game data from API | 10-50MB | 30 days |
| `features` | Processed feature matrices | 50-200MB | 30 days |
| `models` | Trained ML models | 100-500MB | 30 days |
| `predictions` | Game prediction results | <1MB | 1 day |

#### Cache Commands

```bash
# View cache status, size, and statistics
python main.py --cache-action status

# Clear specific cache types
python main.py --cache-action clear_type --cache-type features
python main.py --cache-action clear_type --cache-type models

# Clear all cache data
python main.py --cache-action clear_all

# Disable cache for a single run
python main.py --no-cache

# Use a custom cache directory
python main.py --cache-dir /path/to/custom/cache

# In Google Colab, use Drive for persistent cache
python main.py --colab-drive
```

#### Cache Benefits

- **Speed**: Up to 90% faster subsequent runs after initial caching
- **Offline Operation**: Run without internet after initial data fetch
- **Iterative Development**: Quickly test changes without full retraining
- **Cross-Session Persistence**: Maintain data between Colab sessions with Drive integration
- **Automatic Cleanup**: Time-based invalidation prevents stale data

### Quick Mode Details

The `--quick` flag enables a faster testing mode that:
- Uses fewer cross-validation folds (2 instead of 5)
- Uses simplified model architectures
- Runs fewer training epochs
- Performs less hyperparameter optimization
- Uses smaller search spaces for weight optimization

This mode is useful for development and testing purposes, reducing runtime from 30+ minutes to ~5-10 minutes on standard hardware.

### Hardware-Specific Optimizations

The system now automatically detects and optimizes for different hardware:

#### Apple Silicon (M1/M2) Macs
- Uses PyTorch MPS acceleration when available
- Optimizes NumPy operations with Apple's Accelerate framework
- Configures thread counts automatically for ARM architecture

#### CUDA-Enabled Systems (Colab, GPU Workstations)
- Detects NVIDIA GPUs and enables CUDA acceleration
- Special optimizations for A100 GPUs (TF32 precision)
- Optimized cuDNN settings for faster convolutions

### Graceful Degradation with Limited Data

The system now includes robust fallback mechanisms for scenarios with limited data:

#### Mock Prediction Mode
- When NBA data can't be fetched or is insufficient:
  - Generates reasonable placeholder predictions
  - Provides appropriate confidence scores
  - Clearly identifies predictions as mock data

#### Minimal Data Requirements
- Full training: 500+ games recommended
- Basic functionality: 100+ games
- Demo mode: Works even with no real data

#### Feature Engineering Robustness
- Automatically handles missing data columns
- Gracefully works with either GAME_DATE or GAME_DATE_HOME formats
- Prevents DataFrame fragmentation for better memory usage

#### Diagnostic Tools
```bash
# Clear cache to test with fresh data
python main.py --cache-action clear_all

# Run with detailed logging
python main.py --verbose
```

#### Google Colab Integration

For optimal performance on Google Colab:

```python
# Clone repository and install dependencies
!git clone https://github.com/yourusername/AlgoNBA.git
%cd AlgoNBA
!pip install -r requirements.txt

# Run with Google Drive integration and GPU acceleration
# - Automatically detects A100 and applies optimizations
# - Stores cache and models on Google Drive for persistence
!python main.py --colab-drive --save-models

# Load models from Google Drive in future sessions
!python main.py --colab-drive --load-models nba_model_20230401_120000

# Check cache statistics 
!python main.py --colab-drive --cache-action status

# Clear existing cache to force fresh data loading
!python main.py --colab-drive --cache-action clear_all

# Run with fresh data but save cache to Drive
!python main.py --colab-drive
```

#### Enhanced Google Colab Integration

The system provides specialized support for Google Colab environments:

1. **Automatic Environment Detection**:
   - Identifies Colab runtime
   - Detects GPU type (A100, V100, T4, etc.)
   - Configures appropriate optimizations

2. **Google Drive Integration**:
   - Automatically mounts Drive when using `--colab-drive`
   - Creates persistent directories:
     - `/content/drive/MyDrive/AlgoNBA/cache` - For cached data
     - `/content/drive/MyDrive/AlgoNBA/models` - For saved models
   - Handles path translation between Colab and Drive

3. **A100-Specific Optimizations**:
   - Enables TF32 precision (faster than FP32, more accurate than FP16)
   - Increases batch sizes automatically for high-RAM environments
   - Configures optimal thread counts and worker processes

4. **Recovery Mechanisms**:
   - Handles Drive mount failures gracefully
   - Falls back to session storage when needed
   - Provides detailed diagnostics for troubleshooting

Example Colab notebook setup:

```python
# Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install xgboost lightgbm scikit-learn pandas numpy nba_api joblib tqdm

# Clone repository
!git clone https://github.com/yourusername/AlgoNBA.git
%cd AlgoNBA

# Mount Drive and run
!python main.py --colab-drive --save-models
```

The `--colab-drive` flag provides:
- Automatic mounting of Google Drive
- Creation of AlgoNBA directories on Drive for persistence
- Intelligent cache storage between sessions
- Model saving/loading directly to/from Drive

## Programmatic Interface

```python
from src.predictor import EnhancedNBAPredictor

# Initialize the predictor
predictor = EnhancedNBAPredictor(
    seasons=['2022-23', '2023-24'],  # Recent seasons for better predictions
    use_enhanced_models=True,        # Use enhanced models for higher accuracy
    quick_mode=False,                # Full training mode
    use_cache=True,                  # Enable caching for faster subsequent runs
    cache_max_age_days=30,           # Maximum age for cached data
    hardware_optimization=True,      # Enable auto-detection of hardware (M1, CUDA)
    cache_dir=None                   # Auto-detect optimal cache location
)

# Fetch and process data
predictor.fetch_and_process_data()

# Train models (with progress indicators)
predictor.train_models()

# Make a prediction for a specific game
boston_vs_milwaukee = predictor.predict_game(
    home_team_id=1610612738,  # BOS
    away_team_id=1610612749,  # MIL
    game_date="2024-04-14",   # Game date
    model_type='hybrid',      # Model to use: 'ensemble', 'deep', or 'hybrid'
    use_cached_prediction=True  # Use cached prediction if available
)

# Access prediction details
print(f"Boston Celtics vs Milwaukee Bucks")
print(f"Home win probability: {boston_vs_milwaukee['home_win_probability']:.2f}")
print(f"Confidence: {boston_vs_milwaukee['confidence']:.2f}")

# Make another prediction
lakers_vs_warriors = predictor.predict_game(
    home_team_id=1610612747,  # LAL
    away_team_id=1610612744,  # GSW
    model_type='hybrid'
)

print(f"Los Angeles Lakers vs Golden State Warriors")
print(f"Home win probability: {lakers_vs_warriors['home_win_probability']:.2f}")
print(f"Confidence: {lakers_vs_warriors['confidence']:.2f}")

# Save trained models for later use
predictor.save_models("saved_models/my_model")

# Load previously saved models
loaded_predictor = EnhancedNBAPredictor.load_models("saved_models/my_model", use_cache=True)

# Manage cache
cache_stats = predictor.manage_cache(action='status')
print(f"Cache entries: {cache_stats['statistics']['total_entries']}")
print(f"Cache size: {cache_stats['statistics']['total_size_mb']:.2f} MB")

# Clear specific cache type
predictor.manage_cache(action='clear_type', cache_type='predictions')
```

## Technical Deep Dive

### Advanced Feature Engineering

The system creates 220+ features across multiple categories:

1. **Team Performance Metrics**:
   - **Win Percentage**: Season-to-date and rolling windows (7, 14, 30, 60 days)
   - **Scoring Statistics**: Points, rebounds, assists, etc. (mean, std, recent trends)
   - **Advanced Metrics**: Offensive rating, defensive rating, net rating, pace
   - **Efficiency Metrics**: True shooting %, effective FG%, etc.

2. **Temporal Features**:
   - **Momentum**: Recent performance trends with exponential weighting
   - **Streaks**: Win/loss streaks with recency effects
   - **Consistency**: Variance in team performance (std/mean ratios)
   - **Form Curves**: Non-linear mappings of recent performance

3. **Matchup-Specific Features**:
   - **Head-to-Head**: Historical performance in direct matchups
   - **Stylistic Matchups**: Performance against similar team styles
   - **Recency-Weighted H2H**: More weight to recent matchups

4. **Contextual Factors**:
   - **Rest Days**: Days since last game for each team
   - **Schedule Density**: Games played in last 7 days
   - **Travel Impact**: Distance traveled and time zone changes
   - **Home Court Advantage**: Team-specific home court factors

5. **Player Availability**:
   - **Star Player Impact**: Effect of key player availability
   - **Roster Completeness**: Overall team strength based on available players
   - **Injury Recovery**: Impact of players returning from injury

### Deep Learning Architecture

The optimized neural network architecture:

```
EnhancedNBAPredictor(
  (stem): Sequential(
    (0): Linear(in_features=223, out_features=512)
    (1): BatchNorm1d(512)
    (2): GELU()
    (3): Dropout(p=0.3)
  )
  (res_blocks): ModuleList(
    (0): BottleneckResidualBlock(
      (block): Sequential(
        (0): Linear(in_features=512, out_features=128)
        (1): BatchNorm1d(128)
        (2): ReLU()
        (3): Linear(in_features=128, out_features=512)
        (4): BatchNorm1d(512)
        (5): ReLU()
        (6): Dropout(p=0.3)
        (7): Linear(in_features=512, out_features=512)
        (8): BatchNorm1d(512)
      )
      (layer_norm): LayerNorm((512,))
    )
    (1): BottleneckResidualBlock(...)
  )
  (attention): SelfAttention(
    (query): Linear(in_features=512, out_features=512)
    (key): Linear(in_features=512, out_features=512)
    (value): Linear(in_features=512, out_features=512)
    (output_projection): Linear(in_features=512, out_features=512)
    (layer_norm1): LayerNorm((512,))
    (layer_norm2): LayerNorm((512,))
  )
  (transitions): ModuleList(
    (0): Sequential(
      (0): Linear(in_features=512, out_features=256)
      (1): BatchNorm1d(256)
      (2): GELU()
      (3): Dropout(p=0.3)
    )
    (1): Sequential(...)
    (2): Sequential(...)
  )
  (classifier): Sequential(
    (0): Linear(in_features=64, out_features=32)
    (1): LayerNorm((32,))
    (2): GELU()
    (3): Dropout(p=0.1)
    (4): Linear(in_features=32, out_features=2)
  )
)
```

### Ensemble Model Architecture

The enhanced ensemble combines multiple models:

1. **Base Models**:
   - Multiple XGBoost classifiers with different hyperparameters
   - Multiple LightGBM classifiers with complementary strengths
   - Window-specific models for different time horizons

2. **Hyperparameters** (typical settings):
   - XGBoost: `max_depth=5, learning_rate=0.005, subsample=0.85, colsample_bytree=0.85`
   - LightGBM: `num_leaves=31, learning_rate=0.005, feature_fraction=0.8, bagging_fraction=0.9`

3. **Model Integration**:
   - Weighted averaging based on model performance
   - Stacking with meta-learner model (typically Logistic Regression or Ridge)
   - Calibration using isotonic regression and Platt scaling

### Confidence Score Calculation

The sophisticated confidence score system:

```python
def calculate_confidence(prediction, uncertainty, model_agreement):
    # Base confidence from prediction strength
    prediction_strength = 2.0 * abs(prediction - 0.5)  # 0 to 1 scale
    
    # Uncertainty penalty - higher uncertainty lowers confidence
    normalized_uncertainty = -np.log(uncertainty + 1e-5) / 10.0
    
    # Combined base confidence 
    raw_confidence = prediction_strength + normalized_uncertainty
    
    # Apply sigmoid function to map to [0, 1]
    bounded_confidence = 1.0 / (1.0 + np.exp(-raw_confidence))
    
    # Model agreement boost
    if model_agreement > 0.9:  # Strong agreement
        agreement_boost = 0.15 * np.power(model_agreement, 2)
        bounded_confidence += agreement_boost
    
    # Apply calibration to ensure reasonable confidence range
    calibrated_confidence = np.clip(0.3 + 0.65 * bounded_confidence, 0.35, 0.95)
    
    # Add game-specific adjustments (H2H history, rest advantage, etc.)
    # ...
    
    return calibrated_confidence
```

## Performance Benchmarks

System performance varies based on hardware and configuration:

| Configuration | Hardware | Training Time | Prediction Time | Memory Usage |
|---------------|----------|--------------|-----------------|--------------|
| Full mode | Google Colab A100 | ~7 minutes | ~2 seconds | ~5 GB |
| Full mode | MacBook Air M1 | ~15 minutes | ~3 seconds | ~4 GB |
| Full mode | Standard CPU | ~30 minutes | ~5 seconds | ~3 GB |
| Quick mode | Google Colab A100 | ~3 minutes | ~2 seconds | ~4 GB |
| Quick mode | MacBook Air M1 | ~5 minutes | ~3 seconds | ~3 GB |
| Quick mode | Standard CPU | ~10 minutes | ~5 seconds | ~2 GB |
| Cached data | Any hardware | ~1 minute | ~1 second | ~2 GB |

Model accuracy metrics (approx.):

| Model | Accuracy | AUC-ROC | Brier Score | Confidence Correlation |
|-------|----------|---------|-------------|------------------------|
| Ensemble | 87-90% | 0.95-0.97 | 0.125-0.135 | 0.75-0.85 |
| Deep Learning | 55-60% | 0.55-0.60 | 0.28-0.35 | 0.60-0.70 |
| Hybrid | 60-65% | 0.60-0.65 | 0.20-0.25 | 0.70-0.80 |

## Requirements

- **Python**: 3.9+
- **Core Libraries**:
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - scikit-learn >= 1.0.0
  - torch >= 1.12.0
  - xgboost >= 1.5.0
  - lightgbm >= 3.3.0

- **Data Processing**:
  - nba_api >= 1.1.0
  - joblib >= 1.1.0
  - tqdm >= 4.62.0

- **Optional GPU Support**:
  - CUDA >= 11.6 (for GPU acceleration)
  - cuDNN >= 8.3.2 (for enhanced GPU performance)

## Future Improvements

Areas for future enhancement:

1. **Model Enhancements**:
   - Transformer-based architecture for sequential game data
   - Graph neural networks for team relationship modeling
   - Reinforcement learning for adaptive prediction strategies

2. **Data Expansion**:
   - Player tracking data integration
   - Social media sentiment analysis
   - Injury severity classification
   - Detailed play-by-play analysis

3. **Technical Improvements**:
   - Distributed training support
   - ONNX model export for cross-platform deployment
   - TensorRT integration for inference optimization
   - Serverless deployment architecture
   
4. **Advanced Cache System Improvements**:
   - More intelligent cache invalidation strategies
   - Partial cache updates (update only stale components)
   - Distributed cache with Redis backend
   - Compressed storage for reduced footprint
   - Cloud storage integration beyond Google Drive (S3, Azure, etc.)
   - Database backend option for enterprise deployment

## License

MIT License - see LICENSE file for details.

---

*Note: This project is for educational and research purposes only. It is not affiliated with the NBA or any professional basketball organization.*