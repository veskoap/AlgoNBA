"""
Main predictor class that orchestrates the NBA prediction workflow.
"""
import os
import pickle
import joblib
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Set
from datetime import datetime, date, timedelta

from src.data.data_loader import NBADataLoader
from src.features.feature_processor import NBAFeatureProcessor
from src.features.advanced.player_availability import PlayerAvailabilityProcessor
from src.models.ensemble_model import NBAEnsembleModel
from src.models.enhanced_ensemble import NBAEnhancedEnsembleModel
from src.models.deep_model import DeepModelTrainer
from src.models.enhanced_deep_model import EnhancedDeepModelTrainer
from src.models.hybrid_model import HybridModel
from src.utils.constants import DEFAULT_LOOKBACK_WINDOWS, FEATURE_REGISTRY


class EnhancedNBAPredictor:
    """Main class for NBA prediction system."""
    
    def __init__(self, seasons: List[str], lookback_windows: List[int] = None, 
                use_enhanced_models: bool = True, quick_mode: bool = False,
                use_cache: bool = True, cache_max_age_days: int = 30,
                cache_dir: str = None, hardware_optimization: bool = True,
                selective_cache_components: Dict[str, bool] = None):
        """
        Initialize the NBA prediction system.
        
        Args:
            seasons: List of NBA seasons in format 'YYYY-YY' (e.g., '2022-23')
                    More recent seasons will provide better predictions
            lookback_windows: List of day windows for rolling statistics 
                              (default: [7, 14, 30, 60])
            use_enhanced_models: Whether to use enhanced models or standard ones.
                                 Enhanced models offer higher accuracy but may take
                                 longer to train.
            quick_mode: Whether to run in quick test mode with simplified models.
                        When True:
                        - Uses fewer cross-validation folds (2 instead of 5)
                        - Uses simpler model architectures
                        - Runs fewer training epochs
                        - Performs less hyperparameter optimization
                        Useful for development and testing, but for highest accuracy,
                        set to False.
            use_cache: Whether to use cache for data loading and feature processing
            cache_max_age_days: Maximum age of cached data in days
            cache_dir: Custom directory for cache storage (if None, will auto-detect)
            hardware_optimization: Whether to apply hardware-specific optimizations
                                  (M1/Apple Silicon, CUDA, etc.)
            selective_cache_components: Dictionary defining which components should use cache.
                                       Example: {'data': True, 'features': False, 'models': False}
                                       If None, all components follow use_cache setting.
        """
        self.seasons = seasons
        self.lookback_windows = lookback_windows or DEFAULT_LOOKBACK_WINDOWS
        self.use_enhanced_models = use_enhanced_models
        self.quick_mode = quick_mode
        self.use_cache = use_cache
        self.cache_max_age_days = cache_max_age_days
        self.hardware_optimization = hardware_optimization
        self.device = 'cpu'  # Default device
        self.is_colab = False  # Default Colab detection
        
        # Initialize caching
        from src.utils.cache_manager import CacheManager
        self.cache_manager = CacheManager(cache_dir=cache_dir)
        
        # Set up selective caching if provided
        default_cache_state = use_cache
        self.selective_cache = selective_cache_components or {
            'data': default_cache_state,
            'features': default_cache_state,
            'models': default_cache_state,
            'predictions': default_cache_state
        }
        
        # Auto-detect and configure hardware optimizations if requested
        if hardware_optimization:
            self._configure_hardware_optimizations()
        
        # Initialize components with selective caching support
        self.data_loader = NBADataLoader(
            use_cache=self.selective_cache.get('data', use_cache), 
            cache_max_age_days=cache_max_age_days, 
            cache_dir=cache_dir
        )
        self.feature_processor = NBAFeatureProcessor(self.lookback_windows)
        self.player_processor = PlayerAvailabilityProcessor()
        
        # Initialize appropriate models based on flag
        if use_enhanced_models:
            if self.quick_mode:
                # Use simplified models for quick testing
                self.ensemble_model = NBAEnhancedEnsembleModel(
                    use_calibration=False, 
                    use_stacking=False,
                    n_folds=2  # Use fewer folds for faster testing
                )
                self.deep_model_trainer = EnhancedDeepModelTrainer(
                    use_residual=False, 
                    use_attention=False, 
                    use_mc_dropout=False,
                    epochs=5,  # Very few epochs for quick testing
                    hidden_layers=[64, 32],  # Simplified architecture
                    n_folds=2  # Fewer folds for faster testing
                )
                self.hybrid_model = HybridModel(
                    ensemble_model=self.ensemble_model,
                    deep_model=self.deep_model_trainer,
                    quick_mode=True
                )
            else:
                # Use full models
                self.ensemble_model = NBAEnhancedEnsembleModel()
                self.deep_model_trainer = EnhancedDeepModelTrainer()
                self.hybrid_model = HybridModel()
        else:
            self.ensemble_model = NBAEnsembleModel()
            self.deep_model_trainer = DeepModelTrainer()
            self.hybrid_model = None
            
        # Storage for models and data
        self.games = None
        self.advanced_metrics = None
        self.stats_df = None
        self.features = None
        self.targets = None
        
        # Create a unique cache key for this predictor configuration
        self.config_hash = self._generate_config_hash()
        
    def _configure_hardware_optimizations(self):
        """Configure optimizations for the current hardware platform."""
        import platform
        import os
        
        # Detect platform specifics
        system = platform.system()
        machine = platform.machine()
        
        # Check for TPU availability first (highest priority)
        try:
            # Check if we should skip TPU detection completely (safer option)
            if os.environ.get('ALGONBA_DISABLE_TPU') == '1':
                print("TPU detection disabled by ALGONBA_DISABLE_TPU=1 environment variable")
                self.is_tpu = False
                # Skip TPU detection entirely
            else:
                # Try a more conservative TPU detection approach
                try:
                    # Import torch_xla but don't set any environment variables initially
                    import torch_xla
                    has_torch_xla = True
                    print("torch_xla package detected")
                    
                    # Default to not using TPU
                    self.is_tpu = False
                    
                    # Check if this is called with --use-tpu flag
                    tpu_explicitly_requested = False
                    try:
                        import sys
                        tpu_explicitly_requested = '--use-tpu' in sys.argv
                    except Exception:
                        pass
                    
                    if tpu_explicitly_requested:
                        print("TPU explicitly requested but will run in CPU-only mode for stability")
                        print("To force TPU usage, set environment variable ALGONBA_FORCE_TPU=1")
                        
                        # Only try TPU if explicitly forced via environment variable
                        if os.environ.get('ALGONBA_FORCE_TPU') == '1':
                            print("Force TPU mode enabled by ALGONBA_FORCE_TPU=1 environment variable")
                            try:
                                # Set required environment variables
                                os.environ['PJRT_DEVICE'] = 'TPU'
                                os.environ['XLA_USE_BF16'] = '1'
                                
                                # Import needed modules
                                import torch_xla.core.xla_model as xm
                                
                                # Create device directly without querying topology
                                self.device = xm.xla_device()
                                self.is_tpu = True
                                
                                # Log TPU information
                                print(f"Using TPU device: {self.device}")
                                print("TPU optimizations enabled via direct initialization")
                            except Exception as e:
                                print(f"Failed to initialize TPU, falling back to CPU: {e}")
                                self.is_tpu = False
                    else:
                        print("TPU capability detected but not enabled (use --use-tpu flag)")
                except ImportError:
                    has_torch_xla = False
                    print("torch_xla package not available, TPU support disabled")
                    self.is_tpu = False
        except Exception as e:
            # Catch any unexpected errors in TPU detection
            print(f"Error during TPU setup: {e}")
            print("Falling back to CPU for safety")
            self.is_tpu = False
        
        # Apply Apple Silicon (M1/M2) optimizations if TPU not available
        if system == 'Darwin' and machine == 'arm64':
            os.environ['VECLIB_MAXIMUM_THREADS'] = str(os.cpu_count())
            
            try:
                import torch
                # Use MPS (Metal Performance Shaders) if available
                if torch.backends.mps.is_available():
                    print("Apple Silicon (M1/M2) detected: Using MPS for acceleration")
                    self.device = torch.device('mps')
                else:
                    print("Apple Silicon (M1/M2) detected: MPS not available, using CPU")
                    self.device = torch.device('cpu')
                    
                # Make sure PyTorch is optimized for ARM
                if hasattr(torch, 'set_num_threads'):
                    torch.set_num_threads(os.cpu_count())
            except ImportError:
                print("PyTorch not available, using CPU for computation")
                self.device = 'cpu'
                
        # Check for CUDA GPU availability (for Linux, Windows, Intel Macs)
        elif system in ['Linux', 'Windows'] or (system == 'Darwin' and machine != 'arm64'):
            try:
                import torch
                if torch.cuda.is_available():
                    # Detect if we're on Colab with an A100
                    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else ""
                    
                    if 'A100' in gpu_name:
                        print("NVIDIA A100 GPU detected: Applying A100-specific optimizations")
                        # Enable TF32 precision for A100 GPUs
                        torch.backends.cuda.matmul.allow_tf32 = True
                        torch.backends.cudnn.allow_tf32 = True
                    else:
                        print(f"CUDA GPU detected ({gpu_name}): Enabling GPU acceleration")
                    
                    # Enable cuDNN benchmarking for faster convolutions
                    torch.backends.cudnn.benchmark = True
                    self.device = torch.device('cuda')
                else:
                    print("No CUDA GPU detected, using CPU for computation")
                    self.device = torch.device('cpu')
            except ImportError:
                print("PyTorch not available, using CPU for computation")
                self.device = 'cpu'
                
        # Handle Colab-specific environment optimizations
        try:
            import google.colab
            # We're in Colab - set up appropriate environment
            print("Google Colab environment detected")
            self.is_colab = True
            
            # Check if we have a high-RAM environment
            import psutil
            total_ram = psutil.virtual_memory().total / (1024 ** 3)  # GB
            if total_ram > 25:  # High-RAM Colab instance
                print(f"High-RAM environment detected ({total_ram:.1f} GB): Optimizing for memory usage")
                # Can use larger batch sizes, etc.
            
        except ImportError:
            self.is_colab = False
        
    def _generate_config_hash(self) -> str:
        """
        Generate a unique hash of the current configuration for cache lookups.
        
        Returns:
            str: Hash representing this configuration
        """
        import hashlib
        import json
        
        # Collect configuration parameters
        config = {
            'seasons': sorted(self.seasons),
            'lookback_windows': sorted(self.lookback_windows),
            'use_enhanced_models': self.use_enhanced_models,
            'quick_mode': self.quick_mode
        }
        
        # Generate a stable string representation and hash it
        config_str = json.dumps(config, sort_keys=True)
        hash_obj = hashlib.md5(config_str.encode())
        
        return hash_obj.hexdigest()
        
    def fetch_and_process_data(self) -> None:
        """Fetch NBA data and process it into features."""
        # Check for cached processed features first when features caching is enabled
        if self.use_cache and self.selective_cache.get('features', True):
            cache_params = {
                'config_hash': self.config_hash,
                'lookback_windows': sorted(self.lookback_windows),
                'seasons': sorted(self.seasons)
            }
            
            cached_features = self.cache_manager.get_cache('features', cache_params)
            if cached_features is not None and not self.cache_manager.is_cache_stale('features', cache_params, self.cache_max_age_days):
                print("Using cached processed features...")
                self.games, self.advanced_metrics, self.stats_df, self.features, self.targets = cached_features
                print(f"Loaded cached features: {len(self.features)} samples with {len(self.features.columns)} features")
                return
                
            print("No valid feature cache found, processing data...")
        
        # Load data
        print("Fetching NBA game data...")
        self.games, self.advanced_metrics = self.data_loader.fetch_games(self.seasons)
        
        # Calculate team statistics
        print("Calculating team statistics...")
        self.stats_df = self.feature_processor.calculate_team_stats(self.games)
        
        # Calculate player availability impact (new)
        print("Calculating player availability impact...")
        player_features = self.player_processor.calculate_player_impact_features(self.games)
        
        # First check if player_features has data before attempting merge
        if player_features is not None and not player_features.empty:
            # Log merge details more verbosely
            print(f"Merging {len(player_features)} player availability records with features")
            print(f"Features columns before merge: {self.stats_df.columns[:5]}...")
            print(f"Home player data columns: {player_features.columns[:5]}...")
            
            # Ensure all key merge columns exist
            missing_cols = []
            for col in ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']:
                if col not in player_features.columns:
                    missing_cols.append(col)
                    print(f"Missing column {col} in player_features - creating it")
                    # Add missing column with default values
                    if col == 'GAME_DATE':
                        player_features[col] = pd.to_datetime('2022-10-01')
                    else:
                        player_features[col] = 0
                        
            if missing_cols:
                print(f"Created {len(missing_cols)} missing columns in player_features")
            
            # Ensure both dataframes have matching dtypes for merge columns
            for col in ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']:
                if col in player_features.columns and col in self.stats_df.columns:
                    # Ensure consistent types for merge columns
                    try:
                        if player_features[col].dtype != self.stats_df[col].dtype:
                            print(f"Converting {col} in player_features from {player_features[col].dtype} to {self.stats_df[col].dtype}")
                            player_features[col] = player_features[col].astype(self.stats_df[col].dtype)
                    except Exception as e:
                        print(f"Error converting column {col}: {e} - using safer conversion")
                        # Use a safer conversion method
                        if col == 'GAME_DATE':
                            player_features[col] = pd.to_datetime(player_features[col])
                        else:
                            player_features[col] = pd.to_numeric(player_features[col], errors='coerce').fillna(0).astype(int)
            
            # Merge player features with team stats
            original_len = len(self.stats_df)
            try:
                # Store original stats_df for fallback
                original_stats_df = self.stats_df.copy()
                
                # Attempt merge
                merged_df = self.stats_df.merge(
                    player_features,
                    on=['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'],
                    how='left'
                )
                merged_len = len(merged_df)
                
                if merged_len != original_len:
                    print(f"Warning: Merge changed dataframe length from {original_len} to {merged_len}")
                    # If merge significantly changed the shape, add availability columns to original DataFrame
                    if abs(merged_len - original_len) > 0.1 * original_len:
                        print(f"Significant change in data size detected. Using alternative merge approach")
                        # Restore original and add columns individually
                        self.stats_df = original_stats_df
                        for col in player_features.columns:
                            if col not in ['GAME_DATE', 'TEAM_ID_HOME', 'TEAM_ID_AWAY'] and col not in self.stats_df.columns:
                                print(f"Adding column {col} with default values")
                                # Add missing column with default values
                                if 'IMPACT' in col or 'MOMENTUM' in col:
                                    self.stats_df[col] = 1.0
                                else:
                                    self.stats_df[col] = 0.0
                    else:
                        # Small change, accept the merge
                        self.stats_df = merged_df
                else:
                    # Merge was successful with no change in length
                    self.stats_df = merged_df
                    print(f"Successfully merged player data with {len(self.stats_df)} features")
                    
            except Exception as e:
                print(f"Error merging player availability data: {e}")
                # Handle the error by continuing without merging
                # Add essential columns with default values
                for col in ['PLAYER_IMPACT_HOME', 'PLAYER_IMPACT_AWAY', 'PLAYER_IMPACT_DIFF', 
                           'PLAYER_IMPACT_HOME_MOMENTUM', 'PLAYER_IMPACT_AWAY_MOMENTUM', 'PLAYER_IMPACT_MOMENTUM_DIFF']:
                    if col not in self.stats_df.columns:
                        print(f"Adding essential column {col} with default values")
                        if 'DIFF' in col:
                            self.stats_df[col] = 0.0
                        else:
                            self.stats_df[col] = 1.0
        else:
            print("No player availability data found to merge")
            
        # Fill missing player impact with default values
        if 'PLAYER_IMPACT_HOME' not in self.stats_df.columns:
            print("Adding default values for player availability columns")
            self.stats_df['PLAYER_IMPACT_HOME'] = 1.0
        if 'PLAYER_IMPACT_AWAY' not in self.stats_df.columns:
            self.stats_df['PLAYER_IMPACT_AWAY'] = 1.0
        if 'PLAYER_IMPACT_DIFF' not in self.stats_df.columns:
            self.stats_df['PLAYER_IMPACT_DIFF'] = self.stats_df['PLAYER_IMPACT_HOME'] - self.stats_df['PLAYER_IMPACT_AWAY']
        
        # Prepare features
        print("Preparing features...")
        self.features, self.targets = self.feature_processor.prepare_features(self.stats_df)
        
        # Add game date and target together to avoid fragmentation
        additional_cols = {
            'GAME_DATE': self.stats_df['GAME_DATE'],
            'TARGET': self.targets
        }
        self.features = pd.concat([self.features, pd.DataFrame(additional_cols, index=self.features.index)], axis=1)
        
        # Cache the processed features if feature caching is enabled
        if self.use_cache and self.selective_cache.get('features', True):
            cache_params = {
                'config_hash': self.config_hash,
                'lookback_windows': sorted(self.lookback_windows),
                'seasons': sorted(self.seasons)
            }
            
            cached_data = (self.games, self.advanced_metrics, self.stats_df, self.features, self.targets)
            self.cache_manager.set_cache('features', cache_params, cached_data)
            print(f"Cached processed features: {len(self.features)} samples with {len(self.features.columns)} features")
        
    def train_models(self) -> None:
        """Train all prediction models with caching support."""
        if self.features is None:
            raise ValueError("Features not available. Call fetch_and_process_data first.")
            
        # Check if we have meaningful data for training
        if len(self.features) <= 1:  # 1 or fewer samples isn't enough for meaningful training
            print("Warning: Insufficient data for meaningful training (only {} samples found)".format(len(self.features)))
            print("Will use mock models for demonstration purposes only")
            # Create mock models that will just return default predictions
            if self.use_enhanced_models:
                from sklearn.dummy import DummyClassifier
                # Use 'most_frequent' strategy instead of 'constant' to avoid errors
                dummy = DummyClassifier(strategy='most_frequent')
                # Get a few numeric columns for dummy training
                numeric_cols = [col for col in self.features.columns 
                               if col not in ['GAME_DATE', 'TARGET'] and self.features[col].dtype in [np.float64, np.int64]]
                X = self.features[numeric_cols[:5] if len(numeric_cols) >= 5 else numeric_cols]
                
                # Create a dummy target
                y = np.zeros(len(X), dtype=int)  # Ensure integer type for classification
                dummy.fit(X, y)
                
                # Create simple prediction function that always returns 0.5
                def predict_proba(X):
                    return np.array([[0.5, 0.5]] * len(X))
                
                # Add the predict_proba method to our dummy
                dummy.predict_proba = predict_proba
                
                # Set dummy models for prediction
                self.ensemble_model.models = {'dummy': dummy}
                self.deep_model_trainer.model = dummy
                if self.hybrid_model:
                    self.hybrid_model.ensemble_model = self.ensemble_model
                    self.hybrid_model.deep_model = self.deep_model_trainer
                
                print("Created mock models due to insufficient training data")
                return
        
        # Check for cached trained models when model caching is enabled
        if self.use_cache and self.selective_cache.get('models', True):
            # Create cache params based on feature data and model config
            model_cache_params = {
                'config_hash': self.config_hash,
                'feature_hash': self._get_feature_hash(),
                'model_type': 'hybrid' if self.use_enhanced_models else 'standard',
                'quick_mode': self.quick_mode
            }
            
            cached_models = self.cache_manager.get_cache('models', model_cache_params)
            if cached_models is not None and not self.cache_manager.is_cache_stale('models', model_cache_params, self.cache_max_age_days):
                print("Using cached trained models...")
                if self.use_enhanced_models:
                    # For hybrid model approach
                    self.hybrid_model = cached_models.get('hybrid_model')
                    self.ensemble_model = cached_models.get('ensemble_model')
                    self.deep_model_trainer = cached_models.get('deep_model')
                else:
                    # For separate models approach
                    self.ensemble_model = cached_models.get('ensemble_model')
                    self.deep_model_trainer = cached_models.get('deep_model')
                
                print("Successfully loaded trained models from cache")
                return
            
            print("No valid model cache found, training new models...")
        
        # Train models
        if self.use_enhanced_models:
            # Train the hybrid model (which trains both ensemble and deep models)
            print("\nTraining advanced hybrid model...")
            self.hybrid_model.train(self.features)
        else:
            # Train models separately
            # Train ensemble model
            print("\nTraining ensemble model...")
            self.ensemble_model.train(self.features)
            
            # Train deep model
            print("\nTraining deep learning model...")
            self.deep_model_trainer.train_deep_model(self.features)
        
        # Cache the trained models when model caching is enabled
        if self.use_cache and self.selective_cache.get('models', True):
            model_cache_params = {
                'config_hash': self.config_hash,
                'feature_hash': self._get_feature_hash(),
                'model_type': 'hybrid' if self.use_enhanced_models else 'standard',
                'quick_mode': self.quick_mode
            }
            
            if self.use_enhanced_models:
                cached_models = {
                    'hybrid_model': self.hybrid_model,
                    'ensemble_model': self.ensemble_model,
                    'deep_model': self.deep_model_trainer
                }
            else:
                cached_models = {
                    'ensemble_model': self.ensemble_model,
                    'deep_model': self.deep_model_trainer
                }
                
            self.cache_manager.set_cache('models', model_cache_params, cached_models)
            print("Cached trained models for future use")
    
    def _get_feature_hash(self) -> str:
        """
        Generate a hash of the current feature data for cache verification.
        
        Returns:
            str: Hash representing the feature data
        """
        import hashlib
        
        # Use a sample of the features plus metadata to create a hash
        # This is more efficient than hashing the entire feature set
        # but still captures the essential characteristics
        
        if self.features is None or self.targets is None:
            return "no_features"
            
        # Get shape, column names, and a data sample
        feature_shape = self.features.shape
        feature_columns = sorted(self.features.columns.tolist())
        
        # Take a sample of the actual data values (first 100 rows, every 10th column)
        sample_cols = feature_columns[::10][:20]  # Limit to 20 columns max
        if sample_cols:
            data_sample = self.features[sample_cols].head(100).values.flatten()
            # Convert to strings and join
            data_str = ','.join([str(round(float(x), 4)) if pd.notnull(x) else 'nan' for x in data_sample])
        else:
            data_str = ""
        
        # Create a string to hash
        feature_str = f"{feature_shape}|{','.join(feature_columns)}|{data_str}"
        
        # Generate the hash
        hash_obj = hashlib.md5(feature_str.encode())
        return hash_obj.hexdigest()
        
    def predict(self, 
                features: pd.DataFrame,
                model_type: str = 'hybrid') -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with confidence scores.
        
        Args:
            features: DataFrame containing features
            model_type: Type of model to use ('ensemble', 'deep', or 'hybrid')
            
        Returns:
            tuple: (predictions, confidence_scores)
        """
        if model_type == 'ensemble':
            if self.use_enhanced_models:
                return self.ensemble_model.predict_with_confidence(features)
            else:
                return self.ensemble_model.predict_with_confidence(features)
        elif model_type == 'deep':
            if self.use_enhanced_models:
                preds, uncertainties = self.deep_model_trainer.predict_with_uncertainty(features)
                confidence_scores = self.deep_model_trainer.calculate_confidence_from_uncertainty(preds, uncertainties)
                return preds, confidence_scores
            else:
                predictions = self.deep_model_trainer.predict(features)
                confidence_scores = self.ensemble_model.calculate_confidence_score(predictions, features)
                return predictions, confidence_scores
        elif model_type == 'hybrid':
            if self.use_enhanced_models:
                return self.hybrid_model.predict_with_confidence(features)
            else:
                # Get predictions from both models
                ensemble_preds = self.ensemble_model.predict(features)
                deep_preds = self.deep_model_trainer.predict(features)
                
                # Average the predictions (weighted equally for now)
                hybrid_preds = (ensemble_preds + deep_preds) / 2
                
                # Use ensemble model for confidence scores
                confidence_scores = self.ensemble_model.calculate_confidence_score(hybrid_preds, features)
                return hybrid_preds, confidence_scores
        else:
            raise ValueError("Invalid model_type. Choose from 'ensemble', 'deep', or 'hybrid'.")
            
    def get_feature_importances(self, n: int = 20) -> Dict:
        """
        Get the top n most important features.
        
        Args:
            n: Number of top features to return
            
        Returns:
            dict: Feature names and importance scores
        """
        try:
            if self.use_enhanced_models and self.hybrid_model:
                return self.hybrid_model.get_feature_importances(n)
            else:
                return self.ensemble_model.get_top_features(n)
        except Exception as e:
            print(f"Could not get feature importances: {str(e)}")
            # Return mock feature importances when real models aren't available/trained
            mock_features = {}
            if self.features is not None and len(self.features.columns) > 0:
                # Get some actual feature names if available
                feature_cols = []
                for col in self.features.columns:
                    if col not in ['GAME_DATE', 'TARGET']:
                        try:
                            col_data = self.features[col]
                            if isinstance(col_data, pd.Series) and col_data.dtype in [np.float64, np.int64]:
                                feature_cols.append(col)
                        except Exception:
                            # Skip any problematic columns
                            pass
                
                # Generate mock importance scores for a subset of features
                sample_cols = feature_cols[:min(n, len(feature_cols))]
                if sample_cols:
                    # Generate mock scores that sum to 1.0
                    total = len(sample_cols)
                    for i, col in enumerate(reversed(sample_cols)):
                        # Higher index gets higher importance in this mock data
                        mock_features[col] = (i + 1) / (total * (total + 1) / 2)
                    return mock_features
            
            # If no features available, create generic ones
            for i in range(min(n, 10)):
                feature_name = f"WIN_PCT_DIFF_{(i+1)*7}D"
                mock_features[feature_name] = (10-i) / 55  # Scores sum to 1.0
            return mock_features
    
    def prepare_game_prediction(self, 
                              home_team_id: int, 
                              away_team_id: int,
                              game_date: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare features for a specific game prediction using the standardized feature pipeline.
        
        Args:
            home_team_id: NBA API team ID for home team
            away_team_id: NBA API team ID for away team
            game_date: Game date in format 'YYYY-MM-DD' (default: latest available date)
            
        Returns:
            pd.DataFrame: Features for the game
        """
        if self.stats_df is None:
            raise ValueError("Stats not available. Call fetch_and_process_data first.")
            
        # If no date provided, use the latest available
        if game_date is None:
            game_date = self.stats_df['GAME_DATE'].max()
        else:
            game_date = pd.to_datetime(game_date)
            
        # Check for cached prediction features
        if self.use_cache:
            # Format game date for cache key
            formatted_date = str(game_date) if game_date is not None else "latest"
            
            feature_cache_params = {
                'config_hash': self.config_hash,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'game_date': formatted_date,
                'feature_version': '1.0'  # For cache invalidation if feature logic changes
            }
            
            cached_features = self.cache_manager.get_cache('prediction_features', feature_cache_params)
            if cached_features is not None and not self.cache_manager.is_cache_stale('prediction_features', feature_cache_params, self.cache_max_age_days):
                print(f"Using cached prediction features for {home_team_id} vs {away_team_id}")
                return cached_features
                
        print(f"Preparing prediction for {home_team_id} (home) vs {away_team_id} (away) on {game_date}")
            
        # Find the most recent entry for both teams - critically ensure we use data strictly before game_date to prevent data leakage
        home_stats = self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['GAME_DATE'] < game_date)  # Changed <= to < to prevent data leakage
        ].sort_values('GAME_DATE', ascending=False).iloc[0].copy() if not self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['GAME_DATE'] < game_date)  # Changed <= to < to prevent data leakage
        ].empty else pd.Series()
        
        away_stats = self.stats_df[
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) & 
            (self.stats_df['GAME_DATE'] < game_date)  # Changed <= to < to prevent data leakage
        ].sort_values('GAME_DATE', ascending=False).iloc[0].copy() if not self.stats_df[
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) & 
            (self.stats_df['GAME_DATE'] < game_date)  # Changed <= to < to prevent data leakage
        ].empty else pd.Series()
        
        if home_stats.empty or away_stats.empty:
            # Try to find stats where these teams played in other positions
            alt_home_stats = self.stats_df[
                (self.stats_df['TEAM_ID_AWAY'] == home_team_id) & 
                (self.stats_df['GAME_DATE'] < game_date)
            ].sort_values('GAME_DATE', ascending=False)
            
            alt_away_stats = self.stats_df[
                (self.stats_df['TEAM_ID_HOME'] == away_team_id) & 
                (self.stats_df['GAME_DATE'] < game_date)
            ].sort_values('GAME_DATE', ascending=False)
            
            # Use alternate data if available
            if home_stats.empty and not alt_home_stats.empty:
                print(f"Using away stats for home team {home_team_id}")
                home_stats = alt_home_stats.iloc[0].rename(
                    lambda x: x.replace('_AWAY', '_HOME') if '_AWAY' in x else x
                ).copy()
                
            if away_stats.empty and not alt_away_stats.empty:
                print(f"Using home stats for away team {away_team_id}")
                away_stats = alt_away_stats.iloc[0].rename(
                    lambda x: x.replace('_HOME', '_AWAY') if '_HOME' in x else x
                ).copy()
            
            # If still no data, raise error
            if home_stats.empty or away_stats.empty:
                raise ValueError("Not enough data available for one or both teams.")
            
        # Create a new game entry with these teams
        new_game = pd.Series({
            'GAME_DATE': game_date,
            'GAME_ID_HOME': f'PRED_{home_team_id}_{away_team_id}',  # Add a predictive GAME_ID for compatibility
            'TEAM_ID_HOME': home_team_id,
            'TEAM_ID_AWAY': away_team_id
        })
        
        # Copy features from the most recent games, using only historical data
        for col in home_stats.index:
            if col not in new_game and '_HOME' in col:
                new_game[col] = home_stats[col]
                
        for col in away_stats.index:
            if col not in new_game and '_AWAY' in col:
                new_game[col] = away_stats[col]
                
        # Add head-to-head features - ensuring we only use data strictly before game_date
        h2h_stats = self.stats_df[
            (self.stats_df['TEAM_ID_HOME'] == home_team_id) & 
            (self.stats_df['TEAM_ID_AWAY'] == away_team_id) &
            (self.stats_df['GAME_DATE'] < game_date)  # Changed <= to < to prevent data leakage
        ].sort_values('GAME_DATE', ascending=False)
        
        if not h2h_stats.empty:
            h2h_recent = h2h_stats.iloc[0]
            for col in ['H2H_GAMES', 'H2H_WIN_PCT', 'DAYS_SINCE_H2H', 'LAST_GAME_HOME', 
                       'H2H_AVG_MARGIN', 'H2H_STREAK', 'H2H_HOME_ADVANTAGE', 'H2H_MOMENTUM']:
                if col in h2h_recent:
                    new_game[col] = h2h_recent[col]
        else:
            # Default values if no head-to-head history
            h2h_defaults = {
                'H2H_GAMES': 0,
                'H2H_WIN_PCT': 0.5,
                'DAYS_SINCE_H2H': 365,
                'LAST_GAME_HOME': 0,
                'H2H_AVG_MARGIN': 0,
                'H2H_STREAK': 0,
                'H2H_HOME_ADVANTAGE': 0.5,
                'H2H_MOMENTUM': 0.5
            }
            for col, val in h2h_defaults.items():
                new_game[col] = val
                
        # Add player impact features (default values)
        if 'PLAYER_IMPACT_HOME' not in new_game:
            new_game['PLAYER_IMPACT_HOME'] = 1.0
        if 'PLAYER_IMPACT_AWAY' not in new_game:
            new_game['PLAYER_IMPACT_AWAY'] = 1.0
        if 'PLAYER_IMPACT_DIFF' not in new_game:
            new_game['PLAYER_IMPACT_DIFF'] = new_game['PLAYER_IMPACT_HOME'] - new_game['PLAYER_IMPACT_AWAY']
        if 'PLAYER_IMPACT_HOME_MOMENTUM' not in new_game:
            new_game['PLAYER_IMPACT_HOME_MOMENTUM'] = 1.0
        if 'PLAYER_IMPACT_AWAY_MOMENTUM' not in new_game:
            new_game['PLAYER_IMPACT_AWAY_MOMENTUM'] = 1.0
        if 'PLAYER_IMPACT_MOMENTUM_DIFF' not in new_game:
            new_game['PLAYER_IMPACT_MOMENTUM_DIFF'] = 0.0
            
        # Process features for this game
        game_df = pd.DataFrame([new_game])
        
        # Collect required features from all sources
        required_features = self._get_required_features()
        
        print(f"Identified {len(required_features)} required features for prediction")
        
        # Initialize feature transformer with required features
        self.feature_processor.feature_transformer.register_features(list(required_features))
        
        # Generate prediction features using the standardized pipeline
        print("Generating prediction features using standardized pipeline...")
        prediction_features = self.feature_processor.prepare_enhanced_features(game_df)
        
        # Final validation to ensure all required features are present
        missing_features = []
        missing_feature_values = {}
        
        for feature in required_features:
            if feature not in prediction_features.columns and feature not in ['GAME_DATE', 'TARGET']:
                missing_features.append(feature)
                # Add with default value (0 or 0.5 for probabilities)
                if any(prob_term in feature for prob_term in ['WIN_PCT', 'PROBABILITY', 'H2H_']):
                    missing_feature_values[feature] = 0.5
                else:
                    missing_feature_values[feature] = 0
        
        # Add all missing features at once to avoid fragmentation
        # Prepare data for all features at once to avoid fragmentation
        all_features = {}
        
        # Add missing features to the all_features dictionary
        if missing_feature_values:
            all_features.update(missing_feature_values)
        
        # Add game date if needed
        if 'GAME_DATE' not in prediction_features.columns:
            all_features['GAME_DATE'] = game_date
            
        # Merge all new features at once to avoid fragmentation
        if all_features:
            # Create a DataFrame with all new features and concatenate once
            new_df = pd.DataFrame(all_features, index=prediction_features.index)
            prediction_features = pd.concat([prediction_features, new_df], axis=1)
            
        if missing_features:
            print(f"Added {len(missing_features)} missing features with default values: {missing_features[:5]}...")
        
        # Cache the prediction features
        if self.use_cache:
            # Format game date for cache key
            formatted_date = str(game_date) if game_date is not None else "latest"
            
            feature_cache_params = {
                'config_hash': self.config_hash,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'game_date': formatted_date,
                'feature_version': '1.0'  # For cache invalidation if feature logic changes
            }
            
            self.cache_manager.set_cache('prediction_features', feature_cache_params, prediction_features)
            print(f"Cached prediction features for {home_team_id} vs {away_team_id}")
            
        return prediction_features
        
    def _get_required_features(self) -> Set[str]:
        """
        Get a comprehensive set of all required features from all components.
        
        Returns:
            Set of required feature names
        """
        required_features = set()
        
        # From ensemble model
        if hasattr(self.ensemble_model, 'training_features'):
            required_features.update(self.ensemble_model.training_features)
        
        # From deep model
        if hasattr(self.deep_model_trainer, 'training_features'):
            required_features.update(self.deep_model_trainer.training_features)
            
        # From feature registry (important derived features)
        for base_feature, info in FEATURE_REGISTRY.items():
            if info['type'] in ['derived', 'interaction']:
                # Add all window variants if applicable
                if info['windows']:
                    for window in info['windows']:
                        required_features.add(f"{base_feature}_{window}D")
                else:
                    required_features.add(base_feature)
                    
                # Also add dependencies
                if 'dependencies' in info:
                    for dep in info['dependencies']:
                        if info['windows']:
                            for window in info['windows']:
                                required_features.add(f"{dep}_{window}D")
                        else:
                            required_features.add(dep)
            
        # If we have a record of the training features
        if self.features is not None:
            feature_cols = []
            for col in self.features.columns:
                if col not in ['GAME_DATE', 'TARGET']:
                    try:
                        col_data = self.features[col]
                        if isinstance(col_data, pd.Series):
                            feature_cols.append(col)
                        elif isinstance(col_data, pd.DataFrame) and len(col_data.columns) > 0:
                            # For DataFrame columns, add the first column name with a suffix
                            feature_cols.append(f"{col}_0")
                    except Exception:
                        # Skip problematic columns
                        pass
            required_features.update(feature_cols)
            
        # Remove non-feature columns
        for col in ['GAME_DATE', 'TARGET', 'TEAM_ID_HOME', 'TEAM_ID_AWAY']:
            if col in required_features:
                required_features.remove(col)
                
        return required_features
    
    def predict_game(self, 
                   home_team_id: int, 
                   away_team_id: int,
                   game_date: Optional[str] = None,
                   model_type: str = 'hybrid',
                   use_cached_prediction: bool = True) -> Dict:
        """
        Predict the outcome of a specific game with improved feature compatibility.
        
        Args:
            home_team_id: NBA API team ID for home team
            away_team_id: NBA API team ID for away team
            game_date: Game date in format 'YYYY-MM-DD' (default: latest available date)
            model_type: Type of model to use ('ensemble', 'deep', or 'hybrid')
            use_cached_prediction: Whether to use cached prediction results
            
        Returns:
            dict: Prediction details
        """
        # Check for cached prediction if requested
        if self.use_cache and use_cached_prediction:
            # Format game date for cache key
            formatted_date = str(game_date) if game_date is not None else "latest"
            
            pred_cache_params = {
                'config_hash': self.config_hash,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'game_date': formatted_date,
                'model_type': model_type
            }
            
            cached_prediction = self.cache_manager.get_cache('predictions', pred_cache_params)
            if cached_prediction is not None and not self.cache_manager.is_cache_stale('predictions', pred_cache_params, 1):  # 1 day max age for predictions
                print(f"Using cached prediction for {home_team_id} vs {away_team_id}")
                return cached_prediction
                
        # Check if model is trained
        if self.features is None or len(self.features) <= 1:
            print("Warning: Features not available or insufficient. Using mock prediction.")
            # Get team abbreviations for better output
            from src.utils.constants import TEAM_ID_TO_ABBREV
            home_team_abbrev = TEAM_ID_TO_ABBREV.get(home_team_id, str(home_team_id))
            away_team_abbrev = TEAM_ID_TO_ABBREV.get(away_team_id, str(away_team_id))
            
            # Create a mock prediction with 50% probability
            from datetime import datetime as dt, date
            mock_result = {
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'home_team': home_team_abbrev,
                'away_team': away_team_abbrev,
                'game_date': game_date or date.today().isoformat(),
                'home_win_probability': 0.5,  # Default 50% for mock
                'confidence': 0.4,  # Lower confidence for mock prediction
                'model_type': model_type + " (mock)",
                'prediction_time': dt.now().isoformat(),
                'is_mock': True
            }
            
            # Cache the mock prediction if caching is enabled
            if self.use_cache:
                self.cache_manager.set_cache('predictions', pred_cache_params, mock_result)
                
            return mock_result
        
        # Prepare features for the game - already ensures required features are present
        game_features = self.prepare_game_prediction(home_team_id, away_team_id, game_date)
        
        # Make prediction with the prepared feature set
        # The models' predict methods handle feature alignment internally now
        probs, confidence = self.predict(game_features, model_type)
        
        # Get team abbreviations for better output
        from src.utils.constants import TEAM_ID_TO_ABBREV
        home_team_abbrev = TEAM_ID_TO_ABBREV.get(home_team_id, str(home_team_id))
        away_team_abbrev = TEAM_ID_TO_ABBREV.get(away_team_id, str(away_team_id))
        
        # Format output
        # Use iloc[0] to get the first row value from the DataFrame
        # Handle 'GAME_DATE' column error by first checking if it exists
        game_date_value = game_features['GAME_DATE'].iloc[0] if 'GAME_DATE' in game_features.columns else None
        
        from datetime import datetime as dt
        
        result = {
            'home_team_id': home_team_id,
            'away_team_id': away_team_id,
            'home_team': home_team_abbrev,
            'away_team': away_team_abbrev,
            'game_date': game_date_value,
            'home_win_probability': float(probs[0]),
            'confidence': float(confidence[0]),
            'model_type': model_type,
            'prediction_time': dt.now().isoformat()
        }
        
        # Cache the prediction
        if self.use_cache:
            # Format game date for cache key
            formatted_date = str(game_date) if game_date is not None else "latest"
            
            pred_cache_params = {
                'config_hash': self.config_hash,
                'home_team_id': home_team_id,
                'away_team_id': away_team_id,
                'game_date': formatted_date,
                'model_type': model_type
            }
            
            self.cache_manager.set_cache('predictions', pred_cache_params, result)
        
        return result
        
    def save_models(self, directory: str = 'saved_models') -> str:
        """
        Save all trained models to disk.
        
        Args:
            directory: Directory to save models in
            
        Returns:
            str: Path to the saved model directory
        """
        # Create save directory if it doesn't exist
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = os.path.join(directory, f"nba_model_{timestamp}")
        os.makedirs(save_dir, exist_ok=True)
        
        print(f"Saving models to {save_dir}...")
        
        # Save model configuration
        config = {
            'seasons': self.seasons,
            'lookback_windows': self.lookback_windows,
            'use_enhanced_models': self.use_enhanced_models,
            'quick_mode': self.quick_mode,
            'timestamp': timestamp
        }
        
        with open(os.path.join(save_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(config, f)
        
        # Save the feature processor state
        with open(os.path.join(save_dir, 'feature_processor.pkl'), 'wb') as f:
            pickle.dump(self.feature_processor, f)
        
        # Save the ensemble model
        if self.ensemble_model:
            self.ensemble_model.save_model(os.path.join(save_dir, 'ensemble_model'))
        
        # Save the deep model
        if self.deep_model_trainer:
            self.deep_model_trainer.save_model(os.path.join(save_dir, 'deep_model'))
        
        # Save the hybrid model if available
        if self.hybrid_model:
            self.hybrid_model.save_model(os.path.join(save_dir, 'hybrid_model'))
        
        # Save feature statistics
        if self.features is not None:
            with open(os.path.join(save_dir, 'feature_stats.pkl'), 'wb') as f:
                # Save only the feature names and statistics, not the full data
                feature_stats = {
                    'feature_names': list(self.features.columns),
                    'feature_means': self.features.mean().to_dict(),
                    'feature_stds': self.features.std().to_dict(),
                }
                pickle.dump(feature_stats, f)
        
        print(f"Models successfully saved to {save_dir}")
        
        return save_dir
    
    def manage_cache(self, action: str = 'status', cache_type: str = None) -> Dict:
        """
        Manage the cache system with various operations.
        
        Args:
            action: The action to perform ('status', 'clear', 'clear_type', 'clear_all')
            cache_type: Specific cache type to clear (e.g., 'games', 'features', 'models')
            
        Returns:
            Dict with operation result
        """
        if not self.use_cache:
            return {"status": "Cache is disabled"}
            
        if action == 'status':
            # Get cache statistics
            stats = self.cache_manager.get_cache_stats()
            return {
                "status": "Cache statistics retrieved",
                "statistics": stats
            }
            
        elif action == 'clear_type' and cache_type:
            # Clear specific cache type
            count = self.cache_manager.clear_cache_type(cache_type)
            return {
                "status": f"Cleared {count} entries from {cache_type} cache"
            }
            
        elif action == 'clear_all':
            # Clear all caches
            count = self.cache_manager.clear_all_cache()
            return {
                "status": f"Cleared all cache entries ({count} total)"
            }
            
        return {"status": "Invalid action or missing parameters"}
    
    @classmethod
    def load_models(cls, model_dir: str, use_cache: bool = True, 
                  cache_dir: str = None, hardware_optimization: bool = True) -> 'EnhancedNBAPredictor':
        """
        Load a saved model from disk.
        
        Args:
            model_dir: Directory containing saved model files
            use_cache: Whether to use cache for subsequent operations
            cache_dir: Custom cache directory path
            hardware_optimization: Whether to apply hardware-specific optimizations
            
        Returns:
            EnhancedNBAPredictor: Loaded predictor instance
        """
        # Check if running in Colab and model_dir is a Google Drive path
        try:
            import google.colab
            is_colab = True
            # If model_dir is a Google Drive path but Drive isn't mounted, mount it
            if model_dir.startswith('/content/drive') and not os.path.exists('/content/drive'):
                print("Google Drive not mounted. Mounting now...")
                from google.colab import drive
                drive.mount('/content/drive')
        except ImportError:
            is_colab = False
            
        print(f"Loading models from {model_dir}...")
        
        # Load configuration
        with open(os.path.join(model_dir, 'config.pkl'), 'rb') as f:
            config = pickle.load(f)
        
        # Create a new instance with the same configuration
        predictor = cls(
            seasons=config['seasons'],
            lookback_windows=config['lookback_windows'],
            use_enhanced_models=config['use_enhanced_models'],
            quick_mode=config['quick_mode'],
            use_cache=use_cache,
            cache_dir=cache_dir,
            hardware_optimization=hardware_optimization
        )
        
        # Load feature processor
        with open(os.path.join(model_dir, 'feature_processor.pkl'), 'rb') as f:
            predictor.feature_processor = pickle.load(f)
        
        # Load models
        # Ensemble model
        ensemble_dir = os.path.join(model_dir, 'ensemble_model')
        if os.path.exists(ensemble_dir):
            if config['use_enhanced_models']:
                predictor.ensemble_model = NBAEnhancedEnsembleModel.load_model(ensemble_dir)
            else:
                predictor.ensemble_model = NBAEnsembleModel.load_model(ensemble_dir)
        
        # Deep model
        deep_dir = os.path.join(model_dir, 'deep_model')
        if os.path.exists(deep_dir):
            if config['use_enhanced_models']:
                predictor.deep_model_trainer = EnhancedDeepModelTrainer.load_model(deep_dir)
            else:
                predictor.deep_model_trainer = DeepModelTrainer.load_model(deep_dir)
        
        # Hybrid model
        hybrid_dir = os.path.join(model_dir, 'hybrid_model')
        if os.path.exists(hybrid_dir):
            predictor.hybrid_model = HybridModel.load_model(hybrid_dir)
        
        # Load feature statistics if available
        stats_path = os.path.join(model_dir, 'feature_stats.pkl')
        if os.path.exists(stats_path):
            with open(stats_path, 'rb') as f:
                feature_stats = pickle.load(f)
                # Set as placeholder to enable prediction
                predictor.features = pd.DataFrame(columns=feature_stats['feature_names'])
        
        # Store loaded models in cache if caching is enabled
        if use_cache:
            # Create a cache entry for the loaded models
            model_cache_params = {
                'config_hash': predictor.config_hash,
                'model_type': 'hybrid' if config['use_enhanced_models'] else 'standard',
                'quick_mode': config['quick_mode'],
                'source': 'loaded_from_disk',
                'model_dir': os.path.basename(model_dir)
            }
            
            if config['use_enhanced_models']:
                cached_models = {
                    'hybrid_model': predictor.hybrid_model,
                    'ensemble_model': predictor.ensemble_model,
                    'deep_model': predictor.deep_model_trainer
                }
            else:
                cached_models = {
                    'ensemble_model': predictor.ensemble_model,
                    'deep_model': predictor.deep_model_trainer
                }
                
            predictor.cache_manager.set_cache('models', model_cache_params, cached_models)
            print("Cached loaded models for future use")
        
        print(f"Models successfully loaded from {model_dir}")
        return predictor