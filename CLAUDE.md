# AlgoNBA Development Notes

## Project Structure

AlgoNBA is a sophisticated machine learning system for NBA game predictions with:
- Enhanced prediction models targeting 70%+ accuracy
- Player availability analysis
- Advanced ensemble models (XGBoost + LightGBM)
- Deep learning with residual networks and attention
- Monte Carlo dropout for uncertainty estimation

## Commands

### Running the Application

```bash
# Run with enhanced models (default)
python main.py

# Use standard models instead
python main.py --standard

# Specify seasons for training
python main.py --seasons 2022-23 2023-24

# Run in quick mode for faster testing (development only)
python main.py --quick

# Save trained models to disk
python main.py --save-models

# Load previously saved models
python main.py --load-models saved_models/nba_model_20230401_120000

# Combine options
python main.py --quick --standard --seasons 2022-23 --save-models
```

### Quick Mode Features

The `--quick` flag enables faster development testing by:
- Using fewer cross-validation folds (2 instead of 5)
- Using simplified model architectures
- Running fewer training epochs
- Performing less hyperparameter optimization

## Key Classes

- **EnhancedNBAPredictor**: Main orchestration class
- **NBAEnhancedEnsembleModel**: Advanced ensemble model with XGBoost and LightGBM
- **EnhancedDeepModelTrainer**: Deep learning implementation with residual networks and attention
- **HybridModel**: Meta-model that optimally combines ensemble and deep learning approaches
- **PlayerAvailabilityProcessor**: Handles player impact and availability features
- **EnhancedScaler**: Robust feature scaling with outlier handling

## Recent Improvements

1. **Performance Optimizations**:
   - Added quick mode for faster development iterations
   - Improved memory efficiency in the feature scaling process
   - Added robust handling of DataFrame fragmentation

2. **Error Handling**:
   - Fixed type conversion issues in injury recovery calculations
   - Updated API compatibility for XGBoost and LightGBM
   - Added fallback mechanisms for model training failures

3. **Documentation**:
   - Improved README with up-to-date usage examples
   - Added comprehensive docstrings across all modules
   - Created this CLAUDE.md file for development reference

## Prediction Workflow

1. Fetch historical NBA data
2. Process team statistics and player availability
3. Calculate advanced features
4. Train ensemble and deep learning models
5. Optimize hybrid weighting
6. Make predictions with confidence scores

## License

MIT License - see LICENSE file for details.