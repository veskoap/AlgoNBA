import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple, DefaultDict
from collections import defaultdict
from model_definition import DeepNBAPredictor # Import DeepNBAPredictor

def train_deep_model(X: pd.DataFrame) -> Tuple[List, List]:
    """Train deep neural network model with enhanced architecture."""
    print("\nTraining deep neural network model...")

    # Extract target variable
    y = X['TARGET']
    X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')

    print(f"Training deep model with {len(X)} samples and {len(X.columns)} features")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    models = []
    scalers = []
    fold_metrics = []
    tscv = TimeSeriesSplit(n_splits=5)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining deep model fold {fold}...")

        # Prepare data
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train_scaled).to(device)
        y_train_tensor = torch.LongTensor(y_train.values).to(device)
        X_val_tensor = torch.FloatTensor(X_val_scaled).to(device)
        y_val_tensor = torch.LongTensor(y_val.values).to(device)

        # Initialize model
        model = DeepNBAPredictor(X_train.shape[1]).to(device)
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

    return models, scalers


def train_traditional_model(X: pd.DataFrame, lookback_windows: List[int]) -> Tuple[List, List, List, DefaultDict[str, List]]:
    """Train enhanced ensemble of models with improved stability."""
    print("Training model ensemble...")

    # Extract target variable
    y = X['TARGET']
    X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')

    print(f"Training with {len(X)} samples and {len(X.columns)} features")

    tscv = TimeSeriesSplit(n_splits=5)

    # Initialize tracking
    models = []
    scalers = []
    feature_selectors = []
    fold_metrics = []
    feature_importance_dict = defaultdict(list)

    # First pass: identify consistently important features
    print("Performing initial feature stability analysis...")
    feature_stability = defaultdict(int)
    feature_selector_list = []

    for fold, (train_idx, _) in enumerate(tscv.split(X), 1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]

        # Initial feature selection
        selector = SelectFromModel(
            xgb.XGBClassifier(
                n_estimators=300,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42 + fold
            ),
            threshold='mean'
        )

        selector.fit(X_train, y_train)
        selected_features = X.columns[selector.get_support()].tolist()
        feature_selector_list.append(selected_features)

        for feat in selected_features:
            feature_stability[feat] += 1

    # Identify stable features (selected in majority of folds)
    stable_features = [feat for feat, count in feature_stability.items() if count >= 3]
    print(f"\nIdentified {len(stable_features)} stable features")

    # Main training loop with enhanced monitoring
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nTraining fold {fold}...")

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Use stable features
        feature_mask = X.columns.isin(stable_features)
        X_train_selected = X_train_scaled[:, feature_mask]
        X_val_selected = X_val_scaled[:, feature_mask]

        # Train window-specific models
        window_models = []

        for window in lookback_windows:
            # Get window-specific features
            window_features = [feat for feat in stable_features if f'_{window}D' in feat]
            base_features = [feat for feat in stable_features if '_D' not in feat]
            combined_features = window_features + base_features

            if not combined_features:
                continue

            feature_indices = [stable_features.index(feat) for feat in combined_features]

            # Window-specific model with early stopping
            model = xgb.XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.005,
                subsample=0.85,
                colsample_bytree=0.85,
                min_child_weight=4,
                gamma=0.5,
                reg_alpha=0.3,
                reg_lambda=1.5,
                scale_pos_weight=1,
                random_state=42 + window,
                eval_metric=['logloss', 'auc']
            )

            X_train_window = X_train_selected[:, feature_indices]
            X_val_window = X_val_selected[:, feature_indices]

            # Train with early stopping
            model.fit(
                X_train_window, y_train,
                eval_set=[(X_val_window, y_val)],
                verbose=0
            )

            window_models.append((f'{window}d', model, combined_features))

            # Store feature importance
            importances = model.feature_importances_
            for feat, imp in zip(combined_features, importances):
                feature_importance_dict[feat].append(imp)

        # Store models and scalers
        models.append((window_models, scaler))

        # Evaluate performance
        y_preds = []
        for _, model, feats in window_models:
            feature_indices = [stable_features.index(f) for f in feats]
            y_pred = model.predict_proba(X_val_selected[:, feature_indices])[:, 1]
            y_preds.append(y_pred)

        # Average predictions from all window models
        y_pred_avg = np.mean(y_preds, axis=0)
        y_pred_binary = (y_pred_avg > 0.5).astype(int)

        # Calculate metrics
        acc = accuracy_score(y_val, y_pred_binary)
        brier = brier_score_loss(y_val, y_pred_avg)
        auc = roc_auc_score(y_val, y_pred_avg)

        fold_metrics.append({
            'accuracy': acc,
            'brier_score': brier,
            'auc': auc
        })

        print(f"Fold {fold} Metrics:")
        print(f"Accuracy: {acc:.3f}")
        print(f"Brier Score: {brier:.3f}")
        print(f"AUC-ROC: {auc:.3f}")

    # Print overall performance
    print("\nOverall Model Performance:")
    metrics_df = pd.DataFrame(fold_metrics)
    for metric in metrics_df.columns:
        mean_val = metrics_df[metric].mean()
        std_val = metrics_df[metric].std()
        print(f"{metric}: {mean_val:.3f} ± {std_val:.3f}")

    return models, scalers, feature_selectors, feature_importance_dict

def calculate_confidence_score(predictions: np.ndarray, features: pd.DataFrame) -> np.ndarray:
    """Calculate confidence scores for predictions."""
    confidence_scores = np.zeros(len(predictions))

    try:
        # Factors affecting confidence
        factors = {
            'prediction_margin': 0.3,  # Weight for prediction probability margin
            'sample_size': 0.2,        # Weight for number of previous matches
            'recent_consistency': 0.2,  # Weight for consistency in recent games
            'h2h_history': 0.15,       # Weight for head-to-head history
            'rest_advantage': 0.15     # Weight for rest day advantage
        }

        for i, pred in enumerate(predictions):
            score = 0

            # Prediction margin confidence
            prob_margin = abs(pred - 0.5) * 2  # Scale to [0, 1]
            score += prob_margin * factors['prediction_margin']

            # Sample size confidence
            games_played = features.iloc[i]['WIN_count_HOME_60D']
            sample_size_conf = min(games_played / 20, 1)  # Scale to [0, 1]
            score += sample_size_conf * factors['sample_size']

            # Recent consistency confidence
            consistency = 1 - features.iloc[i]['HOME_CONSISTENCY_30D']  # Lower variance is better
            score += consistency * factors['recent_consistency']

            # Head-to-head confidence
            h2h_games = features.iloc[i]['H2H_GAMES']
            h2h_conf = min(h2h_games / 10, 1)  # Scale to [0, 1]
            score += h2h_conf * factors['h2h_history']

            # Rest advantage confidence
            rest_diff = abs(features.iloc[i]['REST_DIFF'])
            rest_conf = min(rest_diff / 3, 1)  # Scale to [0, 1]
            score += rest_conf * factors['rest_advantage']

            confidence_scores[i] = score

        # Normalize to [0, 1]
        confidence_scores = (confidence_scores - confidence_scores.min()) / (confidence_scores.max() - confidence_scores.min())

    except Exception as e:
        print(f"Error calculating confidence scores: {e}")
        confidence_scores = np.full(len(predictions), 0.5)

    return confidence_scores