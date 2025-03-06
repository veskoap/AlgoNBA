"""
Deep learning models for NBA prediction.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from typing import List, Tuple


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

            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

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
        Make predictions using the trained ensemble of deep models.
        
        Args:
            X: DataFrame containing features
            
        Returns:
            np.ndarray: Prediction probabilities
        """
        if not self.models or not self.scalers:
            raise ValueError("Models not trained yet. Call train_deep_model first.")
            
        # Drop non-feature columns
        X = X.drop(['TARGET', 'GAME_DATE'], axis=1, errors='ignore')
        
        # Get predictions from each model in the ensemble
        all_preds = []
        
        for model, scaler in zip(self.models, self.scalers):
            # Scale features
            X_scaled = scaler.transform(X)
            X_tensor = torch.FloatTensor(X_scaled).to(self.device)
            
            # Make predictions
            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)
                probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.append(probs)
                
        # Average predictions from all models
        ensemble_preds = np.mean(all_preds, axis=0)
        
        return ensemble_preds