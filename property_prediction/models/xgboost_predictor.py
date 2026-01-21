"""
XGBoost Predictor: Property prediction using XGBoost with cross-validation.

Supports:
- Regression (Tg, MAC, Lipophilicity)
- Classification (BBBP)
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Literal
from pathlib import Path
import json

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score


TaskType = Literal['regression', 'classification']


class XGBoostPredictor:
    """
    XGBoost predictor with cross-validation.
    
    Handles both regression and classification tasks.
    """
    
    def __init__(
        self,
        task_type: TaskType = 'regression',
        n_estimators: int = 500,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        early_stopping_rounds: int = 50,
        n_folds: int = 5,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost predictor.
        
        Args:
            task_type: 'regression' or 'classification'
            n_estimators: Maximum number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            early_stopping_rounds: Stop if no improvement in N rounds
            n_folds: Number of cross-validation folds
            random_state: Random seed for reproducibility
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.task_type = task_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.early_stopping_rounds = early_stopping_rounds
        self.n_folds = n_folds
        self.random_state = random_state
        
        # Model storage
        self.models: List[xgb.XGBModel] = []
        self.cv_scores: Dict[str, List[float]] = {}
        self.feature_importance: Optional[np.ndarray] = None
    
    def _create_model(self) -> xgb.XGBModel:
        """Create a new XGBoost model."""
        common_params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'random_state': self.random_state,
            'n_jobs': -1,
            'verbosity': 0,
        }
        
        if self.task_type == 'regression':
            return xgb.XGBRegressor(
                **common_params,
                objective='reg:squarederror',
                eval_metric='mae',
            )
        else:
            return xgb.XGBClassifier(
                **common_params,
                objective='binary:logistic',
                eval_metric='auc',
            )
    
    def train_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Train with cross-validation.
        
        Args:
            X: Feature matrix of shape (N, D)
            y: Target values of shape (N,)
            feature_names: Optional feature names for importance
            verbose: Print progress
            
        Returns:
            Dictionary of mean CV scores
        """
        # Choose CV strategy
        if self.task_type == 'classification':
            kfold = StratifiedKFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        else:
            kfold = KFold(
                n_splits=self.n_folds,
                shuffle=True,
                random_state=self.random_state,
            )
        
        # Initialize score storage
        if self.task_type == 'regression':
            self.cv_scores = {'mae': [], 'rmse': [], 'r2': []}
        else:
            self.cv_scores = {'accuracy': [], 'auc': [], 'f1': []}
        
        self.models = []
        all_importances = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            if verbose:
                print(f"  Fold {fold + 1}/{self.n_folds}...", end=' ')
            
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Create and train model
            model = self._create_model()
            
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            
            # Predict on validation set
            if self.task_type == 'regression':
                y_pred = model.predict(X_val)
                
                mae = mean_absolute_error(y_val, y_pred)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred))
                r2 = r2_score(y_val, y_pred)
                
                self.cv_scores['mae'].append(mae)
                self.cv_scores['rmse'].append(rmse)
                self.cv_scores['r2'].append(r2)
                
                if verbose:
                    print(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}")
            else:
                y_pred = model.predict(X_val)
                y_prob = model.predict_proba(X_val)[:, 1]
                
                acc = accuracy_score(y_val, y_pred)
                auc = roc_auc_score(y_val, y_prob)
                f1 = f1_score(y_val, y_pred)
                
                self.cv_scores['accuracy'].append(acc)
                self.cv_scores['auc'].append(auc)
                self.cv_scores['f1'].append(f1)
                
                if verbose:
                    print(f"Acc: {acc:.4f}, AUC: {auc:.4f}, F1: {f1:.4f}")
            
            self.models.append(model)
            all_importances.append(model.feature_importances_)
        
        # Average feature importance
        self.feature_importance = np.mean(all_importances, axis=0)
        
        # Compute mean scores
        mean_scores = {k: np.mean(v) for k, v in self.cv_scores.items()}
        std_scores = {f'{k}_std': np.std(v) for k, v in self.cv_scores.items()}
        
        if verbose:
            print(f"  Mean CV scores: {mean_scores}")
        
        return {**mean_scores, **std_scores}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using ensemble of CV models.
        
        Args:
            X: Feature matrix of shape (N, D)
            
        Returns:
            Predictions of shape (N,)
        """
        if not self.models:
            raise ValueError("Model not trained. Call train_cv first.")
        
        # Ensemble predictions
        predictions = []
        for model in self.models:
            if self.task_type == 'regression':
                pred = model.predict(X)
            else:
                pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)
        
        # Average predictions
        avg_pred = np.mean(predictions, axis=0)
        
        if self.task_type == 'classification':
            # Convert to class labels
            return (avg_pred >= 0.5).astype(int)
        
        return avg_pred
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probabilities (for classification).
        
        Args:
            X: Feature matrix of shape (N, D)
            
        Returns:
            Probability predictions of shape (N,)
        """
        if self.task_type != 'classification':
            raise ValueError("predict_proba only available for classification")
        
        if not self.models:
            raise ValueError("Model not trained. Call train_cv first.")
        
        # Ensemble probabilities
        probas = []
        for model in self.models:
            prob = model.predict_proba(X)[:, 1]
            probas.append(prob)
        
        return np.mean(probas, axis=0)
    
    def get_feature_importance(
        self,
        feature_names: Optional[List[str]] = None,
        top_k: int = 20,
    ) -> List[Tuple[str, float]]:
        """
        Get top-k most important features.
        
        Args:
            feature_names: Feature names (optional)
            top_k: Number of top features to return
            
        Returns:
            List of (feature_name, importance) tuples
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained. Call train_cv first.")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        # Sort by importance
        indices = np.argsort(self.feature_importance)[::-1][:top_k]
        
        return [(feature_names[i], self.feature_importance[i]) for i in indices]
    
    def save(self, path: str):
        """Save trained models to directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for i, model in enumerate(self.models):
            model.save_model(path / f'model_fold_{i}.json')
        
        # Save metadata
        metadata = {
            'task_type': self.task_type,
            'n_folds': self.n_folds,
            'cv_scores': self.cv_scores,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
            },
        }
        
        with open(path / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance
        if self.feature_importance is not None:
            np.save(path / 'feature_importance.npy', self.feature_importance)
    
    def load(self, path: str):
        """Load trained models from directory."""
        path = Path(path)
        
        # Load metadata
        with open(path / 'metadata.json', 'r') as f:
            metadata = json.load(f)
        
        self.task_type = metadata['task_type']
        self.n_folds = metadata['n_folds']
        self.cv_scores = metadata['cv_scores']
        
        # Load models
        self.models = []
        for i in range(self.n_folds):
            if self.task_type == 'regression':
                model = xgb.XGBRegressor()
            else:
                model = xgb.XGBClassifier()
            model.load_model(path / f'model_fold_{i}.json')
            self.models.append(model)
        
        # Load feature importance
        importance_path = path / 'feature_importance.npy'
        if importance_path.exists():
            self.feature_importance = np.load(importance_path)


def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task_type: TaskType = 'regression',
    feature_names: Optional[List[str]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Run a complete experiment with train/test split.
    
    Args:
        X_train, y_train: Training data
        X_test, y_test: Test data
        task_type: 'regression' or 'classification'
        feature_names: Optional feature names
        verbose: Print progress
        
    Returns:
        Dictionary with CV scores and test scores
    """
    predictor = XGBoostPredictor(task_type=task_type)
    
    # Train with CV on training set
    if verbose:
        print("Training with cross-validation...")
    cv_scores = predictor.train_cv(X_train, y_train, feature_names, verbose)
    
    # Evaluate on test set
    if verbose:
        print("Evaluating on test set...")
    
    if task_type == 'regression':
        y_pred = predictor.predict(X_test)
        
        test_scores = {
            'test_mae': mean_absolute_error(y_test, y_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'test_r2': r2_score(y_test, y_pred),
        }
    else:
        y_pred = predictor.predict(X_test)
        y_prob = predictor.predict_proba(X_test)
        
        test_scores = {
            'test_accuracy': accuracy_score(y_test, y_pred),
            'test_auc': roc_auc_score(y_test, y_prob),
            'test_f1': f1_score(y_test, y_pred),
        }
    
    if verbose:
        print(f"Test scores: {test_scores}")
    
    return {**cv_scores, **test_scores}








