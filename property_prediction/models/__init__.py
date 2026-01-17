"""
Prediction models for property prediction.

Available models:
- XGBoostPredictor: XGBoost with cross-validation
"""

from .xgboost_predictor import XGBoostPredictor

__all__ = [
    'XGBoostPredictor',
]




