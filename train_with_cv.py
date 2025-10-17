"""
Provides utility functions for training specialized XGBoost Regressor models.

This module is designed to support a regime-switching strategy by training 
separate 'expert' models on subsets of the training data partitioned by a 
specified financial regime. The core function, `train_model_with_cv`, 
automates the configuration, cross-validation (CV), and final training 
of the XGBoost model.

The module focuses on robust model evaluation by calculating and reporting 
the Cross-Validation Root Mean Squared Error (RMSE) to gauge out-of-sample 
performance before the final model is fitted and returned.
"""

from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import numpy as np

def train_model_with_cv(df_train, regime, params=None, cv_folds=5, verbose=True):
    """
    Train XGB model with cross-validation scoring
    """
    params = params or dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=42,
        objective="reg:squarederror"
    )
    x_cols = [c for c in df_train.columns if c.startswith("feat_") or c.startswith("sent_")]
    df_sub = df_train[df_train["regime"] == regime]
    if df_sub.empty:
        raise ValueError(f"No rows for {regime}")
    x = df_sub[x_cols].values
    y = df_sub["target"].values
    # Cross-validation before final training
    model = XGBRegressor(**params)
    cv_scores = cross_val_score(model, x, y, cv=cv_folds, scoring='neg_mean_squared_error')
    # Final model training
    model.fit(x, y)
    if verbose:
        print(f"[TRAIN] {regime}: n={len(df_sub)} rows")
        print(
            f"[CV] {regime}: RMSE = {np.sqrt(-cv_scores.mean()):.4f} "
            f"(+/- {np.sqrt(cv_scores.std() * 2):.4f})"
        )
    return model, x_cols, cv_scores
