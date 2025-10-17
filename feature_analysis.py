"""
feature_importance_analysis.py

This module provides a utility function to visualize and analyze feature importances 
from trained machine learning models (e.g., XGBoost, RandomForest). 

It generates a horizontal bar chart of the top N most important features for a given 
model and regime, helping to interpret which input variables contribute most to the 
model’s predictions.

Functions:
    analyze_feature_importance(model, X_cols, regime, top_n=20):
        Plots and returns the sorted feature importances for a trained model.
"""
import pandas as pd
import matplotlib.pyplot as plt

def analyze_feature_importance(model, x_cols, regime, top_n=20):
    """
    Plot and return feature importance for a trained model
    """
    importance_df = pd.DataFrame({
        'feature': x_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    # Plot top features
    plt.figure(figsize=(10, 8))
    top_features = importance_df.head(top_n)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Features - {regime.upper()}')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

    return importance_df
