"""Functions to evaluate the diabetes model."""

import pandas as pd
from sklearn.metrics import roc_auc_score


def compute_roc_auc(
    df: pd.DataFrame, target: str, prediction_column: str = "predictions"
) -> float:
    """Compute the ROC AUC between the real target and the predicted probabilities."""
    return roc_auc_score(df[target], df[prediction_column])
