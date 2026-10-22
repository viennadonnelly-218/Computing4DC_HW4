"""Functions to train the diabetes model and add its predictions."""

import pandas as pd
from sklearn.linear_model import LogisticRegression


def train_model(
    df: pd.DataFrame, features: list[str], target: str
) -> LogisticRegression:
    """Train a logistic regression using the given feature columns and target column."""
    model = LogisticRegression(max_iter=1000)
    model.fit(df[features], df[target])
    return model


def add_predictions(
    df: pd.DataFrame,
    model: LogisticRegression,
    features: list[str],
    column_name: str = "predictions",
) -> pd.DataFrame:
    """Return a copy of df with a new column holding the predicted probability of diabetes."""
    df = df.copy()
    df[column_name] = model.predict_proba(df[features])[:, 1]
    return df
