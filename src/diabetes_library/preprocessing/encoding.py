"""Functions to encode categorical columns of the diabetes data."""

import pandas as pd


def add_one_hot_dummies(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Add one-hot encoded columns for a categorical column."""
    dummies = pd.get_dummies(df[column], prefix=column, dtype=int)
    return pd.concat([df, dummies], axis=1)


def encode_binary(
    df: pd.DataFrame, column: str, positive_value: str, new_column: str
) -> pd.DataFrame:
    """Create a 0/1 column: 1 if df[column] == positive_value, 0 otherwise."""
    df = df.copy()
    df[new_column] = (df[column] == positive_value).astype(int)
    return df
