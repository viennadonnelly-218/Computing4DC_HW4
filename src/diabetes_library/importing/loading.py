"""Functions to load the diabetes data and split it into train and test sets."""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def split_data(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data into train and test sets."""
    train, test = train_test_split(df, test_size=test_size, random_state=random_state)
    return train, test
