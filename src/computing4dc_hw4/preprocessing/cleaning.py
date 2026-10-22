"""Functions to clean missing values in the diabetes data."""

import pandas as pd


def drop_missing_rows(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Remove rows that have NaN values in any of the given columns."""
    return df.dropna(subset=columns)


def fill_missing_with_mean(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Replace NaN values in the given columns with the column mean."""
    df = df.copy()
    df[columns] = df[columns].fillna(df[columns].mean())
    return df
