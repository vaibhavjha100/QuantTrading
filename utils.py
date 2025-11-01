"""
Utility functions for various common tasks.
"""

import pandas as pd


def load_multiindex_csv(file_path):
    """
    Load a CSV file with multi-level indexing.
    For our use case, the first level is 'Ticker' and the second level is 'Date'.

    Parameters:
    file_path (str): Path to the CSV file.

    Returns:
    pd.DataFrame: DataFrame with multi-level index.
    """
    return pd.read_csv(file_path, index_col=[1, 0])

def split_data(df, split_ratio=0.8):
    """
    Split the DataFrame into training and testing sets based on the split ratio.
    Respects multi-level index (Ticker, Date) to ensure proper temporal split per ticker.

    Parameters:
    df (pd.DataFrame): The input DataFrame with multi-level index (Ticker, Date).
    split_ratio (float): The ratio to split the data into training and testing sets.

    Returns:
    pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """

    df = df.copy()

    dates = df.index.get_level_values(0).unique().sort_values()

    df = df.sort_index()

    split_date = dates[int(len(dates) * split_ratio)]

    train_mask = df.index.get_level_values(0) < split_date
    test_mask = df.index.get_level_values(0) >= split_date
    train_df = df[train_mask]
    test_df = df[test_mask]

    return train_df, test_df