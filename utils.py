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

    Parameters:
    df (pd.DataFrame): The input DataFrame(with multi-level index).
    split_ratio (float): The ratio to split the data into training and testing sets.

    Returns:
    pd.DataFrame, pd.DataFrame: Training and testing DataFrames.
    """

    dates = df.index.get_level_values(0).unique().sort_values()

    split_date = dates[int(len(dates) * split_ratio)]

    train_df = df.xs(slice(None, split_date), level=0, drop_level=False)
    test_df = df.xs(slice(split_date, None), level=0, drop_level=False)

    return train_df, test_df