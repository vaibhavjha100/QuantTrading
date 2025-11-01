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

    df = pd.read_csv(file_path, index_col=[1, 0])
    df.index = df.index.set_levels([pd.to_datetime(df.index.levels[0]), df.index.levels[1]])

    # Sort the date index
    df = df.sort_index(level=0)

    return df

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

def get_tax_date(dates, date):
    """
    Get the closest date to the next march 31st from the given date in dates.
    If today is 01-01-2023, the next march 31st is 31-03-2023.
    If today is 05-04-2023, the next march 31st is 31-03-2024.
    If there is no date available after the next march 31st, return the last date in dates.

    Parameters:
    dates (pd.DatetimeIndex): The available dates.
    date (pd.Timestamp): The reference date.
    Returns:
    pd.Timestamp: The closest date to the next march 31st.
    """

    year = date.year

    if (date.month, date.day) > (3, 31):
        year += 1
    tax_date = pd.Timestamp(year=year, month=3, day=31)
    future_dates = dates[dates >= tax_date]
    if len(future_dates) == 0:
        return dates[-1]
    return future_dates[0]