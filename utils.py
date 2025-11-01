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