"""
Module for creating technical features and execution price for OHLCV data.
"""

import pandas as pd
import ta
import numpy as np

def load_ohlcv_data(file_path):
    """
    Load OHLCV data from a CSV file.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: Multi-index DataFrame with OHLCV data.
    """

    df = pd.read_csv("ohlcv_data.csv", index_col=[1, 0])
    return df

def add_technical_indicators(df):
    """
    Add technical indicators to the OHLCV DataFrame.
    Args:
        df (pd.DataFrame): Multi-index DataFrame with OHLCV data.
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """

    pass

def add_execution_price(df, spread_coeff=0.1, sigma_noise=0.01):
    """
    Add execution price to the OHLCV DataFrame.
    There are two components to the execution price:
    1. Spread estimation based on high and low prices.
    2. Random noise to simulate market impact and slippage.

    Args:
        df (pd.DataFrame): Multi-index DataFrame with OHLCV data.
        spread_coeff (float): Coefficient to estimate spread.
        sigma_noise (float): Standard deviation of noise to add.
    Returns:
        pd.DataFrame: DataFrame with added execution price.
    """

    df['mid'] = (df['High'] + df['Low']) / 2
    df['spread_est'] = spread_coeff * (df['High'] - df['Low']) / df['mid']

    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, sigma_noise, size=len(df))

    df['execution_price'] = df['mid'] + (df['spread_est'] / 2) + noise
    df['execution_price'] = df['execution_price'].clip(lower=0)

    return df