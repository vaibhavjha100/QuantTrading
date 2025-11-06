"""
Module to run a specified trading strategy using the strategy module.
"""

from strategy import ExampleStrategy
import logging
import pandas as pd
import utils
import numpy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    # Load data
    df = utils.load_multiindex_csv("features.csv")

    # Split data into training and testing sets
    train_df, test_df = utils.split_data(df, split_ratio=0.8)
    benchmark_df = pd.read_csv("benchmark_data.csv", index_col=0, parse_dates=True)

    logger.info(f"Training data shape: {train_df.shape}")
    logger.info(f"Testing data shape: {test_df.shape}")

    ES = ExampleStrategy(train_df, test_df, tickers=["ABB.NS", "MPHASIS.NS"], benchmark_data=benchmark_df)

    ES.execute_strategy()

    logger.info("Strategy execution completed.")

    ES.print_summary()

    ES.indicators = ["SMA10", "RSI14"]

    ES.plot_candlestick("ABB.NS", start_date="2015-01-01", end_date="2015-03-31")