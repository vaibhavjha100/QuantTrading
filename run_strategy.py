"""
Module to run a specified trading strategy using the strategy module.
"""

import logging
import pandas as pd
import utils
import numpy
import warnings
warnings.filterwarnings("ignore")
import os
import importlib

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

    tickers = ["ABB.NS", "MPHASIS.NS"]
    strategies_dir = "strategies"

    # Get all filenames in strategies directory
    strategy_files = [f for f in os.listdir(strategies_dir) if f.endswith(".py")]

    # Remove .py extension to get module names
    strategy_modules = [os.path.splitext(f)[0] for f in strategy_files]

    for module_name in strategy_modules:
        module = importlib.import_module(f"{strategies_dir}.{module_name}")
        strategy_class = getattr(module, module_name)
        strat = strategy_class(train_df, test_df, tickers=tickers, benchmark=benchmark_df)
        strat.name = module_name
        strat.export_results()
