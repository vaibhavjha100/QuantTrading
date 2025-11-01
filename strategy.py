"""
Module for designing and implementing trading strategies.
"""

import numpy as np
import pandas as pd
import utils

class BaseStrategy:
    """
    Base class for trading strategies.
    """

    def __init__(self, train_data, test_data, initial_cash=1_00_000, tickers=[], transcation_cost=0.003, tax_rate=0.2):
        """
        Initialize the strategy with training and testing data.

        Parameters:
        train_data (pd.DataFrame): Training data with multi-level index (Ticker, Date).
        test_data (pd.DataFrame): Testing data with multi-level index (Ticker, Date).
        initial_cash (float): Initial cash for trading.
        tickers (list): List of tickers to consider.
        transcation_cost (float): Transaction cost rate.
        tax_rate (float): Tax rate on profits.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.initial_cash = initial_cash
        self.tickers = tickers
        self.transaction_cost = transcation_cost
        self.tax_rate = tax_rate
        self.position = {ticker: 0 for ticker in tickers}
        self.cash = initial_cash
        self.portfolio_value = initial_cash
        self.entry_prices = {ticker: 0 for ticker in tickers}
        self.exit_prices = {ticker: 0 for ticker in tickers}
        self.train_dates = train_data.index.get_level_values(0).unique().sort_values()
        self.test_dates = test_data.index.get_level_values(0).unique().sort_values()

        self.history = {
            'Date' : [],
            'Portfolio Value' : [],
            'Cash' : [],
            'Positions' : [],
            'Trades' : []
        }


