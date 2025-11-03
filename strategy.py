"""
Module for designing and implementing trading strategies.
"""

import numpy as np
import pandas as pd
import utils
import random
import matplotlib.pyplot as plt
import seaborn as sns
import mplfinance as mpf

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

        # Slice train and test data to only include specified tickers
        if tickers:
            self.train_data = train_data.loc[train_data.index.get_level_values(1).isin(tickers)]
            self.test_data = test_data.loc[test_data.index.get_level_values(1).isin(tickers)]
        else:
            self.tickers = train_data.index.get_level_values(1).unique().tolist()

        self.train_dates = self.train_data.index.get_level_values(0).unique().sort_values()


        self.test_dates = self.test_data.index.get_level_values(0).unique().sort_values()

        self.tax_loss_carryforward = 0
        self.tax_loss_carryforward_years = 8
        self.tax_paid = 0
        self.transaction_costs_paid = 0

        self.history = {
            'Date' : [],
            'Portfolio Value' : [],
            'Cash' : [],
            'Positions' : [],
            'Trades' : []
        }

    def reset(self):
        """
        Reset the strategy to initial state.
        """
        self.position = {ticker: 0 for ticker in self.tickers}
        self.cash = self.initial_cash
        self.portfolio_value = self.initial_cash
        self.entry_prices = {ticker: 0 for ticker in self.tickers}
        self.exit_prices = {ticker: 0 for ticker in self.tickers}
        self.tax_loss_carryforward = 0
        self.tax_loss_carryforward_years = 8
        self.tax_paid = 0
        self.transaction_costs_paid = 0
        self.history = {
            'Date' : [],
            'Portfolio Value' : [],
            'Cash' : [],
            'Positions' : [],
            'Trades' : []
        }

    def execute_strategy(self, train=True, allocation="equal"):
        """
        Execute the trading strategy on the training or testing data.
        This provides the basic structure for executing the strategy.
        Parameters:
            train (bool): If True, execute on training data; otherwise, on testing data.
            allocation (str): Allocation strategy, e.g., "equal" for equal allocation.
        Returns:
            None
        Saves the trading history and portfolio value over time.
        """

        # Reset the strategy state
        self.reset()

        data = self.train_data if train else self.test_data
        dates = self.train_dates if train else self.test_dates

        for current_date in dates:
            daily_data = data.xs(current_date, level=0)

            trades = []
            signals = []
            for ticker in self.tickers:
                if ticker not in daily_data.index:
                    continue

                # Execution prices
                price = daily_data.loc[ticker]['Execution Price']

                # Slice the data to feed into the strategy
                # Make sure that there is no lookahead bias by only using data up to the current date
                # Data should only be for the ticker in this iteration
                ticker_data = data.xs(ticker, level=1)
                ticker_data_slice = ticker_data[ticker_data.index <= current_date]

                # Call the get_trade_signal method to get the trade signal
                signal = self.get_trade_signal(ticker_data_slice)

                signals.append((ticker, signal, price))

            # Execute trades based on signals

            # First handle sells to free up cash
            sell_tickers = [s for s in signals if s[1] == 'SELL']
            buy_tickers = [s for s in signals if s[1] == 'BUY']
            hold_tickers = [s for s in signals if s[1] == 'HOLD']

            # Remove tickers from sell tickers if we have no position
            sell_tickers = [s for s in sell_tickers if self.position[s[0]] > 0]

            # Execute sells
            if sell_tickers:
                trades.extend([(s[0], 'SELL', self.position[s[0]], s[2]) for s in sell_tickers])
                self.transaction_costs_paid += np.sum([self.position[s[0]] * s[2] * self.transaction_cost for s in sell_tickers])
                self.cash+= np.sum([self.position[s[0]] * s[2] * (1 - self.transaction_cost) for s in sell_tickers])
                self.position.update({s[0]: 0 for s in sell_tickers})

            # Check if it is the tax payment date
            tax_date = utils.get_tax_date(dates, current_date)
            if current_date == tax_date:
                # Set the payment date to the previous date in dates from current_date
                payment_date = dates[dates < tax_date][-1]
                # Call the tax payment method
                tax_payment = self.calculate_tax_payment(dates, payment_date)
                self.cash -= tax_payment

            # Calculate available cash for buys
            available_cash = self.cash

            # Execute buys

            if buy_tickers and available_cash > 0:
                # Calculate the minimum price among buy tickers
                min_price = min([s[2] for s in buy_tickers])

                # If min_price greater than available cash, skip buys
                if min_price > available_cash:
                    buy_tickers = []
                else:
                    if allocation == "equal":
                        allocation_per_ticker = available_cash / len(buy_tickers)

                        # Remove tickers that cannot be bought with the allocated cash
                        affordable_buy_tickers = []
                        for s in buy_tickers:
                            if allocation_per_ticker >= s[2]:
                                affordable_buy_tickers.append(s)

                        buy_tickers = affordable_buy_tickers

                if len(buy_tickers) > 0:
                    allocation_per_ticker = available_cash / len(buy_tickers)
                    for s in buy_tickers:
                        num_shares = int(allocation_per_ticker / s[2])
                        cost = num_shares * s[2] * (1 + self.transaction_cost)
                        if cost <= self.cash:
                            self.position[s[0]] += num_shares
                            self.cash -= cost
                            self.transaction_costs_paid += num_shares * s[2] * self.transaction_cost
                            trades.append((s[0], 'BUY', num_shares, s[2]))


            # Update portfolio value
            self.portfolio_value = self.cash + np.sum([self.position[ticker] * daily_data.loc[ticker]['Execution Price'] for ticker in self.tickers if ticker in daily_data.index])

            # Record history
            self.history['Date'].append(current_date)
            self.history['Portfolio Value'].append(self.portfolio_value)
            self.history['Cash'].append(self.cash)
            self.history['Positions'].append(self.position.copy())
            self.history['Trades'].append(trades)

    def calculate_tax_payment(self, dates, tax_date):
        """
        Calculate the tax payment due on the tax date.

        Parameters:
        dates (pd.DatetimeIndex): The available dates.
        tax_date (pd.Timestamp): The tax payment date.

        Returns:
        float: The tax payment amount.
        """

        # Start date should be 1st April of the previous year
        start_date = pd.Timestamp(year=tax_date.year - 1, month=4, day=1)

        # Slice the date and data to get the relevant period
        period_dates = dates[(dates >= start_date) & (dates <= tax_date)]

        # Use self.history to calculate profit
        # profit is calculated as portfolio value on tax date - portfolio value on start date
        start_value = self.history['Portfolio Value'][self.history['Date'].index(period_dates[0])]
        end_value = self.history['Portfolio Value'][self.history['Date'].index(period_dates[-1])]

        total_profit = end_value - start_value

        # Adjust profit with tax loss carryforward

        if total_profit < 0:
            # Update tax loss carryforward
            self.tax_loss_carryforward += -total_profit
            self.tax_loss_carryforward_years -= 1
            total_profit = 0
        else:
            if self.tax_loss_carryforward > 0:
                if total_profit >= self.tax_loss_carryforward:
                    total_profit -= self.tax_loss_carryforward
                    self.tax_loss_carryforward = 0
                    self.tax_loss_carryforward_years = 8
                    self.tax_paid += total_profit * self.tax_rate
                else:
                    self.tax_loss_carryforward -= total_profit
                    self.tax_loss_carryforward_years -= 1
                    total_profit = 0

        tax_payment = total_profit * self.tax_rate if total_profit > 0 else 0

        return tax_payment

    def get_trade_signal(self, df):
        """
        Abstract method to get trade signal for a given ticker data slice.
        Must be implemented by subclasses.

        Parameters:
        df (pd.DataFrame): DataFrame slice for a specific ticker up to the current date.

        Returns:
        str: Trade signal ('BUY', 'SELL', 'HOLD').
        """
        raise NotImplementedError("Subclasses must implement this method.")

    def plot_candlestick(self, ticker, train=True, start_date=None, end_date=None):
        """
        Plot candlestick chart for a given ticker.

        Parameters:
        ticker (str): Ticker symbol.
        train (bool): If True, plot training data; otherwise, testing data.
        start_date (pd.Timestamp): Start date for the plot.
        end_date (pd.Timestamp): End date for the plot.

        Returns:
        None
        """

        data = self.train_data if train else self.test_data
        dates = self.train_dates if train else self.test_dates

        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = dates[-1]

        df = data.xs(ticker, level=1).copy()
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        df.index.name = 'Date'
        # Make sure index is datetime
        df.index = pd.to_datetime(df.index)
        mpf.plot(df, type='candle', style='charles', title=f'Candlestick chart for {ticker}', volume=True)


class ExampleStrategy(BaseStrategy):
    """
    Example implementation of a trading strategy.
    Buys when the price is below the 20-day moving average and sells when above.
    """

    def get_trade_signal(self, df):
        """
        Get trade signal based on random strategy.

        Parameters:
        df (pd.DataFrame): DataFrame slice for a specific ticker up to the current date.

        Returns:
        str: Trade signal ('BUY', 'SELL', 'HOLD').
        """

        if len(df) < 2:
            return 'HOLD'

        signal = random.choice(['BUY', 'SELL', 'HOLD'])

        return signal




