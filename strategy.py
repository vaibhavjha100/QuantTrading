"""
Module for designing and implementing trading strategies.
"""

import numpy as np
import pandas as pd
import utils
import random
import plotly.graph_objects as go
import os
import pickle

class BaseStrategy:
    """
    Base class for trading strategies.
    """

    def __init__(self, train_data, test_data, initial_cash=1_00_000, tickers=[], transaction_cost=0.003, tax_rate=0.2, rf=0.065, benchmark=None):
        """
        Initialize the strategy with training and testing data.

        Parameters:
        train_data (pd.DataFrame): Training data with multi-level index (Ticker, Date).
        test_data (pd.DataFrame): Testing data with multi-level index (Ticker, Date).
        initial_cash (float): Initial cash for trading.
        tickers (list): List of tickers to consider.
        transaction_cost (float): Transaction cost rate.
        tax_rate (float): Tax rate on profits.
        """
        self.train_data = train_data
        self.test_data = test_data
        self.initial_cash = initial_cash
        self.tickers = tickers
        self.transaction_cost = transaction_cost
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
        self.rf = rf

        self.indicators = []
        self.benchmark = benchmark

        self.name = ''

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
            if self.portfolio_value <= 0:
                print("Portfolio value has dropped to zero or below.")
                print("Declaring bankruptcy and stopping strategy execution.")
                break
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
                ticker_data_slice = ticker_data[ticker_data.index < current_date]

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

        indicators = self.indicators

        history_df = pd.DataFrame(self.history)
        history_df['Date'] = pd.to_datetime(history_df['Date'])
        history_df.set_index('Date', inplace=True)

        history_df.sort_index(inplace=True)



        data = self.train_data if train else self.test_data
        dates = self.train_dates if train else self.test_dates

        if start_date is None:
            start_date = dates[0]
        if end_date is None:
            end_date = dates[-1]

        df = data.xs(ticker, level=1).copy()
        df = df[(df.index >= start_date) & (df.index <= end_date)]

        # Slice trades for the specified date range
        trades = history_df['Trades']
        trades = trades[(history_df.index >= start_date) & (history_df.index <= end_date)]
        # Slice trades for the specified ticker
        trades = trades.apply(lambda x: [trade for trade in x if trade[0] == ticker])



        df.index.name = 'Date'
        # Make sure index is datetime
        df.index = pd.to_datetime(df.index)

        colors = [
            'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray',
            'olive', 'cyan', 'magenta', 'teal', 'navy', 'maroon', 'lime', 'aqua',
            'coral', 'gold', 'indigo', 'khaki', 'lavender', 'salmon', 'sienna',
            'steelblue', 'tomato', 'turquoise', 'violet', 'yellowgreen', 'darkblue',
            'darkgreen', 'darkred', 'darkgray', 'darkmagenta', 'darkcyan',
            'darkseagreen', 'darkslateblue', 'darkslategray', 'darkturquoise',
            'lightblue', 'lightcoral', 'lightcyan', 'lightgreen', 'lightgray',
            'lightpink', 'lightsalmon', 'lightseagreen', 'lightskyblue',
            'lightslategray', 'lightsteelblue', 'lightyellow'
        ]
        fig = go.Figure()

        fig.add_trace(go.Candlestick(x=df.index,
                                     open=df['Open'],
                                     high=df['High'],
                                     low=df['Low'],
                                     close=df['Close'],
                                     name='Candlestick'))
        # Add trade markers
        buy_dates = []
        sell_dates = []
        buy_prices = []
        sell_prices = []

        for trade_date, trade_list in trades.items():
            for trade in trade_list:
                trade_ticker, action, quantity, price = trade

                if action.lower() == 'buy':
                    buy_dates.append(trade_date)
                    buy_prices.append(price)
                elif action.lower() == 'sell':
                    sell_dates.append(trade_date)
                    sell_prices.append(price)

        fig.add_trace(go.Scatter(x=buy_dates, y=buy_prices, mode='markers', name='Buy',
                                 marker=dict(symbol='triangle-up', color='lime', size=12), line=dict(width=1, color='black')))

        fig.add_trace(go.Scatter(x=sell_dates, y=sell_prices, mode='markers', name='Sell',
                                 marker=dict(symbol='triangle-down', color='magenta', size=12), line=dict(width=1, color='black')))

        # Add indicators

        for indicator in indicators:
            if indicator in df.columns:
                color = colors[indicators.index(indicator) % len(colors)]
                # fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color)))

                if any(indicator.startswith(x) for x in ['SMA', 'EMA', 'WMA', 'HMA', 'VWAP']):
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))

                elif any(x in indicator for x in ['RSI', 'Stochastic', 'Williams %R', 'CCI', 'MFI', 'ROC', 'Stochastic RSI']):
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y2'))

                elif any(x in indicator for x in ['MACD', 'ADX', 'DMI', 'Parabolic SAR', 'Aroon', 'Ichimoku']):
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y2'))

                elif 'Bollinger' in indicator:
                    color = "blue"
                    if 'Upper' in indicator:
                        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))
                    elif 'Lower' in indicator:
                        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))
                    else:
                        fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))

                elif any(x in indicator for x in ['ATR', 'Keltner', 'Donchian', 'Standard Deviation']):
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))

                elif any(x in indicator for x in
                         ['OBV', 'VWAP', 'CMF', 'A/D Line', 'Volume Oscillator', 'PVT', 'Volume Profile']):
                    fig.add_trace(go.Bar(x=df.index, y=df[indicator], name=indicator, marker_color=color, yaxis='y3'))

                elif any(x in indicator for x in ['Fibonacci', 'Pivot Points']):
                    # Add as horizontal lines at the last value
                    fig.add_hline(y=df[indicator].iloc[-1], line_dash="dot", annotation_text=indicator, line_color=color)

                else:
                    fig.add_trace(go.Scatter(x=df.index, y=df[indicator], mode='lines', name=indicator, line=dict(width=2, color=color), yaxis='y'))

        #fig.update_layout(title=f'Candlestick chart for {ticker}', yaxis_title='Price', xaxis_title='Date')
        fig.update_layout(
            title=f'Candlestick chart for {ticker}',
            xaxis=dict(title='Date'),
            yaxis=dict(title='Price'),
            yaxis2=dict(title='Indicators', overlaying='y', side='right', showgrid=False),
            yaxis3=dict(title='Volume', anchor='free', overlaying='y', side='right', position=0.95, showgrid=False),
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))

        fig.show()

    def print_summary(self):
        """
        Print a summary of the strategy execution.
        """
        # self.calculate_performance_metrics()

        metrics = self.performance_metrics

        print("\n" + "=" * 80)
        print(" " * 20 + "STRATEGY PERFORMANCE REPORT")
        print("=" * 80)

        for category, stats in metrics.items():
            print(f"\n--- {category} ---")
            for stat_name, value in stats.items():
                if isinstance(value, float):
                    print(f"{stat_name}: {value:,.4f}")
                else:
                    print(f"{stat_name}: {value}")

        print("\n" + "=" * 80)



    def _get_portfolio(self):
        """
        Create a daily portfolio DataFrame from the history.
        """
        portfolio_df = pd.DataFrame(self.history)
        portfolio_df['Date'] = pd.to_datetime(portfolio_df['Date'])
        portfolio_df.set_index('Date', inplace=True)
        portfolio_df.sort_index(inplace=True)
        self.portfolio = portfolio_df

    def _calculate_return_metrics(self, df):
        """
        Calculate return metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing return metrics.
        """
        df = df.copy()

        initial_value = self.initial_cash
        final_value = df['Portfolio Value'].iloc[-1]

        total_return = (final_value - initial_value) / initial_value

        total_gross_profit = final_value - initial_value + self.tax_paid + self.transaction_costs_paid
        total_net_profit = final_value - initial_value

        # Calculate number of years
        num_days = (df.index[-1] - df.index[0]).days
        num_years = num_days / 365.25

        annualized_return = (1 + total_return) ** (1 / num_years) - 1

        return {
            'Total Return': total_return,
            'Annualized Return': annualized_return,
            'Total Gross Profit': total_gross_profit,
            'Total Net Profit': total_net_profit,
            'Number of Years': num_years,
            'Number of Days': len(df)
        }


    def _calculate_risk_metrics(self, df, confidence_level=0.95):
        """
        Calculate risk metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
            confidence_level (float): Confidence level for VaR calculation.
        Returns:
            dict: Dictionary containing risk metrics.
        """

        df = df.copy()

        cumulative_max = df['Portfolio Value'].cummax()
        drawdowns = (df['Portfolio Value'] - cumulative_max) / cumulative_max

        max_drawdown = drawdowns.min()
        avg_drawdown = drawdowns.mean()

        # Calculate maximum drawdown duration
        in_drawdown = drawdowns < 0
        drawdown_durations = []
        current_period = 0

        for is_dd in in_drawdown:
            if is_dd:
                current_period += 1
            else:
                if current_period > 0:
                    drawdown_durations.append(current_period)
                    current_period = 0

        if current_period > 0:
            drawdown_durations.append(current_period)

        max_drawdown_duration = max(drawdown_durations) if drawdown_durations else 0

        # Volatility

        daily_volatility = df['Returns'].std()
        annualized_volatility = daily_volatility * np.sqrt(252)

        # Downside Deviation
        downside_returns = df['Returns'][df['Returns'] < 0]
        daily_downside_deviation = downside_returns.std()
        annualized_downside_deviation = daily_downside_deviation * np.sqrt(252)

        # VaR
        var = np.percentile(df['Returns'], (1 - confidence_level) * 100)

        # CVaR
        cvar = df['Returns'][df['Returns'] <= var].mean()

        return {
            'Max Drawdown': max_drawdown,
            'Avg Drawdown': avg_drawdown,
            'Max Drawdown Duration (days)': max_drawdown_duration,
            'Annualized Volatility': annualized_volatility,
            'Annualized Downside Deviation': annualized_downside_deviation,
            'VaR ({}%)'.format(int(confidence_level * 100)): var,
            'CVaR ({}%)'.format(int(confidence_level * 100)): cvar
        }

    def _calculate_risk_adjusted_metrics(self, df):
        """
        Calculate risk-adjusted metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing risk-adjusted metrics.
        """

        rf = self.rf

        df = df.copy()

        return_metrics = self._calculate_return_metrics(df)
        risk_metrics = self._calculate_risk_metrics(df)

        annualized_return = return_metrics['Annualized Return']
        annualized_volatility = risk_metrics['Annualized Volatility']

        sharpe_ratio = (annualized_return - rf) / annualized_volatility if annualized_volatility != 0 else np.nan

        downside_deviation = risk_metrics['Annualized Downside Deviation']
        sortino_ratio = (annualized_return - rf) / downside_deviation if downside_deviation != 0 else np.nan

        max_drawdown = risk_metrics['Max Drawdown']
        calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else np.nan

        trade_stats = self._calculate_trade_statistics(df)
        win_rate = trade_stats['Win Rate'] if 'Win Rate' in trade_stats else np.nan
        avg_win = trade_stats['Average Win'] if 'Average Win' in trade_stats else np.nan
        avg_loss = trade_stats['Average Loss'] if 'Average Loss' in trade_stats else np.nan
        avg_loss = abs(avg_loss) if not np.isnan(avg_loss) else np.nan

        expectency = (win_rate * avg_win) - ((1 - win_rate) * avg_loss) if not np.isnan(win_rate) and not np.isnan(
            avg_win) and not np.isnan(avg_loss) else np.nan

        return {
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Calmar Ratio': calmar_ratio,
            'Expectency': expectency
        }

    def _calculate_trade_statistics(self, df):
        """
        Calculate trade statistics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing trade statistics.
        """
        all_trades = []
        trade_durations = []
        trade_entry_dates = {}

        for idx, trades in enumerate(self.history['Trades']):
            current_date = self.history['Date'][idx]
            for trade in trades:
                ticker, action, quantity, price = trade

                if action == 'BUY':
                    if ticker not in trade_entry_dates:
                        trade_entry_dates[ticker] = []
                    trade_entry_dates[ticker].append({'entry_date': current_date, 'quantity': quantity, 'entry_price': price})

                elif action == 'SELL' and ticker in trade_entry_dates and trade_entry_dates[ticker]:
                    entries = trade_entry_dates[ticker]
                    remaining_quantity = quantity
                    total_pnl = 0

                    while remaining_quantity > 0 and entries:
                        entry = entries[0]
                        entry_quantity = min(remaining_quantity, entry['quantity'])

                        pnl = (price - entry['entry_price']) * entry_quantity
                        total_pnl += pnl

                        duration = (current_date - entry['entry_date']).days
                        trade_durations.append(duration)

                        if entry['quantity'] <= remaining_quantity:
                            entries.pop(0)
                        else:
                            entry['quantity'] -= entry_quantity

                    all_trades.append(total_pnl)

        num_trades = len(all_trades)

        winning_trades = [trade for trade in all_trades if trade > 0]
        losing_trades = [trade for trade in all_trades if trade <= 0]

        num_wins = len(winning_trades)
        num_losses = len(losing_trades)

        win_rate = num_wins / num_trades if num_trades > 0 else np.nan
        avg_win = np.mean(winning_trades) if num_wins > 0 else np.nan
        avg_loss = np.mean(losing_trades) if num_losses > 0 else np.nan
        largest_win = np.max(winning_trades) if num_wins > 0 else np.nan
        largest_loss = np.min(losing_trades) if num_losses > 0 else np
        avg_trade = np.mean(all_trades) if num_trades > 0 else np.nan
        avg_duration = np.mean(trade_durations) if trade_durations else np.nan

        return {
            'Number of Trades': num_trades,
            'Number of Winning Trades': num_wins,
            'Number of Losing Trades': num_losses,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Largest Win': largest_win,
            'Largest Loss': largest_loss,
            'Average Trade P&L': avg_trade,
            'Average Trade Duration (days)': avg_duration
        }

    def _calculate_streak_metrics(self, df):
        """
        Calculate streak metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing streak metrics.
        """

        df = df.copy()

        daily_pnl = df['Portfolio Value'].diff().fillna(0)

        win_days = (daily_pnl > 0).astype(int)

        streaks = []
        current_streak = 0
        current_type = None

        for is_win in win_days:
            if is_win == current_type:
                current_streak += 1
            else:
                if current_streak > 0:
                    streaks.append((current_type, current_streak))
                current_type = is_win
                current_streak = 1

        if current_streak > 0:
            streaks.append((current_type, current_streak))

        win_streaks = [s for s in streaks if s[0] == 1]
        loss_streaks = [s for s in streaks if s[0] == 0]

        max_consecutive_wins = max([s[1] for s in win_streaks], default=0)
        max_consecutive_losses = max([s[1] for s in loss_streaks], default=0)
        avg_consecutive_wins = np.mean([s[1] for s in win_streaks]) if win_streaks else 0
        avg_consecutive_losses = np.mean([s[1] for s in loss_streaks]) if loss_streaks else 0
        num_win_streaks = len(win_streaks)
        num_loss_streaks = len(loss_streaks)

        return {
            'Max Consecutive Wins': max_consecutive_wins,
            'Max Consecutive Losses': max_consecutive_losses,
            'Avg Consecutive Wins': avg_consecutive_wins,
            'Avg Consecutive Losses': avg_consecutive_losses,
            'Number of Win Streaks': num_win_streaks,
            'Number of Loss Streaks': num_loss_streaks
        }

    def _calculate_portfolio_metrics(self, df):
        """
        Calculate all portfolio metrics.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing all portfolio metrics.
        """

        df = df.copy()

        avg_portfolio_value = df['Portfolio Value'].mean()
        final_portfolio_value = df['Portfolio Value'].iloc[-1]
        avg_cash = df['Cash'].mean()

        total_positions = 0
        position_count = 0

        for positions in df['Positions']:
            for ticker, qty in positions.items():
                if qty > 0:
                    total_positions += qty
                    position_count += 1

        position_count = len(df['Positions'])
        avg_num_positions = total_positions / position_count if position_count > 0 else 0

        total_trade_value = 0

        for trades in df['Trades']:
            for trade in trades:
                ticker, action, quantity, price = trade
                total_trade_value += quantity * price

        portfolio_turnover = total_trade_value / avg_portfolio_value if avg_portfolio_value > 0 else np.nan

        return {
            'Average Portfolio Value': avg_portfolio_value,
            'Final Portfolio Value': final_portfolio_value,
            'Average Cash': avg_cash,
            'Average Number of Positions': avg_num_positions,
            'Portfolio Turnover': portfolio_turnover
        }

    def _calculate_cost_tax_metrics(self, df):
        """
        Calculate cost and tax metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
        Returns:
            dict: Dictionary containing cost and tax metrics.
        """
        total_ret_pct = (self.portfolio_value - self.initial_cash) / self.initial_cash * 100

        # Transaction Costs as % of Returns
        if total_ret_pct != 0:
            transaction_costs_pct_of_returns = (self.transaction_costs_paid / self.initial_cash) / (total_ret_pct / 100) * 100
            tax_pct_of_returns = (self.tax_paid / self.initial_cash) / (total_ret_pct / 100) * 100
        else:
            transaction_costs_pct_of_returns = 0
            tax_pct_of_returns = 0

        return {
            'Total Transaction Costs Paid': self.transaction_costs_paid,
            'Total Tax Paid': self.tax_paid,
            'Transaction Costs as % of Returns': transaction_costs_pct_of_returns,
            'Tax as % of Returns': tax_pct_of_returns
        }

    def _calculate_benchmark_metrics(self, df, benchmark):
        """
        Calculate benchmark comparison metrics for the strategy.
        Parameters:
            df (pd.DataFrame): DataFrame containing portfolio values over time.
            benchmark (pd.Series): DataFrame containing benchmark values over time.
        Returns:
            dict: Dictionary containing benchmark comparison metrics.
        """
        df = df.copy()
        benchmark = benchmark.copy()

        benchmark_returns = benchmark['Returns']
        strategy_returns = df['Returns']

        # Align the indices
        combined = pd.concat([strategy_returns, benchmark_returns], axis=1, join='inner')
        combined.columns = ['Strategy Returns', 'Benchmark Returns']

        # Drop NaN values
        combined.dropna(inplace=True)

        # Calculate Beta
        covariance = combined.cov().iloc[0, 1]
        benchmark_variance = combined['Benchmark Returns'].var()
        beta = covariance / benchmark_variance

        # Calculate Alpha
        rf = self.rf
        annualized_strategy_return = (1 + strategy_returns.mean()) ** 252 - 1
        annualized_benchmark_return = (1 + benchmark_returns.mean()) ** 252 - 1

        alpha = annualized_strategy_return - (rf + beta * (annualized_benchmark_return - rf))

        return {
            'Beta': beta,
            'Alpha': alpha,
            'Correlation with Benchmark': combined.corr().iloc[0, 1]
        }


    def calculate_performance_metrics(self):
        """
        Calculate performance metrics for the strategy.
        """
        if not hasattr(self, 'portfolio'):
            self._get_portfolio()

        df = self.portfolio.copy()
        benchmark = self.benchmark

        df['Returns'] = df['Portfolio Value'].pct_change().fillna(0)
        benchmark['Returns'] = benchmark['Close'].pct_change().fillna(0)

        metrics = {}

        metrics['Returns & Profitability'] = self._calculate_return_metrics(df)
        metrics['Risk'] = self._calculate_risk_metrics(df)
        metrics['Risk-Adjusted Performance'] = self._calculate_risk_adjusted_metrics(df)
        metrics['Trade Statistics'] = self._calculate_trade_statistics(df)
        metrics['Streaks'] = self._calculate_streak_metrics(df)
        metrics['Portfolio'] = self._calculate_portfolio_metrics(df)
        metrics['Costs & Taxes'] = self._calculate_cost_tax_metrics(df)
        metrics['Benchmark Comparison'] = self._calculate_benchmark_metrics(df, benchmark)

        self.performance_metrics = metrics

    def export_results(self):
        """
        Export the created object in a pickle file after running the strategy, getting the portfolio and calculating performance metrics.
        Save the pickele file in the 'results' directory with the strategy name.
        """

        # Execute strategy if not already done
        if not self.history['Date']:
            self.execute_strategy()

        # Get portfolio if not already done
        if not hasattr(self, 'portfolio'):
            self._get_portfolio()

        # Calculate performance metrics if not already done
        if not hasattr(self, 'performance_metrics'):
            self.calculate_performance_metrics()

        file_path = "results/"
        strategy_dir = os.path.dirname(file_path)

        if not os.path.exists(strategy_dir):
            os.makedirs(strategy_dir)

        # Export object as pickle
        pickle_file = os.path.join(strategy_dir, f'{self.name}.pkl')

        with open(pickle_file, 'wb') as f:
            pickle.dump(self, f)


    def load_results(self):
        """
        Load the object from a pickle file.
        Load the pickele file from the 'results' directory with the strategy name.
        Returns:
            Strategy object: Loaded strategy object.
        """
        file_path = "results/"
        strategy_dir = os.path.join(os.path.dirname(file_path), self.name)
        pickle_file = os.path.join(strategy_dir, f'{self.name}.pkl')

        if not os.path.exists(pickle_file):
            raise FileNotFoundError(f"No saved results found for strategy '{self.name}'.")

        with open(pickle_file, 'rb') as f:
            loaded_strategy = pickle.load(f)

        return loaded_strategy







