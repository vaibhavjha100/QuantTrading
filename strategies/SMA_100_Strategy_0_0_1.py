"""
Simple SMA (Simple Moving Average) Strategy Implementation
on crossover with a 100-period lookback.

Version: 0.0.1

Buy: When Close crosses above SMA100
Sell: When Close crosses below SMA100
Hold: Otherwise
"""

from strategy import BaseStrategy

class SMA_100_Strategy_0_0_1(BaseStrategy):
    """
    SMA 100 Strategy Implementation.

    Inherits from BaseStrategy and implements SMA-based trade signals.
    """

    def get_trade_signal(self, df):
        """
        Get trade signal based on SMA strategy.

        Parameters:
        df (pd.DataFrame): DataFrame slice for a specific ticker up to the current date.

        Returns:
        str: Trade signal ('BUY', 'SELL', 'HOLD').
        """

        if len(df) < 101:
            return 'HOLD'  # Not enough data to calculate SMA100

        close_price = df['Close'].iloc[-1]
        sma_100 = df['SMA100'].iloc[-1]

        if close_price > sma_100 and df['Close'].iloc[-2] <= df['SMA100'].iloc[-2]:
            return 'BUY'
        elif close_price < sma_100 and df['Close'].iloc[-2] >= df['SMA100'].iloc[-2]:
            return 'SELL'
        else:
            return 'HOLD'