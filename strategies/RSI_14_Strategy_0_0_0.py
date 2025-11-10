"""
Simple RSI (Relative Strength Index) Strategy Implementation
with a 14-period lookback.

Version: 0.0.0

Buy: When RSI < 30
Sell: When RSI > 70
Hold: Otherwise

The class inherits from BaseStrategy and implements the get_trade_signal method.
"""

from strategy import BaseStrategy

class RSI_14_Strategy_0_0_0(BaseStrategy):
    """
    RSI 14 Strategy Implementation.

    Inherits from BaseStrategy and implements RSI-based trade signals.
    """

    def get_trade_signal(self, df):
        """
        Get trade signal based on RSI strategy.

        Parameters:
        df (pd.DataFrame): DataFrame slice for a specific ticker up to the current date.

        Returns:
        str: Trade signal ('BUY', 'SELL', 'HOLD').
        """

        if len(df) < 15:
            return 'HOLD'  # Not enough data to calculate RSI

        rsi = df['RSI14'].iloc[-1]

        if rsi < 30:
            return 'BUY'
        elif rsi > 70:
            return 'SELL'
        else:
            return 'HOLD'