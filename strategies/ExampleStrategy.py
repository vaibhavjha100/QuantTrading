"""
Example trading strategy module.

This module defines an ExampleStrategy class that inherits from BaseStrategy.

It implements a simple random trade signal generator for demonstration purposes.
"""
import random
from strategy import BaseStrategy

class ExampleStrategy(BaseStrategy):
    """
    Example implementation of a trading strategy.
    Inherits from BaseStrategy and implements a random trade signal generator.
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