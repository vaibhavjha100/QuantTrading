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

    Following technical indicators are added:

    Moving Averages
    SMA5, SMA10, SMA20, SMA21, SMA30, SMA50, SMA100, SMA200 | EMA5, EMA9, EMA10, EMA12, EMA20, EMA21, EMA26, EMA50, EMA200 | WMA10, WMA20, WMA50 | HMA9, HMA16, HMA20 | VWAP (intraday/daily)

    Momentum Oscillators
    RSI9, RSI14, RSI21, RSI25 | Stochastic (5,3,3), (9,3,2), (14,3,3), (14,5,3), (21,5,3), (21,9,3) | Williams %R9, %R14, %R21 | CCI5, CCI14, CCI20, CCI30, CCI50 | MFI14, MFI20 | ROC10, ROC12, ROC25 | Stochastic RSI14, RSI21 | Momentum10, Momentum14, Momentum20

    Trend Indicators
    MACD (5,13,8), (8,17,9), (8,21,5), (12,26,9), (13,30,10), (19,39,9) | ADX7, ADX14, ADX20, ADX28 | DMI14, DMI20 | Parabolic SAR (0.01/0.1), (0.015/0.15), (0.02/0.2), (0.025/0.25) | Supertrend ATR10/M2, ATR10/M2.5, ATR10/M3, ATR14/M3, ATR20/M4 | Aroon14, Aroon25 | Ichimoku (9,26,52), (6,13,26), (10,30,60)

    Volatility Indicators
    Bollinger Bands (10,1.5), (14,2), (20,2), (50,2.5) | ATR10, ATR14, ATR20, ATR21 | Keltner Channel (15,10,1.5), (20,14,2), (20,20,2), (30,21,2.5) | Donchian 10, 20, 50, 55 | Standard Deviation 20, 50

    Volume Indicators
    OBV (cumulative) | VWAP | CMF20, CMF21 | A/D Line (cumulative) | Volume Oscillator (5,28), (14,50) | PVT (cumulative) | Volume Profile

    Support/Resistance
    Fibonacci Retracement (23.6%, 38.2%, 50%, 61.8%, 78.6%) | Fibonacci Extension (100%, 127.2%, 161.8%, 200%, 261.8%) | Pivot Points (Standard, Fibonacci, Camarilla, Woodie)

    Other Indicators
    TRIX12, TRIX15, TRIX20 | Ultimate Oscillator (7,14,28) | Elder Ray13 | Detrended Price Oscillator20 | Envelope (20/2%, 50/5%) | Ease of Movement14

    Args:
        df (pd.DataFrame): Multi-index DataFrame with OHLCV data.
    Returns:
        pd.DataFrame: DataFrame with added technical indicators.
    """

    # Moving Averages

    # Simple Moving Averages (SMA)

    df['SMA5'] = ta.trend.sma_indicator(df['Close'], window=5)
    df['SMA10'] = ta.trend.sma_indicator(df['Close'], window=10)
    df['SMA20'] = ta.trend.sma_indicator(df['Close'], window=20)
    df['SMA21'] = ta.trend.sma_indicator(df['Close'], window=21)
    df['SMA30'] = ta.trend.sma_indicator(df['Close'], window=30)
    df['SMA50'] = ta.trend.sma_indicator(df['Close'], window=50)
    df['SMA100'] = ta.trend.sma_indicator(df['Close'], window=100)
    df['SMA200'] = ta.trend.sma_indicator(df['Close'], window=200)

    # Exponential Moving Averages (EMA)

    df['EMA5'] = ta.trend.ema_indicator(df['Close'], window=5)
    df['EMA9'] = ta.trend.ema_indicator(df['Close'], window=9)
    df['EMA10'] = ta.trend.ema_indicator(df['Close'], window=10)
    df['EMA12'] = ta.trend.ema_indicator(df['Close'], window=12)
    df['EMA20'] = ta.trend.ema_indicator(df['Close'], window=20)
    df['EMA21'] = ta.trend.ema_indicator(df['Close'], window=21)
    df['EMA26'] = ta.trend.ema_indicator(df['Close'], window=26)
    df['EMA50'] = ta.trend.ema_indicator(df['Close'], window=50)
    df['EMA200'] = ta.trend.ema_indicator(df['Close'], window=200)

    # Weighted Moving Averages (WMA)

    df['WMA10'] = ta.trend.wma_indicator(df['Close'], window=10)
    df['WMA20'] = ta.trend.wma_indicator(df['Close'], window=20)
    df['WMA50'] = ta.trend.wma_indicator(df['Close'], window=50)

    # Hull Moving Averages (HMA)

    df['HMA9'] = ta.trend.hull_moving_average(df['Close'], window=9)
    df['HMA16'] = ta.trend.hull_moving_average(df['Close'], window=16)
    df['HMA20'] = ta.trend.hull_moving_average(df['Close'], window=20)

    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = ta.volume.volume_weighted_average_price(df['High'], df['Low'], df['Close'], df['Volume'])

    # Momentum Oscillators

    # Relative Strength Index (RSI)

    df['RSI9'] = ta.momentum.rsi(df['Close'], window=9)
    df['RSI14'] = ta.momentum.rsi(df['Close'], window=14)
    df['RSI21'] = ta.momentum.rsi(df['Close'], window=21)
    df['RSI25'] = ta.momentum.rsi(df['Close'], window=25)

    # Stochastic Oscillator

    df['Stochastic_5_3_3'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=5, smooth_window=3)
    df['Stochastic_9_3_2'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=9, smooth_window=2)
    df['Stochastic_14_3_3'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stochastic_14_5_3'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['Stochastic_21_5_3'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=21, smooth_window=3)
    df['Stochastic_21_9_3'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=21, smooth_window=3)

    # Williams %R

    df['Williams_%R9'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=9)
    df['Williams_%R14'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=14)
    df['Williams_%R21'] = ta.momentum.williams_r(df['High'], df['Low'], df['Close'], lbp=21)

    # Commodity Channel Index (CCI)

    df['CCI5'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=5)
    df['CCI14'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=14)
    df['CCI20'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=20)
    df['CCI30'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=30)
    df['CCI50'] = ta.trend.cci(df['High'], df['Low'], df['Close'], window=50)

    # Money Flow Index (MFI)

    df['MFI14'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    df['MFI20'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=20)

    # Rate of Change (ROC)

    df['ROC10'] = ta.momentum.roc(df['Close'], window=10)
    df['ROC12'] = ta.momentum.roc(df['Close'], window=12)
    df['ROC25'] = ta.momentum.roc(df['Close'], window=25)

    # Stochastic RSI

    df['Stochastic_RSI14'] = ta.momentum.stochrsi(df['Close'], window=14)
    df['Stochastic_RSI21'] = ta.momentum.stochrsi(df['Close'], window=21)

    # Momentum

    df['Momentum10'] = ta.momentum.momentum(df['Close'], window=10)
    df['Momentum14'] = ta.momentum.momentum(df['Close'], window=14)
    df['Momentum20'] = ta.momentum.momentum(df['Close'], window=20)

    # Trend Indicators

    # Moving Average Convergence Divergence (MACD)

    macd_5_13_8 = ta.trend.MACD(df['Close'], window_slow=13, window_fast=5, window_sign=8)
    df['MACD_5_13_8'] = macd_5_13_8.macd()
    macd_8_17_9 = ta.trend.MACD(df['Close'], window_slow=17, window_fast=8, window_sign=9)
    df['MACD_8_17_9'] = macd_8_17_9.macd()
    macd_8_21_5 = ta.trend.MACD(df['Close'], window_slow=21, window_fast=8, window_sign=5)
    df['MACD_8_21_5'] = macd_8_21_5.macd()
    macd_12_26_9 = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['MACD_12_26_9'] = macd_12_26_9.macd()
    macd_13_30_10 = ta.trend.MACD(df['Close'], window_slow=30, window_fast=13, window_sign=10)
    df['MACD_13_30_10'] = macd_13_30_10.macd()
    macd_19_39_9 = ta.trend.MACD(df['Close'], window_slow=39, window_fast=19, window_sign=9)
    df['MACD_19_39_9'] = macd_19_39_9.macd()

    # Average Directional Index (ADX)

    df['ADX7'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=7)
    df['ADX14'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=14)
    df['ADX20'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=20)
    df['ADX28'] = ta.trend.adx(df['High'], df['Low'], df['Close'], window=28)

    # Directional Movement Index (DMI)

    df['DMI14'] = ta.trend.dmi(df['High'], df['Low'], df['Close'], window=14)
    df['DMI20'] = ta.trend.dmi(df['High'], df['Low'], df['Close'], window=20)

    # Parabolic SAR

    df['Parabolic_SAR_0.01_0.1'] = ta.trend.psar(df['High'], df['Low'], df['Close'], step=0.01, max_step=0.1)
    df['Parabolic_SAR_0.015_0.15'] = ta.trend.psar(df['High'], df['Low'], df['Close'], step=0.015, max_step=0.15)
    df['Parabolic_SAR_0.02_0.2'] = ta.trend.psar(df['High'], df['Low'], df['Close'], step=0.02, max_step=0.2)
    df['Parabolic_SAR_0.025_0.25'] = ta.trend.psar(df['High'], df['Low'], df['Close'], step=0.025, max_step=0.25)

    # Supertrend (Using ATR)

    df['Supertrend_ATR10_M2'] = ta.trend.supertrend(df['High'], df['Low'], df['Close'], window=10, multiplier=2)
    df['Supertrend_ATR10_M2.5'] = ta.trend.supertrend(df['High'], df['Low'], df['Close'], window=10, multiplier=2.5)
    df['Supertrend_ATR10_M3'] = ta.trend.supertrend(df['High'], df['Low'], df['Close'], window=10, multiplier=3)
    df['Supertrend_ATR14_M3'] = ta.trend.supertrend(df['High'], df['Low'], df['Close'], window=14, multiplier=3)
    df['Supertrend_ATR20_M4'] = ta.trend.supertrend(df['High'], df['Low'], df['Close'], window=20, multiplier=4)

    # Aroon Indicator

    df['Aroon14'] = ta.trend.aroon_up(df['Close'], window=14) - ta.trend.aroon_down(df['Close'], window=14)
    df['Aroon25'] = ta.trend.aroon_up(df['Close'], window=25) - ta.trend.aroon_down(df['Close'], window=25)

    # Ichimoku

    ichimoku_9_26_52 = ta.trend.IchimokuIndicator(df['High'], df['Low'], window1=9, window2=26, window3=52)
    df['Ichimoku_9_26_52'] = ichimoku_9_26_52.ichimoku_a() - ichimoku_9_26_52.ichimoku_b()
    ichimoku_6_13_26 = ta.trend.IchimokuIndicator(df['High'], df['Low'], window1=6, window2=13, window3=26)
    df['Ichimoku_6_13_26'] = ichimoku_6_13_26.ichimoku_a() - ichimoku_6_13_26.ichimoku_b()
    ichimoku_10_30_60 = ta.trend.IchimokuIndicator(df['High'], df['Low'], window1=10, window2=30, window3=60)
    df['Ichimoku_10_30_60'] = ichimoku_10_30_60.ichimoku_a() - ichimoku_10_30_60.ichimoku_b()

    # Volatility Indicators

    # Bollinger Bands (Both upper and lower bands are added)

    # __ indicates decimal point in column names

    bollinger_10_1__5 = ta.volatility.BollingerBands(df['Close'], window=10, window_dev=1.5)
    df['Bollinger_10_1.5_Upper'] = bollinger_10_1__5.bollinger_hband()
    df['Bollinger_10_1.5_Lower'] = bollinger_10_1__5.bollinger_lband()
    bollinger_14_2 = ta.volatility.BollingerBands(df['Close'], window=14, window_dev=2)
    df['Bollinger_14_2_Upper'] = bollinger_14_2.bollinger_hband()
    df['Bollinger_14_2_Lower'] = bollinger_14_2.bollinger_lband()
    bollinger_20_2 = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['Bollinger_20_2_Upper'] = bollinger_20_2.bollinger_hband()
    df['Bollinger_20_2_Lower'] = bollinger_20_2.bollinger_lband()
    bollinger_50_2__5 = ta.volatility.BollingerBands(df['Close'], window=50, window_dev=2.5)
    df['Bollinger_50_2.5_Upper'] = bollinger_50_2__5.bollinger_hband()
    df['Bollinger_50_2.5_Lower'] = bollinger_50_2__5.bollinger_lband()

    # Average True Range (ATR)

    df['ATR10'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=10)
    df['ATR14'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['ATR20'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=20)
    df['ATR21'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=21)

    # Keltner Channel (Both upper and lower bands are added)

    keltner_15_10_1__5 = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=15, window_atr=10, fillna=False)
    df['Keltner_15_10_1.5_Upper'] = keltner_15_10_1__5.keltner_channel_hband()
    df['Keltner_15_10_1.5_Lower'] = keltner_15_10_1__5.keltner_channel_lband()
    keltner_20_14_2 = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20, window_atr=14, fillna=False)
    df['Keltner_20_14_2_Upper'] = keltner_20_14_2.keltner_channel_hband()
    df['Keltner_20_14_2_Lower'] = keltner_20_14_2.keltner_channel_lband()
    keltner_20_20_2 = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=20, window_atr=20, fillna=False)
    df['Keltner_20_20_2_Upper'] = keltner_20_20_2.keltner_channel_hband()
    df['Keltner_20_20_2_Lower'] = keltner_20_20_2.keltner_channel_lband()
    keltner_30_21_2__5 = ta.volatility.KeltnerChannel(df['High'], df['Low'], df['Close'], window=30, window_atr=21, fillna=False)
    df['Keltner_30_21_2.5_Upper'] = keltner_30_21_2__5.keltner_channel_hband()
    df['Keltner_30_21_2.5_Lower'] = keltner_30_21_2__5.keltner_channel_lband()

    # Donchian Channel (Both upper and lower bands are added)

    donchian_10 = ta.volatility.DonchianChannel(df['High'], df['Low'], window=10)
    df['Donchian_10_Upper'] = donchian_10.donchian_channel_hband()
    df['Donchian_10_Lower'] = donchian_10.donchian_channel_lband()
    donchian_20 = ta.volatility.DonchianChannel(df['High'], df['Low'], window=20)
    df['Donchian_20_Upper'] = donchian_20.donchian_channel_hband()
    df['Donchian_20_Lower'] = donchian_20.donchian_channel_lband()
    donchian_50 = ta.volatility.DonchianChannel(df['High'], df['Low'], window=50)
    df['Donchian_50_Upper'] = donchian_50.donchian_channel_hband()
    df['Donchian_50_Lower'] = donchian_50.donchian_channel_lband()
    donchian_55 = ta.volatility.DonchianChannel(df['High'], df['Low'], window=55)
    df['Donchian_55_Upper'] = donchian_55.donchian_channel_hband()
    df['Donchian_55_Lower'] = donchian_55.donchian_channel_lband()

    # Standard Deviation

    df['STD20'] = ta.volatility.stdev(df['Close'], window=20)
    df['STD50'] = ta.volatility.stdev(df['Close'], window=50)

    # Volume Indicators

    # On-Balance Volume (OBV)

    df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])

    # Chaikin Money Flow (CMF)

    df['CMF20'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=20)
    df['CMF21'] = ta.volume.chaikin_money_flow(df['High'], df['Low'], df['Close'], df['Volume'], window=21)

    # Accumulation/Distribution Line (A/D Line)

    df['AD_Line'] = ta.volume.acc_dist_index(df['High'], df['Low'], df['Close'], df['Volume'])

    # Volume Oscillator

    df['Volume_Oscillator_5_28'] = ta.volume.volume_oscillator(df['Volume'], window_slow=28, window_fast=5)
    df['Volume_Oscillator_14_50'] = ta.volume.volume_oscillator(df['Volume'], window_slow=50, window_fast=14)

    # Price Volume Trend (PVT)

    df['PVT'] = ta.volume.price_volume_trend(df['Close'], df['Volume'])

    # Support/Resistance

    # Fibonacci Retracement
    # To be added



def add_execution_price(df, spread_coeff=0.1, sigma_noise=0.005):
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

    df['Mid'] = (df['High'] + df['Low']) / 2
    df['Spread_est'] = spread_coeff * (df['High'] - df['Low']) / df['Mid']

    np.random.seed(42)  # For reproducibility
    noise = np.random.normal(0, sigma_noise, size=len(df))

    df['Execution Price'] = df['Mid'] + (df['Spread_est'] / 2) + noise
    df['Execution Price'] = df['Execution Price'].clip(lower=0)

    # Drop intermediate columns
    df = df.drop(columns=['Mid', 'Spread_est'])

    return df

def get_features(file_path):
    """
    Load OHLCV data and add technical indicators and execution price.
    Args:
        file_path (str): Path to the CSV file.
    Returns:
        pd.DataFrame: DataFrame with technical indicators and execution price.
    Saves the data to 'features.csv'.
    """

    df = load_ohlcv_data(file_path)
    df = add_technical_indicators(df)
    df = add_execution_price(df)
    df.to_csv("features.csv")
    print("Features data saved to 'features.csv'")
    return df