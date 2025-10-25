"""
Module for collecting OHLCV data through yfinance and saving it to CSV files.
"""

import yfinance as yf
import pandas as pd
import requests
from io import StringIO


def get_tickers():
    """
    Retrieve all tickers in NIFTY 500 automatically.
    Returns:
        list: List of ticker symbols with .NS suffix for NSE.
    """
    try:
        # Method 1: Fetch from NSE India website
        print("Fetching NIFTY 200 constituents from NSE India...")
        url = "https://www.niftyindices.com/IndexConstituent/ind_nifty200list.csv"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }

        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Read CSV data
        df = pd.read_csv(StringIO(response.text))

        # Extract symbols and add .NS suffix for Yahoo Finance
        symbols = df['Symbol'].tolist()
        tickers = [f"{symbol}.NS" for symbol in symbols]

        print(f"Successfully fetched {len(tickers)} tickers from NIFTY 500")
        return tickers

    except Exception as e:
        print(f"Error fetching from NSE: {e}")

        try:
            # Method 2: Fetch from Wikipedia as fallback
            print("Trying Wikipedia as fallback...")
            url = "https://en.wikipedia.org/wiki/NIFTY_500"

            tables = pd.read_html(url)
            # Find the table with company symbols
            for table in tables:
                if 'Symbol' in table.columns or 'Ticker' in table.columns:
                    symbol_col = 'Symbol' if 'Symbol' in table.columns else 'Ticker'
                    symbols = table[symbol_col].dropna().tolist()
                    tickers = [f"{symbol}.NS" for symbol in symbols if isinstance(symbol, str)]

                    if len(tickers) > 100:  # Sanity check
                        print(f"Successfully fetched {len(tickers)} tickers from Wikipedia")
                        return tickers

            raise Exception("Could not find valid ticker table in Wikipedia")

        except Exception as e2:
            print(f"Error fetching from Wikipedia: {e2}")
            print("Please install required packages: pip install requests lxml html5lib")
            raise Exception("Failed to fetch tickers from all sources")

def get_ohlcv_data(tickers, start_date, end_date, interval="1d"):
    """
    Fetch OHLCV data for given tickers using yfinance.
    Args:
        tickers (list): List of ticker symbols.
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' format.
        interval (str): Data interval.
    Returns:
        pd.DataFrame: DataFrame containing OHLCV data in multi-index format.
    Saves the data to 'ohlcv_data.csv'.
    """
    print(f"Fetching OHLCV data for {len(tickers)} tickers...")
    data = pd.DataFrame()

    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        stock_data = yf.download(ticker, start=start_date, end=end_date, interval=interval, multi_level_index=False)
        stock_data['Ticker'] = ticker
        data = pd.concat([data, stock_data])

    # Drop Adjusted Close if exists
    if 'Adj Close' in data.columns:
        data = data.drop(columns=['Adj Close'])

    # Set multi-index
    data.index.name = 'Date'
    data = data.set_index('Ticker', append=True)
    data = data.reorder_levels(['Ticker', 'Date'])

    # Filter out tickers with insufficient data
    bust_tickers = filter_tickers(data, min_start_date="2009-02-01")
    data = data.drop(index=bust_tickers, level='Ticker')

    # Save to CSV
    data.to_csv("ohlcv_data.csv")
    print("OHLCV data saved to 'ohlcv_data.csv'")

    return data

def filter_tickers(data, min_start_date="2009-02-01"):
    """
    Filter tickers based on minimum start date.
    Args:
        data (pd.DataFrame): Multi-index DataFrame with OHLCV data.
        min_start_date (str): Minimum start date in 'YYYY-MM-DD' format.
    Returns:
        list: List of tickers that do not meet the criteria.
    """
    bust_tickers = []
    grouped = data.groupby(level='Ticker')

    for ticker, group in grouped:
        min_date = group.index.get_level_values('Date').min()
        if min_date > pd.to_datetime(min_start_date):
            bust_tickers.append(ticker)

    print(f"Total tickers with min date after {min_start_date}: {len(bust_tickers)}")
    return bust_tickers

if __name__ == "__main__":

    tickers = get_tickers()

    ohlcv_data = get_ohlcv_data(tickers, start_date="2009-01-01", end_date="2025-01-01", interval="1d")

