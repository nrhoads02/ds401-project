"""
stock_technical_indicators.py

This module reads historical equities data (OHLCV) from a CSV file (which contains data for multiple symbols),
reindexes each symbol’s time series to fill in missing intervals based on the specified trading hours,
computes a range of technical indicators, and saves the enriched dataset to a new CSV file.

The computed indicators include:
  - Simple Moving Averages (SMA)
  - Exponential Moving Averages (EMA)
  - Relative Strength Index (RSI)
  - Bollinger Bands (Upper Band, Lower Band, and Moving Average)
  - Moving Average Convergence Divergence (MACD)
  - Average True Range (ATR)
"""

import pandas as pd
import os
import ta

def reindex_symbol_data(df, frequency="5min", start_time="09:30:00", end_time="16:00:00"):
    """
    Reindexes the DataFrame for one symbol to a complete timeline for each trading day.
    
    Parameters:
      df (DataFrame): Data for one symbol with a 'time' column.
      frequency (str): Frequency for the new index (e.g., "5min" for 5-minute bars).
      start_time (str): Trading day start time (in US/Eastern, e.g., "09:30:00").
      end_time (str): Trading day end time (in US/Eastern, e.g., "16:00:00").
    
    Returns:
      DataFrame: Reindexed DataFrame with missing intervals forward-filled.
    """
    # Ensure 'time' is datetime
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time')
    df = df.set_index('time')
    
    new_dfs = []
    # Group by each trading day (the index.date gives a naive date, so we'll use that)
    for day, group in df.groupby(df.index.date):
        # Create start and end timestamps in US/Eastern and convert to UTC.
        start_dt = pd.Timestamp(f"{day} {start_time}", tz="US/Eastern").tz_convert("UTC")
        end_dt = pd.Timestamp(f"{day} {end_time}", tz="US/Eastern").tz_convert("UTC")
        # Create a complete DateTimeIndex for that day
        trading_index = pd.date_range(start=start_dt, end=end_dt, freq=frequency)
        # Reindex the group's DataFrame to this index, forward-filling missing values.
        group = group.reindex(trading_index, method='ffill')
        # Reset index to bring the 'time' back as a column
        group = group.reset_index().rename(columns={"index": "time"})
        new_dfs.append(group)
    
    return pd.concat(new_dfs, ignore_index=True)

def compute_technical_indicators_for_group(df):
    """
    Computes technical indicators for one symbol's DataFrame.
    
    Parameters:
      df (DataFrame): DataFrame containing raw (and reindexed) OHLCV data for one symbol.
      
    Returns:
      DataFrame: The DataFrame with additional technical indicator columns.
    """
    df = df.sort_values('time').reset_index(drop=True)
    
    # Simple Moving Averages (SMA)
    df['sma_20'] = df['close'].rolling(window=20, min_periods=1).mean()
    df['sma_50'] = df['close'].rolling(window=50, min_periods=1).mean()
    
    # Exponential Moving Averages (EMA)
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Relative Strength Index (RSI) using a 14-period window
    df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
    
    # Bollinger Bands using a 20-period window and 2 standard deviations
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_lower'] = bollinger.bollinger_lband()
    df['bb_mavg']  = bollinger.bollinger_mavg()
    
    # MACD: compute MACD line, signal line, and histogram (difference)
    macd = ta.trend.MACD(close=df['close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()
    
    # Average True Range (ATR) using a 14-period window
    atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['atr'] = atr.average_true_range()
    
    return df

def compute_technical_indicators(input_file, output_file, frequency="5min", start_time="09:30:00", end_time="16:00:00"):
    """
    Reads a CSV file containing raw OHLCV data for multiple symbols, reindexes each symbol's data to a full timeline,
    computes technical indicators for each symbol separately, and writes the enriched dataset to a new CSV file.
    
    Parameters:
      input_file (str): Path to the raw CSV file.
      output_file (str): Path where the processed CSV will be saved.
      frequency (str): Frequency for reindexing (e.g., "5min").
      start_time (str): Trading start time in US/Eastern (e.g., "09:30:00").
      end_time (str): Trading end time in US/Eastern (e.g., "16:00:00").
    """
    print(f"Reading raw data from {input_file}...")
    df = pd.read_csv(input_file)
    df['time'] = pd.to_datetime(df['time'])
    
    df_list = []
    for symbol, group in df.groupby('symbol'):
        print(f"Processing symbol: {symbol}")
        # Reindex each symbol's data to fill missing time intervals
        group = reindex_symbol_data(group.copy(), frequency=frequency, start_time=start_time, end_time=end_time)
        group_indicators = compute_technical_indicators_for_group(group)
        # Optionally, add a column for the symbol (if not already present)
        group_indicators["symbol"] = symbol
        df_list.append(group_indicators)
    
    df_processed = pd.concat(df_list, ignore_index=True)
    # Sort overall data by symbol and time
    df_processed = df_processed.sort_values(['symbol', 'time'])
    
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df_processed = df_processed.dropna()
    df_processed.to_csv(output_file, index=False)
    print(f"Processed data with technical indicators saved to {output_file}")

if __name__ == "__main__":
    input_file = os.path.join("data", "raw", "stocks", "bars.csv")
    output_file = os.path.join("data", "processed", "stocks", "bars_with_indicators.csv")
    compute_technical_indicators(input_file, output_file, frequency="5min", start_time="09:30:00", end_time="16:00:00")
