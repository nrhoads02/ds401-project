import polars as pl

def calculate_ema(df: pl.DataFrame, smoothing: float, days: int) -> pl.Series:
    """
    Calculate the Exponential Moving Average (EMA) for the 'close' column.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing a 'close' column.
        smoothing (float): The smoothing constant.
        days (int): The period over which to calculate the EMA.
        
    Returns:
        pl.Series: EMA values as a Polars Series.
    """
    close = df["close"].to_list()
    ema_values = [close[0]]
    factor = smoothing / (1 + days)
    for i in range(1, len(close)):
        prev_ema = ema_values[-1]
        current_close = close[i]
        ema_values.append(current_close * factor + prev_ema * (1 - factor))
    return pl.Series(ema_values)

def calculate_rsi(df: pl.DataFrame, window: int) -> pl.Series:
    """
    Calculate the Relative Strength Index (RSI) over a given window.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing a 'close' column.
        window (int): The lookback period for RSI.
        
    Returns:
        pl.Series: RSI values.
    """
    close = df["close"].to_list()
    # Compute differences between consecutive closes.
    delta = [0] + [close[i] - close[i - 1] for i in range(1, len(close))]
    gain = [d if d > 0 else 0 for d in delta]
    loss = [-d if d < 0 else 0 for d in delta]
    
    avg_gain = []
    avg_loss = []
    for i in range(len(close)):
        if i < window:
            avg_gain.append(None)
            avg_loss.append(None)
        elif i == window:
            avg_gain_value = sum(gain[1:window+1]) / window
            avg_loss_value = sum(loss[1:window+1]) / window
            avg_gain.append(avg_gain_value)
            avg_loss.append(avg_loss_value)
        else:
            prev_gain = avg_gain[-1] if avg_gain[-1] is not None else 0
            prev_loss = avg_loss[-1] if avg_loss[-1] is not None else 0
            avg_gain.append((prev_gain * (window - 1) + gain[i]) / window)
            avg_loss.append((prev_loss * (window - 1) + loss[i]) / window)
    
    rsi_values = []
    for g, l in zip(avg_gain, avg_loss):
        if g is None or l is None or l == 0:
            rsi_values.append(None)
        else:
            rs = g / l
            rsi_values.append(100 - (100 / (1 + rs)))
    return pl.Series(rsi_values)

def calculate_macd(df: pl.DataFrame, fast_window: int = 12, slow_window: int = 26, signal_window: int = 9):
    """
    Calculate MACD, Signal Line, and MACD Histogram for the 'close' column.
    
    Returns:
        tuple(pl.Series, pl.Series, pl.Series): MACD line, Signal line, and Histogram.
    """
    fast_smoothing = 2 / (fast_window + 1)
    slow_smoothing = 2 / (slow_window + 1)
    signal_smoothing = 2 / (signal_window + 1)
    
    fast_ema = calculate_ema(df, fast_smoothing, fast_window)
    slow_ema = calculate_ema(df, slow_smoothing, slow_window)
    
    macd_line = fast_ema - slow_ema

    # Compute Signal Line using EMA on MACD values.
    macd_list = macd_line.to_list()
    signal_values = [macd_list[0]]
    factor = signal_smoothing
    for i in range(1, len(macd_list)):
        prev_signal = signal_values[-1]
        signal_values.append(macd_list[i] * factor + prev_signal * (1 - factor))
    signal_line = pl.Series(signal_values)
    macd_histogram = macd_line - signal_line
    return macd_line, signal_line, macd_histogram

def calculate_bollinger_bands(df: pl.DataFrame, window: int, multiplier: float = 2.0):
    """
    Calculate Bollinger Bands for the 'close' column.
    
    Parameters:
        df (pl.DataFrame): DataFrame with a 'close' column.
        window (int): The rolling window size.
        multiplier (float): The multiplier for the rolling standard deviation.
        
    Returns:
        tuple(pl.Series, pl.Series, pl.Series): (SMA, Upper Band, Lower Band)
    """
    sma = df["close"].rolling_mean(window_size=window)
    rolling_std = df["close"].rolling_std(window_size=window)
    upper_band = sma + multiplier * rolling_std
    lower_band = sma - multiplier * rolling_std
    return sma, upper_band, lower_band

def calculate_stochastic_oscillator(df: pl.DataFrame, window: int):
    """
    Calculate the Stochastic Oscillator (%K and %D).
    
    Parameters:
        df (pl.DataFrame): DataFrame containing 'high', 'low', and 'close'.
        window (int): The lookback window.
        
    Returns:
        tuple(pl.Series, pl.Series): %K and %D values.
    """
    low_n = df["low"].rolling_min(window_size=window)
    high_n = df["high"].rolling_max(window_size=window)
    k = 100 * (df["close"] - low_n) / (high_n - low_n)
    d = k.rolling_mean(window_size=3)
    return k, d

def calculate_cmf(df: pl.DataFrame, window: int):
    """
    Calculate the Chaikin Money Flow (CMF).
    
    Parameters:
        df (pl.DataFrame): DataFrame containing 'high', 'low', 'close', and 'volume'.
        window (int): The rolling window.
        
    Returns:
        pl.Series: CMF values.
    """
    mfm = (df["close"] - df["low"]) / (df["high"] - df["low"])
    mfv = mfm * df["volume"]
    rolling_mfv_sum = mfv.rolling_sum(window_size=window)
    rolling_vol_sum = df["volume"].rolling_sum(window_size=window)
    return rolling_mfv_sum / rolling_vol_sum

def calculate_fibonacci_retracement(df: pl.DataFrame) -> dict:
    """
    Calculate Fibonacci retracement levels based on the entire DataFrame range.
    
    Returns:
        dict: Mapping of level names to their corresponding price values.
    """
    high = df["high"].max()
    low = df["low"].min()
    price_range = high - low
    return {
        "Fibonacci_0": low,
        "Fibonacci_23.6": low + price_range * 0.236,
        "Fibonacci_38.2": low + price_range * 0.382,
        "Fibonacci_50": low + price_range * 0.5,
        "Fibonacci_61.8": low + price_range * 0.618,
        "Fibonacci_100": high
    }

def calculate_parabolic_sar(df: pl.DataFrame, acceleration_factor: float = 0.02, max_acceleration: float = 0.2) -> pl.Series:
    """
    Calculate the Parabolic SAR for the given OHLC data.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing 'high', 'low', and 'close'.
        acceleration_factor (float): The step value.
        max_acceleration (float): Maximum step (currently unused, kept for signature consistency).
        
    Returns:
        pl.Series: Parabolic SAR values.
    """
    n = df.height
    sar = [None] * n
    ep = [None] * n
    trend = [None] * n
    close_list = df["close"].to_list()
    high_list = df["high"].to_list()
    low_list = df["low"].to_list()
    
    sar[0] = close_list[0]
    ep[0] = high_list[0]
    trend[0] = 1  # Assume uptrend initially
    
    for i in range(1, n):
        if trend[i - 1] == 1:
            sar[i] = sar[i - 1] + acceleration_factor * (ep[i - 1] - sar[i - 1])
            ep[i] = max(high_list[i], ep[i - 1])
            if sar[i] > close_list[i]:
                sar[i] = ep[i - 1]
                trend[i] = -1
                ep[i] = low_list[i]
            else:
                trend[i] = 1
        else:
            sar[i] = sar[i - 1] + acceleration_factor * (ep[i - 1] - sar[i - 1])
            ep[i] = min(low_list[i], ep[i - 1])
            if sar[i] < close_list[i]:
                sar[i] = ep[i - 1]
                trend[i] = 1
                ep[i] = high_list[i]
            else:
                trend[i] = -1
    return pl.Series(sar)

def calculate_ichimoku_cloud(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate Ichimoku Cloud components.
    
    Returns:
        pl.DataFrame: DataFrame with columns: Tenkan_Sen, Kijun_Sen, Senkou_Span_A, Senkou_Span_B, Chikou_Span.
    """
    tenkan = (df["high"].rolling_max(window_size=9) + df["low"].rolling_min(window_size=9)) / 2
    kijun = (df["high"].rolling_max(window_size=26) + df["low"].rolling_min(window_size=26)) / 2
    senkou_span_a = ((tenkan + kijun) / 2).shift(26)
    senkou_span_b = ((df["high"].rolling_max(window_size=52) + df["low"].rolling_min(window_size=52)) / 2).shift(26)
    chikou = df["close"].shift(-26)
    return pl.DataFrame({
        "Tenkan_Sen": tenkan,
        "Kijun_Sen": kijun,
        "Senkou_Span_A": senkou_span_a,
        "Senkou_Span_B": senkou_span_b,
        "Chikou_Span": chikou
    })

def bullish_engulfing(df: pl.DataFrame) -> pl.Series:
    """
    Identify bullish engulfing candlestick patterns.
    """
    return (df["close"] > df["open"]) & \
           (df["close"].shift(1) < df["open"].shift(1)) & \
           (df["open"] < df["close"].shift(1))

def bearish_engulfing(df: pl.DataFrame) -> pl.Series:
    """
    Identify bearish engulfing candlestick patterns.
    """
    return (df["close"] < df["open"]) & \
           (df["close"].shift(1) > df["open"].shift(1)) & \
           (df["open"] > df["close"].shift(1))

def doji(df: pl.DataFrame, threshold: float = 0.1) -> pl.Series:
    """
    Identify doji candlestick patterns based on a threshold.
    """
    return ((df["close"] - df["open"]).abs() / df["open"]) < threshold

def morning_star(df: pl.DataFrame) -> pl.Series:
    """
    Identify morning star candlestick pattern.
    """
    return (df["close"].shift(2) < df["open"].shift(2)) & \
           (df["close"].shift(1) > df["open"].shift(1)) & \
           (df["close"] > df["open"].shift(1))

def add_candlestick_patterns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add candlestick pattern indicator columns to the DataFrame.
    """
    return df.with_columns([
        bullish_engulfing(df).alias("Bullish_Engulfing"),
        bearish_engulfing(df).alias("Bearish_Engulfing"),
        doji(df).alias("Doji"),
        morning_star(df).alias("Morning_Star")
    ])

def calculate_pivot_points(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate pivot points and associated support/resistance levels.
    """
    pivot = (df["high"] + df["low"] + df["close"]) / 3
    s1 = 2 * pivot - df["high"]
    r1 = 2 * pivot - df["low"]
    s2 = pivot - (df["high"] - df["low"])
    r2 = pivot + (df["high"] - df["low"])
    s3 = df["low"] - 2 * (df["high"] - pivot)
    r3 = df["high"] + 2 * (pivot - df["low"])
    return df.with_columns([
        pivot.alias("Pivot_Point"),
        s1.alias("S1"),
        r1.alias("R1"),
        s2.alias("S2"),
        r2.alias("R2"),
        s3.alias("S3"),
        r3.alias("R3")
    ])

def add_technical_indicators(df: pl.DataFrame) -> pl.DataFrame:
    """
    Add a comprehensive set of technical indicators to the OHLCV DataFrame.
    
    Parameters:
        df (pl.DataFrame): Input DataFrame containing at least the columns: date, open, high, low, close, volume.
        
    Returns:
        pl.DataFrame: DataFrame augmented with various technical indicator columns.
    """
    # Log Returns
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1)).log().alias("log_returns")
    )

    # Simple Moving Averages (SMA)
    df = df.with_columns([
        pl.col("close").rolling_mean(window_size=3).alias("SMA_3"),
        pl.col("close").rolling_mean(window_size=5).alias("SMA_5"),
        pl.col("close").rolling_mean(window_size=10).alias("SMA_10"),
        pl.col("close").rolling_mean(window_size=20).alias("SMA_20"),
    ])
    
    # Exponential Moving Averages (EMA)
    df = df.with_columns([
        calculate_ema(df, 2, 3).alias("EMA_3"),
        calculate_ema(df, 2, 5).alias("EMA_5"),
        calculate_ema(df, 2, 10).alias("EMA_10"),
        calculate_ema(df, 2, 20).alias("EMA_20"),
    ])
    
    # RSI Indicators
    df = df.with_columns([
        calculate_rsi(df, 7).alias("RSI_7"),
        calculate_rsi(df, 9).alias("RSI_9"),
        calculate_rsi(df, 14).alias("RSI_14"),
        calculate_rsi(df, 21).alias("RSI_21"),
        calculate_rsi(df, 30).alias("RSI_30")
    ])
    
    # MACD
    macd_line, signal_line, macd_histogram = calculate_macd(df)
    df = df.with_columns([
        macd_line.alias("MACD"),
        signal_line.alias("Signal_Line"),
        macd_histogram.alias("MACD_Histogram")
    ])
    
    # Bollinger Bands
    bb_mid20, bb_up20, bb_low20 = calculate_bollinger_bands(df, 20)
    bb_mid50, bb_up50, bb_low50 = calculate_bollinger_bands(df, 50)
    df = df.with_columns([
        bb_mid20.alias("BB_Middle_20"),
        bb_up20.alias("BB_Upper_20"),
        bb_low20.alias("BB_Lower_20"),
        bb_mid50.alias("BB_Middle_50"),
        bb_up50.alias("BB_Upper_50"),
        bb_low50.alias("BB_Lower_50")
    ])
    
    # Stochastic Oscillator
    stoch_k_14, stoch_d_14 = calculate_stochastic_oscillator(df, 14)
    stoch_k_9, stoch_d_9 = calculate_stochastic_oscillator(df, 9)
    stoch_k_21, stoch_d_21 = calculate_stochastic_oscillator(df, 21)
    df = df.with_columns([
        stoch_k_14.alias("Stochastic_%K_14"),
        stoch_d_14.alias("Stochastic_%D_14"),
        stoch_k_9.alias("Stochastic_%K_9"),
        stoch_d_9.alias("Stochastic_%D_9"),
        stoch_k_21.alias("Stochastic_%K_21"),
        stoch_d_21.alias("Stochastic_%D_21")
    ])
    
    # Chaikin Money Flow (CMF)
    df = df.with_columns([
        calculate_cmf(df, 10).alias("CMF_10"),
        calculate_cmf(df, 20).alias("CMF_20"),
        calculate_cmf(df, 50).alias("CMF_50"),
        calculate_cmf(df, 100).alias("CMF_100")
    ])
    
    # Fibonacci Retracement Levels (added as constant columns)
    fib_levels = calculate_fibonacci_retracement(df)
    for col_name, value in fib_levels.items():
        df = df.with_columns(pl.lit(value).alias(col_name))
    
    # Parabolic SAR
    df = df.with_columns([
        calculate_parabolic_sar(df).alias("Parabolic_SAR")
    ])
    
    # Ichimoku Cloud Components
    ichimoku = calculate_ichimoku_cloud(df)
    df = df.hstack(ichimoku)
    
    # Candlestick Patterns
    df = add_candlestick_patterns(df)
    
    # Pivot Points and Support/Resistance Levels
    df = calculate_pivot_points(df)
    
    return df

if __name__ == "__main__":
    # Read the OHLCV CSV and parse the 'date' column
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
    ohlcv = ohlcv.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # Add technical indicators to the DataFrame
    ohlcv_with_indicators = add_technical_indicators(ohlcv)

    print(ohlcv_with_indicators.estimated_size("mb"))
    
    # Print the resulting DataFrame
    print(ohlcv_with_indicators)
