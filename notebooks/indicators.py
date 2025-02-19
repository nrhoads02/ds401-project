import polars as pl

ohlcv_data = pl.read_csv("/Users/ryanfreidhoff/Desktop/DS 401/ds401-project/Data/Raw/stocks/csv/ohlcv.csv")

ohlcv_data = ohlcv_data.rename({"act_symbol": "Symbol"})

# Filter rows where Symbol is "AAPL"
aapl = ohlcv_data.filter(pl.col("Symbol") == "AAPL")

# Exponential Moving Average column calculated using the close
# smoothing is a constant, typically set to 2
# days is the period of days, typically set to 20
def calculate_ema(df, smoothing, days):
    close = df["close"]
    ema = [close[0]]  # Initialize EMA with the first close value

    for i in range(1, len(close)):
        prev_ema = ema[-1]
        current_close = close[i]
        if prev_ema is None:
            ema.append(current_close)
        else:
            ema.append(current_close * (smoothing)/(1+days) + prev_ema * (1 - (smoothing)/(1+days)))

    return pl.Series(ema)

# Add column for the Simple Moving Average (SMA) of Close using period 20
aapl = aapl.with_columns(
    [
        pl.col("close").rolling_mean(window_size=3).alias("SMA_3"),  # Add 3-period SMA
        pl.col("close").rolling_mean(window_size=5).alias("SMA_5"),  # Add 5-period SMA
        pl.col("close").rolling_mean(window_size=10).alias("SMA_10"),  # Add 10-period SMA
        pl.col("close").rolling_mean(window_size=20).alias("SMA_20"),  # Add 20-period SMA
        calculate_ema(aapl, 2, 3).alias("EMA_3"), # Add 3-period EMA
        calculate_ema(aapl, 2, 5).alias("EMA_5"), # Add 5-period EMA
        calculate_ema(aapl, 2, 10).alias("EMA_10"), # Add 10-period EMA
        calculate_ema(aapl, 2, 20).alias("EMA_20") # Add 20-period EMA
    ]
)

def calculate_rsi(df, window):
    delta = df["close"].diff().fill_null(0)  # Fill missing values with 0
    
    # Use Polars expressions for element-wise operations
    gain = pl.when(delta > 0).then(delta).otherwise(0)
    loss = pl.when(delta < 0).then(-delta).otherwise(0)
    
    # Calculate rolling means for gains and losses
    avg_gain = gain.rolling_mean(window)
    avg_loss = loss.rolling_mean(window)
    
    # Calculate the Relative Strength (RS) and RSI
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

# Add the RSI column to the DataFrame
aapl = aapl.with_columns(
    [
        calculate_rsi(aapl, 7).alias("RSI_7"), # Add 7 period RSI
        calculate_rsi(aapl, 9).alias("RSI_9"), # Add 9 period RSI
        calculate_rsi(aapl, 14).alias("RSI_14"), # Add 14 period RSI
        calculate_rsi(aapl, 21).alias("RSI_21"), # Add 21 period RSI
        calculate_rsi(aapl, 30).alias("RSI_30") # Add 30 period RSI
    ]
)

# Function to calculate MACD
def calculate_macd(df, fast_window=12, slow_window=26, signal_window=9):
    # Calculate the smoothing factor for each EMA (2 / (window + 1))
    fast_smoothing = 2 / (fast_window + 1)
    slow_smoothing = 2 / (slow_window + 1)
    signal_smoothing = 2 / (signal_window + 1)

    # Calculate Fast EMA and Slow EMA using custom function
    fast_ema = calculate_ema(df, fast_smoothing, fast_window)
    slow_ema = calculate_ema(df, slow_smoothing, slow_window)
    
    # Calculate MACD Line (difference between Fast and Slow EMAs)
    macd_line = fast_ema - slow_ema
    
    # Calculate Signal Line (EMA of the MACD line)
    signal_line = calculate_ema(df, signal_smoothing, signal_window)
    
    # Calculate MACD Histogram (optional)
    macd_histogram = macd_line - signal_line
    
    return macd_line, signal_line, macd_histogram

# Add MACD, Signal Line, and Histogram to the DataFrame
aapl = aapl.with_columns([
    calculate_macd(aapl)[0].alias("MACD"),  # MACD Line
    calculate_macd(aapl)[1].alias("Signal Line"),  # Signal Line
    calculate_macd(aapl)[2].alias("MACD Histogram")  # MACD Histogram (optional)
])

# Calculate Bollinger Bands
def calculate_bollinger_bands(df, window, multiplier=2.0):
    # Calculate the rolling mean (SMA)
    sma = df["close"].rolling_mean(window)
    
    # Calculate the rolling standard deviation
    rolling_std = df["close"].rolling_std(window)
    
    # Calculate the Upper and Lower Bands
    upper_band = sma + multiplier * rolling_std
    lower_band = sma - multiplier * rolling_std
    
    return sma, upper_band, lower_band

# Add the Bollinger Bands columns to the DataFrame
aapl = aapl.with_columns(
    [
        calculate_bollinger_bands(aapl, 20)[0].alias("BB_Middle_20"),   # Middle Band (SMA) for 20 period
        calculate_bollinger_bands(aapl, 20)[1].alias("BB_Upper_20"),    # Upper Band for 20 period
        calculate_bollinger_bands(aapl, 20)[2].alias("BB_Lower_20"),    # Lower Band for 20 period
        calculate_bollinger_bands(aapl, 50)[0].alias("BB_Middle_50"),   # Middle Band (SMA) for 50 period
        calculate_bollinger_bands(aapl, 50)[1].alias("BB_Upper_50"),    # Upper Band for 50 period
        calculate_bollinger_bands(aapl, 50)[2].alias("BB_Lower_50")     # Lower Band for 50 period
    ]
)

# Calculate Stochastic Oscillator
def calculate_stochastic_oscillator(df, window):
    # Calculate the rolling min (lowest low) and max (highest high) over the window period
    low_n = df["low"].rolling_min(window)
    high_n = df["high"].rolling_max(window)
    
    # Calculate %K (Stochastic Oscillator)
    K = 100 * (df["close"] - low_n) / (high_n - low_n)
    
    # Calculate %D (3-period SMA of %K)
    D = K.rolling_mean(3)
    
    return K, D

# Add Stochastic Oscillator columns to the DataFrame
aapl = aapl.with_columns(
    [
        calculate_stochastic_oscillator(aapl, 14)[0].alias("Stochastic_%K_14"),  # %K for 14 periods
        calculate_stochastic_oscillator(aapl, 14)[1].alias("Stochastic_%D_14"),  # %D (SMA of %K) for 14 periods
        calculate_stochastic_oscillator(aapl, 9)[0].alias("Stochastic_%K_9"),    # %K for 9 periods
        calculate_stochastic_oscillator(aapl, 9)[1].alias("Stochastic_%D_9"),    # %D (SMA of %K) for 9 periods
        calculate_stochastic_oscillator(aapl, 21)[0].alias("Stochastic_%K_21"),  # %K for 21 periods
        calculate_stochastic_oscillator(aapl, 21)[1].alias("Stochastic_%D_21")   # %D (SMA of %K) for 21 periods
    ]
)

# Calculate Chaikin Money Flow (CMF)
def calculate_cmf(df, window):
    # Calculate Money Flow Multiplier (MFM)
    mfm = (df["close"] - df["low"]) / (df["high"] - df["low"])
    
    # Calculate Money Flow Volume (MFV)
    mfv = mfm * df["volume"]
    
    # Calculate the rolling sum of MFV and volume over the window
    rolling_mfv_sum = mfv.rolling_sum(window)
    rolling_vol_sum = df["volume"].rolling_sum(window)
    
    # Calculate CMF as the ratio of the rolling sums
    cmf = rolling_mfv_sum / rolling_vol_sum
    
    return cmf

# Add CMF columns to the DataFrame
aapl = aapl.with_columns(
    [
        calculate_cmf(aapl, 10).alias("CMF_10"),   # CMF for 10 periods
        calculate_cmf(aapl, 20).alias("CMF_20"),   # CMF for 20 periods
        calculate_cmf(aapl, 50).alias("CMF_50"),   # CMF for 50 periods
        calculate_cmf(aapl, 100).alias("CMF_100")  # CMF for 100 periods
    ]
)

# Calculate Fibonacci retracement levels
def calculate_fibonacci_retracement(df, high_col, low_col):
    # Get the maximum and minimum price in the given window
    high = df[high_col].max()
    low = df[low_col].min()
    
    # Calculate the difference between high and low
    price_range = high - low
    
    # Calculate Fibonacci levels as a percentage of the price range
    fib_levels = {
        "0%": low,
        "23.6%": low + price_range * 0.236,
        "38.2%": low + price_range * 0.382,
        "50%": low + price_range * 0.5,
        "61.8%": low + price_range * 0.618,
        "100%": high
    }
    
    # Create a Polars expression for each level
    fib_expr = [pl.lit(fib_levels[key]).alias(f"Fibonacci_{key}") for key in fib_levels]
    
    return fib_expr

# Add Fibonacci retracement levels to the DataFrame
aapl = aapl.with_columns(
    calculate_fibonacci_retracement(aapl, "high", "low")
)

# Function to calculate Parabolic SAR
def calculate_parabolic_sar(df, acceleration_factor=0.02, max_acceleration=0.2):
    sar_values = [None] * len(df)  # List to hold SAR values
    ep_values = [None] * len(df)   # List to hold the Extreme Points (EP)
    trend = [None] * len(df)       # List to track uptrend (1) or downtrend (-1)
    
    # Initial SAR, EP, and trend
    sar_values[0] = df["close"][0]
    ep_values[0] = df["high"][0]  # For the first period, assuming an uptrend
    trend[0] = 1  # Starting with an uptrend
    
    # Start calculating SAR for each subsequent period
    for i in range(1, len(df)):
        if trend[i - 1] == 1:  # Uptrend
            sar_values[i] = sar_values[i - 1] + acceleration_factor * (ep_values[i - 1] - sar_values[i - 1])
            # Update EP (Extreme Point) to the highest high in the uptrend
            ep_values[i] = max(df["high"][i], ep_values[i - 1])
            
            # If SAR exceeds the price, switch to a downtrend
            if sar_values[i] > df["close"][i]:
                sar_values[i] = ep_values[i - 1]
                trend[i] = -1
                ep_values[i] = df["low"][i]  # Set new EP to the lowest low for downtrend
        else:  # Downtrend
            sar_values[i] = sar_values[i - 1] + acceleration_factor * (ep_values[i - 1] - sar_values[i - 1])
            # Update EP (Extreme Point) to the lowest low in the downtrend
            ep_values[i] = min(df["low"][i], ep_values[i - 1])
            
            # If SAR goes below the price, switch to an uptrend
            if sar_values[i] < df["close"][i]:
                sar_values[i] = ep_values[i - 1]
                trend[i] = 1
                ep_values[i] = df["high"][i]  # Set new EP to the highest high for uptrend
    
    return pl.DataFrame({
        "Parabolic_SAR": sar_values
    })

# Add Parabolic SAR to the DataFrame
aapl = aapl.with_columns(
    calculate_parabolic_sar(aapl).select("Parabolic_SAR")
)

# Calculate Ichimoku Cloud components
def calculate_ichimoku_cloud(df):
    # Tenkan-sen (Conversion Line) - 9 periods
    tenkan_sen = (df["high"].rolling_max(9) + df["low"].rolling_min(9)) / 2
    
    # Kijun-sen (Base Line) - 26 periods
    kijun_sen = (df["high"].rolling_max(26) + df["low"].rolling_min(26)) / 2
    
    # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2 shifted 26 periods ahead
    senkou_span_a = (tenkan_sen + kijun_sen) / 2
    senkou_span_a = senkou_span_a.shift(26)
    
    # Senkou Span B (Leading Span B) - 52 periods
    senkou_span_b = (df["high"].rolling_max(52) + df["low"].rolling_min(52)) / 2
    senkou_span_b = senkou_span_b.shift(26)
    
    # Chikou Span (Lagging Span) - 26 periods back
    chikou_span = df["close"].shift(-26)
    
    # Add all Ichimoku Cloud components to the DataFrame
    return df.with_columns(
        [
            tenkan_sen.alias("Tenkan_Sen"),
            kijun_sen.alias("Kijun_Sen"),
            senkou_span_a.alias("Senkou_Span_A"),
            senkou_span_b.alias("Senkou_Span_B"),
            chikou_span.alias("Chikou_Span")
        ]
    )

# Add Ichimoku Cloud components to the DataFrame
aapl = calculate_ichimoku_cloud(aapl)

# Bullish Engulfing
def bullish_engulfing(df):
    return (df["close"] > df["open"]) & (df["close"].shift(1) < df["open"].shift(1)) & (df["open"] < df["close"].shift(1))

# Bearish Engulfing
def bearish_engulfing(df):
    return (df["close"] < df["open"]) & (df["close"].shift(1) > df["open"].shift(1)) & (df["open"] > df["close"].shift(1))

# Doji
def doji(df, threshold=0.1):
    return (df["close"] - df["open"]).abs() / df["open"] < threshold

# Morning Star (Three Candle Pattern)
def morning_star(df):
    # Morning star pattern is identified by:
    # 1. A bearish candle
    # 2. A small body candle (indecision)
    # 3. A bullish candle that closes higher than the previous bullish candle
    return (df["close"].shift(2) < df["open"].shift(2)) & (df["close"].shift(1) > df["open"].shift(1)) & (df["close"] > df["open"].shift(1))

# Adding the candlestick patterns to the DataFrame
def add_candlestick_patterns(df):
    return df.with_columns(
        [
            bullish_engulfing(df).alias("Bullish_Engulfing"),
            bearish_engulfing(df).alias("Bearish_Engulfing"),
            doji(df).alias("Doji"),
            morning_star(df).alias("Morning_Star")
        ]
    )

# Add candlestick pattern columns to your DataFrame
aapl = add_candlestick_patterns(aapl)

# Calculate Pivot Points and Support/Resistance Levels
def calculate_pivot_points(df):
    # Calculate the Pivot Point (P)
    pivot_point = (df["high"] + df["low"] + df["close"]) / 3
    
    # Calculate the Support and Resistance levels
    s1 = 2 * pivot_point - df["high"]
    r1 = 2 * pivot_point - df["low"]
    
    s2 = pivot_point - (df["high"] - df["low"])
    r2 = pivot_point + (df["high"] - df["low"])
    
    s3 = df["low"] - 2 * (df["high"] - pivot_point)
    r3 = df["high"] + 2 * (pivot_point - df["low"])
    
    # Add the pivot points and levels to the DataFrame
    return df.with_columns(
        [
            pivot_point.alias("Pivot_Point"),
            s1.alias("S1"),
            r1.alias("R1"),
            s2.alias("S2"),
            r2.alias("R2"),
            s3.alias("S3"),
            r3.alias("R3")
        ]
    )

# Add Pivot Points and Levels to the DataFrame
aapl = calculate_pivot_points(aapl)

print(aapl)