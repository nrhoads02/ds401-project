import polars as pl
import math

def add_technical_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add technical indicators with windowed VWAP, windowed OBV, SMA/EMA ratio,
    Donchian Range, for all indicator columns (per asset).
    
    Optimized version that maintains all functionality while improving performance.
    Normalization removed and moved to a separate script.
    Intermediate calculation columns are properly cleaned up.
    
    Added:
    - Parkinson's volatility for windows 10, 30, 90
    - Fixed YZ volatility lag direction (forward-looking)
    """
    print("Calculating technical indicators...")

    # Windows used throughout calculations
    windows = [10, 30, 90]
    
    # Initial sorting
    df = df.sort(["date", "act_symbol"])
    
    # --- Step 1: Log Returns calculation ---
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("act_symbol"))
        .log()
        .alias("log_returns")
    )
    
    # --- Step 2: Basic trend indicators (SMA, EMA, STD) ---
    trend_cols = []
    for n in windows:
        trend_cols.extend([
            pl.col("close").rolling_mean(n).over("act_symbol").alias(f"SMA_{n}"),
            pl.col("close").ewm_mean(span=n, adjust=False).over("act_symbol").alias(f"EMA_{n}"),
            pl.col("close").rolling_std(n).over("act_symbol").alias(f"STD_{n}")
        ])
    df = df.with_columns(trend_cols)
    
    # --- Step 3: SMA/EMA Ratio ---
    ratio_cols = [
        (pl.col(f"SMA_{n}") / pl.col(f"EMA_{n}")).alias(f"SMA_EMA_ratio_{n}")
        for n in windows
    ]
    df = df.with_columns(ratio_cols)
    
    # --- Step 4: RSI calculation ---
    # Calculate price delta first
    df = df.with_columns(
        pl.col("close").diff().over("act_symbol").alias("price_delta")
    )
    
    # Then calculate RSI components and final RSI for each window
    for window in windows:
        # First prepare the gain and loss columns
        df = df.with_columns([
            pl.when(pl.col("price_delta") > 0).then(pl.col("price_delta")).otherwise(0).alias(f"gain_{window}"),
            pl.when(pl.col("price_delta") < 0).then(-pl.col("price_delta")).otherwise(0).alias(f"loss_{window}")
        ])
        
        # Calculate average gain and loss
        df = df.with_columns([
            pl.col(f"gain_{window}").rolling_mean(window).over("act_symbol").alias(f"avg_gain_{window}"),
            pl.col(f"loss_{window}").rolling_mean(window).over("act_symbol").alias(f"avg_loss_{window}")
        ])
        
        # Calculate RSI
        df = df.with_columns(
            (100 - 100 / (1 + pl.col(f"avg_gain_{window}") / pl.col(f"avg_loss_{window}"))).alias(f"RSI_{window}")
        )
    
    # --- Step 5: MACD of Historical Volatility ---
    # Calculate volatility components
    df = df.with_columns([
        pl.col("log_returns").rolling_std(12).over("act_symbol").alias("vol_12"),
        pl.col("log_returns").rolling_std(26).over("act_symbol").alias("vol_26")
    ])
    
    # Calculate MACD line, signal line, and histogram
    df = df.with_columns(
        (pl.col("vol_12") - pl.col("vol_26")).alias("vol_macd_line")
    )
    
    df = df.with_columns(
        pl.col("vol_macd_line")
        .ewm_mean(span=9, adjust=False).over("act_symbol")
        .alias("vol_signal_line")
    )
    
    df = df.with_columns(
        (pl.col("vol_macd_line") - pl.col("vol_signal_line")).alias("vol_macd_hist")
    )
    
    # --- Step 6: Typical Price (used by multiple indicators) ---
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
    )
    
    # --- Step 7: Chaikin Money Flow (CMF) ---
    # Calculate money flow multiplier
    df = df.with_columns(
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .alias("money_flow_multiplier")
    )
    
    # Calculate CMF for each window
    for window in windows:
        df = df.with_columns([
            (pl.col("money_flow_multiplier") * pl.col("volume")).rolling_sum(window).over("act_symbol")
            .alias(f"money_flow_volume_{window}"),
            pl.col("volume").rolling_sum(window).over("act_symbol")
            .alias(f"volume_sum_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"money_flow_volume_{window}") / pl.col(f"volume_sum_{window}"))
            .alias(f"CMF_{window}")
        )
    
    # --- Step 8: Commodity Channel Index (CCI) ---
    for window in windows:
        # Calculate typical price SMA
        df = df.with_columns(
            pl.col("typical_price").rolling_mean(window).over("act_symbol").alias(f"tp_sma_{window}")
        )
        
        # Calculate deviation and mean absolute deviation
        df = df.with_columns(
            (pl.col("typical_price") - pl.col(f"tp_sma_{window}")).alias(f"tp_dev_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"tp_dev_{window}").abs().rolling_mean(window).over("act_symbol").alias(f"tp_mad_{window}")
        )
        
        # Calculate CCI
        df = df.with_columns(
            (pl.col(f"tp_dev_{window}") / (0.015 * pl.col(f"tp_mad_{window}"))).alias(f"CCI_{window}")
        )
    
    # --- Step 9: Average True Range (ATR) ---
    # Calculate previous close (needed for each window separately to avoid conflicts)
    for window in windows:
        df = df.with_columns(
            pl.col("close").shift(1).over("act_symbol").alias(f"prev_close_{window}")
        )
        
        # Calculate true range
        df = df.with_columns(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col(f"prev_close_{window}")).abs(),
                (pl.col("low") - pl.col(f"prev_close_{window}")).abs()
            ).alias(f"tr_{window}")
        )
        
        # Calculate ATR
        df = df.with_columns(
            pl.col(f"tr_{window}").rolling_mean(window).over("act_symbol").alias(f"ATR_{window}")
        )
    
    # --- Step 10: Parkinson's Volatility (Added) ---
    # Constant for Parkinson's Volatility calculation
    parkinson_constant = 1.0 / (4.0 * math.log(2.0))
    
    # Calculate squared log range for each day
    df = df.with_columns(
        ((pl.col("high") / pl.col("low")).log().pow(2) * parkinson_constant)
        .alias("parkinson_daily")
    )
    
    # Calculate Parkinson's Volatility for each window
    for window in windows:
        df = df.with_columns(
            (pl.col("parkinson_daily").rolling_mean(window).over("act_symbol") * window)
            .sqrt()
            .alias(f"ParkinsonVol_{window}")
        )
    
    # --- Step 11: Statistical Measures (Skewness, Kurtosis) ---
    for window in windows:
        # Calculate mean and std
        df = df.with_columns([
            pl.col("log_returns").rolling_mean(window).over("act_symbol").alias(f"mean_{window}"),
            pl.col("log_returns").rolling_std(window).over("act_symbol").alias(f"std_{window}")
        ])
        
        # Calculate deviation
        df = df.with_columns(
            (pl.col("log_returns") - pl.col(f"mean_{window}")).alias(f"deviation_{window}")
        )
        
        # --- Skewness ---
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(3).alias(f"dev_cubed_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_cubed_{window}").rolling_mean(window).over("act_symbol").alias(f"mean_dev_cubed_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"mean_dev_cubed_{window}") / pl.col(f"std_{window}").pow(3))
            .alias(f"Skewness_{window}")
        )
        
        # --- Kurtosis ---
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(4).alias(f"dev_fourth_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_fourth_{window}").rolling_mean(window).over("act_symbol").alias(f"mean_dev_fourth_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"mean_dev_fourth_{window}") / pl.col(f"std_{window}").pow(4) - 3)
            .alias(f"Kurtosis_{window}")
        )
    
    # --- Step 12: Quantile Spreads ---
    for window in windows:
        # Calculate quantiles
        df = df.with_columns([
            pl.col("log_returns").rolling_quantile(0.95, window_size=window, interpolation='linear')
            .over("act_symbol").alias(f"q95_{window}"),
            pl.col("log_returns").rolling_quantile(0.05, window_size=window, interpolation='linear')
            .over("act_symbol").alias(f"q05_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"q95_{window}") - pl.col(f"q05_{window}")).alias(f"QuantileSpread_{window}_Extreme")
        )
        
        df = df.with_columns([
            pl.col("log_returns").rolling_quantile(0.75, window_size=window, interpolation='linear')
            .over("act_symbol").alias(f"q75_{window}"),
            pl.col("log_returns").rolling_quantile(0.25, window_size=window, interpolation='linear')
            .over("act_symbol").alias(f"q25_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"q75_{window}") - pl.col(f"q25_{window}")).alias(f"QuantileSpread_{window}_IQR")
        )
    
    # --- Step 13: Donchian Range ---
    for window in windows:
        df = df.with_columns([
            pl.col("high").rolling_max(window).over("act_symbol").alias(f"high_max_{window}"),
            pl.col("low").rolling_min(window).over("act_symbol").alias(f"low_min_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"high_max_{window}") - pl.col(f"low_min_{window}"))
            .alias(f"DonchianRange_{window}")
        )
    
    # --- Step 14: OBV and Windowed OBV ---
    # Previous close specifically for OBV
    df = df.with_columns(
        pl.col("close").shift(1).over("act_symbol").alias("prev_close_obv")
    )
    
    # Calculate volume direction
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("prev_close_obv"))
          .then(pl.col("volume"))
          .when(pl.col("close") < pl.col("prev_close_obv"))
          .then(-pl.col("volume"))
          .otherwise(0)
          .alias("volume_direction")
    )
    
    # Calculate OBV
    df = df.with_columns(
        pl.col("volume_direction").cum_sum().over("act_symbol").alias("OBV")
    )
    
    # Calculate OBV change for each window
    for window in windows:
        df = df.with_columns(
            (pl.col("OBV") - pl.col("OBV").shift(window).over("act_symbol"))
            .alias(f"OBV_change_{window}")
        )
    
    # --- Step 15: Windowed VWAP and VWAP Deviation ---
    for window in windows:
        df = df.with_columns([
            ((pl.col("typical_price") * pl.col("volume"))
              .rolling_sum(window).over("act_symbol")).alias(f"tp_vol_sum_{window}"),
            pl.col("volume").rolling_sum(window).over("act_symbol")
              .alias(f"vwap_volume_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"tp_vol_sum_{window}") / pl.col(f"vwap_volume_{window}"))
            .alias(f"VWAP_{window}")
        )
        
        df = df.with_columns(
            (pl.col("close") - pl.col(f"VWAP_{window}"))
            .alias(f"VWAP_deviation_{window}")
        )
    
    # --- Step 16: Yang-Zhang Volatility ---
    for window in windows:
        k = 0.34 / (1 + (window + 1) / (window - 1))
        
        # Calculate open-close and close-open log returns
        df = df.with_columns([
            (pl.col("close") / pl.col("open").shift(1).over("act_symbol")).log()
            .alias(f"open_close_{window}"),
            (pl.col("open") / pl.col("close").shift(1).over("act_symbol")).log()
            .alias(f"close_open_{window}")
        ])
        
        # Calculate variances
        df = df.with_columns([
            pl.col(f"open_close_{window}").rolling_var(window).over("act_symbol")
            .alias(f"vol_oc_{window}"),
            pl.col(f"close_open_{window}").rolling_var(window).over("act_symbol")
            .alias(f"vol_co_{window}"),
            pl.col("log_returns").rolling_var(window).over("act_symbol")
            .alias(f"vol_cc_{window}")
        ])
        
        # Calculate Yang-Zhang volatility
        df = df.with_columns(
            ((k * pl.col(f"vol_oc_{window}")) + 
             ((1 - k) * pl.col(f"vol_co_{window}")) + 
             pl.col(f"vol_cc_{window}")).sqrt()
            .alias(f"YZVol_{window}")
        )
        
        # Calculate forward-looking version (reversed from original) - using negative shift
        # This will give the value from "window" days in the future
        df = df.with_columns(
            pl.col(f"YZVol_{window}").shift(-window).over("act_symbol")
            .alias(f"YZVol_{window}_future")
        )
    
    # --- Step 17: Clean up intermediate calculation columns ---
    # Define columns to keep (base columns + final indicator columns)
    base_cols = ["act_symbol", "date", "open", "high", "low", "close", "volume", "log_returns"]
    
    # Final indicator columns (only keep the actual indicators, not intermediate calculations)
    indicator_cols = [
        "OBV", "vol_macd_line", "vol_signal_line", "vol_macd_hist", "vol_12", "vol_26"
    ]
    
    # Add window-specific indicators
    for window in windows:
        indicator_cols.extend([
            f"SMA_{window}", f"EMA_{window}", f"STD_{window}", f"SMA_EMA_ratio_{window}",
            f"RSI_{window}", f"ATR_{window}", f"CMF_{window}", f"CCI_{window}",
            f"DonchianRange_{window}", f"OBV_change_{window}", f"VWAP_{window}",
            f"VWAP_deviation_{window}", f"Skewness_{window}", f"Kurtosis_{window}",
            f"QuantileSpread_{window}_Extreme", f"QuantileSpread_{window}_IQR",
            f"YZVol_{window}", f"YZVol_{window}_future", f"ParkinsonVol_{window}"
        ])
    
    cols_to_keep = base_cols + indicator_cols
    
    # Select only the specified columns
    df = df.select(cols_to_keep)
    
    return df

if __name__ == "__main__":
    print("Loading OHLCV data...")
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv").with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    print("Starting indicator calculations...")
    try:
        ohlcv_with_indicators = (
            ohlcv.lazy()
            .pipe(add_technical_indicators)
            .collect()
        )
        
        print(f"Success! Data size: {ohlcv_with_indicators.estimated_size('mb')} MB")
        print(f"Number of columns: {len(ohlcv_with_indicators.columns)}")
        print(ohlcv_with_indicators.columns)
        
        # Save the processed data
        # ohlcv_with_indicators.write_csv("data/processed/ohlcv_with_indicators.csv")
        # print("Saved processed data to data/processed/ohlcv_with_indicators.csv")
        
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")