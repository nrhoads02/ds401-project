import polars as pl

def add_technical_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add volatility-focused technical indicators using lazy operations."""
    
    # Initial sorting
    df = df.sort(["date", "act_symbol"])
    
    # Base returns calculation
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("act_symbol"))
        .log()
        .alias("log_returns")
    )
    
    # Trend Indicators (SMA/EMA)
    sma_ema_columns = []
    for n in [3, 5, 10, 20, 50]:
        sma_ema_columns.extend([
            pl.col("close").rolling_mean(n).over("act_symbol").alias(f"SMA_{n}"),
            pl.col("close").ewm_mean(span=n, adjust=False).over("act_symbol").alias(f"EMA_{n}")
        ])
    df = df.with_columns(sma_ema_columns)
    
    # Momentum Indicators
    # RSI (Relative Strength Index)
    df = df.with_columns(
        pl.col("close").diff(1).over("act_symbol").alias("delta")
    )
    
    # For each RSI window, compute gains, losses and then the RSI
    rsi_columns = []
    for window in [7, 9, 14, 21, 30]:
        gain = pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0)
        loss = pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0)
        
        avg_gain = gain.rolling_mean(window).over("act_symbol")
        avg_loss = loss.rolling_mean(window).over("act_symbol")
        
        rsi_columns.append((100 - 100 / (1 + avg_gain / avg_loss)).alias(f"RSI_{window}"))
    
    df = df.with_columns(rsi_columns).drop("delta")
    
    # MACD components - FIXED
    df = df.with_columns([
        pl.col("close").ewm_mean(span=12, adjust=False).over("act_symbol").alias("fast_ema"),
        pl.col("close").ewm_mean(span=26, adjust=False).over("act_symbol").alias("slow_ema")
    ])
    
    df = df.with_columns(
        (pl.col("fast_ema") - pl.col("slow_ema")).alias("macd_line")
    )
    
    df = df.with_columns(
        pl.col("macd_line").ewm_mean(span=9, adjust=False).over("act_symbol").alias("signal_line")
    )
    
    df = df.with_columns(
        (pl.col("macd_line") - pl.col("signal_line")).alias("macd_histogram")
    )
    
    df = df.drop(["fast_ema", "slow_ema"])
    
    # Volatility Metrics
    # Bollinger Components (SMA, Std, Width)
    bollinger_columns = []
    for window in [20, 50]:
        sma = pl.col("close").rolling_mean(window).over("act_symbol")
        std = pl.col("close").rolling_std(window).over("act_symbol")
        bollinger_columns.extend([
            sma.alias(f"BB_SMA_{window}"),
            std.alias(f"BB_STD_{window}"),
            (4 * std).alias(f"BB_WIDTH_{window}")
        ])
    df = df.with_columns(bollinger_columns)
    
    # Chaikin Money Flow
    df = df.with_columns(
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .alias("money_flow_multiplier")
    )

    for window in [10, 20, 50, 100]:
        df = df.with_columns([
            (pl.col("money_flow_multiplier") * pl.col("volume")).rolling_sum(window).over("act_symbol")
            .alias(f"money_flow_volume_{window}"),
            pl.col("volume").rolling_sum(window).over("act_symbol")
            .alias(f"total_volume_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"money_flow_volume_{window}") / pl.col(f"total_volume_{window}"))
            .alias(f"CMF_{window}")
        )
        
        df = df.drop([f"money_flow_volume_{window}", f"total_volume_{window}"])
    
    df = df.drop("money_flow_multiplier")
    
    # Commodity Channel Index (CCI)
    for window in [14, 20]:
        df = df.with_columns(
            ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias(f"tp_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"tp_{window}").rolling_mean(window).over("act_symbol").alias(f"sma_tp_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"tp_{window}") - pl.col(f"sma_tp_{window}")).abs().alias(f"dev_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_{window}").rolling_mean(window).over("act_symbol").alias(f"mad_{window}")
        )
        
        df = df.with_columns(
            ((pl.col(f"tp_{window}") - pl.col(f"sma_tp_{window}")) / (0.015 * pl.col(f"mad_{window}"))).alias(f"CCI_{window}")
        )
        
        df = df.drop([f"tp_{window}", f"sma_tp_{window}", f"dev_{window}", f"mad_{window}"])
    
    # Average True Range
    for window in [14, 20]:
        df = df.with_columns(
            pl.col("close").shift(1).over("act_symbol").alias(f"prev_close_{window}")
        )
        
        df = df.with_columns(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col(f"prev_close_{window}")).abs(),
                (pl.col("low") - pl.col(f"prev_close_{window}")).abs()
            ).alias(f"tr_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"tr_{window}").rolling_mean(window).over("act_symbol").alias(f"ATR_{window}")
        )
        
        df = df.drop([f"prev_close_{window}", f"tr_{window}"])
    
    # Statistical Measures 
    for window in [20, 50]:
        df = df.with_columns([
            pl.col("log_returns").rolling_mean(window).over("act_symbol").alias(f"mean_{window}"),
            pl.col("log_returns").rolling_std(window).over("act_symbol").alias(f"std_{window}")
        ])
        
        df = df.with_columns(
            (pl.col("log_returns") - pl.col(f"mean_{window}")).alias(f"deviation_{window}")
        )
        
        # Skewness calculation
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(3).alias(f"dev_cubed_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_cubed_{window}").rolling_mean(window).over("act_symbol").alias(f"mean_dev_cubed_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"mean_dev_cubed_{window}") / pl.col(f"std_{window}").pow(3)).alias(f"Skewness_{window}")
        )
        
        # Kurtosis calculation
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(4).alias(f"dev_fourth_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_fourth_{window}").rolling_mean(window).over("act_symbol").alias(f"mean_dev_fourth_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"mean_dev_fourth_{window}") / pl.col(f"std_{window}").pow(4) - 3).alias(f"Kurtosis_{window}")
        )
        
        # Clean up temp columns
        df = df.drop([
            f"mean_{window}", f"std_{window}", f"deviation_{window}", 
            f"dev_cubed_{window}", f"mean_dev_cubed_{window}",
            f"dev_fourth_{window}", f"mean_dev_fourth_{window}"
        ])
    
    # Quantile Spreads
    for window in [20, 50]:
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
        
        df = df.drop([f"q95_{window}", f"q05_{window}", f"q75_{window}", f"q25_{window}"])
    
    # Volume-based Indicators
    # OBV
    df = df.with_columns(
        pl.col("close").shift(1).over("act_symbol").alias("prev_close")
    )
    
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("prev_close"))
        .then(pl.col("volume"))
        .when(pl.col("close") < pl.col("prev_close"))
        .then(-pl.col("volume"))
        .otherwise(0)
        .alias("volume_direction")
    )
    
    df = df.with_columns(
        pl.col("volume_direction").cum_sum().over("act_symbol").alias("OBV")
    )
    
    df = df.drop(["prev_close", "volume_direction"])
    
    # VWAP Deviation
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3 * pl.col("volume"))
        .alias("vwap_numerator")
    )
    
    df = df.with_columns([
        pl.col("vwap_numerator").cum_sum().over("act_symbol").alias("cum_vwap_numerator"),
        pl.col("volume").cum_sum().over("act_symbol").alias("cum_volume")
    ])
    
    df = df.with_columns(
        (pl.col("cum_vwap_numerator") / pl.col("cum_volume")).alias("VWAP")
    )
    
    df = df.with_columns(
        (pl.col("close") - pl.col("VWAP")).alias("VWAP_Deviation")
    )
    
    df = df.drop(["vwap_numerator", "cum_vwap_numerator", "cum_volume", "VWAP"])
    
    # Volume Percentile
    for window in [20, 50]:
        df = df.with_columns(
            pl.col("volume").rolling_quantile(0.5, window_size=window).over("act_symbol")
            .alias(f"vol_med_{window}")
        )
        
        df = df.with_columns(
            (pl.col("volume") > pl.col(f"vol_med_{window}")).alias(f"vol_above_med_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"vol_above_med_{window}").cast(pl.Float64).alias(f"vol_ind_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"vol_ind_{window}").rolling_mean(window).over("act_symbol")
            .mul(100).alias(f"VolumePercentile_{window}")
        )
        
        df = df.drop([f"vol_med_{window}", f"vol_above_med_{window}", f"vol_ind_{window}"])
    
    # Range-based Indicators
    # Donchian Range
    for window in [20, 50]:
        df = df.with_columns([
            pl.col("high").rolling_max(window).over("act_symbol").alias(f"high_max_{window}"),
            pl.col("low").rolling_min(window).over("act_symbol").alias(f"low_min_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"high_max_{window}") - pl.col(f"low_min_{window}")).alias(f"DonchianRange_{window}")
        )
        
        df = df.drop([f"high_max_{window}", f"low_min_{window}"])
    
    # Parkinson Volatility
    for window in [21, 26, 30, 50]:
        df = df.with_columns(
            (pl.col("high") / pl.col("low")).log().pow(2).alias(f"hl_ratio_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"hl_ratio_{window}").rolling_mean(window).over("act_symbol").alias(f"mean_hl_ratio_{window}")
        )
        
        df = df.with_columns(
            (pl.col(f"mean_hl_ratio_{window}") / (4 * pl.lit(2).log())).sqrt()
            .alias(f"ParkinsonVol_{window}")
        )
        
        df = df.drop([f"hl_ratio_{window}", f"mean_hl_ratio_{window}"])
    
    # Yang-Zhang Volatility
    for window in [21, 26, 30, 50]:
        k = 0.34 / (1 + (window + 1) / (window - 1))
        
        df = df.with_columns([
            (pl.col("close") / pl.col("open").shift(1).over("act_symbol")).log()
            .alias(f"open_close_{window}"),
            (pl.col("open") / pl.col("close").shift(1).over("act_symbol")).log()
            .alias(f"close_open_{window}")
        ])
        
        df = df.with_columns([
            pl.col(f"open_close_{window}").rolling_var(window).over("act_symbol")
            .alias(f"vol_oc_{window}"),
            pl.col(f"close_open_{window}").rolling_var(window).over("act_symbol")
            .alias(f"vol_co_{window}"),
            pl.col("log_returns").rolling_var(window).over("act_symbol")
            .alias(f"vol_cc_{window}")
        ])
        
        df = df.with_columns(
            ((k * pl.col(f"vol_oc_{window}")) + 
             ((1 - k) * pl.col(f"vol_co_{window}")) + 
             pl.col(f"vol_cc_{window}")).sqrt()
            .alias(f"YZVol_{window}")
        )
        
        df = df.drop([
            f"open_close_{window}", f"close_open_{window}", 
            f"vol_oc_{window}", f"vol_co_{window}", f"vol_cc_{window}"
        ])
    
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
        print(ohlcv_with_indicators.columns)
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}")