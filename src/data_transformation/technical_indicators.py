import polars as pl

def add_technical_indicators(df: pl.LazyFrame) -> pl.LazyFrame:
    """Add technical indicators using lazy operations."""
    df = df.sort(["date", "act_symbol"])
    
    # Log Returns and volatility features
    df = df.with_columns(
        (pl.col("close") / pl.col("close").shift(1).over("act_symbol")).log().alias("log_returns")
    ).with_columns(
        pl.col("log_returns").pow(2).alias("log_returns_sq"),
        pl.col("log_returns").shift(1).over("act_symbol").alias("log_returns_lag1"),
        pl.col("log_returns").shift(1).over("act_symbol").pow(2).alias("log_returns_lag1_sq")
    )
    
    # Trend Indicators
    for n in [3, 5, 10, 20, 50]:
        df = df.with_columns(
            pl.col("close").rolling_mean(window_size=n).over("act_symbol").alias(f"SMA_{n}"),
            pl.col("close").ewm_mean(span=n, adjust=False).over("act_symbol").alias(f"EMA_{n}")
        )
    
    # RSI (Relative Strength Index)
    df = df.with_columns(
        pl.col("close").diff(1).over("act_symbol").alias("delta")
    )
    
    # For each RSI window, compute gains, losses and then the RSI.
    for window in [7, 9, 14, 21, 30]:
        gain = pl.when(pl.col("delta") > 0).then(pl.col("delta")).otherwise(0)
        loss = pl.when(pl.col("delta") < 0).then(-pl.col("delta")).otherwise(0)
        
        avg_gain = gain.rolling_mean(window).over("act_symbol")
        avg_loss = loss.rolling_mean(window).over("act_symbol")
        
        rsi = (100 - 100 / (1 + avg_gain / avg_loss)).alias(f"RSI_{window}")
        
        df = df.with_columns(rsi)
    
    df = df.drop("delta")
    
    # MACD (Fixed implementation)
    df = df.with_columns(
        fast_ema=pl.col("close").ewm_mean(span=12, adjust=False).over("act_symbol"),
        slow_ema=pl.col("close").ewm_mean(span=26, adjust=False).over("act_symbol")
    ).with_columns(
        macd_line=pl.col("fast_ema") - pl.col("slow_ema")
    ).with_columns(
        signal_line=pl.col("macd_line").ewm_mean(span=9, adjust=False).over("act_symbol")
    ).with_columns(
        macd_histogram=pl.col("macd_line") - pl.col("signal_line")
    ).drop(["fast_ema", "slow_ema"])
    
    # Volatility Indicators
    def _bollinger(window: int) -> list[pl.Expr]:
        sma = pl.col("close").rolling_mean(window).over("act_symbol")
        std = pl.col("close").rolling_std(window).over("act_symbol")
        return [
            sma.alias(f"BB_Middle_{window}"),
            (sma + 2 * std).alias(f"BB_Upper_{window}"),
            (sma - 2 * std).alias(f"BB_Lower_{window}")
        ]
    
    for window in [20, 50]:
        df = df.with_columns(_bollinger(window))
    
    # Volume Indicators
    def _cmf(window: int) -> pl.Expr:
        mfm = (pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))
        return (mfm * pl.col("volume")).rolling_sum(window).over("act_symbol") / pl.col("volume").rolling_sum(window).over("act_symbol")
    
    for window in [10, 20, 50, 100]:
        df = df.with_columns(_cmf(window).alias(f"CMF_{window}"))
    
    # Stochastic Oscillator
    for window in [9, 14, 21]:
        df = df.with_columns(
            low_min=pl.col("low").rolling_min(window).over("act_symbol"),
            high_max=pl.col("high").rolling_max(window).over("act_symbol")
        ).with_columns(
            k=100 * (pl.col("close") - pl.col("low_min")) / (pl.col("high_max") - pl.col("low_min"))
        ).with_columns(
            d=pl.col("k").rolling_mean(3).over("act_symbol")
        ).rename({
            "k": f"Stochastic_%K_{window}",
            "d": f"Stochastic_%D_{window}"
        }).drop(["low_min", "high_max"])
    
    # Ichimoku Cloud
    df = df.with_columns(
        tenkan_high=pl.col("high").rolling_max(9).over("act_symbol"),
        tenkan_low=pl.col("low").rolling_min(9).over("act_symbol"),
        kijun_high=pl.col("high").rolling_max(26).over("act_symbol"),
        kijun_low=pl.col("low").rolling_min(26).over("act_symbol"),
        senkou_high=pl.col("high").rolling_max(52).over("act_symbol"),
        senkou_low=pl.col("low").rolling_min(52).over("act_symbol")
    ).with_columns(
        tenkan=(pl.col("tenkan_high") + pl.col("tenkan_low")) / 2,
        kijun=(pl.col("kijun_high") + pl.col("kijun_low")) / 2,
        senkou_b=(pl.col("senkou_high") + pl.col("senkou_low")) / 2
    ).with_columns(
        senkou_a=((pl.col("tenkan") + pl.col("kijun")) / 2).shift(26).over("act_symbol"),
        senkou_b=pl.col("senkou_b").shift(26).over("act_symbol"),
        chikou=pl.col("close").shift(-26).over("act_symbol")
    ).drop([
        "tenkan_high", "tenkan_low", 
        "kijun_high", "kijun_low",
        "senkou_high", "senkou_low"
    ]).rename({
        "tenkan": "Tenkan_Sen",
        "kijun": "Kijun_Sen",
        "senkou_a": "Senkou_Span_A",
        "senkou_b": "Senkou_Span_B",
        "chikou": "Chikou_Span"
    })
    
    return df

if __name__ == "__main__":
    # Read the OHLCV CSV and parse the 'date' column
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
    ohlcv = ohlcv.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # Add technical indicators to the DataFrame
    ohlcv_with_indicators = (
        ohlcv.lazy()
        .pipe(add_technical_indicators)
        .collect()
    )

    print(ohlcv_with_indicators.estimated_size("mb"))
    
    # Print the resulting DataFrame
    print(ohlcv_with_indicators)