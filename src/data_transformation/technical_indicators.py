import polars as pl
import math
import numpy as np  # Needed for array operations
from scipy import interpolate  # Add this import for cubic spline

def add_technical_indicators(df: pl.LazyFrame, calculate_future_cols: bool = True) -> pl.LazyFrame:
    """Add technical indicators with enhanced stochastic volatility modeling capabilities.
    
    Parameters:
    -----------
    df : pl.LazyFrame
        Input data frame with OHLCV data
    calculate_future_cols : bool, default=True
        Whether to calculate forward-looking (future) indicators which are used as targets
        Set to False to save time and memory when only using features for inference
    
    Returns:
    --------
    pl.LazyFrame: DataFrame with technical indicators added
    """
    print("Calculating technical indicators...")

    # Define multiple window sizes for different calculations
    heston_windows = [10, 15, 20, 25, 30, 35]  # Windows for Heston model params
    vol_windows = [10, 20, 35]  # For most volatility metrics 
    core_windows = [10, 20, 35]  # For core technical indicators (reduced from previous)
    
    # Initial sorting
    df = df.sort(["date", "act_symbol"])
    
    # --- Step 1: Log Returns calculation ---
    # Improved log returns calculation with proper checks for division by zero
    df = df.with_columns(
        pl.when((pl.col("close").shift(1).over("act_symbol") > 0) & (pl.col("close") > 0))
        .then((pl.col("close") / pl.col("close").shift(1).over("act_symbol")).log())
        .otherwise(None)
        .alias("log_returns")
    )

    # --- Step 1.1: Split positive and negative returns (needed for volatility skew) ---
    df = df.with_columns([
        pl.when(pl.col("log_returns") > 0)
            .then(pl.col("log_returns"))
            .otherwise(0)
            .alias("pos_returns"),
            
        pl.when(pl.col("log_returns") < 0)
            .then(-pl.col("log_returns"))  # Make positive for easier calculations
            .otherwise(0)
            .alias("neg_returns")
    ])
    
    # --- Step 2: Basic trend indicators (SMA, EMA, STD) ---
    trend_cols = []
    for n in vol_windows:
        trend_cols.extend([
            pl.col("close").rolling_mean(n).over("act_symbol").alias(f"SMA_{n}"),
            pl.col("close").ewm_mean(span=n, adjust=False).over("act_symbol").alias(f"EMA_{n}"),
            pl.col("close").rolling_std(n).over("act_symbol").alias(f"STD_{n}")
        ])
    df = df.with_columns(trend_cols)
    
    # --- Step 3: SMA/EMA Ratio ---
    ratio_cols = [
        (pl.when(pl.col(f"EMA_{n}") > 0.00001)
         .then(pl.col(f"SMA_{n}") / pl.col(f"EMA_{n}"))
         .otherwise(1.0))
        .alias(f"SMA_EMA_ratio_{n}")
        for n in core_windows
    ]
    df = df.with_columns(ratio_cols)
    
    # --- Step 4: RSI calculation ---
    # Calculate price delta first
    df = df.with_columns(
        pl.col("close").diff().over("act_symbol").alias("price_delta")
    )
    
    # Then calculate RSI components and final RSI for core windows
    for window in core_windows:
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
        
        # Calculate RSI with improved handling for division by zero
        df = df.with_columns(
            pl.when(pl.col(f"avg_loss_{window}") > 0.00001)
            .then(100 - 100 / (1 + pl.col(f"avg_gain_{window}") / pl.col(f"avg_loss_{window}")))
            .otherwise(pl.when(pl.col(f"avg_gain_{window}") > 0).then(100.0).otherwise(50.0))
            .clip(0, 100)  # Ensure RSI is within valid range
            .alias(f"RSI_{window}")
        )
    
    # --- Step 5: Typical Price (used by multiple indicators) ---
    df = df.with_columns(
        ((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price")
    )
    
    # --- Step 6: Chaikin Money Flow (CMF) ---
    # Calculate money flow multiplier with checks for division by zero
    df = df.with_columns(
        pl.when(pl.col("high") - pl.col("low") > 0.00001)
        .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
        .otherwise(0.5)  # Default to middle value for flat price periods
        .alias("money_flow_multiplier")
    )
    
    # Calculate CMF for core windows
    for window in core_windows:
        df = df.with_columns([
            (pl.col("money_flow_multiplier") * pl.col("volume")).rolling_sum(window).over("act_symbol")
            .alias(f"money_flow_volume_{window}"),
            pl.col("volume").rolling_sum(window).over("act_symbol")
            .alias(f"volume_sum_{window}")
        ])
        
        # Add check for division by zero
        df = df.with_columns(
            pl.when(pl.col(f"volume_sum_{window}") > 0.00001)
            .then(pl.col(f"money_flow_volume_{window}") / pl.col(f"volume_sum_{window}"))
            .otherwise(0.5)  # Default to neutral value
            .clip(-1, 1)  # Clip to normal range
            .alias(f"CMF_{window}")
        )
    
    # --- Step 7: Commodity Channel Index (CCI) ---
    for window in core_windows:
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
        
        # Calculate CCI with check for division by zero
        df = df.with_columns(
            pl.when(pl.col(f"tp_mad_{window}") > 0.00001)
            .then((pl.col(f"tp_dev_{window}") / (0.015 * pl.col(f"tp_mad_{window}"))).clip(-666.666667, 666.666667))
            .otherwise(0.0)  # Default to neutral value
            .alias(f"CCI_{window}")
        )
    
    # --- Step 8: Average True Range (ATR) ---
    # Calculate previous close (needed for each window separately to avoid conflicts)
    for window in vol_windows:
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
    
    # --- Step 9: Parkinson's Volatility ---
    # Constant for Parkinson's Volatility calculation
    parkinson_constant = 1.0 / (4.0 * math.log(2.0))
    
    # Calculate squared log range for each day with handling for potential edge cases
    df = df.with_columns(
        pl.when((pl.col("high") > 0) & (pl.col("low") > 0) & (pl.col("high") > pl.col("low")))
        .then((pl.col("high") / pl.col("low")).log().pow(2) * parkinson_constant)
        .otherwise(0.0)
        .alias("parkinson_daily")
    )
    
    # Calculate Parkinson's Volatility for each window
    for window in vol_windows:
        df = df.with_columns(
            pl.when(pl.col("parkinson_daily").rolling_mean(window).over("act_symbol") > 0)
            .then((pl.col("parkinson_daily").rolling_mean(window).over("act_symbol") * window).sqrt())
            .otherwise(
                # Fallback to standard deviation when Parkinson can't be calculated
                pl.col("log_returns").rolling_std(window, min_periods=max(5, window//5)).over("act_symbol")
            )
            .clip(0, 1.0)  # Reasonable bounds for volatility 
            .alias(f"ParkinsonVol_{window}")
        )
    
    # --- Step 10: Statistical Measures (Skewness, Kurtosis) --- 
    # Using heston_windows instead of vol_windows for consistency
    for window in heston_windows:
        # Calculate mean and std with minimum observations
        df = df.with_columns([
            pl.col("log_returns")
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"mean_{window}"),
            
            pl.col("log_returns")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"std_{window}")
        ])
        
        # Calculate deviation with finite value checking
        df = df.with_columns(
            pl.when(pl.col(f"mean_{window}").is_finite() & (pl.col(f"std_{window}") > 0.00001))
            .then(pl.col("log_returns") - pl.col(f"mean_{window}"))
            .otherwise(0.0)  # Use 0 instead of None to avoid propagating nulls
            .alias(f"deviation_{window}")
        )
        
        # --- Skewness with proper safeguards ---
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(3).alias(f"dev_cubed_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_cubed_{window}")
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"mean_dev_cubed_{window}")
        )
        
        df = df.with_columns(
            pl.when(
                (pl.col(f"std_{window}") > 0.00001) &
                pl.col(f"mean_dev_cubed_{window}").is_finite() &
                pl.col(f"std_{window}").is_finite()
            )
            .then((pl.col(f"mean_dev_cubed_{window}") / pl.col(f"std_{window}").pow(3)).clip(-10, 10)) # Clip extreme skewness
            .otherwise(0)
            .alias(f"Skewness_{window}")
        )
        
        # --- Kurtosis with proper safeguards ---
        df = df.with_columns(
            pl.col(f"deviation_{window}").pow(4).alias(f"dev_fourth_{window}")
        )
        
        df = df.with_columns(
            pl.col(f"dev_fourth_{window}")
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"mean_dev_fourth_{window}")
        )
        
        df = df.with_columns(
            pl.when(
                (pl.col(f"std_{window}") > 0.00001) &
                pl.col(f"mean_dev_fourth_{window}").is_finite() &
                pl.col(f"std_{window}").is_finite()
            )
            .then(((pl.col(f"mean_dev_fourth_{window}") / pl.col(f"std_{window}").pow(4)) - 3).clip(-5, 20))
            .otherwise(0)
            .alias(f"Kurtosis_{window}")
        )
    
    # --- Step 11: Quantile Spreads ---
    # Calculate for both vol_windows and heston_windows to ensure we have data for curvature
    for window in heston_windows:
        # Calculate quantiles with minimum periods
        df = df.with_columns(
            pl.col("log_returns").rolling_quantile(quantile=0.95, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q95_{window}")
        )
        
        df = df.with_columns(
            pl.col("log_returns").rolling_quantile(quantile=0.05, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q05_{window}")
        )
        
        # Calculate extreme spread
        df = df.with_columns(
            (pl.col(f"q95_{window}") - pl.col(f"q05_{window}")).clip(0, 0.5)  # Clip to reasonable volatility range
            .alias(f"QuantileSpread_{window}_Extreme")
        )
        
        # Then the 75th and 25th percentiles
        df = df.with_columns(
            pl.col("log_returns").rolling_quantile(quantile=0.75, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q75_{window}")
        )
        
        df = df.with_columns(
            pl.col("log_returns").rolling_quantile(quantile=0.25, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q25_{window}")
        )
        
        # Calculate IQR spread
        df = df.with_columns(
            (pl.col(f"q75_{window}") - pl.col(f"q25_{window}")).clip(0.00001, 0.25)  # Ensure non-zero and reasonable
            .alias(f"QuantileSpread_{window}_IQR")
        )
    
    # --- Step 12: Donchian Range ---
    for window in vol_windows:
        df = df.with_columns([
            pl.col("high").rolling_max(window).over("act_symbol").alias(f"high_max_{window}"),
            pl.col("low").rolling_min(window).over("act_symbol").alias(f"low_min_{window}")
        ])
        
        df = df.with_columns(
            (pl.col(f"high_max_{window}") - pl.col(f"low_min_{window}"))
            .alias(f"DonchianRange_{window}")
        )
    
    # --- Step 13: OBV and Windowed OBV ---
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
    
    # Calculate OBV change for core windows only
    for window in core_windows:
        df = df.with_columns(
            (pl.col("OBV") - pl.col("OBV").shift(window).over("act_symbol"))
            .alias(f"OBV_change_{window}")
        )
    
    # --- Step 14: Windowed VWAP and VWAP Deviation ---
    for window in core_windows:
        df = df.with_columns([
            ((pl.col("typical_price") * pl.col("volume"))
              .rolling_sum(window).over("act_symbol")).alias(f"tp_vol_sum_{window}"),
            pl.col("volume").rolling_sum(window).over("act_symbol")
              .alias(f"vwap_volume_{window}")
        ])
        
        df = df.with_columns(
            # Add check for division by zero
            pl.when(pl.col(f"vwap_volume_{window}") > 0.00001)
            .then(pl.col(f"tp_vol_sum_{window}") / pl.col(f"vwap_volume_{window}"))
            .otherwise(pl.col("typical_price"))  # Default to typical price when volume is zero
            .alias(f"VWAP_{window}")
        )
        
        df = df.with_columns(
            (pl.col("close") - pl.col(f"VWAP_{window}")).clip(-1000, 1000)  # Clip extreme values
            .alias(f"VWAP_deviation_{window}")
        )
    
    # --- Step 15: Yang-Zhang Volatility (Using heston_windows) ---
    for window in heston_windows:
        k = 0.34 / (1 + (window + 1) / (window - 1))
        
        # Calculate open-close and close-open log returns with proper null handling
        df = df.with_columns([
            pl.when((pl.col("open").shift(1).over("act_symbol") > 0) & (pl.col("close") > 0))
            .then((pl.col("close") / pl.col("open").shift(1).over("act_symbol")).log())
            .otherwise(None)
            .alias(f"open_close_{window}"),
            
            pl.when((pl.col("close").shift(1).over("act_symbol") > 0) & (pl.col("open") > 0))
            .then((pl.col("open") / pl.col("close").shift(1).over("act_symbol")).log())
            .otherwise(None)
            .alias(f"close_open_{window}")
        ])
        
        # Calculate variances with minimum observation requirement
        df = df.with_columns([
            pl.col(f"open_close_{window}")
            .rolling_var(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_oc_{window}"),
            
            pl.col(f"close_open_{window}")
            .rolling_var(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_co_{window}"),
            
            pl.col("log_returns")
            .rolling_var(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_cc_{window}")
        ])
        
        # Calculate Yang-Zhang volatility with comprehensive safeguards
        df = df.with_columns(
            pl.when(
                pl.col(f"vol_oc_{window}").is_finite() & 
                pl.col(f"vol_co_{window}").is_finite() & 
                pl.col(f"vol_cc_{window}").is_finite() &
                (pl.col(f"vol_oc_{window}") > 0) &
                (pl.col(f"vol_co_{window}") > 0) &
                (pl.col(f"vol_cc_{window}") > 0)
            )
            .then(
                ((k * pl.col(f"vol_oc_{window}")) + 
                ((1 - k) * pl.col(f"vol_co_{window}")) + 
                pl.col(f"vol_cc_{window}")).sqrt()
            )
            .otherwise(
                # Fallback to standard deviation when YZ can't be calculated properly
                pl.col("log_returns")
                .rolling_std(window, min_periods=max(5, window//5))
                .over("act_symbol")
            )
            .clip(0.001, 0.5)  # Apply reasonable bounds for volatility
            .alias(f"YZVol_{window}")
        )

        # Calculate forward-looking version if requested
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"YZVol_{window}").shift(-window).over("act_symbol")
                .alias(f"YZVol_{window}_future")
            )
    
    # --- Step 16: Upside and Downside Volatility for Heston windows ---
    for window in heston_windows:
        # Calculate upside and downside volatility with minimum observations
        df = df.with_columns([
            pl.col("pos_returns")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"upside_vol_{window}"),
            
            pl.col("neg_returns")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"downside_vol_{window}")
        ])
        
        # Calculate volatility skew with improved handling
        df = df.with_columns(
            pl.when((pl.col(f"YZVol_{window}") > 0.00001) &
                   pl.col(f"upside_vol_{window}").is_finite() &
                   pl.col(f"downside_vol_{window}").is_finite())
            .then((pl.col(f"downside_vol_{window}") - pl.col(f"upside_vol_{window}")) / 
                 pl.col(f"YZVol_{window}"))
            .otherwise(0.0)  # Default to zero skew
            .clip(-1.0, 1.0)  # Ensure skew is within reasonable bounds
            .alias(f"VolSkew_{window}")
        )
        
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"VolSkew_{window}").shift(-window).over("act_symbol")
                .alias(f"VolSkew_{window}_future")
            )
        
    # --- Step 17: VolCurvature for Heston windows ---
    for window in heston_windows:
        # Use quantile-based curvature with comprehensive safeguards
        df = df.with_columns(
            pl.when(
                (pl.col(f"QuantileSpread_{window}_IQR") > 0.0001) &
                pl.col(f"QuantileSpread_{window}_Extreme").is_finite() &
                pl.col(f"QuantileSpread_{window}_IQR").is_finite()
            )
            .then(
                ((pl.col(f"QuantileSpread_{window}_Extreme") / pl.col(f"QuantileSpread_{window}_IQR")) - 1.0)
                .clip(-5.0, 5.0)  # More restrictive clipping to prevent extreme values
            )
            .otherwise(
                # Fall back to kurtosis-based measure with clipping for stability
                pl.col(f"Kurtosis_{window}").clip(-3.0, 3.0) / 6.0
            )
            .alias(f"VolCurvature_{window}")
        )
        
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"VolCurvature_{window}").shift(-window).over("act_symbol")
                .alias(f"VolCurvature_{window}_future")
            )
    
    # --- Step 18: Wing Ratio calculation for Heston windows ---
    for window in heston_windows:
        # Calculate 1% and 99% quantiles for wing ratio with minimum periods
        df = df.with_columns([
            pl.col("log_returns").rolling_quantile(quantile=0.01, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q01_{window}"),
            
            pl.col("log_returns").rolling_quantile(quantile=0.99, interpolation="linear", 
                                                 window_size=window, min_periods=max(5, window//5))
            .over("act_symbol").alias(f"q99_{window}")
        ])
        
        # Calculate wing ratio with comprehensive safety checks
        df = df.with_columns(
            pl.when(
                (pl.col(f"q99_{window}").abs() > 0.0001) &
                pl.col(f"q01_{window}").is_finite() &
                pl.col(f"q99_{window}").is_finite()
            )
            .then((pl.col(f"q01_{window}").abs() / pl.col(f"q99_{window}").abs()).clip(0.1, 5.0))  # More restrictive clip
            .otherwise(1.0)  # Default to balanced risk when calculation is unsafe
            .alias(f"WingRatio_{window}")
        )

        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"WingRatio_{window}").shift(-window).over("act_symbol")
                .alias(f"WingRatio_{window}_future")
            )
    
    # --- Step 19: Mean Reversion Speed (Enhanced Half-life Method) ---
    for window in heston_windows:
        # Calculate lag based on window size for efficiency
        lag = max(1, min(5, window // 20))
        
        # Calculate auto-correlation with a single lag for efficiency
        df = df.with_columns([
            pl.col(f"YZVol_{window}").shift(lag).over("act_symbol").alias(f"vol_lag_{window}"),
            
            # Calculate mean for centered product
            pl.col(f"YZVol_{window}")
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_mean_{window}")
        ])
        
        # Calculate centered product (covariance numerator)
        df = df.with_columns(
            ((pl.col(f"YZVol_{window}") - pl.col(f"vol_mean_{window}")) * 
            (pl.col(f"vol_lag_{window}") - pl.col(f"vol_mean_{window}")))
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"cov_num_{window}")
        )
        
        # Calculate variance (denominator)
        df = df.with_columns(
            (pl.col(f"YZVol_{window}") - pl.col(f"vol_mean_{window}")).pow(2)
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_var_{window}")
        )
        
        # Calculate auto-correlation (single pass)
        df = df.with_columns(
            pl.when(pl.col(f"vol_var_{window}") > 0.00001)
            .then(pl.col(f"cov_num_{window}") / pl.col(f"vol_var_{window}"))
            .otherwise(0.5)
            .clip(-0.99, 0.99)  # Ensure valid
            .alias(f"ac_{window}")
        )
        
        # Convert to mean reversion coefficient (1 - autocorrelation)
        # Higher value means stronger mean reversion
        df = df.with_columns(
            (1.0 - pl.col(f"ac_{window}"))
            .clip(0.01, 0.999)  # Ensure reasonable range
            .alias(f"MeanReversion_{window}")
        )
        
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"MeanReversion_{window}").shift(-window).over("act_symbol")
                .alias(f"MeanReversion_{window}_future")
            )
            
    # --- Step 20: Volatility of Volatility (Heston σᵥ parameter) ---
    for window in heston_windows:
        # Calculate daily changes in volatility
        df = df.with_columns(
            pl.col(f"YZVol_{window}").diff().over("act_symbol").alias(f"YZVol_{window}_diff")
        )
        
        # Calculate rolling standard deviation of volatility changes with minimum periods
        df = df.with_columns(
            pl.col(f"YZVol_{window}_diff")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .clip(0, 0.1)  
            .alias(f"VolOfVol_{window}")
        )
                
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"VolOfVol_{window}").shift(-window).over("act_symbol")
                .alias(f"VolOfVol_{window}_future")
            )
    
    # --- Step 21: Price-Volatility Correlation (Heston ρ parameter) ---
    for window in heston_windows:
        # Calculate the mean of the product of returns and vol changes with proper filtering
        df = df.with_columns(
            (pl.col("log_returns") * pl.col(f"YZVol_{window}_diff"))
            .rolling_mean(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"return_vol_cov_{window}")
        )
        
        # Calculate the standard deviations with minimum periods
        df = df.with_columns([
            pl.col("log_returns")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"return_std_{window}"),
            
            pl.col(f"YZVol_{window}_diff")
            .rolling_std(window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"vol_diff_std_{window}")
        ])
        
        # Calculate correlation with comprehensive safety checks
        df = df.with_columns(
            pl.when(
                (pl.col(f"return_std_{window}") > 0.00001) & 
                (pl.col(f"vol_diff_std_{window}") > 0.00001) &
                pl.col(f"return_vol_cov_{window}").is_finite() &
                pl.col(f"return_std_{window}").is_finite() &
                pl.col(f"vol_diff_std_{window}").is_finite()
            )
            .then(
                (pl.col(f"return_vol_cov_{window}") / 
                (pl.col(f"return_std_{window}") * pl.col(f"vol_diff_std_{window}")))
                .clip(-1.0, 1.0)  
            )
            .otherwise(0.0)  
            .alias(f"PriceVolCorr_{window}")
        )
        
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"PriceVolCorr_{window}").shift(-window).over("act_symbol")
                .alias(f"PriceVolCorr_{window}_future")
            )
        
    # --- Step 22: Volatility Regime Indicators (Optimized for Memory) ---
    for window in vol_windows:
        df = df.with_columns(
            pl.when(pl.col(f"YZVol_{window}").shift(5).over("act_symbol") > 0.00001)
            .then(
                (pl.col(f"YZVol_{window}") - pl.col(f"YZVol_{window}").shift(5).over("act_symbol")) /
                pl.col(f"YZVol_{window}").shift(5).over("act_symbol")
            )
            .otherwise(0.0) 
            .clip(-1.0, 1.0) 
            .alias(f"VolTrend_{window}")
        )

    # --- Step 23: Volatility Intensity (Optimized Tail-weighted Method with 0-1 Scaling) ---
    for window in heston_windows:
        # Calculate only the necessary quantiles for efficiency
        # We use absolute returns for more efficient calculation of tail behavior
        df = df.with_columns(
            pl.col("log_returns").abs().alias(f"abs_returns_{window}")
        )
        
        # Calculate key quantiles in a single pass for efficiency
        df = df.with_columns([
            pl.col(f"abs_returns_{window}")
            .rolling_quantile(quantile=0.95, window_size=window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"q95_{window}_tail"),
            
            pl.col(f"abs_returns_{window}")
            .rolling_quantile(quantile=0.75, window_size=window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"q75_{window}_tail"),
            
            pl.col(f"abs_returns_{window}")
            .rolling_quantile(quantile=0.50, window_size=window, min_periods=max(5, window//5))
            .over("act_symbol")
            .alias(f"q50_{window}_tail")
        ])
        
        # Calculate upper/median ratio with bounded values
        df = df.with_columns(
            pl.when(pl.col(f"q50_{window}_tail") > 0.00001)
            .then((pl.col(f"q95_{window}_tail") / pl.col(f"q50_{window}_tail")).clip(1.0, 10.0))
            .otherwise(2.0)  # Reasonable default
            .alias(f"tail_ratio_{window}")
        )
        
        # Calculate interquartile ratio with bounded values
        df = df.with_columns(
            pl.when(pl.col(f"q50_{window}_tail") > 0.00001)
            .then((pl.col(f"q75_{window}_tail") / pl.col(f"q50_{window}_tail")).clip(1.0, 5.0))
            .otherwise(1.5)  # Reasonable default
            .alias(f"iqr_ratio_{window}")
        )
        
        # Calculate raw tail intensity (still with ~0.8-1.2 range)
        df = df.with_columns(
            (0.6 * pl.col(f"tail_ratio_{window}") / (1.0 + 0.6 * pl.col(f"tail_ratio_{window}")) + 
            0.4 * pl.col(f"iqr_ratio_{window}") / (1.0 + 0.4 * pl.col(f"iqr_ratio_{window}")))
            .alias(f"raw_intensity_{window}")
        )
        
        # Rescale to 0-1 and center at 0.5
        # We know the range is approximately 0.8 to 1.2 from the statistics
        # Map 0.8 -> 0.3 and 1.2 -> 0.7 to center at 0.5 with appropriate scaling
        df = df.with_columns(
            (((pl.col(f"raw_intensity_{window}") - 0.8) / 0.4) * 0.4 + 0.3)
            .clip(0.0, 1.0)  # Ensure final values are in 0-1 range
            .alias(f"VolIntensity_{window}")
        )
        
        # Forward-looking version
        if calculate_future_cols:
            df = df.with_columns(
                pl.col(f"VolIntensity_{window}").shift(-window).over("act_symbol")
                .alias(f"VolIntensity_{window}_future")
            )
        
    # --- Step 24: Volatility Ratios ---
    # IntradayToFullVol (Parkinson/YZVol)
    for window in vol_windows:
        df = df.with_columns(
            pl.when(pl.col(f"YZVol_{window}") > 0.00001)
            .then((pl.col(f"ParkinsonVol_{window}") / pl.col(f"YZVol_{window}")).clip(0, 10))
            .otherwise(1.0) 
            .alias(f"IntradayToFullVol_{window}")
        )

    # VolTermRatio (term structure steepness)
    # Key term structure pairs
    term_structure_pairs = [
        (10, 15), (10, 20), (10, 35), (15, 20), (15, 35), (20, 25), (20, 30), (20, 35)
    ]
    
    # Calculate for each pair with proper safeguards
    for short_window, long_window in term_structure_pairs:
        ratio_name = f"VolTermRatio_{short_window}_{long_window}"
        df = df.with_columns(
            pl.when(
                (pl.col(f"YZVol_{long_window}") > 0.00001) &
                pl.col(f"YZVol_{short_window}").is_finite() &
                pl.col(f"YZVol_{long_window}").is_finite()
            )
            .then((pl.col(f"YZVol_{short_window}") / pl.col(f"YZVol_{long_window}")).clip(0.1, 5.0))
            .otherwise(1.0) 
            .alias(ratio_name)
        )
    
    # --- Step 25: Clean up intermediate calculation columns ---
    # Define columns to keep (base columns + final indicator columns)
    base_cols = ["act_symbol", "date", "open", "high", "low", "close", "volume", "log_returns"]

    # Final indicator columns (only keep the actual indicators, not intermediate calculations)
    indicator_cols = ["OBV"]

    # Add standard indicators for vol_windows
    for window in vol_windows:
        indicator_cols.extend([
            f"SMA_{window}", f"EMA_{window}", f"STD_{window}",
            f"ATR_{window}", f"DonchianRange_{window}",
            f"ParkinsonVol_{window}",
            # Volatility regime indicators
            f"VolTrend_{window}", f"IntradayToFullVol_{window}"
        ])

    # Add core window indicators (limited to shorter windows)
    for window in core_windows:
        indicator_cols.extend([
            f"SMA_EMA_ratio_{window}", f"RSI_{window}", f"CMF_{window}", f"CCI_{window}",
            f"OBV_change_{window}", f"VWAP_{window}", f"VWAP_deviation_{window}"
        ])

    # Add all heston-related indicators for all heston windows
    for window in heston_windows:
        # Base indicators always included
        base_heston_indicators = [
            f"YZVol_{window}",
            f"QuantileSpread_{window}_Extreme", f"QuantileSpread_{window}_IQR",
            f"Skewness_{window}", f"Kurtosis_{window}", 
            f"VolSkew_{window}", f"VolCurvature_{window}", f"WingRatio_{window}",
            f"MeanReversion_{window}", f"VolOfVol_{window}", f"PriceVolCorr_{window}",
            f"VolIntensity_{window}"
        ]
        indicator_cols.extend(base_heston_indicators)
        
        # Add future indicators only if requested
        if calculate_future_cols:
            future_indicators = [
                f"YZVol_{window}_future", f"VolSkew_{window}_future",
                f"VolCurvature_{window}_future", f"WingRatio_{window}_future",
                f"MeanReversion_{window}_future", f"VolOfVol_{window}_future",
                f"PriceVolCorr_{window}_future", f"VolIntensity_{window}_future"
            ]
            indicator_cols.extend(future_indicators)

    # Add volatility term structure metrics
    for short_window, long_window in term_structure_pairs:
        indicator_cols.append(f"VolTermRatio_{short_window}_{long_window}")
    
    # Select only the specified columns
    cols_to_keep = base_cols + indicator_cols

    # First, define the numeric columns (all columns except "act_symbol" and "date")
    numeric_cols_to_keep = [col for col in cols_to_keep if col not in ["act_symbol", "date"]]

    # Select all columns from cols_to_keep, but when filtering, only apply is_finite to the numeric columns
    df = df.select(cols_to_keep).filter(pl.all_horizontal(pl.col(numeric_cols_to_keep).is_finite()))
        
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