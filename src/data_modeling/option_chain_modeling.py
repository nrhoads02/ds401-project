import numpy as np
import polars as pl
import datetime
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Optional, Tuple, Any
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
from scipy.interpolate import griddata

# ======================================================
# Data Loading and Date Handling Functions
# ======================================================

def find_nearest_date(
    df: pl.DataFrame,
    target_date: Union[str, datetime.date],
    date_column: str = "date",
    backward: bool = True
) -> Optional[Union[datetime.date, str]]:
    """
    Find the nearest available date in a DataFrame.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame to search in
    target_date : Union[str, datetime.date]
        Target date to find
    date_column : str
        Name of the date column
    backward : bool
        If True, search backward in time; if False, search forward
        
    Returns:
    --------
    Optional[Union[datetime.date, str]]
        Nearest available date, or None if DataFrame is empty
    """
    if df.height == 0:
        return None
    
    # Convert target_date to datetime.date if it's a string
    if isinstance(target_date, str):
        target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    
    # Get all unique dates
    all_dates = df[date_column].unique().sort()
    
    # Convert to numpy array for easier comparison
    date_array = np.array([d if isinstance(d, datetime.date) else 
                          datetime.datetime.strptime(str(d), "%Y-%m-%d").date() 
                          for d in all_dates])
    
    # Find the index of the closest date
    if backward:
        valid_dates = date_array[date_array <= target_date]
        if len(valid_dates) == 0:
            # If no earlier date, use the earliest available
            return date_array[0] if len(date_array) > 0 else None
        return valid_dates[-1]  # The most recent date that's <= target_date
    else:
        valid_dates = date_array[date_array >= target_date]
        if len(valid_dates) == 0:
            # If no later date, use the latest available
            return date_array[-1] if len(date_array) > 0 else None
        return valid_dates[0]  # The earliest date that's >= target_date

def load_option_chain_data(
    df: pl.DataFrame,
    stock: str,
    target_date: Union[str, datetime.date]
) -> Tuple[pl.DataFrame, Union[datetime.date, str]]:
    """
    Load option chain data for the nearest available date.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with option chain data
    stock : str
        Stock symbol to analyze
    target_date : Union[str, datetime.date]
        Target date for analysis
        
    Returns:
    --------
    Tuple[pl.DataFrame, Union[datetime.date, str]]
        Filtered DataFrame and the actual date used
    """
    # Filter by stock
    stock_df = df.filter(pl.col("act_symbol") == stock)
    
    if stock_df.height == 0:
        raise ValueError(f"No data found for stock {stock}")
    
    # Find nearest available date
    actual_date = find_nearest_date(stock_df, target_date, "date", backward=True)
    
    if actual_date is None:
        raise ValueError(f"No data available for {stock} on or before {target_date}")
    
    # Filter by date
    result_df = stock_df.filter(pl.col("date") == actual_date)
    
    # Ensure date and expiration are proper date types
    if result_df["date"].dtype != pl.Date:
        result_df = result_df.with_columns([
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
        ])
    
    if result_df["expiration"].dtype != pl.Date:
        result_df = result_df.with_columns([
            pl.col("expiration").str.strptime(pl.Date, "%Y-%m-%d").alias("expiration")
        ])
    
    return result_df, actual_date

def load_ohlcv_data(
    df: pl.DataFrame,
    stock: str,
    target_date: Union[str, datetime.date]
) -> Tuple[pl.DataFrame, Union[datetime.date, str]]:
    """
    Load OHLCV data for the nearest available date.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with OHLCV data
    stock : str
        Stock symbol to analyze
    target_date : Union[str, datetime.date]
        Target date for analysis
        
    Returns:
    --------
    Tuple[pl.DataFrame, Union[datetime.date, str]]
        Filtered DataFrame and the actual date used
    """
    # Filter by stock
    stock_df = df.filter(pl.col("act_symbol") == stock)
    
    if stock_df.height == 0:
        raise ValueError(f"No data found for stock {stock}")
    
    # Find nearest available date
    actual_date = find_nearest_date(stock_df, target_date, "date", backward=True)
    
    if actual_date is None:
        raise ValueError(f"No data available for {stock} on or before {target_date}")
    
    # Filter by date
    result_df = stock_df.filter(pl.col("date") == actual_date)
    
    # Ensure date is a proper date type
    if result_df["date"].dtype != pl.Date:
        result_df = result_df.with_columns([
            pl.col("date").str.strptime(pl.Date, "%Y-%m-%d").alias("date")
        ])
    
    return result_df, actual_date

def generate_prediction_data(
    model_results: Dict,
    ohlcv_df: pl.DataFrame,
    stock: str,
    target_date: Union[str, datetime.date],
    lgbm_modeling: Any  # The imported module
) -> Tuple[pl.DataFrame, Union[datetime.date, str]]:
    """
    Generate prediction data using LGBM model.
    
    Parameters:
    -----------
    model_results : Dict
        Results dictionary from LGBM model training
    ohlcv_df : pl.DataFrame
        DataFrame with OHLCV data
    stock : str
        Stock symbol to analyze
    target_date : Union[str, datetime.date]
        Target date for analysis
    lgbm_modeling : Any
        The imported LGBM modeling module
        
    Returns:
    --------
    Tuple[pl.DataFrame, Union[datetime.date, str]]
        Prediction DataFrame and the actual date used
    """
    # Filter OHLCV data for the stock
    stock_df = ohlcv_df.filter(pl.col("act_symbol") == stock)
    
    # Find nearest available date
    actual_date = find_nearest_date(stock_df, target_date, "date", backward=True)
    
    if actual_date is None:
        raise ValueError(f"No data available for {stock} on or before {target_date}")
    
    # Convert to string if it's a datetime.date
    if isinstance(actual_date, datetime.date):
        date_str = actual_date.strftime("%Y-%m-%d")
    else:
        date_str = actual_date
    
    # Generate predictions
    pred_df = lgbm_modeling.predict_for_visualization(
        model_results=model_results,
        df=ohlcv_df,
        stock=stock,
        date=date_str
    )
    
    return pred_df, actual_date

# ======================================================
# Surface Generation Functions
# ======================================================

def create_option_vol_surface(
    option_df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    option_type: str = 'Call',
    vol_scale: float = 1.0  # Scale factor for volatility values
) -> Dict:
    """
    Create implied volatility surface from option chain data.
    
    Parameters:
    -----------
    option_df : pl.DataFrame
        DataFrame with option chain data
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    option_type : str
        Option type to visualize ('Call' or 'Put')
    vol_scale : float
        Scaling factor for volatility values (use 100.0 if vols are in decimal form but need percentage)
        
    Returns:
    --------
    Dict
        Dictionary with surface data and metadata
    """
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Find nearest available date in the dataset
    actual_date = find_nearest_date(option_df, date, "date", backward=True)
    if actual_date is None:
        raise ValueError(f"No data available for {stock} on or before {date}")
    
    print(f"Using data from {actual_date} for analysis (requested date: {date})")
    
    # Filter data for the specific stock, date, and option type
    filtered_df = option_df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") == actual_date) &
        (pl.col("call_put") == option_type)
    )
    
    if filtered_df.height == 0:
        raise ValueError(f"No {option_type} option data found for {stock} on {actual_date}")
    
    # Only include options that expire after our target date
    filtered_df = filtered_df.filter(pl.col("expiration") > actual_date)
    
    if filtered_df.height == 0:
        raise ValueError(f"No {option_type} options with future expiration found for {stock} on {actual_date}")
    
    # Calculate days to expiration
    filtered_df = filtered_df.with_columns([
        ((pl.col("expiration") - actual_date).dt.total_days()).alias("days_to_expiration")
    ])
    
    # Print summary to help diagnose
    print(f"Found {filtered_df.height} options for {stock} {option_type} on {actual_date}")
    print(f"Days to expiration range: {filtered_df['days_to_expiration'].min()} to {filtered_df['days_to_expiration'].max()}")
    print(f"Strike price range: {filtered_df['strike'].min()} to {filtered_df['strike'].max()}")
    
    # Get unique strike prices and days to expiration
    days_to_exp = filtered_df["days_to_expiration"].unique().sort().to_numpy()
    
    # Get strikes, focusing on those that have data
    all_strikes = filtered_df["strike"].unique().sort().to_numpy()
    
    # If we have too many strikes, select a subset
    if len(all_strikes) > 20:
        # Get the range of strikes with most data
        # Find the "at the money" point as a reference
        close_price = filtered_df.select(pl.col("strike") - 
                                        (pl.col("bid") + pl.col("ask"))/2).abs().min()
        atm_index = np.abs(all_strikes - close_price).argmin()
        
        # Select strikes around the ATM point
        start_idx = max(0, atm_index - 10)
        end_idx = min(len(all_strikes), atm_index + 11)
        strikes = all_strikes[start_idx:end_idx]
    else:
        strikes = all_strikes
    
    print(f"Using {len(days_to_exp)} expiration dates and {len(strikes)} strike prices for surface")
    
    # Create a grid for the surface
    strike_grid, days_grid = np.meshgrid(strikes, days_to_exp)
    
    # Initialize the value grid with NaNs
    vol_grid = np.full(strike_grid.shape, np.nan)
    
    # Create lookup dictionary for fast access
    lookup_dict = {}
    for row in filtered_df.select(["days_to_expiration", "strike", "vol"]).iter_rows():
        days, strike, vol = row
        lookup_dict[(days, strike)] = vol
    
    # Fill in the value grid using the lookup dictionary
    for i, day in enumerate(days_to_exp):
        for j, strike in enumerate(strikes):
            key = (day, strike)
            if key in lookup_dict:
                vol_grid[i, j] = lookup_dict[key] * vol_scale
    
    # Count how many values we have before interpolation
    valid_before = np.sum(~np.isnan(vol_grid))
    filled_pct_before = valid_before / vol_grid.size
    print(f"Valid data points before interpolation: {valid_before} out of {vol_grid.size} ({filled_pct_before:.1%})")
    
    # Fill in missing values using interpolation if possible
    if valid_before >= 5:  # Need at least 5 valid points for interpolation
        # Create mask of valid points
        mask = ~np.isnan(vol_grid)
        
        # Extract coordinates and values of valid points
        x_valid = strike_grid[mask]
        y_valid = days_grid[mask]
        z_valid = vol_grid[mask]
        
        # Perform interpolation
        points = np.column_stack((x_valid, y_valid))
        try:
            # Try linear interpolation (more stable than cubic)
            grid_z = griddata(points, z_valid, (strike_grid, days_grid), method='linear')
            
            # If we still have NaNs, try nearest
            if np.any(np.isnan(grid_z)):
                nan_mask = np.isnan(grid_z)
                grid_z_nearest = griddata(points, z_valid, (strike_grid, days_grid), method='nearest')
                grid_z[nan_mask] = grid_z_nearest[nan_mask]
            
            # Only update original grid where we had NaNs
            nan_mask = np.isnan(vol_grid)
            vol_grid[nan_mask] = grid_z[nan_mask]
            
            # Report on interpolation
            valid_after = np.sum(~np.isnan(vol_grid))
            interpolated = valid_after - valid_before
            print(f"Interpolated {interpolated} additional points")
        except Exception as e:
            print(f"Interpolation failed: {str(e)}")
    
    # Calculate percent filled
    filled_pct = np.sum(~np.isnan(vol_grid)) / vol_grid.size
    print(f"Final valid data points: {np.sum(~np.isnan(vol_grid))} out of {vol_grid.size} ({filled_pct:.1%})")
    
    # Return as a dictionary with all the data needed for visualization
    return {
        "type": "option_market",
        "subtype": option_type,
        "x": strike_grid,
        "y": days_grid,
        "z": vol_grid,
        "x_label": "Strike Price",
        "y_label": "Days to Expiration",
        "z_label": "Implied Volatility",
        "title": f"{stock} {option_type} Option Implied Volatility Surface on {actual_date}",
        "description": f"Market implied volatility surface from {option_type} options",
        "date": actual_date,
        "filled_pct": filled_pct
    }

def create_model_vol_surface(
    pred_df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    windows: Optional[List[int]] = None,
    vol_scale: float = 1.0,
    pred_suffix: str = "_future"
) -> Dict:
    """
    Create volatility surface from model predictions using continuous volatility
    intensity and proper skew scaling.
    
    Parameters:
    -----------
    pred_df : pl.DataFrame
        DataFrame with model predictions
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    windows : List[int], optional
        List of time windows to include
    vol_scale : float
        Scaling factor for volatility values
    pred_suffix : str
        Suffix for prediction columns
        
    Returns:
    --------
    Dict
        Dictionary with surface data and metadata
    """
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Filter for stock and date
    stock_data = pred_df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") == date)
    )
    
    if stock_data.height == 0:
        raise ValueError(f"No data found for {stock} on {date}")
    
    # Extract stock data row for direct indexing like original code
    stock_cols = stock_data.columns
    stock_row = stock_data.row(0)
    
    # Determine available windows if not specified
    if windows is None:
        # Use the same function from original code
        available_windows = []
        for col in stock_cols:
            if col.startswith("YZVol_") and col.endswith(pred_suffix):
                try:
                    window_str = col.replace("YZVol_", "").replace(pred_suffix, "")
                    window = int(window_str)
                    available_windows.append(window)
                except ValueError:
                    continue
        
        windows = sorted(available_windows)
    
    if not windows:
        raise ValueError(f"No prediction windows found with suffix '{pred_suffix}'")
    
    # Convert windows to days and years exactly like original code
    time_days = windows
    times = np.array([w/252 for w in windows])  # Convert to years

    # Moneyness values (log(K/S))
    moneyness = np.linspace(-0.15, 0.15, 11)
    
    # Create output grid - swap dimensions to make moneyness first and time second
    Z = np.zeros((len(times), len(moneyness)))
    
    # --- Get term structure of volatility (ATM volatilities) ---
    atm_vols = []
    for days in time_days:
        vol_windows = []
        for col in stock_cols:
            if col.startswith('YZVol_') and not col.endswith('_pred'):
                try:
                    win = int(col.replace('YZVol_', '').replace(pred_suffix, ''))
                    vol_windows.append(win)
                except ValueError:
                    continue
        
        if not vol_windows:
            atm_vols.append(0.25)  # Default if no data
            continue
            
        closest_win = min(vol_windows, key=lambda x: abs(x - days))
        future_col = f'YZVol_{closest_win}{pred_suffix}'
        
        if future_col in stock_cols:
            col_idx = stock_cols.index(future_col)
            atm_vols.append(stock_row[col_idx] * vol_scale)
        else:
            # Use current volatility as fallback
            current_col = f'YZVol_{closest_win}' 
            if current_col in stock_cols:
                col_idx = stock_cols.index(current_col)
                atm_vols.append(stock_row[col_idx] * vol_scale)
            else:
                atm_vols.append(0.25 * vol_scale)  # Default if no data
    
    # Smooth the ATM volatility curve if we have enough points
    if len(atm_vols) >= 4:
        # Convert to numpy for easier manipulation
        x_days = np.array(time_days)
        y_vols = np.array(atm_vols)
        
        # Simple smoothing using rolling average
        window_size = 3
        weights = np.ones(window_size) / window_size
        # Pad the ends to avoid edge effects
        padded_vols = np.pad(y_vols, (window_size//2, window_size//2), mode='edge')
        smoothed_vols = np.convolve(padded_vols, weights, mode='valid')
        
        # Use the smoothed values but keep original values at endpoints
        atm_vols = list(smoothed_vols)
    
    # Scale the volatility range to be more reasonable
    # Find min and max volatility
    min_vol = min(atm_vols)
    max_vol = max(atm_vols)
    
    # If the range is too large, compress it
    if max_vol / min_vol > 4.0:
        # Compress the range by bringing extremes closer to the mean
        mean_vol = np.mean(atm_vols)
        compression_factor = 0.6  # Lower values = more compression
        atm_vols = [mean_vol + compression_factor * (vol - mean_vol) for vol in atm_vols]
    
    # --- Core model parameters ---
    # Get key Heston parameters from a medium-term window for base model
    medium_term_idx = len(time_days) // 2
    medium_term_days = time_days[medium_term_idx]
    medium_term_t = times[medium_term_idx]
    
    # Get mean reversion parameter
    mr_windows = []
    for col in stock_cols:
        if col.startswith('MeanReversion_') and col.endswith(pred_suffix):
            try:
                win = int(col.replace('MeanReversion_', '').replace(pred_suffix, ''))
                mr_windows.append(win)
            except ValueError:
                continue
    
    if mr_windows:
        closest_mr_win = min(mr_windows, key=lambda x: abs(x - medium_term_days))
        mr_col = f'MeanReversion_{closest_mr_win}{pred_suffix}'
        if mr_col in stock_cols:
            col_idx = stock_cols.index(mr_col)
            base_kappa = stock_row[col_idx]
        else:
            base_kappa = 1.5  # Default mean reversion
    else:
        base_kappa = 1.5
        
    # Volatility of volatility
    vv_windows = []
    for col in stock_cols:
        if col.startswith('VolOfVol_') and col.endswith(pred_suffix):
            try:
                win = int(col.replace('VolOfVol_', '').replace(pred_suffix, ''))
                vv_windows.append(win)
            except ValueError:
                continue
    
    if vv_windows:
        closest_vv_win = min(vv_windows, key=lambda x: abs(x - medium_term_days))
        vv_col = f'VolOfVol_{closest_vv_win}{pred_suffix}'
        if vv_col in stock_cols:
            col_idx = stock_cols.index(vv_col)
            base_volvol = stock_row[col_idx]
        else:
            base_volvol = 0.3  # Default volvol
    else:
        base_volvol = 0.3
        
    # Price-vol correlation
    corr_windows = []
    for col in stock_cols:
        if col.startswith('PriceVolCorr_') and col.endswith(pred_suffix):
            try:
                win = int(col.replace('PriceVolCorr_', '').replace(pred_suffix, ''))
                corr_windows.append(win)
            except ValueError:
                continue
    
    if corr_windows:
        closest_corr_win = min(corr_windows, key=lambda x: abs(x - medium_term_days))
        corr_col = f'PriceVolCorr_{closest_corr_win}{pred_suffix}'
        if corr_col in stock_cols:
            col_idx = stock_cols.index(corr_col)
            base_rho = stock_row[col_idx]
        else:
            base_rho = -0.5  # Default correlation (negative for equities)
    else:
        base_rho = -0.5
    
    # Fix-up parameters for numerical stability
    base_kappa = np.clip(base_kappa, 0.1, 5.0)
    base_volvol = np.clip(base_volvol, 0.01, 1.0)
    base_rho = np.clip(base_rho, -0.9, 0.5)
    
    # Scale down vol-of-vol to reduce the overall volatility range
    base_volvol *= 0.7  # Reduce by 30%
    
    # Process each time slice with term-structure consistency
    for i, (t, days) in enumerate(zip(times, time_days)):
        # Get tenor-specific volatility intensity (continuous value scaled 0 to 1)
        vol_intensity = 0.5  # Default
        intensity_cols = []
        for col in stock_cols:
            if col.startswith('VolIntensity_') and col.endswith(pred_suffix):
                try:
                    win = int(col.replace('VolIntensity_', '').replace(pred_suffix, ''))
                    intensity_cols.append((win, col))
                except ValueError:
                    continue
        
        if intensity_cols:
            closest_win = min(intensity_cols, key=lambda x: abs(x[0] - days))[0]
            for win, col in intensity_cols:
                if win == closest_win:
                    col_idx = stock_cols.index(col)
                    vol_intensity = stock_row[col_idx]
                    break
        
        # Get the ATM volatility for this tenor
        atm_vol = atm_vols[i]
        
        # IMPROVED: Use continuous functions of vol_intensity and tenor for parameter adjustments
        
        # 1. Tenor-based adjustments using continuous functions
        # Short-term factor: higher for short tenors, approaches 1.0 for longer tenors
        tenor_factor = 1.0 + 0.5 * np.exp(-5.0 * t)
        
        # 2. Continuous intensity-based adjustments
        # Instead of discrete ranges, use smooth functions of intensity
        # Linear component: increases from min to max as intensity goes from 0 to 1
        intensity_linear = 0.85 + 0.35 * vol_intensity
        
        # Nonlinear component: accelerates as intensity increases (stronger effect at high intensity)
        intensity_nonlinear = vol_intensity ** 2
        
        # 3. Apply adjustments to parameters
        # For kappa (mean reversion)
        kappa_adj = 1.0 + 0.3 * (vol_intensity ** 1.2)  # Slightly nonlinear
        kappa = base_kappa * kappa_adj
        
        # For volvol (volatility of volatility)
        # Short tenors get higher volvol adjustment
        volvol_tenor_adj = 1.0 - 0.15 * (1.0 - np.exp(-3.0 * t))
        # Intensity affects volvol more strongly
        volvol_intensity_adj = intensity_linear * (1.0 + 0.2 * intensity_nonlinear)
        volvol = base_volvol * volvol_tenor_adj * volvol_intensity_adj
        
        # For rho (correlation)
        # Shorter tenors have stronger correlation adjustment
        rho_tenor_adj = 1.0 + 0.1 * np.exp(-4.0 * t)
        # Intensity has nonlinear effect on correlation strength
        rho_intensity_adj = 1.0 + 0.3 * vol_intensity + 0.2 * vol_intensity**2
        rho = base_rho * rho_tenor_adj * rho_intensity_adj
        
        # Ensure parameters remain within bounds
        kappa = np.clip(kappa, 0.1, 10.0)
        volvol = np.clip(volvol, 0.01, 2.0)
        rho = np.clip(rho, -0.95, 0.7)
        
        # Convert volatility to variance (v = σ²)
        v0 = (atm_vol / vol_scale)**2
        
        # Approximate long-term variance based on term structure
        if i < len(times) - 1:
            forward_vol = atm_vols[min(i + 1, len(atm_vols) - 1)] / vol_scale
            theta = forward_vol**2
        else:
            theta = ((atm_vol / vol_scale) * 0.97)**2
        
        theta = max(0.0004, theta)
        
        # Apply term structure effect with mean-reversion
        v_t = v0 + (theta - v0) * (1 - np.exp(-kappa * t))
        base_vol = np.sqrt(v_t) * vol_scale
        
        surface_slice = np.ones_like(moneyness) * base_vol
        
        # IMPROVED: Enhance skew calculation to properly scale with vol_scale
        
        # 1. Calculate skew effect with proper scaling
        # The key insight: skew effects should be proportional to base_vol
        skew_strength = np.sqrt(t) * (1 - np.exp(-kappa * t)) / (kappa * t) * 0.3
        
        # Scale skew with volatility intensity
        skew_intensity_factor = 1.0 + 0.4 * vol_intensity
        skew_strength *= skew_intensity_factor
        
        # Calculate basic skew effect
        skew_effect = rho * volvol * moneyness * skew_strength
        
        # 2. Calculate curvature effect with proper scaling
        curvature_strength = (1 - rho**2) * volvol**2 * t * np.exp(-kappa * t) * 0.15
        
        # Scale curvature with volatility intensity (more pronounced for high intensity)
        curve_intensity_factor = 1.0 + 0.5 * vol_intensity**1.5
        curvature_strength *= curve_intensity_factor
        
        curvature_effect = curvature_strength * moneyness**2
        
        # 3. Apply wing effects with continuous intensity scaling
        wing_left = np.zeros_like(moneyness)
        wing_right = np.zeros_like(moneyness)
        
        # Left wing (put side)
        left_mask = moneyness < -0.05
        if np.any(left_mask):
            wing_strength = 0.15 * np.power(np.abs(moneyness[left_mask] + 0.05), 1.5) * np.exp(-0.5 * t)
            # Scale wing effect with intensity
            wing_intensity = 0.8 + 0.4 * vol_intensity
            wing_left[left_mask] = wing_strength * wing_intensity
        
        # Right wing (call side)
        right_mask = moneyness > 0.05
        if np.any(right_mask):
            wing_strength = 0.1 * np.power(np.abs(moneyness[right_mask] - 0.05), 1.5) * np.exp(-0.5 * t)
            # Scale wing effect with intensity
            wing_intensity = 0.8 + 0.3 * vol_intensity
            wing_right[right_mask] = wing_strength * wing_intensity * 0.5  # Asymmetric effect
        
        # Asymmetry based on rho
        wing_effect = -1.0 * wing_left + wing_right
        
        # 4. Combine all effects
        total_effect = 1.0 + skew_effect + curvature_effect + wing_effect
        
        # Apply to base volatility - multiplicative effect preserves proportionality
        surface_slice = base_vol * total_effect
        
        # --- Fine-tune with known values if available ---
        skew_windows = []
        curve_windows = []
        
        # Collect skew windows
        for col in stock_cols:
            if col.startswith('VolSkew_') and col.endswith(pred_suffix):
                try:
                    win = int(col.replace('VolSkew_', '').replace(pred_suffix, ''))
                    skew_windows.append(win)
                except ValueError:
                    continue
        
        # Collect curve windows
        for col in stock_cols:
            if col.startswith('VolCurvature_') and col.endswith(pred_suffix):
                try:
                    win = int(col.replace('VolCurvature_', '').replace(pred_suffix, ''))
                    curve_windows.append(win)
                except ValueError:
                    continue
        
        if skew_windows and curve_windows:
            closest_skew_win = min(skew_windows, key=lambda x: abs(x - days))
            closest_curve_win = min(curve_windows, key=lambda x: abs(x - days))
            
            skew_col = f'VolSkew_{closest_skew_win}{pred_suffix}'
            curve_col = f'VolCurvature_{closest_curve_win}{pred_suffix}'
            
            if skew_col in stock_cols and curve_col in stock_cols:
                skew_idx = stock_cols.index(skew_col)
                curve_idx = stock_cols.index(curve_col)
                target_skew = stock_row[skew_idx]
                target_curve = max(0, min(3.0, stock_row[curve_idx]))
                
                atm_idx = np.abs(moneyness).argmin()
                down_idx = np.abs(moneyness + 0.05).argmin()
                up_idx = np.abs(moneyness - 0.05).argmin()
                
                current_skew = (surface_slice[down_idx] - surface_slice[up_idx]) / 0.1
                current_curve = (surface_slice[down_idx] + surface_slice[up_idx] - 2 * surface_slice[atm_idx]) / 0.05**2
                
                # Scale adjustments with intensity for more pronounced effects with high intensity
                skew_adjust_factor = 0.2 * (1.0 + 0.3 * vol_intensity)
                curve_adjust_factor = 0.07 * (1.0 + 0.4 * vol_intensity)
                
                # Calculate adjustments
                skew_adjust = (target_skew - current_skew) * skew_adjust_factor * moneyness
                curve_adjust = (target_curve - current_curve) * curve_adjust_factor * moneyness**2
                
                # Scale the max adjustments with base volatility to maintain proportionality
                max_skew_adjust = 0.15 * base_vol
                max_curve_adjust = 0.07 * base_vol
                
                skew_adjust = np.clip(skew_adjust, -max_skew_adjust, max_skew_adjust)
                curve_adjust = np.clip(curve_adjust, -max_curve_adjust, max_curve_adjust)
                
                # Apply adjustments additively
                surface_slice = surface_slice + skew_adjust + curve_adjust
        
        # Apply bounds proportional to the base volatility
        surface_slice = np.clip(surface_slice, 0.6 * base_vol, 1.4 * base_vol)
        Z[i, :] = surface_slice
    
    # --- Smoothing across term structure ---
    for j in range(len(moneyness)):
        Z[:, j] = savgol_smooth(Z[:, j])
    
    # Create meshgrids for output
    strike_grid, days_grid = np.meshgrid(moneyness, time_days)
    
    # Return surface data dictionary
    return {
        "type": "model_prediction",
        "subtype": "YZVol",
        "x": strike_grid,
        "y": days_grid,
        "z": Z,
        "x_label": "Moneyness (log(K/S))",
        "y_label": "Forecast Horizon (Days)",
        "z_label": "Predicted Volatility",
        "title": f"{stock} Model Predicted Volatility Surface on {date}",
        "description": "Model predicted future realized volatility",
        "date": date,
        "windows": windows,
        "pred_values": atm_vols
    }

def create_realized_vol_surface(
    df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    windows: Optional[List[int]] = None,
    vol_scale: float = 1.0,
    future_suffix: str = "_future"
) -> Dict:
    """
    Create volatility surface from realized future values using continuous volatility
    intensity and proper skew scaling.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with realized values
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    windows : List[int], optional
        List of time windows to include
    vol_scale : float
        Scaling factor for volatility values
    future_suffix : str
        Suffix for future realized columns
        
    Returns:
    --------
    Dict
        Dictionary with surface data and metadata
    """
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Filter for stock and date
    stock_data = df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") == date)
    )
    
    if stock_data.height == 0:
        raise ValueError(f"No data found for {stock} on {date}")
    
    # Extract stock data row for direct indexing like original code
    stock_cols = stock_data.columns
    stock_row = stock_data.row(0)
    
    # Determine available windows if not specified
    if windows is None:
        # Use the same function from original code
        available_windows = []
        for col in stock_cols:
            if col.startswith("YZVol_") and col.endswith(future_suffix):
                try:
                    window_str = col.replace("YZVol_", "").replace(future_suffix, "")
                    window = int(window_str)
                    available_windows.append(window)
                except ValueError:
                    continue
        
        windows = sorted(available_windows)
    
    if not windows:
        raise ValueError(f"No realized volatility windows found with suffix '{future_suffix}'")
    
    # Convert windows to days and years exactly like original code
    time_days = windows
    times = np.array([w/252 for w in windows])  # Convert to years

    # Moneyness values (log(K/S))
    moneyness = np.linspace(-0.15, 0.15, 11)
    
    # Create output grid - swap dimensions to make moneyness first and time second
    Z = np.zeros((len(times), len(moneyness)))
    
    # --- Get term structure of volatility (ATM volatilities) ---
    atm_vols = []
    for days in time_days:
        vol_windows = []
        for col in stock_cols:
            if col.startswith('YZVol_') and not col.endswith('_pred'):
                try:
                    win = int(col.replace('YZVol_', '').replace(future_suffix, ''))
                    vol_windows.append(win)
                except ValueError:
                    continue
        
        if not vol_windows:
            atm_vols.append(0.25)  # Default if no data
            continue
            
        closest_win = min(vol_windows, key=lambda x: abs(x - days))
        future_col = f'YZVol_{closest_win}{future_suffix}'
        
        if future_col in stock_cols:
            col_idx = stock_cols.index(future_col)
            atm_vols.append(stock_row[col_idx] * vol_scale)
        else:
            # Use current volatility as fallback
            current_col = f'YZVol_{closest_win}' 
            if current_col in stock_cols:
                col_idx = stock_cols.index(current_col)
                atm_vols.append(stock_row[col_idx] * vol_scale)
            else:
                atm_vols.append(0.25 * vol_scale)  # Default if no data
    
    # Smooth the ATM volatility curve if we have enough points
    if len(atm_vols) >= 4:
        # Convert to numpy for easier manipulation
        x_days = np.array(time_days)
        y_vols = np.array(atm_vols)
        
        # Simple smoothing using rolling average
        window_size = 3
        weights = np.ones(window_size) / window_size
        # Pad the ends to avoid edge effects
        padded_vols = np.pad(y_vols, (window_size//2, window_size//2), mode='edge')
        smoothed_vols = np.convolve(padded_vols, weights, mode='valid')
        
        # Use the smoothed values but keep original values at endpoints
        atm_vols = list(smoothed_vols)
    
    # Scale the volatility range to be more reasonable
    # Find min and max volatility
    min_vol = min(atm_vols)
    max_vol = max(atm_vols)
    
    # If the range is too large, compress it
    if max_vol / min_vol > 4.0:
        # Compress the range by bringing extremes closer to the mean
        mean_vol = np.mean(atm_vols)
        compression_factor = 0.6  # Lower values = more compression
        atm_vols = [mean_vol + compression_factor * (vol - mean_vol) for vol in atm_vols]
    
    # --- Core model parameters ---
    # Get key Heston parameters from a medium-term window for base model
    medium_term_idx = len(time_days) // 2
    medium_term_days = time_days[medium_term_idx]
    medium_term_t = times[medium_term_idx]
    
    # Get mean reversion parameter (with adjustments for realized surface)
    mr_windows = []
    for col in stock_cols:
        if col.startswith('MeanReversion_') and col.endswith(future_suffix):
            try:
                win = int(col.replace('MeanReversion_', '').replace(future_suffix, ''))
                mr_windows.append(win)
            except ValueError:
                continue
    
    if mr_windows:
        closest_mr_win = min(mr_windows, key=lambda x: abs(x - medium_term_days))
        mr_col = f'MeanReversion_{closest_mr_win}{future_suffix}'
        if mr_col in stock_cols:
            col_idx = stock_cols.index(mr_col)
            base_kappa = stock_row[col_idx] * 0.9  # Reduce a bit for realized
        else:
            base_kappa = 1.5 * 0.9
    else:
        base_kappa = 1.5 * 0.9
        
    # Volatility of volatility (with adjustments for realized surface)
    vv_windows = []
    for col in stock_cols:
        if col.startswith('VolOfVol_') and col.endswith(future_suffix):
            try:
                win = int(col.replace('VolOfVol_', '').replace(future_suffix, ''))
                vv_windows.append(win)
            except ValueError:
                continue
    
    if vv_windows:
        closest_vv_win = min(vv_windows, key=lambda x: abs(x - medium_term_days))
        vv_col = f'VolOfVol_{closest_vv_win}{future_suffix}'
        if vv_col in stock_cols:
            col_idx = stock_cols.index(vv_col)
            base_volvol = stock_row[col_idx] * 0.85  # Reduce for realized
        else:
            base_volvol = 0.3 * 0.85
    else:
        base_volvol = 0.3 * 0.85
        
    # Price-vol correlation (with adjustments for realized surface)
    corr_windows = []
    for col in stock_cols:
        if col.startswith('PriceVolCorr_') and col.endswith(future_suffix):
            try:
                win = int(col.replace('PriceVolCorr_', '').replace(future_suffix, ''))
                corr_windows.append(win)
            except ValueError:
                continue
    
    if corr_windows:
        closest_corr_win = min(corr_windows, key=lambda x: abs(x - medium_term_days))
        corr_col = f'PriceVolCorr_{closest_corr_win}{future_suffix}'
        if corr_col in stock_cols:
            col_idx = stock_cols.index(corr_col)
            base_rho = stock_row[col_idx] * 0.9  # Reduce for realized
        else:
            base_rho = -0.5 * 0.9
    else:
        base_rho = -0.5 * 0.9
    
    # Fix-up parameters for numerical stability
    base_kappa = np.clip(base_kappa, 0.1, 4.5)
    base_volvol = np.clip(base_volvol, 0.01, 0.8)
    base_rho = np.clip(base_rho, -0.85, 0.4)
    
    # Scale down vol-of-vol to reduce the overall volatility range
    base_volvol *= 0.7  # Reduce by 30%
    
    # Process each time slice with term-structure consistency
    for i, (t, days) in enumerate(zip(times, time_days)):
        # Get tenor-specific volatility intensity (continuous value scaled 0 to 1)
        vol_intensity = 0.5  # Default
        intensity_cols = []
        for col in stock_cols:
            if col.startswith('VolIntensity_') and col.endswith(future_suffix):
                try:
                    win = int(col.replace('VolIntensity_', '').replace(future_suffix, ''))
                    intensity_cols.append((win, col))
                except ValueError:
                    continue
        
        if intensity_cols:
            closest_win = min(intensity_cols, key=lambda x: abs(x[0] - days))[0]
            for win, col in intensity_cols:
                if win == closest_win:
                    col_idx = stock_cols.index(col)
                    vol_intensity = stock_row[col_idx]
                    break
        
        # Get the ATM volatility for this tenor
        atm_vol = atm_vols[i]
        
        # IMPROVED: Use continuous functions of vol_intensity for realized surface
        # Parameters are more muted compared to model surface
        
        # 1. Tenor-based adjustments using continuous functions
        # Short-term factor: higher for short tenors, approaches 1.0 for longer tenors
        tenor_factor = 1.0 + 0.4 * np.exp(-5.5 * t)  # Slightly milder than model
        
        # 2. Continuous intensity-based adjustments
        # Linear component: increases from min to max as intensity goes from 0 to 1
        intensity_linear = 0.88 + 0.24 * vol_intensity  # More muted range
        
        # Nonlinear component: accelerates as intensity increases (milder than model)
        intensity_nonlinear = vol_intensity ** 1.8  # Less aggressive
        
        # 3. Apply adjustments to parameters - more muted for realized
        # For kappa (mean reversion)
        kappa_adj = 1.0 + 0.25 * (vol_intensity ** 1.1)  # Slightly milder
        kappa = base_kappa * kappa_adj
        
        # For volvol (volatility of volatility)
        volvol_tenor_adj = 1.0 - 0.12 * (1.0 - np.exp(-3.0 * t))
        volvol_intensity_adj = intensity_linear * (1.0 + 0.15 * intensity_nonlinear)
        volvol = base_volvol * volvol_tenor_adj * volvol_intensity_adj
        
        # For rho (correlation)
        rho_tenor_adj = 1.0 + 0.08 * np.exp(-4.0 * t)
        rho_intensity_adj = 1.0 + 0.25 * vol_intensity + 0.15 * vol_intensity**2
        rho = base_rho * rho_tenor_adj * rho_intensity_adj
        
        # Ensure parameters remain within bounds
        kappa = np.clip(kappa, 0.1, 8.0)
        volvol = np.clip(volvol, 0.01, 1.5)
        rho = np.clip(rho, -0.9, 0.6)
        
        # Convert volatility to variance (v = σ²)
        v0 = (atm_vol / vol_scale)**2
        
        # Approximate long-term variance based on term structure
        if i < len(times) - 1:
            forward_vol = atm_vols[min(i + 1, len(atm_vols) - 1)] / vol_scale
            theta = forward_vol**2
        else:
            theta = ((atm_vol / vol_scale) * 0.98)**2  # More gradual decay for realized
        
        theta = max(0.0004, theta)
        
        # Apply term structure effect with mean-reversion
        v_t = v0 + (theta - v0) * (1 - np.exp(-kappa * t))
        base_vol = np.sqrt(v_t) * vol_scale
        
        surface_slice = np.ones_like(moneyness) * base_vol
        
        # IMPROVED: Enhance skew calculation to properly scale with vol_scale
        # Use more muted effects for realized surface
        
        # 1. Calculate skew effect with proper scaling
        skew_strength = np.sqrt(t) * (1 - np.exp(-kappa * t)) / (kappa * t) * 0.25  # Reduced vs model
        
        # Scale skew with volatility intensity (milder than model)
        skew_intensity_factor = 1.0 + 0.3 * vol_intensity
        skew_strength *= skew_intensity_factor
        
        # Calculate basic skew effect
        skew_effect = rho * volvol * moneyness * skew_strength
        
        # 2. Calculate curvature effect with proper scaling
        curvature_strength = (1 - rho**2) * volvol**2 * t * np.exp(-kappa * t) * 0.12  # Reduced vs model
        
        # Scale curvature with volatility intensity (more muted than model)
        curve_intensity_factor = 1.0 + 0.4 * vol_intensity**1.4
        curvature_strength *= curve_intensity_factor
        
        curvature_effect = curvature_strength * moneyness**2
        
        # 3. Apply wing effects with continuous intensity scaling
        wing_left = np.zeros_like(moneyness)
        wing_right = np.zeros_like(moneyness)
        
        # Left wing (put side) - more muted for realized
        left_mask = moneyness < -0.05
        if np.any(left_mask):
            wing_strength = 0.12 * np.power(np.abs(moneyness[left_mask] + 0.05), 1.5) * np.exp(-0.6 * t)
            # Scale wing effect with intensity
            wing_intensity = 0.75 + 0.3 * vol_intensity
            wing_left[left_mask] = wing_strength * wing_intensity * 0.8  # 20% reduction vs model
        
        # Right wing (call side) - more muted for realized
        right_mask = moneyness > 0.05
        if np.any(right_mask):
            wing_strength = 0.08 * np.power(np.abs(moneyness[right_mask] - 0.05), 1.5) * np.exp(-0.6 * t)
            # Scale wing effect with intensity
            wing_intensity = 0.75 + 0.25 * vol_intensity
            wing_right[right_mask] = wing_strength * wing_intensity * 0.4  # More muted asymmetric effect
        
        # Asymmetry based on rho
        wing_effect = -1.0 * wing_left + wing_right
        
        # 4. Combine all effects
        total_effect = 1.0 + skew_effect + curvature_effect + wing_effect
        
        # Apply to base volatility - multiplicative effect preserves proportionality
        surface_slice = base_vol * total_effect
        
        # --- Fine-tune with known values if available ---
        skew_windows = []
        curve_windows = []
        
        # Collect skew windows
        for col in stock_cols:
            if col.startswith('VolSkew_') and col.endswith(future_suffix):
                try:
                    win = int(col.replace('VolSkew_', '').replace(future_suffix, ''))
                    skew_windows.append(win)
                except ValueError:
                    continue
        
        # Collect curve windows
        for col in stock_cols:
            if col.startswith('VolCurvature_') and col.endswith(future_suffix):
                try:
                    win = int(col.replace('VolCurvature_', '').replace(future_suffix, ''))
                    curve_windows.append(win)
                except ValueError:
                    continue
        
        if skew_windows and curve_windows:
            closest_skew_win = min(skew_windows, key=lambda x: abs(x - days))
            closest_curve_win = min(curve_windows, key=lambda x: abs(x - days))
            
            skew_col = f'VolSkew_{closest_skew_win}{future_suffix}'
            curve_col = f'VolCurvature_{closest_curve_win}{future_suffix}'
            
            if skew_col in stock_cols and curve_col in stock_cols:
                skew_idx = stock_cols.index(skew_col)
                curve_idx = stock_cols.index(curve_col)
                target_skew = stock_row[skew_idx] * 0.85  # Reduce for realized
                target_curve = max(0, min(2.5, stock_row[curve_idx] * 0.8))  # Reduce for realized
                
                atm_idx = np.abs(moneyness).argmin()
                down_idx = np.abs(moneyness + 0.05).argmin()
                up_idx = np.abs(moneyness - 0.05).argmin()
                
                current_skew = (surface_slice[down_idx] - surface_slice[up_idx]) / 0.1
                current_curve = (surface_slice[down_idx] + surface_slice[up_idx] - 2 * surface_slice[atm_idx]) / 0.05**2
                
                # Scale adjustments with intensity (more muted than model)
                skew_adjust_factor = 0.18 * (1.0 + 0.25 * vol_intensity)
                curve_adjust_factor = 0.06 * (1.0 + 0.3 * vol_intensity)
                
                # Calculate adjustments
                skew_adjust = (target_skew - current_skew) * skew_adjust_factor * moneyness
                curve_adjust = (target_curve - current_curve) * curve_adjust_factor * moneyness**2
                
                # Scale the max adjustments with base volatility to maintain proportionality
                max_skew_adjust = 0.13 * base_vol  # Reduced vs model
                max_curve_adjust = 0.06 * base_vol  # Reduced vs model
                
                skew_adjust = np.clip(skew_adjust, -max_skew_adjust, max_skew_adjust)
                curve_adjust = np.clip(curve_adjust, -max_curve_adjust, max_curve_adjust)
                
                # Apply adjustments additively
                surface_slice = surface_slice + skew_adjust + curve_adjust
        
        # Apply tighter bounds for realized volatility
        surface_slice = np.clip(surface_slice, 0.7 * base_vol, 1.3 * base_vol)
        Z[i, :] = surface_slice
    
    # --- Smoothing across term structure ---
    for j in range(len(moneyness)):
        Z[:, j] = savgol_smooth(Z[:, j])
    
    # Create meshgrids for output
    strike_grid, days_grid = np.meshgrid(moneyness, time_days)
    
    # Return surface data dictionary
    return {
        "type": "realized",
        "subtype": "YZVol",
        "x": strike_grid,
        "y": days_grid,
        "z": Z,
        "x_label": "Moneyness (log(K/S))",
        "y_label": "Future Period (Days)",
        "z_label": "Realized Volatility",
        "title": f"{stock} Realized Future Volatility Surface from {date}",
        "description": "Realized future volatility (Yang-Zhang)",
        "date": date,
        "windows": windows,
        "realized_values": atm_vols
    }

def savgol_smooth(y, window_length=5, polyorder=2):
    """
    Apply a Savitzky-Golay filter to smooth data.
    Falls back to simpler moving average if not enough points.
    """
    if len(y) < window_length:
        if len(y) <= 3:
            return y
        
        window_size = min(3, len(y) - 2)
        weights = np.ones(window_size) / window_size
        padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
        smoothed = np.convolve(padded, weights, mode='valid')
        return smoothed
    
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length, polyorder)
    except ImportError:
        window_size = min(window_length, len(y) - 2)
        weights = np.ones(window_size) / window_size
        padded = np.pad(y, (window_size//2, window_size//2), mode='edge')
        return np.convolve(padded, weights, mode='valid')

def create_volatility_visualization_dashboard(
    option_df: Optional[pl.DataFrame] = None, 
    model_df: Optional[pl.DataFrame] = None,
    realized_df: Optional[pl.DataFrame] = None,
    stock: str = None,
    date: Union[str, datetime.date] = None,
    option_type: str = 'Call',
    windows: Optional[List[int]] = None,
    vol_scale_factors: Dict[str, float] = {'market': 1.0, 'model': 10.0, 'realized': 10.0},
    vol_range: Tuple[float, float] = (0.0, 0.7)  # Fixed volatility range
) -> go.Figure:
    """
    Create a comprehensive dashboard-style visualization with consistent volatility scaling.
    
    Parameters:
    -----------
    option_df : pl.DataFrame, optional
        DataFrame with option chain data
    model_df : pl.DataFrame, optional
        DataFrame with model predictions
    realized_df : pl.DataFrame, optional
        DataFrame with realized values
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    option_type : str
        Option type to analyze ('Call' or 'Put')
    windows : List[int], optional
        List of specific time windows to include
    vol_scale_factors : Dict[str, float]
        Scaling factors for different volatility types
    vol_range : Tuple[float, float]
        Fixed range for volatility axis (min, max)
        
    Returns:
    --------
    go.Figure
        Plotly dashboard figure with multiple volatility visualizations
    """
    # Verify we have at least one data source
    if option_df is None and model_df is None and realized_df is None:
        raise ValueError("At least one data source (option_df, model_df, or realized_df) must be provided")
    
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Create necessary surfaces
    surfaces = {}
    
    if option_df is not None:
        try:
            surfaces['market'] = create_option_vol_surface(
                option_df=option_df,
                stock=stock,
                date=date,
                option_type=option_type,
                vol_scale=vol_scale_factors.get('market', 1.0)
            )
        except Exception as e:
            print(f"Could not create market surface: {e}")
    
    if model_df is not None:
        try:
            surfaces['model'] = create_model_vol_surface(
                pred_df=model_df,
                stock=stock,
                date=date,
                windows=windows,
                vol_scale=vol_scale_factors.get('model', 10.0)
            )
        except Exception as e:
            print(f"Could not create model surface: {e}")
    
    if realized_df is not None:
        try:
            surfaces['realized'] = create_realized_vol_surface(
                df=realized_df,
                stock=stock,
                date=date,
                windows=windows,
                vol_scale=vol_scale_factors.get('realized', 10.0)
            )
        except Exception as e:
            print(f"Could not create realized surface: {e}")
    
    if not surfaces:
        raise ValueError("Could not create any volatility surfaces")
    
    # Create dashboard layout based on the number of surfaces
    n_surfaces = len(surfaces)
    
    # Create the appropriate subplot specs
    if n_surfaces == 1:
        # Single surface - create a 2x2 layout with surface, term structure, skew, and parameters
        n_rows = 2
        n_cols = 2
        specs = [
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
        subplot_titles = [
            "Volatility Surface", 
            "Term Structure (ATM)",
            "Volatility Smile",
            "Surface Parameters"
        ]
        
    elif n_surfaces == 2:
        # Two surfaces - create a 2x2 layout with surfaces on top and comparisons below
        n_rows = 2
        n_cols = 2
        specs = [
            [{"type": "scene"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
        
        # Get the surface types
        surface_types = list(surfaces.keys())
        subplot_titles = [
            f"{surface_types[0].capitalize()} Volatility Surface", 
            f"{surface_types[1].capitalize()} Volatility Surface",
            "Term Structure Comparison",
            "Volatility Skew Comparison"
        ]
        
    elif n_surfaces == 3:
        # Three surfaces - create a 2x3 layout with surfaces on top and comparisons below
        n_rows = 2
        n_cols = 3
        specs = [
            [{"type": "scene"}, {"type": "scene"}, {"type": "scene"}],
            [{"type": "xy", "colspan": 3}, None, None]
        ]
        
        subplot_titles = [
            "Market Volatility Surface", 
            "Model Predicted Volatility Surface",
            "Realized Volatility Surface",
            "Term Structure & Skew Comparison"
        ]
        
    else:
        # Default case - shouldn't happen but just in case
        n_rows = 1
        n_cols = 1
        specs = [[{"type": "scene"}]]
        subplot_titles = ["Volatility Surface"]
    
    # Create subplot
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        specs=specs,
        subplot_titles=subplot_titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.05
    )
    
    # Add each surface
    colorscales = {
        'market': 'Viridis',
        'model': 'Plasma',
        'realized': 'Cividis'
    }
    
    # Track the locations for each surface to add plots
    surface_locations = {}
    
    # Add 3D surfaces
    if n_surfaces == 1:
        # Single surface case
        surface_type = list(surfaces.keys())[0]
        surface_data = surfaces[surface_type]
        
        X, Y = np.meshgrid(np.unique(surface_data["x"]), np.unique(surface_data["y"]))
        Z = surface_data["z"]
        
        fig.add_trace(
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale=colorscales.get(surface_type, 'Viridis'),
                colorbar=dict(
                    title=surface_data["z_label"],
                    len=0.5,
                    y=0.8,
                    cmin=vol_range[0],
                    cmax=vol_range[1]
                ),
                cmin=vol_range[0],
                cmax=vol_range[1],
                lighting=dict(ambient=0.8, diffuse=0.2, roughness=0.9, specular=0.1, fresnel=0),
                contours={
                    "x": {"show": True, "color":"black", "width": 1},
                    "y": {"show": True, "color":"black", "width": 1},
                    "z": {"show": True, "color":"black", "width": 1}
                }
            ),
            row=1, col=1
        )
        
        surface_locations[surface_type] = (1, 1)
        
        # Add term structure plot
        atm_idx = len(np.unique(surface_data["x"])) // 2
        y_vals = np.unique(surface_data["y"])
        atm_vols = Z[:, atm_idx]
        
        # Apply cubic spline for smoothing if we have enough points
        if len(y_vals) >= 4:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(y_vals, atm_vols)
            smooth_days = np.linspace(min(y_vals), max(y_vals), 100)
            smooth_vols = cs(smooth_days)
            
            # Plot the smoothed line
            fig.add_trace(
                go.Scatter(
                    x=smooth_days,
                    y=smooth_vols,
                    mode='lines',
                    name=f'{surface_type.capitalize()} ATM (Smoothed)',
                    line=dict(color='blue', width=2.5)
                ),
                row=1, col=2
            )
            
            # Add the original points
            fig.add_trace(
                go.Scatter(
                    x=y_vals,
                    y=atm_vols,
                    mode='markers',
                    name=f'{surface_type.capitalize()} ATM (Data)',
                    marker=dict(color='blue', size=8, symbol='circle')
                ),
                row=1, col=2
            )
        else:
            # Just plot the line if we don't have enough points for spline
            fig.add_trace(
                go.Scatter(
                    x=y_vals,
                    y=atm_vols,
                    mode='lines+markers',
                    name=f'{surface_type.capitalize()} ATM Vol',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=1, col=2
            )
        
        # Set fixed y-axis range for term structure plot
        fig.update_yaxes(range=vol_range, row=1, col=2)
        
        # Add skew plot - select a few representative days
        x_vals = np.unique(surface_data["x"])
        n_curves = min(5, len(y_vals))
        selected_indices = np.linspace(0, len(y_vals)-1, n_curves).astype(int)
        selected_days = [y_vals[i] for i in selected_indices]
        
        # Create a viridis-like color palette
        colors = [
            'rgb(68, 1, 84)',    # Dark purple
            'rgb(59, 82, 139)',   # Blue
            'rgb(33, 144, 140)',  # Teal
            'rgb(93, 201, 99)',   # Green
            'rgb(253, 231, 37)'   # Yellow
        ]
        
        # Ensure we don't try to use more colors than we have
        colors = colors[:n_curves]
        
        for i, day_idx in enumerate(selected_indices):
            day = y_vals[day_idx]
            smile = Z[day_idx, :]
            
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=smile,
                    mode='lines',
                    name=f'{day:.0f}d',
                    line=dict(color=colors[i], width=2)
                ),
                row=2, col=1
            )
        
        # Set fixed y-axis range for skew plot
        fig.update_yaxes(range=vol_range, row=2, col=1)
        
        # Add parameter summary table
        params_text = f"<b>{surface_type.capitalize()} Volatility Surface Parameters:</b><br><br>"
        
        if 'windows' in surface_data:
            windows = surface_data.get('windows', [])
            values = surface_data.get('pred_values' if surface_type == 'model' else 'realized_values', [])
            
            params_text += "<b>Term Structure:</b><br>"
            if len(windows) > 0 and len(values) > 0:
                term_items = []
                for w, v in zip(windows, values):
                    if not np.isnan(v):
                        term_items.append(f"{w}d: {v:.4f}")
                        
                params_text += ", ".join(term_items)
        
        # Add as annotation in the last cell
        fig.add_annotation(
            text=params_text,
            x=0.5, y=0.5,
            xref="x4", yref="y4",
            showarrow=False,
            font=dict(size=11),
            align="center",
            bordercolor="gray",
            borderwidth=1,
            borderpad=5,
            bgcolor="white",
            opacity=0.8
        )
        
        # Remove axes from the parameter cell
        fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)
        fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=2, col=2)
        
        # Set fixed z-axis range for 3D plot
        fig.update_scenes(
            zaxis=dict(range=vol_range),
            row=1, col=1
        )
        
    else:
        # Multiple surfaces case
        for i, (surface_type, surface_data) in enumerate(surfaces.items()):
            if n_surfaces == 2:
                row, col = 1, i + 1
            elif n_surfaces == 3:
                row, col = 1, i + 1
            else:
                # Default placement
                row, col = (i // n_cols) + 1, (i % n_cols) + 1
                
            surface_locations[surface_type] = (row, col)
            
            X, Y = np.meshgrid(np.unique(surface_data["x"]), np.unique(surface_data["y"]))
            Z = surface_data["z"]
            
            fig.add_trace(
                go.Surface(
                    z=Z,
                    x=X,
                    y=Y,
                    colorscale=colorscales.get(surface_type, 'Viridis'),
                    colorbar=dict(
                        title=surface_data["z_label"],
                        len=0.5,
                        y=0.8 - 0.3 * (i % 3),
                        cmin=vol_range[0],
                        cmax=vol_range[1]
                    ),
                    cmin=vol_range[0],
                    cmax=vol_range[1],
                    lighting=dict(ambient=0.8, diffuse=0.2, roughness=0.9, specular=0.1, fresnel=0),
                    contours={
                        "x": {"show": True, "color":"black", "width": 1},
                        "y": {"show": True, "color":"black", "width": 1},
                        "z": {"show": True, "color":"black", "width": 1}
                    },
                    name=f"{surface_type.capitalize()}"
                ),
                row=row, col=col
            )
            
            # Update scene settings with fixed z-axis range
            fig.update_scenes(
                xaxis_title=surface_data["x_label"],
                yaxis_title=surface_data["y_label"],
                zaxis_title=surface_data["z_label"],
                zaxis=dict(range=vol_range),
                aspectratio=dict(x=1.2, y=1.0, z=0.7),
                camera=dict(eye=dict(x=1.5, y=-1.5, z=1)),
                row=row, col=col
            )
        
        # Add comparison plots for multiple surfaces
        if n_surfaces == 2:
            # Two surfaces - Add term structure comparison
            colors = {'market': 'blue', 'model': 'red', 'realized': 'green'}
            comparison_row, comparison_col = 2, 1
            
            for surface_type, surface_data in surfaces.items():
                # Extract at-the-money volatilities
                atm_idx = len(np.unique(surface_data["x"])) // 2
                y_vals = np.unique(surface_data["y"])
                Z = surface_data["z"]
                atm_vols = Z[:, atm_idx]
                
                # Apply cubic spline if we have enough points
                if len(y_vals) >= 4:
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(y_vals, atm_vols)
                    smooth_days = np.linspace(min(y_vals), max(y_vals), 100)
                    smooth_vols = cs(smooth_days)
                    
                    # Plot the smoothed line
                    fig.add_trace(
                        go.Scatter(
                            x=smooth_days,
                            y=smooth_vols,
                            mode='lines',
                            name=f'{surface_type.capitalize()} ATM (Smoothed)',
                            line=dict(color=colors.get(surface_type, 'purple'), width=2.5)
                        ),
                        row=comparison_row, col=comparison_col
                    )
                    
                    # Add the original points
                    fig.add_trace(
                        go.Scatter(
                            x=y_vals,
                            y=atm_vols,
                            mode='markers',
                            name=f'{surface_type.capitalize()} ATM (Data)',
                            marker=dict(color=colors.get(surface_type, 'purple'), size=8, symbol='circle'),
                            showlegend=False
                        ),
                        row=comparison_row, col=comparison_col
                    )
                else:
                    # Just plot the line if we don't have enough points for spline
                    fig.add_trace(
                        go.Scatter(
                            x=y_vals,
                            y=atm_vols,
                            mode='lines+markers',
                            name=f'{surface_type.capitalize()} ATM Vol',
                            line=dict(color=colors.get(surface_type, 'purple'), width=2),
                            marker=dict(size=8)
                        ),
                        row=comparison_row, col=comparison_col
                    )
            
            # Set fixed y-axis range for term structure comparison
            fig.update_yaxes(range=vol_range, row=comparison_row, col=comparison_col)
            
            # Add skew comparison for a specific tenor
            # Choose a common tenor (days to expiration) if possible
            comparison_row, comparison_col = 2, 2
            common_tenors = []
            
            for surface_data in surfaces.values():
                common_tenors.extend(np.unique(surface_data["y"]))
            
            common_tenor = np.median(common_tenors)
            
            # Add skew lines for each surface
            for surface_type, surface_data in surfaces.items():
                x_vals = np.unique(surface_data["x"])
                y_vals = np.unique(surface_data["y"])
                Z = surface_data["z"]
                
                # Find closest tenor
                closest_tenor_idx = np.abs(y_vals - common_tenor).argmin()
                closest_tenor = y_vals[closest_tenor_idx]
                
                # Extract smile for this tenor
                smile = Z[closest_tenor_idx, :]
                
                fig.add_trace(
                    go.Scatter(
                        x=x_vals,
                        y=smile,
                        mode='lines',
                        name=f'{surface_type.capitalize()} ({closest_tenor:.0f}d)',
                        line=dict(color=colors.get(surface_type, 'purple'), width=2)
                    ),
                    row=comparison_row, col=comparison_col
                )
                
                # Add vertical line at ATM point
                atm_point = 0.0 if "Moneyness" in surface_data["x_label"] else np.median(x_vals)
                fig.add_shape(
                    type="line",
                    x0=atm_point, y0=0,
                    x1=atm_point, y1=1,
                    yref="paper",
                    line=dict(color="black", width=1, dash="dash"),
                    row=comparison_row, col=comparison_col
                )
            
            # Set fixed y-axis range for skew comparison
            fig.update_yaxes(range=vol_range, row=comparison_row, col=comparison_col)
        
        elif n_surfaces == 3:
            # Three surfaces - Add combined comparison plot
            colors = {'market': 'blue', 'model': 'red', 'realized': 'green'}
            comparison_row, comparison_col = 2, 1
            
            # First add term structure comparison
            for surface_type, surface_data in surfaces.items():
                # Extract at-the-money volatilities
                atm_idx = len(np.unique(surface_data["x"])) // 2
                y_vals = np.unique(surface_data["y"])
                Z = surface_data["z"]
                atm_vols = Z[:, atm_idx]
                
                # Apply cubic spline if we have enough points
                if len(y_vals) >= 4:
                    from scipy.interpolate import CubicSpline
                    cs = CubicSpline(y_vals, atm_vols)
                    smooth_days = np.linspace(min(y_vals), max(y_vals), 100)
                    smooth_vols = cs(smooth_days)
                    
                    # Plot the smoothed line
                    fig.add_trace(
                        go.Scatter(
                            x=smooth_days,
                            y=smooth_vols,
                            mode='lines',
                            name=f'{surface_type.capitalize()} Term Structure',
                            line=dict(color=colors.get(surface_type, 'purple'), width=2)
                        ),
                        row=comparison_row, col=comparison_col
                    )
                else:
                    # Just plot the line if we don't have enough points for spline
                    fig.add_trace(
                        go.Scatter(
                            x=y_vals,
                            y=atm_vols,
                            mode='lines+markers',
                            name=f'{surface_type.capitalize()} Term Structure',
                            line=dict(color=colors.get(surface_type, 'purple'), width=2),
                            marker=dict(size=6)
                        ),
                        row=comparison_row, col=comparison_col
                    )
            
            # Set fixed y-axis range for term structure comparison
            fig.update_yaxes(range=vol_range, row=comparison_row, col=comparison_col)
            
            # Add vertical separator between term structure and skew
            fig.add_shape(
                type="line",
                x0=0.5, y0=0,
                x1=0.5, y1=1,
                xref="paper", yref="paper",
                line=dict(color="black", width=1, dash="dash"),
                row=comparison_row, col=comparison_col
            )
            
            # Add annotations to clarify the sections
            fig.add_annotation(
                text="Term Structure Comparison",
                x=0.25, y=1.05,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12),
                row=comparison_row, col=comparison_col
            )
            
            fig.add_annotation(
                text="Volatility Skew Comparison",
                x=0.75, y=1.05,
                xref="paper", yref="paper",
                showarrow=False,
                font=dict(size=12),
                row=comparison_row, col=comparison_col
            )
            
            # Choose a common tenor for skew comparison
            common_tenors = []
            for surface_data in surfaces.values():
                common_tenors.extend(np.unique(surface_data["y"]))
            
            common_tenor = np.median(common_tenors)
            
            # Add skew comparison
            for surface_type, surface_data in surfaces.items():
                x_vals = np.unique(surface_data["x"])
                y_vals = np.unique(surface_data["y"])
                Z = surface_data["z"]
                
                # Find closest tenor
                closest_tenor_idx = np.abs(y_vals - common_tenor).argmin()
                closest_tenor = y_vals[closest_tenor_idx]
                
                # Extract smile for this tenor
                smile = Z[closest_tenor_idx, :]
                
                # Offset the x-values to position on the right side of the figure
                x_offset = 0.5  # Start at the midpoint
                x_spread = 0.5 / len(x_vals)  # Spread across right half
                
                # Create a new x-axis that spreads the smile across the right side
                new_x = np.linspace(x_offset + x_spread, 1.0 - x_spread, len(x_vals))
                
                fig.add_trace(
                    go.Scatter(
                        x=new_x,
                        y=smile,
                        mode='lines',
                        name=f'{surface_type.capitalize()} Skew ({closest_tenor:.0f}d)',
                        line=dict(color=colors.get(surface_type, 'purple'), width=2),
                        showlegend=False
                    ),
                    row=comparison_row, col=comparison_col
                )
                
                # Add vertical line at ATM point in the right section
                atm_idx = len(x_vals) // 2  # Middle of moneyness range
                atm_x = new_x[atm_idx]
                
                fig.add_shape(
                    type="line",
                    x0=atm_x, y0=0,
                    x1=atm_x, y1=1,
                    yref="paper",
                    line=dict(color="black", width=1, dash="dash"),
                    row=comparison_row, col=comparison_col
                )
    
    # Update axes for all comparison plots
    for row in range(1, n_rows + 1):
        for col in range(1, n_cols + 1):
            # Skip 3D scenes
            if row == 1 and col <= n_surfaces:
                continue
                
            # Skip parameter box in single surface case
            if n_surfaces == 1 and row == 2 and col == 2:
                continue
                
            fig.update_xaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(0,0,0,0.2)',
                row=row, col=col
            )
            
            fig.update_yaxes(
                showgrid=True,
                gridwidth=1,
                gridcolor='rgba(0,0,0,0.1)',
                zeroline=True,
                zerolinewidth=1,
                zerolinecolor='rgba(0,0,0,0.2)',
                row=row, col=col
            )
    
    # Update term structure plot labels
    if n_surfaces <= 2:
        term_row, term_col = (2, 1) if n_surfaces == 2 else (1, 2)
        fig.update_xaxes(title_text="Days to Expiration", row=term_row, col=term_col)
        fig.update_yaxes(title_text="Volatility", row=term_row, col=term_col)
    
    # Update skew comparison plot labels
    if n_surfaces == 2:
        skew_row, skew_col = 2, 2
        fig.update_xaxes(title_text="Moneyness (log(K/S))", row=skew_row, col=skew_col)
        fig.update_yaxes(title_text="Volatility", row=skew_row, col=skew_col)
    
    # Format the date properly for the title
    if isinstance(date, str):
        formatted_date = date
    elif hasattr(date, 'strftime'):
        formatted_date = date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(date).split()[0]  # Extract just the date part
    
    # Update overall layout
    fig.update_layout(
        title=f"Volatility Analysis for {stock} on {formatted_date}",
        height=900 if n_surfaces > 1 else 800,
        width=1200,
        margin=dict(l=65, r=50, b=65, t=90),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig




def create_greeks_surface(
    option_df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    greek: str = 'delta',
    option_type: str = 'Call'
) -> Dict:
    """
    Create Greeks surface from option chain data.
    
    Parameters:
    -----------
    option_df : pl.DataFrame
        DataFrame with option chain data
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    greek : str
        Greek to visualize ('delta', 'gamma', 'theta', 'vega', or 'rho')
    option_type : str
        Option type to visualize ('Call' or 'Put')
        
    Returns:
    --------
    Dict
        Dictionary with surface data and metadata
    """
    # Validate the greek parameter
    valid_greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
    if greek not in valid_greeks:
        raise ValueError(f"Invalid greek: {greek}. Must be one of {valid_greeks}")
    
    # Filter for the option type
    filtered_df = option_df.filter(pl.col("call_put") == option_type)
    
    if filtered_df.height == 0:
        raise ValueError(f"No {option_type} option data found for {stock} on {date}")
    
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Calculate days to expiration
    filtered_df = filtered_df.with_columns([
        ((pl.col("expiration") - date).dt.total_days()).alias("days_to_expiration")
    ])
    
    # Get unique strike prices and days to expiration
    strikes = filtered_df["strike"].unique().sort().to_numpy()
    days_to_exp = filtered_df["days_to_expiration"].unique().sort().to_numpy()
    
    # Create a grid for the surface
    strike_grid, days_grid = np.meshgrid(strikes, days_to_exp)
    
    # Initialize the value grid with NaNs
    greek_grid = np.full(strike_grid.shape, np.nan)
    
    # Create lookup dictionary for fast access
    lookup_dict = {}
    for row in filtered_df.select(["days_to_expiration", "strike", greek]).iter_rows():
        days, strike, value = row
        lookup_dict[(days, strike)] = value
    
    # Fill in the value grid
    for i, day in enumerate(days_to_exp):
        for j, strike in enumerate(strikes):
            key = (day, strike)
            if key in lookup_dict:
                greek_grid[i, j] = lookup_dict[key]
    
    # Fill in missing values using interpolation if possible
    if np.sum(~np.isnan(greek_grid)) > 5:  # Need at least 5 valid points for interpolation
        # Create mask of valid points
        mask = ~np.isnan(greek_grid)
        
        # Extract coordinates and values of valid points
        x_valid = strike_grid[mask]
        y_valid = days_grid[mask]
        z_valid = greek_grid[mask]
        
        # Perform interpolation
        points = np.column_stack((x_valid, y_valid))
        try:
            # Try cubic interpolation first
            grid_z = griddata(points, z_valid, (strike_grid, days_grid), method='cubic')
            
            # Fall back to linear for remaining NaNs
            if np.any(np.isnan(grid_z)):
                nan_mask = np.isnan(grid_z)
                grid_z_linear = griddata(points, z_valid, (strike_grid, days_grid), method='linear')
                grid_z[nan_mask] = grid_z_linear[nan_mask]
            
            # Only update original grid where we had NaNs
            nan_mask = np.isnan(greek_grid)
            greek_grid[nan_mask] = grid_z[nan_mask]
        except Exception as e:
            warnings.warn(f"Interpolation failed: {str(e)}")
    
    # Return as a dictionary with all the data needed for visualization
    return {
        "type": "option_greek",
        "subtype": f"{option_type}_{greek}",
        "x": strike_grid,
        "y": days_grid,
        "z": greek_grid,
        "x_label": "Strike Price",
        "y_label": "Days to Expiration",
        "z_label": greek.capitalize(),
        "title": f"{stock} {option_type} Option {greek.capitalize()} Surface on {date}",
        "description": f"Market {greek} surface from {option_type} options",
        "date": date
    }

def estimate_greeks_from_volatility(
    surface_data: Dict,
    risk_free_rate: float = 0.02,
    current_price: Optional[float] = None
) -> Dict[str, Dict]:
    """
    Estimate option Greeks from a volatility surface.
    
    Parameters:
    -----------
    surface_data : Dict
        Dictionary with volatility surface data
    risk_free_rate : float
        Risk-free interest rate (annual)
    current_price : float, optional
        Current stock price (if None, will use median strike or surface's current_price)
        
    Returns:
    --------
    Dict[str, Dict]
        Dictionary of Greek surfaces (delta, gamma, theta, vega, rho)
    """
    # Extract surface data
    strike_grid = surface_data["x"]
    days_grid = surface_data["y"]
    vol_grid = surface_data["z"]
    
    # Determine current stock price
    if current_price is None:
        if "current_price" in surface_data:
            current_price = surface_data["current_price"]
        else:
            # Use median strike as proxy
            unique_strikes = np.unique(strike_grid)
            current_price = np.median(unique_strikes)
    
    # Initialize Greek grids
    delta_grid = np.full_like(vol_grid, np.nan)
    gamma_grid = np.full_like(vol_grid, np.nan)
    theta_grid = np.full_like(vol_grid, np.nan)
    vega_grid = np.full_like(vol_grid, np.nan)
    rho_grid = np.full_like(vol_grid, np.nan)
    
    # Determine if this is a call or put surface
    is_call = "Call" in surface_data["title"] or "call" in surface_data["subtype"].lower()
    
    # Import scipy stats here to avoid potential import issues
    from scipy.stats import norm
    
    # Iterate over the surface
    for i in range(strike_grid.shape[0]):
        for j in range(strike_grid.shape[1]):
            # Skip if volatility is NaN
            if np.isnan(vol_grid[i, j]):
                continue
            
            # Extract values for this point
            strike = strike_grid[i, j]
            days = days_grid[i, j]
            vol = vol_grid[i, j]
            
            # Convert to years for option pricing
            T = days / 365.0
            
            # Skip if expiration is too close or zero (to avoid division by zero)
            if T <= 0.001:
                continue
            
            # Skip invalid strikes
            if strike <= 0:
                continue
                
            # Calculate d1 and d2 (Black-Scholes)
            try:
                d1 = (np.log(current_price / strike) + (risk_free_rate + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
                d2 = d1 - vol * np.sqrt(T)
                
                # Calculate normal CDF and PDF
                N_d1 = norm.cdf(d1)
                N_d2 = norm.cdf(d2)
                N_neg_d1 = norm.cdf(-d1)
                N_neg_d2 = norm.cdf(-d2)
                n_d1 = norm.pdf(d1)
                
                # Calculate Greeks for calls and puts
                if is_call:
                    # Call delta
                    delta_grid[i, j] = N_d1
                    
                    # Call gamma (same for calls and puts)
                    gamma_grid[i, j] = n_d1 / (current_price * vol * np.sqrt(T))
                    
                    # Call theta (daily)
                    theta_grid[i, j] = (-((current_price * vol * n_d1) / (2 * np.sqrt(T))) - 
                                        risk_free_rate * strike * np.exp(-risk_free_rate * T) * N_d2) / 365.0
                    
                    # Call vega (for 1% change in vol)
                    vega_grid[i, j] = current_price * np.sqrt(T) * n_d1 / 100.0
                    
                    # Call rho (for 1% change in rate)
                    rho_grid[i, j] = strike * T * np.exp(-risk_free_rate * T) * N_d2 / 100.0
                else:
                    # Put delta
                    delta_grid[i, j] = N_d1 - 1
                    
                    # Put gamma (same for calls and puts)
                    gamma_grid[i, j] = n_d1 / (current_price * vol * np.sqrt(T))
                    
                    # Put theta (daily)
                    theta_grid[i, j] = (-((current_price * vol * n_d1) / (2 * np.sqrt(T))) + 
                                        risk_free_rate * strike * np.exp(-risk_free_rate * T) * N_neg_d2) / 365.0
                    
                    # Put vega (for 1% change in vol)
                    vega_grid[i, j] = current_price * np.sqrt(T) * n_d1 / 100.0
                    
                    # Put rho (for 1% change in rate)
                    rho_grid[i, j] = -strike * T * np.exp(-risk_free_rate * T) * N_neg_d2 / 100.0
            except Exception:
                # Skip any calculation errors
                continue
    
    # Create surface data dictionaries for each Greek
    option_type = "Call" if is_call else "Put"
    date = surface_data["date"]
    stock = surface_data["title"].split()[0]  # Extract stock symbol from title
    
    greeks = {}
    
    greeks["delta"] = {
        "type": "estimated_greek",
        "subtype": f"{option_type}_delta",
        "x": strike_grid,
        "y": days_grid,
        "z": delta_grid,
        "x_label": surface_data["x_label"],
        "y_label": surface_data["y_label"],
        "z_label": "Delta",
        "title": f"{stock} Estimated {option_type} Delta Surface on {date}",
        "description": f"Estimated delta from {surface_data['type']} volatility",
        "date": date
    }
    
    greeks["gamma"] = {
        "type": "estimated_greek",
        "subtype": f"{option_type}_gamma",
        "x": strike_grid,
        "y": days_grid,
        "z": gamma_grid,
        "x_label": surface_data["x_label"],
        "y_label": surface_data["y_label"],
        "z_label": "Gamma",
        "title": f"{stock} Estimated {option_type} Gamma Surface on {date}",
        "description": f"Estimated gamma from {surface_data['type']} volatility",
        "date": date
    }
    
    greeks["theta"] = {
        "type": "estimated_greek",
        "subtype": f"{option_type}_theta",
        "x": strike_grid,
        "y": days_grid,
        "z": theta_grid,
        "x_label": surface_data["x_label"],
        "y_label": surface_data["y_label"],
        "z_label": "Theta (Daily)",
        "title": f"{stock} Estimated {option_type} Theta Surface on {date}",
        "description": f"Estimated theta from {surface_data['type']} volatility",
        "date": date
    }
    
    greeks["vega"] = {
        "type": "estimated_greek",
        "subtype": f"{option_type}_vega",
        "x": strike_grid,
        "y": days_grid,
        "z": vega_grid,
        "x_label": surface_data["x_label"],
        "y_label": surface_data["y_label"],
        "z_label": "Vega (per 1% vol change)",
        "title": f"{stock} Estimated {option_type} Vega Surface on {date}",
        "description": f"Estimated vega from {surface_data['type']} volatility",
        "date": date
    }
    
    greeks["rho"] = {
        "type": "estimated_greek",
        "subtype": f"{option_type}_rho",
        "x": strike_grid,
        "y": days_grid,
        "z": rho_grid,
        "x_label": surface_data["x_label"],
        "y_label": surface_data["y_label"],
        "z_label": "Rho (per 1% rate change)",
        "title": f"{stock} Estimated {option_type} Rho Surface on {date}",
        "description": f"Estimated rho from {surface_data['type']} volatility",
        "date": date
    }
    
    return greeks

# ======================================================
# Surface Comparison and Visualization Functions
# ======================================================

def interpolate_surface(
    surface_data: Dict,
    new_x: np.ndarray,
    new_y: np.ndarray
) -> Dict:
    """
    Interpolate a surface onto a new grid.
    
    Parameters:
    -----------
    surface_data : Dict
        Dictionary with surface data
    new_x : np.ndarray
        New x-grid coordinates
    new_y : np.ndarray
        New y-grid coordinates
        
    Returns:
    --------
    Dict
        Interpolated surface data
    """
    # Extract original coordinates and values
    x_orig = surface_data["x"].flatten()
    y_orig = surface_data["y"].flatten()
    z_orig = surface_data["z"].flatten()
    
    # Remove NaN values
    valid_mask = ~np.isnan(z_orig)
    x_valid = x_orig[valid_mask]
    y_valid = y_orig[valid_mask]
    z_valid = z_orig[valid_mask]
    
    # Check if we have enough valid points
    if len(z_valid) < 4:
        raise ValueError("Not enough valid points for interpolation")
    
    # Create points arrays
    points = np.column_stack((x_valid, y_valid))
    
    # Create new grid
    new_X, new_Y = np.meshgrid(new_x, new_y)
    
    # Interpolate
    try:
        # Try cubic first
        new_Z = griddata(points, z_valid, (new_X, new_Y), method='cubic')
        
        # Fill remaining NaNs with linear interpolation
        if np.any(np.isnan(new_Z)):
            mask_nan = np.isnan(new_Z)
            new_Z_linear = griddata(points, z_valid, (new_X, new_Y), method='linear')
            new_Z[mask_nan] = new_Z_linear[mask_nan]
            
            # If still NaN, try nearest
            if np.any(np.isnan(new_Z)):
                mask_nan = np.isnan(new_Z)
                new_Z_nearest = griddata(points, z_valid, (new_X, new_Y), method='nearest')
                new_Z[mask_nan] = new_Z_nearest[mask_nan]
    except Exception:
        # Fall back to nearest neighbor if other methods fail
        new_Z = griddata(points, z_valid, (new_X, new_Y), method='nearest')
    
    # Create new surface data dictionary
    new_surface = surface_data.copy()
    new_surface["x"] = new_X
    new_surface["y"] = new_Y
    new_surface["z"] = new_Z
    new_surface["description"] = f"Interpolated: {surface_data['description']}"
    
    return new_surface

def transform_surface_to_common_space(
    surface: Dict,
    reference_price: float = 100.0,
    max_days: int = 180
) -> Dict:
    """
    Transform a surface to a common coordinate space for comparison.
    
    Parameters:
    -----------
    surface : Dict
        Surface data dictionary
    reference_price : float
        Reference price to normalize strikes (as 100% moneyness)
    max_days : int
        Maximum days to include in the transformed space
        
    Returns:
    --------
    Dict
        Transformed surface in common space
    """
    # Determine the current price for moneyness calculation
    if "current_price" in surface:
        current_price = surface["current_price"]
    else:
        # Use median strike as proxy
        unique_strikes = np.unique(surface["x"])
        current_price = np.median(unique_strikes)
    
    # Create moneyness values
    x_vals = np.unique(surface["x"])
    moneyness = x_vals / current_price * reference_price
    
    # Create days values, capped at max_days
    y_vals = np.unique(surface["y"])
    y_vals = y_vals[y_vals <= max_days]
    
    # Create normalized grid
    new_X, new_Y = np.meshgrid(moneyness, y_vals)
    
    # Initialize new z values
    new_Z = np.full(new_X.shape, np.nan)
    
    # Map original values to normalized grid
    for i, y in enumerate(y_vals):
        for j, m in enumerate(moneyness):
            # Find corresponding original coordinates
            orig_x = m / reference_price * current_price
            orig_y = y
            
            # Find closest point in original grid
            x_idx = np.argmin(np.abs(x_vals - orig_x))
            y_idx = np.argmin(np.abs(y_vals - orig_y))
            
            # Map the value
            if y_idx < len(y_vals) and x_idx < len(x_vals):
                # Find the original point
                mask = (np.isclose(surface["y"], y_vals[y_idx])) & (np.isclose(surface["x"], x_vals[x_idx]))
                if np.any(mask):
                    new_Z[i, j] = surface["z"][mask][0]
    
    # Fill NaN values with interpolation
    if np.sum(~np.isnan(new_Z)) > 5:  # Need enough points for interpolation
        mask = ~np.isnan(new_Z)
        x_valid = new_X[mask]
        y_valid = new_Y[mask]
        z_valid = new_Z[mask]
        
        points = np.column_stack((x_valid, y_valid))
        
        try:
            # Try cubic interpolation
            grid_z = griddata(points, z_valid, (new_X, new_Y), method='cubic')
            
            # Fall back to linear for remaining NaNs
            if np.any(np.isnan(grid_z)):
                nan_mask = np.isnan(grid_z)
                grid_z_linear = griddata(points, z_valid, (new_X, new_Y), method='linear')
                grid_z[nan_mask] = grid_z_linear[nan_mask]
                
                # Fall back to nearest for any remaining NaNs
                if np.any(np.isnan(grid_z)):
                    nan_mask = np.isnan(grid_z)
                    grid_z_nearest = griddata(points, z_valid, (new_X, new_Y), method='nearest')
                    grid_z[nan_mask] = grid_z_nearest[nan_mask]
            
            # Update new_Z
            new_Z = grid_z
        except Exception:
            # Keep original values if interpolation fails
            pass
    
    # Create transformed surface
    transformed = surface.copy()
    transformed["x"] = new_X
    transformed["y"] = new_Y
    transformed["z"] = new_Z
    transformed["x_label"] = "Moneyness (% of ATM)"
    transformed["description"] = f"Normalized: {surface['description']}"
    transformed["reference_price"] = reference_price
    
    return transformed

def align_surfaces_for_comparison(
    surface1: Dict,
    surface2: Dict,
    normalize_moneyness: bool = True
) -> Tuple[Dict, Dict]:
    """
    Align two surfaces to a common grid for comparison.
    
    Parameters:
    -----------
    surface1 : Dict
        First surface data dictionary
    surface2 : Dict
        Second surface data dictionary
    normalize_moneyness : bool
        Whether to normalize strikes to moneyness
        
    Returns:
    --------
    Tuple[Dict, Dict]
        Aligned surface data dictionaries
    """
    # If both surfaces are already in a common space, just use them directly
    if (normalize_moneyness and 
        "Moneyness" in surface1["x_label"] and 
        "Moneyness" in surface2["x_label"]):
        # Already normalized, just need to align grids
        pass
    elif normalize_moneyness:
        # Transform to common moneyness space
        try:
            surface1 = transform_surface_to_common_space(surface1)
            surface2 = transform_surface_to_common_space(surface2)
        except Exception as e:
            warnings.warn(f"Surface normalization failed: {str(e)}. Using original surfaces.")
    
    # Extract grids
    x1 = np.unique(surface1["x"])
    y1 = np.unique(surface1["y"])
    x2 = np.unique(surface2["x"])
    y2 = np.unique(surface2["y"])
    
    # Create common grid
    common_x = np.sort(np.unique(np.concatenate([x1, x2])))
    common_y = np.sort(np.unique(np.concatenate([y1, y2])))
    
    # Limit common grid to points that exist in both surfaces
    x_min = max(np.min(x1), np.min(x2))
    x_max = min(np.max(x1), np.max(x2))
    y_min = max(np.min(y1), np.min(y2))
    y_max = min(np.max(y1), np.max(y2))
    
    common_x = common_x[(common_x >= x_min) & (common_x <= x_max)]
    common_y = common_y[(common_y >= y_min) & (common_y <= y_max)]
    
    if len(common_x) < 2 or len(common_y) < 2:
        # Not enough common points, create artificial grid
        warnings.warn("Not enough common points between surfaces. Using artificial grid.")
        common_x = np.linspace(x_min, x_max, 10)
        common_y = np.linspace(y_min, y_max, 10)
    
    # Interpolate both surfaces to common grid
    try:
        aligned_surface1 = interpolate_surface(surface1, common_x, common_y)
        aligned_surface2 = interpolate_surface(surface2, common_x, common_y)
    except Exception as e:
        # Handle interpolation failure by creating simpler grid
        warnings.warn(f"Interpolation failed: {str(e)}. Using simplified grid.")
        common_x = np.linspace(x_min, x_max, 5)
        common_y = np.linspace(y_min, y_max, 5)
        
        try:
            aligned_surface1 = interpolate_surface(surface1, common_x, common_y)
            aligned_surface2 = interpolate_surface(surface2, common_x, common_y)
        except Exception:
            # Last resort - reuse original surfaces
            warnings.warn("Simplified interpolation also failed. Using original surfaces.")
            aligned_surface1 = surface1
            aligned_surface2 = surface2
    
    return aligned_surface1, aligned_surface2

def calculate_surface_difference(
    surface1: Dict,
    surface2: Dict,
    relative: bool = False
) -> Dict:
    """
    Calculate the difference between two surfaces.
    
    Parameters:
    -----------
    surface1 : Dict
        First surface data dictionary
    surface2 : Dict
        Second surface data dictionary
    relative : bool
        If True, calculate relative difference (percentage)
        
    Returns:
    --------
    Dict
        Surface difference data dictionary
    """
    # Align surfaces
    try:
        aligned_s1, aligned_s2 = align_surfaces_for_comparison(surface1, surface2)
    except Exception as e:
        warnings.warn(f"Surface alignment failed: {str(e)}. Using original surfaces.")
        aligned_s1, aligned_s2 = surface1, surface2
    
    # Calculate difference
    if relative:
        # Avoid division by zero
        denominator = np.maximum(0.0001, np.abs(aligned_s2["z"]))
        diff_z = (aligned_s1["z"] - aligned_s2["z"]) / denominator * 100.0
        z_label = "Relative Difference (%)"
    else:
        diff_z = aligned_s1["z"] - aligned_s2["z"]
        z_label = f"Difference ({surface1['z_label']} - {surface2['z_label']})"
    
    # Create difference surface dictionary
    difference = {
        "type": "difference",
        "subtype": f"{surface1['subtype']}_vs_{surface2['subtype']}",
        "x": aligned_s1["x"],
        "y": aligned_s1["y"],
        "z": diff_z,
        "x_label": aligned_s1["x_label"],
        "y_label": aligned_s1["y_label"],
        "z_label": z_label,
        "title": f"Difference: {surface1['title'].split(' on ')[0]} vs {surface2['title'].split(' on ')[0]}",
        "description": f"Difference between {surface1['description']} and {surface2['description']}",
        "date": surface1["date"],
        "source1": surface1["type"],
        "source2": surface2["type"],
        "relative": relative
    }
    
    return difference

def calculate_surface_metrics(
    surface1: Dict,
    surface2: Dict
) -> Dict:
    """
    Calculate comparison metrics between two surfaces.
    
    Parameters:
    -----------
    surface1 : Dict
        First surface data dictionary
    surface2 : Dict
        Second surface data dictionary
        
    Returns:
    --------
    Dict
        Dictionary with comparison metrics
    """
    # Align surfaces
    try:
        aligned_s1, aligned_s2 = align_surfaces_for_comparison(surface1, surface2)
    except Exception as e:
        warnings.warn(f"Surface alignment failed in metrics calculation: {str(e)}. Using original surfaces.")
        aligned_s1, aligned_s2 = surface1, surface2
    
    # Calculate difference
    diff = aligned_s1["z"] - aligned_s2["z"]
    
    # Create mask for valid data points
    valid_mask = ~np.isnan(aligned_s1["z"]) & ~np.isnan(aligned_s2["z"]) & ~np.isnan(diff)
    
    # Check if we have enough valid points
    if np.sum(valid_mask) < 5:
        return {
            "mse": np.nan,
            "mae": np.nan,
            "max_diff": np.nan,
            "correlation": np.nan,
            "avg_value1": np.nanmean(aligned_s1["z"]) if np.any(~np.isnan(aligned_s1["z"])) else np.nan,
            "avg_value2": np.nanmean(aligned_s2["z"]) if np.any(~np.isnan(aligned_s2["z"])) else np.nan,
            "avg_pct_diff": np.nan,
            "valid_points": np.sum(valid_mask)
        }
    
    # Extract valid values
    z1_valid = aligned_s1["z"][valid_mask]
    z2_valid = aligned_s2["z"][valid_mask]
    diff_valid = diff[valid_mask]
    
    # Calculate metrics
    mse = np.mean(diff_valid ** 2)
    mae = np.mean(np.abs(diff_valid))
    max_diff = np.max(np.abs(diff_valid))
    
    # Calculate correlation
    if len(z1_valid) > 1:
        correlation = np.corrcoef(z1_valid, z2_valid)[0, 1]
    else:
        correlation = np.nan
    
    # Calculate average values
    avg_s1 = np.mean(z1_valid)
    avg_s2 = np.mean(z2_valid)
    
    # Calculate percentage difference
    pct_diff = np.mean(np.abs(diff_valid) / np.maximum(0.0001, np.abs(z2_valid))) * 100
    
    # Return metrics
    return {
        "mse": mse,
        "mae": mae,
        "max_diff": max_diff,
        "correlation": correlation,
        "avg_value1": avg_s1,
        "avg_value2": avg_s2,
        "avg_pct_diff": pct_diff,
        "valid_points": np.sum(valid_mask)
    }

def plot_surface(
    surface_data: Dict,
    colorscale: str = 'Viridis',
    show_term_structure: bool = True,
    show_skew: bool = True,
    vol_range: Tuple[float, float] = (0.0, 0.7),  # Fixed volatility range for consistent visualization
    height: int = 800,
    width: int = 1200
) -> go.Figure:
    """
    Create a comprehensive figure with 3D surface, term structure, and volatility smile plots
    with consistent volatility scaling.
    
    Parameters:
    -----------
    surface_data : Dict
        Surface data dictionary
    colorscale : str
        Colorscale to use for the 3D surface
    show_term_structure : bool
        Whether to include term structure plot
    show_skew : bool
        Whether to include volatility skew plot
    vol_range : Tuple[float, float]
        Fixed range for volatility axis (min, max)
    height : int
        Height of the figure in pixels
    width : int
        Width of the figure in pixels
        
    Returns:
    --------
    go.Figure
        Plotly figure with multiple views of the volatility surface
    """
    # Extract data
    z = surface_data["z"]
    x = np.unique(surface_data["x"])  # Moneyness or Strike values
    y = np.unique(surface_data["y"])  # Days to expiration
    
    # Create subplot layout based on which plots to include
    if show_term_structure and show_skew:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[[{"type": "scene", "rowspan": 2}, {"type": "xy"}],
                   [None, {"type": "xy"}]],
            subplot_titles=["3D Volatility Surface", "Term Structure (ATM)", 
                            "", "Volatility Smile"]
        )
        scene_col, scene_row = 1, 1
        term_col, term_row = 2, 1
        skew_col, skew_row = 2, 2
    elif show_term_structure:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=["3D Volatility Surface", "Term Structure (ATM)"]
        )
        scene_col, scene_row = 1, 1
        term_col, term_row = 2, 1
        skew_col, skew_row = None, None
    elif show_skew:
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "scene"}, {"type": "xy"}]],
            subplot_titles=["3D Volatility Surface", "Volatility Smile"]
        )
        scene_col, scene_row = 1, 1
        term_col, term_row = None, None
        skew_col, skew_row = 2, 1
    else:
        fig = make_subplots(
            rows=1, cols=1,
            specs=[[{"type": "scene"}]],
            subplot_titles=["3D Volatility Surface"]
        )
        scene_col, scene_row = 1, 1
        term_col, term_row = None, None
        skew_col, skew_row = None, None
    
    # Add 3D surface plot with fixed volatility range
    # FIXED: cmin/cmax are now only applied to the Surface, not the colorbar
    fig.add_trace(
        go.Surface(
            z=z,
            x=x,
            y=y,
            colorscale=colorscale,
            colorbar=dict(
                title=surface_data["z_label"],
                len=0.75,
                y=0.5
            ),
            # Simple surface without complex lighting
            lighting=dict(ambient=0.8, diffuse=0.2, roughness=0.9, specular=0.1, fresnel=0),
            contours={
                "x": {"show": True, "color":"black", "width": 1},
                "y": {"show": True, "color":"black", "width": 1},
                "z": {"show": True, "color":"black", "width": 1}
            },
            # Set surface color scale range to match vol_range
            cmin=vol_range[0],
            cmax=vol_range[1]
        ),
        row=scene_row, col=scene_col
    )
    
    # Add term structure plot if requested
    if show_term_structure and term_col and term_row:
        # Extract at-the-money volatilities 
        atm_idx = len(x) // 2
        atm_vols = z[:, atm_idx]
        
        # Apply cubic spline for smoothing if we have enough points
        if len(y) >= 4:
            from scipy.interpolate import CubicSpline
            cs = CubicSpline(y, atm_vols)
            smooth_days = np.linspace(min(y), max(y), 100)
            smooth_vols = cs(smooth_days)
            
            # Plot the smoothed line
            fig.add_trace(
                go.Scatter(
                    x=smooth_days,
                    y=smooth_vols,
                    mode='lines',
                    name='ATM Volatility (Smoothed)',
                    line=dict(color='blue', width=2.5)
                ),
                row=term_row, col=term_col
            )
            
            # Add the original points
            fig.add_trace(
                go.Scatter(
                    x=y,
                    y=atm_vols,
                    mode='markers',
                    name='ATM Volatility (Data Points)',
                    marker=dict(color='blue', size=8, symbol='circle')
                ),
                row=term_row, col=term_col
            )
        else:
            # Just plot the line if we don't have enough points for spline
            fig.add_trace(
                go.Scatter(
                    x=y,
                    y=atm_vols,
                    mode='lines+markers',
                    name='ATM Volatility',
                    line=dict(color='blue', width=2),
                    marker=dict(size=8)
                ),
                row=term_row, col=term_col
            )
        
        # Add bands around term structure
        vol_std = np.std(atm_vols) * 0.5
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([y, y[::-1]]),
                y=np.concatenate([atm_vols + vol_std, (atm_vols - vol_std)[::-1]]),
                fill='toself',
                fillcolor='rgba(0, 0, 255, 0.1)',
                line=dict(color='rgba(0, 0, 255, 0)'),
                showlegend=False,
                hoverinfo='skip'
            ),
            row=term_row, col=term_col
        )
        
        # Update term structure layout with fixed y-axis range
        fig.update_xaxes(
            title_text="Days to Expiration",
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            row=term_row, col=term_col
        )
        fig.update_yaxes(
            title_text=surface_data["z_label"],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            range=vol_range,  # Set fixed y-axis range
            row=term_row, col=term_col
        )
    
    # Add volatility smile plot if requested
    if show_skew and skew_col and skew_row:
        # Select a few representative days to expiration
        n_curves = min(5, len(y))
        selected_indices = np.linspace(0, len(y)-1, n_curves).astype(int)
        selected_days = [y[i] for i in selected_indices]
        
        # Create a viridis-like color palette
        colors = [
            'rgb(68, 1, 84)',    # Dark purple
            'rgb(59, 82, 139)',   # Blue
            'rgb(33, 144, 140)',  # Teal
            'rgb(93, 201, 99)',   # Green
            'rgb(253, 231, 37)'   # Yellow
        ]
        
        # Ensure we don't try to use more colors than we have
        colors = colors[:n_curves]
        
        for i, day_idx in enumerate(selected_indices):
            day = y[day_idx]
            smile = z[day_idx, :]
            
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=smile,
                    mode='lines',
                    name=f'{day:.0f}d',
                    line=dict(color=colors[i], width=2)
                ),
                row=skew_row, col=skew_col
            )
        
        # Add vertical line at ATM point
        atm_point = 0.0 if "Moneyness" in surface_data["x_label"] else np.median(x)
        fig.add_shape(
            type="line",
            x0=atm_point, y0=0,
            x1=atm_point, y1=1,
            yref="paper",
            line=dict(color="black", width=1, dash="dash"),
            row=skew_row, col=skew_col
        )
        
        # Update skew plot layout with fixed y-axis range
        fig.update_xaxes(
            title_text=surface_data["x_label"],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            row=skew_row, col=skew_col
        )
        fig.update_yaxes(
            title_text=surface_data["z_label"],
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(0,0,0,0.1)',
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor='rgba(0,0,0,0.2)',
            range=vol_range,  # Set fixed y-axis range
            row=skew_row, col=skew_col
        )
    
    # Update 3D scene properties with fixed z-axis range
    fig.update_scenes(
        xaxis_title=surface_data["x_label"],
        yaxis_title=surface_data["y_label"],
        zaxis_title=surface_data["z_label"],
        zaxis=dict(range=vol_range),  # Set fixed z-axis range for 3D plot
        aspectratio=dict(x=1.2, y=1.0, z=0.7),
        camera=dict(eye=dict(x=1.5, y=-1.5, z=1)),
        row=scene_row, col=scene_col
    )
    
    # Update overall layout
    fig.update_layout(
        title=surface_data["title"],
        height=height,
        width=width,
        margin=dict(l=65, r=50, b=65, t=90),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def compare_surfaces(
    surface1: Dict,
    surface2: Dict,
    show_difference: bool = True,
    relative_diff: bool = False,
    vol_range: Tuple[float, float] = (0.0, 0.7),  # Fixed volatility range
    diff_range: Tuple[float, float] = (-0.2, 0.2)  # Range for difference plot
) -> go.Figure:
    """
    Create a comparison visualization of two surfaces with consistent scaling.
    
    Parameters:
    -----------
    surface1 : Dict
        First surface data dictionary
    surface2 : Dict
        Second surface data dictionary
    show_difference : bool
        Whether to include the difference surface
    relative_diff : bool
        Whether to show relative (percentage) differences
    vol_range : Tuple[float, float]
        Fixed range for volatility axis (min, max)
    diff_range : Tuple[float, float]
        Range for difference plot
        
    Returns:
    --------
    go.Figure
        Plotly figure with surface comparison
    """
    # Try to align surfaces
    try:
        aligned_s1, aligned_s2 = align_surfaces_for_comparison(surface1, surface2)
    except Exception as e:
        warnings.warn(f"Surface alignment failed: {str(e)}. Using original surfaces.")
        aligned_s1, aligned_s2 = surface1, surface2
    
    # Calculate difference if requested
    if show_difference:
        try:
            diff_surface = calculate_surface_difference(surface1, surface2, relative=relative_diff)
            
            # Create subplot figure with 3 plots
            fig = make_subplots(
                rows=1, 
                cols=3,
                specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
                subplot_titles=[
                    surface1["title"].split(" on ")[0],
                    surface2["title"].split(" on ")[0],
                    diff_surface["title"]
                ]
            )
            
            # Add surfaces with consistent scaling
            fig.add_trace(
                go.Surface(
                    z=aligned_s1["z"],
                    x=aligned_s1["x"],
                    y=aligned_s1["y"],
                    colorscale='Viridis',
                    colorbar=dict(
                        title=aligned_s1["z_label"],
                        len=0.5,
                        y=0.8,
                        x=0.15,
                        cmin=vol_range[0],
                        cmax=vol_range[1]
                    ),
                    cmin=vol_range[0],
                    cmax=vol_range[1]
                ),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Surface(
                    z=aligned_s2["z"],
                    x=aligned_s2["x"],
                    y=aligned_s2["y"],
                    colorscale='Plasma',
                    colorbar=dict(
                        title=aligned_s2["z_label"],
                        len=0.5,
                        y=0.8,
                        x=0.48,
                        cmin=vol_range[0],
                        cmax=vol_range[1]
                    ),
                    cmin=vol_range[0],
                    cmax=vol_range[1]
                ),
                row=1, col=2
            )
            
            # For difference plot, use a diverging colorscale centered at zero
            fig.add_trace(
                go.Surface(
                    z=diff_surface["z"],
                    x=diff_surface["x"],
                    y=diff_surface["y"],
                    colorscale='RdBu_r',  # Red-White-Blue with red for negative
                    colorbar=dict(
                        title=diff_surface["z_label"], 
                        len=0.5,
                        y=0.8,
                        x=0.85,
                        cmin=diff_range[0],
                        cmax=diff_range[1]
                    ),
                    cmin=diff_range[0],
                    cmax=diff_range[1]
                ),
                row=1, col=3
            )
            
            # Update layout
            fig.update_layout(
                title=f"Comparison: {surface1['title'].split(' on ')[0]} vs {surface2['title'].split(' on ')[0]}",
                height=600,
                width=1200
            )
            
            # Update scenes with consistent z-axis scaling
            fig.update_scenes(
                xaxis_title=aligned_s1["x_label"],
                yaxis_title=aligned_s1["y_label"],
                zaxis_title=aligned_s1["z_label"],
                zaxis=dict(range=vol_range),
                row=1, col=1
            )
            
            fig.update_scenes(
                xaxis_title=aligned_s2["x_label"],
                yaxis_title=aligned_s2["y_label"],
                zaxis_title=aligned_s2["z_label"],
                zaxis=dict(range=vol_range),
                row=1, col=2
            )
            
            # Difference plot gets its own range
            diff_range_actual = diff_range if not relative_diff else (-50, 50)
            fig.update_scenes(
                xaxis_title=diff_surface["x_label"],
                yaxis_title=diff_surface["y_label"],
                zaxis_title=diff_surface["z_label"],
                zaxis=dict(range=diff_range_actual),
                row=1, col=3
            )
        except Exception as e:
            warnings.warn(f"Failed to create difference surface: {str(e)}. Showing only two surfaces.")
            show_difference = False
    
    if not show_difference:
        # Create subplot figure with 2 plots
        fig = make_subplots(
            rows=1, 
            cols=2,
            specs=[[{'type': 'scene'}, {'type': 'scene'}]],
            subplot_titles=[
                surface1["title"].split(" on ")[0],
                surface2["title"].split(" on ")[0]
            ]
        )
        
        # Add surfaces with consistent scaling
        fig.add_trace(
            go.Surface(
                z=aligned_s1["z"],
                x=aligned_s1["x"],
                y=aligned_s1["y"],
                colorscale='Viridis',
                colorbar=dict(
                    title=aligned_s1["z_label"],
                    len=0.5,
                    y=0.8,
                    x=0.15,
                    cmin=vol_range[0],
                    cmax=vol_range[1]
                ),
                cmin=vol_range[0],
                cmax=vol_range[1]
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Surface(
                z=aligned_s2["z"],
                x=aligned_s2["x"],
                y=aligned_s2["y"],
                colorscale='Plasma',
                colorbar=dict(
                    title=aligned_s2["z_label"],
                    len=0.5,
                    y=0.8,
                    x=0.85,
                    cmin=vol_range[0],
                    cmax=vol_range[1]
                ),
                cmin=vol_range[0],
                cmax=vol_range[1]
            ),
            row=1, col=2
        )
        
        # Update layout
        fig.update_layout(
            title=f"Comparison: {surface1['title'].split(' on ')[0]} vs {surface2['title'].split(' on ')[0]}",
            height=600,
            width=1000
        )
        
        # Update scenes with consistent z-axis scaling
        fig.update_scenes(
            xaxis_title=aligned_s1["x_label"],
            yaxis_title=aligned_s1["y_label"],
            zaxis_title=aligned_s1["z_label"],
            zaxis=dict(range=vol_range),
            row=1, col=1
        )
        
        fig.update_scenes(
            xaxis_title=aligned_s2["x_label"],
            yaxis_title=aligned_s2["y_label"],
            zaxis_title=aligned_s2["z_label"],
            zaxis=dict(range=vol_range),
            row=1, col=2
        )
    
    return fig

def visualize_vol_term_structure(
    option_surface: Dict,
    model_surface: Dict,
    realized_surface: Optional[Dict] = None,
    vol_range: Tuple[float, float] = (0.0, 0.7)  # Fixed volatility range
) -> go.Figure:
    """
    Visualize volatility term structure comparison with consistent y-axis scaling.
    
    Parameters:
    -----------
    option_surface : Dict
        Option implied volatility surface data
    model_surface : Dict
        Model predicted volatility surface data
    realized_surface : Dict, optional
        Realized volatility surface data
    vol_range : Tuple[float, float]
        Fixed range for volatility axis (min, max)
        
    Returns:
    --------
    go.Figure
        Plotly figure with term structure comparison
    """
    # Try to normalize to common space first
    try:
        norm_option = transform_surface_to_common_space(option_surface)
        norm_model = transform_surface_to_common_space(model_surface)
        if realized_surface is not None:
            norm_realized = transform_surface_to_common_space(realized_surface)
        else:
            norm_realized = None
    except Exception as e:
        warnings.warn(f"Surface normalization failed: {str(e)}. Using original surfaces.")
        norm_option = option_surface
        norm_model = model_surface
        norm_realized = realized_surface
    
    # Extract at-the-money volatilities
    # For normalized surfaces, this is at x = 100
    atm_strike = 100.0 if "Moneyness" in norm_option["x_label"] else np.median(np.unique(norm_option["x"]))
    
    # Extract term structures - option surface
    opt_days = np.unique(norm_option["y"])
    opt_vols = []
    
    for day in opt_days:
        # Find all points with this expiration
        mask = np.isclose(norm_option["y"], day)
        x_vals = norm_option["x"][mask]
        z_vals = norm_option["z"][mask]
        
        # Find valid values
        valid_mask = ~np.isnan(z_vals)
        if np.sum(valid_mask) == 0:
            opt_vols.append(np.nan)
            continue
            
        x_valid = x_vals[valid_mask]
        z_valid = z_vals[valid_mask]
        
        # Find closest strike to ATM
        closest_idx = np.abs(x_valid - atm_strike).argmin()
        opt_vols.append(z_valid[closest_idx])
    
    # For model surface
    model_days = np.unique(norm_model["y"])
    model_vols = []
    
    for day in model_days:
        # Find all points with this expiration
        mask = np.isclose(norm_model["y"], day)
        x_vals = norm_model["x"][mask]
        z_vals = norm_model["z"][mask]
        
        # Find valid values
        valid_mask = ~np.isnan(z_vals)
        if np.sum(valid_mask) == 0:
            model_vols.append(np.nan)
            continue
            
        x_valid = x_vals[valid_mask]
        z_valid = z_vals[valid_mask]
        
        # Find closest strike to ATM
        closest_idx = np.abs(x_valid - atm_strike).argmin()
        model_vols.append(z_valid[closest_idx])
    
    # Extract realized values if available
    real_days = []
    real_vols = []
    
    if norm_realized is not None:
        real_days = np.unique(norm_realized["y"])
        
        for day in real_days:
            # Find all points with this expiration
            mask = np.isclose(norm_realized["y"], day)
            x_vals = norm_realized["x"][mask]
            z_vals = norm_realized["z"][mask]
            
            # Find valid values
            valid_mask = ~np.isnan(z_vals)
            if np.sum(valid_mask) == 0:
                real_vols.append(np.nan)
                continue
                
            x_valid = x_vals[valid_mask]
            z_valid = z_vals[valid_mask]
            
            # Find closest strike to ATM
            closest_idx = np.abs(x_valid - atm_strike).argmin()
            real_vols.append(z_valid[closest_idx])
    
    # Create figure
    fig = go.Figure()
    
    # Add option term structure
    fig.add_trace(
        go.Scatter(
            x=opt_days,
            y=opt_vols,
            mode='lines+markers',
            name='Option Implied Volatility',
            line=dict(color='blue', width=2)
        )
    )
    
    # Add model term structure
    fig.add_trace(
        go.Scatter(
            x=model_days,
            y=model_vols,
            mode='lines+markers',
            name='Model Predicted Volatility',
            line=dict(color='red', width=2, dash='dash')
        )
    )
    
    # Add realized term structure if available
    if norm_realized is not None and len(real_days) > 0:
        fig.add_trace(
            go.Scatter(
                x=real_days,
                y=real_vols,
                mode='lines+markers',
                name='Realized Volatility',
                line=dict(color='green', width=2, dash='dot')
            )
        )
    
    # Update layout with fixed y-axis range
    stock = option_surface["title"].split()[0]
    date = option_surface["date"]
    
    fig.update_layout(
        title=f"{stock} Volatility Term Structure Comparison (ATM) on {date}",
        xaxis_title="Days to Expiration / Forecast Horizon",
        yaxis_title="Volatility",
        yaxis=dict(range=vol_range),  # Set fixed y-axis range
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        width=800,
        height=500
    )
    
    return fig

def visualize_vol_skew(
    option_surface: Dict,
    model_surface: Dict,
    realized_surface: Optional[Dict] = None,
    target_expiration: Optional[float] = None,
    vol_range: Tuple[float, float] = (0.0, 0.7)  # Fixed volatility range
) -> go.Figure:
    """
    Visualize volatility skew comparison for a specific expiration with consistent y-axis scaling.
    
    Parameters:
    -----------
    option_surface : Dict
        Option implied volatility surface data
    model_surface : Dict
        Model predicted volatility surface data
    realized_surface : Dict, optional
        Realized volatility surface data
    target_expiration : float, optional
        Target expiration/horizon to visualize (if None, use the median)
    vol_range : Tuple[float, float]
        Fixed range for volatility axis (min, max)
        
    Returns:
    --------
    go.Figure
        Plotly figure with skew comparison
    """
    # Try to normalize to common space first
    try:
        norm_option = transform_surface_to_common_space(option_surface)
        norm_model = transform_surface_to_common_space(model_surface)
        if realized_surface is not None:
            norm_realized = transform_surface_to_common_space(realized_surface)
        else:
            norm_realized = None
    except Exception as e:
        warnings.warn(f"Surface normalization failed: {str(e)}. Using original surfaces.")
        norm_option = option_surface
        norm_model = model_surface
        norm_realized = realized_surface
    
    # If target expiration not specified, use the median or a common value
    opt_expirations = np.unique(norm_option["y"])
    model_expirations = np.unique(norm_model["y"])
    
    # Find common expirations
    common_exps = np.intersect1d(opt_expirations, model_expirations)
    
    if len(common_exps) > 0:
        # Use median of common expirations
        if target_expiration is None:
            target_expiration = np.median(common_exps)
            
        # Find closest to target
        closest_exp = common_exps[np.abs(common_exps - target_expiration).argmin()]
    else:
        # No common expirations, use median of all
        all_exps = np.union1d(opt_expirations, model_expirations)
        if target_expiration is None:
            target_expiration = np.median(all_exps)
            
        # Find closest in each surface
        opt_exp = opt_expirations[np.abs(opt_expirations - target_expiration).argmin()]
        model_exp = model_expirations[np.abs(model_expirations - target_expiration).argmin()]
        
        # Use the midpoint
        closest_exp = (opt_exp + model_exp) / 2
    
    # Extract skews for closest expirations
    # Option skew
    opt_exp = opt_expirations[np.abs(opt_expirations - closest_exp).argmin()]
    opt_strikes = []
    opt_vols = []
    
    for i in range(norm_option["z"].shape[0]):
        for j in range(norm_option["z"].shape[1]):
            if np.isclose(norm_option["y"][i, j], opt_exp):
                opt_strikes.append(norm_option["x"][i, j])
                opt_vols.append(norm_option["z"][i, j])
    
    # Sort by strike
    if len(opt_strikes) > 0:
        opt_idx = np.argsort(opt_strikes)
        opt_strikes = np.array(opt_strikes)[opt_idx]
        opt_vols = np.array(opt_vols)[opt_idx]
    
    # Model skew
    model_exp = model_expirations[np.abs(model_expirations - closest_exp).argmin()]
    model_strikes = []
    model_vols = []
    
    for i in range(norm_model["z"].shape[0]):
        for j in range(norm_model["z"].shape[1]):
            if np.isclose(norm_model["y"][i, j], model_exp):
                model_strikes.append(norm_model["x"][i, j])
                model_vols.append(norm_model["z"][i, j])
    
    # Sort by strike
    if len(model_strikes) > 0:
        model_idx = np.argsort(model_strikes)
        model_strikes = np.array(model_strikes)[model_idx]
        model_vols = np.array(model_vols)[model_idx]
    
    # Realized skew if available
    real_strikes = []
    real_vols = []
    real_exp = None
    
    if norm_realized is not None:
        real_expirations = np.unique(norm_realized["y"])
        if len(real_expirations) > 0:
            real_exp = real_expirations[np.abs(real_expirations - closest_exp).argmin()]
            
            for i in range(norm_realized["z"].shape[0]):
                for j in range(norm_realized["z"].shape[1]):
                    if np.isclose(norm_realized["y"][i, j], real_exp):
                        real_strikes.append(norm_realized["x"][i, j])
                        real_vols.append(norm_realized["z"][i, j])
            
            # Sort by strike
            if len(real_strikes) > 0:
                real_idx = np.argsort(real_strikes)
                real_strikes = np.array(real_strikes)[real_idx]
                real_vols = np.array(real_vols)[real_idx]
    
    # Create figure
    fig = go.Figure()
    
    # Add option skew if we have data
    if len(opt_strikes) > 0:
        fig.add_trace(
            go.Scatter(
                x=opt_strikes,
                y=opt_vols,
                mode='lines+markers',
                name=f'Option IV (T={opt_exp:.0f}d)',
                line=dict(color='blue', width=2)
            )
        )
    
    # Add model skew if we have data
    if len(model_strikes) > 0:
        fig.add_trace(
            go.Scatter(
                x=model_strikes,
                y=model_vols,
                mode='lines+markers',
                name=f'Model Vol (T={model_exp:.0f}d)',
                line=dict(color='red', width=2, dash='dash')
            )
        )
    
    # Add realized skew if available and we have data
    if len(real_strikes) > 0:
        fig.add_trace(
            go.Scatter(
                x=real_strikes,
                y=real_vols,
                mode='lines+markers',
                name=f'Realized Vol (T={real_exp:.0f}d)',
                line=dict(color='green', width=2, dash='dot')
            )
        )
    
    # Add vertical line at ATM (moneyness = 100.0 if normalized)
    if "Moneyness" in norm_option["x_label"]:
        atm_strike = 100.0
    else:
        # Use median as approximate ATM
        all_strikes = np.concatenate([opt_strikes, model_strikes])
        if len(real_strikes) > 0:
            all_strikes = np.concatenate([all_strikes, real_strikes])
        atm_strike = np.median(all_strikes)
        
    fig.add_vline(x=atm_strike, line=dict(color="black", width=1, dash="dash"))
    
    # Update layout with fixed y-axis range
    stock = option_surface["title"].split()[0]
    date = option_surface["date"]
    
    fig.update_layout(
        title=f"{stock} Volatility Skew Comparison (T≈{closest_exp:.0f} days) on {date}",
        xaxis_title=norm_option["x_label"],
        yaxis_title="Volatility",
        yaxis=dict(range=vol_range),  # Set fixed y-axis range
        legend=dict(x=0.01, y=0.99, bordercolor="Black", borderwidth=1),
        width=800,
        height=500
    )
    
    return fig

def create_integrated_dashboard(
    option_df: pl.DataFrame,
    pred_df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    option_type: str = 'Call',
    vol_scale_factor: float = 1.0  # Scaling factor for volatility values
) -> Dict:
    """
    Create an integrated dashboard with all visualizations.
    
    Parameters:
    -----------
    option_df : pl.DataFrame
        DataFrame with option chain data
    pred_df : pl.DataFrame
        DataFrame with model predictions and realized values
    stock : str
        Stock symbol to analyze
    date : Union[str, datetime.date]
        Reference date for analysis
    option_type : str
        Option type to analyze ('Call' or 'Put')
    vol_scale_factor : float
        Scaling factor for volatility values (use 1.0 if all in same scale, higher if need to convert decimal to percent)
        
    Returns:
    --------
    Dict
        Dictionary with all figures and metrics
    """
    # Normalize date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    print(f"Creating dashboard for {stock} on {date}...")
    
    # Create surfaces
    print("Generating market implied volatility surface...")
    try:
        market_surface = create_option_vol_surface(option_df, stock, date, option_type, vol_scale=vol_scale_factor)
    except Exception as e:
        print(f"Error creating market surface: {e}")
        market_surface = None
    
    print("Generating model predicted volatility surface...")
    try:
        # Scale factor for model predictions: decimal to percentage
        # From diagnostic, we know predictions are ~0.02-0.05 while option IVs are ~0.20-0.57
        # Assuming model predictions are in decimal form, multiply by 10 for similar scale
        model_surface = create_model_vol_surface(
            pred_df, stock, date, vol_scale=vol_scale_factor*10, pred_suffix="_future"
        )
    except Exception as e:
        print(f"Error creating model surface: {e}")
        model_surface = None
    
    print("Generating realized volatility surface...")
    try:
        # Use the same scaling as model predictions
        realized_surface = create_realized_vol_surface(
            pred_df, stock, date, vol_scale=vol_scale_factor*10, future_suffix="_future"
        )
    except Exception as e:
        print(f"Warning: Could not create realized volatility surface: {e}")
        realized_surface = None
    
    # Skip further processing if essential surfaces missing
    if market_surface is None or model_surface is None:
        print("Critical error: Could not create market or model surface. Aborting dashboard creation.")
        return {
            "stock": stock,
            "date": date,
            "option_type": option_type,
            "error": "Could not create essential surfaces"
        }
    
    # Create comparison figures
    print("Creating surface comparisons...")
    try:
        market_vs_model_fig = compare_surfaces(market_surface, model_surface)
        metrics = {}
        metrics["market_vs_model"] = calculate_surface_metrics(market_surface, model_surface)
    except Exception as e:
        print(f"Error creating market vs model comparison: {e}")
        market_vs_model_fig = None
        metrics = {"market_vs_model": {"error": str(e)}}
    
    if realized_surface is not None:
        try:
            market_vs_realized_fig = compare_surfaces(market_surface, realized_surface)
            model_vs_realized_fig = compare_surfaces(model_surface, realized_surface)
            
            metrics["market_vs_realized"] = calculate_surface_metrics(market_surface, realized_surface)
            metrics["model_vs_realized"] = calculate_surface_metrics(model_surface, realized_surface)
        except Exception as e:
            print(f"Error creating realized comparisons: {e}")
            market_vs_realized_fig = None
            model_vs_realized_fig = None
    else:
        market_vs_realized_fig = None
        model_vs_realized_fig = None
    
    # Create term structure and skew visualizations
    print("Creating term structure and skew visualizations...")
    try:
        term_structure_fig = visualize_vol_term_structure(
            market_surface, model_surface, realized_surface
        )
    except Exception as e:
        print(f"Error creating term structure: {e}")
        term_structure_fig = None
    
    try:
        skew_fig = visualize_vol_skew(
            market_surface, model_surface, realized_surface
        )
    except Exception as e:
        print(f"Error creating skew figure: {e}")
        skew_fig = None
    
    # Estimate Greeks
    print("Estimating Greeks...")
    try:
        market_greeks = estimate_greeks_from_volatility(market_surface)
        model_greeks = estimate_greeks_from_volatility(model_surface)
        
        if realized_surface is not None:
            realized_greeks = estimate_greeks_from_volatility(realized_surface)
        else:
            realized_greeks = None
    except Exception as e:
        print(f"Error estimating greeks: {e}")
        market_greeks = None
        model_greeks = None
        realized_greeks = None
    
    # Create Greek comparisons for delta
    print("Creating Greek comparisons...")
    if market_greeks is not None and model_greeks is not None:
        try:
            market_vs_model_delta_fig = compare_surfaces(
                market_greeks["delta"], model_greeks["delta"]
            )
        except Exception as e:
            print(f"Error creating delta comparison: {e}")
            market_vs_model_delta_fig = None
        
        if realized_greeks is not None:
            try:
                market_vs_realized_delta_fig = compare_surfaces(
                    market_greeks["delta"], realized_greeks["delta"]
                )
            except Exception as e:
                print(f"Error creating realized delta comparison: {e}")
                market_vs_realized_delta_fig = None
        else:
            market_vs_realized_delta_fig = None
    else:
        market_vs_model_delta_fig = None
        market_vs_realized_delta_fig = None
    
    # Compile results
    print("Compiling dashboard...")
    dashboard = {
        "stock": stock,
        "date": date,
        "option_type": option_type,
        "surfaces": {
            "market": market_surface,
            "model": model_surface,
            "realized": realized_surface
        },
        "greeks": {
            "market": market_greeks,
            "model": model_greeks,
            "realized": realized_greeks
        },
        "comparison_figures": {
            "market_vs_model": market_vs_model_fig,
            "market_vs_realized": market_vs_realized_fig,
            "model_vs_realized": model_vs_realized_fig,
            "term_structure": term_structure_fig,
            "skew": skew_fig,
            "market_vs_model_delta": market_vs_model_delta_fig,
            "market_vs_realized_delta": market_vs_realized_delta_fig
        },
        "metrics": metrics
    }
    
    print("Dashboard creation complete!")
    return dashboard

def create_summary_report(dashboard: Dict) -> Dict:
    """
    Create a summary report from the dashboard.
    
    Parameters:
    -----------
    dashboard : Dict
        Dashboard dictionary
        
    Returns:
    --------
    Dict
        Summary report with key metrics and insights
    """
    stock = dashboard["stock"]
    date = dashboard["date"]
    option_type = dashboard["option_type"]
    
    # Extract key metrics
    metrics = dashboard["metrics"]
    
    # Create summary
    summary = {
        "stock": stock,
        "date": date,
        "option_type": option_type,
    }
    
    # Add market vs model metrics if available
    if "market_vs_model" in metrics and not isinstance(metrics["market_vs_model"], dict):
        summary["market_vs_model"] = {
            "mse": metrics["market_vs_model"]["mse"],
            "mae": metrics["market_vs_model"]["mae"],
            "correlation": metrics["market_vs_model"]["correlation"],
            "avg_market_vol": metrics["market_vs_model"]["avg_value1"],
            "avg_model_vol": metrics["market_vs_model"]["avg_value2"],
            "avg_pct_diff": metrics["market_vs_model"]["avg_pct_diff"],
            "valid_points": metrics["market_vs_model"].get("valid_points", 0)
        }
    else:
        # Default values if metrics not available
        summary["market_vs_model"] = {
            "mse": np.nan,
            "mae": np.nan,
            "correlation": np.nan,
            "avg_market_vol": np.nan,
            "avg_model_vol": np.nan,
            "avg_pct_diff": np.nan,
            "valid_points": 0
        }
    
    # Add other metrics if available
    if "market_vs_realized" in metrics and not isinstance(metrics["market_vs_realized"], dict):
        summary["market_vs_realized"] = {
            "mse": metrics["market_vs_realized"]["mse"],
            "mae": metrics["market_vs_realized"]["mae"],
            "correlation": metrics["market_vs_realized"]["correlation"],
            "avg_market_vol": metrics["market_vs_realized"]["avg_value1"],
            "avg_realized_vol": metrics["market_vs_realized"]["avg_value2"],
            "avg_pct_diff": metrics["market_vs_realized"]["avg_pct_diff"],
            "valid_points": metrics["market_vs_realized"].get("valid_points", 0)
        }
    
    if "model_vs_realized" in metrics and not isinstance(metrics["model_vs_realized"], dict):
        summary["model_vs_realized"] = {
            "mse": metrics["model_vs_realized"]["mse"],
            "mae": metrics["model_vs_realized"]["mae"],
            "correlation": metrics["model_vs_realized"]["correlation"],
            "avg_model_vol": metrics["model_vs_realized"]["avg_value1"],
            "avg_realized_vol": metrics["model_vs_realized"]["avg_value2"],
            "avg_pct_diff": metrics["model_vs_realized"]["avg_pct_diff"],
            "valid_points": metrics["model_vs_realized"].get("valid_points", 0)
        }
    
    # Generate insights
    insights = []
    
    # Compare average volatilities if available
    try:
        market_vol = summary["market_vs_model"]["avg_market_vol"]
        model_vol = summary["market_vs_model"]["avg_model_vol"]
        
        if not np.isnan(market_vol) and not np.isnan(model_vol):
            if market_vol > model_vol * 1.1:
                insights.append(f"Market implied volatility ({market_vol:.1%}) is significantly higher than model-predicted volatility ({model_vol:.1%}), suggesting the market expects more volatility than the model predicts.")
            elif model_vol > market_vol * 1.1:
                insights.append(f"Model-predicted volatility ({model_vol:.1%}) is significantly higher than market implied volatility ({market_vol:.1%}), suggesting potential trading opportunities if the model is accurate.")
            else:
                insights.append(f"Market implied volatility ({market_vol:.1%}) is roughly in line with model-predicted volatility ({model_vol:.1%}).")
    except (KeyError, TypeError):
        pass
    
    # Check correlation if available
    try:
        correlation = summary["market_vs_model"]["correlation"]
        valid_points = summary["market_vs_model"]["valid_points"]
        
        if not np.isnan(correlation) and valid_points > 5:
            if correlation > 0.8:
                insights.append(f"High correlation ({correlation:.2f}) between market and model volatility surfaces indicates similar term structure and skew patterns.")
            elif correlation < 0.5:
                insights.append(f"Low correlation ({correlation:.2f}) between market and model volatility surfaces suggests significant differences in volatility patterns.")
    except (KeyError, TypeError):
        pass
    
    # Add realized vol insights if available
    try:
        if "market_vs_realized" in summary:
            realized_vol = summary["market_vs_realized"]["avg_realized_vol"]
            market_vol = summary["market_vs_realized"]["avg_market_vol"]
            valid_points = summary["market_vs_realized"]["valid_points"]
            
            if not np.isnan(realized_vol) and not np.isnan(market_vol) and valid_points > 5:
                market_realized_diff = (market_vol / realized_vol - 1) * 100
                
                if market_vol > realized_vol * 1.15:
                    insights.append(f"Market implied volatility was {market_realized_diff:.1f}% higher than the subsequent realized volatility, suggesting an overpricing of options.")
                elif realized_vol > market_vol * 1.15:
                    insights.append(f"Realized volatility was {-market_realized_diff:.1f}% higher than the market implied volatility, suggesting an underpricing of options.")
    except (KeyError, TypeError):
        pass
    
    try:
        if "model_vs_realized" in summary:
            model_realized_corr = summary["model_vs_realized"]["correlation"]
            valid_points = summary["model_vs_realized"]["valid_points"]
            
            if not np.isnan(model_realized_corr) and valid_points > 5:
                insights.append(f"Model predictions had a {model_realized_corr:.2f} correlation with subsequent realized volatility.")
    except (KeyError, TypeError):
        pass
    
    # If no insights were generated, add a default insight
    if not insights:
        insights.append("Insufficient data for detailed analysis. Consider using a date with more comprehensive option chain data and model predictions.")
    
    summary["insights"] = insights
    
    return summary

def visualize_summary_report(summary: Dict) -> go.Figure:
    """
    Create a visual summary report.
    
    Parameters:
    -----------
    summary : Dict
        Summary report dictionary
        
    Returns:
    --------
    go.Figure
        Plotly figure with summary visualizations
    """
    # Extract data
    stock = summary["stock"]
    date = summary["date"]
    
    # Create comparison bar chart
    categories = []
    market_vals = []
    model_vals = []
    realized_vals = []
    
    # Prepare volatility comparison data
    if "market_vs_model" in summary:
        categories.append("Average Volatility")
        
        market_vol = summary["market_vs_model"].get("avg_market_vol", np.nan)
        model_vol = summary["market_vs_model"].get("avg_model_vol", np.nan)
        
        market_vals.append(market_vol)
        model_vals.append(model_vol)
        
        if "market_vs_realized" in summary:
            realized_vol = summary["market_vs_realized"].get("avg_realized_vol", np.nan)
            realized_vals.append(realized_vol)
        else:
            realized_vals.append(None)
    
    # Create figure
    try:
        fig = make_subplots(
            rows=2, cols=2,
            specs=[
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "table", "colspan": 2}, {}]
            ],
            subplot_titles=[
                "Volatility Comparison", 
                "Performance Metrics",
                "Key Insights"
            ],
            vertical_spacing=0.1,
            horizontal_spacing=0.05
        )
        
        # Add bar chart for volatility comparison
        if categories:
            x = ["Market", "Model"]
            if realized_vals[0] is not None and not np.isnan(realized_vals[0]):
                x.append("Realized")
            
            y = [market_vals[0], model_vals[0]]
            if realized_vals[0] is not None and not np.isnan(realized_vals[0]):
                y.append(realized_vals[0])
            
            colors = ['blue', 'red']
            if realized_vals[0] is not None and not np.isnan(realized_vals[0]):
                colors.append('green')
            
            # Format text labels, handling NaN values
            text_labels = []
            for val in y:
                if np.isnan(val):
                    text_labels.append("N/A")
                else:
                    text_labels.append(f"{val:.1%}")
            
            fig.add_trace(
                go.Bar(
                    x=x,
                    y=y,
                    marker_color=colors,
                    text=text_labels,
                    textposition="outside"
                ),
                row=1, col=1
            )
        
        # Add scatter plot for metrics
        metrics_x = ["MSE", "MAE", "Correlation"]
        
        if "market_vs_model" in summary:
            metrics_y_market_model = [
                summary["market_vs_model"].get("mse", np.nan),
                summary["market_vs_model"].get("mae", np.nan),
                summary["market_vs_model"].get("correlation", np.nan)
            ]
            
            # Filter out NaN values
            valid_x = []
            valid_y = []
            for i, val in enumerate(metrics_y_market_model):
                if not np.isnan(val):
                    valid_x.append(metrics_x[i])
                    valid_y.append(val)
            
            if valid_x:
                fig.add_trace(
                    go.Scatter(
                        x=valid_x,
                        y=valid_y,
                        mode="markers+lines",
                        name="Market vs Model",
                        marker=dict(size=10, color="purple")
                    ),
                    row=1, col=2
                )
        
        if "market_vs_realized" in summary:
            metrics_y_market_realized = [
                summary["market_vs_realized"].get("mse", np.nan),
                summary["market_vs_realized"].get("mae", np.nan),
                summary["market_vs_realized"].get("correlation", np.nan)
            ]
            
            # Filter out NaN values
            valid_x = []
            valid_y = []
            for i, val in enumerate(metrics_y_market_realized):
                if not np.isnan(val):
                    valid_x.append(metrics_x[i])
                    valid_y.append(val)
            
            if valid_x:
                fig.add_trace(
                    go.Scatter(
                        x=valid_x,
                        y=valid_y,
                        mode="markers+lines",
                        name="Market vs Realized",
                        marker=dict(size=10, color="orange")
                    ),
                    row=1, col=2
                )
        
        if "model_vs_realized" in summary:
            metrics_y_model_realized = [
                summary["model_vs_realized"].get("mse", np.nan),
                summary["model_vs_realized"].get("mae", np.nan),
                summary["model_vs_realized"].get("correlation", np.nan)
            ]
            
            # Filter out NaN values
            valid_x = []
            valid_y = []
            for i, val in enumerate(metrics_y_model_realized):
                if not np.isnan(val):
                    valid_x.append(metrics_x[i])
                    valid_y.append(val)
            
            if valid_x:
                fig.add_trace(
                    go.Scatter(
                        x=valid_x,
                        y=valid_y,
                        mode="markers+lines",
                        name="Model vs Realized",
                        marker=dict(size=10, color="brown")
                    ),
                    row=1, col=2
                )
        
        # Add table with insights
        insights = summary.get("insights", ["No insights available"])
        cells = []
        for i, insight in enumerate(insights):
            cells.append([f"{i+1}. {insight}"])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=["Key Insights"],
                    fill_color="lightgray",
                    align="left",
                    font=dict(size=12)
                ),
                cells=dict(
                    values=cells,
                    fill_color="white",
                    align="left",
                    font=dict(size=11),
                    height=30
                )
            ),
            row=2, col=1
        )
        
        # Update layout
        fig.update_layout(
            title=f"{stock} Volatility Analysis Summary - {date}",
            height=800,
            width=1000,
            showlegend=True
        )
        
        # Update axes
        fig.update_xaxes(title_text="Source", row=1, col=1)
        fig.update_yaxes(title_text="Volatility", row=1, col=1)
        
        fig.update_xaxes(title_text="Metric", row=1, col=2)
        fig.update_yaxes(title_text="Value", row=1, col=2)
        
        return fig
    except Exception as e:
        # Create simple fallback figure if there's an error
        fallback_fig = go.Figure()
        fallback_fig.add_annotation(
            text=f"Error creating summary visualization: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="red")
        )
        
        fallback_fig.update_layout(
            title=f"{stock} Volatility Analysis Summary - {date}",
            height=400,
            width=800
        )
        
        return fallback_fig

# ======================================================
# Main Interface Functions
# ======================================================

def analyze_volatility(
    option_chain_df: pl.DataFrame,
    ohlcv_df: pl.DataFrame,
    model_results: Dict,
    lgbm_modeling: Any,  # The imported module
    stock: str,
    target_date: Union[str, datetime.date],
    option_type: str = 'Call',
    create_report: bool = True
) -> Dict:
    """
    Comprehensive volatility analysis combining market implied volatility,
    model predictions, and realized volatility.
    
    Parameters:
    -----------
    option_chain_df : pl.DataFrame
        DataFrame with option chain data
    ohlcv_df : pl.DataFrame
        DataFrame with OHLCV data
    model_results : Dict
        Results dictionary from LGBM model training
    lgbm_modeling : Any
        The imported LGBM modeling module
    stock : str
        Stock symbol to analyze
    target_date : Union[str, datetime.date]
        Target date for analysis
    option_type : str
        Option type to analyze ('Call' or 'Put')
    create_report : bool
        Whether to create a summary report
        
    Returns:
    --------
    Dict
        Dictionary with all analysis results
    """
    # Normalize date format
    if isinstance(target_date, str):
        target_date = datetime.datetime.strptime(target_date, "%Y-%m-%d").date()
    
    print(f"Starting volatility analysis for {stock} on {target_date}...")
    
    # Load data for the nearest available dates
    print("Loading option chain data...")
    option_data, option_date = load_option_chain_data(option_chain_df, stock, target_date)
    
    print(f"Using option chain data from {option_date} for {stock}")
    
    print("Loading OHLCV data...")
    ohlcv_data, ohlcv_date = load_ohlcv_data(ohlcv_df, stock, target_date)
    
    print(f"Using OHLCV data from {ohlcv_date} for {stock}")
    
    print("Generating model predictions...")
    pred_df, pred_date = generate_prediction_data(model_results, ohlcv_df, stock, target_date, lgbm_modeling)
    
    print(f"Using prediction date {pred_date} for {stock}")
    
    # Create dashboard
    dashboard = create_integrated_dashboard(
        option_df=option_data,
        pred_df=pred_df,
        stock=stock,
        date=option_date,
        option_type=option_type,
        vol_scale_factor=1.0  # Adjust based on your data scales
    )
    
    # Create summary report if requested
    if create_report:
        print("Creating summary report...")
        try:
            summary = create_summary_report(dashboard)
            summary_fig = visualize_summary_report(summary)
            dashboard["summary"] = summary
            dashboard["summary_fig"] = summary_fig
        except Exception as e:
            print(f"Error creating summary report: {e}")
            dashboard["summary_error"] = str(e)
    
    return dashboard

def demo_analysis(
    option_chain_df: pl.DataFrame,
    ohlcv_df: pl.DataFrame,
    model_results: Dict,
    lgbm_modeling: Any,  # The imported module
    stock: str = "AAPL",
    target_date: str = "2020-02-03",
    option_type: str = "Put"
) -> None:
    """
    Demonstrate the volatility analysis functionality.
    
    Parameters:
    -----------
    option_chain_df : pl.DataFrame
        DataFrame with option chain data
    ohlcv_df : pl.DataFrame
        DataFrame with OHLCV data
    model_results : Dict
        Results dictionary from LGBM model training
    lgbm_modeling : Any
        The imported LGBM modeling module
    stock : str
        Stock symbol to analyze
    target_date : str
        Target date for analysis
    option_type : str
        Option type to analyze ('Call' or 'Put')
    """
    print(f"Running demonstration for {stock} on {target_date}...")
    
    # Run analysis
    results = analyze_volatility(
        option_chain_df=option_chain_df,
        ohlcv_df=ohlcv_df,
        model_results=model_results,
        lgbm_modeling=lgbm_modeling,
        stock=stock,
        target_date=target_date,
        option_type=option_type
    )
    
    # Display key visualizations
    print("\nShowing key visualizations...")
    
    # Show summary if available
    if "summary_fig" in results:
        print("\n1. Analysis Summary")
        results["summary_fig"].show()
    
    # Show market implied volatility surface
    if "surfaces" in results and "market" in results["surfaces"]:
        print("\n2. Market Implied Volatility Surface")
        market_fig = plot_surface(results["surfaces"]["market"])
        market_fig.show()
    
    # Show model predicted volatility surface
    if "surfaces" in results and "model" in results["surfaces"]:
        print("\n3. Model Predicted Volatility Surface")
        model_fig = plot_surface(results["surfaces"]["model"], colorscale="Plasma")
        model_fig.show()
    
    # Show comparison
    if "comparison_figures" in results and "market_vs_model" in results["comparison_figures"]:
        print("\n4. Volatility Surface Comparison")
        results["comparison_figures"]["market_vs_model"].show()
    
    # Show term structure
    if "comparison_figures" in results and "term_structure" in results["comparison_figures"]:
        print("\n5. Volatility Term Structure")
        results["comparison_figures"]["term_structure"].show()
    
    # Show skew comparison
    if "comparison_figures" in results and "skew" in results["comparison_figures"]:
        print("\n6. Volatility Skew Comparison")
        results["comparison_figures"]["skew"].show()
    
    # Show delta comparison
    if "comparison_figures" in results and "market_vs_model_delta" in results["comparison_figures"]:
        print("\n7. Delta Surface Comparison")
        results["comparison_figures"]["market_vs_model_delta"].show()
    
    print("\nDemonstration complete!")
