import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import polars as pl
import datetime
from typing import List, Union, Dict, Tuple, Optional
from scipy.interpolate import CubicSpline

def get_tenor_specific_intensity(stock_data, days):
    """
    Find the volatility intensity for a specific tenor.
    Intensity is scaled between 0 and 1.
    
    Parameters:
    ----------
    stock_data : pl.DataFrame
        Stock data with volatility metrics
    days : int
        Target tenor in days
        
    Returns:
    -------
    float: Volatility intensity (0.0-1.0)
    """
    stock_cols = stock_data.columns
    stock_row = stock_data.row(0)
    
    prefix = 'VolIntensity_'
    future_suffix = '_future'
    
    # First try to find exact match
    intensity_col = f'{prefix}{days}'
    if intensity_col in stock_cols:
        col_idx = stock_cols.index(intensity_col)
        return stock_row[col_idx]
    
    # Look for future indicator
    future_intensity_col = f'{prefix}{days}{future_suffix}'
    if future_intensity_col in stock_cols:
        col_idx = stock_cols.index(future_intensity_col)
        return stock_row[col_idx]
    
    # Find all available windows with this prefix
    intensity_windows = []
    for col in stock_cols:
        if col.startswith(prefix) and not col.endswith(future_suffix):
            try:
                win = int(col[len(prefix):])  # Extract the window size
                intensity_windows.append(win)
            except ValueError:
                continue
    
    if not intensity_windows:
        # Fall back to default value
        return 0.5
    
    # Find closest window
    closest_win = min(intensity_windows, key=lambda x: abs(x - days))
    closest_col = f'{prefix}{closest_win}'
    
    # Use future value if available, otherwise current
    future_col = f'{prefix}{closest_win}{future_suffix}'
    if future_col in stock_cols:
        col_idx = stock_cols.index(future_col)
        return stock_row[col_idx]
    else:
        col_idx = stock_cols.index(closest_col)
        return stock_row[col_idx]

def detect_available_windows(stock_data):
    """
    Detect all available time windows in the data for volatility metrics.
    Also looks for '_pred' suffix columns for compatibility with prediction data.
    """
    # First try regular volatility columns
    vol_windows = safely_get_windows(stock_data, 'YZVol_')
    
    # If no windows found, try prediction columns
    if not vol_windows:
        for col in stock_data.columns:
            if col.startswith('YZVol_') and ('_future' in col or '_pred' in col):
                try:
                    # Extract the window size - get the number between 'YZVol_' and '_future' or '_pred'
                    window_part = col.replace('YZVol_', '').split('_')[0]
                    window = int(window_part)
                    vol_windows.append(window)
                except (ValueError, IndexError):
                    continue
        vol_windows = sorted(list(set(vol_windows)))  # Remove duplicates and sort
    
    if not vol_windows:
        # If still no volatility windows, try looking for any numeric windows in any column
        all_windows = []
        for col in stock_data.columns:
            for prefix in ['YZVol_', 'VolSkew_', 'VolCurvature_', 'WingRatio_', 'MeanReversion_']:
                if col.startswith(prefix):
                    # Try to extract window from column name
                    try:
                        # Remove prefix and split by underscore
                        parts = col[len(prefix):].split('_')
                        # First part might be the window number
                        if parts and parts[0].isdigit():
                            all_windows.append(int(parts[0]))
                    except ValueError:
                        continue
        return sorted(list(set(all_windows)))
    
    # Return all volatility windows
    return sorted(vol_windows)

def safely_get_windows(stock_data, prefix):
    """
    Safely detect available windows for a given parameter prefix using Polars.
    Returns all windows that have the given prefix in column names.
    Also handles prediction columns with '_pred' suffix.
    """
    windows = []
    
    # Get all windows from columns with this prefix
    for col in stock_data.columns:
        if col.startswith(prefix):
            # Try to extract the numeric part after the prefix
            try:
                # Handle both regular and prediction columns
                col_without_prefix = col[len(prefix):]
                # If it has _future_pred, _future, or _pred, remove that part
                if '_future_pred' in col_without_prefix:
                    window_part = col_without_prefix.split('_future_pred')[0]
                elif '_future' in col_without_prefix:
                    window_part = col_without_prefix.split('_future')[0]
                elif '_pred' in col_without_prefix:
                    window_part = col_without_prefix.split('_pred')[0]
                else:
                    window_part = col_without_prefix
                
                # Try to convert the remaining part to an integer
                window = int(window_part)
                windows.append(window)
            except ValueError:
                # Skip if not a valid window number
                continue
    
    # For volatility metrics, we want windows that have either future values or predictions
    if prefix in ['YZVol_', 'VolSkew_', 'VolCurvature_', 'WingRatio_', 'MeanReversion_', 'VolOfVol_', 'PriceVolCorr_']:
        windows_with_values = []
        for window in windows:
            future_col = f"{prefix}{window}_future"
            pred_col = f"{prefix}{window}_future_pred"
            alt_pred_col = f"{prefix}{window}_pred"
            
            if (future_col in stock_data.columns or 
                pred_col in stock_data.columns or 
                alt_pred_col in stock_data.columns):
                windows_with_values.append(window)
        windows = windows_with_values
    
    return sorted(windows)

def heston_vol_approx(moneyness, tau, v0, theta, kappa, sigma, rho):
    """
    Improved approximation of the Heston model implied volatility with better numerical stability
    and more realistic behavior for extreme parameter values and short tenors.
    
    Parameters:
    ----------
    moneyness : float or array
        Log-moneyness log(K/S)
    tau : float
        Time to expiration in years
    v0 : float
        Initial variance
    theta : float
        Long-term variance
    kappa : float
        Mean reversion speed
    sigma : float
        Volatility of variance
    rho : float
        Price-volatility correlation
        
    Returns:
    -------
    float or array : Implied volatility
    """
    # Parameter bounds for stability
    v0 = max(0.0001, min(0.25, v0))  # 1% to 50% volatility
    theta = max(0.0001, min(0.25, theta))
    kappa = max(0.1, min(10.0, kappa))  # Reasonable mean reversion range
    sigma = max(0.01, min(2.0, sigma))  # Bound vol-of-vol
    rho = max(-0.95, min(0.95, rho))  # Avoid extreme correlation
    
    # Effective time factor (prevents extreme behavior for very short options)
    tau_eff = max(0.01, tau)  # Minimum effective time of 0.01 years (~2-3 days) 
    
    # ATM volatility level
    atm_vol = np.sqrt(v0)
    
    # Effective variance calculation with smoother blending
    # This gives the term-structure effect
    weight = 1.0 - np.exp(-kappa * tau_eff)
    v_eff = v0 * (1.0 - weight) + theta * weight
    
    # Base volatility (ATM level)
    base_vol = np.sqrt(v_eff)
    
    # First-order skew effect (ATM slope)
    # More conservative scaling for very short expirations
    skew_strength = np.clip(1.0 - np.exp(-2 * tau_eff), 0.05, 1.0)
    # Scale down skew effect to reduce volatility range
    skew_effect = rho * sigma * moneyness * skew_strength * 0.2  # Reduced from 0.4
    
    # Second-order curvature effect (smile)
    # Higher for shorter terms but with reasonable bounds
    curve_factor = (1 - rho**2) * sigma**2 * np.exp(-0.5 * tau_eff)
    curve_factor = min(curve_factor, 3.0)  # Cap for stability (reduced from 5.0)
    # Scale down curvature effect to reduce volatility range
    smile_effect = curve_factor * (moneyness**2) * 0.1  # Reduced from 0.2
    
    # Wing adjustment with smoother transition
    left_wing = np.where(
        moneyness < -0.05, 
        0.15 * np.power(np.abs(moneyness + 0.05), 1.5) * np.exp(-0.5 * tau_eff),
        0
    )
    
    right_wing = np.where(
        moneyness > 0.05, 
        0.1 * np.power(np.abs(moneyness - 0.05), 1.5) * np.exp(-0.5 * tau_eff),
        0
    )
    
    # Asymmetry adjustment based on correlation
    rho_effect = 0.3 * rho  # Reduced from 0.5
    asymmetry = rho_effect * (left_wing - 0.5 * right_wing) 
    
    # Combine effects with more controlled scaling
    # For very short maturities, scale back the effects proportionally
    if tau < 0.04:  # Less than ~10 days
        time_scale = np.sqrt(tau / 0.04)  # Square root scaling
        total_vol = base_vol * (1.0 + (skew_effect + smile_effect + asymmetry) * time_scale)
    else:
        total_vol = base_vol * (1.0 + skew_effect + smile_effect + asymmetry)
    
    # More conservative bounds to reduce the range of volatility values
    total_vol = np.maximum(total_vol, 0.5 * base_vol)  # Minimum 50% of ATM vol
    total_vol = np.minimum(total_vol, 1.5 * base_vol)  # Maximum 150% of ATM vol
    
    return total_vol

def generate_vol_surface(stock_data, times, moneyness, time_days):
    """
    Generate a volatility surface using a robust model-based approach with continuous
    volatility intensity adjustments for more realistic surfaces.
    
    Parameters:
    ----------
    stock_data : pl.DataFrame
        DataFrame row with stock volatility metrics
    times : np.array
        Array of time points (in years)
    moneyness : np.array
        Array of moneyness values (log(K/S))
    time_days : list
        Array of time points (in days)
        
    Returns:
    -------
    np.array : 2D array of volatility values
    """
    # Create output grid - swap dimensions to make moneyness first and time second
    Z = np.zeros((len(times), len(moneyness)))
    stock_cols = stock_data.columns
    stock_row = stock_data.row(0)
    
    # --- Get term structure of volatility (ATM volatilities) ---
    atm_vols = []
    for days in time_days:
        vol_windows = safely_get_windows(stock_data, 'YZVol_')
        if not vol_windows:
            atm_vols.append(0.25)  # Default if no data
            continue
            
        closest_win = min(vol_windows, key=lambda x: abs(x - days))
        future_col = f'YZVol_{closest_win}_future'
        
        if future_col in stock_cols:
            col_idx = stock_cols.index(future_col)
            atm_vols.append(stock_row[col_idx])
        else:
            # Use current volatility as fallback
            current_col = f'YZVol_{closest_win}' 
            if current_col in stock_cols:
                col_idx = stock_cols.index(current_col)
                atm_vols.append(stock_row[col_idx])
            else:
                atm_vols.append(0.25)  # Default if no data
    
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
    
    mr_windows = safely_get_windows(stock_data, 'MeanReversion_')
    if mr_windows:
        closest_mr_win = min(mr_windows, key=lambda x: abs(x - medium_term_days))
        mr_col = f'MeanReversion_{closest_mr_win}_future'
        if mr_col in stock_cols:
            col_idx = stock_cols.index(mr_col)
            base_kappa = stock_row[col_idx]
        else:
            base_kappa = 1.5  # Default mean reversion
    else:
        base_kappa = 1.5
        
    # Volatility of volatility
    vv_windows = safely_get_windows(stock_data, 'VolOfVol_')
    if vv_windows:
        closest_vv_win = min(vv_windows, key=lambda x: abs(x - medium_term_days))
        vv_col = f'VolOfVol_{closest_vv_win}_future'
        if vv_col in stock_cols:
            col_idx = stock_cols.index(vv_col)
            base_volvol = stock_row[col_idx]
        else:
            base_volvol = 0.3  # Default volvol
    else:
        base_volvol = 0.3
        
    # Price-vol correlation
    corr_windows = safely_get_windows(stock_data, 'PriceVolCorr_')
    if corr_windows:
        closest_corr_win = min(corr_windows, key=lambda x: abs(x - medium_term_days))
        corr_col = f'PriceVolCorr_{closest_corr_win}_future'
        if corr_col in stock_cols:
            col_idx = stock_cols.index(corr_col)
            base_rho = stock_row[col_idx]
        else:
            base_rho = -0.5  # Default correlation (negative for equities)
    else:
        base_rho = -0.5
    
    # --- Generate consistent surface across term structure ---
    
    # Fix-up parameters for numerical stability
    base_kappa = np.clip(base_kappa, 0.1, 5.0)
    base_volvol = np.clip(base_volvol, 0.01, 1.0)
    base_rho = np.clip(base_rho, -0.9, 0.5)
    
    # Scale down vol-of-vol to reduce the overall volatility range
    base_volvol *= 0.7  # Reduce by 30%
    
    # Process each time slice with term-structure consistency
    for i, (t, days) in enumerate(zip(times, time_days)):
        # Get tenor-specific volatility intensity (continuous value scaled 0 to 1)
        vol_intensity = get_tenor_specific_intensity(stock_data, days)
        
        atm_vol = atm_vols[i]
        
        # Scale parameters appropriately with tenor and volatility intensity
        # Use more moderate scaling to reduce overall volatility range
        if t < 0.05:  # Very short term (<13 days)
            time_factor = 0.05 / max(0.01, t)
            time_factor = min(time_factor, 2.0)  # Reduced from 3.0
            kappa = base_kappa
            volvol = base_volvol * 0.85  # Increased from 0.8
            rho = base_rho * 1.1  # Reduced from 1.2
        elif t < 0.2:  # Short term (< ~50 days)
            time_factor = np.sqrt(0.2 / t)
            time_factor = min(time_factor, 1.5)  # Reduced from 2.0
            kappa = base_kappa
            volvol = base_volvol * 0.9
            rho = base_rho * 1.05  # Reduced from 1.1
        elif t > 0.5:  # Long term (> ~125 days)
            time_factor = np.cbrt(t / 0.5)
            kappa = base_kappa * 0.95  # Increased from 0.9
            volvol = base_volvol * 0.85 / time_factor  # Increased from 0.8
            rho = base_rho * 0.95  # Increased from 0.9
        else:  # Medium term
            kappa = base_kappa
            volvol = base_volvol
            rho = base_rho
        
        # Volatility intensity adjustments using continuous scale (0 to 1)
        # Use more moderate scaling to reduce overall volatility range
        if vol_intensity <= 0.33:  # Normal volatility range
            intensity_factor = 0.85 + (vol_intensity / 0.33) * 0.15  # Reduced range
            volvol *= intensity_factor
            rho = base_rho * intensity_factor
        elif vol_intensity <= 0.66:  # Elevated volatility range
            elevated_factor = 1.0 + ((vol_intensity - 0.33) / 0.33) * 0.2  # Reduced from 0.3
            volvol *= elevated_factor
            rho_factor = 1.0 + ((vol_intensity - 0.33) / 0.33) * 0.15  # Reduced from 0.2
            rho = max(-0.95, base_rho * rho_factor)
        else:  # High volatility range
            high_factor = 1.2 + ((vol_intensity - 0.66) / (1 - 0.66)) * 0.15  # Reduced from 1.3 and 0.2
            high_factor = min(high_factor, 1.5)  # Reduced from 1.8
            volvol *= high_factor
            rho_factor = 1.15 + ((vol_intensity - 0.66) / (1 - 0.66)) * 0.2  # Reduced from 1.2 and 0.3
            rho_factor = min(rho_factor, 1.4)  # Reduced from 1.7
            rho = max(-0.95, base_rho * rho_factor)
        
        # Convert volatility to variance (v = σ²)
        v0 = atm_vol**2
        
        # Approximate long-term variance based on term structure
        if i < len(times) - 1:
            forward_vol = atm_vols[min(i + 1, len(atm_vols) - 1)]
            theta = forward_vol**2
        else:
            theta = (atm_vol * 0.97)**2  # Increased from 0.95
        
        theta = max(0.0004, theta)
        
        # --- Apply a more robust model for each slice ---
        v_t = v0 + (theta - v0) * (1 - np.exp(-kappa * t))
        base_vol = np.sqrt(v_t)
        
        surface_slice = np.ones_like(moneyness) * base_vol
        
        # 1. Skew effect - scale down to reduce volatility range
        skew_strength = np.sqrt(t) * (1 - np.exp(-kappa * t)) / (kappa * t) * 0.3  # Reduced from 0.4
        skew_effect = rho * volvol * moneyness * skew_strength
        
        # 2. Curvature effect - scale down to reduce volatility range
        curvature_strength = (1 - rho**2) * volvol**2 * t * np.exp(-kappa * t) * 0.15  # Reduced from 0.2
        curvature_effect = curvature_strength * moneyness**2
        
        total_effect = 1.0 + skew_effect + curvature_effect
        surface_slice = base_vol * total_effect
        
        # --- Fine-tune with known values if available ---
        skew_windows = safely_get_windows(stock_data, 'VolSkew_')
        curve_windows = safely_get_windows(stock_data, 'VolCurvature_')
        
        if skew_windows and curve_windows:
            closest_skew_win = min(skew_windows, key=lambda x: abs(x - days))
            closest_curve_win = min(curve_windows, key=lambda x: abs(x - days))
            
            skew_col = f'VolSkew_{closest_skew_win}_future'
            curve_col = f'VolCurvature_{closest_curve_win}_future'
            
            if skew_col in stock_cols and curve_col in stock_cols:
                skew_idx = stock_cols.index(skew_col)
                curve_idx = stock_cols.index(curve_col)
                target_skew = stock_row[skew_idx]
                target_curve = max(0, min(3.0, stock_row[curve_idx]))  # Reduced max from 5.0
                
                atm_idx = np.abs(moneyness).argmin()
                down_idx = np.abs(moneyness + 0.05).argmin()
                up_idx = np.abs(moneyness - 0.05).argmin()
                
                current_skew = (surface_slice[down_idx] - surface_slice[up_idx]) / 0.1
                current_curve = (surface_slice[down_idx] + surface_slice[up_idx] - 2 * surface_slice[atm_idx]) / 0.05**2
                
                # Scale down adjustments to reduce volatility range
                skew_adjust = (target_skew - current_skew) * 0.2 * moneyness  # Reduced from 0.3
                curve_adjust = (target_curve - current_curve) * 0.07 * moneyness**2  # Reduced from 0.1
                
                max_skew_adjust = 0.15 * base_vol  # Reduced from 0.2
                max_curve_adjust = 0.07 * base_vol  # Reduced from 0.1
                
                skew_adjust = np.clip(skew_adjust, -max_skew_adjust, max_skew_adjust)
                curve_adjust = np.clip(curve_adjust, -max_curve_adjust, max_curve_adjust)
                
                surface_slice = surface_slice + skew_adjust + curve_adjust
        
        # More moderate bounds to reduce volatility range
        surface_slice = np.clip(surface_slice, 0.6 * base_vol, 1.4 * base_vol)  # Tightened from 0.25-2.0
        Z[i, :] = surface_slice
    
    # --- Smoothing across term structure ---
    for j in range(len(moneyness)):
        Z[:, j] = savgol_smooth(Z[:, j])
    
    for i in range(len(times) - 1):
        if times[i+1] > times[i]:
            var_t1 = Z[i, :]**2 
            var_t2 = Z[i+1, :]**2
            var_growth = var_t2 - var_t1
            neg_growth_mask = var_growth < 0
            if np.any(neg_growth_mask):
                min_growth = 0.0001 * (times[i+1] - times[i])
                var_t2[neg_growth_mask] = var_t1[neg_growth_mask] + min_growth
                Z[i+1, :] = np.sqrt(var_t2)
    
    return Z

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

def visualize_vol_surface(data, stock_symbol, reference_date, windows=None, num_windows=None, print_params=True):
    """
    Visualize the volatility surface for a stock at a specific date.
    
    Parameters:
    ----------
    data : pl.DataFrame
        DataFrame with volatility metrics
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL')
    reference_date : date or str
        Reference date for the surface (e.g., '2020-03-15')
    windows : list, optional
        Specific time windows to use (in days)
    num_windows : int, optional
        Number of windows to select if windows not specified
    print_params : bool, optional
        Whether to print parameters to console (default: True)
        
    Returns:
    -------
    matplotlib figure object
    """
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    stock_data = data.filter(
        (pl.col('act_symbol') == stock_symbol) & 
        (pl.col('date') == reference_date)
    )
    
    if stock_data.height == 0:
        raise ValueError(f"No data found for {stock_symbol} on {reference_date}")
    
    available_windows = detect_available_windows(stock_data)
    
    if not available_windows:
        raise ValueError(f"No volatility windows found for {stock_symbol}")
    
    if windows is None:
        selected_windows = available_windows
        if num_windows and len(available_windows) > num_windows:
            sorted_windows = sorted(available_windows)
            indices = np.unique(np.geomspace(0, len(sorted_windows)-1, num_windows).astype(int))
            selected_windows = [sorted_windows[i] for i in indices]
    else:
        selected_windows = []
        for win in windows:
            closest = min(available_windows, key=lambda x: abs(x - win))
            if closest not in selected_windows:
                selected_windows.append(closest)
    
    selected_windows = sorted(list(set(selected_windows)))
    
    print(f"Using {len(selected_windows)} windows: {selected_windows}")
    
    if print_params:
        print_volatility_parameters(data, stock_symbol, reference_date, selected_windows)
    
    times = np.array([w/252 for w in selected_windows])
    time_labels = [f'{w}d' for w in selected_windows]
    
    moneyness = np.linspace(-0.12, 0.12, 11)
    
    Z = generate_vol_surface(stock_data, times, moneyness, selected_windows)
    
    # Create meshgrid with time (days) and moneyness
    X, Y = np.meshgrid(moneyness, selected_windows)
    
    vols = []
    for days in selected_windows:
        vol_col = f'YZVol_{days}_future'
        if vol_col in stock_data.columns:
            vols.append(stock_data.select(vol_col).row(0)[0])
        else:
            vol_windows = safely_get_windows(stock_data, 'YZVol_')
            if vol_windows:
                closest_window = min(vol_windows, key=lambda x: abs(x - days))
                vols.append(stock_data.select(f'YZVol_{closest_window}_future').row(0)[0])
            else:
                vols.append(0.2)
    
    vols = np.array(vols)
    
    fig = plt.figure(figsize=(16, 12))
    
    # Format the date properly for the title
    if isinstance(reference_date, str):
        formatted_date = reference_date
    elif hasattr(reference_date, 'strftime'):
        formatted_date = reference_date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(reference_date).split()[0]  # Extract just the date part
    
    # Modified title to clearly indicate these are "future values" not predictions
    fig.suptitle(f'Volatility Surface (Future Values) for {stock_symbol} on {formatted_date}', fontsize=16)
    
    ax1 = fig.add_subplot(221, projection='3d')
    
    surf = ax1.plot_surface(X, Y, Z, cmap=cm.viridis, linewidth=0, antialiased=True, alpha=0.8)
    fig.colorbar(surf, ax=ax1, shrink=0.6, label='Implied Volatility')
    
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlabel('Moneyness (log(K/S))')
    ax1.set_ylabel('Days to Expiration')
    ax1.set_zlabel('Implied Volatility')
    ax1.set_title('3D Volatility Surface')
    ax1.view_init(elev=30, azim=-60)
    
    ax2 = fig.add_subplot(222)
    
    if len(selected_windows) >= 4:
        smooth_days = np.linspace(min(selected_windows), max(selected_windows), 100)
        cs = CubicSpline(selected_windows, vols)
        smooth_vols = cs(smooth_days)
        ax2.plot(smooth_days, smooth_vols, '-', linewidth=2, color='blue', alpha=0.7)
        ax2.plot(selected_windows, vols, 'o', color='blue', markersize=6)
    else:
        ax2.plot(selected_windows, vols, 'o-', linewidth=2, color='blue', markersize=6)
    
    vol_std = np.std(vols) * 0.5
    ax2.fill_between(selected_windows, vols - vol_std, vols + vol_std, color='blue', alpha=0.15)
    
    ax2.set_xlabel('Days to Expiration')
    ax2.set_ylabel('Volatility')
    ax2.set_title('Volatility Term Structure')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(selected_windows) * 1.05)
    ax2.set_ylim(max(0, min(vols) * 0.9), max(vols) * 1.1)
    
    ax3 = fig.add_subplot(223)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_windows)))
    
    for i, days in enumerate(selected_windows):
        smile_vol = Z[i, :]
        ax3.plot(moneyness, smile_vol, '-', color=colors[i], linewidth=2, label=f'{days}d')
    
    ax3.axvline(x=0, color='black', linestyle='--', alpha=0.5, label='ATM')
    
    ax3.set_xlabel('Moneyness (log(K/S))')
    ax3.set_ylabel('Implied Volatility')
    ax3.set_title('Volatility Smile at Different Expirations')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(5, len(selected_windows)), frameon=True)
    
    ax4 = fig.add_subplot(224)
    ax4.axis('off')
    
    params_text = create_parameter_summary(stock_data, stock_symbol, selected_windows, vols)
    
    fontsize = 11
    if len(selected_windows) > 8:
        fontsize = 10
    if len(selected_windows) > 10:
        fontsize = 9
        
    ax4.text(0.5, 0.5, params_text, va='center', ha='center', fontsize=fontsize, 
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8))
    
    plt.tight_layout(rect=[0, 0, 1, 0.95], h_pad=1.0, w_pad=1.0)
    
    return fig

def visualize_from_predictions(prediction_df, stock_symbol, reference_date, windows=None, num_windows=None, print_params=True):
    """
    Visualize volatility surface from prediction data generated by LGBM models.
    
    Parameters:
    ----------
    prediction_df : pl.DataFrame
        DataFrame containing a single row with predicted volatility metrics
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL')
    reference_date : date or str
        Reference date for the surface (e.g., '2020-03-15')
    windows : list, optional
        Specific time windows to use (in days)
    num_windows : int, optional
        Number of windows to select if windows not specified
    print_params : bool, optional
        Whether to print parameters to console (default: True)
        
    Returns:
    -------
    matplotlib figure object
    """
    # Use our existing visualization function, but first make sure the data is properly formatted
    if prediction_df.height != 1:
        raise ValueError("Prediction DataFrame must contain exactly one row.")
    
    # Verify the date in the prediction dataframe
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    # Extract the date from the prediction dataframe
    pred_date = prediction_df["date"][0]
    
    # Create a new DataFrame with the predicted values where _pred suffix is removed
    # This allows it to be used with our existing visualization function
    cleaned_df = {"act_symbol": stock_symbol, "date": pred_date}
    
    # Process columns to prioritize predictions
    # First, find all prediction columns (ending with _pred)
    pred_cols = {}
    for col in prediction_df.columns:
        if col.endswith('_pred'):
            orig_col = col[:-5]  # Remove _pred suffix
            pred_cols[orig_col] = col
    
    # Then, handle all columns in prediction_df
    for col in prediction_df.columns:
        # Skip if it's act_symbol or date (already handled)
        if col in ['act_symbol', 'date']:
            continue
        
        # If this is a prediction column (_pred suffix), we already handled it in the pred_cols dictionary
        if col.endswith('_pred'):
            continue
            
        # If we have a prediction for this column, don't add the non-prediction version
        if col in pred_cols:
            continue
            
        # Otherwise, add the regular column
        cleaned_df[col] = prediction_df[col][0]
    
    # Now add all prediction values (with _pred suffix removed)
    for orig_col, pred_col in pred_cols.items():
        cleaned_df[orig_col] = prediction_df[pred_col][0]
    
    # Convert to DataFrame
    viz_ready_df = pl.DataFrame([cleaned_df])
    
    # Call the existing visualization function but override the title after
    fig = visualize_vol_surface(
        data=viz_ready_df,
        stock_symbol=stock_symbol,
        reference_date=pred_date,
        windows=windows,
        num_windows=num_windows,
        print_params=print_params
    )
    
    # Update the title to clearly indicate these are model predictions
    if isinstance(pred_date, str):
        formatted_date = pred_date
    elif hasattr(pred_date, 'strftime'):
        formatted_date = pred_date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(pred_date).split()[0]
    
    fig.suptitle(f'Volatility Surface (Model Predictions) for {stock_symbol} on {formatted_date}', fontsize=16)
    
    return fig

def create_plotly_vol_surface(data, stock_symbol, reference_date, windows=None, num_windows=None):
    """
    Create an interactive Plotly 3D surface plot of the volatility surface.
    
    Parameters:
    ----------
    data : pl.DataFrame
        DataFrame with volatility metrics
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL')
    reference_date : date or str
        Reference date for the surface (e.g., '2020-03-15')
    windows : list, optional
        Specific time windows to use (in days)
    num_windows : int, optional
        Number of windows to select if windows not specified
        
    Returns:
    -------
    plotly.graph_objects.Figure
    """
    try:
        import plotly.graph_objects as go
    except ImportError:
        raise ImportError("Plotly is required for interactive visualization. Install with: pip install plotly")
    
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    stock_data = data.filter(
        (pl.col('act_symbol') == stock_symbol) & 
        (pl.col('date') == reference_date)
    )
    
    if stock_data.height == 0:
        raise ValueError(f"No data found for {stock_symbol} on {reference_date}")
    
    available_windows = detect_available_windows(stock_data)
    
    if not available_windows:
        raise ValueError(f"No volatility windows found for {stock_symbol}")
    
    if windows is None:
        selected_windows = available_windows
        if num_windows and len(available_windows) > num_windows:
            sorted_windows = sorted(available_windows)
            indices = np.unique(np.geomspace(0, len(sorted_windows)-1, num_windows).astype(int))
            selected_windows = [sorted_windows[i] for i in indices]
    else:
        selected_windows = []
        for win in windows:
            closest = min(available_windows, key=lambda x: abs(x - win))
            if closest not in selected_windows:
                selected_windows.append(closest)
    
    selected_windows = sorted(list(set(selected_windows)))
    
    times = np.array([w/252 for w in selected_windows])
    
    moneyness = np.linspace(-0.12, 0.12, 11)
    
    Z = generate_vol_surface(stock_data, times, moneyness, selected_windows)
    
    # Format the date properly for the title
    if isinstance(reference_date, str):
        formatted_date = reference_date
    elif hasattr(reference_date, 'strftime'):
        formatted_date = reference_date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(reference_date).split()[0]  # Extract just the date part
    
    # Create the surface with days on y-axis
    fig = go.Figure(data=[go.Surface(
        x=moneyness,
        y=selected_windows,  # Use days directly
        z=Z,
        colorscale='Viridis',
        colorbar=dict(title="Implied Volatility", thickness=20, len=0.8),
        contours={
            "x": {"show": True, "start": -0.12, "end": 0.12, "size": 0.05, "color":"black"},
            "y": {"show": True, "start": min(selected_windows), "end": max(selected_windows), "size": (max(selected_windows)-min(selected_windows))/5, "color":"black"},
            "z": {"show": True, "start": Z.min(), "end": Z.max(), "size": (Z.max()-Z.min())/10}
        },
    )])
    
    fig.update_layout(
        title=dict(
            text=f'Interactive Volatility Surface (Future Values) for {stock_symbol} on {formatted_date}',
            font=dict(size=20),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis_title='Moneyness (log(K/S))',
            yaxis_title='Days to Expiration',
            zaxis_title='Implied Volatility',
            xaxis=dict(tickmode='array', tickvals=[-0.12, -0.075, -0.033, 0, 0.033, 0.075, 0.12]),
            yaxis=dict(tickmode='array', tickvals=selected_windows, ticktext=[f'{d}d' for d in selected_windows]),
            aspectratio=dict(x=1.2, y=1, z=0.7),
            camera=dict(eye=dict(x=1.5, y=-1.5, z=1))
        ),
        width=900,
        height=700,
        margin=dict(l=65, r=50, b=65, t=90),
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    # Add intensity annotations for all available windows
    annotations = []
    for i, days in enumerate(selected_windows):
        intensity = get_tenor_specific_intensity(stock_data, days)
        
        # Assign regime name based on intensity value
        if intensity < 0.33:
            regime_name = "Normal"
            color = "blue"
        elif intensity < 0.66:
            regime_name = "Elevated"
            color = "orange"
        else:
            regime_name = "High"
            color = "red"
            
        # Add annotation for each window's volatility regime
        annotations.append(dict(
            x=0.95,
            y=0.9 - i * 0.03,
            xref="paper",
            yref="paper",
            text=f"{days}d: {regime_name} Volatility (Intensity: {intensity:.2f})",
            showarrow=False,
            font=dict(
                size=10,
                color=color
            )
        ))
    
    fig.update_layout(annotations=annotations)
    
    # Update hover template to show days instead of years
    fig.update_traces(
        hovertemplate='<b>Moneyness</b>: %{x:.3f}<br>' +
                      '<b>Days to Expiration</b>: %{y}<br>' +
                      '<b>Volatility</b>: %{z:.4f}<extra></extra>'
    )
    
    return fig

def create_plotly_from_predictions(prediction_df, stock_symbol, reference_date, windows=None, num_windows=None):
    """
    Create an interactive Plotly 3D surface plot from LGBM prediction data.
    
    Parameters:
    ----------
    prediction_df : pl.DataFrame
        DataFrame containing a single row with predicted volatility metrics
    stock_symbol : str
        Stock ticker symbol (e.g., 'AAPL')
    reference_date : date or str
        Reference date for the surface (e.g., '2020-03-15')
    windows : list, optional
        Specific time windows to use (in days)
    num_windows : int, optional
        Number of windows to select if windows not specified
        
    Returns:
    -------
    plotly.graph_objects.Figure
    """
    # Similar to visualize_from_predictions but for Plotly
    if prediction_df.height != 1:
        raise ValueError("Prediction DataFrame must contain exactly one row.")
    
    # Verify the date in the prediction dataframe
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    # Extract the date from the prediction dataframe
    pred_date = prediction_df["date"][0]
    
    # Create a new DataFrame with the predicted values where _pred suffix is removed
    cleaned_df = {"act_symbol": stock_symbol, "date": pred_date}
    
    # Process columns to prioritize predictions
    # First, find all prediction columns (ending with _pred)
    pred_cols = {}
    for col in prediction_df.columns:
        if col.endswith('_pred'):
            orig_col = col[:-5]  # Remove _pred suffix
            pred_cols[orig_col] = col
    
    # Then, handle all columns in prediction_df
    for col in prediction_df.columns:
        # Skip if it's act_symbol or date (already handled)
        if col in ['act_symbol', 'date']:
            continue
        
        # If this is a prediction column (_pred suffix), we already handled it
        if col.endswith('_pred'):
            continue
            
        # If we have a prediction for this column, don't add the non-prediction version
        if col in pred_cols:
            continue
            
        # Otherwise, add the regular column
        cleaned_df[col] = prediction_df[col][0]
    
    # Now add all prediction values (with _pred suffix removed)
    for orig_col, pred_col in pred_cols.items():
        cleaned_df[orig_col] = prediction_df[pred_col][0]
    
    # Convert to DataFrame
    viz_ready_df = pl.DataFrame([cleaned_df])
    
    # Call the existing Plotly visualization function
    fig = create_plotly_vol_surface(
        data=viz_ready_df,
        stock_symbol=stock_symbol,
        reference_date=pred_date,
        windows=windows,
        num_windows=num_windows
    )
    
    # Update the title to clearly indicate these are model predictions
    if isinstance(pred_date, str):
        formatted_date = pred_date
    elif hasattr(pred_date, 'strftime'):
        formatted_date = pred_date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(pred_date).split()[0]
        
    fig.update_layout(
        title=dict(
            text=f'Interactive Volatility Surface (Model Predictions) for {stock_symbol} on {formatted_date}',
            font=dict(size=20),
            x=0.5,
            xanchor='center'
        )
    )
    
    return fig

def create_parameter_summary(stock_data, stock_symbol, selected_windows, vols):
    """Create a formatted parameter summary for the figure text box with comprehensive parameter display."""
    stock_cols = stock_data.columns
    stock_row = stock_data.row(0)
    
    params_text = f"Heston Model Parameters for {stock_symbol}\n\n"
    
    # Display volatility term structure - show all windows
    term_structure_text = []
    
    for i, days in enumerate(selected_windows):
        term_structure_text.append(f"{days}d: {vols[i]:.4f}")
    
    # Split into multiple lines if there are many windows
    if len(selected_windows) > 5:
        # Create chunks of values
        chunks = []
        chunk_size = 5
        for i in range(0, len(term_structure_text), chunk_size):
            chunks.append(", ".join(term_structure_text[i:i+chunk_size]))
        
        params_text += "Volatility Term Structure:\n" + "\n".join(chunks) + "\n\n"
    else:
        params_text += "Volatility Term Structure: " + ", ".join(term_structure_text) + "\n\n"
    
    # Get medium term window for reference
    medium_idx = len(selected_windows) // 2
    medium_window = selected_windows[medium_idx]
    
    # Get Heston model parameters
    params_text += "Heston Parameters (Medium Term):\n"
    
    # Mean reversion parameter
    mr_windows = safely_get_windows(stock_data, 'MeanReversion_')
    if mr_windows:
        closest_win = min(mr_windows, key=lambda x: abs(x - medium_window))
        mr_col = f'MeanReversion_{closest_win}_future'
        if mr_col in stock_cols:
            col_idx = stock_cols.index(mr_col)
            kappa = stock_row[col_idx]
            params_text += f"  Mean Reversion (κ): {kappa:.4f}"
        else:
            params_text += f"  Mean Reversion (κ): 2.0 (default)"
    else:
        params_text += f"  Mean Reversion (κ): 2.0 (default)"
    
    # Volatility of volatility parameter
    vv_windows = safely_get_windows(stock_data, 'VolOfVol_')
    if vv_windows:
        closest_win = min(vv_windows, key=lambda x: abs(x - medium_window))
        vv_col = f'VolOfVol_{closest_win}_future'
        if vv_col in stock_cols:
            col_idx = stock_cols.index(vv_col)
            volvol = stock_row[col_idx]
            params_text += f", Vol of Vol (σᵥ): {volvol:.4f}\n"
        else:
            params_text += f", Vol of Vol (σᵥ): 0.3 (default)\n"
    else:
        params_text += f", Vol of Vol (σᵥ): 0.3 (default)\n"
    
    # Price-vol correlation parameter
    pv_windows = safely_get_windows(stock_data, 'PriceVolCorr_')
    if pv_windows:
        closest_win = min(pv_windows, key=lambda x: abs(x - medium_window))
        pv_col = f'PriceVolCorr_{closest_win}_future'
        if pv_col in stock_cols:
            col_idx = stock_cols.index(pv_col)
            rho = stock_row[col_idx]
            params_text += f"  Price-Vol Corr (ρ): {rho:.4f}\n"
        else:
            params_text += f"  Price-Vol Corr (ρ): -0.7 (default)\n"
    else:
        params_text += f"  Price-Vol Corr (ρ): -0.7 (default)\n"
    
    # Volatility intensity for all windows
    params_text += "\nVolatility Regime & Intensity (Continuous, 0-1 scale):\n"
    
    # Display intensity for all windows, possibly in multiple lines
    intensity_text = []
    for win in selected_windows:
        intensity = get_tenor_specific_intensity(stock_data, win)
            
        # Assign regime based on intensity
        if intensity < 0.33:
            regime_text = "Normal"
        elif intensity < 0.66:
            regime_text = "Elevated"
        else:
            regime_text = "High"
            
        intensity_text.append(f"{win}d: {regime_text} (Intensity: {intensity:.2f})")
    
    # Split into multiple lines if there are many windows
    if len(selected_windows) > 3:
        # Create chunks of values
        chunks = []
        chunk_size = 3
        for i in range(0, len(intensity_text), chunk_size):
            chunks.append(", ".join(intensity_text[i:i+chunk_size]))
        
        params_text += "  " + "\n  ".join(chunks) + "\n"
    else:
        params_text += "  " + ", ".join(intensity_text) + "\n"
    
    # Skew and curvature parameters for all windows
    params_text += "\nSkew & Curvature Parameters:\n"
    
    # Collect skew and curve parameters for all windows
    skew_curve_text = []
    for win in selected_windows:
        # Try to get skew parameter
        skew_val = "N/A"
        skew_windows = safely_get_windows(stock_data, 'VolSkew_')
        if skew_windows:
            closest_skew_win = min(skew_windows, key=lambda x: abs(x - win))
            skew_col = f'VolSkew_{closest_skew_win}_future'
            if skew_col in stock_cols:
                col_idx = stock_cols.index(skew_col)
                skew_val = f"{stock_row[col_idx]:.4f}"
        
        # Try to get curvature parameter
        curve_val = "N/A"
        curve_windows = safely_get_windows(stock_data, 'VolCurvature_')
        if curve_windows:
            closest_curve_win = min(curve_windows, key=lambda x: abs(x - win))
            curve_col = f'VolCurvature_{closest_curve_win}_future'
            if curve_col in stock_cols:
                col_idx = stock_cols.index(curve_col)
                curve_val = f"{stock_row[col_idx]:.4f}"
        
        skew_curve_text.append(f"{win}d: Skew={skew_val}, Curve={curve_val}")
    
    # Split into multiple lines
    if len(selected_windows) > 3:
        # Display all parameters but on separate lines
        params_text += "  " + "\n  ".join(skew_curve_text)
    else:
        # If there are only a few windows, can display them all on one line
        params_text += "  " + ", ".join(skew_curve_text)
    
    return params_text

def print_volatility_parameters(data, stock_symbol, reference_date, selected_windows=None):
    """
    Print volatility parameters for a stock on a specific date using Polars.
    
    Parameters:
    ----------
    data : pl.DataFrame
        DataFrame with volatility metrics
    stock_symbol : str
        Stock ticker symbol
    reference_date : date or str
        Reference date
    selected_windows : list, optional
        List of time windows to display
        
    Returns:
    -------
    None (prints to console)
    """
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    stock_data = data.filter(
        (pl.col('act_symbol') == stock_symbol) & 
        (pl.col('date') == reference_date)
    )
    
    if stock_data.height == 0:
        print(f"No data found for {stock_symbol} on {reference_date}")
        return
    
    stock_row = stock_data.row(0)
    stock_cols = stock_data.columns
    
    if selected_windows is None:
        available_windows = detect_available_windows(stock_data)
        if not available_windows:
            print(f"No volatility windows found for {stock_symbol}")
            return
        selected_windows = available_windows
    
    # Format the date properly for the display
    if isinstance(reference_date, str):
        formatted_date = reference_date
    elif hasattr(reference_date, 'strftime'):
        formatted_date = reference_date.strftime('%Y-%m-%d')
    else:
        formatted_date = str(reference_date).split()[0]  # Extract just the date part
    
    print(f"\n===== Volatility Parameters for {stock_symbol} on {formatted_date} =====\n")
    
    print("Volatility Term Structure:")
    for win in selected_windows:
        vol_col = f'YZVol_{win}_future'
        if vol_col in stock_cols:
            col_idx = stock_cols.index(vol_col)
            print(f"  {win}-day: {stock_row[col_idx]:.4f}")
        else:
            vol_windows = safely_get_windows(stock_data, 'YZVol_')
            if vol_windows:
                closest_window = min(vol_windows, key=lambda x: abs(x - win))
                closest_col = f'YZVol_{closest_window}_future'
                col_idx = stock_cols.index(closest_col)
                print(f"  {win}-day: {stock_row[col_idx]:.4f} (approx.)")
    
    print("\nHeston Model Parameters:")
    
    mr_windows = safely_get_windows(stock_data, 'MeanReversion_')
    if mr_windows:
        medium_window = selected_windows[len(selected_windows)//2]
        closest_win = min(mr_windows, key=lambda x: abs(x - medium_window))
        mr_col = f'MeanReversion_{closest_win}_future'
        if mr_col in stock_cols:
            col_idx = stock_cols.index(mr_col)
            print(f"  Mean Reversion Speed (κ): {stock_row[col_idx]:.4f}")
        else:
            print(f"  Mean Reversion Speed (κ): 2.0 (default)")
    else:
        print(f"  Mean Reversion Speed (κ): 2.0 (default)")
    
    vv_windows = safely_get_windows(stock_data, 'VolOfVol_')
    if vv_windows:
        medium_window = selected_windows[len(selected_windows)//2]
        closest_win = min(vv_windows, key=lambda x: abs(x - medium_window))
        vv_col = f'VolOfVol_{closest_win}_future'
        if vv_col in stock_cols:
            col_idx = stock_cols.index(vv_col)
            print(f"  Volatility of Volatility (σᵥ): {stock_row[col_idx]:.4f}")
        else:
            print(f"  Volatility of Volatility (σᵥ): 0.3 (default)")
    else:
        print(f"  Volatility of Volatility (σᵥ): 0.3 (default)")
    
    pv_windows = safely_get_windows(stock_data, 'PriceVolCorr_')
    if pv_windows:
        medium_window = selected_windows[len(selected_windows)//2]
        closest_win = min(pv_windows, key=lambda x: abs(x - medium_window))
        pv_col = f'PriceVolCorr_{closest_win}_future'
        if pv_col in stock_cols:
            col_idx = stock_cols.index(pv_col)
            print(f"  Price-Volatility Correlation (ρ): {stock_row[col_idx]:.4f}")
        else:
            print(f"  Price-Volatility Correlation (ρ): -0.7 (default)")
    else:
        print(f"  Price-Volatility Correlation (ρ): -0.7 (default)")
    
    print("\nVolatility Intensity (Continuous, 0-1 scale):")
    for win in selected_windows:
        intensity = get_tenor_specific_intensity(stock_data, win)
        
        # Map intensity to regime name
        if intensity < 0.33:
            regime_text = "Normal"
        elif intensity < 0.66:
            regime_text = "Elevated"
        else:
            regime_text = "High"
            
        print(f"  {win}-day: {regime_text} (Intensity: {intensity:.2f})")
    
    print("\nSkew & Curvature Parameters:")
    for win in selected_windows:
        skew_windows = safely_get_windows(stock_data, 'VolSkew_')
        if skew_windows:
            closest = min(skew_windows, key=lambda x: abs(x - win))
            skew_col = f'VolSkew_{closest}_future'
            if skew_col in stock_cols:
                skew = stock_data.select(skew_col).row(0)[0]
                print(f"  {win}d: Skew={skew:.4f}", end="")
            else:
                print(f"  {win}d: Skew=N/A", end="")
        else:
            print(f"  {win}d: Skew=N/A", end="")
        
        curve_windows = safely_get_windows(stock_data, 'VolCurvature_')
        if curve_windows:
            closest = min(curve_windows, key=lambda x: abs(x - win))
            curve_col = f'VolCurvature_{closest}_future'
            if curve_col in stock_cols:
                curve = stock_data.select(curve_col).row(0)[0]
                print(f", Curve={curve:.4f}")
            else:
                print(f", Curve=N/A")
        else:
            print(f", Curve=N/A")
    
    print("\n==================================================\n")

def compare_predicted_vs_actual(
    prediction_df: pl.DataFrame,
    actual_df: pl.DataFrame,
    stock_symbol: str,
    reference_date: Union[str, datetime.date],
    windows: List[int] = None,
    target_pattern: str = "YZVol_{window}_future"
) -> plt.Figure:
    """
    Compare predicted volatility surface with actual data.
    
    Parameters:
    -----------
    prediction_df : pl.DataFrame
        DataFrame with model predictions
    actual_df : pl.DataFrame
        DataFrame with actual values
    stock_symbol : str
        Stock symbol to visualize
    reference_date : Union[str, datetime.date]
        Reference date for comparison
    windows : List[int], optional
        Specific windows to include in the comparison
    target_pattern : str
        Pattern for target columns with {window} placeholder
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure with the comparison
    """
    # Normalize date formats
    if isinstance(reference_date, str):
        try:
            year, month, day = map(int, reference_date.split('-'))
            reference_date = pl.date(year, month, day)
        except ValueError:
            reference_date = pl.Date(datetime.datetime.strptime(reference_date, "%Y-%m-%d"))
    
    # Filter data for this specific stock and date
    pred_row = prediction_df.filter(
        (pl.col("act_symbol") == stock_symbol) &
        (pl.col("date") == reference_date)
    )
    
    actual_row = actual_df.filter(
        (pl.col("act_symbol") == stock_symbol) &
        (pl.col("date") == reference_date)
    )
    
    if pred_row.height == 0 or actual_row.height == 0:
        raise ValueError(f"Missing data for {stock_symbol} on {reference_date}")
    
    # Determine available windows from the pattern
    if windows is None:
        # Try to detect windows from actual data
        actual_windows = []
        for col in actual_row.columns:
            if col.startswith("YZVol_") and col.endswith("_future"):
                try:
                    window = int(col.replace("YZVol_", "").replace("_future", ""))
                    actual_windows.append(window)
                except ValueError:
                    continue
        
        if not actual_windows:
            raise ValueError(f"No volatility windows found in actual data for {stock_symbol}")
        
        windows = sorted(actual_windows)
    
    # Extract predicted and actual values
    predicted_values = []
    actual_values = []
    available_windows = []
    
    for window in windows:
        target_col = target_pattern.format(window=window)
        pred_col = f"{target_col}_pred"
        
        if pred_col in pred_row.columns and target_col in actual_row.columns:
            pred_val = pred_row[pred_col][0]
            actual_val = actual_row[target_col][0]
            
            # Skip if either value is null
            if pred_val is None or actual_val is None or np.isnan(pred_val) or np.isnan(actual_val):
                continue
                
            predicted_values.append(pred_val)
            actual_values.append(actual_val)
            available_windows.append(window)
    
    if not available_windows:
        raise ValueError(f"No matching windows found for comparison")
    
    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Convert to numpy for easier manipulation
    windows_array = np.array(available_windows)
    predicted_array = np.array(predicted_values)
    actual_array = np.array(actual_values)
    
    # Term structure plot
    ax1.plot(windows_array, predicted_array, 'o-', color='blue', label='Predicted')
    ax1.plot(windows_array, actual_array, 'o-', color='red', label='Actual')
    ax1.fill_between(windows_array, predicted_array, actual_array, color='gray', alpha=0.2)
    
    ax1.set_xlabel('Days to Expiration')
    ax1.set_ylabel('Volatility')
    ax1.set_title(f'Volatility Term Structure Comparison\n{stock_symbol} on {reference_date}')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Scatter plot with diagonal line
    ax2.scatter(actual_array, predicted_array, color='blue', alpha=0.8)
    
    # Add labels for each point
    for i, window in enumerate(available_windows):
        ax2.annotate(f"{window}d", 
                    (actual_array[i], predicted_array[i]),
                    textcoords="offset points",
                    xytext=(0, 5),
                    ha='center')
    
    # Calculate error metrics
    rmse = np.sqrt(np.mean((predicted_array - actual_array) ** 2))
    mae = np.mean(np.abs(predicted_array - actual_array))
    mape = np.mean(np.abs((actual_array - predicted_array) / actual_array)) * 100
    
    # Plot the diagonal line (perfect predictions)
    min_val = min(min(actual_array), min(predicted_array)) * 0.9
    max_val = max(max(actual_array), max(predicted_array)) * 1.1
    ax2.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    ax2.set_xlabel('Actual Volatility')
    ax2.set_ylabel('Predicted Volatility')
    ax2.set_title(f'Prediction Accuracy\nRMSE: {rmse:.4f}, MAE: {mae:.4f}, MAPE: {mape:.2f}%')
    ax2.grid(True, alpha=0.3)
    
    # Set equal aspect ratio for the scatter plot
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    return fig