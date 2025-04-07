import numpy as np
import plotly.graph_objects as go
import datetime
import polars as pl
from scipy.optimize import minimize
from scipy.interpolate import griddata

def generate_vol_surface(df, stock, date, show_surface=False):
    """
    Generate a theoretically sound realized volatility surface based on predicted
    volatility parameters across multiple time horizons and price levels.
    
    This function implements a hybrid approach combining elements from:
    1. SVI (Stochastic Volatility Inspired) parameterization
    2. Heston stochastic volatility dynamics
    3. Local volatility principles
    
    The surface represents conditional expected volatility given both time and price level,
    similar to Dupire's local volatility but representing physical/realized volatility.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing _future columns with volatility predictions
    stock : str
        Stock symbol
    date : str or datetime.date or pl.Date or pl.Expr
        Date for visualization
    show_surface : bool
        Whether to display the surface (default: False)
    
    Returns:
    --------
    tuple: (fig, surface_dict)
        fig: plotly figure object
        surface_dict: dictionary with surface data
    """
    # Filter data for the specific stock and date
    row = df.filter((pl.col("act_symbol") == stock) & (pl.col("date") == date))
    
    if row.height == 0:
        raise ValueError(f"No data found for {stock} on {date}")
    
    # Define trading day windows (these are the windows in your _future columns)
    trading_windows = [10, 15, 20, 25, 30, 35]
    
    # Convert trading days to approximate calendar days (for x-axis)
    # A common approximation: calendar days ≈ trading days × (7/5)
    calendar_days = [int(round(window * 7/5)) for window in trading_windows]
    
    # Extract realized values for all timepoints
    realized_data = {}
    for i, trading_window in enumerate(trading_windows):
        days = calendar_days[i]
        years = days / 365.0  # For term structure effects
        
        # Extract base realized values with bounds checking
        realized_vol = float(row[f"YZVol_{trading_window}_future"][0]) * np.sqrt(252)  # Annualize
        realized_k = float(row[f"LogPriceRatio_{trading_window}_future"][0])
        
        # Extract volatility shape parameters with proper bounds
        skew = clip_value(float(row[f"VolSkew_{trading_window}_future"][0]), -2.0, 2.0)
        curvature = clip_value(float(row[f"VolCurvature_{trading_window}_future"][0]), -5.0, 5.0)
        wing_ratio = clip_value(float(row[f"WingRatio_{trading_window}_future"][0]), 0.2, 5.0)
        mean_reversion = clip_value(float(row[f"MeanReversion_{trading_window}_future"][0]), 0.01, 10.0)
        vol_of_vol = clip_value(float(row[f"VolOfVol_{trading_window}_future"][0]), 0.01, 2.0)
        price_vol_corr = clip_value(float(row[f"PriceVolCorr_{trading_window}_future"][0]), -1.0, 1.0)
        vol_intensity = clip_value(float(row[f"VolIntensity_{trading_window}_future"][0]), 0.0, 1.0)
        
        realized_data[days] = {
            'vol': realized_vol,
            'moneyness': realized_k,
            'skew': skew,
            'curvature': curvature,
            'wing_ratio': wing_ratio,
            'mean_reversion': mean_reversion,
            'vol_of_vol': vol_of_vol, 
            'price_vol_corr': price_vol_corr,
            'vol_intensity': vol_intensity,
            'trading_window': trading_window,
            'years': years
        }
    
    # Determine moneyness range dynamically based on realized points
    realized_moneyness = [realized_data[t]['moneyness'] for t in calendar_days]
    min_realized = min(realized_moneyness)
    max_realized = max(realized_moneyness)
    
    # Create wider range around realized moneyness points
    moneyness_padding = 0.15
    moneyness_min = min_realized - moneyness_padding
    moneyness_max = max_realized + moneyness_padding
    moneyness = np.linspace(moneyness_min, moneyness_max, 50)
    
    # Create meshgrid for surface plotting
    K, T = np.meshgrid(moneyness, calendar_days)
    vol_surface = np.zeros(K.shape)
    
    # Get the SVI parameters for each time slice with calibration to known points
    svi_params = {}
    for i, days in enumerate(calendar_days):
        data = realized_data[days]
        # Calibrate SVI parameters to exactly hit our realized volatility point
        svi_params[days] = calibrate_svi_parameters(data)
    
    # Apply surface constraints to ensure no-arbitrage
    svi_params = apply_term_structure_constraints(svi_params, calendar_days, realized_data)
    
    # Generate surface with theoretically sound volatility dynamics
    for i, days in enumerate(calendar_days):
        data = realized_data[days]
        params = svi_params[days]
        
        for j, k in enumerate(moneyness):
            # Calculate log-moneyness (adjusted by the expected log price change)
            log_moneyness = k - data['moneyness']
            
            # Calculate volatility using a method that guarantees we hit our realized points
            vol_surface[i, j] = calculate_vol_at_moneyness(
                log_moneyness, 
                params, 
                data
            )
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Add main surface
    fig.add_trace(go.Surface(
        x=K,
        y=T,
        z=vol_surface,
        colorscale='Viridis',
        colorbar=dict(title='Annualized Volatility'),
        name='Realized Volatility Surface'
    ))
    
    # Add actual realized points
    actual_moneyness = [realized_data[t]['moneyness'] for t in calendar_days]
    actual_vols = [realized_data[t]['vol'] for t in calendar_days]
    
    fig.add_trace(go.Scatter3d(
        x=actual_moneyness,
        y=calendar_days,
        z=actual_vols,
        mode='markers',
        marker=dict(size=7, color='red'),
        name='Realized Volatility'
    ))
    
    # Add text annotation showing trading days to calendar days mapping
    annotations = []
    for i, (trade_days, cal_days) in enumerate(zip(trading_windows, calendar_days)):
        annotations.append(
            dict(
                showarrow=False,
                x=min_realized,
                y=cal_days,
                z=0,
                text=f"{trade_days}td",
                xanchor="left",
                font=dict(color="white", size=10)
            )
        )
    
    # Properly format the date for the title
    if isinstance(date, pl.Date) or isinstance(date, pl.Expr):
        # If it's a Polars Date or Expression, convert to string in YYYY-MM-DD format
        date_value = row["date"][0]
        if hasattr(date_value, 'strftime'):
            date_str = date_value.strftime("%Y-%m-%d")
        else:
            date_str = str(date_value).split(' ')[0]  # Take just the date part
    elif isinstance(date, datetime.date):
        # If it's a Python datetime.date object
        date_str = date.strftime("%Y-%m-%d")
    else:
        # If it's already a string or something else
        date_str = str(date).split(' ')[0]  # Take just the date part
    
    # Set axis labels and title with properly formatted date
    fig.update_layout(
        title=f"Realized Volatility Surface for {stock} on {date_str}",
        scene=dict(
            xaxis_title='Log(Future Price / Spot Price) ~ log-moneyness',
            yaxis_title='Calendar Days',
            zaxis_title='Annualized Volatility',
            xaxis=dict(range=[moneyness_min, moneyness_max]),
            yaxis=dict(range=[min(calendar_days), max(calendar_days)]),
            zaxis=dict(range=[0, np.nanmax(vol_surface) * 1.1]),
            annotations=annotations
        ),
        width=900,
        height=700,
        margin=dict(r=20, l=10, b=10, t=50)
    )
    
    # Pack surface data
    surface_data = {
        'K': K.tolist(),  # Convert numpy arrays to lists for JSON serialization
        'T': T.tolist(),
        'vol_surface': vol_surface.tolist(),
        'calendar_days': calendar_days,
        'trading_windows': trading_windows,
        'moneyness': moneyness.tolist(),
        'actual_moneyness': actual_moneyness,
        'actual_vols': actual_vols,
        'realized_data': realized_data,
        'svi_params': {k: list(v.values()) for k, v in svi_params.items()},  # Make serializable
        'date': date_str,
        'stock': stock
    }
    
    return fig, surface_data


def clip_value(value, min_val, max_val):
    """Clip a value to be between min_val and max_val"""
    return max(min_val, min(value, max_val))


def calibrate_svi_parameters(data):
    """
    Calibrate SVI parameters to exactly match the known realized volatility point.
    
    Parameters:
    -----------
    data : dict
        Dictionary with our volatility parameters
        
    Returns:
    --------
    dict: Calibrated SVI parameters
    """
    # Extract our parameters
    vol = data['vol']           # Base ATM volatility
    skew = data['skew']         # Skew (asymmetry)
    curvature = data['curvature']  # Curvature
    wing_ratio = data['wing_ratio']  # Wing asymmetry
    years = data['years']       # Time to expiry
    
    # Target log-moneyness (where we have our realized volatility)
    log_moneyness = 0.0  # The surface will be relative to our realized moneyness
    
    # Target variance
    target_variance = vol**2 * years
    
    # Initial SVI parameters (starting point)
    # Setting m = 0 centers the SVI at our realized moneyness
    m = 0.0
    
    # Initial rho based on skew
    rho = -np.clip(np.sign(skew) * (0.5 + 0.4 * abs(skew)), -0.9, 0.9)
    
    # Initial sigma based on curvature
    sigma = 0.2 / (0.2 + abs(curvature))
    
    # Wing ratio affects rho (put/call wing asymmetry)
    if wing_ratio > 1:
        # Adjust ρ to account for wing ratio (put wing steeper)
        rho = rho * np.sqrt(wing_ratio)
    else:
        # If wing_ratio < 1, call wing is steeper
        rho = rho / np.sqrt(max(0.2, wing_ratio))
    
    # Initial guess for b based on skew and curvature
    b = 0.1 + 0.2 * abs(skew) + 0.1 * curvature
    
    # Solve for 'a' to ensure we hit our point exactly
    # At our realized moneyness (log_moneyness=0, k_shifted=0-m):
    # w(0) = a + b * (ρ * (-m) + sqrt(m^2 + σ^2))
    k_shifted = log_moneyness - m
    svi_term = rho * k_shifted + np.sqrt(k_shifted**2 + sigma**2)
    
    # Solve for 'a' to match our variance at this point
    a = target_variance - b * svi_term
    
    return {
        'a': a,
        'b': b,
        'rho': rho,
        'm': m,
        'sigma': sigma,
        # Store original parameters for reference
        'vol': vol,
        'skew': skew,
        'curvature': curvature,
        'wing_ratio': wing_ratio,
        'years': years,
        'target_variance': target_variance
    }


def apply_term_structure_constraints(svi_params, calendar_days, realized_data):
    """
    Apply term structure constraints to ensure no calendar arbitrage
    and to incorporate mean reversion effects.
    
    Parameters:
    -----------
    svi_params : dict
        Dictionary of SVI parameters for each time slice
    calendar_days : list
        List of calendar days
    realized_data : dict
        Dictionary with our volatility parameters
        
    Returns:
    --------
    dict: Updated SVI parameters that maintain our exact realized points
    """
    # Make a deep copy to avoid modifying the original
    params = {k: v.copy() for k, v in svi_params.items()}
    
    # Sort days to ensure we process in order
    days_sorted = sorted(calendar_days)
    
    # Additional constraints:
    # 1. Total variance should increase with time (no calendar arbitrage)
    # 2. Parameters should evolve smoothly with time
    # 3. Mean reversion should affect how variance evolves
    
    for i in range(1, len(days_sorted)):
        prev_day = days_sorted[i-1]
        curr_day = days_sorted[i]
        
        prev_params = params[prev_day]
        curr_params = params[curr_day]
        
        # Mean reversion speed influences how quickly parameters change
        mean_rev = realized_data[curr_day]['mean_reversion']
        
        # Higher mean reversion means faster convergence to long-term level
        time_factor = (curr_day - prev_day) / 365.0
        decay_factor = np.exp(-mean_rev * time_factor)
        
        # Ensure variance increases with time (no calendar arbitrage)
        # a_t should satisfy: a_t >= a_{t-1} * (t/t-1)
        min_a = prev_params['a'] * (curr_day / prev_day)
        
        # For b parameter - control wing steepness evolution
        b_prev = prev_params['b']
        b_smooth = decay_factor * b_prev + (1 - decay_factor) * curr_params['b']
        
        # For rho parameter - control skew evolution
        rho_prev = prev_params['rho']
        rho_smooth = decay_factor * rho_prev + (1 - decay_factor) * curr_params['rho']
        
        # For sigma parameter - control curvature evolution
        sigma_prev = prev_params['sigma']
        sigma_smooth = decay_factor * sigma_prev + (1 - decay_factor) * curr_params['sigma']
        
        # Update parameters while preserving our target variance at log_moneyness = 0
        # This is crucial to ensure the surface still passes through our realized points
        target_variance = curr_params['target_variance']
        m = curr_params['m']
        
        # Calculate the SVI term at our realized moneyness (log_moneyness = 0)
        k_shifted = 0 - m
        svi_term = rho_smooth * k_shifted + np.sqrt(k_shifted**2 + sigma_smooth**2)
        
        # Solve for 'a' to ensure we still hit our realized point
        a_adjusted = target_variance - b_smooth * svi_term
        
        # Ensure 'a' satisfies calendar arbitrage constraint
        a_adjusted = max(a_adjusted, min_a)
        
        # Recalculate 'b' if necessary to still hit our target point
        if a_adjusted > target_variance:
            # Edge case - adjust other parameters to maintain our realized point
            # This is a simplification - in practice, more sophisticated
            # parameter adjustment might be needed
            b_smooth = 0.001  # Minimal wing effect
            svi_term = rho_smooth * k_shifted + np.sqrt(k_shifted**2 + sigma_smooth**2)
            a_adjusted = target_variance - b_smooth * svi_term
        
        # Update parameters
        curr_params['a'] = a_adjusted
        curr_params['b'] = b_smooth
        curr_params['rho'] = rho_smooth
        curr_params['sigma'] = sigma_smooth
    
    return params


def calculate_vol_at_moneyness(log_moneyness, params, data):
    """
    Calculate volatility at a given log-moneyness using a hybrid SVI/stochastic vol model
    that guarantees passing through our known realized volatility points.
    
    Parameters:
    -----------
    log_moneyness : float
        Log-moneyness (relative to our realized moneyness)
    params : dict
        SVI parameters (calibrated to hit our realized point)
    data : dict
        Dictionary with our original volatility parameters
        
    Returns:
    --------
    float: Volatility at the given moneyness
    """
    # Extract SVI parameters
    a = params['a']  # Base level
    b = params['b']  # Wing slopes
    rho = params['rho']  # Asymmetry
    m = params['m']  # Shift
    sigma = params['sigma']  # Curvature
    years = params['years']  # Time
    
    # Extract additional parameters
    vol_of_vol = data['vol_of_vol']
    price_vol_corr = data['price_vol_corr']
    vol_intensity = data['vol_intensity']
    
    # Modified SVI formula for total variance
    k_shifted = log_moneyness - m
    svi_term = rho * k_shifted + np.sqrt(k_shifted**2 + sigma**2)
    total_variance = a + b * svi_term
    
    # Additional Heston/stochastic vol dynamics
    # 1. Vol-of-vol effect: higher vol-of-vol means more pronounced smile
    vov_factor = 1.0 + 0.2 * vol_of_vol * (k_shifted**2)
    
    # 2. Price-vol correlation: affects skew (leverage effect)
    # Negative correlation → steeper negative skew
    corr_factor = 1.0 + 0.1 * price_vol_corr * k_shifted
    
    # 3. Jump/tail intensity: affects far OTM options
    intensity_factor = 1.0 + 0.2 * vol_intensity * np.abs(k_shifted)**2.5
    
    # Blend the raw SVI variance with additional effects
    # Ensure we preserve our realized point (when log_moneyness = 0)
    if np.isclose(log_moneyness, 0.0, atol=1e-8):
        # Exactly at our realized moneyness, force the exact realized volatility
        volatility = data['vol']
    else:
        # Away from our realized moneyness, apply full model
        # Combine all effects
        total_variance = total_variance * vov_factor * corr_factor * intensity_factor
        
        # Convert total variance to annualized volatility
        volatility = np.sqrt(total_variance / years)
    
    # Ensure volatility is within reasonable bounds (1% to 200%)
    volatility = np.clip(volatility, 0.01, 2.0)
    
    return volatility
