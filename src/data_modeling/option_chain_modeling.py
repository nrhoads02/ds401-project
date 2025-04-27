# option_chain_modeling.py
# ---------------------------------------------------------------
# Options chain surface visualization and analysis
# Generates volatility surface from options chain data and enables
# comparison with LGBM model surface to identify pricing inefficiencies
# ---------------------------------------------------------------

import os
import datetime as _dt
from typing import Dict, Any, List, Tuple, Union, Optional

import numpy as np
import polars as pl
import plotly.graph_objects as go
import logging

from src.data_transformation.stock_adjustments import adjust_option_splits

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def find_nearest_date(df: pl.DataFrame, target_date: Union[str, _dt.date, _dt.datetime, np.datetime64], 
                     symbol: str = None, search_backwards: bool = True) -> Optional[_dt.date]:
    """
    Find the nearest available date to target_date in the DataFrame.
    
    Args:
        df: DataFrame containing a 'date' column
        target_date: The target date to find the nearest match for
        symbol: Optional stock symbol to filter by
        search_backwards: If True, only consider dates earlier than or equal to target_date
        
    Returns:
        The nearest date found, or None if no valid date is available
    """
    # Convert target_date to datetime.date for consistent comparison
    if isinstance(target_date, str):
        td = _dt.date.fromisoformat(target_date)
    elif isinstance(target_date, _dt.datetime):
        td = target_date.date()
    elif isinstance(target_date, np.datetime64):
        td = target_date.astype('datetime64[D]').item().date()
    elif isinstance(target_date, _dt.date):
        td = target_date
    else:
        raise TypeError(f"Cannot convert {type(target_date)} to date")
    
    # Filter by symbol if provided
    if symbol:
        df_filtered = df.filter(pl.col("act_symbol") == symbol)
    else:
        df_filtered = df
    
    if df_filtered.is_empty():
        return None
    
    # Get unique dates and convert to Python dates for easy comparison
    unique_dates = df_filtered.select("date").unique().sort("date")
    dates_list = []
    for d in unique_dates["date"].to_list():
        try:
            if isinstance(d, str):
                dates_list.append(_dt.date.fromisoformat(d))
            elif isinstance(d, _dt.datetime):
                dates_list.append(d.date())
            elif isinstance(d, np.datetime64):
                dates_list.append(d.astype('datetime64[D]').item().date())
            elif isinstance(d, _dt.date):
                dates_list.append(d)
            else:
                logger.warning(f"Skipping date {d} of type {type(d)}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Could not convert date {d}: {e}")
    
    if not dates_list:
        return None
    
    # Find the nearest date according to direction
    if search_backwards:
        # Find the nearest date that is <= target_date
        valid_dates = [d for d in dates_list if d <= td]
        if not valid_dates:
            return None
        return max(valid_dates)  # Latest date that is <= target_date
    else:
        # Find absolute nearest date (either before or after)
        return min(dates_list, key=lambda d: abs((d - td).days))

def generate_options_surface(
    option_chain_df: pl.DataFrame,
    stock: str,
    trade_date: Union[str, _dt.date, _dt.datetime, np.datetime64],
    option_type: str = "call",  # 'call', 'put', or 'average'
    k_range: Tuple[float, float] = (-1.0, 1.0),  # Default log-moneyness range
    k_grid_points: int = 101,    # Default grid density
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generates a volatility surface from options chain data for a given stock and date.
    
    Args:
        option_chain_df: DataFrame with options chain data
        stock: Stock symbol
        trade_date: The date for which to generate the surface
        option_type: Type of options to use ('call', 'put', or 'average')
        k_range: Range for log-moneyness 'k'
        k_grid_points: Number of points to use for the strike (k) dimension
        
    Returns:
        A tuple (K_mesh, T_mesh, Z) representing the surface arrays (Strike, Time, Vol),
        or None if surface generation fails
    """
    logger.info(f"Generating options chain surface for {stock} on {trade_date}, type={option_type}") #

    logger.info("Adjusting option chain data for stock splits...")
    option_chain_raw = option_chain_df
    option_chain_df = adjust_option_splits(option_chain_raw, verbose=False)
    logger.info("Option chain split adjustment complete.")

    # Find nearest date in the (now potentially adjusted) options chain data
    nearest_date = find_nearest_date(option_chain_df, trade_date, stock) #
    if nearest_date is None:
        logger.error(f"No options data found for {stock} on or before {trade_date} (after potential adjustment)") # Modified log message
        return None

    logger.info(f"Using nearest available options data from: {nearest_date}") #

    # Filter by date and stock using the adjusted data
    filtered_df = option_chain_df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date").cast(pl.Date) == nearest_date)
    ) #
    
    if filtered_df.is_empty():
        logger.error(f"No options data found for {stock} on {nearest_date} after filtering")
        return None
    
    # Get the current stock price (using representative strike price)
    # In a real implementation, you might want to get this from the OHLCV data
    # but for simplicity we'll extract from filtered_df
    try:
        # Calculate average implied volatility for ATM options to get reasonable S0 estimate
        atm_options = filtered_df.filter(
            (pl.col("call_put") == "Call") & 
            (pl.col("delta") >= 0.45) & 
            (pl.col("delta") <= 0.55)
        )
        
        if atm_options.is_empty():
            # Fallback - get the middle strike
            unique_strikes = filtered_df["strike"].unique().sort()
            S0 = unique_strikes[len(unique_strikes) // 2]
        else:
            # Use weighted average of ATM option strikes
            S0 = atm_options.select(
                ((pl.col("strike") * pl.col("vol")).sum() / 
                 pl.col("vol").sum()).alias("S0")
            ).item()
        
        logger.info(f"Using spot price of {S0} for {stock} on {nearest_date}")
    except Exception as e:
        logger.error(f"Error calculating spot price: {e}")
        # Fallback - use median strike as S0
        try:
            S0 = filtered_df.select(pl.col("strike").median()).item()
            logger.warning(f"Using median strike {S0} as fallback spot price")
        except:
            logger.error("Could not determine spot price, aborting")
            return None
    
    # Filter by option type
    if option_type.lower() == "call":
        type_filtered_df = filtered_df.filter(pl.col("call_put") == "Call")
    elif option_type.lower() == "put":
        type_filtered_df = filtered_df.filter(pl.col("call_put") == "Put")
    else:  # 'average' - we'll process both types separately and average them
        calls_df = filtered_df.filter(pl.col("call_put") == "Call")
        puts_df = filtered_df.filter(pl.col("call_put") == "Put")
        
        # Check if we have both calls and puts
        if calls_df.is_empty() or puts_df.is_empty():
            logger.warning(f"Missing {'calls' if calls_df.is_empty() else 'puts'} for {stock} on {nearest_date}")
            # Fall back to the one we have
            type_filtered_df = puts_df if calls_df.is_empty() else calls_df
            logger.warning(f"Using only {'puts' if calls_df.is_empty() else 'calls'} instead of average")
        else:
            # Process calls and puts separately and combine later
            call_surface = generate_options_surface(
                option_chain_raw, stock, nearest_date, "call", k_range, k_grid_points
            )
            put_surface = generate_options_surface(
                option_chain_raw, stock, nearest_date, "put", k_range, k_grid_points
            )
            
            if call_surface and put_surface:
                # Extract data
                call_K, call_T, call_Z = call_surface
                put_K, put_T, put_Z = put_surface
                
                # Average implied volatilities
                avg_Z = (call_Z + put_Z) / 2.0
                
                # Use the K and T from calls (they should be the same as puts)
                logger.info(f"Created average surface from call and put surfaces")
                return call_K, call_T, avg_Z
            else:
                if call_surface:
                    logger.warning("Could not generate put surface, using only calls")
                    return call_surface
                elif put_surface:
                    logger.warning("Could not generate call surface, using only puts")
                    return put_surface
                else:
                    logger.error("Could not generate either call or put surface")
                    return None

    # Check if we have data after filtering by type
    if 'type_filtered_df' in locals() and type_filtered_df.is_empty():
        logger.error(f"No {option_type} options found for {stock} on {nearest_date}")
        return None
    else:
        # Get unique expirations
        unique_expirations = type_filtered_df.select(pl.col("expiration").unique().sort())
        
        if unique_expirations.is_empty():
            logger.error(f"No valid expirations found for {stock} {option_type} options on {nearest_date}")
            return None
        
        expirations_list = []
        for exp in unique_expirations["expiration"].to_list():
            try:
                if isinstance(exp, str):
                    expirations_list.append(_dt.date.fromisoformat(exp))
                elif isinstance(exp, _dt.datetime):
                    expirations_list.append(exp.date())
                elif isinstance(exp, np.datetime64):
                    expirations_list.append(exp.astype('datetime64[D]').item().date())
                elif isinstance(exp, _dt.date):
                    expirations_list.append(exp)
                else:
                    logger.warning(f"Skipping expiration {exp} of type {type(exp)}")
            except (ValueError, TypeError) as e:
                logger.warning(f"Could not convert expiration {exp}: {e}")
        
        if not expirations_list:
            logger.error(f"No valid expiration dates could be parsed")
            return None
        
        # Calculate calendar days to expiration
        trade_dt = nearest_date
        t_cal_list = [(exp_date - trade_dt).days for exp_date in expirations_list]
        
        # Create empty surface grid
        n_expirations = len(t_cal_list)
        n_strikes = k_grid_points
        
        # Create log-moneyness grid and convert to strikes
        k_grid = np.linspace(k_range[0], k_range[1], n_strikes)
        K_grid = S0 * np.exp(k_grid)
        
        # Initialize surface arrays
        Z = np.full((n_expirations, n_strikes), np.nan)
        K_mesh = np.zeros((n_expirations, n_strikes))
        T_mesh = np.zeros((n_expirations, n_strikes))
        
        # For each expiration, process volatility data
        for t_idx, (expiration, t_cal) in enumerate(zip(expirations_list, t_cal_list)):
            # Filter by expiration
            exp_df = type_filtered_df.filter(
                pl.col("expiration").cast(pl.Date) == expiration
            )
            
            # Skip if no data for this expiration
            if exp_df.is_empty():
                logger.warning(f"No data for expiration {expiration}, skipping")
                continue
                
            # Sort by strike
            exp_df = exp_df.sort("strike")
            strikes = exp_df["strike"].to_list()
            
            # Use implied volatility from 'vol' column
            vols = exp_df["vol"].to_list()
            
            # Basic data validation
            if len(strikes) != len(vols) or len(strikes) == 0:
                logger.warning(f"Invalid data for expiration {expiration}, skipping")
                continue
                
            # Check for NaN or invalid values
            valid_data = [(s, v) for s, v in zip(strikes, vols) 
                          if np.isfinite(s) and np.isfinite(v) and v > 0]
            
            if not valid_data:
                logger.warning(f"No valid vol data for expiration {expiration}, skipping")
                continue
                
            # Unpack valid data
            valid_strikes, valid_vols = zip(*valid_data)
            
            # Interpolate volatility for each point in the K grid
            for k_idx, K in enumerate(K_grid):
                # Find closest strikes for interpolation
                idx = np.searchsorted(valid_strikes, K)
                
                if idx == 0:  # K is smaller than the smallest strike
                    Z[t_idx, k_idx] = valid_vols[0]
                elif idx == len(valid_strikes):  # K is larger than the largest strike
                    Z[t_idx, k_idx] = valid_vols[-1]
                else:  # Linear interpolation between strikes
                    k_low, k_high = valid_strikes[idx-1], valid_strikes[idx]
                    v_low, v_high = valid_vols[idx-1], valid_vols[idx]
                    
                    # Prevent division by zero
                    if k_high == k_low:
                        Z[t_idx, k_idx] = v_low
                    else:
                        # Linear interpolation
                        alpha = (K - k_low) / (k_high - k_low)
                        Z[t_idx, k_idx] = v_low + alpha * (v_high - v_low)
                
                # Fill mesh values
                K_mesh[t_idx, k_idx] = K
                T_mesh[t_idx, k_idx] = t_cal
        
        # Check if we have any valid data
        if np.all(np.isnan(Z)):
            logger.error(f"No valid volatility data could be interpolated")
            return None
            
        # Final surface is ready
        return K_mesh, T_mesh, Z

def plot_options_surface(
    K_mesh: np.ndarray, 
    T_mesh: np.ndarray, 
    Z: np.ndarray,
    stock: str,
    trade_date: Union[str, _dt.date, _dt.datetime, np.datetime64],
    option_type: str = "call",
    show_surface: bool = False
) -> go.Figure:
    """
    Create and display a 3D surface plot of options volatility.
    
    Args:
        K_mesh: 2D array of strike prices
        T_mesh: 2D array of expiration times (calendar days)
        Z: 2D array of volatility values
        stock: Stock symbol
        trade_date: Trade date
        option_type: Type of options used ('call', 'put', or 'average')
        show_surface: If True, displays the plot
        
    Returns:
        Plotly Figure object
    """
    # Convert date for display
    if isinstance(trade_date, str):
        td = _dt.date.fromisoformat(trade_date)
    elif isinstance(trade_date, _dt.datetime):
        td = trade_date.date()
    elif isinstance(trade_date, np.datetime64):
        td = trade_date.astype('datetime64[D]').item().date()
    elif isinstance(trade_date, _dt.date):
        td = trade_date
    else:
        td = str(trade_date)
    
    # Create figure
    fig = go.Figure()
    
    # Define contour settings
    contour_settings = {
        "x": {"show": False}, 
        "y": {"show": False},
        "z": {
            "show": True,
            "highlight": True,
            "highlightcolor": "limegreen",
            "project": {"z": True}
        }
    }
    
    # Add the volatility surface
    fig.add_trace(go.Surface(
        x=K_mesh,
        y=T_mesh,
        z=Z,
        name=f"{option_type.capitalize()} Option IV",
        colorscale="Viridis",
        showlegend=True,
        colorbar=dict(title='Implied Vol (σ)', thickness=20, len=0.75),
        contours=contour_settings,
        hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                      "<b>T:</b> %{y:.0f} cd<br>" +
                      "<b>IV σ:</b> %{z:.4f}<extra></extra>",
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2)
    ))
    
    # Calculate dynamic axis ranges
    x_min = np.nanmin(K_mesh) * 0.95
    x_max = np.nanmax(K_mesh) * 1.05
    z_min = max(0, np.nanmin(Z) * 0.9)
    z_max = np.nanmax(Z) * 1.1
    
    # Configure plot layout
    fig.update_layout(
        title=f"{stock} {option_type.capitalize()} Options Implied Volatility Surface @ {td}",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Days to Expiration (T)",
            zaxis_title="Implied Volatility (σ)",
            xaxis_range=[x_min, x_max],
            zaxis_range=[z_min, z_max],
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Adjust camera view
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.7, y=-1.7, z=1.0)
    ))
    
    # Display the figure if requested
    if show_surface:
        fig.show()
    
    return fig

def compare_surfaces(
    option_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lgbm_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    stock: str,
    trade_date: Union[str, _dt.date, _dt.datetime, np.datetime64],
    option_type: str = "call",
    show_realized_points: bool = True,
    realized_points: Optional[Tuple[List[float], List[float], List[float]]] = None,
    spot_price: Optional[float] = None,
    spot_line_data: Optional[Tuple[List[float], List[float]]] = None,
    show_surface: bool = False
) -> go.Figure:
    """
    Create a visualization comparing options and LGBM volatility surfaces.
    
    Args:
        option_surface: Tuple (K_mesh, T_mesh, Z) for options surface
        lgbm_surface: Tuple (K_mesh, T_mesh, Z) for LGBM model surface
        stock: Stock symbol
        trade_date: Trade date
        option_type: Type of options used
        show_realized_points: Whether to show realized volatility points
        realized_points: Optional tuple of (strikes, times, sigmas) for realized points
        spot_price: Optional current stock price for S0 line
        spot_line_data: Optional tuple of (times, sigmas) for S0 line
        show_surface: If True, displays the plot
        
    Returns:
        Plotly Figure object with both surfaces
    """
    # Convert date for display
    if isinstance(trade_date, str):
        td = _dt.date.fromisoformat(trade_date)
    elif isinstance(trade_date, _dt.datetime):
        td = trade_date.date()
    elif isinstance(trade_date, np.datetime64):
        td = trade_date.astype('datetime64[D]').item().date()
    elif isinstance(trade_date, _dt.date):
        td = trade_date
    else:
        td = str(trade_date)
    
    opt_K_mesh, opt_T_mesh, opt_Z = option_surface
    lgbm_K_mesh, lgbm_T_mesh, lgbm_Z = lgbm_surface
    
    # Create figure
    fig = go.Figure()
    
    # Define contour settings for options surface
    opt_contour_settings = {
        "x": {"show": False}, 
        "y": {"show": False},
        "z": {
            "show": True,
            "highlight": True,
            "highlightcolor": "limegreen",
            "project": {"z": True}
        }
    }
    
    # Define contour settings for LGBM surface
    lgbm_contour_settings = {
        "x": {"show": False}, 
        "y": {"show": False},
        "z": {
            "show": True,
            "highlight": True,
            "highlightcolor": "red",
            "project": {"z": True}
        }
    }
    
    # Add the options volatility surface
    fig.add_trace(go.Surface(
        x=opt_K_mesh,
        y=opt_T_mesh,
        z=opt_Z,
        name=f"{option_type.capitalize()} Options IV",
        colorscale="Viridis",
        showlegend=True,
        colorbar=dict(title='IV (σ)', thickness=20, len=0.75),
        contours=opt_contour_settings,
        hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                      "<b>T:</b> %{y:.0f} cd<br>" +
                      "<b>IV σ:</b> %{z:.4f}<extra></extra>",
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2),
        opacity=0.5  # Make slightly transparent
    ))
    
    # Add the LGBM model surface
    fig.add_trace(go.Surface(
        x=lgbm_K_mesh,
        y=lgbm_T_mesh,
        z=lgbm_Z,
        name="LGBM Model RV",
        colorscale="Plasma",  # Different colorscale
        showlegend=True,
        showscale=False,  # Don't show a second colorbar
        contours=lgbm_contour_settings,
        hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                      "<b>T:</b> %{y:.0f} cd<br>" +
                      "<b>RV σ:</b> %{z:.4f}<extra></extra>",
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2),
        opacity=0.5  # More transparent
    ))
    
    # Add realized volatility points if available and requested
    if show_realized_points and realized_points is not None:
        realized_strikes, realized_T, realized_sigma = realized_points
        if realized_strikes and realized_T and realized_sigma:
            fig.add_trace(go.Scatter3d(
                x=realized_strikes,
                y=realized_T,
                z=realized_sigma,
                mode="markers",
                marker=dict(color="red", size=5, symbol='diamond'),
                name="Realized Points",
                showlegend=True,
                hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                            "<b>T:</b> %{y:.0f} cd<br>" +
                            "<b>Realized σ:</b> %{z:.4f}<extra></extra>"
            ))
            logger.info(f"Added {len(realized_strikes)} realized points to comparison plot")
    
    # Add S0 line (historical realized vol at current spot price) if available
    if spot_price is not None and spot_line_data is not None:
        s0_line_T, s0_line_sigma = spot_line_data
        if s0_line_T and s0_line_sigma:
            # Sort points by time for a smooth line
            s0_line_points = sorted(zip(s0_line_T, s0_line_sigma))
            s0_T_sorted = [p[0] for p in s0_line_points]
            s0_sigma_sorted = [p[1] for p in s0_line_points]
            
            fig.add_trace(go.Scatter3d(
                x=[spot_price] * len(s0_T_sorted),   # Constant K = S0
                y=s0_T_sorted,               # T values
                z=s0_sigma_sorted,           # Historical realized sigma at S0
                mode='lines+markers',
                marker=dict(color='cyan', size=3),
                line=dict(color='cyan', width=4),
                name=f'Hist. RV @ Spot K={spot_price:.2f}',
                showlegend=True,
                hovertemplate="<b>Strike K:</b> %{x:.2f} (Spot)<br>" +
                            "<b>T:</b> %{y:.0f} cd<br>" +
                            "<b>Hist. RV σ:</b> %{z:.4f}<extra></extra>"
            ))
            logger.info(f"Added S0 line with {len(s0_T_sorted)} historical RV points to comparison plot")
    
    # Calculate range for axes to include both surfaces and additional points
    # First determine min/max from both main surfaces
    x_values = np.concatenate([opt_K_mesh.flatten(), lgbm_K_mesh.flatten()])
    y_values = np.concatenate([opt_T_mesh.flatten(), lgbm_T_mesh.flatten()])
    z_values = np.concatenate([opt_Z.flatten(), lgbm_Z.flatten()])
    
    # Add realized points if available
    if show_realized_points and realized_points is not None:
        realized_strikes, realized_T, realized_sigma = realized_points
        if realized_strikes:
            x_values = np.append(x_values, realized_strikes)
        if realized_T:
            y_values = np.append(y_values, realized_T)
        if realized_sigma:
            z_values = np.append(z_values, realized_sigma)
    
    # Add S0 line data if available
    if spot_price is not None:
        x_values = np.append(x_values, spot_price)
    if spot_line_data is not None:
        s0_line_T, s0_line_sigma = spot_line_data
        if s0_line_T:
            y_values = np.append(y_values, s0_line_T)
        if s0_line_sigma:
            z_values = np.append(z_values, s0_line_sigma)
    
    # Calculate ranges with padding
    x_min = np.nanmin(x_values) * 0.95
    x_max = np.nanmax(x_values) * 1.05
    y_min = np.nanmin(y_values) * 0.95
    y_max = np.nanmax(y_values) * 1.05
    z_min = max(0, np.nanmin(z_values) * 0.9)
    z_max = np.nanmax(z_values) * 1.1
    
    # Configure plot layout
    fig.update_layout(
        title=f"{stock} Options IV vs LGBM Model RV @ {td}",
        scene=dict(
            xaxis_title="Strike Price (K)",
            yaxis_title="Days to Expiration (T)",
            zaxis_title="Volatility (σ)",
            xaxis_range=[x_min, x_max],
            yaxis_range=[y_min, y_max],
            zaxis_range=[z_min, z_max],
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=10, r=10, b=10, t=50),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    # Adjust camera view
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.7, y=-1.7, z=1.0)
    ))
    
    # Display the figure if requested
    if show_surface:
        fig.show()
    
    return fig

def calculate_pricing_difference(
    option_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lgbm_surface: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Calculate the percentage difference between options IV and LGBM RV surfaces.
    
    Args:
        option_surface: Tuple (K_mesh, T_mesh, Z) for options surface
        lgbm_surface: Tuple (K_mesh, T_mesh, Z) for LGBM model surface
        
    Returns:
        Tuple containing:
        - diff_Z: 2D array of percentage differences
        - stats: Dictionary with statistics about the differences
    """
    opt_K_mesh, opt_T_mesh, opt_Z = option_surface
    lgbm_K_mesh, lgbm_T_mesh, lgbm_Z = lgbm_surface
    
    # Initialize statistics
    stats = {
        "max_diff": float('-inf'),
        "max_diff_k": None,
        "max_diff_t": None,
        "min_diff": float('inf'),
        "min_diff_k": None,
        "min_diff_t": None,
        "mean_diff": None,
        "median_diff": None
    }
    
    # We need to interpolate the surfaces to a common grid
    # For simplicity, let's use the options surface grid as our base
    
    # Create interpolated LGBM Z values on the options grid
    from scipy.interpolate import griddata
    
    # Flatten the meshes for interpolation
    lgbm_points = np.column_stack((lgbm_K_mesh.flatten(), lgbm_T_mesh.flatten()))
    lgbm_values = lgbm_Z.flatten()
    
    # Target points (options mesh)
    opt_points = np.column_stack((opt_K_mesh.flatten(), opt_T_mesh.flatten()))
    
    # Interpolate LGBM values onto options grid
    interp_lgbm_Z = griddata(
        lgbm_points, lgbm_values, opt_points, 
        method='linear', fill_value=np.nan
    ).reshape(opt_Z.shape)
    
    # Calculate percentage difference: (IV - RV) / RV * 100
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_Z = (opt_Z - interp_lgbm_Z) / interp_lgbm_Z * 100
    
    # Replace infinities with NaN
    diff_Z[~np.isfinite(diff_Z)] = np.nan
    
    # Calculate statistics for non-NaN values
    valid_diffs = diff_Z[~np.isnan(diff_Z)]
    if len(valid_diffs) > 0:
        # Find max difference (most overpriced)
        max_idx = np.nanargmax(diff_Z)
        max_i, max_j = np.unravel_index(max_idx, diff_Z.shape)
        stats["max_diff"] = diff_Z[max_i, max_j]
        stats["max_diff_k"] = opt_K_mesh[max_i, max_j]
        stats["max_diff_t"] = opt_T_mesh[max_i, max_j]
        
        # Find min difference (most underpriced)
        min_idx = np.nanargmin(diff_Z)
        min_i, min_j = np.unravel_index(min_idx, diff_Z.shape)
        stats["min_diff"] = diff_Z[min_i, min_j]
        stats["min_diff_k"] = opt_K_mesh[min_i, min_j]
        stats["min_diff_t"] = opt_T_mesh[min_i, min_j]
        
        # Overall statistics
        stats["mean_diff"] = np.nanmean(diff_Z)
        stats["median_diff"] = np.nanmedian(diff_Z)
    
    return diff_Z, stats