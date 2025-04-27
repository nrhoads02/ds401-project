# vol_surface_visualization.py
# ---------------------------------------------------------------
# Plot a realized‐volatility‐surface using A/B split models for each horizon.
# Calls the prediction logic from surface_lgbm_modeling.
# ---------------------------------------------------------------

import os
import datetime as _dt
from typing import Dict, Any, List, Tuple, Union, Optional

import numpy as np
import polars as pl
import plotly.graph_objects as go
import joblib
import logging 

# Import necessary functions from the modeling script
# Using absolute imports from project structure
from src.data_modeling.surface_lgbm_modeling import predict_surface, K_RANGE, K_GRID_POINTS, HORIZONS

logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

def _to_date(x: Union[str, _dt.date, _dt.datetime, np.datetime64]) -> _dt.date:
    """Convert various date formats to date object."""
    if isinstance(x, str): return _dt.date.fromisoformat(x)
    if isinstance(x, _dt.datetime): return x.date()
    if isinstance(x, np.datetime64): return x.astype('datetime64[D]').item().date()
    if isinstance(x, _dt.date): return x
    raise TypeError(f"Cannot convert {type(x)} to date")

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
    td = _to_date(target_date)
    
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
            dates_list.append(_to_date(d))
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

# ────────────────────────────────────────────────────────────────
# Surface generation
# ────────────────────────────────────────────────────────────────
def generate_vol_surface(
    models: Dict[str, Any],
    df: pl.DataFrame,
    stock: str,
    trade_date: Union[str, _dt.date, _dt.datetime, np.datetime64],
    k_range: Tuple[float, float] = K_RANGE,
    k_grid_points: int = K_GRID_POINTS,
    show_surface: bool = False
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[Dict[str, Any]]]]:
    """
    Builds and optionally plots the sigma(K,T) volatility surface for a given stock and date.
    It calls the `predict_surface` function to get the predicted surface and overlays
    realized volatility points if available in the input DataFrame `df`.

    Args:
        models: Model dictionary returned by `load_surface_models`, containing loaded
                model boosters, feature column names, stock splits, etc.
        df: DataFrame with stock data, including features and future realized vol/returns
            needed for plotting realized points.
        stock: Stock symbol to plot.
        trade_date: The specific date for which the surface should be generated.
        k_range: Range for log-moneyness 'k'
        k_grid_points: Number of points to use for the strike (k) dimension of the surface.
        show_surface: If True, displays the interactive Plotly figure.

    Returns:
        A tuple (K_mesh, T_mesh, Z, extra_data) representing the surface arrays and additional info,
        or None if the surface generation fails.
        extra_data contains realized points and S0 line data for reuse in other visualizations.
    """
    logger.info(f"Generating surface for {stock} on {trade_date} with {k_grid_points} strike points...")

    # Find nearest date in the dataframe
    nearest_date = find_nearest_date(df, trade_date, stock)
    if nearest_date is None:
        logger.error(f"No data found for {stock} on or before {trade_date}")
        return None
    
    logger.info(f"Using nearest available date: {nearest_date}")

    # --- Call the core prediction function from the modeling script ---
    try:
        # `predict_surface` handles model selection (A/B), feature preparation (including interactions),
        # grid generation, and prediction logic based on the trained models.
        result_tuple = predict_surface(
            model_dict=models,
            df=df,
            stock=stock,
            trade_date=nearest_date,  # Use nearest date
            k_range=k_range,
            k_grid_points=k_grid_points
        )
        # Check if prediction returned valid data
        if result_tuple is None:
             logger.error(f"predict_surface returned None for {stock} on {nearest_date}. Cannot generate plot.")
             return None
        K_mesh, T_mesh, Z = result_tuple
        logger.info(f"Successfully generated predicted surface for {stock} on {nearest_date}.")

    except ValueError as ve:
        # Handle specific errors like missing data or model issues from predict_surface
        logger.error(f"Value error during surface prediction for {stock} on {nearest_date}: {ve}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during prediction
        logger.error(f"Unexpected error during surface prediction for {stock} on {nearest_date}: {e}", exc_info=True)
        return None

    # --- Prepare Realized Data for Overlay ---
    realized_strikes, realized_T, realized_sigma = [], [], []
    s0_line_T, s0_line_sigma_hist = [], []
    S0 = None
    
    # Additional data to return
    extra_data = {
        "S0": None,
        "realized_points": None,
        "spot_line_data": None,
        "nearest_date": nearest_date
    }
    
    try:
        # Filter the dataframe for the specific stock and nearest date
        today = df.filter(
            (pl.col("act_symbol") == stock) & 
            (pl.col("date").cast(pl.Date) == nearest_date)
        ).head(1) # Ensure only one row

        if not today.is_empty():
            # Convert the row to a dictionary for easy access
            row_dict = today.row(0, named=True)
            S0 = row_dict.get("close") # Get closing price
            extra_data["S0"] = S0

            if S0 is not None and np.isfinite(S0):
                # Iterate through the horizons defined in the model dictionary
                loaded_horizons = models.get("horizons", HORIZONS) # Use horizons from loaded model if available
                for h in loaded_horizons:
                    rv_col = f"rv_{h}d_future"
                    lr_col = f"log_ret_future_{h}"
                    # Check if future realized vol and return exist for this horizon
                    if rv_col in row_dict and lr_col in row_dict:
                        rv = row_dict[rv_col]
                        lr = row_dict[lr_col]
                        # Ensure values are valid numbers and realized vol is non-negative
                        if rv is not None and lr is not None and np.isfinite(rv) and np.isfinite(lr) and rv >= 0:
                            try:
                                # Calculate annualized volatility from realized variance
                                sigma0 = np.sqrt(rv / (h / 252.0)) # Assuming 252 trading days
                                # Calculate time in calendar days (consistent with T_mesh)
                                Tcal = int(round(h * 7/5))
                                # Append data for the realized point marker
                                realized_strikes.append(S0 * np.exp(lr)) # Strike price K = S0 * exp(k)
                                realized_T.append(Tcal)                 # Time T
                                realized_sigma.append(sigma0)           # Volatility sigma
                            except (ValueError, ZeroDivisionError, TypeError) as calc_err:
                                logger.warning(f"Could not calculate realized point for H={h} due to: {calc_err}")
                    
                    # 2. Extract data for S0 line (uses HISTORICAL volatility)
                    rv_hist_col = f"rv_{h}d"  # Historical realized vol column
                    rv_for_s0_line = row_dict.get(rv_hist_col)
                    if rv_for_s0_line is not None and np.isfinite(rv_for_s0_line) and rv_for_s0_line >= 0:
                        try:
                            # Calculate annualized volatility from historical RV
                            sigma0_hist_at_S0 = np.sqrt(rv_for_s0_line / (h / 252.0))
                            s0_line_T.append(Tcal)
                            s0_line_sigma_hist.append(sigma0_hist_at_S0)
                        except Exception as e: 
                            logger.warning(f"Error calculating S0 line sigma H={h}: {e}")
            else:
                 logger.warning(f"Could not retrieve valid close price S0 for {stock} on {nearest_date}.")
        else:
            logger.warning(f"No data found in DataFrame for {stock} on {nearest_date} to plot realized points.")

    except Exception as e:
         logger.error(f"Error preparing realized data points for {stock} on {nearest_date}: {e}", exc_info=True)
    
    # Store the realized points and S0 line data
    if realized_strikes:
        extra_data["realized_points"] = (realized_strikes, realized_T, realized_sigma)
    if s0_line_T:
        extra_data["spot_line_data"] = (s0_line_T, s0_line_sigma_hist)

    # --- Create Plotly Figure ---
    fig = go.Figure()

    # Define contour settings
    contour_settings = {
        "x": {"show": False}, # Disable contours on the x-plane projection
        "y": {"show": False}, # Disable contours on the y-plane projection
        "z": {                 # Configure contours on the z-plane projection
            "show": True,          # Enable z-contours
            "highlight": True,     # Highlight contours on hover
            "highlightcolor": "limegreen", # Color for highlighted contour lines
            "project": {"z": True},# Project contours onto the z-plane (floor)
        }
    }

    # Add the predicted volatility surface trace
    fig.add_trace(go.Surface(
        x=K_mesh,
        y=T_mesh,
        z=Z,
        name="LGBM Predicted Surface",
        showlegend=True,
        colorscale="Viridis", # Color scheme for the surface
        colorbar=dict(title='Ann. Vol (σ)', thickness=20, len=0.75), # Label and style for the color bar
        contours = contour_settings, # Apply contour settings
        hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                      "<b>T:</b> %{y:.0f} cd<br>" +
                      "<b>Predicted σ:</b> %{z:.4f}<extra></extra>",
        lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2) # Adjust surface lighting
    ))

    # Add the realized volatility points as scatter markers if available
    if realized_strikes:
        fig.add_trace(go.Scatter3d(
            x=realized_strikes,
            y=realized_T,
            z=realized_sigma,
            mode="markers", # Display as markers
            marker=dict(color="red", size=5, symbol='diamond'), # Marker style
            name="Realized Points", # Legend entry
            showlegend=True,
            hovertemplate="<b>Strike K:</b> %{x:.2f}<br>" +
                          "<b>T:</b> %{y:.0f} cd<br>" +
                          "<b>Realized σ:</b> %{z:.4f}<extra></extra>"
        ))
        logger.info(f"Added {len(realized_strikes)} realized points to the plot.")
    else:
         logger.info("No realized points were available or calculable to plot.")

    # Add S0 line (historical realized vol at current spot price)
    if S0 is not None and np.isfinite(S0) and s0_line_T:
        # Sort points by time for a smooth line
        s0_line_points = sorted(zip(s0_line_T, s0_line_sigma_hist))
        s0_T_sorted = [p[0] for p in s0_line_points]
        s0_sigma_sorted = [p[1] for p in s0_line_points]
        
        fig.add_trace(go.Scatter3d(
            x=[S0] * len(s0_T_sorted),   # Constant K = S0
            y=s0_T_sorted,               # T values
            z=s0_sigma_sorted,           # Historical realized sigma at S0
            mode='lines+markers',
            marker=dict(color='cyan', size=3),
            line=dict(color='cyan', width=4),
            name=f'Hist. RV @ Spot K={S0:.2f}',
            showlegend=True,
            hovertemplate="<b>Strike K:</b> %{x:.2f} (Spot)<br>" +
                          "<b>T:</b> %{y:.0f} cd<br>" +
                          "<b>Hist. RV σ:</b> %{z:.4f}<extra></extra>"
        ))
        logger.info(f"Adding S0 line with {len(s0_T_sorted)} historical RV points.")
    else: 
        logger.warning("Could not plot S0 line (missing S0 or historical RV data).")

    # Calculate axis ranges
    if S0 is not None and np.isfinite(S0):
        x_min = min(S0*0.85, np.nanmin(realized_strikes)*0.85 if realized_strikes else S0*0.85)
        x_max = max(S0*1.15, np.nanmax(realized_strikes)*1.15 if realized_strikes else S0*1.15)
    else:
        x_min = np.nanmin(K_mesh) * 0.95
        x_max = np.nanmax(K_mesh) * 1.05
        
    z_min = min(np.nanmin(Z)*0.85, np.nanmin(realized_sigma)*0.85 if realized_sigma else np.nanmin(Z)*0.85)
    z_max = max(np.nanmax(Z)*1.15, np.nanmax(realized_sigma)*1.15 if realized_sigma else np.nanmax(Z)*1.15)

    # --- Configure Plot Layout ---
    fig.update_layout(
        title=f"{stock} LGBM Volatility Surface @ {nearest_date}", # Plot title
        scene=dict(
            # Axis labels
            xaxis_title="Strike Price (K)",
            yaxis_title="Time to Maturity (Calendar Days, T)",
            zaxis_title="Annualized Volatility (σ)",
            # Set axis ranges
            xaxis_range=[x_min, x_max],
            zaxis_range=[z_min, z_max],
            aspectratio=dict(x=1.5, y=1.5, z=1) # Adjust aspect ratio for better view
        ),
        margin=dict(l=10, r=10, b=10, t=50), # Adjust margins
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01) # Position legend in top left
    )

    # Adjust camera angle for a good initial view
    fig.update_layout(scene_camera=dict(
        eye=dict(x=1.7, y=-1.7, z=1.0) # Adjust x, y, z for desired viewpoint
    ))

    # Show the plot if requested
    if show_surface:
        logger.info("Displaying interactive surface plot...")
        fig.show()
    else:
         logger.info("Surface plot generated but not displayed (show_surface=False).")

    # Return the generated surface arrays plus extra data
    return K_mesh, T_mesh, Z, extra_data