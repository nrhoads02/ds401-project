# alt_vol_surface_visualization.py
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
import logging # Added for logging potential issues

# Import necessary functions from the modeling script
# Adjust the import path based on your project structure
try:
    # Assumes modeling script is in a sibling directory 'src/data_modeling'
    from src.data_modeling.surface_lgbm_modeling import predict_surface, _to_date, K_RANGE, K_GRID_POINTS, HORIZONS
except ImportError:
    # Fallback if the src structure isn't used (e.g., running from same directory)
    try:
        from surface_lgbm_modeling import predict_surface, _to_date, K_RANGE, K_GRID_POINTS, HORIZONS
    except ImportError:
         raise ImportError("Could not import required functions from surface_lgbm_modeling. Please ensure the file is accessible.")


logger = logging.getLogger(__name__)
# Basic logging configuration if not already configured elsewhere
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

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
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
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
        k_grid_points: Number of points to use for the strike (k) dimension of the surface.
        show_surface: If True, displays the interactive Plotly figure.

    Returns:
        A tuple (K_mesh, T_mesh, Z) representing the surface arrays (Strike, Time, Vol),
        or None if the surface generation fails.
    """
    logger.info(f"Generating surface for {stock} on {trade_date} with {k_grid_points} strike points...")

    # --- Call the core prediction function from the modeling script ---
    try:
        # `predict_surface` handles model selection (A/B), feature preparation (including interactions),
        # grid generation, and prediction logic based on the trained models.
        result_tuple = predict_surface(
            model_dict=models,
            df=df,
            stock=stock,
            trade_date=trade_date,
            k_range=k_range,
            k_grid_points=k_grid_points
        )
        # Check if prediction returned valid data
        if result_tuple is None:
             logger.error(f"predict_surface returned None for {stock} on {trade_date}. Cannot generate plot.")
             return None
        K_mesh, T_mesh, Z = result_tuple
        logger.info(f"Successfully generated predicted surface for {stock} on {trade_date}.")

    except ValueError as ve:
        # Handle specific errors like missing data or model issues from predict_surface
        logger.error(f"Value error during surface prediction for {stock} on {trade_date}: {ve}")
        return None
    except Exception as e:
        # Handle any other unexpected errors during prediction
        logger.error(f"Unexpected error during surface prediction for {stock} on {trade_date}: {e}", exc_info=True)
        return None

    # --- Prepare Realized Data for Overlay ---
    realized_strikes, realized_T, realized_sigma = [], [], []
    try:
        td = _to_date(trade_date)
        # Filter the dataframe for the specific stock and trade date
        today = df.filter(
            (pl.col("act_symbol") == stock) & (pl.col("date").cast(pl.Date) == td)
        ).head(1) # Ensure only one row

        if not today.is_empty():
            # Convert the row to a dictionary for easy access
            row_dict = today.row(0, named=True)
            S0 = row_dict.get("close") # Get closing price

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
            else:
                 logger.warning(f"Could not retrieve valid close price S0 for {stock} on {td}.")
        else:
            logger.warning(f"No data found in DataFrame for {stock} on {td} to plot realized points.")

    except Exception as e:
         logger.error(f"Error preparing realized data points for {stock} on {td}: {e}", exc_info=True)


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
        name="Predicted Surface",
        colorscale="Viridis", # Color scheme for the surface
        colorbar=dict(title='Ann. Vol (σ)', thickness=20, len=0.75), # Label and style for the color bar
        contours = contour_settings, # Apply contour settings
        hoverinfo='x+y+z', # Information to show on hover
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
            name="Realized Points" # Legend entry
        ))
        logger.info(f"Added {len(realized_strikes)} realized points to the plot.")
    else:
         logger.info("No realized points were available or calculable to plot.")


    # --- Configure Plot Layout ---
    fig.update_layout(
        title=f"{stock} Volatility Surface @ {td}", # Plot title
        scene=dict(
            # Axis labels
            xaxis_title="Strike Price (K)",
            yaxis_title="Time to Maturity (Calendar Days, T)",
            zaxis_title="Annualized Volatility (σ)",
            # Set axis ranges - ensure z starts at 0 or slightly below min vol
            xaxis_range=[min(S0*0.85, np.nanmin(realized_strikes)*0.85),max(S0*1.15, np.nanmax(realized_strikes)*1.15)],
            zaxis_range=[min(np.nanmin(Z)*0.85,np.nanmin(realized_sigma)*0.85), 
                         max(np.nanmax(Z)*1.15,np.nanmax(realized_sigma)*1.15)],
            aspectratio=dict(x=1.5, y=1.5, z=1) # Adjust aspect ratio for better view
        ),
        margin=dict(l=10, r=10, b=10, t=50) # Adjust margins
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

    # Return the generated surface arrays
    return K_mesh, T_mesh, Z
