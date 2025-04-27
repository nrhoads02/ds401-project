# streamlit_app.py
import os
import sys

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Standard libraries
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime
import logging
from typing import Dict, Any, List, Tuple, Union, Optional
from scipy.interpolate import griddata # Needed for 2D interpolation

# --- Call set_page_config() as the first Streamlit command ---
st.set_page_config(page_title="Volatility Surface App", layout="wide")

# Project-specific imports
try:
    from src.data_transformation.transformation_pipeline import transformation_pipeline
    from src.data_transformation.stock_adjustments import adjust_option_splits
    from src.data_extraction.dataframe_loader import load_data
    from src.data_modeling.surface_lgbm_modeling import (
        load_surface_models,
        HORIZONS,
        K_RANGE, # Still needed for *generating* surface points
        K_GRID_POINTS, # Used for defining grid density
    )
    from src.data_modeling.vol_surface_visualization import (
        generate_vol_surface,
        find_nearest_date,
        _to_date
    )
    from src.data_modeling.option_chain_modeling import (
        generate_options_surface,
        # plot_options_surface, # Defined below
        compare_surfaces, # Defined below
        # calculate_pricing_difference # Replaced by calculate_pricing_difference_on_axes
    )
except ImportError as e:
    st.error(f"Failed to import project modules. Ensure the script is run from the 'app/' directory or the project structure is correct. Error: {e}")
    sys.exit(1)


# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
AXIS_PADDING_FACTOR = 0.15 # Controls padding for individual surface axis calculation
# Updated color constants for better contrast and visibility
OPTIONS_CLOSER_COLOR = 'rgba(0, 180, 0, 0.9)'  # Bright green for Options closer (swapped)
LGBM_CLOSER_COLOR = 'rgba(255, 20, 147, 0.9)'  # Hot pink for LGBM closer (swapped)
DEFAULT_RV_COLOR = 'rgba(255, 165, 0, 0.9)'  # Orange (default if only one surface or equal)
S0_LINE_COLOR = 'rgba(0, 191, 255, 0.9)'  # DeepSkyBlue for S0 line
TARGET_GRID_RESOLUTION = 50 # Number of points for K and T axes on common comparison grid
SURFACE_OPACITY = 0.95 # Opacity for 3D surfaces
SHARED_SURFACE_OPACITY = 0.65 # Opacity for shared surfaces

# -----------------------------------------------------------------------------
# About text
# -----------------------------------------------------------------------------
ABOUT_TEXT = r"""
### What Is an Option and Why Does Volatility Matter?

An **option** is a contract giving the right—but not the obligation—to buy or sell a stock at a fixed price on or before a certain date.

Options are priced largely based on **volatility**, which measures how much the stock's price moves. Higher volatility raises the chance of large price swings, making options more expensive.


---


### Implied vs. Local Volatility Surfaces

- **Implied volatility** is the market's expectation of future volatility, backed out of option prices across strikes and expiries.

- A **local volatility surface** goes deeper: it asks

  > *"What instantaneous volatility at each stock price and future date would make today's option prices consistent under a risk‑neutral model?"*

  One inverts the option pricing model (Dupire's formula) to recover this surface, which is essential for pricing more complex derivatives.


---


### Realized Local Volatility Surface

The **Realized Local Volatility Surface (RLVS)** brings these ideas to historical data:


1. **Realized volatility** is what actually happened: how much the stock price varied over a past window.

2. A simple realized‑vol curve is only a single line for the one path the stock took.

3. The RLVS estimates the conditional expectation

   $$
   \sigma_{\mathrm{real}}(K, T)
   \;=\;
   \mathbb{E}\bigl[\mathrm{RealizedVol}\,\bigm|\tfrac{S_T}{S_0}=e^{k}\bigr],
   $$

   where $K = S_0 e^{k}$ is the strike and $T$ is the time to expiry.

   It tells you,

   > *"Had the stock ended at strike \(K\) in \(T\) days, what volatility would we have realized?"*


---


### Comparing Options IV and LGBM Model RV

This app allows you to:

1. Visualize the **LGBM-modeled realized volatility surface** based on historical data
2. Generate an **options chain implied volatility surface** from market prices
3. **Compare both surfaces** to identify potential pricing inefficiencies

When Options IV is significantly higher than the LGBM RV, this might indicate:
- Market expectations of higher future volatility
- Risk premium for certain strike/expiry combinations
- Potential overpricing that could present trading opportunities

**Note:** All strike prices (K) displayed in the plots and tables are **split-adjusted** to be comparable across different dates.
"""

# -----------------------------------------------------------------------------
# Helper functions for comparisons and Plotting
# -----------------------------------------------------------------------------

def interpolate_surface_at_points(
    surface_K_mesh: np.ndarray,
    surface_T_mesh: np.ndarray,
    surface_Z: np.ndarray,
    points_K: Union[List[float], np.ndarray], # Accept list or array
    points_T: Union[List[float], np.ndarray], # Accept list or array
    method='linear' # Allow specifying interpolation method
) -> Optional[np.ndarray]: # Return numpy array or None
    """
    Interpolates the surface Z value at the given target (K, T) points using griddata.
    Returns a flat numpy array of interpolated Z values, or None if interpolation fails.
    """
    target_K_flat = np.asarray(points_K).flatten()
    target_T_flat = np.asarray(points_T).flatten()

    if target_K_flat.size == 0 or target_T_flat.size == 0 or target_K_flat.size != target_T_flat.size:
        logger.error("Interpolation failed: Invalid target points K or T.")
        return None

    # Check if source surface data is valid
    if surface_K_mesh is None or surface_T_mesh is None or surface_Z is None or \
       surface_K_mesh.size == 0 or surface_T_mesh.size == 0 or surface_Z.size == 0 or \
       surface_K_mesh.shape != surface_T_mesh.shape or surface_K_mesh.shape != surface_Z.shape:
        logger.warning("Interpolation failed: Input source surface data is invalid or empty or has mismatched shapes.")
        # Return array of NaNs with the expected size
        return np.full(target_K_flat.shape, np.nan)

    k_flat, t_flat, z_flat = surface_K_mesh.flatten(), surface_T_mesh.flatten(), surface_Z.flatten()

    # Filter out points where source K, T, or Z is NaN/inf
    valid_indices = np.isfinite(k_flat) & np.isfinite(t_flat) & np.isfinite(z_flat)
    if not np.any(valid_indices):
        logger.warning("Interpolation failed: Source surface K, T, or Z contains only non-finite values.")
        return np.full(target_K_flat.shape, np.nan)

    # Create the array of known points (K, T) and corresponding Z values
    points_for_interp = np.column_stack((k_flat[valid_indices], t_flat[valid_indices]))
    values_for_interp = z_flat[valid_indices]

    # Target points where we want to interpolate
    target_points = np.column_stack((target_K_flat, target_T_flat))

    try:
        # Use griddata for 2D interpolation
        logger.debug(f"Calling griddata with {points_for_interp.shape[0]} source points and {target_points.shape[0]} target points, method='{method}'")
        interpolated_z = griddata(points_for_interp, values_for_interp, target_points, method=method)
        # griddata returns NaN for points outside the convex hull (linear) or based on fill_value
        logger.debug("griddata call successful.")
        return interpolated_z # Return the flat numpy array
    except Exception as e:
        logger.error(f"Interpolation using griddata failed: {e}", exc_info=True)
        return None # Indicate failure

def calculate_point_differences(
    surface_name: str,
    surface_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], # Allow None
    realized_points: Optional[Tuple[List[float], List[float], List[float]]]
) -> Optional[Dict[str, Any]]:
    """ Calculates differences between a surface and realized points using interpolation. """
    if realized_points is None or not realized_points[0]: return None

    realized_K, realized_T, realized_sigma = realized_points
    # Check if surface_tuple is valid before unpacking
    if not isinstance(surface_tuple, tuple) or len(surface_tuple) != 3 or \
       surface_tuple[0] is None or surface_tuple[1] is None or surface_tuple[2] is None:
         logger.warning(f"Cannot calculate point differences for {surface_name}: Invalid or None surface_tuple.")
         # Still return details structure but with NaNs
         interpolated_Z = [None] * len(realized_K)
    else:
        surface_K_mesh, surface_T_mesh, surface_Z = surface_tuple
        # Use the interpolation helper function
        interpolated_Z_array = interpolate_surface_at_points(surface_K_mesh, surface_T_mesh, surface_Z, realized_K, realized_T)
        interpolated_Z = interpolated_Z_array.tolist() if interpolated_Z_array is not None else [None] * len(realized_K)


    differences, abs_differences, point_details = [], [], []
    for i, real_s in enumerate(realized_sigma):
        # Handle potential index error if interpolation failed unexpectedly
        interp_s = interpolated_Z[i] if i < len(interpolated_Z) else None
        # Ensure realized sigma is valid
        is_real_s_valid = np.isfinite(real_s)
        # Check if interpolated value is valid (non-None and finite)
        is_interp_s_valid = interp_s is not None and np.isfinite(interp_s)

        detail = {
            "K": realized_K[i],
            "T": realized_T[i],
            "Realized σ": real_s if is_real_s_valid else np.nan,
            f"{surface_name} σ": interp_s if is_interp_s_valid else np.nan,
            "Difference": np.nan
        }
        # Calculate difference only if both realized and interpolated values are valid
        if is_real_s_valid and is_interp_s_valid:
            diff = real_s - interp_s
            differences.append(diff)
            abs_differences.append(abs(diff))
            detail["Difference"] = diff

        point_details.append(detail)

    if not differences: # No valid comparisons could be made
        logger.warning(f"No valid differences calculated between {surface_name} and realized points.")
        # Return details even if averages are NaN
        return {"average_diff": np.nan, "average_abs_diff": np.nan, "point_details": point_details}

    avg_diff = np.mean(differences)
    avg_abs_diff = np.mean(abs_differences)
    return {"average_diff": avg_diff, "average_abs_diff": avg_abs_diff, "point_details": point_details}

def display_point_differences(
    surface_name: str,
    diff_results: Optional[Dict[str, Any]],
    realized_available: bool
):
    """ Displays the calculated differences in an expander. """
    if not realized_available:
        # Check if diff_results exist but maybe calculation failed
        if diff_results and diff_results.get("point_details"):
            st.info(f"Realized volatility points are not available/shown, but comparison to {surface_name} was attempted.")
        else:
             st.info("Realized volatility points are not available or not shown for comparison.")
        return

    if diff_results:
        with st.expander(f"Comparison: {surface_name} vs. Realized Volatility Points"):
            avg_abs_diff = diff_results.get('average_abs_diff', np.nan)
            avg_diff = diff_results.get('average_diff', np.nan)

            st.metric(f"Average Absolute Difference |Realized σ - {surface_name} σ|",
                      f"{avg_abs_diff:.4f}" if np.isfinite(avg_abs_diff) else "N/A",
                      help="Average absolute difference. Lower is better.")
            st.metric(f"Average Signed Difference (Realized σ - {surface_name} σ)",
                      f"{avg_diff:.4f}" if np.isfinite(avg_diff) else "N/A",
                      help="Average signed difference. Positive means Realized > Predicted on average.")

            point_details = diff_results.get("point_details")
            if point_details:
                st.markdown("##### Differences per Point:")
                try:
                    # Create DataFrame, handle potential errors
                    diff_df = pl.DataFrame(point_details)
                    # Add '$' prefix to K column for display if it exists
                    if "K" in diff_df.columns:
                       diff_df = diff_df.with_columns(
                           pl.col("K").cast(pl.Float64).map_elements(lambda k: f"${k:.2f}", return_dtype=pl.Utf8).alias("K_Display")
                       )
                       # Select and order columns for display
                       display_cols_order = ["K_Display", "T", "Realized σ", f"{surface_name} σ", "Difference"]
                       # Filter out columns that don't exist in df
                       cols_to_display = [col for col in display_cols_order if col in diff_df.columns or col == "K_Display"]
                       diff_df_display = diff_df.select(cols_to_display).rename({"K_Display": "K ($)"}) # Rename K for display
                    else:
                        # Fallback if K column missing
                        display_cols_order = ["T", "Realized σ", f"{surface_name} σ", "Difference"]
                        cols_to_display = [col for col in display_cols_order if col in diff_df.columns]
                        diff_df_display = diff_df.select(cols_to_display)

                    # Format numeric columns
                    diff_df_display = diff_df_display.with_columns(pl.col(pl.Float64).round(4))

                    st.dataframe(diff_df_display, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not display difference details as DataFrame: {e}")
                    # Fallback to showing raw list of dictionaries
                    # st.json(point_details, expanded=False)
            else:
                st.warning("No difference details available.")
    else:
        # This case might be hit if calculate_point_differences returned None
        st.warning(f"Could not calculate differences between {surface_name} surface and realized points (function returned None).")


# --- Helper function for RV point coloring ---
def get_rv_point_colors_and_hovers(
    realized_points: Tuple[List[float], List[float], List[float]],
    lgbm_surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    options_surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    option_type: str = "Avg" # Needed for naming
) -> Tuple[List[str], List[str]]:
    """
    Determines colors and hover text for RV points based on proximity to surfaces.
    Uses interpolation to get surface values at RV points.
    """
    realized_K, realized_T, realized_sigma = realized_points
    n_points = len(realized_K)
    colors = [DEFAULT_RV_COLOR] * n_points
    hovers = [""] * n_points

    # Interpolate both surfaces at all realized points
    interp_lgbm_array, interp_opts_array = None, None
    if lgbm_surface and lgbm_surface[0] is not None: # Check surface validity
        interp_lgbm_array = interpolate_surface_at_points(lgbm_surface[0], lgbm_surface[1], lgbm_surface[2], realized_K, realized_T)
    if options_surface and options_surface[0] is not None: # Check surface validity
        interp_opts_array = interpolate_surface_at_points(options_surface[0], options_surface[1], options_surface[2], realized_K, realized_T)

    # Convert numpy arrays (or None) to lists for easier iteration, handle None result from interp
    interp_lgbm = interp_lgbm_array.tolist() if interp_lgbm_array is not None else [None] * n_points
    interp_opts = interp_opts_array.tolist() if interp_opts_array is not None else [None] * n_points


    for i in range(n_points):
        real_s = realized_sigma[i]
        k_val = realized_K[i]
        t_val = realized_T[i]

        lgbm_s = interp_lgbm[i] # Already None if interpolation failed
        opts_s = interp_opts[i] # Already None if interpolation failed

        # Check if the realized sigma itself is valid
        is_real_s_valid = np.isfinite(real_s)
        if not is_real_s_valid:
            lgbm_s_text = f"{lgbm_s:.4f}" if lgbm_s is not None and np.isfinite(lgbm_s) else 'N/A'
            opts_s_text = f"{opts_s:.4f}" if opts_s is not None and np.isfinite(opts_s) else 'N/A'
            hover_text = (f"<b>Realized Point</b><br>"
                          f"K: ${k_val:.2f}, T: {t_val:.0f} cd<br>" # Added $
                          f"Realized σ: NaN<br>"
                          f"LGBM RV σ: {lgbm_s_text}<br>" # Use full name
                          f"Options IV ({option_type}) σ: {opts_s_text}<br>" # Use full name
                          "<extra></extra>")
            colors[i] = 'grey' # Indicate invalid realized point
            hovers[i] = hover_text
            continue

        # Calculate absolute differences, check if interp result is finite
        diff_lgbm = abs(real_s - lgbm_s) if lgbm_s is not None and np.isfinite(lgbm_s) else np.inf
        diff_opts = abs(real_s - opts_s) if opts_s is not None and np.isfinite(opts_s) else np.inf

        hover_text = (f"<b>Realized Point</b><br>"
                      f"K: ${k_val:.2f}, T: {t_val:.0f} cd<br>" # Added $
                      f"Realized σ: {real_s:.4f}<br>")

        # Determine color and add details to hover text
        lgbm_valid_interp = (lgbm_s is not None and np.isfinite(lgbm_s))
        opts_valid_interp = (opts_s is not None and np.isfinite(opts_s))

        if lgbm_valid_interp and opts_valid_interp: # Both surfaces interpolated successfully
            hover_text += f"LGBM RV σ: {lgbm_s:.4f} (Diff: {diff_lgbm:.4f})<br>" # Use full name
            hover_text += f"Options IV ({option_type}) σ: {opts_s:.4f} (Diff: {diff_opts:.4f})<br>" # Use full name
            # Use a small tolerance for equality check
            if np.isclose(diff_lgbm, diff_opts):
                hover_text += "<i>Roughly equidistant</i>"
                colors[i] = DEFAULT_RV_COLOR
            elif diff_lgbm < diff_opts:
                colors[i] = LGBM_CLOSER_COLOR
                hover_text += "<i>Closer: LGBM RV</i>" # Use full name
            else: # diff_opts < diff_lgbm
                colors[i] = OPTIONS_CLOSER_COLOR
                hover_text += f"<i>Closer: Options IV ({option_type})</i>" # Use full name
        elif lgbm_valid_interp: # Only LGBM interpolated successfully
            hover_text += f"LGBM RV σ: {lgbm_s:.4f} (Diff: {diff_lgbm:.4f})<br>" # Use full name
            hover_text += f"Options IV ({option_type}) σ: N/A<br>" # Use full name
            colors[i] = DEFAULT_RV_COLOR # Keep default for consistency
        elif opts_valid_interp: # Only Options interpolated successfully
            hover_text += f"LGBM RV σ: N/A<br>" # Use full name
            hover_text += f"Options IV ({option_type}) σ: {opts_s:.4f} (Diff: {diff_opts:.4f})<br>" # Use full name
            colors[i] = DEFAULT_RV_COLOR # Keep default
        else: # Neither interpolated successfully
             hover_text += "LGBM RV σ: N/A<br>" # Use full name
             hover_text += f"Options IV ({option_type}) σ: N/A<br>" # Use full name
             colors[i] = DEFAULT_RV_COLOR # No comparison possible

        hover_text += "<extra></extra>"
        hovers[i] = hover_text

    return colors, hovers


# --- LGBM Plotting function with specific axis logic ---
def create_plotly_figure(
    K_mesh: np.ndarray,
    T_mesh: np.ndarray,
    Z: np.ndarray,
    today_data: Optional[pl.DataFrame],
    stock: str,
    trade_date: datetime.date,
    loaded_horizons: List[int],
    show_realized: bool,
    show_s0: bool
) -> Tuple[go.Figure, Dict[str, Any]]:
    """
    Creates the Plotly 3D surface plot for LGBM with SPECIFIC axis scaling
    based on S0 and realized points only. Also returns calculated axis ranges and overlay data.
    """
    fig = go.Figure()
    S0, row_dict = np.nan, {}
    if today_data is not None and not today_data.is_empty():
        try:
            # Ensure today_data is not empty before accessing row
            row_dict = today_data.row(0, named=True)
            S0 = row_dict.get("close", np.nan)
            if S0 is None or S0 <= 0: S0 = np.nan # More robust check
        except IndexError:
            logger.warning(f"Cannot access row 0 for {trade_date}, today_data might be empty despite check.")
            row_dict = {} # Ensure row_dict is a dict
        except Exception as e:
            logger.error(f"Error extracting row data for {trade_date}: {e}")
            row_dict = {}

    realized_strikes, realized_T, realized_sigma = [], [], []
    s0_line_T, s0_line_sigma_hist = [], []

    # Extract overlay data
    if row_dict and not np.isnan(S0):
        for h in loaded_horizons:
            Tcal = int(round(h * 7/5)) # Convert business days horizon to calendar days T
            # Realized points (Future RV converted to annualized sigma at future K)
            rv_future_col, lr_future_col = f"rv_{h}d_future", f"log_ret_future_{h}"
            if rv_future_col in row_dict and lr_future_col in row_dict:
                rv_fut = row_dict.get(rv_future_col)
                lr_fut = row_dict.get(lr_future_col)
                # Check if values are not None and are finite numbers
                if rv_fut is not None and lr_fut is not None and np.isfinite(rv_fut) and np.isfinite(lr_fut) and rv_fut >= 0:
                    try:
                        # Calculate annualized sigma from realized variance
                        # Avoid division by zero if h=0
                        if h > 0:
                           sigma0_fut = np.sqrt(rv_fut / (h / 252.0)) # Annualize
                        else: sigma0_fut = np.nan # Define sigma as NaN if horizon is 0
                        realized_K = S0 * np.exp(lr_fut) # Calculate the strike where this RV was realized
                        # Only add if sigma is finite and positive
                        if np.isfinite(sigma0_fut) and sigma0_fut > 0:
                            realized_strikes.append(realized_K)
                            realized_T.append(Tcal)
                            realized_sigma.append(sigma0_fut)
                    except (ValueError, FloatingPointError, ZeroDivisionError) as math_err:
                         logger.warning(f"Math error calculating realized point for h={h}: {math_err}")
                         pass # Ignore math errors

            # S0 line (Historical RV converted to annualized sigma at current K=S0)
            rv_hist_col = f"rv_{h}d"
            rv_for_s0_line = row_dict.get(rv_hist_col)
            if rv_for_s0_line is not None and np.isfinite(rv_for_s0_line) and rv_for_s0_line >= 0:
                try:
                    # Avoid division by zero if h=0
                    if h > 0:
                       sigma0_hist_at_S0 = np.sqrt(rv_for_s0_line / (h / 252.0)) # Annualize
                    else: sigma0_hist_at_S0 = np.nan # Define as NaN if h=0
                    # Only add if sigma is finite and positive
                    if np.isfinite(sigma0_hist_at_S0) and sigma0_hist_at_S0 > 0:
                        s0_line_T.append(Tcal)
                        s0_line_sigma_hist.append(sigma0_hist_at_S0)
                except (ValueError, FloatingPointError, ZeroDivisionError) as math_err:
                    logger.warning(f"Math error calculating S0 line point for h={h}: {math_err}")
                    pass # Ignore math errors

    # --- Axis Range Calculation (LGBM specific logic) ---
    lgbm_k_axis_elements = []
    lgbm_sigma_axis_elements = []
    realized_points_data = None
    spot_line_data = None

    # 1. Determine K range based ONLY on S0 and Realized K (if shown)
    if not np.isnan(S0): lgbm_k_axis_elements.append(S0)
    if show_realized and realized_strikes:
        lgbm_k_axis_elements.extend(k for k in realized_strikes if np.isfinite(k)) # Filter non-finite strikes
        # Store valid realized points
        valid_realized_indices = [i for i, (k, t, s) in enumerate(zip(realized_strikes, realized_T, realized_sigma)) if np.isfinite(k) and np.isfinite(t) and np.isfinite(s)]
        if valid_realized_indices:
            realized_points_data = (
                [realized_strikes[i] for i in valid_realized_indices],
                [realized_T[i] for i in valid_realized_indices],
                [realized_sigma[i] for i in valid_realized_indices]
            )

    # Calculate K range with padding
    if lgbm_k_axis_elements:
        # Filter finite elements before min/max
        finite_k_elements = [k for k in lgbm_k_axis_elements if np.isfinite(k)]
        if finite_k_elements:
            lgbm_k_min_calc = np.min(finite_k_elements)
            lgbm_k_max_calc = np.max(finite_k_elements)
            k_range_val = lgbm_k_max_calc - lgbm_k_min_calc
            k_padding = max(k_range_val * AXIS_PADDING_FACTOR / 2.0, 1.0) # Ensure some minimal padding
            lgbm_k_min = max(0.0, lgbm_k_min_calc - k_padding) # Prevent negative strikes
            lgbm_k_max = lgbm_k_max_calc + k_padding
        else: # Fallback if only non-finite K elements (e.g., S0 is NaN)
            lgbm_k_min = np.nanmin(K_mesh) * 0.9 if np.any(np.isfinite(K_mesh)) else 0
            lgbm_k_max = np.nanmax(K_mesh) * 1.1 if np.any(np.isfinite(K_mesh)) else 100
    else: # Fallback if S0 is NaN and no realized points shown/valid
        lgbm_k_min = np.nanmin(K_mesh) * 0.9 if np.any(np.isfinite(K_mesh)) else 0
        lgbm_k_max = np.nanmax(K_mesh) * 1.1 if np.any(np.isfinite(K_mesh)) else 100 # Default max K

    # 2. Determine Sigma range based ONLY on S0 line sigmas and Realized sigmas (if shown)
    #    and surface Z values *within the calculated K range*
    if show_s0 and not np.isnan(S0) and s0_line_T:
        # Store valid spot line points
        valid_spot_indices = [i for i, (t, s) in enumerate(zip(s0_line_T, s0_line_sigma_hist)) if np.isfinite(t) and np.isfinite(s)]
        if valid_spot_indices:
            s0_T_sorted = [s0_line_T[i] for i in valid_spot_indices]
            s0_sigma_sorted = [s0_line_sigma_hist[i] for i in valid_spot_indices]
            # Sort by T for plotting
            sorted_pairs = sorted(zip(s0_T_sorted, s0_sigma_sorted))
            s0_T_sorted = [p[0] for p in sorted_pairs]
            s0_sigma_sorted = [p[1] for p in sorted_pairs]
            lgbm_sigma_axis_elements.extend(s0_sigma_sorted)
            spot_line_data = (s0_T_sorted, s0_sigma_sorted) # Store for return

    if show_realized and realized_points_data: # Use the filtered data
        lgbm_sigma_axis_elements.extend(realized_points_data[2]) # Add realized sigma values

    # Filter surface Z based on K range
    # Add check for Z validity
    if Z is not None and Z.size > 0 and K_mesh is not None and K_mesh.size == Z.size:
         k_flat = K_mesh.flatten()
         z_flat = Z.flatten()
         # Ensure mask is boolean and handles NaNs correctly
         within_k_range_mask = (k_flat >= lgbm_k_min) & (k_flat <= lgbm_k_max) & np.isfinite(z_flat)
         if np.any(within_k_range_mask):
             lgbm_sigma_axis_elements.extend(z_flat[within_k_range_mask].tolist())

    # Calculate Sigma range with padding
    if lgbm_sigma_axis_elements:
        finite_sigma_elements = [s for s in lgbm_sigma_axis_elements if np.isfinite(s)]
        if finite_sigma_elements:
            lgbm_sigma_min_calc = np.min(finite_sigma_elements)
            lgbm_sigma_max_calc = np.max(finite_sigma_elements)
            sigma_range_val = lgbm_sigma_max_calc - lgbm_sigma_min_calc
            sigma_padding = max(sigma_range_val * AXIS_PADDING_FACTOR / 2.0, 0.01) # Ensure minimal padding
            lgbm_sigma_min = max(0.0, lgbm_sigma_min_calc - sigma_padding) # Prevent negative sigma
            lgbm_sigma_max = lgbm_sigma_max_calc + sigma_padding
        else: # Fallback if all elements were NaN/inf
            lgbm_sigma_min, lgbm_sigma_max = 0.0, 1.0
    else: # Fallback if no elements to consider
        lgbm_sigma_min, lgbm_sigma_max = 0.0, 1.0 # Default sigma range

    # Ensure max > min, handle potential edge cases
    if lgbm_k_max <= lgbm_k_min: lgbm_k_max = lgbm_k_min + 10.0 # Add arbitrary range if min=max
    if lgbm_sigma_max <= lgbm_sigma_min: lgbm_sigma_max = lgbm_sigma_min + 0.1 # Add arbitrary range

    # --- Plotting ---
    # Plot main surface only if Z is valid
    if Z is not None and np.any(np.isfinite(Z)):
        fig.add_trace(go.Surface(
            x=K_mesh, y=T_mesh, z=Z, name="LGBM Predicted RV", showlegend=True,
            colorscale="Plasma", opacity=SURFACE_OPACITY, # Use constant
            colorbar=dict(title='LGBM RV (σ)', thickness=15, len=0.6, y=0.8),
            contours={"z": {"show": True, "highlight": True, "highlightcolor": "limegreen", "project": {"z": True}}},
            hovertemplate="<b>LGBM RV</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>", # Added $
            lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2)
        ))
    else:
        logger.warning("LGBM surface Z data is invalid or empty, not plotting surface.")

    # Plot overlays using the filtered/validated data
    if show_realized and realized_points_data:
        fig.add_trace(go.Scatter3d(
            x=realized_points_data[0], y=realized_points_data[1], z=realized_points_data[2], mode="markers",
            marker=dict(color=DEFAULT_RV_COLOR, size=5, symbol='diamond'),
            name="Realized Points", showlegend=True,
            hovertemplate="<b>Realized Point</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" # Added $
        ))

    if show_s0 and spot_line_data and not np.isnan(S0):
        fig.add_trace(go.Scatter3d(
            x=[S0] * len(spot_line_data[0]), y=spot_line_data[0], z=spot_line_data[1], mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=3), line=dict(color=S0_LINE_COLOR, width=4),
            name=f'Hist. RV @ Spot K=${S0:.2f}', showlegend=True, # Added $
            hovertemplate="<b>Hist RV@Spot</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" # Added $
        ))

    # Get T range for consistency across plots (from the input mesh)
    t_min = np.nanmin(T_mesh) if np.any(np.isfinite(T_mesh)) else 0
    t_max = np.nanmax(T_mesh) if np.any(np.isfinite(T_mesh)) else 365 # Default max T

    fig.update_layout(
        title=f"{stock} LGBM Predicted RV Surface @ {trade_date}", # Added "Surface"
        scene=dict(
            xaxis_title="Strike Price (K, $)", # Added $
            yaxis_title="Time to Maturity (Days, T)",
            zaxis_title="Annualized Volatility (σ)",
            xaxis_range=[lgbm_k_min, lgbm_k_max], # Apply SPECIFIC ranges
            zaxis_range=[lgbm_sigma_min, lgbm_sigma_max],
            yaxis_range=[t_min, t_max], # Use T range from mesh
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=10, r=10, b=10, t=50), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0))
    )

    # Return calculated axes and overlay data
    axis_ranges = {
        "k_min": lgbm_k_min, "k_max": lgbm_k_max,
        "sigma_min": lgbm_sigma_min, "sigma_max": lgbm_sigma_max,
        "t_min": t_min, "t_max": t_max,
        "S0": S0, # Include S0 used for calculations
        "realized_points": realized_points_data, # Pass filtered data
        "spot_line_data": spot_line_data # Pass filtered data
    }
    return fig, axis_ranges


# --- Options Plotting function with specific axis logic ---
def plot_options_surface(
    opt_K_mesh: np.ndarray,
    opt_T_mesh: np.ndarray,
    opt_Z: np.ndarray,
    stock: str,
    trade_date: datetime.date,
    option_type: str,
    axis_ranges: Dict[str, float], # Uses pre-calculated ranges based on *actual* chain K and surface IV
    show_realized: bool = False,
    realized_points: Optional[Tuple[List[float], List[float], List[float]]] = None, # Expects filtered data
    show_s0: bool = False,
    spot_price: Optional[float] = None, # Expects S0 value if available
    spot_line_data: Optional[Tuple[List[int], List[float]]] = None # Expects filtered data
) -> go.Figure:
    """
    Plots the 3D options implied volatility surface with optional overlays,
    using axis ranges derived from option chain K and surface IV.
    """
    fig = go.Figure()

    # Plot main surface only if Z is valid
    if opt_Z is not None and np.any(np.isfinite(opt_Z)):
        fig.add_trace(go.Surface(
            x=opt_K_mesh, y=opt_T_mesh, z=opt_Z,
            name=f"Options IV ({option_type.capitalize()})", showlegend=True,
            colorscale="Viridis", opacity=SURFACE_OPACITY, # Use constant
            colorbar=dict(title='Options IV (σ)', thickness=15, len=0.6, y=0.8),
            contours={"z": {"show": True, "highlight": True, "highlightcolor": "yellow", "project": {"z": True}}},
            hovertemplate="<b>Options IV</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>", # Added $
            lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2)
        ))
    else:
        logger.warning("Options surface Z data is invalid or empty, not plotting surface.")

    # Plot overlays using the filtered/validated data passed in
    if show_realized and realized_points and realized_points[0]: # Check if list is not empty
        fig.add_trace(go.Scatter3d(
            x=realized_points[0], y=realized_points[1], z=realized_points[2], mode="markers",
            marker=dict(color=DEFAULT_RV_COLOR, size=5, symbol='diamond'),
            name="Realized Points", showlegend=True,
            hovertemplate="<b>Realized Pt</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" # Added $
        ))

    if show_s0 and spot_price is not None and np.isfinite(spot_price) and spot_line_data:
        s0_T_sorted, s0_sigma_sorted = spot_line_data
        fig.add_trace(go.Scatter3d(
            x=[spot_price] * len(s0_T_sorted), y=s0_T_sorted, z=s0_sigma_sorted, mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=3), line=dict(color=S0_LINE_COLOR, width=4),
            name=f'Hist. RV @ Spot K=${spot_price:.2f}', showlegend=True, # Added $
            hovertemplate="<b>Hist RV@Spot</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" # Added $
        ))

    # Update layout using PRE-CALCULATED axis ranges passed in
    # Provide default ranges if axis_ranges is None or keys are missing
    k_min = axis_ranges.get("k_min", 0) if axis_ranges else 0
    k_max = axis_ranges.get("k_max", 100) if axis_ranges else 100
    sigma_min = axis_ranges.get("sigma_min", 0) if axis_ranges else 0
    sigma_max = axis_ranges.get("sigma_max", 1.0) if axis_ranges else 1.0
    t_min = axis_ranges.get("t_min", 0) if axis_ranges else 0
    t_max = axis_ranges.get("t_max", 365) if axis_ranges else 365

    fig.update_layout(
        title=f"{stock} Options IV Surface ({option_type.capitalize()}) @ {trade_date}", # Added "Surface"
        scene=dict(
            xaxis_title="Strike Price (K, $)", # Added $
            yaxis_title="Time to Maturity (Days, T)",
            zaxis_title="Implied Volatility (σ)",
            xaxis_range=[k_min, k_max],
            zaxis_range=[sigma_min, sigma_max],
            yaxis_range=[t_min, t_max],
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=10, r=10, b=10, t=50), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0))
    )
    return fig


# --- Comparison Plotting function with specific axis logic and colored RV points ---
def compare_surfaces(
    options_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lgbm_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    stock: str,
    trade_date: datetime.date,
    option_type: str,
    # Pass explicit axes for range setting
    k_axis: np.ndarray,
    t_axis: np.ndarray,
    sigma_range: Tuple[float, float],
    show_realized: bool = False,
    realized_points: Optional[Tuple[List[float], List[float], List[float]]] = None, # Expects filtered data
    show_s0: bool = False,
    spot_price: Optional[float] = None, # Expects S0 value
    spot_line_data: Optional[Tuple[List[int], List[float]]] = None # Expects filtered data
) -> go.Figure:
    """
    Compares surfaces using explicitly defined common axes ranges and colors RV points based on proximity.
    """
    fig = go.Figure()
    # Check validity before unpacking
    opt_K, opt_T, opt_Z = (options_surface if isinstance(options_surface, tuple) and len(options_surface)==3 else (None, None, None))
    lgbm_K, lgbm_T, lgbm_Z = (lgbm_surface if isinstance(lgbm_surface, tuple) and len(lgbm_surface)==3 else (None, None, None))


    # Add Surfaces (check validity before adding)
    if opt_Z is not None and np.any(np.isfinite(opt_Z)):
        fig.add_trace(go.Surface(
            x=opt_K, y=opt_T, z=opt_Z, name=f"Options IV ({option_type.capitalize()})",
            colorscale="Viridis", opacity=SURFACE_OPACITY, showscale=True, # Use constant
            colorbar=dict(title='Options IV', thickness=15, len=0.6, y=0.8, x=1.02), # Shifted x slightly
            hovertemplate="<b>Options IV</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>", showlegend=True # Added $
        ))
    if lgbm_Z is not None and np.any(np.isfinite(lgbm_Z)):
        fig.add_trace(go.Surface(
            x=lgbm_K, y=lgbm_T, z=lgbm_Z, name="LGBM RV", # Use simpler name in legend
            colorscale="Plasma", opacity=SHARED_SURFACE_OPACITY, showscale=True, # Use constant
            colorbar=dict(title='LGBM RV', thickness=15, len=0.6, y=0.15, x=1.02), # Shifted x slightly
            hovertemplate="<b>LGBM RV</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>", showlegend=True # Added $
        ))

    # Add Realized Points with conditional coloring
    if show_realized and realized_points and realized_points[0]: # Check list not empty
        # --- Get colors and hovers based on proximity ---
        # Pass the potentially invalid surfaces too, coloring function handles Nones
        rv_colors, rv_hovers = get_rv_point_colors_and_hovers(
            realized_points, lgbm_surface, options_surface, option_type.capitalize()
        )
        fig.add_trace(go.Scatter3d(
            x=realized_points[0], y=realized_points[1], z=realized_points[2], mode="markers",
            marker=dict(color=rv_colors, size=6, symbol='diamond', line=dict(color='black', width=1)), # Use calculated colors
            name="Realized Points", showlegend=True,
            hovertemplate=rv_hovers # Use calculated hovers (already includes $)
        ))

    # Add S0 Line (using filtered data passed in)
    if show_s0 and spot_price is not None and np.isfinite(spot_price) and spot_line_data:
        s0_T_sorted, s0_sigma_sorted = spot_line_data
        fig.add_trace(go.Scatter3d(
            x=[spot_price] * len(s0_T_sorted), y=s0_T_sorted, z=s0_sigma_sorted, mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=4, line=dict(color='black', width=1)),
            line=dict(color=S0_LINE_COLOR, width=5),
            name=f'Hist. RV @ Spot K=${spot_price:.2f}', showlegend=True, # Added $
            hovertemplate="<b>Hist RV@Spot</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>" # Added $
        ))

    # Update layout using the min/max of the explicitly passed axes
    k_min, k_max = k_axis.min(), k_axis.max()
    t_min, t_max = t_axis.min(), t_axis.max()
    sigma_min, sigma_max = sigma_range

    fig.update_layout(
        title=f"{stock} Comparison: Options IV vs LGBM RV @ {trade_date}", # Use full names
        scene=dict(
            xaxis_title="Strike Price (K, $)", # Added $
            yaxis_title="Time to Maturity (Days, T)",
            zaxis_title="Volatility (σ)",
            xaxis_range=[k_min, k_max],
            yaxis_range=[t_min, t_max],
            zaxis_range=[sigma_min, sigma_max],
            aspectratio=dict(x=1.5, y=1.5, z=1)
        ),
        margin=dict(l=0, r=40, b=0, t=40), # Adjusted margins slightly for colorbars
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.8, y=-1.8, z=0.9))
    )
    return fig


# --- Options Axis Calculation ---
def calculate_options_axis_ranges(
    options_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    # option_chain_df should be ADJUSTED data for calculating display range
    option_chain_df: Optional[pl.DataFrame] = None,
    stock: Optional[str] = None,
    date: Optional[datetime.date] = None
) -> Optional[Dict[str, float]]:
    """
    Calculate axis ranges for options chain display:
    K based on ACTUAL *adjusted* strikes from option_chain_df,
    Sigma based on the generated SURFACE IV.
    Returns None if essential data is missing.
    """
    # Validate options_surface structure
    if not isinstance(options_surface, tuple) or len(options_surface) != 3:
        logger.warning("Cannot calculate options axis ranges: Invalid options_surface structure.")
        return None
    opt_K_mesh, opt_T_mesh, opt_Z = options_surface
    if opt_K_mesh is None or opt_T_mesh is None or opt_Z is None:
        logger.warning("Cannot calculate options axis ranges: Surface contains None.")
        return None

    # 1. Determine K range based on actual *adjusted* strikes in the option chain df for that date
    actual_k_min, actual_k_max = None, None
    if option_chain_df is not None and not option_chain_df.is_empty() and stock is not None and date is not None:
        # Use the passed df (should be adjusted) to get strike range
        strike_range = get_options_strike_range(option_chain_df, stock, date)
        if strike_range:
            raw_k_min, raw_k_max = strike_range
            # Check range is valid before using
            if np.isfinite(raw_k_min) and np.isfinite(raw_k_max) and raw_k_max > raw_k_min:
                actual_k_min = raw_k_min
                actual_k_max = raw_k_max
                logger.info(f"Using actual adjusted strike range from options chain: ${actual_k_min:.2f} to ${actual_k_max:.2f}")
            else:
                 logger.warning(f"Invalid strike range ({raw_k_min}, {raw_k_max}) from adjusted options data for {stock} on {date}. Cannot determine K axis.")
                 return None
        else:
            logger.warning(f"No adjusted options data found for {stock} on {date} to determine strike range. Cannot determine K axis.")
            return None
    else:
         logger.warning("Missing data (adjusted chain_df, stock, or date) to determine options K axis.")
         return None

    # 2. Determine Sigma range based ONLY on the generated Surface IV values
    if opt_Z is not None and opt_Z.size > 0:
        valid_Z = opt_Z[np.isfinite(opt_Z)] # Filter non-finite values
        if valid_Z.size > 0:
            sigma_min_calc = np.min(valid_Z)
            sigma_max_calc = np.max(valid_Z)
            sigma_range_val = sigma_max_calc - sigma_min_calc
            # Apply padding, ensuring it's not excessively large for small ranges
            sigma_padding = max(min(sigma_range_val * AXIS_PADDING_FACTOR / 2.0, 0.1), 0.01)
            opt_sigma_min = max(0.0, sigma_min_calc - sigma_padding)
            opt_sigma_max = sigma_max_calc + sigma_padding
        else: # Surface Z exists but contains only NaNs/infs
             logger.warning("Options surface Z contains only non-finite values, using default sigma range.")
             opt_sigma_min, opt_sigma_max = 0.0, 1.0
    else: # Fallback if surface Z is None or empty
        logger.warning("Options surface Z is invalid or empty, using default sigma range.")
        opt_sigma_min, opt_sigma_max = 0.0, 1.0

    # 3. Get T range from the mesh
    t_min = np.nanmin(opt_T_mesh) if np.any(np.isfinite(opt_T_mesh)) else 0
    t_max = np.nanmax(opt_T_mesh) if np.any(np.isfinite(opt_T_mesh)) else 365

    # Use actual K range directly (padding often not needed/desired for options)
    opt_k_min = actual_k_min
    opt_k_max = actual_k_max

    # Ensure max > min (already checked for K)
    if opt_sigma_max <= opt_sigma_min: opt_sigma_max = opt_sigma_min + 0.1

    return {
        "k_min": opt_k_min, "k_max": opt_k_max,
        "sigma_min": opt_sigma_min, "sigma_max": opt_sigma_max,
        "t_min": t_min, "t_max": t_max
    }


# --- Helper to get options strike range ---
def get_options_strike_range(option_chain_df, stock, date):
    """
    Get actual min/max strike from the provided options chain dataframe
    for a specific stock and date. Assumes the df contains the relevant strikes
    (e.g., pass adjusted df if calculating range for adjusted surface display).
    """
    if option_chain_df is None or option_chain_df.is_empty(): return None
    try:
        # Filter for the specific stock and date
        # Ensure date comparison works correctly (cast df column if needed)
        # Ensure date is datetime.date object
        target_date = _to_date(date)
        if target_date is None:
             logger.error(f"Invalid date format provided to get_options_strike_range: {date}")
             return None

        filtered_df = option_chain_df.filter(
            (pl.col("act_symbol") == stock) &
            (pl.col("date").cast(pl.Date) == target_date)
        )
        if filtered_df.is_empty():
             # Log a warning instead of error, as data might just not exist for that date
             logger.warning(f"No options data found for {stock} on {target_date} in get_options_strike_range using provided df.")
             return None
        # Calculate min/max strike from the filtered data
        min_strike = filtered_df["strike"].min()
        max_strike = filtered_df["strike"].max()
        # Add check for valid range (non-null, finite, max > min)
        if min_strike is None or max_strike is None or not np.isfinite(min_strike) or not np.isfinite(max_strike) or max_strike <= min_strike:
            logger.warning(f"Invalid strike range calculated for {stock} on {target_date}: min={min_strike}, max={max_strike}")
            return None
        return (min_strike, max_strike)
    except pl.ColumnNotFoundError:
        logger.error(f"Missing required columns ('act_symbol', 'date', 'strike') in option_chain_df for strike range calculation.")
        return None
    except Exception as e:
        # Log error with more context
        logger.error(f"Error getting options strike range for {stock} on {date}: {e}", exc_info=True)
        return None


# --- [REVISED] Function calculating pricing difference on EXPLICIT common grid axes ---
def calculate_pricing_difference_on_axes(
    options_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    lgbm_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    target_k_axis: np.ndarray, # Explicit 1D K axis for common grid
    target_t_axis: np.ndarray, # Explicit 1D T axis for common grid
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], # diff_Z, K_mesh, T_mesh
           Optional[np.ndarray], Optional[np.ndarray], # opt_Z_common, lgbm_Z_common
           Dict[str, Any]]: # stats
    """
    Calculates the percentage difference (Options IV - LGBM RV) / LGBM RV * 100
    by interpolating BOTH surfaces onto a common grid defined by the explicit
    target_k_axis and target_t_axis.

    Args:
        options_surface: Tuple (K_mesh, T_mesh, Z_values) for Options IV.
        lgbm_surface: Tuple (K_mesh, T_mesh, Z_values) for LGBM RV.
        target_k_axis: 1D numpy array defining the K points for the common grid.
        target_t_axis: 1D numpy array defining the T points for the common grid.

    Returns:
        Tuple of (
            diff_Z: Matrix of percentage differences on the common grid (or None if failed),
            target_K_mesh: The K mesh of the common grid,
            target_T_mesh: The T mesh of the common grid,
            opt_Z_common: Options IV values interpolated onto common grid (or None),
            lgbm_Z_common: LGBM RV values interpolated onto common grid (or None),
            statistics_dict: Calculated metrics (max_diff, min_diff, etc.) on the common grid.
        )
    """
    # Initialize default stats and return structure
    stats = {
        "max_diff": np.nan, "min_diff": np.nan, "mean_diff": np.nan,
        "max_diff_k": np.nan, "max_diff_t": np.nan,
        "min_diff_k": np.nan, "min_diff_t": np.nan,
    }
    default_return = (None, None, None, None, None, stats)

    # --- Input Validation ---
    if not isinstance(options_surface, tuple) or len(options_surface) != 3 or \
       not all(isinstance(arr, np.ndarray) and arr.size > 0 for arr in options_surface):
        logger.error("Invalid or empty options_surface input for difference calculation.")
        return default_return
    opt_K_mesh, opt_T_mesh, opt_Z = options_surface

    if not isinstance(lgbm_surface, tuple) or len(lgbm_surface) != 3 or \
       not all(isinstance(arr, np.ndarray) and arr.size > 0 for arr in lgbm_surface):
        logger.error("Invalid or empty lgbm_surface input for difference calculation.")
        return default_return
    lgbm_K_mesh, lgbm_T_mesh, lgbm_Z = lgbm_surface

    if opt_K_mesh.shape != opt_T_mesh.shape or opt_K_mesh.shape != opt_Z.shape:
        logger.error("Mismatched shapes within options_surface.")
        return default_return
    if lgbm_K_mesh.shape != lgbm_T_mesh.shape or lgbm_K_mesh.shape != lgbm_Z.shape:
        logger.error("Mismatched shapes within lgbm_surface.")
        return default_return

    if target_k_axis is None or target_t_axis is None or target_k_axis.ndim != 1 or target_t_axis.ndim != 1 or \
       target_k_axis.size == 0 or target_t_axis.size == 0:
        logger.error("Invalid target_k_axis or target_t_axis provided.")
        return default_return

    # --- Create Common Target Grid from Axes ---
    target_K_mesh, target_T_mesh = np.meshgrid(target_k_axis, target_t_axis)
    target_points_k_flat = target_K_mesh.flatten()
    target_points_t_flat = target_T_mesh.flatten()
    logger.info(f"Created common target grid with shape {target_K_mesh.shape}")

    # --- Interpolate BOTH Surfaces onto Common Grid ---
    logger.info(f"Interpolating Options surface onto common {target_K_mesh.shape} grid...")
    opt_Z_interp_flat = interpolate_surface_at_points(
        opt_K_mesh, opt_T_mesh, opt_Z,
        target_points_k_flat, target_points_t_flat,
        method='linear'
    )
    if opt_Z_interp_flat is None:
        logger.error("Options surface interpolation failed.")
        return default_return
    opt_Z_common = opt_Z_interp_flat.reshape(target_K_mesh.shape)
    logger.info("Options interpolation complete.")

    logger.info(f"Interpolating LGBM surface onto common {target_K_mesh.shape} grid...")
    lgbm_Z_interp_flat = interpolate_surface_at_points(
        lgbm_K_mesh, lgbm_T_mesh, lgbm_Z,
        target_points_k_flat, target_points_t_flat,
        method='linear'
    )
    if lgbm_Z_interp_flat is None:
        logger.error("LGBM surface interpolation failed.")
        return default_return # Return default with None for Z values
    lgbm_Z_common = lgbm_Z_interp_flat.reshape(target_K_mesh.shape)
    logger.info("LGBM interpolation complete.")


    # --- Calculate Percentage Difference ---
    with np.errstate(divide='ignore', invalid='ignore'):
        diff_Z = (opt_Z_common - lgbm_Z_common) / lgbm_Z_common * 100
        # Replace inf/-inf (division by zero) and NaN (0/0 or nan involved) with NaN
        diff_Z[~np.isfinite(diff_Z)] = np.nan

    # --- Calculate Statistics on the Common Grid ---
    valid_diff_mask = np.isfinite(diff_Z)
    if np.any(valid_diff_mask):
        valid_diffs = diff_Z[valid_diff_mask]
        stats["max_diff"] = np.nanmax(valid_diffs)
        stats["min_diff"] = np.nanmin(valid_diffs)
        stats["mean_diff"] = np.nanmean(valid_diffs)

        # Find indices of max/min on the common grid
        if np.isfinite(stats["max_diff"]):
             max_indices = np.where(np.isclose(diff_Z, stats["max_diff"]) & valid_diff_mask)
             if len(max_indices[0]) > 0:
                 idx = (max_indices[0][0], max_indices[1][0]) # row, col index
                 stats["max_diff_t"] = target_T_mesh[idx]
                 stats["max_diff_k"] = target_K_mesh[idx]

        if np.isfinite(stats["min_diff"]):
            min_indices = np.where(np.isclose(diff_Z, stats["min_diff"]) & valid_diff_mask)
            if len(min_indices[0]) > 0:
                idx = (min_indices[0][0], min_indices[1][0]) # row, col index
                stats["min_diff_t"] = target_T_mesh[idx]
                stats["min_diff_k"] = target_K_mesh[idx]
        logger.info(f"Difference stats calculated over {np.sum(valid_diff_mask)} valid points on common grid.")
    else:
        logger.warning("No valid difference points found after interpolating both surfaces onto common grid.")
        # Keep default NaN stats

    # Return the difference matrix, the common grid meshes, the interpolated Z values, and stats
    return diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats


# --- Heatmap Creation function ---
def create_pricing_difference_plot(
    diff_Z: np.ndarray,
    K_mesh: np.ndarray,  # K mesh corresponding to diff_Z rows/cols (common grid K)
    T_mesh: np.ndarray,  # T mesh corresponding to diff_Z rows/cols (common grid T)
    opt_Z_common: np.ndarray, # Options IV values on common grid
    lgbm_Z_common: np.ndarray, # LGBM RV values on common grid
    stats: Dict[str, Any], # Statistics dictionary, including locations of min/max
    stock: str,
    trade_date: datetime.date,
    option_type: str # Added for context in hover
) -> go.Figure:
    """
    Creates a heatmap of pricing differences using the calculated difference matrix (diff_Z)
    and the corresponding common grid K_mesh and T_mesh. Includes detailed hover info.
    """
    fig = go.Figure()

    # Check if diff_Z and meshes are valid before plotting heatmap
    if diff_Z is None or K_mesh is None or T_mesh is None or \
       opt_Z_common is None or lgbm_Z_common is None or \
       diff_Z.size == 0 or K_mesh.size == 0 or T_mesh.size == 0 or \
       diff_Z.shape != K_mesh.shape or diff_Z.shape != T_mesh.shape or \
       diff_Z.shape != opt_Z_common.shape or diff_Z.shape != lgbm_Z_common.shape:
        logger.warning("Cannot create heatmap: Invalid input diff_Z, K/T mesh, or common Z values, or mismatched shapes.")
        # Return an empty figure with a title indicating failure
        fig.update_layout(title=f"Pricing Difference Heatmap Generation Failed for {stock} @ {trade_date}")
        return fig

    # Use standard go.Heatmap for better hover support
    # Stack all necessary data for the hover template
    customdata = np.dstack((K_mesh, T_mesh, opt_Z_common, lgbm_Z_common))

    fig.add_trace(go.Heatmap(
            z=diff_Z,
            x=K_mesh[0, :], # K labels for x-axis (from common grid)
            y=T_mesh[:, 0], # T labels for y-axis (from common grid)
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="OptionsIV-LGBMRV<br>Diff (%)", thickness=15), # Updated label
            customdata=customdata, # Provide K, T, OptZ, LgbmZ
            hovertemplate=( "<b>Difference Details</b><br>"
                            "K: $%{customdata[0]:.2f}<br>" # Added $
                            "T: %{customdata[1]:.0f}d<br>"
                            "Options IV: %{customdata[2]:.4f}<br>"
                            "LGBM RV: %{customdata[3]:.4f}<br>"
                            "Difference: %{z:.2f}%<extra></extra>"),
            hoverongaps=False
     ))

    # Add markers for Max Overpricing (Positive Diff) and Max Underpricing (Negative Diff)
    # Check if the stats and coordinates are valid numbers
    if stats and np.isfinite(stats.get('max_diff_k', np.nan)) and np.isfinite(stats.get('max_diff_t', np.nan)):
        max_diff_val = stats.get('max_diff', np.nan)
        fig.add_trace(go.Scatter(
            x=[stats['max_diff_k']],
            y=[stats['max_diff_t']],
            mode="markers",
            marker=dict(symbol="circle", size=10, color="rgba(255,0,0,0.9)", line=dict(width=1, color="black")),
            name=f"Max Over (+{max_diff_val:.1f}%)" if np.isfinite(max_diff_val) else "Max Over (N/A)",
            hovertemplate=(f"<b>Max Over (Opt > LGBM)</b><br>" # Updated label
                           f"K: $%{{x:.2f}}, T: %{{y:.0f}}d<br>" # Added $
                           f"Diff: {max_diff_val:.2f}%<extra></extra>") if np.isfinite(max_diff_val) else "Max Overpricing (N/A)"
        ))

    if stats and np.isfinite(stats.get('min_diff_k', np.nan)) and np.isfinite(stats.get('min_diff_t', np.nan)):
        min_diff_val = stats.get('min_diff', np.nan)
        fig.add_trace(go.Scatter(
            x=[stats['min_diff_k']],
            y=[stats['min_diff_t']],
            mode="markers",
            marker=dict(symbol="circle", size=10, color="rgba(0,0,255,0.9)", line=dict(width=1, color="black")),
            name=f"Max Under ({min_diff_val:.1f}%)" if np.isfinite(min_diff_val) else "Max Under (N/A)",
            hovertemplate=(f"<b>Max Under (Opt < LGBM)</b><br>" # Updated label
                           f"K: $%{{x:.2f}}, T: %{{y:.0f}}d<br>" # Added $
                           f"Diff: {min_diff_val:.2f}%<extra></extra>") if np.isfinite(min_diff_val) else "Max Underpricing (N/A)"
        ))

    fig.update_layout(
        title=f"{stock} Options IV vs LGBM RV Difference (%) @ {trade_date}", # Use full names
        xaxis_title="Strike Price (K, $)", # Added $
        yaxis_title="Days to Expiration (T)",
        yaxis_autorange='reversed', # Optional: Put shorter expiries at the top
        width=None, # Let streamlit manage width
        height=600,
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
    )
    # Set axis range based on the mesh extents used for labels (common grid)
    # Ensure axes exist before calling min/max
    if K_mesh.size > 0:
       fig.update_xaxes(range=[np.nanmin(K_mesh[0, :]), np.nanmax(K_mesh[0, :])])
    if T_mesh.size > 0:
       fig.update_yaxes(range=[np.nanmin(T_mesh[:, 0]), np.nanmax(T_mesh[:, 0])])


    return fig


# -----------------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------------
def main():
    # Tabs and Title
    viz_tab, about_tab = st.tabs(["Visualizer", "About"])
    with about_tab: st.markdown(ABOUT_TEXT)
    with viz_tab:
        st.title("Volatility Surface Visualizer")
        st.markdown("Compare LGBM-modeled Realized Volatility (RV) and Options Implied Volatility (IV) surfaces.")
        st.caption("Note: All strike prices (K) are split-adjusted.") # Added split note

        # --- Sidebar Inputs ---
        st.sidebar.header("Surface Parameters")
        # Default list, attempt to load from file
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSLA"]
        try:
            # Construct path relative to the script location
            symbols_path = os.path.join(project_root, 'data', 'processed', 'symbols.txt')
            if os.path.exists(symbols_path):
                with open(symbols_path, 'r') as f:
                    symbols_from_file = sorted([s.strip().upper() for s in f if s.strip()])
                    if symbols_from_file: # Use file contents only if not empty
                         symbols = symbols_from_file
                    else:
                         st.sidebar.warning("symbols.txt is empty, using default list.")
            else:
                 st.sidebar.warning(f"symbols.txt not found at {symbols_path}, using default list.")
        except Exception as e:
            st.sidebar.error(f"Error loading symbols: {e}")
        # Select stock, handle case where default AAPL might not be in the list
        default_symbol = "AAPL"
        if default_symbol not in symbols and symbols: default_symbol = symbols[0]
        elif not symbols: default_symbol = "AAPL"; symbols=["AAPL"] # Fallback if loading failed completely
        stock = st.sidebar.selectbox("Stock Symbol", symbols, index=symbols.index(default_symbol))

        # Date input with a reasonable default
        default_date = datetime.date(2021, 1, 25)
        trade_date = st.sidebar.date_input("Trade Date", default_date)

        st.sidebar.header("Visualization Settings")
        vis_mode = st.sidebar.radio("Select View", ["LGBM Model RV", "Options Chain IV", "Compare Surfaces"], index=2, key="vis_mode_radio", horizontal=True)

        # Option type only relevant if showing Options IV
        option_type = "Average" # Default
        if vis_mode != "LGBM Model RV":
            option_type = st.sidebar.radio("Option Type (for IV)", ["Call", "Put", "Average"], index=2, horizontal=True) # Default to Average

        with st.sidebar.expander("Display Options", expanded=True):
            show_realized_points = st.checkbox("Show Realized Points", value=True, help="Show actual future realized volatility points (if available).")
            show_s0_line = st.checkbox("Show Historical S0 Line", value=True, help="Show historical realized volatility at the current spot price (S0).")
            # Comparison-specific options
            show_diff_heatmap = False
            show_stats = False
            show_rv_comparison = False
            if vis_mode == "Compare Surfaces":
                show_diff_heatmap = st.checkbox("Show Options IV vs LGBM RV Heatmap", value=True, help="Show a 2D heatmap of the percentage difference between Options IV and LGBM RV.") # Updated label
                show_stats = st.checkbox("Show Options IV vs LGBM RV Stats", value=True, help="Display summary statistics of the difference between the surfaces.") # Updated label
                show_rv_comparison = st.checkbox("Show Surface vs Realized", value=True, help="Compare how close each surface is to the realized volatility points.")

        generate = st.sidebar.button("Generate Plot(s)", key="generate_button", type="primary", use_container_width=True)

        # --- Main Content Area ---
        if generate:
            # Initialize state variables for this run
            ohlcv_df, option_chain_raw, option_chain_adj = None, None, None # Keep raw and adjusted options separate
            transformed_df = None
            nearest_ohlcv_date, nearest_options_date = None, None
            lgbm_surface, options_surface = None, None # Store as (K, T, Z) tuples
            lgbm_axis_ranges, options_axis_ranges = None, None # Store axis dicts
            lgbm_models_dict, lgbm_today_data = None, None
            error_occurred = False
            loading_placeholder = st.empty()

            try:
                # --- Step 1 & 2: Load Data & Models ---
                with loading_placeholder.status(f"Processing {stock} near {trade_date}...", expanded=True):
                    st.write("Loading OHLCV Data...")
                    try:
                        ohlcv_df = load_data("ohlcv", stock)
                        if ohlcv_df is None or ohlcv_df.is_empty(): raise ValueError("OHLCV data is empty or failed to load.")
                        # Find nearest date *after* loading successfully
                        nearest_ohlcv_date = find_nearest_date(ohlcv_df, trade_date, stock)
                        if nearest_ohlcv_date is None: raise ValueError(f"No suitable OHLCV date found near {trade_date}.")
                        st.write(f"Found OHLCV data for date: {nearest_ohlcv_date}")
                    except Exception as e:
                        st.error(f"Failed to load or find OHLCV data: {e}")
                        error_occurred = True # Stop processing if essential OHLCV fails

                    # Load options only if needed and OHLCV succeeded
                    if not error_occurred and vis_mode != "LGBM Model RV":
                        st.write("Loading Options Chain Data...")
                        try:
                            option_chain_raw = load_data("option_chain", stock)
                            if option_chain_raw is None or option_chain_raw.is_empty(): raise ValueError("Options chain data is empty or failed to load.")
                            # Find nearest date in raw data first
                            nearest_options_date = find_nearest_date(option_chain_raw, trade_date, stock)
                            if nearest_options_date is None: raise ValueError(f"No suitable options date found near {trade_date}.")
                            st.write(f"Found Options data for date: {nearest_options_date}")
                            # Adjust splits *after* confirming data exists for the nearest date
                            st.write("Adjusting Options for Splits...")
                            option_chain_adj = adjust_option_splits(option_chain_raw)
                            if option_chain_adj is None or option_chain_adj.is_empty(): raise ValueError("Options data became empty after split adjustment.")
                        except Exception as e:
                            st.warning(f"Failed to load or process Options data: {e}. Options surface may not be generated.")
                            # Don't set error_occurred = True, allow potential LGBM-only view

                    # Load LGBM models only if needed and OHLCV succeeded
                    if not error_occurred and vis_mode != "Options Chain IV":
                         st.write("Loading LGBM Models...")
                         try:
                            lgbm_models_dict = load_surface_models()
                            if not lgbm_models_dict or 'models' not in lgbm_models_dict or not lgbm_models_dict['models']:
                                raise ValueError("LGBM models dictionary is invalid or empty.")
                            st.write("LGBM Models loaded.")
                         except Exception as e:
                             st.warning(f"Failed to load LGBM Models: {e}. LGBM surface cannot be generated.")
                             lgbm_models_dict = None # Ensure it's None if loading failed

                    # Transform OHLCV data if loaded successfully and needed for LGBM
                    # Or if needed for S0 line / realized points overlays
                    if not error_occurred and ohlcv_df is not None and (vis_mode != "Options Chain IV" or show_s0_line or show_realized_points):
                        st.write("Transforming OHLCV Data...")
                        try:
                            transformed_df = transformation_pipeline(ohlcv_df)
                            if transformed_df is None or transformed_df.is_empty(): raise ValueError("Data transformation returned empty DataFrame.")
                            # Get data for the specific date needed for LGBM surface / overlays
                            lgbm_today_data = transformed_df.filter(
                                (pl.col("act_symbol") == stock) &
                                (pl.col("date").cast(pl.Date) == nearest_ohlcv_date) # Use found OHLCV date
                            ).head(1) # Ensure only one row
                            if lgbm_today_data.is_empty():
                                logger.warning(f"No transformed data found for {stock} on {nearest_ohlcv_date}.")
                                lgbm_today_data = None # Set to None if no data for the specific date
                            st.write("Data transformation complete.")
                        except Exception as e:
                            st.error(f"Data transformation failed: {e}")
                            error_occurred = True # Stop if transformation fails

                loading_placeholder.empty() # Clear status

                # Re-check dates after loading attempts
                if not error_occurred:
                    date_msgs = []
                    if nearest_ohlcv_date: date_msgs.append(f"OHLCV: `{nearest_ohlcv_date}`")
                    if nearest_options_date: date_msgs.append(f"Options: `{nearest_options_date}`")
                    if date_msgs: st.info(f"Using nearest available data dates: {', '.join(date_msgs)}")
                    elif vis_mode == "LGBM Model RV": # If only LGBM was needed but failed date find
                         st.error("Could not find a valid OHLCV date.")
                         error_occurred = True
                    # Allow continuing if options date failed but LGBM might work

                # --- Step 3: Generate Surfaces & Calculate Axes ---
                if not error_occurred:
                    generation_placeholder = st.empty()
                    with generation_placeholder.status(f"Generating surfaces & calculating axes...", expanded=True):

                        # Generate LGBM Surface & Axes (if model loaded and data available)
                        if vis_mode != "Options Chain IV" and lgbm_models_dict and transformed_df is not None and nearest_ohlcv_date:
                             st.write("Generating LGBM RV Surface & Axes...") # Updated Name
                             try:
                                 # Generate surface requires the transformed data, not just today's row
                                 result = generate_vol_surface(
                                     lgbm_models_dict, transformed_df, stock, nearest_ohlcv_date,
                                     K_RANGE, K_GRID_POINTS, False # Use False for include_history
                                 )
                                 if result and result[2] is not None and np.any(np.isfinite(result[2])): # Check if Z is valid
                                     lgbm_surface = (result[0], result[1], result[2]) # K, T, Z
                                     # --- Calculate LGBM specific axes using the helper ---
                                     # Pass today's data row for overlays
                                     _, lgbm_axis_ranges = create_plotly_figure(
                                          *lgbm_surface, lgbm_today_data, stock, nearest_ohlcv_date,
                                          lgbm_models_dict.get("horizons", HORIZONS),
                                          show_realized=show_realized_points, show_s0=show_s0_line
                                     )
                                     st.write("LGBM RV Surface generated.") # Updated Name
                                 else:
                                      st.warning("LGBM RV surface generation failed or resulted in invalid data.") # Updated Name
                                      lgbm_surface = None # Ensure it's None
                                      lgbm_axis_ranges = None
                             except Exception as e:
                                 st.warning(f"Error generating LGBM RV surface/axes: {e}") # Updated Name
                                 lgbm_surface = None # Ensure it's None on error
                                 lgbm_axis_ranges = None

                        # Generate Options Surface & Axes (if data loaded and date found)
                        # Apply user's fix here for raw vs adjusted data usage
                        if vis_mode != "LGBM Model RV" and option_chain_raw is not None and option_chain_adj is not None and nearest_options_date:
                            st.write("Generating Options IV Surface & Axes...") # Updated Name
                            try:
                                # Use the raw options data for surface generation (assuming it handles splits)
                                options_surface_result = generate_options_surface(
                                    option_chain_raw, stock, nearest_options_date,
                                    option_type.lower(), K_RANGE, K_GRID_POINTS
                                )
                                if options_surface_result and options_surface_result[2] is not None and np.any(np.isfinite(options_surface_result[2])):
                                    options_surface = options_surface_result # Store (K, T, Z)
                                    # --- Calculate Options specific axes using the helper ---
                                    # Pass the adj options data for accurate strike range calculation for display
                                    options_axis_ranges = calculate_options_axis_ranges(
                                        options_surface, option_chain_adj, stock, nearest_options_date
                                    )
                                    if options_axis_ranges is None: # Handle failure in axis calc
                                        st.warning("Failed to calculate Options IV axis ranges (check options data for the date). Options IV surface may not display correctly.") # Use full name
                                        options_surface = None # Prevent plotting if axes failed
                                    else:
                                         st.write("Options IV Surface generated.") # Use full name
                                else:
                                    st.warning("Options IV surface generation failed or resulted in invalid data.") # Use full name
                                    options_surface = None # Ensure it's None
                                    options_axis_ranges = None
                            except Exception as e:
                                st.warning(f"Error generating Options IV surface/axes: {e}") # Updated Name
                                options_surface = None # Ensure it's None on error
                                options_axis_ranges = None

                    generation_placeholder.empty()

                # --- Step 4: Plotting and Comparisons ---
                if error_occurred: st.error("Processing stopped due to earlier errors. Cannot generate plots.")
                else:
                    # Check if *any* surface was generated successfully
                    if not lgbm_surface and not options_surface:
                         st.error(f"Could not generate any volatility surfaces for {stock} on {trade_date}. Check data availability and logs.")
                         # Exit early if no surfaces generated
                         return # Stops further execution in this Streamlit run


                    # Extract overlay data (might be None if LGBM failed or overlays disabled)
                    # Use lgbm_axis_ranges if available, otherwise try to get S0 separately
                    spot_price = np.nan
                    realized_points_data = None
                    spot_line_data = None
                    if lgbm_axis_ranges:
                        realized_points_data = lgbm_axis_ranges.get("realized_points")
                        spot_price = lgbm_axis_ranges.get("S0")
                        spot_line_data = lgbm_axis_ranges.get("spot_line_data")
                    elif lgbm_today_data is not None and not lgbm_today_data.is_empty(): # Try to get S0 even if axis calc failed
                        try:
                           spot_price = lgbm_today_data.row(0, named=True).get("close", np.nan)
                        except IndexError: pass # Ignore if row doesn't exist

                    # Determine if realized points are actually available AND requested
                    realized_available = show_realized_points and (realized_points_data is not None) and (len(realized_points_data[0]) > 0)

                    # --- LGBM View ---
                    if vis_mode == "LGBM Model RV":
                        if lgbm_surface and lgbm_axis_ranges:
                            st.subheader(f"LGBM Predicted RV Surface ({stock} @ {nearest_ohlcv_date})") # Updated Title
                            # Re-create plot using the calculated axes
                            fig_lgbm, _ = create_plotly_figure(
                                *lgbm_surface, lgbm_today_data, stock, nearest_ohlcv_date,
                                lgbm_models_dict.get("horizons", HORIZONS) if lgbm_models_dict else HORIZONS, # Handle None models_dict
                                show_realized=show_realized_points, show_s0=show_s0_line
                            )
                            st.plotly_chart(fig_lgbm, use_container_width=True)
                            # Comparison vs Realized (only if requested and available)
                            if show_rv_comparison:
                                lgbm_diffs = calculate_point_differences("LGBM RV", lgbm_surface, realized_points_data) # Use full name
                                display_point_differences("LGBM RV", lgbm_diffs, realized_available) # Use full name
                        else: st.error("Could not generate LGBM RV surface or determine its axes. Check logs.") # Updated Name

                    # --- Options View ---
                    elif vis_mode == "Options Chain IV":
                        if options_surface and options_axis_ranges: # Check axes are valid too
                            st.subheader(f"Options IV Surface ({option_type}, {stock} @ {nearest_options_date})") # Updated Title
                            fig_opts = plot_options_surface(
                                *options_surface, stock, nearest_options_date, option_type,
                                options_axis_ranges, # Pass calculated specific axes
                                show_realized=show_realized_points, realized_points=realized_points_data,
                                show_s0=show_s0_line, spot_price=spot_price, spot_line_data=spot_line_data
                            )
                            st.plotly_chart(fig_opts, use_container_width=True)
                            # Comparison vs Realized (only if requested and available)
                            if show_rv_comparison:
                                options_diffs = calculate_point_differences(f"Options IV ({option_type})", options_surface, realized_points_data) # Use full name
                                display_point_differences(f"Options IV ({option_type})", options_diffs, realized_available) # Use full name
                        else: st.error("Could not generate Options IV surface or determine its axes. Check options data exists for the selected date and logs.") # Updated Name

                    # --- Comparison View ---
                    elif vis_mode == "Compare Surfaces":
                        if lgbm_surface and options_surface and lgbm_axis_ranges and options_axis_ranges:
                            plot_date_ref = nearest_options_date if nearest_options_date else nearest_ohlcv_date
                            st.subheader(f"Comparison: Options IV vs LGBM RV ({stock} around {plot_date_ref})") # Use full names

                            # --- Define Common Axes for Plotting and Difference Calculation ---
                            # K: Union of the individual ranges
                            combined_k_min = min(lgbm_axis_ranges["k_min"], options_axis_ranges["k_min"])
                            combined_k_max = max(lgbm_axis_ranges["k_max"], options_axis_ranges["k_max"])
                            # Sigma: Union of the individual ranges
                            combined_sigma_min = min(lgbm_axis_ranges["sigma_min"], options_axis_ranges["sigma_min"])
                            combined_sigma_max = max(lgbm_axis_ranges["sigma_max"], options_axis_ranges["sigma_max"])
                             # T: Union of the individual ranges for visualization extent
                            combined_t_min_vis = min(lgbm_axis_ranges["t_min"], options_axis_ranges["t_min"])
                            combined_t_max_vis = max(lgbm_axis_ranges["t_max"], options_axis_ranges["t_max"])

                            # Create the explicit 1D axes arrays using these ranges
                            # Ensure range is valid before using linspace
                            if combined_k_max > combined_k_min:
                                common_k_axis = np.linspace(combined_k_min, combined_k_max, TARGET_GRID_RESOLUTION)
                            else:
                                common_k_axis = np.array([combined_k_min]) # Single point if min=max
                            if combined_t_max_vis > combined_t_min_vis:
                                common_t_axis = np.linspace(combined_t_min_vis, combined_t_max_vis, TARGET_GRID_RESOLUTION)
                            else:
                                common_t_axis = np.array([combined_t_min_vis]) # Single point

                            common_sigma_range = (combined_sigma_min, combined_sigma_max)

                            # --- Plot Combined Surface using Common Axes ---
                            fig_compare = compare_surfaces(
                                options_surface, lgbm_surface, stock, plot_date_ref, option_type,
                                k_axis=common_k_axis, # Pass explicit axes
                                t_axis=common_t_axis,
                                sigma_range=common_sigma_range,
                                show_realized=show_realized_points, realized_points=realized_points_data,
                                show_s0=show_s0_line, spot_price=spot_price, spot_line_data=spot_line_data
                            )
                            st.plotly_chart(fig_compare, use_container_width=True)

                            # --- Pricing Difference Calculation (on common axes) ---
                            # Initialize results
                            diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats = None, None, None, None, None, {}
                            try:
                                st.write(f"Calculating pricing differences on common {len(common_k_axis)}x{len(common_t_axis)} grid...")
                                # Calculate difference on the common grid defined by the axes
                                diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats = calculate_pricing_difference_on_axes(
                                    options_surface, lgbm_surface,
                                    target_k_axis=common_k_axis, # Pass the defined axes
                                    target_t_axis=common_t_axis
                                )
                            except Exception as e:
                                st.error(f"Error calculating pricing differences: {e}")
                                logger.error(f"Pricing difference calculation error: {e}", exc_info=True)


                            # --- Display Comparison Sections ---
                            col1, col2 = st.columns(2)

                            with col1: # Options IV vs LGBM RV Difference (Stats & Heatmap)
                                if show_stats or show_diff_heatmap:
                                    st.markdown("##### Options IV vs LGBM RV Difference (on Common Grid)") # Use full names
                                    # Display Stats if calculated and requested
                                    if show_stats and stats and np.isfinite(stats.get('mean_diff', np.nan)):
                                        stat_cols=st.columns(3)
                                        help_max = f"K=${stats.get('max_diff_k', np.nan):.1f}, T={stats.get('max_diff_t', np.nan):.0f}d" if np.isfinite(stats.get('max_diff_k', np.nan)) else "N/A" # Added $
                                        help_min = f"K=${stats.get('min_diff_k', np.nan):.1f}, T={stats.get('min_diff_t', np.nan):.0f}d" if np.isfinite(stats.get('min_diff_k', np.nan)) else "N/A" # Added $
                                        help_avg = f"Avg(OptIV-LGBMRV)% over {np.sum(np.isfinite(diff_Z))} points on common grid" if diff_Z is not None else "N/A" # Updated help

                                        stat_cols[0].metric("Max Over", f"{stats.get('max_diff', np.nan):.2f}%", help=help_max)
                                        stat_cols[1].metric("Max Under", f"{stats.get('min_diff', np.nan):.2f}%", help=help_min)
                                        stat_cols[2].metric("Avg Diff", f"{stats.get('mean_diff', np.nan):.2f}%", help=help_avg)
                                    elif show_stats:
                                         st.info("Difference statistics could not be calculated (check surface validity and interpolation).")

                                    # Display Heatmap if calculated and requested
                                    # Check results from difference calculation are valid
                                    if show_diff_heatmap and diff_Z is not None and target_K_mesh is not None and target_T_mesh is not None \
                                       and opt_Z_common is not None and lgbm_Z_common is not None and stats:
                                        try:
                                            heatmap_fig = create_pricing_difference_plot(
                                                diff_Z,
                                                target_K_mesh, # Use K mesh from common grid result
                                                target_T_mesh, # Use T mesh from common grid result
                                                opt_Z_common,  # Pass common grid Options IV Z values
                                                lgbm_Z_common, # Pass common grid LGBM RV Z values
                                                stats, stock, plot_date_ref, # Use consistent date ref
                                                option_type # Pass for hover context
                                            )
                                            st.plotly_chart(heatmap_fig, use_container_width=True)
                                        except Exception as e:
                                            st.error(f"Error creating pricing difference heatmap: {e}")
                                            logger.error(f"Heatmap creation error: {e}", exc_info=True)
                                    elif show_diff_heatmap:
                                         st.info("Difference heatmap could not be generated (difference calculation failed or result invalid).")

                            with col2: # Surface vs Realized Comparison
                                if show_rv_comparison:
                                    st.markdown("##### Surface Accuracy vs Realized Volatility")
                                    if realized_available:
                                        # Calculate differences for both surfaces vs realized points
                                        lgbm_diffs = calculate_point_differences("LGBM RV", lgbm_surface, realized_points_data) # Use full name
                                        options_diffs = calculate_point_differences(f"Options IV ({option_type})", options_surface, realized_points_data) # Use full name

                                        # Display comparison metrics if both calculations succeeded
                                        # Check specifically if average_abs_diff is present and finite
                                        lgbm_avg_abs_diff = lgbm_diffs.get('average_abs_diff', np.nan) if lgbm_diffs else np.nan
                                        opts_avg_abs_diff = options_diffs.get('average_abs_diff', np.nan) if options_diffs else np.nan

                                        if np.isfinite(lgbm_avg_abs_diff) and np.isfinite(opts_avg_abs_diff):
                                            st.metric("Avg Abs Diff (LGBM RV vs Real)", f"{lgbm_avg_abs_diff:.4f}") # Use full name
                                            st.metric(f"Avg Abs Diff (Options IV vs Real)", f"{opts_avg_abs_diff:.4f}") # Use full name
                                            if np.isclose(lgbm_avg_abs_diff, opts_avg_abs_diff):
                                                 st.info("Surfaces have similar average difference to realized points.")
                                            elif lgbm_avg_abs_diff < opts_avg_abs_diff:
                                                 st.success("LGBM RV surface is closer to realized points on average.") # Use full name
                                            else: # opts_avg_abs_diff < lgbm_avg_abs_diff
                                                 st.success("Options IV surface is closer to realized points on average.") # Use full name
                                        elif np.isfinite(lgbm_avg_abs_diff):
                                             st.metric("Avg Abs Diff (LGBM RV vs Real)", f"{lgbm_avg_abs_diff:.4f}") # Use full name
                                             st.metric(f"Avg Abs Diff (Options IV vs Real)", "N/A") # Use full name
                                             st.info("Only LGBM RV vs Realized difference could be calculated.") # Use full name
                                        elif np.isfinite(opts_avg_abs_diff):
                                             st.metric("Avg Abs Diff (LGBM RV vs Real)", "N/A") # Use full name
                                             st.metric(f"Avg Abs Diff (Options IV vs Real)", f"{opts_avg_abs_diff:.4f}") # Use full name
                                             st.info("Only Options IV vs Realized difference could be calculated.") # Use full name
                                        else:
                                             st.metric("Avg Abs Diff (LGBM RV vs Real)", "N/A") # Use full name
                                             st.metric(f"Avg Abs Diff (Options IV vs Real)", "N/A") # Use full name
                                             st.warning("Could not calculate average absolute differences vs realized points.")

                                        # Display detailed breakdowns in expanders (even if averages are NaN)
                                        if lgbm_diffs: display_point_differences("LGBM RV", lgbm_diffs, True) # Use full name
                                        if options_diffs: display_point_differences(f"Options IV ({option_type})", options_diffs, True) # Use full name

                                    else: # Realized points not available/shown
                                        st.info("Realized points comparison requires 'Show Realized Points' to be checked and data to be available.")

                        # Handle cases where only one surface was successful in Comparison mode
                        elif lgbm_surface and lgbm_axis_ranges: # Only LGBM available
                            st.warning("Options IV surface failed to generate. Showing only LGBM RV.") # Use full name
                            st.subheader(f"LGBM Predicted RV Surface ({stock} @ {nearest_ohlcv_date})") # Use full name
                            fig_lgbm, _ = create_plotly_figure(*lgbm_surface, lgbm_today_data, stock, nearest_ohlcv_date, lgbm_models_dict.get("horizons", HORIZONS) if lgbm_models_dict else HORIZONS, show_realized_points, show_s0_line)
                            st.plotly_chart(fig_lgbm, use_container_width=True)
                            if show_rv_comparison:
                                lgbm_diffs = calculate_point_differences("LGBM RV", lgbm_surface, realized_points_data) # Use full name
                                display_point_differences("LGBM RV", lgbm_diffs, realized_available) # Use full name
                        elif options_surface and options_axis_ranges: # Only Options available
                            st.warning("LGBM RV surface failed to generate. Showing only Options IV.") # Use full name
                            st.subheader(f"Options IV Surface ({option_type}, {stock} @ {nearest_options_date})") # Use full name
                            fig_opts = plot_options_surface(*options_surface, stock, nearest_options_date, option_type, options_axis_ranges, show_realized_points, realized_points_data, show_s0_line, spot_price, spot_line_data)
                            st.plotly_chart(fig_opts, use_container_width=True)
                            if show_rv_comparison:
                                options_diffs = calculate_point_differences(f"Options IV ({option_type})", options_surface, realized_points_data) # Use full name
                                display_point_differences(f"Options IV ({option_type})", options_diffs, realized_available) # Use full name
                        else: # Neither generated successfully
                            st.error("Neither LGBM RV nor Options IV surface could be generated successfully for comparison.") # Use full name

            # --- General Error Handling ---
            except FileNotFoundError as fnf:
                st.error(f"File Not Found Error: {fnf}. Please check data paths.")
                logger.error(f"FNF Error: {fnf}", exc_info=True)
            except ImportError as ime:
                st.error(f"Import Error: {ime}. Check project structure and dependencies.")
                logger.error(f"Import Error: {ime}", exc_info=True)
            except ValueError as ve:
                # Catch specific value errors if needed, otherwise generic message
                st.error(f"Data Error: {ve}. Check input data validity or selection.")
                logger.error(f"Value Error: {ve}", exc_info=True)
            except MemoryError:
                st.error("Memory Error: The application ran out of memory. Try reducing the date range or complexity.")
                logger.error("Memory Error occurred", exc_info=True)
            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")
                logger.error(f"Unexpected Error: {e}", exc_info=True)
        else:
            st.info("Select parameters in the sidebar and click 'Generate Plot(s)' to visualize the volatility surfaces.")

if __name__ == "__main__":
    # Set logger level (optional, can be configured elsewhere)
    logging.basicConfig(level=logging.INFO) # Consider changing to DEBUG for more verbose logs if needed
    main()