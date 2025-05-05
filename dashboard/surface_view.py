# surface_view.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Union, Optional
from scipy.interpolate import griddata # Needed for 2D interpolation

# Project-specific imports (relative imports might work if structure is correct)
try:
    from src.data_modeling.surface_lgbm_modeling import predict_surface # Predict RV surface
    from src.data_modeling.option_chain_modeling import generate_options_surface # Predict IV surface
except ImportError as e:
    st.error(f"Surface View: Failed to import modeling modules: {e}")
    st.stop()

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# -----------------------------------------------------------------------------
# Constants (Moved from original app)
# -----------------------------------------------------------------------------
AXIS_PADDING_FACTOR = 0.15 # Controls padding for individual surface axis calculation
OPTIONS_CLOSER_COLOR = 'rgba(0, 180, 0, 0.9)'  # Bright green for Options closer
LGBM_CLOSER_COLOR = 'rgba(255, 20, 147, 0.9)'  # Hot pink for LGBM closer
DEFAULT_RV_COLOR = 'rgba(255, 165, 0, 0.9)'  # Orange (default if only one surface or equal)
S0_LINE_COLOR = 'rgba(0, 191, 255, 0.9)'  # DeepSkyBlue for Hist RV @ Spot K line
TARGET_GRID_RESOLUTION = 50 # Number of points for K and T axes on common comparison grid
SURFACE_OPACITY = 0.95 # Opacity for 3D surfaces
SHARED_SURFACE_OPACITY = 0.65 # Opacity for shared surfaces

# -----------------------------------------------------------------------------
# Helper functions (Moved from original app)
# -----------------------------------------------------------------------------

# --- Date conversion utility ---
# Note: This is duplicated from the main app, consider a shared utils file later
def _to_date(x: Union[str, datetime.date, datetime.datetime, np.datetime64]) -> datetime.date:
    """Convert various date formats to date object."""
    if isinstance(x, str): return datetime.date.fromisoformat(x)
    if isinstance(x, datetime.datetime): return x.date()
    if isinstance(x, np.datetime64): return x.astype('datetime64[D]').item().date()
    if isinstance(x, datetime.date): return x
    raise TypeError(f"Cannot convert {type(x)} to date")

# --- Interpolation Helper ---
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
        return np.full(target_K_flat.shape, np.nan) # Return NaNs

    k_flat, t_flat, z_flat = surface_K_mesh.flatten(), surface_T_mesh.flatten(), surface_Z.flatten()
    valid_indices = np.isfinite(k_flat) & np.isfinite(t_flat) & np.isfinite(z_flat)
    if not np.any(valid_indices):
        logger.warning("Interpolation failed: Source surface K, T, or Z contains only non-finite values.")
        return np.full(target_K_flat.shape, np.nan) # Return NaNs

    points_for_interp = np.column_stack((k_flat[valid_indices], t_flat[valid_indices]))
    values_for_interp = z_flat[valid_indices]
    target_points = np.column_stack((target_K_flat, target_T_flat))

    try:
        logger.debug(f"Calling griddata with {points_for_interp.shape[0]} source points and {target_points.shape[0]} target points, method='{method}'")
        interpolated_z = griddata(points_for_interp, values_for_interp, target_points, method=method)
        logger.debug("griddata call successful.")
        return interpolated_z
    except Exception as e:
        logger.error(f"Interpolation using griddata failed: {e}", exc_info=True)
        return None

# --- Difference Calculation Helper ---
def calculate_point_differences(
    surface_name: str,
    surface_tuple: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]], # Allow None
    realized_points: Optional[Tuple[List[float], List[float], List[float]]]
) -> Optional[Dict[str, Any]]:
    """ Calculates differences between a surface and realized points using interpolation. """
    if realized_points is None or not realized_points[0]: return None

    realized_K, realized_T, realized_sigma = realized_points
    interpolated_Z = [None] * len(realized_K) # Default

    if isinstance(surface_tuple, tuple) and len(surface_tuple) == 3 and all(s is not None for s in surface_tuple):
        surface_K_mesh, surface_T_mesh, surface_Z = surface_tuple
        interpolated_Z_array = interpolate_surface_at_points(surface_K_mesh, surface_T_mesh, surface_Z, realized_K, realized_T)
        if interpolated_Z_array is not None:
             interpolated_Z = interpolated_Z_array.tolist()
        else:
             logger.warning(f"Interpolation failed for {surface_name}, cannot calculate differences.")
             # Keep interpolated_Z as [None, ...]
    else:
         logger.warning(f"Cannot calculate point differences for {surface_name}: Invalid or None surface_tuple provided.")
         # Keep interpolated_Z as [None, ...]

    differences, abs_differences, point_details = [], [], []
    for i, real_s in enumerate(realized_sigma):
        interp_s = interpolated_Z[i] if i < len(interpolated_Z) else None
        is_real_s_valid = np.isfinite(real_s)
        is_interp_s_valid = interp_s is not None and np.isfinite(interp_s)

        detail = { "K": realized_K[i], "T": realized_T[i], "Realized σ": real_s if is_real_s_valid else np.nan,
                   f"{surface_name} σ": interp_s if is_interp_s_valid else np.nan, "Difference": np.nan }

        if is_real_s_valid and is_interp_s_valid:
            diff = real_s - interp_s
            differences.append(diff)
            abs_differences.append(abs(diff))
            detail["Difference"] = diff
        point_details.append(detail)

    if not differences:
        logger.warning(f"No valid differences calculated between {surface_name} and realized points.")
        return {"average_diff": np.nan, "average_abs_diff": np.nan, "point_details": point_details}

    return {"average_diff": np.mean(differences), "average_abs_diff": np.mean(abs_differences), "point_details": point_details}

# --- Difference Display Helper ---
def display_point_differences(
    surface_name: str,
    diff_results: Optional[Dict[str, Any]],
    realized_available: bool
):
    """ Displays the calculated differences in an expander. """
    if not realized_available:
        st.info(f"Realized volatility points are not available/shown for comparison with {surface_name}.")
        return

    if diff_results:
        with st.expander(f"Comparison: {surface_name} vs. Realized Volatility Points"):
            avg_abs_diff = diff_results.get('average_abs_diff', np.nan)
            avg_diff = diff_results.get('average_diff', np.nan)

            col1, col2 = st.columns(2)
            col1.metric(f"Avg Abs Diff |Realized σ - {surface_name} σ|", f"{avg_abs_diff:.4f}" if np.isfinite(avg_abs_diff) else "N/A",
                      help="Average absolute difference. Lower is better.")
            col2.metric(f"Avg Signed Diff (Realized σ - {surface_name} σ)", f"{avg_diff:.4f}" if np.isfinite(avg_diff) else "N/A",
                      help="Average signed difference. Positive means Realized > Predicted on average.")

            point_details = diff_results.get("point_details")
            if point_details:
                st.markdown("###### Differences per Point:")
                try:
                    diff_df = pl.DataFrame(point_details)
                    if "K" in diff_df.columns:
                       diff_df = diff_df.with_columns(pl.col("K").cast(pl.Float64).map_elements(lambda k: f"${k:.2f}", return_dtype=pl.Utf8).alias("K_Display"))
                       display_cols_order = ["K_Display", "T", "Realized σ", f"{surface_name} σ", "Difference"]
                       cols_to_display = [col for col in display_cols_order if col in diff_df.columns or col == "K_Display"]
                       diff_df_display = diff_df.select(cols_to_display).rename({"K_Display": "K ($)"})
                    else:
                        display_cols_order = ["T", "Realized σ", f"{surface_name} σ", "Difference"]
                        cols_to_display = [col for col in display_cols_order if col in diff_df.columns]
                        diff_df_display = diff_df.select(cols_to_display)

                    diff_df_display = diff_df_display.with_columns(pl.col(pl.Float64).round(4))
                    st.dataframe(diff_df_display, use_container_width=True, hide_index=True)
                except Exception as e:
                    st.warning(f"Could not display difference details as DataFrame: {e}")
            else:
                st.warning("No difference details available.")
    else:
        st.warning(f"Could not calculate differences between {surface_name} surface and realized points.")


# --- RV Point Coloring Helper ---
def get_rv_point_colors_and_hovers(
    realized_points: Tuple[List[float], List[float], List[float]],
    lgbm_surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    options_surface: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]],
    option_type: str = "Avg"
) -> Tuple[List[str], List[str]]:
    """ Determines colors and hover text for RV points based on proximity to surfaces. """
    realized_K, realized_T, realized_sigma = realized_points
    n_points = len(realized_K)
    colors = [DEFAULT_RV_COLOR] * n_points
    hovers = [""] * n_points

    interp_lgbm_array, interp_opts_array = None, None
    if isinstance(lgbm_surface, tuple) and len(lgbm_surface)==3 and all(s is not None for s in lgbm_surface):
        interp_lgbm_array = interpolate_surface_at_points(lgbm_surface[0], lgbm_surface[1], lgbm_surface[2], realized_K, realized_T)
    if isinstance(options_surface, tuple) and len(options_surface)==3 and all(s is not None for s in options_surface):
        interp_opts_array = interpolate_surface_at_points(options_surface[0], options_surface[1], options_surface[2], realized_K, realized_T)

    interp_lgbm = interp_lgbm_array.tolist() if interp_lgbm_array is not None else [None] * n_points
    interp_opts = interp_opts_array.tolist() if interp_opts_array is not None else [None] * n_points

    for i in range(n_points):
        real_s = realized_sigma[i]
        k_val, t_val = realized_K[i], realized_T[i]
        lgbm_s, opts_s = interp_lgbm[i], interp_opts[i]
        is_real_s_valid = np.isfinite(real_s)

        if not is_real_s_valid:
            lgbm_s_text = f"{lgbm_s:.4f}" if lgbm_s is not None and np.isfinite(lgbm_s) else 'N/A'
            opts_s_text = f"{opts_s:.4f}" if opts_s is not None and np.isfinite(opts_s) else 'N/A'
            hover_text = (f"<b>Realized Point</b><br>K: ${k_val:.2f}, T: {t_val:.0f} cd<br>"
                          f"Realized σ: NaN<br>LGBM RV σ: {lgbm_s_text}<br>Options IV ({option_type}) σ: {opts_s_text}<br><extra></extra>")
            colors[i] = 'grey'
            hovers[i] = hover_text
            continue

        diff_lgbm = abs(real_s - lgbm_s) if lgbm_s is not None and np.isfinite(lgbm_s) else np.inf
        diff_opts = abs(real_s - opts_s) if opts_s is not None and np.isfinite(opts_s) else np.inf
        hover_text = (f"<b>Realized Point</b><br>K: ${k_val:.2f}, T: {t_val:.0f} cd<br>Realized σ: {real_s:.4f}<br>")

        lgbm_valid_interp = (lgbm_s is not None and np.isfinite(lgbm_s))
        opts_valid_interp = (opts_s is not None and np.isfinite(opts_s))

        if lgbm_valid_interp and opts_valid_interp:
            hover_text += f"LGBM RV σ: {lgbm_s:.4f} (Diff: {diff_lgbm:.4f})<br>"
            hover_text += f"Options IV ({option_type}) σ: {opts_s:.4f} (Diff: {diff_opts:.4f})<br>"
            if np.isclose(diff_lgbm, diff_opts): hover_text += "<i>Roughly equidistant</i>"; colors[i] = DEFAULT_RV_COLOR
            elif diff_lgbm < diff_opts: colors[i] = LGBM_CLOSER_COLOR; hover_text += "<i>Closer: LGBM RV</i>"
            else: colors[i] = OPTIONS_CLOSER_COLOR; hover_text += f"<i>Closer: Options IV ({option_type})</i>"
        elif lgbm_valid_interp:
            hover_text += f"LGBM RV σ: {lgbm_s:.4f} (Diff: {diff_lgbm:.4f})<br>Options IV ({option_type}) σ: N/A<br>"; colors[i] = DEFAULT_RV_COLOR
        elif opts_valid_interp:
            hover_text += f"LGBM RV σ: N/A<br>Options IV ({option_type}) σ: {opts_s:.4f} (Diff: {diff_opts:.4f})<br>"; colors[i] = DEFAULT_RV_COLOR
        else:
            hover_text += "LGBM RV σ: N/A<br>Options IV ({option_type}) σ: N/A<br>"; colors[i] = DEFAULT_RV_COLOR

        hover_text += "<extra></extra>"
        hovers[i] = hover_text
    return colors, hovers

# --- Overlay Data Extraction Helper ---
def extract_overlay_data(
    today_data: Optional[pl.DataFrame],
    loaded_horizons: List[int]
) -> Tuple[Optional[float], Optional[Tuple], Optional[Tuple]]:
    """ Extracts data needed for overlays (Realized Points and Hist RV @ Spot K line). """
    spot_price, realized_points_data, spot_line_data = None, None, None
    realized_strikes, realized_T, realized_sigma = [], [], []
    s0_line_T, s0_line_sigma_hist = [], []

    if today_data is None or today_data.is_empty(): return spot_price, realized_points_data, spot_line_data

    try:
        row_dict = today_data.row(0, named=True)
        spot_price = row_dict.get("close", np.nan)
        if spot_price is None or not np.isfinite(spot_price) or spot_price <= 0:
            logger.warning(f"Invalid spot price found: {spot_price}"); spot_price = np.nan
            return spot_price, realized_points_data, spot_line_data

        for h in loaded_horizons:
            Tcal = int(round(h * 7/5))
            # Realized points
            rv_future_col, lr_future_col = f"rv_{h}d_future", f"log_ret_future_{h}"
            if rv_future_col in row_dict and lr_future_col in row_dict:
                rv_fut, lr_fut = row_dict.get(rv_future_col), row_dict.get(lr_future_col)
                if rv_fut is not None and lr_fut is not None and np.isfinite(rv_fut) and np.isfinite(lr_fut) and rv_fut >= 0:
                    try:
                        sigma0_fut = np.sqrt(rv_fut / (h / 252.0)) if h > 0 else np.nan
                        realized_K = spot_price * np.exp(lr_fut)
                        if np.isfinite(sigma0_fut) and sigma0_fut > 0:
                            realized_strikes.append(realized_K); realized_T.append(Tcal); realized_sigma.append(sigma0_fut)
                    except Exception as math_err: logger.warning(f"Math error calc realized point h={h}: {math_err}")

            # Hist RV @ Spot K line
            rv_hist_col = f"rv_{h}d"
            rv_for_s0_line = row_dict.get(rv_hist_col)
            if rv_for_s0_line is not None and np.isfinite(rv_for_s0_line) and rv_for_s0_line >= 0:
                try:
                    sigma0_hist_at_S0 = np.sqrt(rv_for_s0_line / (h / 252.0)) if h > 0 else np.nan
                    if np.isfinite(sigma0_hist_at_S0) and sigma0_hist_at_S0 > 0:
                        s0_line_T.append(Tcal); s0_line_sigma_hist.append(sigma0_hist_at_S0)
                except Exception as math_err: logger.warning(f"Math error calc Hist RV@Spot point h={h}: {math_err}")

    except IndexError: logger.warning("Cannot access row 0 for overlay data extraction.")
    except Exception as e: logger.error(f"Error extracting overlay data: {e}", exc_info=True)

    valid_realized_indices = [i for i, (k, t, s) in enumerate(zip(realized_strikes, realized_T, realized_sigma)) if np.isfinite(k) and np.isfinite(t) and np.isfinite(s)]
    if valid_realized_indices:
        realized_points_data = ([realized_strikes[i] for i in valid_realized_indices], [realized_T[i] for i in valid_realized_indices], [realized_sigma[i] for i in valid_realized_indices])

    valid_spot_indices = [i for i, (t, s) in enumerate(zip(s0_line_T, s0_line_sigma_hist)) if np.isfinite(t) and np.isfinite(s)]
    if valid_spot_indices:
        s0_T_valid = [s0_line_T[i] for i in valid_spot_indices]; s0_sigma_valid = [s0_line_sigma_hist[i] for i in valid_spot_indices]
        sorted_pairs = sorted(zip(s0_T_valid, s0_sigma_valid))
        spot_line_data = ([p[0] for p in sorted_pairs], [p[1] for p in sorted_pairs])

    return spot_price, realized_points_data, spot_line_data

# -----------------------------------------------------------------------------
# Plotting Functions (Moved from original app)
# -----------------------------------------------------------------------------

# --- LGBM Plotting ---
def create_plotly_figure(
    K_mesh: np.ndarray, T_mesh: np.ndarray, Z: np.ndarray, stock: str, trade_date: datetime.date,
    show_realized: bool, show_s0: bool, spot_price: Optional[float],
    realized_points_data: Optional[Tuple], spot_line_data: Optional[Tuple]
) -> Tuple[go.Figure, Dict[str, Any]]:
    """ Creates the Plotly 3D surface plot for LGBM. """
    fig = go.Figure()
    S0 = spot_price if spot_price is not None and np.isfinite(spot_price) else np.nan
    k_axis_elements, sigma_axis_elements = [], []

    if not np.isnan(S0): k_axis_elements.append(S0)
    if show_realized and realized_points_data: k_axis_elements.extend([k for k in realized_points_data[0] if np.isfinite(k)])

    k_min, k_max = np.nan, np.nan
    if k_axis_elements:
        finite_k = [k for k in k_axis_elements if np.isfinite(k)]
        if finite_k:
            k_min_calc, k_max_calc = np.min(finite_k), np.max(finite_k)
            k_pad = max((k_max_calc - k_min_calc) * AXIS_PADDING_FACTOR / 2.0, 1.0)
            k_min, k_max = max(0.0, k_min_calc - k_pad), k_max_calc + k_pad
    if np.isnan(k_min): k_min = np.nanmin(K_mesh) * 0.9 if np.any(np.isfinite(K_mesh)) else 0
    if np.isnan(k_max): k_max = np.nanmax(K_mesh) * 1.1 if np.any(np.isfinite(K_mesh)) else 100

    if show_s0 and spot_line_data and not np.isnan(S0): sigma_axis_elements.extend([s for s in spot_line_data[1] if np.isfinite(s)])
    if show_realized and realized_points_data: sigma_axis_elements.extend([s for s in realized_points_data[2] if np.isfinite(s)])
    if Z is not None and Z.size > 0 and K_mesh is not None and K_mesh.size == Z.size:
        k_flat, z_flat = K_mesh.flatten(), Z.flatten()
        mask = (k_flat >= k_min) & (k_flat <= k_max) & np.isfinite(z_flat)
        if np.any(mask): sigma_axis_elements.extend(z_flat[mask].tolist())

    sigma_min, sigma_max = 0.0, 1.0
    if sigma_axis_elements:
        finite_s = [s for s in sigma_axis_elements if np.isfinite(s)]
        if finite_s:
            s_min_calc, s_max_calc = np.min(finite_s), np.max(finite_s)
            s_pad = max((s_max_calc - s_min_calc) * AXIS_PADDING_FACTOR / 2.0, 0.01)
            sigma_min, sigma_max = max(0.0, s_min_calc - s_pad), s_max_calc + s_pad

    if k_max <= k_min: k_max = k_min + 10.0
    if sigma_max <= sigma_min: sigma_max = sigma_min + 0.1
    t_min = np.nanmin(T_mesh) if np.any(np.isfinite(T_mesh)) else 0
    t_max = np.nanmax(T_mesh) if np.any(np.isfinite(T_mesh)) else 365

    if Z is not None and np.any(np.isfinite(Z)):
        fig.add_trace(go.Surface(
            x=K_mesh, y=T_mesh, z=Z, name="LGBM Predicted RV", showlegend=True, colorscale="Plasma",
            opacity=SURFACE_OPACITY, colorbar=dict(title='LGBM RV (σ)', thickness=15, len=0.6, y=0.8),
            contours={"z": {"show": True, "highlight": True, "highlightcolor": "limegreen", "project": {"z": True}}},
            hovertemplate="<b>LGBM RV</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>",
            lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2) ))
    else: logger.warning("LGBM surface Z data invalid, not plotting.")

    if show_realized and realized_points_data:
        fig.add_trace(go.Scatter3d( x=realized_points_data[0], y=realized_points_data[1], z=realized_points_data[2], mode="markers",
            marker=dict(color=DEFAULT_RV_COLOR, size=5, symbol='diamond'), name="Realized Points", showlegend=True,
            hovertemplate="<b>Realized Point</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" ))

    if show_s0 and spot_line_data and not np.isnan(S0):
        fig.add_trace(go.Scatter3d( x=[S0] * len(spot_line_data[0]), y=spot_line_data[0], z=spot_line_data[1], mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=3), line=dict(color=S0_LINE_COLOR, width=4),
            name=f'Hist. RV @ Spot K=${S0:.2f}', showlegend=True,
            hovertemplate="<b>Hist RV@Spot</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" ))

    fig.update_layout( title=f"{stock} LGBM Predicted RV Surface @ {trade_date}",
        scene=dict( xaxis_title="Strike Price (K, $)", yaxis_title="Time to Maturity (Days, T)", zaxis_title="Annualized Volatility (σ)",
                    xaxis_range=[k_min, k_max], zaxis_range=[sigma_min, sigma_max], yaxis_range=[t_min, t_max],
                    aspectratio=dict(x=1.5, y=1.5, z=1) ),
        margin=dict(l=10, r=10, b=10, t=50), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0)) )

    axis_ranges = { "k_min": k_min, "k_max": k_max, "sigma_min": sigma_min, "sigma_max": sigma_max, "t_min": t_min, "t_max": t_max }
    return fig, axis_ranges

# --- Options Plotting ---
# Helper to get actual strike range from adjusted options data
def get_options_strike_range(option_chain_df: Optional[pl.DataFrame], stock: str, date: datetime.date) -> Optional[Tuple[float, float]]:
    """ Get actual min/max adjusted strike from the options chain dataframe for a specific stock and date. """
    if option_chain_df is None or option_chain_df.is_empty(): return None
    try:
        target_date = _to_date(date)
        if target_date is None: logger.error(f"Invalid date format: {date}"); return None

        filtered_df = option_chain_df.filter((pl.col("act_symbol") == stock) & (pl.col("date").cast(pl.Date) == target_date))
        if filtered_df.is_empty(): logger.warning(f"No options data for {stock} on {target_date} in get_options_strike_range."); return None

        min_strike, max_strike = filtered_df["strike"].min(), filtered_df["strike"].max()
        if min_strike is None or max_strike is None or not np.isfinite(min_strike) or not np.isfinite(max_strike) or max_strike <= min_strike:
            logger.warning(f"Invalid strike range for {stock} on {target_date}: min={min_strike}, max={max_strike}"); return None
        return (min_strike, max_strike)
    except Exception as e: logger.error(f"Error getting options strike range for {stock} on {date}: {e}", exc_info=True); return None

# Helper to calculate options axis ranges including overlays
def calculate_options_axis_ranges(
    options_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    show_realized: bool, realized_points_data: Optional[Tuple], show_s0: bool,
    spot_price: Optional[float], spot_line_data: Optional[Tuple],
    option_chain_df: Optional[pl.DataFrame], stock: Optional[str], date: Optional[datetime.date]
) -> Optional[Dict[str, float]]:
    """ Calculate axis ranges for options chain display including overlays. """
    if not isinstance(options_surface, tuple) or len(options_surface) != 3 or any(s is None for s in options_surface):
        logger.warning("Cannot calc options axis ranges: Invalid options_surface."); return None
    opt_K_mesh, opt_T_mesh, opt_Z = options_surface

    actual_k_min, actual_k_max = None, None
    strike_range = get_options_strike_range(option_chain_df, stock, date)
    if strike_range:
        raw_k_min, raw_k_max = strike_range
        if np.isfinite(raw_k_min) and np.isfinite(raw_k_max) and raw_k_max > raw_k_min:
            actual_k_min, actual_k_max = raw_k_min, raw_k_max
            logger.info(f"Using actual strike range from options: ${actual_k_min:.2f} to ${actual_k_max:.2f}")
        else: logger.warning(f"Invalid strike range ({raw_k_min}, {raw_k_max}) for {stock}/{date}."); return None
    else: logger.warning(f"No adjusted options data for {stock}/{date} for strike range."); return None

    opt_sigma_axis_elements = []
    if opt_Z is not None and opt_Z.size > 0:
        valid_Z = opt_Z[np.isfinite(opt_Z)]
        if valid_Z.size > 0: opt_sigma_axis_elements.extend(valid_Z.tolist())
    if show_s0 and spot_line_data and spot_price is not None and np.isfinite(spot_price):
        opt_sigma_axis_elements.extend([s for s in spot_line_data[1] if np.isfinite(s)])
    if show_realized and realized_points_data:
        opt_sigma_axis_elements.extend([s for s in realized_points_data[2] if np.isfinite(s)])

    opt_sigma_min, opt_sigma_max = 0.0, 1.0
    if opt_sigma_axis_elements:
        finite_sigma = [s for s in opt_sigma_axis_elements if np.isfinite(s)]
        if finite_sigma:
            s_min_calc, s_max_calc = np.min(finite_sigma), np.max(finite_sigma)
            s_pad = max(min((s_max_calc - s_min_calc) * AXIS_PADDING_FACTOR / 2.0, 0.1), 0.01)
            opt_sigma_min, opt_sigma_max = max(0.0, s_min_calc - s_pad), s_max_calc + s_pad
        else: logger.warning("Options sigma axis elements only non-finite, using default.")
    else: logger.warning("No valid data for Options sigma axis, using default.")

    t_min = np.nanmin(opt_T_mesh) if np.any(np.isfinite(opt_T_mesh)) else 0
    t_max = np.nanmax(opt_T_mesh) if np.any(np.isfinite(opt_T_mesh)) else 365
    if opt_sigma_max <= opt_sigma_min: opt_sigma_max = opt_sigma_min + 0.1

    return { "k_min": actual_k_min, "k_max": actual_k_max, "sigma_min": opt_sigma_min, "sigma_max": opt_sigma_max,
             "t_min": t_min, "t_max": t_max }

# Main options plotting function
def plot_options_surface(
    opt_K_mesh: np.ndarray, opt_T_mesh: np.ndarray, opt_Z: np.ndarray, stock: str, trade_date: datetime.date,
    option_type: str, axis_ranges: Dict[str, float], show_realized: bool = False, realized_points_data: Optional[Tuple] = None,
    show_s0: bool = False, spot_price: Optional[float] = None, spot_line_data: Optional[Tuple] = None
) -> go.Figure:
    """ Plots the 3D options implied volatility surface with optional overlays. """
    fig = go.Figure()
    S0 = spot_price if spot_price is not None and np.isfinite(spot_price) else np.nan

    if opt_Z is not None and np.any(np.isfinite(opt_Z)):
        fig.add_trace(go.Surface(
            x=opt_K_mesh, y=opt_T_mesh, z=opt_Z, name=f"Options IV ({option_type.capitalize()})", showlegend=True, colorscale="Viridis",
            opacity=SURFACE_OPACITY, colorbar=dict(title='Options IV (σ)', thickness=15, len=0.6, y=0.8),
            contours={"z": {"show": True, "highlight": True, "highlightcolor": "yellow", "project": {"z": True}}},
            hovertemplate="<b>Options IV</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>",
            lighting=dict(ambient=0.7, diffuse=0.7, specular=0.2) ))
    else: logger.warning("Options surface Z data invalid, not plotting.")

    if show_realized and realized_points_data:
        fig.add_trace(go.Scatter3d( x=realized_points_data[0], y=realized_points_data[1], z=realized_points_data[2], mode="markers",
            marker=dict(color=DEFAULT_RV_COLOR, size=5, symbol='diamond'), name="Realized Points", showlegend=True,
            hovertemplate="<b>Realized Pt</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" ))

    if show_s0 and spot_line_data and not np.isnan(S0):
        fig.add_trace(go.Scatter3d( x=[S0] * len(spot_line_data[0]), y=spot_line_data[0], z=spot_line_data[1], mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=3), line=dict(color=S0_LINE_COLOR, width=4),
            name=f'Hist. RV @ Spot K=${S0:.2f}', showlegend=True,
            hovertemplate="<b>Hist RV@Spot</b><br>K: $%{x:.2f}, T: %{y:.0f} cd<br>σ: %{z:.4f}<extra></extra>" ))

    k_min = axis_ranges.get("k_min", 0) if axis_ranges else 0
    k_max = axis_ranges.get("k_max", 100) if axis_ranges else 100
    sigma_min = axis_ranges.get("sigma_min", 0) if axis_ranges else 0
    sigma_max = axis_ranges.get("sigma_max", 1.0) if axis_ranges else 1.0
    t_min = axis_ranges.get("t_min", 0) if axis_ranges else 0
    t_max = axis_ranges.get("t_max", 365) if axis_ranges else 365

    fig.update_layout( title=f"{stock} Options IV Surface ({option_type.capitalize()}) @ {trade_date}",
        scene=dict( xaxis_title="Strike Price (K, $)", yaxis_title="Time to Maturity (Days, T)", zaxis_title="Implied Volatility (σ)",
                    xaxis_range=[k_min, k_max], zaxis_range=[sigma_min, sigma_max], yaxis_range=[t_min, t_max],
                    aspectratio=dict(x=1.5, y=1.5, z=1) ),
        margin=dict(l=10, r=10, b=10, t=50), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.7, y=-1.7, z=1.0)) )
    return fig

# --- Comparison Plotting ---
def compare_surfaces(
    options_surface: Optional[Tuple], lgbm_surface: Optional[Tuple], stock: str, trade_date: datetime.date,
    option_type: str, k_axis: np.ndarray, t_axis: np.ndarray, sigma_range: Tuple[float, float],
    show_realized: bool = False, realized_points_data: Optional[Tuple] = None, show_s0: bool = False,
    spot_price: Optional[float] = None, spot_line_data: Optional[Tuple] = None
) -> go.Figure:
    """ Compares surfaces using explicitly defined common axes ranges. """
    fig = go.Figure()
    S0 = spot_price if spot_price is not None and np.isfinite(spot_price) else np.nan
    opt_K, opt_T, opt_Z = (options_surface if isinstance(options_surface, tuple) and len(options_surface)==3 and all(s is not None for s in options_surface) else (None, None, None))
    lgbm_K, lgbm_T, lgbm_Z = (lgbm_surface if isinstance(lgbm_surface, tuple) and len(lgbm_surface)==3 and all(s is not None for s in lgbm_surface) else (None, None, None))

    if opt_Z is not None and np.any(np.isfinite(opt_Z)):
        fig.add_trace(go.Surface(
            x=opt_K, y=opt_T, z=opt_Z, name=f"Options IV ({option_type.capitalize()})", colorscale="Viridis",
            opacity=SURFACE_OPACITY, showscale=True, colorbar=dict(title='Options IV', thickness=15, len=0.6, y=0.8, x=1.02),
            hovertemplate="<b>Options IV</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>", showlegend=True ))
    if lgbm_Z is not None and np.any(np.isfinite(lgbm_Z)):
        fig.add_trace(go.Surface(
            x=lgbm_K, y=lgbm_T, z=lgbm_Z, name="LGBM RV", colorscale="Plasma", opacity=SHARED_SURFACE_OPACITY,
            showscale=True, colorbar=dict(title='LGBM RV', thickness=15, len=0.6, y=0.15, x=1.02),
            hovertemplate="<b>LGBM RV</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>", showlegend=True ))

    if show_realized and realized_points_data:
        rv_colors, rv_hovers = get_rv_point_colors_and_hovers(realized_points_data, lgbm_surface, options_surface, option_type.capitalize())
        fig.add_trace(go.Scatter3d( x=realized_points_data[0], y=realized_points_data[1], z=realized_points_data[2], mode="markers",
            marker=dict(color=rv_colors, size=6, symbol='diamond', line=dict(color='black', width=1)),
            name="Realized Points", showlegend=True, hovertemplate=rv_hovers ))

    if show_s0 and spot_line_data and not np.isnan(S0):
        fig.add_trace(go.Scatter3d( x=[S0] * len(spot_line_data[0]), y=spot_line_data[0], z=spot_line_data[1], mode='lines+markers',
            marker=dict(color=S0_LINE_COLOR, size=4, line=dict(color='black', width=1)), line=dict(color=S0_LINE_COLOR, width=5),
            name=f'Hist. RV @ Spot K=${S0:.2f}', showlegend=True,
            hovertemplate="<b>Hist RV@Spot</b><br>K:$%{x:.2f}, T:%{y:.0f}d<br>σ:%{z:.4f}<extra></extra>" ))

    k_min, k_max = k_axis.min(), k_axis.max()
    t_min, t_max = t_axis.min(), t_axis.max()
    sigma_min, sigma_max = sigma_range

    fig.update_layout( title=f"{stock} Comparison: Options IV vs LGBM RV @ {trade_date}",
        scene=dict( xaxis_title="Strike Price (K, $)", yaxis_title="Time to Maturity (Days, T)", zaxis_title="Volatility (σ)",
                    xaxis_range=[k_min, k_max], yaxis_range=[t_min, t_max], zaxis_range=[sigma_min, sigma_max],
                    aspectratio=dict(x=1.5, y=1.5, z=1) ),
        margin=dict(l=0, r=40, b=0, t=40), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        scene_camera=dict(eye=dict(x=1.8, y=-1.8, z=0.9)) )
    return fig


# --- Difference Calculation on Common Axes ---
def calculate_pricing_difference_on_axes(
    options_surface: Optional[Tuple], lgbm_surface: Optional[Tuple], target_k_axis: np.ndarray, target_t_axis: np.ndarray,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
    """ Calculates percentage difference interpolating BOTH surfaces onto common grid axes. """
    stats = {"max_diff": np.nan, "min_diff": np.nan, "mean_diff": np.nan, "max_diff_k": np.nan, "max_diff_t": np.nan, "min_diff_k": np.nan, "min_diff_t": np.nan}
    default_return = (None, None, None, None, None, stats)

    if not isinstance(options_surface, tuple) or len(options_surface) != 3 or not all(isinstance(arr, np.ndarray) and arr.size > 0 for arr in options_surface):
        logger.error("Invalid options_surface for diff calc."); return default_return
    opt_K_mesh, opt_T_mesh, opt_Z = options_surface
    if not isinstance(lgbm_surface, tuple) or len(lgbm_surface) != 3 or not all(isinstance(arr, np.ndarray) and arr.size > 0 for arr in lgbm_surface):
        logger.error("Invalid lgbm_surface for diff calc."); return default_return
    lgbm_K_mesh, lgbm_T_mesh, lgbm_Z = lgbm_surface
    if opt_K_mesh.shape != opt_T_mesh.shape or opt_K_mesh.shape != opt_Z.shape or \
       lgbm_K_mesh.shape != lgbm_T_mesh.shape or lgbm_K_mesh.shape != lgbm_Z.shape:
        logger.error("Mismatched surface shapes for diff calc."); return default_return
    if target_k_axis is None or target_t_axis is None or target_k_axis.ndim != 1 or target_t_axis.ndim != 1 or target_k_axis.size == 0 or target_t_axis.size == 0:
        logger.error("Invalid target axes for diff calc."); return default_return

    target_K_mesh, target_T_mesh = np.meshgrid(target_k_axis, target_t_axis)
    target_points_k_flat, target_points_t_flat = target_K_mesh.flatten(), target_T_mesh.flatten()
    logger.info(f"Created common diff grid shape {target_K_mesh.shape}")

    opt_Z_interp_flat = interpolate_surface_at_points(opt_K_mesh, opt_T_mesh, opt_Z, target_points_k_flat, target_points_t_flat)
    if opt_Z_interp_flat is None: logger.error("Options surface interp failed."); return default_return
    opt_Z_common = opt_Z_interp_flat.reshape(target_K_mesh.shape)

    lgbm_Z_interp_flat = interpolate_surface_at_points(lgbm_K_mesh, lgbm_T_mesh, lgbm_Z, target_points_k_flat, target_points_t_flat)
    if lgbm_Z_interp_flat is None: logger.error("LGBM surface interp failed."); return default_return
    lgbm_Z_common = lgbm_Z_interp_flat.reshape(target_K_mesh.shape)

    with np.errstate(divide='ignore', invalid='ignore'):
        diff_Z = (opt_Z_common - lgbm_Z_common) / lgbm_Z_common * 100
        diff_Z[~np.isfinite(diff_Z)] = np.nan

    valid_diff_mask = np.isfinite(diff_Z)
    if np.any(valid_diff_mask):
        valid_diffs = diff_Z[valid_diff_mask]
        stats["max_diff"], stats["min_diff"], stats["mean_diff"] = np.nanmax(valid_diffs), np.nanmin(valid_diffs), np.nanmean(valid_diffs)
        if np.isfinite(stats["max_diff"]):
             max_indices = np.where(np.isclose(diff_Z, stats["max_diff"]) & valid_diff_mask)
             if len(max_indices[0]) > 0: idx = (max_indices[0][0], max_indices[1][0]); stats["max_diff_t"], stats["max_diff_k"] = target_T_mesh[idx], target_K_mesh[idx]
        if np.isfinite(stats["min_diff"]):
            min_indices = np.where(np.isclose(diff_Z, stats["min_diff"]) & valid_diff_mask)
            if len(min_indices[0]) > 0: idx = (min_indices[0][0], min_indices[1][0]); stats["min_diff_t"], stats["min_diff_k"] = target_T_mesh[idx], target_K_mesh[idx]
        logger.info(f"Diff stats calculated over {np.sum(valid_diff_mask)} valid points.")
    else: logger.warning("No valid diff points found on common grid.")

    return diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats


# --- Difference Heatmap Plotting ---
def create_pricing_difference_plot(
    diff_Z: np.ndarray, K_mesh: np.ndarray, T_mesh: np.ndarray, opt_Z_common: np.ndarray, lgbm_Z_common: np.ndarray,
    stats: Dict[str, Any], stock: str, trade_date: datetime.date, option_type: str
) -> go.Figure:
    """ Creates a heatmap of pricing differences. """
    fig = go.Figure()
    if diff_Z is None or K_mesh is None or T_mesh is None or opt_Z_common is None or lgbm_Z_common is None or \
       diff_Z.size == 0 or K_mesh.size == 0 or T_mesh.size == 0 or diff_Z.shape != K_mesh.shape or \
       diff_Z.shape != opt_Z_common.shape or diff_Z.shape != lgbm_Z_common.shape:
        logger.warning("Cannot create heatmap: Invalid input diff_Z, K/T mesh, or common Z values, or mismatched shapes.")
        fig.update_layout(title=f"Pricing Difference Heatmap Failed for {stock} @ {trade_date}")
        return fig

    customdata = np.dstack((K_mesh, T_mesh, opt_Z_common, lgbm_Z_common))
    fig.add_trace(go.Heatmap( z=diff_Z, x=K_mesh[0, :], y=T_mesh[:, 0], colorscale="RdBu_r", zmid=0,
            colorbar=dict(title="OptionsIV-LGBMRV<br>Diff (%)", thickness=15), customdata=customdata,
            hovertemplate=( "<b>Difference Details</b><br>K: $%{customdata[0]:.2f}<br>T: %{customdata[1]:.0f}d<br>"
                            "Options IV: %{customdata[2]:.4f}<br>LGBM RV: %{customdata[3]:.4f}<br>"
                            "Difference: %{z:.2f}%<extra></extra>"), hoverongaps=False ))

    if stats and np.isfinite(stats.get('max_diff_k', np.nan)) and np.isfinite(stats.get('max_diff_t', np.nan)):
        max_diff_val = stats.get('max_diff', np.nan)
        fig.add_trace(go.Scatter( x=[stats['max_diff_k']], y=[stats['max_diff_t']], mode="markers", name=f"Max Over (+{max_diff_val:.1f}%)",
            marker=dict(symbol="circle", size=10, color="rgba(255,0,0,0.9)", line=dict(width=1, color="black")),
            hovertemplate=(f"<b>Max Over (Opt > LGBM)</b><br>K: $%{{x:.2f}}, T: %{{y:.0f}}d<br>Diff: {max_diff_val:.2f}%<extra></extra>") if np.isfinite(max_diff_val) else "" ))
    if stats and np.isfinite(stats.get('min_diff_k', np.nan)) and np.isfinite(stats.get('min_diff_t', np.nan)):
        min_diff_val = stats.get('min_diff', np.nan)
        fig.add_trace(go.Scatter( x=[stats['min_diff_k']], y=[stats['min_diff_t']], mode="markers", name=f"Max Under ({min_diff_val:.1f}%)",
            marker=dict(symbol="circle", size=10, color="rgba(0,0,255,0.9)", line=dict(width=1, color="black")),
            hovertemplate=(f"<b>Max Under (Opt < LGBM)</b><br>K: $%{{x:.2f}}, T: %{{y:.0f}}d<br>Diff: {min_diff_val:.2f}%<extra></extra>") if np.isfinite(min_diff_val) else "" ))

    fig.update_layout( title=f"{stock} Options IV vs LGBM RV Difference (%) @ {trade_date}", xaxis_title="Strike Price (K, $)", yaxis_title="Days to Expiration (T)",
        yaxis_autorange='reversed', width=None, height=600, legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99) )
    if K_mesh.size > 0: fig.update_xaxes(range=[np.nanmin(K_mesh[0, :]), np.nanmax(K_mesh[0, :])])
    if T_mesh.size > 0: fig.update_yaxes(range=[np.nanmin(T_mesh[:, 0]), np.nanmax(T_mesh[:, 0])])
    return fig

# -----------------------------------------------------------------------------
# Main Display Function for this View
# -----------------------------------------------------------------------------
def display_surface_view(loaded_data: Dict[str, Any], params: Dict[str, Any]):
    """
    Handles the logic and display for the Volatility Surface view.

    Args:
        loaded_data: Dictionary containing pre-loaded dataframes, models, etc.
                     Keys: "stock", "trade_date", "nearest_ohlcv_date", "nearest_options_date",
                           "transformed_df", "option_chain_adj", "lgbm_models_dict", "horizons".
        params: Dictionary containing sidebar parameters for this view.
                Keys: "vis_mode", "option_type", "show_realized_points", "show_hist_rv_line",
                      "show_diff_heatmap", "show_stats", "show_rv_comparison".
    """
    st.header("Volatility Surface Analysis")

    # Unpack loaded data and parameters
    stock = loaded_data["stock"]
    trade_date = loaded_data["trade_date"] # User requested date
    nearest_ohlcv_date = loaded_data["nearest_ohlcv_date"]
    nearest_options_date = loaded_data["nearest_options_date"]
    transformed_df = loaded_data["transformed_df"]
    option_chain_adj = loaded_data["option_chain_adj"]
    lgbm_models_dict = loaded_data["lgbm_models_dict"]
    horizons = loaded_data["horizons"]

    vis_mode = params["vis_mode"]
    option_type = params["option_type"]
    show_realized_points = params["show_realized_points"]
    show_hist_rv_line = params["show_hist_rv_line"]
    show_diff_heatmap = params["show_diff_heatmap"]
    show_stats = params["show_stats"]
    show_rv_comparison = params["show_rv_comparison"]

    # Initialize required variables
    lgbm_surface, options_surface = None, None
    lgbm_axis_ranges, options_axis_ranges = None, None
    spot_price, realized_points_data, spot_line_data = None, None, None
    error_occurred = False
    generation_placeholder = st.empty()

    try:
        with generation_placeholder.status("Extracting overlay data & generating surfaces...", expanded=True):

            # --- Extract Overlay Data ---
            st.write("Extracting overlay data (Realized Points, Hist RV @ Spot)...")
            lgbm_today_data = None
            if transformed_df is not None and nearest_ohlcv_date:
                 lgbm_today_data = transformed_df.filter(
                    (pl.col("act_symbol") == stock) &
                    (pl.col("date").cast(pl.Date) == nearest_ohlcv_date)
                 ).head(1)
            if lgbm_today_data is not None and not lgbm_today_data.is_empty():
                 spot_price, realized_points_data, spot_line_data = extract_overlay_data(
                     lgbm_today_data, horizons # Use horizons from loaded models/data
                 )
                 if spot_price is None or np.isnan(spot_price): st.warning("Could not determine spot price for overlays.")
                 if realized_points_data is None: st.warning("Could not extract realized points data.")
                 if spot_line_data is None: st.warning("Could not extract historical RV line data.")
            else:
                st.warning(f"No transformed data found for {stock} on {nearest_ohlcv_date} to extract overlays.")

            # --- Generate Surfaces & Calculate Axes ---
            # Generate LGBM Surface & Axes (if needed)
            if vis_mode != "Options Chain IV":
                 st.write("Generating LGBM RV Surface & Axes...")
                 try:
                     result = predict_surface(lgbm_models_dict, transformed_df, stock, nearest_ohlcv_date)
                     if result and result[2] is not None and np.any(np.isfinite(result[2])):
                         lgbm_surface = (result[0], result[1], result[2])
                         _, lgbm_axis_ranges = create_plotly_figure( # Use plotting func to calc axes incl. overlays
                              *lgbm_surface, stock, nearest_ohlcv_date,
                              show_realized=show_realized_points, show_s0=show_hist_rv_line,
                              spot_price=spot_price, realized_points_data=realized_points_data, spot_line_data=spot_line_data
                         )
                         st.write("LGBM RV Surface generated.")
                     else:
                          st.warning("LGBM RV surface generation failed or resulted in invalid data.")
                          lgbm_surface = None; lgbm_axis_ranges = None
                 except Exception as e:
                     st.warning(f"Error generating LGBM RV surface/axes: {e}")
                     lgbm_surface = None; lgbm_axis_ranges = None

            # Generate Options Surface & Axes (if needed)
            if vis_mode != "LGBM Model RV":
                st.write("Generating Options IV Surface & Axes...")
                try:
                    # Note: generate_options_surface uses the *raw* options chain before adjustment internally
                    # but we pass the *adjusted* chain to calculate_options_axis_ranges for correct K range
                    raw_options_for_gen = loaded_data.get("option_chain_raw", option_chain_adj) # Fallback to adjusted if raw missing
                    options_surface_result = generate_options_surface(
                        raw_options_for_gen, stock, nearest_options_date, option_type.lower()
                    )
                    if options_surface_result and options_surface_result[2] is not None and np.any(np.isfinite(options_surface_result[2])):
                        options_surface = options_surface_result
                        options_axis_ranges = calculate_options_axis_ranges(
                            options_surface, show_realized=show_realized_points, realized_points_data=realized_points_data,
                            show_s0=show_hist_rv_line, spot_price=spot_price, spot_line_data=spot_line_data,
                            option_chain_df=option_chain_adj, stock=stock, date=nearest_options_date # Pass adjusted chain here
                        )
                        if options_axis_ranges is None:
                            st.warning("Failed to calculate Options IV axis ranges.")
                            options_surface = None
                        else: st.write("Options IV Surface generated.")
                    else:
                        st.warning("Options IV surface generation failed or resulted in invalid data.")
                        options_surface = None; options_axis_ranges = None
                except Exception as e:
                    st.warning(f"Error generating Options IV surface/axes: {e}")
                    options_surface = None; options_axis_ranges = None

        generation_placeholder.empty()

        # --- Plotting and Comparisons ---
        if vis_mode == "LGBM Model RV" and not lgbm_surface: error_occurred = True; st.error(f"Could not generate LGBM RV surface.")
        if vis_mode == "Options Chain IV" and not options_surface: error_occurred = True; st.error(f"Could not generate Options IV surface.")
        if vis_mode == "Compare Surfaces" and (not lgbm_surface or not options_surface): error_occurred = True; st.error(f"Could not generate both surfaces for comparison.")

        if not error_occurred:
            realized_available = show_realized_points and (realized_points_data is not None)
            hist_rv_available = show_hist_rv_line and (spot_price is not None and np.isfinite(spot_price)) and (spot_line_data is not None)

            # --- LGBM View ---
            if vis_mode == "LGBM Model RV":
                st.subheader(f"LGBM Predicted RV Surface ({stock} @ {nearest_ohlcv_date})")
                fig_lgbm, _ = create_plotly_figure( *lgbm_surface, stock, nearest_ohlcv_date,
                    show_realized=show_realized_points, show_s0=show_hist_rv_line,
                    spot_price=spot_price, realized_points_data=realized_points_data, spot_line_data=spot_line_data )
                st.plotly_chart(fig_lgbm, use_container_width=True)
                # This comparison section is only relevant in compare view, but keep logic structure similar
                # if show_rv_comparison: # Logic exists below

            # --- Options View ---
            elif vis_mode == "Options Chain IV":
                st.subheader(f"Options IV Surface ({option_type}, {stock} @ {nearest_options_date})")
                fig_opts = plot_options_surface( *options_surface, stock, nearest_options_date, option_type, options_axis_ranges,
                    show_realized=show_realized_points, realized_points_data=realized_points_data,
                    show_s0=show_hist_rv_line, spot_price=spot_price, spot_line_data=spot_line_data )
                st.plotly_chart(fig_opts, use_container_width=True)
                # if show_rv_comparison: # Logic exists below

            # --- Comparison View ---
            elif vis_mode == "Compare Surfaces":
                 plot_date_ref = nearest_options_date # Use options date as reference
                 st.subheader(f"Comparison: Options IV vs LGBM RV ({stock} @ {plot_date_ref})")

                 # --- Define Common Axes ---
                 combined_k_min = min(lgbm_axis_ranges["k_min"], options_axis_ranges["k_min"])
                 combined_k_max = max(lgbm_axis_ranges["k_max"], options_axis_ranges["k_max"])
                 combined_sigma_min = min(lgbm_axis_ranges["sigma_min"], options_axis_ranges["sigma_min"])
                 combined_sigma_max = max(lgbm_axis_ranges["sigma_max"], options_axis_ranges["sigma_max"])
                 combined_t_min = min(lgbm_axis_ranges["t_min"], options_axis_ranges["t_min"])
                 combined_t_max = max(lgbm_axis_ranges["t_max"], options_axis_ranges["t_max"])

                 common_k_axis = np.linspace(combined_k_min, combined_k_max, TARGET_GRID_RESOLUTION) if combined_k_max > combined_k_min else np.array([combined_k_min])
                 common_t_axis = np.linspace(combined_t_min, combined_t_max, TARGET_GRID_RESOLUTION) if combined_t_max > combined_t_min else np.array([combined_t_min])
                 common_sigma_range = (combined_sigma_min, combined_sigma_max)

                 # --- Plot Combined Surface ---
                 fig_compare = compare_surfaces(
                     options_surface, lgbm_surface, stock, plot_date_ref, option_type,
                     k_axis=common_k_axis, t_axis=common_t_axis, sigma_range=common_sigma_range,
                     show_realized=show_realized_points, realized_points_data=realized_points_data,
                     show_s0=show_hist_rv_line, spot_price=spot_price, spot_line_data=spot_line_data
                 )
                 st.plotly_chart(fig_compare, use_container_width=True)
                 if realized_available:
                     st.caption( f"**Realized Point Colors:** "
                                 f"<span style='color:{LGBM_CLOSER_COLOR}; font-weight:bold;'>● Pink:</span> Closer to LGBM RV. "
                                 f"<span style='color:{OPTIONS_CLOSER_COLOR}; font-weight:bold;'>● Green:</span> Closer to Options IV. "
                                 f"<span style='color:{DEFAULT_RV_COLOR}; font-weight:bold;'>● Orange:</span> Roughly Equidistant/N/A.",
                                 unsafe_allow_html=True )

                 # --- Pricing Difference Calculation ---
                 diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats = None, None, None, None, None, {}
                 try:
                     st.write(f"Calculating pricing differences on common {len(common_k_axis)}x{len(common_t_axis)} grid...")
                     diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common, stats = calculate_pricing_difference_on_axes(
                         options_surface, lgbm_surface, target_k_axis=common_k_axis, target_t_axis=common_t_axis
                     )
                 except Exception as e: st.error(f"Error calculating pricing differences: {e}")

                 # --- Display Comparison Sections ---
                 col1, col2 = st.columns(2)
                 with col1: # Diff Stats & Heatmap
                     if show_stats or show_diff_heatmap:
                         st.markdown("##### Options IV vs LGBM RV Difference (on Common Grid)")
                         if show_stats and stats and np.isfinite(stats.get('mean_diff', np.nan)):
                             stat_cols=st.columns(3)
                             help_max = f"K=${stats.get('max_diff_k', np.nan):.1f}, T={stats.get('max_diff_t', np.nan):.0f}d" if np.isfinite(stats.get('max_diff_k', np.nan)) else "N/A"
                             help_min = f"K=${stats.get('min_diff_k', np.nan):.1f}, T={stats.get('min_diff_t', np.nan):.0f}d" if np.isfinite(stats.get('min_diff_k', np.nan)) else "N/A"
                             help_avg = f"Avg(OptIV-LGBMRV)% over {np.sum(np.isfinite(diff_Z))} pts" if diff_Z is not None else "N/A"
                             stat_cols[0].metric("Max Over", f"{stats.get('max_diff', np.nan):.2f}%", help=help_max)
                             stat_cols[1].metric("Max Under", f"{stats.get('min_diff', np.nan):.2f}%", help=help_min)
                             stat_cols[2].metric("Avg Diff", f"{stats.get('mean_diff', np.nan):.2f}%", help=help_avg)
                         elif show_stats: st.info("Difference statistics could not be calculated.")

                         if show_diff_heatmap and diff_Z is not None and target_K_mesh is not None and target_T_mesh is not None \
                            and opt_Z_common is not None and lgbm_Z_common is not None and stats:
                             try:
                                 heatmap_fig = create_pricing_difference_plot(
                                     diff_Z, target_K_mesh, target_T_mesh, opt_Z_common, lgbm_Z_common,
                                     stats, stock, plot_date_ref, option_type
                                 )
                                 st.plotly_chart(heatmap_fig, use_container_width=True)
                             except Exception as e: st.error(f"Error creating pricing difference heatmap: {e}")
                         elif show_diff_heatmap: st.info("Difference heatmap could not be generated.")

                 with col2: # Surface vs Realized Comparison
                     if show_rv_comparison:
                         st.markdown("##### Surface Accuracy vs Realized Volatility")
                         if realized_available:
                             lgbm_diffs = calculate_point_differences("LGBM RV", lgbm_surface, realized_points_data)
                             options_diffs = calculate_point_differences(f"Options IV ({option_type})", options_surface, realized_points_data)
                             lgbm_avg_abs_diff = lgbm_diffs.get('average_abs_diff', np.nan) if lgbm_diffs else np.nan
                             opts_avg_abs_diff = options_diffs.get('average_abs_diff', np.nan) if options_diffs else np.nan

                             if np.isfinite(lgbm_avg_abs_diff) and np.isfinite(opts_avg_abs_diff):
                                 st.metric("Avg Abs Diff (LGBM RV vs Real)", f"{lgbm_avg_abs_diff:.4f}")
                                 st.metric(f"Avg Abs Diff (Options IV vs Real)", f"{opts_avg_abs_diff:.4f}")
                                 if np.isclose(lgbm_avg_abs_diff, opts_avg_abs_diff): st.info("Surfaces have similar average difference to realized points.")
                                 elif lgbm_avg_abs_diff < opts_avg_abs_diff: st.success("LGBM RV surface is closer to realized points on average.")
                                 else: st.success("Options IV surface is closer to realized points on average.")
                             elif np.isfinite(lgbm_avg_abs_diff): st.metric("Avg Abs Diff (LGBM RV vs Real)", f"{lgbm_avg_abs_diff:.4f}"); st.metric(f"Avg Abs Diff (Options IV vs Real)", "N/A"); st.info("Only LGBM RV vs Realized calculated.")
                             elif np.isfinite(opts_avg_abs_diff): st.metric("Avg Abs Diff (LGBM RV vs Real)", "N/A"); st.metric(f"Avg Abs Diff (Options IV vs Real)", f"{opts_avg_abs_diff:.4f}"); st.info("Only Options IV vs Realized calculated.")
                             else: st.metric("Avg Abs Diff (LGBM RV vs Real)", "N/A"); st.metric(f"Avg Abs Diff (Options IV vs Real)", "N/A"); st.warning("Could not calculate average absolute differences vs realized points.")

                             if lgbm_diffs: display_point_differences("LGBM RV", lgbm_diffs, True)
                             if options_diffs: display_point_differences(f"Options IV ({option_type})", options_diffs, True)
                         else: st.info("Realized points comparison requires 'Show Realized Points' checked and data available.")

            # Handle cases where only one surface was successful in Comparison mode (redundant now?)
            # This logic might be simplified as the error checks above handle the main failure cases.

    except Exception as e:
        st.error(f"An error occurred in the surface view display: {e}")
        logger.error(f"Surface View Error: {e}", exc_info=True)