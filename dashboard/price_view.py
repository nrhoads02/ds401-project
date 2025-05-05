# dashboard/price_view.py
# price_view.py
import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime
import logging
import sys
import os
from typing import Dict, Any, List, Tuple, Union, Optional
from scipy.stats import norm # For Black-Scholes calculation
from scipy.interpolate import griddata, interp1d # For interpolation tasks

# Project-specific imports
try:
    # Import model prediction function and constants
    from src.data_modeling.surface_lgbm_modeling import predict_surface, K_RANGE, K_GRID_POINTS
    # Import function to generate IV surface from options data
    from src.data_modeling.option_chain_modeling import generate_options_surface
    # Import shared helper functions and constants from the sibling surface_view module
    from dashboard.surface_view import extract_overlay_data, interpolate_surface_at_points, _to_date, AXIS_PADDING_FACTOR
except ImportError as e:
    # Display error and stop if imports fail
    st.error(f"Price View: Failed to import modules: {e}")
    st.stop()

# Add project root to Python path for module resolution if necessary
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging for this module
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    # Set basic logging configuration if no handlers are present
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Constants specific to price view
TARGET_GRID_RESOLUTION_PRICE = 50 # Default grid resolution for price surfaces
S0_LINE_COLOR = 'cyan' # Color for the Spot Price line on heatmaps
S0_LINE_DASH = 'dot'   # Dash style for the Spot Price line

# -----------------------------------------------------------------------------
# Black-Scholes-Merton Implementation
# -----------------------------------------------------------------------------

def bsm_price_vectorized(S0, K, T_years, r, q, sigma, option_type):
    """
    Calculates Black-Scholes-Merton option prices using vectorized NumPy operations
    for efficiency. Handles potential calculation errors and invalid inputs gracefully.
    """
    with np.errstate(divide='ignore', invalid='ignore'): # Suppress expected warnings
        d1 = (np.log(S0 / K) + (r - q + 0.5 * sigma**2) * T_years) / (sigma * np.sqrt(T_years))
        d2 = d1 - sigma * np.sqrt(T_years)
        if option_type.lower() == 'call':
            price = (S0 * np.exp(-q * T_years) * norm.cdf(d1)) - (K * np.exp(-r * T_years) * norm.cdf(d2))
        elif option_type.lower() == 'put':
            price = (K * np.exp(-r * T_years) * norm.cdf(-d2)) - (S0 * np.exp(-q * T_years) * norm.cdf(-d1))
        else: raise ValueError("option_type must be 'call' or 'put'")
    price[~np.isfinite(price)] = 0.0 # Set non-finite results to 0
    price = np.maximum(0.0, price) # Ensure non-negative price
    return price

# -----------------------------------------------------------------------------
# Helper Functions for Price View
# -----------------------------------------------------------------------------

def calculate_price_surface(
    vol_surface: Tuple[np.ndarray, np.ndarray, np.ndarray], # (K_mesh, T_mesh, Vol_Z)
    S0: float, r: float, q: float, option_type: str
) -> Optional[np.ndarray]:
    """
    Calculates option prices across a given volatility surface using the BSM model.
    """
    K_mesh, T_mesh_days, vol_Z = vol_surface
    price_Z = np.full_like(vol_Z, np.nan) # Initialize price grid

    if S0 is None or not np.isfinite(S0) or S0 <= 0: return None # Validate S0

    # Mask for valid BSM inputs
    valid_mask = (np.isfinite(K_mesh) & (K_mesh > 0) &
                  np.isfinite(T_mesh_days) & (T_mesh_days > 0) &
                  np.isfinite(vol_Z) & (vol_Z > 0))

    K_valid, T_days_valid, sigma_valid = K_mesh[valid_mask], T_mesh_days[valid_mask], vol_Z[valid_mask]
    T_years_valid = T_days_valid / 365.0 # Convert T to years

    if K_valid.size > 0: # Proceed only if there are valid points
        try:
            prices_valid = bsm_price_vectorized(S0, K_valid, T_years_valid, r, q, sigma_valid, option_type)
            price_Z[valid_mask] = prices_valid # Place results back into grid
        except Exception as e: logger.error(f"BSM calculation failed: {e}"); return None

    price_Z[~valid_mask & np.isfinite(price_Z)] = 0.0 # Handle points outside valid mask

    # Final checks
    if np.all(np.isnan(price_Z)): logger.error("Price surface is all NaNs."); return None
    elif not np.any(np.isfinite(price_Z[valid_mask])): logger.warning("No valid finite prices calculated.")

    return price_Z


def _calculate_dynamic_axis_range(
    S0: float, K_mesh: np.ndarray, T_mesh: np.ndarray,
    realized_points_data: Optional[Tuple], consider_realized_for_axis: bool,
    sim_k_range: Optional[Tuple[float, float]] = None
) -> Tuple[float, float, float, float]:
    """
    Internal helper to calculate dynamic K and T axis ranges for heatmaps.
    Combines simulation range priority with ensuring all original data is shown.
    """
    # --- K-Axis Calculation ---
    k_axis_elements = []
    if S0 and np.isfinite(S0): k_axis_elements.append(S0)
    if consider_realized_for_axis and realized_points_data:
        k_axis_elements.extend([k for k in realized_points_data[0] if np.isfinite(k)])

    k_min_orig, k_max_orig = np.nan, np.nan # Calculate 'original' range based on S0/realized/padding
    if k_axis_elements:
        finite_k = [k for k in k_axis_elements if np.isfinite(k)]
        if finite_k:
            k_min_calc, k_max_calc = np.min(finite_k), np.max(finite_k)
            k_pad = max((k_max_calc - k_min_calc)*AXIS_PADDING_FACTOR/2.0, S0*0.05 if (S0 and np.isfinite(S0)) else 1.0)
            k_min_orig, k_max_orig = max(0.0, k_min_calc - k_pad), k_max_calc + k_pad
        elif S0 and np.isfinite(S0): k_min_orig, k_max_orig = S0 * 0.85, S0 * 1.15

    # Fallback using K_mesh range if needed
    if np.isnan(k_min_orig) or np.isnan(k_max_orig) or k_max_orig <= k_min_orig:
        if np.any(np.isfinite(K_mesh)):
            k_min_mesh, k_max_mesh = np.nanmin(K_mesh), np.nanmax(K_mesh)
            if np.isfinite(k_min_mesh) and np.isfinite(k_max_mesh) and k_max_mesh > k_min_mesh:
                k_pad = max((k_max_mesh-k_min_mesh)*AXIS_PADDING_FACTOR/2.0, 1.0)
                k_min_orig, k_max_orig = max(0.0, k_min_mesh-k_pad), k_max_mesh+k_pad
            else: k_min_orig, k_max_orig = 0, (S0*1.5 if (S0 and np.isfinite(S0)) else 100)
        else: k_min_orig, k_max_orig = 0, (S0*1.5 if (S0 and np.isfinite(S0)) else 100)
        if k_max_orig <= k_min_orig: k_max_orig = k_min_orig + 10

    # --- Determine Final K Render Range, considering simulation range ---
    k_min_render, k_max_render = k_min_orig, k_max_orig
    if sim_k_range is not None:
        sim_k_min, sim_k_max = sim_k_range
        if np.isfinite(sim_k_min) and np.isfinite(sim_k_max):
            # Take the minimum of the original and sim starts, max of the ends
            k_min_render = min(k_min_orig, sim_k_min)
            k_max_render = max(k_max_orig, sim_k_max)
            k_min_render = max(0.0, k_min_render) # Ensure non-negative K
            if k_max_render <= k_min_render: # Ensure positive range width
                k_max_render = k_min_render + max(1.0, S0*0.1 if (S0 and np.isfinite(S0)) else 1.0)

    # --- T-Axis Calculation ---
    t_min_render = np.nanmin(T_mesh) if np.any(np.isfinite(T_mesh)) else 0
    t_max_render = np.nanmax(T_mesh) if np.any(np.isfinite(T_mesh)) else 365
    if t_max_render <= t_min_render: t_max_render = t_min_render + 30

    return k_min_render, k_max_render, t_min_render, t_max_render


def create_price_heatmap(
    price_Z: np.ndarray, K_mesh: np.ndarray, T_mesh: np.ndarray, title: str,
    colorbar_title: str, S0: float, colorscale: str = "Viridis", zmin=None, zmax=None,
    itm_otm_line: Optional[Tuple[np.ndarray, np.ndarray]] = None, option_type: str = 'call',
    realized_points_data: Optional[Tuple] = None, consider_realized_for_axis: bool = False,
    sim_k_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Creates a 2D Plotly heatmap for option prices, with flexible axis scaling and S0 line.
    """
    fig = go.Figure()

    # --- Input Validation ---
    if price_Z is None or K_mesh is None or T_mesh is None or \
       price_Z.size == 0 or K_mesh.size == 0 or T_mesh.size == 0 or \
       price_Z.shape != K_mesh.shape or price_Z.shape != T_mesh.shape:
        logger.warning(f"Cannot create heatmap '{title}': Invalid data/shapes.")
        fig.update_layout(title=f"{title} (Data Error)"); return fig

    # --- Calculate Axis Ranges ---
    k_min_render, k_max_render, t_min_render, t_max_render = _calculate_dynamic_axis_range(
        S0, K_mesh, T_mesh, realized_points_data, consider_realized_for_axis, sim_k_range
    )

    # --- Heatmap Trace ---
    text_labels = np.array([[f"${p:.2f}" if np.isfinite(p) and p > 0.001 else "" for p in row] for row in price_Z])
    fig.add_trace(go.Heatmap(
        z=price_Z, x=K_mesh[0, :], y=T_mesh[:, 0], colorscale=colorscale,
        colorbar=dict(title=colorbar_title, thickness=15), zmin=zmin, zmax=zmax,
        text=text_labels, texttemplate="%{text}", textfont={"size": 12, "color": "white"},
        customdata=np.dstack((K_mesh, T_mesh)),
        hovertemplate=("<b>Price Details</b><br>K: $%{customdata[0]:.2f}<br>T: %{customdata[1]:.0f}d<br>"
                       "Price: $%{z:.4f}<extra></extra>"),
        hoverongaps=False
    ))

    # --- Add Vertical Line for Spot Price (S0) ---
    if S0 and np.isfinite(S0):
        fig.add_trace(go.Scatter(
            x=[S0, S0],  # Constant X value = S0
            y=[t_min_render, t_max_render], # Span the full T-axis range
            mode='lines',
            line=dict(color=S0_LINE_COLOR, width=2, dash=S0_LINE_DASH),
            name=f'Spot Price (S0=${S0:.2f})', # Legend entry
            hoverinfo='skip' # No separate hover for this line
        ))

    # --- Add ITM/OTM Boundary Line Overlay ---
    if itm_otm_line is not None:
        line_T, line_K = itm_otm_line
        if line_T is not None and line_K is not None and len(line_T) > 0 and len(line_K) == len(line_T):
             sorted_indices = np.argsort(line_T); line_T_s, line_K_s = line_T[sorted_indices], line_K[sorted_indices]
             valid_t_mask = (line_T_s >= t_min_render) & (line_T_s <= t_max_render); line_T_p, line_K_p = line_T_s[valid_t_mask], line_K_s[valid_t_mask]
             if len(line_T_p) > 1:
                 fig.add_trace(go.Scatter(x=line_K_p, y=line_T_p, mode='lines', line=dict(color='white', width=3, dash='dash'),
                                          name='Actual ITM/OTM Boundary', hoverinfo='skip'))
                 anno_text = "Right: ITM<br>Left: OTM" if option_type=='call' else "Left: ITM<br>Right: OTM"
                 fig.add_annotation(x=0.95, y=0.05, xref="paper", yref="paper", text=anno_text, showarrow=False,
                                    font=dict(color="white", size=12), bgcolor="rgba(0,0,0,0.5)", bordercolor="white", borderwidth=1)
             else: logger.info("ITM/OTM boundary line not plotted: Not enough points in range.")
        else: logger.warning("ITM/OTM boundary line data invalid.")

    # --- Final Figure Layout ---
    fig.update_layout(title=title, xaxis_title="Strike Price (K, $)", yaxis_title="Days to Expiration (T)",
                      yaxis_autorange='reversed', width=None, height=600,
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.6)')) # Legend background slightly transparent
    fig.update_xaxes(range=[k_min_render, k_max_render]) # Apply K range
    fig.update_yaxes(range=[t_max_render, t_min_render]) # Apply reversed T range

    return fig


def calculate_itm_otm_boundary(
    transformed_df: pl.DataFrame, stock: str, trade_date: datetime.date, S0: float,
    T_axis_days: np.ndarray, horizons: List[int]
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Calculates the strike price boundary between ITM/OTM based on actual future returns.
    """
    if S0 is None or not np.isfinite(S0) or S0 <= 0: return None # Validate S0
    try: # Get data for the specific date
         today_data = transformed_df.filter( (pl.col("act_symbol") == stock) & (pl.col("date") == trade_date) ).head(1)
         if today_data.is_empty(): logger.warning(f"No data for {stock}/{trade_date} for boundary."); return None
         row_dict = today_data.row(0, named=True)
    except Exception as e: logger.error(f"Error getting data for boundary: {e}"); return None

    # Find available, valid future log returns
    available_horizons = []; future_cols = {}
    for h in horizons:
         col_name = f"log_ret_future_{h}"
         if col_name in row_dict:
              val = row_dict[col_name]
              if val is not None and np.isfinite(val): available_horizons.append(h); future_cols[h] = val
    if not available_horizons: logger.warning(f"No valid future log returns found for {stock}/{trade_date}."); return None
    available_horizons.sort()

    # Calculate boundary K for each T
    boundary_T, boundary_K = [], []
    for T_days in T_axis_days:
        if not np.isfinite(T_days) or T_days <= 0: continue
        h_approx = int(round(T_days * 5 / 7))
        if h_approx == 0: continue
        closest_h = min(available_horizons, key=lambda h: abs(h - h_approx))
        lr_fut = future_cols[closest_h]
        try: # Calculate future price S_T = boundary K
             S_T = S0 * np.exp(lr_fut)
             if np.isfinite(S_T) and S_T > 0: boundary_T.append(T_days); boundary_K.append(S_T)
        except Exception: pass # Ignore calculation errors for single points

    if not boundary_T: logger.warning(f"No valid boundary points calculated."); return None
    return np.array(boundary_T), np.array(boundary_K) # Return as numpy arrays


# -----------------------------------------------------------------------------
# Trading Strategy Simulation
# -----------------------------------------------------------------------------
def simulate_trading_strategy(
    lgbm_price_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    market_price_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    itm_otm_boundary_data: Optional[Tuple[np.ndarray, np.ndarray]],
    option_type: str, S0: float, mispricing_threshold: float, strike_padding: float
) -> Optional[Dict[str, Any]]:
    """
    Simulates trading based on model vs market price differences within a strike range.
    Returns statistics and a grid of trade actions.
    """
    # --- Input Validation ---
    if not lgbm_price_surface or not market_price_surface: return None
    if S0 is None or not np.isfinite(S0) or S0 <= 0: return None
    # P&L calculation requires boundary data
    if itm_otm_boundary_data is None: logger.warning("Sim: Boundary data missing, P&L calculation skipped.")

    K_mesh, T_mesh, lgbm_price_Z = lgbm_price_surface
    _, _, market_price_Z = market_price_surface # Assumes same grid
    boundary_T, boundary_K_actual = itm_otm_boundary_data if itm_otm_boundary_data else (None, None)

    if not (K_mesh.shape == T_mesh.shape == lgbm_price_Z.shape == market_price_Z.shape): return None
    if boundary_T is not None and len(boundary_T) != len(boundary_K_actual): return None

    # --- Setup Simulation ---
    total_profit, long_profit, short_profit = 0.0, 0.0, 0.0
    long_wins, short_wins, long_trades, short_trades = 0, 0, 0, 0
    skipped_strike_range, skipped_no_boundary = 0, 0
    trade_action_grid = np.zeros_like(K_mesh, dtype=int) # Grid for trade actions
    min_k_trade, max_k_trade = S0 * (1.0 - strike_padding), S0 * (1.0 + strike_padding) # K filter range

    # --- Create Boundary Interpolation Function ---
    boundary_interp = None
    if boundary_T is not None and len(boundary_T) >= 2:
        sort_idx = np.argsort(boundary_T)
        boundary_T_s, boundary_K_s = boundary_T[sort_idx], boundary_K_actual[sort_idx]
        try: # Use linear interpolation, fill with edge values outside range
            boundary_interp = interp1d(boundary_T_s, boundary_K_s, kind='linear',
                                       bounds_error=False, fill_value=(boundary_K_s[0], boundary_K_s[-1]))
        except Exception as e: logger.error(f"Sim: Boundary interp failed: {e}"); boundary_interp = None
    elif boundary_T is not None: logger.warning("Sim: Not enough boundary points (<2) for P&L interp.")

    # --- Iterate Grid and Simulate ---
    for i in range(K_mesh.shape[0]):
        for j in range(K_mesh.shape[1]):
            K, T = K_mesh[i, j], T_mesh[i, j]
            lgbm_p, mkt_p = lgbm_price_Z[i, j], market_price_Z[i, j]

            # Skip invalid grid points
            if not (np.isfinite(K) and K > 0 and np.isfinite(T) and T > 0 and
                    np.isfinite(lgbm_p) and lgbm_p >= 0 and np.isfinite(mkt_p) and mkt_p >= 0): continue
            # Apply K range filter
            if not (min_k_trade <= K <= max_k_trade): skipped_strike_range += 1; continue

            # Determine Trade Action
            trade_action_num = 0
            if lgbm_p > mkt_p * (1.0 + mispricing_threshold): trade_action_num = 1; long_trades += 1
            elif lgbm_p < mkt_p * (1.0 - mispricing_threshold): trade_action_num = -1; short_trades += 1
            trade_action_grid[i, j] = trade_action_num # Store action in grid

            # Calculate P&L if trade occurred and boundary is usable
            if trade_action_num != 0:
                 actual_K_at_T = np.nan
                 if boundary_interp:
                     try:
                         actual_K_at_T = boundary_interp(T)
                         if not np.isfinite(actual_K_at_T): actual_K_at_T = np.nan; skipped_no_boundary += 1
                     except Exception: actual_K_at_T = np.nan; skipped_no_boundary += 1
                 else: skipped_no_boundary += 1 # Cannot calc P&L if no interpolator

                 if np.isfinite(actual_K_at_T): # Proceed if actual K found
                     intrinsic = max(0.0, actual_K_at_T - K) if option_type=='call' else max(0.0, K - actual_K_at_T)
                     profit = (intrinsic - mkt_p) if trade_action_num == 1 else (mkt_p - intrinsic)
                     total_profit += profit
                     if trade_action_num == 1: long_profit += profit; long_wins += (profit > 0)
                     else: short_profit += profit; short_wins += (profit > 0)

    # --- Compile Results ---
    total_sim_trades = long_trades + short_trades
    results = {
        "total_trades": total_sim_trades, "total_profit": total_profit,
        "avg_profit_per_trade": total_profit / total_sim_trades if total_sim_trades > 0 else 0,
        "long_trades": long_trades, "long_profit": long_profit, "long_wins": long_wins,
        "long_win_rate": long_wins / long_trades if long_trades > 0 else 0,
        "short_trades": short_trades, "short_profit": short_profit, "short_wins": short_wins,
        "short_win_rate": short_wins / short_trades if short_trades > 0 else 0,
        "skipped_strike_range": skipped_strike_range, "skipped_no_boundary": skipped_no_boundary,
        "strike_range_used": (min_k_trade, max_k_trade), "trade_action_grid": trade_action_grid
    }
    return results


def create_trade_heatmap(
    trade_action_grid: np.ndarray, K_mesh: np.ndarray, T_mesh: np.ndarray, title: str, S0: float,
    realized_points_data: Optional[Tuple] = None, consider_realized_for_axis: bool = False,
    sim_k_range: Optional[Tuple[float, float]] = None
) -> go.Figure:
    """
    Creates a 2D Plotly heatmap visualizing simulated trade actions (Buy/Sell/None), with S0 line.
    """
    fig = go.Figure()

    # --- Input Validation ---
    if trade_action_grid is None or K_mesh is None or T_mesh is None or \
       trade_action_grid.shape != K_mesh.shape or trade_action_grid.shape != T_mesh.shape:
        logger.warning(f"Cannot create trade heatmap '{title}': Invalid data/shapes.")
        fig.update_layout(title=f"{title} (Data Error)"); return fig

    # --- Calculate Axis Ranges ---
    k_min_render, k_max_render, t_min_render, t_max_render = _calculate_dynamic_axis_range(
        S0, K_mesh, T_mesh, realized_points_data, consider_realized_for_axis, sim_k_range
    )

    # --- Heatmap Trace ---
    action_text_map = {1: "Buy", -1: "Sell", 0: "No Trade"} # Map numbers to text for hover
    hover_text_labels = np.array([[action_text_map.get(action, "N/A") for action in row] for row in trade_action_grid])
    fig.add_trace(go.Heatmap(
        z=trade_action_grid, x=K_mesh[0, :], y=T_mesh[:, 0],
        colorscale='RdBu', zmid=0, zmin=-1, zmax=1, # Diverging Red(-1)/Blue(+1) scale centered at 0
        colorbar=dict(title='Trade Action', tickvals=[-1, 0, 1], ticktext=['Sell', 'None', 'Buy'], thickness=15),
        customdata=np.dstack((K_mesh, T_mesh, hover_text_labels)),
        hovertemplate=("<b>Trade Simulation</b><br>K: $%{customdata[0]:.2f}<br>T: %{customdata[1]:.0f}d<br>"
                       "Action: %{customdata[2]}<extra></extra>"),
        hoverongaps=False
    ))

    # --- Add Vertical Line for Spot Price (S0) ---
    if S0 and np.isfinite(S0):
        fig.add_trace(go.Scatter(
            x=[S0, S0], y=[t_min_render, t_max_render], mode='lines',
            line=dict(color=S0_LINE_COLOR, width=2, dash=S0_LINE_DASH),
            name=f'Spot Price (S0=${S0:.2f})', hoverinfo='skip'
        ))

    # --- Final Figure Layout ---
    fig.update_layout(title=title, xaxis_title="Strike Price (K, $)", yaxis_title="Days to Expiration (T)",
                      yaxis_autorange='reversed', width=None, height=600,
                      legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99, bgcolor='rgba(255,255,255,0.6)'))
    fig.update_xaxes(range=[k_min_render, k_max_render]) # Apply K range
    fig.update_yaxes(range=[t_max_render, t_min_render]) # Apply reversed T range

    return fig


# -----------------------------------------------------------------------------
# Main Display Function for this View
# -----------------------------------------------------------------------------
def display_price_view(loaded_data: Dict[str, Any], params: Dict[str, Any]):
    """
    Handles the logic and display for the Execution Price view.
    """
    st.header("Execution Price Analysis")

    # --- Unpack Data and Parameters ---
    stock, calculation_date, nearest_options_date = loaded_data["stock"], loaded_data["nearest_ohlcv_date"], loaded_data["nearest_options_date"]
    transformed_df, lgbm_models_dict, horizons = loaded_data["transformed_df"], loaded_data["lgbm_models_dict"], loaded_data["horizons"]
    option_type, r, q = params["option_type"], params["risk_free_rate"], params["dividend_yield"]
    show_lgbm_hm, show_market_hm, show_diff_hm, show_trade_hm = params["show_lgbm_price_heatmap"], params["show_market_price_heatmap"], params["show_price_diff_heatmap"], params["show_trade_heatmap"]
    show_itm_overlay, show_sim_stats = params["show_itm_overlay"], params["show_trading_sim_stats"]
    sim_threshold, sim_strike_padding = params["trading_threshold"], params["strike_padding"]
    st.info(f"Analysis: {calculation_date} (S0, Model RV, Actual Outcome). Market Prices: {nearest_options_date}.")

    # --- Initialize State ---
    lgbm_rv_surface, market_iv_surface, lgbm_price_surface, market_price_surface = None, None, None, None
    spot_price, realized_points_data, itm_otm_boundary_data, sim_results = None, None, None, None
    error_occurred = False
    view_placeholder = st.empty()

    # --- Data Generation and Calculation ---
    try:
        with view_placeholder.status("Processing...", expanded=True): # Status message
            # 1. Get S0 & Realized Points
            st.write("Extracting Spot Price (S0)...")
            today_data = transformed_df.filter((pl.col("act_symbol") == stock) & (pl.col("date") == calculation_date)).head(1) if transformed_df is not None else None
            if today_data is not None and not today_data.is_empty():
                spot_price, realized_points_data, _ = extract_overlay_data(today_data, horizons)
                if not (spot_price and np.isfinite(spot_price) and spot_price > 0): error_occurred = True; st.error(f"Invalid S0: {spot_price}")
                else: st.write(f"Using S0 = ${spot_price:.2f}")
            else: error_occurred = True; st.error(f"No data for {stock}/{calculation_date} for S0.")
            consider_realized = (realized_points_data is not None)

            # 2. Generate LGBM RV Surface
            if not error_occurred: st.write("Generating LGBM RV surface..."); # Continue processing steps...
            if not error_occurred:
                try: lgbm_rv_surface = predict_surface(lgbm_models_dict, transformed_df, stock, calculation_date)
                except Exception as e: st.error(f"LGBM RV Error: {e}"); error_occurred=True; lgbm_rv_surface = None
                if not (lgbm_rv_surface and lgbm_rv_surface[2] is not None and np.any(np.isfinite(lgbm_rv_surface[2]))):
                    st.warning("LGBM RV gen failed/invalid."); error_occurred=True; lgbm_rv_surface = None
                else: st.write("LGBM RV surface generated.")

            # 3. Generate Market IV Surface
            if not error_occurred: st.write("Generating Market IV surface..."); # Continue processing steps...
            if not error_occurred:
                try:
                    raw_opts = loaded_data.get("option_chain_raw", loaded_data.get("option_chain_adj"))
                    market_iv_surface = generate_options_surface(raw_opts, stock, nearest_options_date, "average")
                    if not (market_iv_surface and market_iv_surface[2] is not None and np.any(np.isfinite(market_iv_surface[2]))): market_iv_surface = None
                    else: st.write("Market IV surface generated.")
                except Exception as e: st.warning(f"Market IV Error: {e}"); market_iv_surface = None

            # 4. Calculate LGBM Price Surface
            if not error_occurred and lgbm_rv_surface: st.write("Calculating LGBM prices..."); # Continue processing steps...
            if not error_occurred and lgbm_rv_surface:
                lgbm_price_Z = calculate_price_surface(lgbm_rv_surface, spot_price, r, q, option_type)
                if lgbm_price_Z is not None: lgbm_price_surface = (lgbm_rv_surface[0], lgbm_rv_surface[1], lgbm_price_Z); st.write("LGBM prices calculated.")
                else: st.warning("Failed LGBM price calc."); lgbm_price_surface = None

            # 5. Calculate Market Price Surface (on common grid)
            if not error_occurred and market_iv_surface and lgbm_rv_surface: st.write("Calculating Market prices..."); # Continue processing steps...
            if not error_occurred and market_iv_surface and lgbm_rv_surface:
                lgbm_K, lgbm_T, _ = lgbm_rv_surface; mkt_K, mkt_T, mkt_V = market_iv_surface
                target_K_f, target_T_f = lgbm_K.flatten(), lgbm_T.flatten(); valid_mask = np.isfinite(target_K_f)&np.isfinite(target_T_f)
                if np.any(valid_mask):
                    mkt_V_interp = interpolate_surface_at_points(mkt_K, mkt_T, mkt_V, target_K_f[valid_mask], target_T_f[valid_mask])
                    if mkt_V_interp is not None and np.any(np.isfinite(mkt_V_interp)):
                        mkt_V_grid = np.full(lgbm_K.shape, np.nan); mkt_V_grid[valid_mask.reshape(lgbm_K.shape)] = mkt_V_interp
                        mkt_P_Z = calculate_price_surface((lgbm_K, lgbm_T, mkt_V_grid), spot_price, r, q, option_type)
                        if mkt_P_Z is not None: market_price_surface = (lgbm_K, lgbm_T, mkt_P_Z); st.write("Market prices calculated.")
                        else: st.warning("Failed Market price calc."); market_price_surface = None
                    else: st.warning("Failed Market IV interp."); market_price_surface = None
                else: st.warning("LGBM grid invalid for Market IV interp."); market_price_surface = None

            # 6. Calculate ITM/OTM Boundary
            needs_boundary = show_itm_overlay or show_sim_stats or show_trade_hm
            if not error_occurred and needs_boundary and lgbm_price_surface: st.write("Calculating ITM/OTM boundary..."); # Continue...
            if not error_occurred and needs_boundary and lgbm_price_surface:
                 T_axis = lgbm_price_surface[1][:, 0]
                 itm_otm_boundary_data = calculate_itm_otm_boundary(transformed_df, stock, calculation_date, spot_price, T_axis, horizons)
                 if itm_otm_boundary_data: st.write("Boundary calculated.")
                 else: st.warning("Could not calculate ITM/OTM boundary.")

            # 7. Run Simulation
            needs_simulation = show_sim_stats or show_trade_hm
            if not error_occurred and needs_simulation:
                can_simulate = lgbm_price_surface and market_price_surface and (spot_price and np.isfinite(spot_price)) # Boundary needed only for P&L stats, not heatmap
                if can_simulate:
                    st.write("Running simulation...")
                    try: sim_results = simulate_trading_strategy(lgbm_price_surface, market_price_surface, itm_otm_boundary_data, option_type, spot_price, sim_threshold, sim_strike_padding)
                    except Exception as e: st.error(f"Simulation Error: {e}"); sim_results = None
                    if sim_results: st.write("Simulation complete.")
                    else: st.warning("Simulation failed or returned no results.")
                else: st.warning("Skipping simulation (missing inputs).")

        view_placeholder.empty() # Clear status

        # --- Display Visualizations ---
        if not error_occurred:
            # Z-axis range for price heatmaps
            all_p = []
            if lgbm_price_surface: all_p.append(lgbm_price_surface[2][np.isfinite(lgbm_price_surface[2])])
            if market_price_surface: all_p.append(market_price_surface[2][np.isfinite(market_price_surface[2])])
            zmin_p, zmax_p = 0, 1
            if all_p: flat_p = np.concatenate(all_p);
            if flat_p.size > 0: zmin_p, zmax_p = max(0., np.percentile(flat_p,1)), np.percentile(flat_p,99); # Calc range...
            if zmin_p >= zmax_p: zmin_p, zmax_p = max(0., np.min(flat_p)), np.max(flat_p);
            if zmax_p <= zmin_p: zmax_p = zmin_p + max(0.1, zmin_p*0.1)

            # K-axis range based on simulation padding
            sim_k_range = sim_results.get("strike_range_used") if sim_results else (spot_price * (1 - sim_strike_padding), spot_price * (1 + sim_strike_padding)) if spot_price else None

            # --- Display Heatmaps ---
            title_suffix = f"@ {calculation_date} (S0=${spot_price:.2f})" if spot_price else f"@ {calculation_date}"
            # LGBM Price
            if show_lgbm_hm and lgbm_price_surface:
                st.plotly_chart(create_price_heatmap(lgbm_price_surface[2], lgbm_price_surface[0], lgbm_price_surface[1], f"{stock} LGBM Expected {option_type.capitalize()} Price {title_suffix}",
                    "Model Price ($)", S0=spot_price, zmin=zmin_p, zmax=zmax_p, itm_otm_line=itm_otm_boundary_data if show_itm_overlay else None, option_type=option_type,
                    realized_points_data=realized_points_data, consider_realized_for_axis=consider_realized, sim_k_range=sim_k_range), use_container_width=True)
            elif show_lgbm_hm: st.warning("LGBM Price heatmap cannot be displayed.")
            # Market Price
            if show_market_hm and market_price_surface:
                st.plotly_chart(create_price_heatmap(market_price_surface[2], market_price_surface[0], market_price_surface[1], f"{stock} Market Implied {option_type.capitalize()} Price {title_suffix}",
                    "Market Price ($)", S0=spot_price, colorscale="Viridis", zmin=zmin_p, zmax=zmax_p, itm_otm_line=itm_otm_boundary_data if show_itm_overlay else None, option_type=option_type,
                    realized_points_data=realized_points_data, consider_realized_for_axis=consider_realized, sim_k_range=sim_k_range), use_container_width=True)
            elif show_market_hm: st.warning("Market Price heatmap cannot be displayed.")
            # Price Difference
            if show_diff_hm and lgbm_price_surface and market_price_surface:
                diff_Z = market_price_surface[2] - lgbm_price_surface[2]; valid_d = diff_Z[np.isfinite(diff_Z)]; zmin_d,zmax_d=-1,1
                if valid_d.size > 0: max_abs = np.percentile(np.abs(valid_d), 98); max_abs=max(max_abs,0.1); zmin_d,zmax_d=-max_abs, max_abs
                fig_diff = create_price_heatmap(diff_Z, lgbm_price_surface[0], lgbm_price_surface[1], f"{stock} Price Diff (Market - LGBM) {title_suffix}", "Price Diff ($)", S0=spot_price, colorscale="RdBu_r", zmin=zmin_d, zmax=zmax_d,
                                                 itm_otm_line=itm_otm_boundary_data if show_itm_overlay else None, option_type=option_type, realized_points_data=realized_points_data, consider_realized_for_axis=consider_realized, sim_k_range=sim_k_range)
                diff_txt = np.array([[f"{d:+.2f}" if np.isfinite(d) else "" for d in r] for r in diff_Z]); cdata=np.dstack((lgbm_price_surface[0],lgbm_price_surface[1],market_price_surface[2],lgbm_price_surface[2]))
                fig_diff.update_traces(text=diff_txt, texttemplate="%{text}", textfont={"size":12,"color":"black"}, customdata=cdata, hovertemplate="Diff Details...<br>Mkt:${customdata[2]:.4f}<br>LGBM:${customdata[3]:.4f}<br>Diff:${z:.4f}<extra></extra>")
                st.plotly_chart(fig_diff, use_container_width=True)
            elif show_diff_hm: st.warning("Price Difference heatmap cannot be displayed.")
            # Trade Heatmap
            if show_trade_hm:
                if sim_results and 'trade_action_grid' in sim_results and lgbm_price_surface:
                     st.plotly_chart(create_trade_heatmap(sim_results['trade_action_grid'], lgbm_price_surface[0], lgbm_price_surface[1], f"{stock} Sim Trade Actions ({option_type.capitalize()}) {title_suffix}", S0=spot_price,
                                                       realized_points_data=realized_points_data, consider_realized_for_axis=consider_realized, sim_k_range=sim_k_range), use_container_width=True)
                     k_range = sim_results.get("strike_range_used", (np.nan, np.nan)); st.caption(f"Actions shown for K ≈ [${k_range[0]:.2f}, ${k_range[1]:.2f}$] using {sim_threshold:.1%} threshold.")
                else: st.warning("Trade Action heatmap cannot be displayed (simulation failed or prerequisites missing).")

            # --- Display Simulation Statistics ---
            if show_sim_stats:
                st.subheader("Trading Strategy Simulation Results")
                if sim_results: # Display stats if simulation ran
                    with st.expander("Strategy Explanation & Parameters Used", expanded=False):
                         k_range = sim_results.get("strike_range_used", (np.nan, np.nan)); st.markdown(f"**Params:** K Range≈[${k_range[0]:.2f}, ${k_range[1]:.2f}$], Threshold={sim_threshold:.1%}. P&L requires boundary data.")
                    st.markdown("##### Summary Statistics"); # ... [Display metrics as before] ...
                    if sim_results["total_trades"] > 0:
                           cols = st.columns(3); cols[0].metric("Trades", f"{sim_results['total_trades']:,}"); cols[1].metric("Total P&L ($)", f"{sim_results['total_profit']:,.2f}"); cols[2].metric("Avg P&L ($)", f"{sim_results['avg_profit_per_trade']:,.2f}"); st.divider()
                           st.markdown("<h6>Longs (Model > Mkt)</h6>", unsafe_allow_html=True); cols_l = st.columns(3); cols_l[0].metric("Trades", f"{sim_results['long_trades']:,}"); cols_l[1].metric("P&L ($)", f"{sim_results['long_profit']:,.2f}"); cols_l[2].metric("Win Rate", f"{sim_results['long_win_rate']:.1%}")
                           st.markdown("<h6>Shorts (Model < Mkt)</h6>", unsafe_allow_html=True); cols_s = st.columns(3); cols_s[0].metric("Trades", f"{sim_results['short_trades']:,}"); cols_s[1].metric("P&L ($)", f"{sim_results['short_profit']:,.2f}"); cols_s[2].metric("Win Rate", f"{sim_results['short_win_rate']:.1%}")
                           sk, sb = sim_results.get("skipped_strike_range",0), sim_results.get("skipped_no_boundary",0)
                           if sk > 0: st.caption(f"ℹ️ {sk:,} pts ignored by K range filter.")
                           if sb > 0: st.caption(f"ℹ️ P&L unknown for {sb:,} trades (boundary data issue).")
                    else: st.info("No trades triggered with current filters."); sk = sim_results.get("skipped_strike_range",0);
                    if sk > 0: st.caption(f"ℹ️ {sk:,} pts ignored by K range filter.")

                else: st.warning("Simulation stats unavailable (simulation did not run or failed).")

    # --- Error Handling for Display ---
    except Exception as e:
        st.error(f"An critical error occurred during the price view display: {e}")
        logger.error(f"Price View Display Error: {e}", exc_info=True)