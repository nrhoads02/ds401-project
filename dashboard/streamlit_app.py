# dashboard/streamlit_app.py
# streamlit_app.py
import os
import sys
import streamlit as st
import datetime
import logging
import numpy as np
import polars as pl

# Add project root to path for module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- Call set_page_config() as the first Streamlit command ---
st.set_page_config(page_title="Volatility & Price App", layout="wide")

# Configure logger
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# --- Project-specific imports ---
try:
    # Import necessary functions from other project modules
    from src.data_transformation.transformation_pipeline import transformation_pipeline
    from src.data_transformation.stock_adjustments import adjust_option_splits
    from src.data_extraction.dataframe_loader import load_data, get_available_symbols
    from src.data_modeling.surface_lgbm_modeling import load_surface_models, HORIZONS, K_RANGE, K_GRID_POINTS
    from src.data_modeling.vol_surface_visualization import find_nearest_date, _to_date # Date helpers
    from src.data_modeling.option_chain_modeling import generate_options_surface # For market IV surface
    from dashboard import price_view, surface_view # Import view modules
except ImportError as e:
    # Display error and stop if imports fail
    st.error(f"Failed to import project modules. Ensure the 'src' directory is in the correct path relative to streamlit_app.py and all dependencies are installed. Error: {e}")
    st.stop()


# -----------------------------------------------------------------------------
# About text (Revised - No Math Mode for Variables)
# -----------------------------------------------------------------------------
ABOUT_TEXT = r"""
### Core Concepts: Options and Volatility

An **option** is a financial contract that gives the buyer the *right*, but not the obligation, to either buy (call option) or sell (put option) an underlying asset (like a stock) at a specified price (the **strike price**, K) on or before a certain date (the **expiration date**, T).

The price of an option is heavily influenced by **volatility**, which measures the magnitude of the underlying asset's price fluctuations. Understanding volatility is crucial for option pricing and trading. There are two main types relevant here:

1.  **Realized Volatility (RV):** This is the *actual*, historical volatility observed over a specific past period. It's a measure of how much the stock price *actually* moved.
2.  **Implied Volatility (IV):** This is the market's *expectation* of future volatility over the life of the option. It's not directly observed but is *implied* by the current market prices of options. High IV suggests the market expects large price swings, making options more expensive, while low IV suggests smaller expected swings.

### LGBM Predicted RV

This application utilizes machine learning models (specifically LightGBM) trained on historical market data. These models aim to **predict future Realized Volatility**, conditional on the underlying stock price finishing at a specific strike price (K) after a certain time (T, in days).

We denote this model prediction as $\sigma_{LGBM}(K, T)$. It represents the model's best estimate of what the actual volatility *will be* over the period T, given the stock ends at price K.

---

### View 1: Volatility Surface Analysis

This view focuses on visualizing and comparing different volatility surfaces in 3D space, plotted against Strike Price (K) and Time to Maturity (T).

* **LGBM Model RV Surface:** Shows the $\sigma_{LGBM}(K, T)$ predicted by our machine learning model. This represents the model's view of future volatility across different strikes and expirations.
* **Options Chain IV Surface:** Shows the Implied Volatility (IV) derived from current market option prices (Calls, Puts, or an Average). This represents the *market's consensus* view of future volatility.
* **Compare Surfaces:** Allows overlaying the LGBM RV and Market IV surfaces for direct comparison. Differences highlight where the model's volatility prediction diverges from the market's expectation.

**Overlays & Comparisons:**

* **Realized Points:** Shows the *actual* future realized volatility calculated from the stock's subsequent price movement (if data is available). These points serve as a ground truth to evaluate the accuracy of both the model and the market. Points are colored based on which surface (LGBM or Market IV) they were closer to.
* **Historical RV @ Spot K:** Shows a line representing the *past* realized volatility calculated up to the selected date, specifically at the current spot price (S0). This provides historical context.
* **Difference Heatmap/Stats (Compare Mode):** Quantifies the percentage difference between the Market IV and LGBM RV surfaces, highlighting areas of maximum divergence.

---

### View 2: Execution Price Analysis

This view translates volatility expectations into theoretical option prices using the **Black-Scholes-Merton (BSM) model**. It helps assess potential mispricings relative to the market and actual outcomes.

* **LGBM Expected Price Surface:** Calculates theoretical option prices using the BSM model fed with the **LGBM Predicted RV ($\sigma_{LGBM}$)**. This shows what the option *should* be worth according to our model's volatility forecast.
* **Market Implied Price Surface:** Calculates theoretical option prices using the BSM model fed with the **Market Implied Volatility (IV)**. This effectively reconstructs the market's pricing surface based on its own volatility expectations.
* **Price Difference Heatmap:** Shows the dollar difference (Market Price - LGBM Price) across the K-T grid. Positive values mean the market price is higher than the model predicts (potential sell opportunity based on the model); negative values mean the market price is lower (potential buy opportunity based on the model).
* **Executed Trade Heatmap:** Visualizes where the simulation (described below) would have triggered a Buy (Blue) or Sell (Red) trade based on the price difference exceeding the defined threshold within the allowed strike range.

**Overlays & Simulation:**

* **Actual ITM/OTM Boundary:** A dashed line showing the final stock price achieved for each expiration time (T). This indicates whether options at different strikes (K) actually finished In-The-Money (profitable exercise) or Out-of-The-Money (worthless exercise) based on reality.
    * *For Calls:* Right of the line is ITM ($ Final Price > K $).
    * *For Puts:* Left of the line is ITM ($ Final Price < K $).
* **Trading Strategy Simulation:** Evaluates a simple, hypothetical trading strategy:
    * It considers trades only within a specified **strike range** around the spot price.
    * It triggers a **Buy** if the LGBM Price is significantly *higher* than the Market Price (by a chosen **threshold**), suggesting the market undervalues the option according to the model.
    * It triggers a **Sell** if the LGBM Price is significantly *lower* than the Market Price, suggesting the market overvalues the option.
    * The simulation then calculates the Profit & Loss (P&L) of these hypothetical trades based on the *actual* option payout determined by the ITM/OTM boundary line. This provides a historical perspective on whether acting on the model's perceived mispricings (within the set filters) would have been profitable. *Note: This ignores transaction costs and other real-world factors.*

**Key Adjustment:** All strike prices (K) displayed in plots and tables are **split-adjusted** to ensure comparability across different dates and historical stock splits.
"""

# -----------------------------------------------------------------------------
# Main App Logic
# -----------------------------------------------------------------------------
def main():
    # Configure tabs: Main visualizer and the About section
    viz_tab, about_tab = st.tabs(["Visualizer", "About"])
    with about_tab:
        # Display the descriptive About text
        st.markdown(ABOUT_TEXT)
    with viz_tab:
        # Main application title and introduction
        st.title("Volatility & Price Surface Visualizer")
        st.markdown("Analyze LGBM modeled surfaces, market option surfaces, and expected pricing.")
        st.caption("Note: All strike prices (K) are split-adjusted.")

        # --- Sidebar Inputs ---
        st.sidebar.header("Parameters")

        # Load available stock symbols
        symbols = ["AAPL", "MSFT", "GOOGL", "NVDA", "AMD", "TSLA"] # Default list
        try:
            # Attempt to load symbols dynamically from data source
            available_symbols = get_available_symbols("ohlcv")
            if available_symbols:
                 symbols = available_symbols
            else:
                 st.sidebar.warning("Could not load symbols dynamically, using default list.")
        except Exception as e:
            st.sidebar.warning(f"Error loading symbols: {e}. Using default list.")

        # Stock selection dropdown
        default_symbol = "AAPL"
        # Ensure default is in the list, otherwise use the first available symbol
        if default_symbol not in symbols and symbols: default_symbol = symbols[0]
        elif not symbols: default_symbol = "AAPL"; symbols=["AAPL"] # Absolute fallback
        # Find index safely
        stock_idx = symbols.index(default_symbol) if default_symbol in symbols else 0
        stock = st.sidebar.selectbox("Stock Symbol", symbols, index=stock_idx)

        # Date selection input
        default_date = datetime.date(2024, 1, 2)
        trade_date = st.sidebar.date_input("Trade Date", default_date)

        # Main view selection (Volatility or Price)
        selected_view = st.sidebar.radio("Select View", ["Volatility Surface", "Execution Price"], index=0, horizontal=True)

        # Header for view-specific settings
        st.sidebar.header("View Settings")

        # Initialize parameter dictionaries for each view
        surface_view_params = {}
        price_view_params = {}

        # --- Settings specific to Volatility Surface View ---
        if selected_view == "Volatility Surface":
            # Select type of surface(s) to display
            vis_mode = st.sidebar.radio("Surface Type", ["LGBM Model RV", "Options Chain IV", "Compare Surfaces"], index=2, key="vis_mode_radio", horizontal=True)
            option_type_iv = "Average" # Default IV type for volatility view
            # Only show IV type selection if Market IV is being displayed
            if vis_mode != "LGBM Model RV":
                option_type_iv = st.sidebar.radio("Option Type (for IV)", ["Call", "Put", "Average"], index=2, horizontal=True)

            # Expander for display options within the volatility view
            with st.sidebar.expander("Display Options (Volatility)", expanded=True):
                show_realized_points = st.checkbox("Show Realized Points", value=True, help="Show actual future realized volatility points (if available).")
                show_hist_rv_line = st.checkbox("Show Historical RV @ Spot K", value=True, help="Show historical realized volatility at the current spot price (S0).")
                # Comparison options are only relevant in "Compare Surfaces" mode
                show_diff_heatmap_vol = False
                show_stats_vol = False
                show_rv_comparison = False
                if vis_mode == "Compare Surfaces":
                    show_diff_heatmap_vol = st.checkbox("Show IV vs RV Difference Heatmap", value=True, help="Show a 2D heatmap of the percentage difference between Options IV and LGBM RV.")
                    show_stats_vol = st.checkbox("Show IV vs RV Difference Stats", value=True, help="Display summary statistics of the difference between the surfaces.")
                    show_rv_comparison = st.checkbox("Show Surface vs Realized Accuracy", value=True, help="Compare how close each surface is to the realized volatility points.")

            # Package Volatility View settings into its dictionary
            surface_view_params = {
                "vis_mode": vis_mode,
                "option_type": option_type_iv,
                "show_realized_points": show_realized_points,
                "show_hist_rv_line": show_hist_rv_line,
                "show_diff_heatmap": show_diff_heatmap_vol,
                "show_stats": show_stats_vol,
                "show_rv_comparison": show_rv_comparison
            }

        # --- Settings specific to Execution Price View ---
        elif selected_view == "Execution Price":
            # Option type selection for BSM pricing (Calls or Puts)
            option_type_price = st.sidebar.radio("Option Type (for Pricing)", ["Call", "Put"], index=0, horizontal=True)
            # Input parameters for the BSM model
            risk_free_rate = st.sidebar.number_input("Risk-Free Rate (Annualized)", min_value=0.0, max_value=0.5, value=0.02, step=0.005, format="%.3f")
            dividend_yield = st.sidebar.number_input("Dividend Yield (Annualized)", min_value=0.0, max_value=0.5, value=0.00, step=0.005, format="%.3f")

            # Expander for display options within the price view
            with st.sidebar.expander("Display Options (Pricing)", expanded=True):
                 # Toggles for various price-related heatmaps
                 show_lgbm_price_heatmap = st.checkbox("Show LGBM Expected Price Heatmap", value=True)
                 show_market_price_heatmap = st.checkbox("Show Market Implied Price Heatmap", value=True)
                 show_price_diff_heatmap = st.checkbox("Show Price Difference Heatmap", value=True)
                 show_trade_heatmap = st.checkbox("Show Executed Trade Heatmap", value=True)
                 # Toggle for the actual outcome overlay
                 show_itm_overlay = st.checkbox("Show Actual ITM/OTM Overlay", value=True, help="Overlay the boundary based on actual future price.")
                 # Separator before simulation-specific controls
                 st.divider()
                 # Toggle and parameters for the trading simulation
                 show_trading_sim_stats = st.checkbox("Show Trading Strategy Simulation Stats", value=True, help="Simulate trading based on model vs market price difference and show results.")
                 sim_threshold = st.slider(
                     "Sim: Mispricing Threshold (%)",
                     min_value=0.0, max_value=25.0, value=10.0, step=0.5, format="%.1f%%",
                     help="Minimum % difference between Model and Market price to trigger a simulated trade."
                 ) / 100.0 # Convert percentage to decimal
                 sim_k_padding = st.slider(
                     "Sim: Strike Range Padding (%) around S0",
                     min_value=0.0, max_value=50.0, value=10.0, step=1.0, format="%.0f%%",
                     help="Simulate trades only for strikes within this % range above/below the Spot Price (S0)."
                 ) / 100.0 # Convert percentage to decimal

            # Package Price View settings into its dictionary
            price_view_params = {
                "option_type": option_type_price,
                "risk_free_rate": risk_free_rate,
                "dividend_yield": dividend_yield,
                "show_lgbm_price_heatmap": show_lgbm_price_heatmap,
                "show_market_price_heatmap": show_market_price_heatmap,
                "show_price_diff_heatmap": show_price_diff_heatmap,
                "show_trade_heatmap": show_trade_heatmap, # Pass new toggle
                "show_itm_overlay": show_itm_overlay,
                "show_trading_sim_stats": show_trading_sim_stats,
                # Pass the simulation parameters
                "trading_threshold": sim_threshold,
                "strike_padding": sim_k_padding
            }

        # Main button to trigger the data loading and visualization generation
        generate = st.sidebar.button("Generate Visualization", key="generate_button", type="primary", use_container_width=True)

        # --- Main Content Area Logic ---
        if generate:
            # Initialize variables for data loading state
            ohlcv_df, option_chain_raw, option_chain_adj = None, None, None
            transformed_df = None
            lgbm_models_dict = None
            error_occurred = False # Flag to track if any critical step fails
            loading_placeholder = st.empty() # Placeholder for status messages during loading
            nearest_ohlcv_date = None
            nearest_options_date = None

            try:
                # --- Step 1: Load and Prepare Core Data ---
                # Use a status indicator during data loading
                with loading_placeholder.status(f"Processing {stock} near {trade_date}...", expanded=True):
                    # Load historical OHLCV data
                    st.write("Loading OHLCV Data...")
                    try:
                        ohlcv_df = load_data("ohlcv", stock)
                        if ohlcv_df is None or ohlcv_df.is_empty(): raise ValueError("OHLCV data is empty or failed to load.")
                        # Find the nearest date with available OHLCV data
                        nearest_ohlcv_date = find_nearest_date(ohlcv_df, trade_date, stock)
                        if nearest_ohlcv_date is None: raise ValueError(f"No suitable OHLCV date found near {trade_date}.")
                        st.write(f"Found OHLCV data for date: {nearest_ohlcv_date}")
                    except Exception as e:
                        st.error(f"Failed to load or find OHLCV data: {e}")
                        error_occurred = True # Critical failure

                    # Load options chain data if OHLCV loaded successfully
                    if not error_occurred:
                        st.write("Loading Options Chain Data...")
                        try:
                            option_chain_raw = load_data("options", stock)
                            if option_chain_raw is None or option_chain_raw.is_empty(): raise ValueError("Options chain data is empty or failed to load.")
                            # Find the nearest date with available options data
                            nearest_options_date = find_nearest_date(option_chain_raw, trade_date, stock)
                            if nearest_options_date is None: raise ValueError(f"No suitable options date found near {trade_date}.")
                            st.write(f"Found Options data for date: {nearest_options_date}")

                            # Ensure date columns have the correct Polars Date type before adjustment
                            for col, dtype in [('date', pl.Date), ('expiration', pl.Date)]:
                                 if col in option_chain_raw.columns and option_chain_raw[col].dtype != dtype:
                                      try: option_chain_raw = option_chain_raw.with_columns(pl.col(col).cast(dtype))
                                      except Exception as cast_e: raise ValueError(f"Failed to cast options column '{col}' to {dtype}: {cast_e}") from cast_e

                            # Adjust option strike prices for historical stock splits
                            st.write("Adjusting Options for Splits...")
                            option_chain_adj = adjust_option_splits(option_chain_raw)
                            if option_chain_adj is None or option_chain_adj.is_empty(): raise ValueError("Options data became empty after split adjustment.")
                            st.write("Options split adjustment complete.")
                        except Exception as e:
                            st.error(f"Failed to load or process Options data: {e}. Cannot proceed.")
                            error_occurred = True # Critical failure

                    # Load pre-trained LGBM models if previous steps succeeded
                    if not error_occurred:
                         st.write("Loading LGBM Models...")
                         try:
                            # Load models specific to the selected stock
                            lgbm_models_dict = load_surface_models(stock_symbol=stock)
                            # Basic validation of the loaded models dictionary
                            if not lgbm_models_dict or 'models' not in lgbm_models_dict or not lgbm_models_dict['models']:
                                raise ValueError("LGBM models dictionary is invalid or empty.")
                            st.write("LGBM Models loaded.")
                         except Exception as e:
                             st.error(f"Failed to load LGBM Models: {e}. Cannot proceed.")
                             error_occurred = True # Critical failure

                    # Transform OHLCV data (create features) if previous steps succeeded
                    if not error_occurred and ohlcv_df is not None:
                        st.write("Transforming OHLCV Data (Calculating Features)...")
                        try:
                            # Ensure date column is correct type before transformation
                            if 'date' in ohlcv_df.columns and ohlcv_df['date'].dtype != pl.Date:
                                 try: ohlcv_df = ohlcv_df.with_columns(pl.col('date').cast(pl.Date))
                                 except Exception as cast_e: raise ValueError(f"Failed to cast OHLCV 'date' column before transform: {cast_e}") from cast_e

                            # Run the transformation pipeline, calculating future columns needed for Price View
                            transformed_df = transformation_pipeline(ohlcv_df, calculate_future_cols=True)
                            if transformed_df is None or transformed_df.is_empty(): raise ValueError("Data transformation returned empty DataFrame.")
                            st.write("Data transformation complete.")
                        except Exception as e:
                            st.error(f"Data transformation failed: {e}")
                            error_occurred = True # Critical failure
                            transformed_df = None

                loading_placeholder.empty() # Remove the status message

                # Final check for essential dates before proceeding to visualization
                if not error_occurred:
                    date_msgs = []
                    if nearest_ohlcv_date: date_msgs.append(f"OHLCV/Features: `{nearest_ohlcv_date}`")
                    if nearest_options_date: date_msgs.append(f"Options: `{nearest_options_date}`")
                    if date_msgs: st.info(f"Using nearest available data dates: {', '.join(date_msgs)}")

                    # Price view requires both dates; other views require their respective dates
                    if selected_view == "Execution Price" and (not nearest_ohlcv_date or not nearest_options_date):
                         st.error("Could not find required dates for both OHLCV and Options data. Price View requires both.")
                         error_occurred = True
                    elif not nearest_ohlcv_date and selected_view != "Options Chain IV":
                         st.error("Could not find required date for OHLCV/Features data.")
                         error_occurred = True
                    elif not nearest_options_date and selected_view != "LGBM Model RV":
                         st.error("Could not find required date for Options data.")
                         error_occurred = True

                # --- Step 2: Call the Appropriate View Function ---
                if not error_occurred:
                    # Package all successfully loaded and processed data
                    loaded_data = {
                        "stock": stock,
                        "trade_date": trade_date, # Original user request
                        "nearest_ohlcv_date": nearest_ohlcv_date, # Date for features/LGBM
                        "nearest_options_date": nearest_options_date, # Date for market IV/prices
                        "transformed_df": transformed_df, # Data with features
                        "option_chain_adj": option_chain_adj, # Split-adjusted options
                        "option_chain_raw": option_chain_raw, # Raw options (if needed)
                        "lgbm_models_dict": lgbm_models_dict, # Models
                        "horizons": lgbm_models_dict.get("horizons", HORIZONS) # Model horizons
                    }

                    # Call the display function corresponding to the selected view
                    if selected_view == "Volatility Surface":
                        surface_view.display_surface_view(loaded_data, surface_view_params)
                    elif selected_view == "Execution Price":
                        price_view.display_price_view(loaded_data, price_view_params)

            # --- General Error Handling for Data Loading/Setup Phase ---
            except FileNotFoundError as fnf:
                st.error(f"File Not Found Error: {fnf}. Please check data paths and ensure parquet/model files exist.")
                logger.error(f"FNF Error: {fnf}", exc_info=True)
            except ImportError as ime:
                st.error(f"Import Error: {ime}. Check project structure and dependencies.")
                logger.error(f"Import Error: {ime}", exc_info=True)
            except ValueError as ve:
                # Often indicates data integrity issues (missing columns, wrong types, etc.)
                st.error(f"Data Error: {ve}. Check input data validity or selection.")
                logger.error(f"Value Error: {ve}", exc_info=True)
            except MemoryError:
                st.error("Memory Error: The application ran out of memory. Try reducing the date range or using a less data-intensive stock.")
                logger.error("Memory Error occurred", exc_info=True)
            except Exception as e:
                # Catch-all for any other unexpected errors during setup
                st.error(f"An unexpected error occurred during data loading or setup: {e}")
                logger.error(f"Unexpected Setup Error: {e}", exc_info=True)
        else:
             # Displayed initially before the user clicks "Generate Visualization"
            st.info("Select parameters in the sidebar and click 'Generate Visualization' to view the results.")

# Standard Python entry point check
if __name__ == "__main__":
    main()