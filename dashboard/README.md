# Volatility & Price Surface Dashboard

This dashboard provides tools for analyzing and visualizing stock volatility surfaces and associated option pricing, based on both market data and machine learning predictions.

## Overview

The application allows users to explore:

1. **Volatility Surfaces:** Compare Implied Volatility (IV) derived from market option prices against future Realized Volatility (RV) predicted by a LightGBM (LGBM) machine learning model. Visualizations are plotted against strike price (K) and time to maturity (T).
2. **Execution Price Analysis:** Translate volatility expectations (both market IV and LGBM RV) into theoretical option prices using the Black-Scholes-Merton (BSM) model. This view helps identify potential mispricings and simulates a simple trading strategy based on these differences.

## Key Concepts

* **Realized Volatility (RV):** The actual, historical volatility observed over a past period. The LGBM model predicts *future* RV.
* **Implied Volatility (IV):** The market's expectation of future volatility, derived from current option prices.
* **Volatility Surface:** A 3D plot showing volatility (IV or RV) as a function of option strike price (K) and time to maturity (T).
* **LGBM Predicted RV:** The dashboard uses pre-trained LightGBM models to predict future Realized Volatility ($\sigma_{LGBM}(K, T)$) based on historical data and calculated features.
* **Black-Scholes-Merton (BSM):** A standard model used to calculate the theoretical price of options based on inputs like stock price, strike price, time, risk-free rate, dividend yield, and volatility.

## Views

### 1. Volatility Surface Analysis

* **Functionality:** Visualizes and compares different volatility surfaces.
* **Modes:**
  * `LGBM Model RV`: Shows the volatility surface predicted by the LGBM model.
  * `Options Chain IV`: Shows the Implied Volatility surface derived from market option prices (Call, Put, or Average).
  * `Compare Surfaces`: Overlays the LGBM RV and Market IV surfaces.
* **Overlays:**
  * `Realized Points`: Plots actual future realized volatility (if available) for comparison against the surfaces. Points are colored based on proximity to either the LGBM or Market IV surface.
  * `Historical RV @ Spot K`: Shows past realized volatility at the current spot price for context.
  * `Difference Heatmap/Stats`: Quantifies the percentage difference between Market IV and LGBM RV in compare mode.

### 2. Execution Price Analysis

* **Functionality:** Analyzes theoretical option prices based on different volatility inputs and simulates trades.
* **Surfaces:**
  * `LGBM Expected Price`: Theoretical option prices calculated using BSM with LGBM Predicted RV.
  * `Market Implied Price`: Theoretical option prices calculated using BSM with Market Implied Volatility.
* **Visualizations:**
  * `Price Difference Heatmap`: Shows the dollar difference (Market Price - LGBM Price).
  * `Executed Trade Heatmap`: Visualizes where the simulation triggered Buy/Sell trades based on the price difference exceeding a threshold.
* **Overlays & Simulation:**
  * `Actual ITM/OTM Boundary`: Shows the final stock price relative to strike prices, indicating actual option payout.
  * `Trading Strategy Simulation`: Evaluates hypothetical trades based on model vs. market price differences within a defined strike range and threshold, calculating potential P&L based on actual outcomes.

## How to Use

1. **Select View:** Choose between "Volatility Surface" or "Execution Price" analysis in the sidebar.
2. **Set Parameters:**
    * Enter a `Stock Symbol` (e.g., AAPL).
    * Select a `Trade Date`. The application will use the nearest available data date.
    * Configure view-specific settings (e.g., Option Type for IV/Pricing, Risk-Free Rate, Display Options, Simulation Thresholds) in the sidebar.
3. **Generate:** Click "Generate Visualization".
4. **Analyze:** Interact with the generated plots and tables. Use the "About" tab for detailed explanations of concepts and terms.

## Underlying Data and Models

* **Data Sources:** Primarily financial data sourced from Dolt databases (`stocks` and `options`) and CBOE index files. Data is processed into Parquet format for efficiency.
* **Transformations:** Includes adjustments for stock splits, calculation of numerous technical indicators (see `src/data_transformation/INDICATORS.md`), and joining of CBOE index data. This is managed by the `transformation_pipeline.py`.
* **Modeling:** Uses pre-trained LightGBM models (`.lgb.z` files) to predict realized volatility surfaces. Models are loaded selectively based on the chosen stock symbol using functions in `surface_lgbm_modeling.py`. Options data is processed using `option_chain_modeling.py`.

*Note: All strike prices (K) displayed are split-adjusted for historical comparability.*
