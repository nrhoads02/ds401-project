# DS 4010 Project Report: Volatility Surface Modeling

**Team:** DERN (Ryan Freidhoff, Nic Rhoads, Dakota Rossi, Emiliano Saucedo)

**Dashboard Link:** [Volatility Dashboard](https://ds401-dern-volatility-dashboard.streamlit.app/)

**Repository Link:** [GitHub Repository](https://github.com/nrhoads02/ds401-project)

## 1. Application Goal

### 1.1. Project Objective

The primary goal of this project was to develop a tool for analyzing and visualizing stock volatility. Specifically, we aimed to:

1. Model future **Realized Volatility (RV)**, which is the actual, observed volatility of a stock over a future period, conditional on the stock price ending at a specific strike price (K) after a certain time (T, in days). We denote this model prediction as $\sigma_{LGBM}(K, T)$.
2. Compare this model-predicted RV against **Implied Volatility (IV)**, which is the market's expectation of future volatility derived from current option prices.
3. Visualize these volatilities as 3D surfaces plotted against strike price (K) and time to maturity (T).
4. Extend the analysis to theoretical option pricing using the Black-Scholes-Merton (BSM) model, comparing model-based prices to market-implied prices to identify potential discrepancies.
5. Present these analyses in an interactive dashboard suitable for educational and exploratory purposes.

### 1.2. Motivation and Target Audience

Volatility is a cornerstone concept in finance, particularly in options trading, risk management, and portfolio construction. Understanding how volatility behaves across different strike prices and time horizons (the "volatility surface") provides insights crucial to applications like trading with derivatives.

Our target audience includes fellow Data Science students, finance students, researchers, and potentially retail investors interested in:

* Learning about different types of volatility (Realized vs. Implied).
* Exploring how machine learning models can be applied to predict financial metrics like volatility.
* Visualizing the complex relationship between volatility, strike price, and time.
* Understanding the connection between volatility surfaces and option pricing.
* Seeing a practical example of building a data pipeline and dashboard for financial analysis, especially handling large datasets.

## 2. Data Pipeline (The Backend)

Detailed financial data can use massive amounts of resources like storage and memory, so ensuring that we have a robust, efficient, and scalable data pipeline is essential.

### 2.1. Data Collection (ETL)

#### 2.1.1. Data Sources

Our project utilized several data sources:

1. **DoltHub Repositories (Initial Source):** We initially sourced our primary stock and options data from public DoltHub repositories. DoltHub functions similarly to GitHub but is optimized for versioning large datasets.
    * `stocks` database: Contained daily Open, High, Low, Close, and Volume (OHLCV) data, plus stock split and dividend information dating back potentially to 2011 for over 2000 equities. (See [`data/raw/stocks`](data/raw/stocks) for structure details).
    * `options` database: Contained option chain data (quotes, greeks like delta, implied volatility) and historical volatility metrics, generally reported weekly since 2019. (See [`data/raw/options`](data/raw/options)).
2. **CBOE Index Data (Auxiliary Source):** We downloaded historical data for various CBOE volatility indices (e.g., VIX, VVIX) directly as CSV files from the CBOE website, stored in [`data/raw/cboe`](data/raw/cboe).
3. **Processed Parquet Files (Final Source for Application):** Due to performance and deployment constraints with accessing DoltHub directly from the dashboard, we processed the core data (OHLCV, options, splits) into partitioned Apache Parquet files stored within our GitHub repository. This became the primary source for the Streamlit application. See [`data/parquet/`](data/parquet/) for details.

#### 2.1.2. Data Acquisition

* **Dolt:** Initially, data was cloned from DoltHub using the Dolt command-line interface. Python scripts (like [`src/data_extraction/dolt_csv_export.py`](src/data_extraction/dolt_csv_export.py)) were used to export specific tables to local CSV files for easier manipulation.
* **CBOE:** CSV files were manually downloaded from the CBOE website.
* **Application Loading:** The final dashboard uses a custom Python module ([`src/data_extraction/dataframe_loader.py`](src/data_extraction/dataframe_loader.py)) to load data efficiently from the partitioned Parquet files stored in the repository.

#### 2.1.3. Data Cleaning and Preprocessing

A couple of initial cleaning steps were necessary:

1. **Duplicate Split Removal:** The raw split data contained duplicate entries for the same split event. We implemented logic to identify and keep only the most recent record for each unique split event, addressing potential inconsistencies, found within [`src/data_transformation/stock_adjustments.py`](src/data_transformation/stock_adjustments.py).
2. **Handling Incomplete Tickers:** Some stocks had incomplete histories within our timeframe. We filtered the dataset to only include tickers present on both the first and last dates of the dataset and having no missing OHLCV values in between (logic also in `stock_adjustments.py`). This ensured data consistency for time-series analysis but reduced the number of available stocks.

#### 2.1.4. Data Transformation

The core transformation involved enriching the cleaned OHLCV data with features relevant for volatility modeling:

1. **Split Adjustments:** Historical OHLCV prices and volumes, as well as option strike prices, were adjusted for stock splits using the cleaned split data. A reverse cumulative adjustment factor was calculated for each stock to ensure comparability over time ([`src/data_transformation/stock_adjustments.py`](src/data_transformation/stock_adjustments.py)).
2. **Technical Indicator Generation:** A comprehensive set of technical indicators was calculated using the Polars library for efficiency. This included moving averages, standard deviation, RSI, ATR, realized variance measures, volume-based indicators, and more. Full details and formulas can be found in [`src/data_transformation/INDICATORS.md`](src/data_transformation/INDICATORS.md).
3. **Target Variable Generation:** For modeling, we calculated future realized volatility (`rv_hd_future`) and future log returns (`log_ret_future_h`) by shifting the calculated `rv_hd` and rolling log returns forward by `h` days. These served as the precursors to the model's target variable ($\sigma$) and the conditional variable (log-moneyness, $k$).
4. **CBOE Index Joining:** The CBOE index data was left joined to the main stock dataframe based on the `date` column using the script [`src/data_transformation/cboe_index_join.py`](src/data_transformation/cboe_index_join.py).
5. **Orchestration:** These steps were combined into a single pipeline function ([`src/data_transformation/transformation_pipeline.py`](src/data_transformation/transformation_pipeline.py)) that leverages Polars' lazy evaluation and streaming capabilities to handle the large dataframes efficiently.

#### 2.1.5. Data Storage for Dashboard

* **Initial Challenge:** Our raw datasets, especially after adding indicators, were several gigabytes in size. Storing these large CSVs directly in GitHub was impractical, and accessing them via DoltHub within the Streamlit Cloud's resource limits proved too slow and unreliable.
* **Solution: Partitioned Parquet:** We converted the essential datasets (OHLCV, adjusted options, splits) into the Apache Parquet format using [`src/data_extraction/parquet_converter.py`](src/data_extraction/parquet_converter.py). Parquet offers efficient compression and columnar storage, significantly reducing file size. Crucially, we partitioned these files by stock symbol (`act_symbol`) using a fixed number of partition files per dataset.
* **Efficient Loading:** We created metadata files (`partition_metadata.json`) mapping symbols to partition files. Our data loader ([`src/data_extraction/dataframe_loader.py`](src/data_extraction/dataframe_loader.py)) uses this metadata to load *only* the necessary partition files for the user-selected stock(s), drastically reducing memory usage and load times in the Streamlit app compared to loading the entire dataset. This approach allowed us to store the data within the GitHub repository limits and lets us maintain fast dashboard performance.

### 2.2. Modeling

#### 2.2.1. Modeling Goal and Purpose

The core modeling effort aimed to **predict future realized volatility surfaces**. The purpose was predictive: given the market conditions and technical indicators up to a specific trade date, what is the expected realized volatility ($\sigma$) over a future horizon ($T$), conditional on the stock price ending at a specific strike ($K$) relative to the starting price ($S_0$)?

This predicted surface ( $\sigma_{LGBM}(K, T)$ ) serves multiple purposes in the application:

1. **Direct Visualization:** Allows users to see the model's expectation of future volatility structure.
2. **Comparison with Market IV:** Highlighting differences between the model's prediction and the market's expectation (Implied Volatility) can suggest potential mispricings or areas where the model disagrees with market consensus.
3. **Input for Pricing:** The predicted volatility is fed into the Black-Scholes-Merton model to generate theoretical option prices based on the model's forecast.

#### 2.2.2. Investigated Approaches (Not Used)

* **GARCH Models:** Generalized Autoregressive Conditional Heteroskedasticity models are standard for modeling volatility time series. We researched GARCH but determined it would be too computationally expensive to fit potentially thousands of models required for our dashboard scope.
* **XGBoost:** While powerful, initial experiments suggested that LightGBM offered comparable or better performance with potentially faster training times on our large dataset.
* **Neural Networks:** Neural Network approaches with LSTM layers would theoretically work best with our data, but our experiments proved training these kinds of models would be too slow for our applications.
* **Ad-hoc SVI Surface Generation (Initial Idea):** Our initial focus was generating Heston-like parameters that we would use with an ad-hoc SVI inspired equation to generate volatility surfaces. This generated visually appealing surfaces, however the results were not interpretable as these models are built for implied volatility and do not integrate cleanly with our ultimate comparison goals. Therefore, we gravitated towards a 'local volatility' interpretation, and attempted to come up with a surface model that would align with this implementation.

#### 2.2.3. Final Model: LightGBM for Conditional Realized Volatility

* **Model Name:** LightGBM (Light Gradient Boosting Machine)
* **Response Variable:** Future Annualized Realized Volatility ($\sigma$). Derived from `rv_hd_future`: $\sigma = \sqrt{\text{rv\_hd\_future} / (h/252.0)}$, where $h$ is the prediction horizon in trading days.
* **Explanatory Variables:**
  * A wide range of technical indicators (see [`src/data_transformation/INDICATORS.md`](src/data_transformation/INDICATORS.md)).
  * Joined CBOE volatility index values.
  * Log-moneyness ( $k = log(K/S_0)$ ), corresponding to `log_ret_future_h` during training.
  * Interaction Terms (e.g., $k^2$, $k \times \text{VIX}$, $k \times \text{RV}$).
  * Feature Selection: A predefined set of low-importance features were removed before training.
* **Model Description:** We used LightGBM, a gradient boosting framework efficient for large datasets. Our full implementation is found in [`src/data_modeling/surface_lgbm_modeling.py`](src/data_modeling/surface_lgbm_modeling.py):
  * **Separate Models per Horizon:** Distinct models were trained for each future horizon ($h \in \{10, 15, ..., 35\}$ days).
  * **A/B Cross-Validation:** Stocks were split into two groups (A and B). For each horizon, Model A was trained on Group B/validated on A, and Model B was trained on Group A/validated on B. The application loads the appropriate model (A or B) that *excluded* the selected stock during its training.
  * **Sample Weighting:** Training samples were weighted based on the absolute value of log-moneyness ($|k|$) to emphasize learning the volatility smile/skew.
  * **GPU Training:** Leveraged GPU acceleration for faster training.
  * **Hyperparameters:** Optimized for large datasets (high `num_leaves`, small `learning_rate`, early stopping).
* **Software Implementation:** Python with `lightgbm`, `polars`, `numpy`, `joblib`.
* **Model Estimation & Storage:**
  * Models are **pre-estimated** offline.
  * Trained LightGBM boosters are saved to disk, compressed using `zlib` (`.lgb.z`) to ensure files remain within GitHub size restraints.
  * Models and metadata are stored in dated subdirectories within [`data/models/surface_lgbm/`](data/models/surface_lgbm/).
  * The application loads the required pre-trained models using parallel processing.

### 2.3. Dashboard Construction

#### 2.3.1. Technology

* **Framework and Hosting:** Streamlit
* **Language:** Python
* **Modeling:** LightGBM
* **Plotting:** Plotly
* **Data Handling:** Polars, NumPy
* **Data Storage:** Dolt, Parquet

#### 2.3.2. Layout and Logic

The dashboard contains multiple views controlled by a sidebar radio button:

1. **Volatility Surface View ([`dashboard/surface_view.py`](dashboard/surface_view.py)):** Visualizes and compares volatility surfaces (LGBM RV, Market IV, Comparison) using 3D plots. Includes controls for overlays and comparison metrics.
2. **Execution Price View ([`dashboard/price_view.py`](dashboard/price_view.py)):** Translates volatility surfaces into theoretical option prices using BSM. Displays price heatmaps (LGBM, Market, Difference), simulates a simple trading strategy based on price differences, and visualizes outcomes. Includes controls for BSM parameters and simulation settings.
3. **About Tab ([`streamlit_app.py`](streamlit_app.py)):** Provides context and definitions to help users understand the application.

This layout separates volatility surface visualization and analysis from the pricing/strategy application, letting users focus on their area of interest.

#### 2.3.3. User Inputs

* **Global:** Stock Symbol (Selectbox), Trade Date (Date Input), View Selection (Radio).
* **Volatility View:** Surface Type (Radio), Option Type for IV (Radio), Display Options (Checkboxes for overlays/comparisons).
* **Price View:** Option Type for Pricing (Radio), BSM Parameters (Number Inputs), Display Options (Checkboxes for heatmaps/overlay), Simulation Controls (Checkbox, Sliders for threshold/strike range).

These inputs enable users to customize the analysis extensively.

#### 2.3.4. Outputs

* **Volatility View:** Interactive 3D Plotly surfaces, 3D scatter/line overlays, 2D difference heatmap (compare mode), comparison metrics.
* **Price View:** Interactive 2D Plotly heatmaps (prices, difference, trade outcomes), ITM/OTM boundary line overlay, simulation statistics tables/metrics.
* **About Tab:** Formatted explanatory text.

Outputs update dynamically based on user selections.

#### 2.3.5. Technology Assessment

* **Streamlit:** Easy Python integration and rapid development. Good widget selection and layout options. Performance heavily relies on efficient backend code and data loading (our Parquet strategy was crucial). Streamlit Cloud deployment is very convenient but has resource limits.
* **Plotly:** Excellent for interactive 3D/2D plots, offering fine control but with a slightly steeper learning curve than simpler libraries. Essential for visualizing surfaces effectively.

## 3. Application Learning

### 3.1. Key Learnings

* **Volatility Dynamics:** Gained deeper insight into the volatility smile/skew by comparing model predictions (RV) with market expectations (IV). Differences highlighted potential model disagreements or market inefficiencies.
* **Modeling Challenges:** Predicting future RV is complex. Feature engineering (esp. log-moneyness $k$ and interactions) and sample weighting were vital. High accuracy across all conditions remains challenging.
* **Data Handling is Crucial:** The project underscored the engineering effort needed for large financial datasets. The partitioned Parquet strategy was essential for dashboard feasibility. Efficient libraries like Polars were key.
* **Model vs. Market:** The basic trading simulation showed how model-market discrepancies *could* be traded, but also emphasized that profitability depends on the model's accuracy and actual market movements, not just the predicted difference.

### 3.2. Choices Made

* **Model:** Chose LightGBM for efficiency and performance on large tabular data.
* **Data Storage:** Adopted partitioned Parquet in GitHub for dashboard compatibility and performance.
* **Dashboard Framework:** Selected Streamlit for rapid Python-based development.
* **Target Variable:** Modeled conditional future RV ( $\sigma(k,T)$ ) to separate volatility forecasting from pricing model specifics.
* **Feature Engineering:** Focused on standard indicators plus explicit interaction terms involving log-moneyness ($k$).

### 3.3. Model Support for Goals

The LightGBM approach directly supported the goals by predicting RV surfaces for visualization, enabling comparison with market IV, and providing the necessary volatility input for the BSM pricing model and subsequent trading simulation in the second dashboard view.

## 4. Discussion

### 4.1. Project Summary

This project successfully developed a data pipeline and interactive dashboard for exploring stock volatility surfaces. We processed large financial datasets, engineered features, and trained LightGBM models to predict conditional future realized volatility. The Streamlit dashboard visualizes model predictions, compares them to market-implied volatility, and extends the analysis to theoretical option pricing and strategy simulation. Key technical contributions include the implementation of an efficient partitioned Parquet data loading and data handling strategy, integration of LightGBM models for volatility predictions conditioned on moneyness and time, and 3-D visualizations of volatility surfaces utilizing Plotly.

### 4.2. Next Steps

Potential future improvements include:

* **More Sophisticated Models:** Explore deep learning architectures (Transformers, TFT, NBEATS), advanced financial volatility models (Heston, SABR), and ticker-specific volatility models like GARCH, either as features or as an alternative to our LightGBM model.
* **Data Enhancements:** Incorporate higher-frequency data or alternative data sources (sentiment, macro indicators, news articles).
* **Improved User Experience:** Add more interactivity (point-and-click details, model explanations), side-by-side comparisons, and more realistic trading simulation options.
* **Refine Surface Generation:** Investigate advanced interpolation or fitting techniques for smoother and more robust surface generation, especially from potentially sparse options data.

### 4.3. Unique Aspects

* **Large Data Handling:** Developing the partitioned Parquet strategy to manage multi-gigabyte datasets within project and deployment constraints.
* **Conditional Volatility Modeling:** Using ML to model volatility conditional on future log-moneyness ($k$) from widely accessible OHLCV data (as opposed to expensive data from options markets).
* **Comprehensive User Experience:** Users can visualize and interact with multiple kinds of volatility surfaces, derivative pricing estimations, and trading simulations.

### 4.4. Availability

* **Code:** The complete source code is available in this GitHub repository: [GitHub Repository](https://github.com/nrhoads02/ds401-project)
* **Dashboard:** The live dashboard is hosted on Streamlit Community Cloud: [Volatility Dashboard](https://ds401-dern-volatility-dashboard.streamlit.app/)
* **Maintenance:** As a capstone project, long-term maintenance is not guaranteed. While all of the necessary code and data files should remain hosted within our repo, the dashboard and its corresponding code are ultimately provided as-is.
