# DS 4010 Project Report: Volatility Surface Modeling

**Team:** DERN (Ryan Freidhoff, Nic Rhoads, Dakota Rossi, Emiliano Saucedo)

**Dashboard Link:** [Volatility Dashboard](https://ds401-dern-volatility-dashboard.streamlit.app/)

**Repository Link:** [GitHub Repository](https://github.com/nrhoads02/ds401-project)

**Intended Audience:** New students in DS 4010

## Table of Contents

- [DS 4010 Project Report: Volatility Surface Modeling](#ds-4010-project-report-volatility-surface-modeling)
  - [Table of Contents](#table-of-contents)
  - [1. Application Goal](#1-application-goal)
    - [1.1. Project Objective](#11-project-objective)
    - [1.2. Understanding Key Concepts: Volatility and Options](#12-understanding-key-concepts-volatility-and-options)
    - [1.3. Motivation and Target Audience](#13-motivation-and-target-audience)
  - [2. Data Pipeline (The Backend)](#2-data-pipeline-the-backend)
    - [2.1. Data Collection (ETL)](#21-data-collection-etl)
      - [2.1.1. Data Sources](#211-data-sources)
      - [2.1.2. Data Acquisition](#212-data-acquisition)
      - [2.1.3. Data Cleaning and Preprocessing](#213-data-cleaning-and-preprocessing)
      - [2.1.4. Data Transformation](#214-data-transformation)
      - [2.1.5. Data Storage for Dashboard](#215-data-storage-for-dashboard)
    - [2.2. Modeling](#22-modeling)
      - [2.2.1. Modeling Goal and Purpose](#221-modeling-goal-and-purpose)
      - [2.2.2. Investigated Approaches (Not Used)](#222-investigated-approaches-not-used)
      - [2.2.3. Final Model: LightGBM for Conditional Realized Volatility](#223-final-model-lightgbm-for-conditional-realized-volatility)
    - [2.3. Dashboard Construction](#23-dashboard-construction)
      - [2.3.1. Technology](#231-technology)
      - [2.3.2. Layout and Logic](#232-layout-and-logic)
      - [2.3.3. User Inputs](#233-user-inputs)
      - [2.3.4. Outputs](#234-outputs)
      - [2.3.5. Technology Assessment](#235-technology-assessment)
  - [3. Application Learning](#3-application-learning)
    - [3.1. Key Learnings](#31-key-learnings)
    - [3.2. Choices Made](#32-choices-made)
    - [3.3. Model Support for Goals](#33-model-support-for-goals)
  - [4. Discussion](#4-discussion)
    - [4.1. Project Summary](#41-project-summary)
    - [4.2. Next Steps](#42-next-steps)
    - [4.3. Unique Aspects](#43-unique-aspects)
    - [4.4. Limitations](#44-limitations)
    - [4.5. Availability](#45-availability)

## 1. Application Goal

### 1.1. Project Objective

The primary goal of this project was to develop a tool for analyzing and visualizing stock volatility. Specifically, we aimed to:

1. Model future **Realized Volatility (RV)**, which is the actual, observed volatility of a stock over a future period, conditional on the stock price ending at a specific strike price ($K$) after a certain time ($T$, in days). We denote this model prediction as $\sigma_{LGBM}(K, T)$.
2. Compare this model-predicted RV against **Implied Volatility (IV)**, which is the market's expectation of future volatility derived from current option prices.
3. Visualize these volatilities as 3D surfaces plotted against strike price ($K$) and time to maturity ($T$).
4. Extend the analysis to theoretical option pricing using the Black-Scholes-Merton (BSM) model, comparing model-based prices to market-implied prices to identify potential discrepancies.
5. Present these analyses in an interactive dashboard suitable for educational and exploratory purposes.

### 1.2. Understanding Key Concepts: Volatility and Options

To fully appreciate the project, a few foundational concepts are essential, especially for those new to financial markets.

**Options Contracts:** An option is a financial contract that gives the buyer the *right*, but not the obligation, to either buy (a "call" option) or sell (a "put" option) an underlying asset, such as a stock, at a specified price (the **strike price**, denoted as $K$) on or before a certain date (the **expiration date**, denoted as $T$). The price of an option is influenced by several factors, with volatility being one of the most critical.

**Volatility:** In finance, volatility is a statistical measure of the dispersion of returns for a given security or market index. In simpler terms, it quantifies how much the price of an asset is likely to fluctuate over a given period. Higher volatility means the price can swing dramatically, while lower volatility suggests more stable prices. There are several ways to measure and interpret volatility:

- **Realized Volatility (RV):** This is the *actual* volatility observed in a stock's price movements over a specific *past* period. It's a historical measure. In this project, we aim to *predict future* Realized Volatility.
  - **Our Project's RV Calculation:** We calculate a daily measure of realized variance using a method that accounts for intraday price ranges and overnight price jumps, often referred to as a "Parkinson-plus-jumps" approach. The daily realized variance is composed of three main parts:
    1. **Intraday Parkinson Component:** Captures volatility during trading hours, based on the high and low prices of the day. Calculated as: $\frac{1}{4 \ln 2} \left(\ln\left(\frac{\text{High}}{\text{Low}}\right)\right)^2$.
    2. **Open-to-Close Jump Component:** Measures the volatility from the market open to market close. Calculated as: $\left(\ln\left(\frac{\text{Close}}{\text{Open}}\right)\right)^2$.
    3. **Overnight Jump Component:** Accounts for price changes between the previous day's close and the current day's open. Calculated as: $\left(\ln\left(\frac{\text{Open}}{\text{Close}_{\text{prev}}}\right)\right)^2$.  

    - The sum of these three components gives our `parkinson_plus_jumps_daily` variance.  
    *(Refer to [`src/data_transformation/technical_indicators.py`](src/data_transformation/technical_indicators.py) and [`src/data_transformation/INDICATORS.md`](src/data_transformation/INDICATORS.md) for precise formulas and implementation details of these components).*  
    This daily variance is then summed over a future prediction horizon of $h$ trading days to obtain `rv_hd_future`. The annualized Realized Volatility ($\sigma$) that our model predicts is then derived from this future variance: $\sigma = \sqrt{rv\\_hd\\_future / (h/252.0)}$, assuming 252 trading days in a year.

- **Implied Volatility (IV):** This is the market's *expectation* of future volatility over the life of an option. It is not directly observed but is "implied" by the current market prices of options. If options are expensive, it implies the market expects high volatility, and vice-versa. Our dashboard compares our model's RV predictions against this market-derived IV.

- **VIX (CBOE Volatility Index):** Often referred to as the "fear index," the VIX is a widely followed measure of the stock market's expectation of volatility based on S&P 500 index options. We incorporate VIX and similar CBOE indices as features in our model.

**Log-Moneyness ($k$):** In options analysis, moneyness describes the relationship between an option's strike price and the current price of the underlying asset. Log-moneyness, denoted as $k$, is often defined as $k = \ln(K/S_0)$, where $K$ is the strike price and $S_0$ is the current stock price. It provides a standardized way to represent how far an option's strike is from the current price. Our model predicts RV conditional on this $k$ (using `log_ret_future_h` as its proxy during training, where $k = \ln(S_T/S_0)$ with $S_T$ being the price at horizon $h$).

**Black-Scholes-Merton (BSM) Model:** The BSM model is a mathematical model used to determine the theoretical price of European-style options. It uses factors like the underlying stock price, strike price, time to expiration, risk-free interest rate, dividend yield, and importantly, the expected volatility of the stock. In our project, we use the BSM model to translate different volatility estimates (our predicted RV and market IV) into theoretical option prices.

### 1.3. Motivation and Target Audience

Volatility is a cornerstone concept in finance, particularly in options trading, risk management, and portfolio construction. Understanding how volatility behaves across different strike prices and time horizons (the "volatility surface") provides insights crucial to applications like trading with derivatives.

Our target audience includes fellow Data Science students, finance students, researchers, and potentially retail investors interested in:

- Learning about different types of volatility and their implications.
- Exploring how machine learning models can be applied to predict financial metrics like volatility.
- Visualizing the complex relationship between volatility, strike price, and time.
- Understanding the connection between volatility surfaces and option pricing.
- Seeing a practical example of building a data pipeline and dashboard for financial analysis, especially handling large datasets.

## 2. Data Pipeline (The Backend)

Detailed financial data can consume massive amounts of resources like storage and memory, so ensuring that we have a robust, efficient, and scalable data pipeline is essential for a project of this nature. Our pipeline was developed to handle the initial extraction, cleaning, transformation, and final storage of data in a format suitable for both modeling and efficient dashboard operation.

### 2.1. Data Collection (ETL)

The Extract, Transform, Load (ETL) process formed the foundation of our project, involving several stages to prepare the data for analysis and modeling.

#### 2.1.1. Data Sources

Our project utilized several data sources to gather a comprehensive financial dataset. Initially, we sourced our primary stock and options data from public DoltHub repositories; DoltHub functions similarly to GitHub but is optimized for versioning large datasets. The `stocks` database provided daily Open, High, Low, Close, and Volume (OHLCV) data, along with stock split and dividend information for over 2000 equities, potentially dating back to 2011 (see [`data/raw/stocks`](data/raw/stocks) for structure details). The `options` database contained option chain data (including quotes and Greeks like delta) and historical volatility metrics, generally reported weekly since 2019 (see [`data/raw/options`](data/raw/options)). As an auxiliary source, we downloaded historical data for various CBOE volatility indices (e.g., VIX, VVIX) directly as CSV files from the CBOE website, stored in [`data/raw/cboe`](data/raw/cboe). Due to performance and deployment constraints with accessing DoltHub directly from the dashboard, the core data (OHLCV, options, splits) was ultimately processed into partitioned Apache Parquet files stored within our GitHub repository. This Parquet dataset became the final, primary source for the Streamlit application (see [`data/parquet/`](data/parquet/) for details).

#### 2.1.2. Data Acquisition

The process of acquiring data varied by source. Initially, data was cloned from DoltHub repositories using the Dolt command-line interface. We then used Python scripts, such as [`src/data_extraction/dolt_csv_export.py`](src/data_extraction/dolt_csv_export.py), to export specific tables from the local Dolt databases into CSV files, which allowed for easier manipulation during the early stages of development. The CBOE index data was obtained through manual download of CSV files from their official website. For the final dashboard application, data loading was handled by a custom Python module, [`src/data_extraction/dataframe_loader.py`](src/data_extraction/dataframe_loader.py), designed to efficiently read from the partitioned Parquet files stored within the project's GitHub repository.

#### 2.1.3. Data Cleaning and Preprocessing

Before the data could be used for feature engineering and modeling, a couple of crucial cleaning steps were necessary. Firstly, the raw stock split data contained some duplicate entries for the same split event. To address potential inconsistencies arising from this, we implemented logic to identify and retain only the most recent record for each unique split event, with this process detailed within [`src/data_transformation/stock_adjustments.py`](src/data_transformation/stock_adjustments.py). Secondly, we observed that some stocks had incomplete price histories within our defined timeframe (e.g., delisted before the end date, listed after the start date, or having missing OHLCV values in between). To ensure data consistency for time-series analysis, we filtered the dataset to include only those tickers that were present on both the first and last dates of our dataset and had no missing OHLCV values during that period. This logic, also found in the `stock_adjustments.py` script, while ensuring data quality, did reduce the total number of available stocks for analysis.

#### 2.1.4. Data Transformation

The core of our data preparation involved transforming the cleaned OHLCV data by enriching it with features relevant for volatility modeling. This multi-step process was orchestrated as follows:

1. **Split Adjustments:** Historical OHLCV prices and volumes, and importantly, option strike prices, were adjusted to account for stock splits. This was done using the cleaned split data, where a reverse cumulative adjustment factor was calculated for each stock. This step, detailed in [`src/data_transformation/stock_adjustments.py`](src/data_transformation/stock_adjustments.py), ensures that price and volume data are comparable across time, even if splits have occurred.
2. **Technical Indicator Generation:** A comprehensive set of technical indicators was computed from the adjusted OHLCV data. We utilized the Polars library for its efficiency in handling large datasets for these calculations. The indicators included various moving averages (Simple and Exponential), standard deviation, Relative Strength Index (RSI), Average True Range (ATR), our custom realized variance measures (detailed in Section 1.2), and several volume-based indicators. A complete list and the specific formulas used can be found in our documentation at [`src/data_transformation/INDICATORS.md`](src/data_transformation/INDICATORS.md).
3. **Target Variable Generation:** For the purpose of training our predictive models, we needed to create our target variables. We calculated future realized volatility (`rv_hd_future`) and future log returns (`log_ret_future_h`) by shifting the already computed `rv_hd` (rolling sum of daily realized variance) and rolling log returns forward by $h$ days. These forward-shifted values served as the precursors to the model's target variable, $\sigma$ (annualized future RV), and the conditional variable, log-moneyness ($k$).
4. **CBOE Index Joining:** Data from the CBOE volatility indices (like VIX) was left-joined to the main stock dataframe. This join was performed on the `date` column, aligning these market-wide volatility measures with the daily stock data, using the script [`src/data_transformation/cboe_index_join.py`](src/data_transformation/cboe_index_join.py).
5. **Orchestration:** All these transformation steps were consolidated into a single pipeline function, available in [`src/data_transformation/transformation_pipeline.py`](src/data_transformation/transformation_pipeline.py). This pipeline was designed to leverage Polars' lazy evaluation and streaming capabilities, allowing for efficient processing of the large dataframes without excessive memory consumption.

#### 2.1.5. Data Storage for Dashboard

Handling our large datasets for the interactive dashboard presented a significant challenge. Our raw datasets, particularly after the addition of numerous technical indicators, grew to several gigabytes in size. Storing these as large CSV files directly in GitHub was impractical due to file size limits, and accessing them from DoltHub within the Streamlit Cloud's resource constraints (especially memory and processing time) proved too slow and unreliable for a responsive user experience.

Our solution was to convert the essential datasets (OHLCV, split-adjusted options, and stock splits) into the Apache Parquet format, a process managed by our script [`src/data_extraction/parquet_converter.py`](src/data_extraction/parquet_converter.py). Parquet files offer efficient columnar storage and compression, which significantly reduced the overall file size. More importantly, we implemented a partitioning strategy where these Parquet files were divided based on the stock symbol (`act_symbol`), creating a fixed number of smaller partition files for each dataset.

To enable efficient loading, we generated metadata files (`partition_metadata.json`) that map each stock symbol to its corresponding partition file(s). Our custom data loader, [`src/data_extraction/dataframe_loader.py`](src/data_extraction/dataframe_loader.py), utilizes this metadata. When a user selects a stock in the dashboard, the loader intelligently reads *only* the necessary partition files containing that stock's data, rather than loading the entire multi-gigabyte dataset into memory. This approach drastically reduced memory usage and data load times in the Streamlit application, allowing us to store the data within GitHub's repository limits and, crucially, maintain fast and responsive dashboard performance for the end-user.

### 2.2. Modeling

With the data prepared, the next phase focused on developing predictive models for future realized volatility.

#### 2.2.1. Modeling Goal and Purpose

The central modeling effort aimed to **predict future realized volatility surfaces**. The purpose of this model is predictive: given the market conditions and a suite of technical indicators observed up to a specific trade date, the model estimates the expected realized volatility ($\sigma$) over a defined future horizon ($T$), conditional on the stock price ultimately reaching a specific strike price ($K$) relative to its starting price ($S_0$).

This predicted surface, denoted as $\sigma_{LGBM}(K, T)$, serves multiple key purposes within our application. Firstly, it allows for **direct visualization** of the model's expectation of the future volatility structure across different strikes and maturities. Secondly, it enables **comparison with market-implied volatility (IV)**; highlighting discrepancies between the model's RV prediction and the market's IV can indicate potential mispricings or areas where the model's forecast diverges from market consensus. Lastly, the predicted volatility acts as a critical **input for option pricing**, where it is fed into the Black-Scholes-Merton model to generate theoretical option prices based on our model's volatility forecast.

#### 2.2.2. Investigated Approaches (Not Used)

During our project, we explored several modeling approaches before settling on our final LightGBM implementation. **GARCH (Generalized Autoregressive Conditional Heteroskedasticity) models**, while standard for time-series volatility modeling, were considered but ultimately deemed too computationally expensive. Fitting potentially thousands of individual GARCH models (e.g., one per stock or stock cluster) for the scope of our dashboard and dataset size was not feasible. We also initially considered **XGBoost**, another powerful gradient boosting framework. However, early experiments suggested that LightGBM could offer comparable, if not better, performance with the benefit of potentially faster training times on our large dataset, leading us to focus our efforts there. **Neural Network approaches**, particularly those with LSTM (Long Short-Term Memory) layers, are theoretically well-suited for time-series data like ours. However, our experiments indicated that training these types_of models to a satisfactory level of performance would be too slow and resource-intensive for the project's timeframe and application needs. Lastly, our initial concept for surface generation involved an **ad-hoc SVI (Stochastic Volatility Inspired) approach**, where we aimed to generate Heston-like parameters to plug into an SVI-style equation. While this produced visually appealing surfaces, the results were difficult to interpret in the context of realized volatility, as SVI models are primarily designed for implied volatility. This led us to pivot towards a 'local volatility' style interpretation, attempting to model the surface more directly.

#### 2.2.3. Final Model: LightGBM for Conditional Realized Volatility

Our final modeling choice was the LightGBM (Light Gradient Boosting Machine), a decision driven by its efficiency and strong performance on large, tabular datasets.

The **response variable** for our model is the Future Annualized Realized Volatility ($\sigma$). This is derived from the `rv_hd_future` (the sum of daily Parkinson-plus-jumps variances over an $h$-day future horizon) using the formula: $\sigma = \sqrt{rv\_hd\_future / (h/252.0)}$, where $h$ is the prediction horizon in trading days.

The **explanatory variables** used to predict this $\sigma$ include a wide array of features:
A comprehensive set of technical indicators (detailed in [`src/data_transformation/INDICATORS.md`](src/data_transformation/INDICATORS.md)) calculated from historical OHLCV data;
Values from joined CBOE volatility indices (e.g., VIX);
Log-moneyness, defined as $k = \ln(K/S_0)$, which during training corresponds to the `log_ret_future_h` (the log return over the horizon $h$);
Interaction terms, such as $k^2$, $k \times \text{VIX}$, and $k \times \text{RV}$, designed to help the model capture non-linear relationships characteristic of volatility surfaces.
Before training, a predefined set of features identified as having low importance or being redundant were explicitly removed.

The **model description** and our specific implementation details can be found in [`src/data_modeling/surface_lgbm_modeling.py`](src/data_modeling/surface_lgbm_modeling.py). Key characteristics include:
Training distinct models for each future horizon ($h \in \{10, 15, \dots, 35\}$ days), allowing each model to specialize in patterns relevant to its specific forecast period.
Employing an A/B cross-validation scheme: stocks were divided into two groups (A and B). For each horizon, one model (Model A) was trained using Group B stocks and validated on Group A, while another (Model B) was trained on Group A and validated on Group B. When making predictions for a selected stock, the application intelligently loads the model (either A or B) that did *not* include that particular stock in its training set, ensuring out-of-sample predictions.
Weighting training samples based on the absolute value of log-moneyness ($|k|$), to give more emphasis to learning the volatility smile and skew effects often observed further from the at-the-money point.
Leveraging GPU acceleration during training to significantly reduce computation time.
Utilizing hyperparameters optimized for large datasets, such as a high number of leaves per tree (`num_leaves`), a small learning rate, and early stopping criteria to prevent overfitting.

The **software implementation** was done in Python, relying heavily on the `lightgbm` library for the modeling itself, `polars` for efficient data manipulation, `numpy` for numerical operations, and `joblib` for saving model metadata.

Regarding **model estimation and storage**:
All models are pre-estimated offline due to the substantial time required for training.
The trained LightGBM booster objects are saved to disk. To manage file sizes effectively, especially for storage on GitHub, models are compressed using `zlib` and stored with a `.lgb.z` extension if they exceed a certain size threshold.
These models, along with their associated metadata (like feature names used for each horizon and the A/B stock splits), are organized into dated subdirectories within [`data/models/surface_lgbm/`](data/models/surface_lgbm/).
The dashboard application then loads these required pre-trained models dynamically using parallel processing for improved responsiveness.

In terms of **model performance**:
Training the complete set of models for all horizons on the full dataset (using a stride of 1 for sampling) took approximately 6 hours on our available hardware.
Across the different prediction horizons, the Root Mean Squared Error (RMSE) for $\sigma$ typically ranged from 0.3 to 0.4. The $R^2$ values consistently exceeded 0.5, indicating that the models could explain a substantial portion of the variance in future realized volatility.
An interesting observation from the feature importance analysis was that $k$-related features (log-moneyness and its interactions) were generally less important than initially anticipated. This may have contributed to the model producing volatility smiles and skews that were flatter than those typically observed in implied volatility markets.

### 2.3. Dashboard Construction

The dashboard serves as the interactive front-end for users to explore the volatility surfaces and pricing analyses generated by our backend pipeline and models.

#### 2.3.1. Technology

The dashboard was built using **Streamlit** as the primary web framework, chosen for its ease of use and strong integration with Python. All data processing and modeling components were also developed in **Python**. For the machine learning aspect, **LightGBM** was the chosen library. Data manipulation was handled efficiently by **Polars** and **NumPy**. Interactive visualizations, particularly the 3D surfaces and 2D heatmaps, were created using **Plotly**. The initial data was sourced from **Dolt** databases, but for the final application, data was stored and accessed from **Parquet** files. The application is hosted on Streamlit Community Cloud.

#### 2.3.2. Layout and Logic

The dashboard is organized into multiple views, selectable via a sidebar radio button, to provide a structured user experience. The primary script orchestrating the UI is [`streamlit_app.py`](streamlit_app.py).

The **Volatility Surface View**, managed by [`dashboard/surface_view.py`](dashboard/surface_view.py), is dedicated to the visualization and comparison of different volatility surfaces. Users can choose to display the LightGBM model's predicted Realized Volatility (RV), the market's Implied Volatility (IV) surface derived from option prices, or a direct comparison of the two. This view features interactive 3D plots and includes controls for overlaying contextual data like realized volatility points and historical RV lines, as well as metrics for comparing surface accuracies.

The **Execution Price View**, handled by [`dashboard/price_view.py`](dashboard/price_view.py), translates these volatility surfaces into theoretical option prices using the Black-Scholes-Merton (BSM) model. It displays various 2D heatmaps, including the model-implied (LGBM) price, the market-implied price, and the dollar difference between them. This view also features a simple trading strategy simulation based on discrepancies between the model and market prices, visualizing potential trade outcomes. Users can adjust BSM parameters and simulation settings here.

Finally, an **About Tab**, integrated within the main `streamlit_app.py` script, provides textual context and definitions for key financial terms and explains the dashboard's functionalities, aiming to help users better understand the presented analyses. This multi-view layout allows users to focus on either the direct volatility analysis or its implications for option pricing and trading strategies.

#### 2.3.3. User Inputs

The dashboard offers a range of user inputs to allow for customized analysis. Globally, users can select the **Stock Symbol** from a dropdown list and specify the **Trade Date** using a calendar input. A primary **View Selection** radio button allows switching between the "Volatility Surface" and "Execution Price" analysis modes.

Within the **Volatility Surface View**, users can choose the **Surface Type** (LGBM RV, Options IV, or Compare Surfaces) and, if viewing options-related data, the **Option Type for IV** (Call, Put, or Average). Checkboxes under "Display Options" allow toggling overlays such as "Show Realized Points" and "Show Historical RV @ Spot K," and in comparison mode, elements like the "IV vs RV Difference Heatmap" and "Surface vs Realized Accuracy" metrics.

In the **Execution Price View**, users select the **Option Type for Pricing** (Call or Put) for the BSM calculations. They can input numerical **BSM Parameters** like the Risk-Free Rate and Dividend Yield. "Display Options" here include checkboxes for various heatmaps (LGBM Expected Price, Market Implied Price, Price Difference, Executed Trade Heatmap) and an overlay for the "Actual ITM/OTM Boundary." Simulation controls include a checkbox to "Show Trading Strategy Simulation Stats" and sliders to adjust the "Mispricing Threshold (%)" and "Strike Range Padding (%) around S0" for the trading simulation. These extensive input options enable users to tailor the visualizations and analyses to their specific interests.

#### 2.3.4. Outputs

The dashboard dynamically generates several outputs based on user selections. In the **Volatility Surface View**, the primary outputs are interactive 3D Plotly surfaces representing the $\sigma_{LGBM}(K, T)$ and/or Market IV. These can be augmented with 3D scatter points showing actual Realized Points and 3D lines for the Historical RV @ Spot K. When in comparison mode, a 2D Plotly heatmap displays the percentage difference between the Implied Volatility and the model's Realized Volatility, alongside tables and metrics comparing the average differences between the surfaces and the realized points.

The **Execution Price View** primarily features interactive 2D Plotly heatmaps. These visualize the LGBM Expected Price, the Market Implied Price, the dollar Price Difference (Market - LGBM), and the outcome of the Simulated Trades (colored by P&L with symbols for buy/sell actions). An important overlay in this view is a line indicating the actual In-The-Money (ITM) / Out-of-The-Money (OTM) boundary, derived from subsequent price movements. Additionally, this view presents metrics and tables summarizing the results of the trading strategy simulation, such as total Profit & Loss (P&L) and Win Rates. The **About Tab** simply outputs formatted explanatory text. All graphical and textual outputs are designed to update dynamically as user inputs are changed, facilitating an exploratory analysis experience.

#### 2.3.5. Technology Assessment

Our choice of **Streamlit** for the framework and hosting proved highly effective due to its seamless Python integration and the ability to rapidly develop and iterate on the dashboard. Creating interactive widgets and managing layout was generally straightforward. However, we learned that Streamlit's performance is heavily dependent on efficient backend Python code and, critically, data loading strategies. Our initial struggles with large datasets necessitated the optimization efforts described earlier, particularly the move to a partitioned Parquet data backend. The resource limits of Streamlit Community Cloud (e.g., RAM) also reinforced the need for this efficiency. Deployment via GitHub integration was notably simple and user-friendly.

For plotting, **Plotly** was an excellent choice. It enabled the creation of interactive, publication-quality visualizations, especially for the complex 3D surfaces which are central to our project. Plotly offers fine-grained control over plot appearance and hover-information, which enhances the user's ability to explore the data. While it has a slightly steeper learning curve compared to simpler plotting libraries, its capabilities were essential for effectively visualizing the volatility surfaces and related heatmaps.

## 3. Application Learning

This project provided numerous insights into financial modeling, data engineering, and dashboard development.

### 3.1. Key Learnings

A significant portion of our learning involved diving into financial research to understand existing approaches to volatility modeling and identify techniques suitable for our data constraints and project goals. We gained considerable insight into the dynamics of the **volatility smile/skew** by comparing our various model predictions (for Realized Volatility) with the Implied Volatility derived from options markets; the observed differences often highlighted areas where our model's forecasts diverged from market expectations or pointed to potential market inefficiencies.

The process of **modeling and predicting future Realized Volatility** itself proved to be complex. We learned that careful feature engineering, especially the inclusion of log-moneyness ($k$) and its interaction terms, along with appropriate sample weighting, were vital for the model to capture any non-linear patterns. However, achieving high accuracy across all market conditions and for all stocks remains a formidable challenge. A key limitation we encountered was the inability to evaluate out-of-sample performance on truly forward-looking data due to the historical cutoff of our primary dataset. This restricted our ability to definitively validate the model's robustness and generalizability to future, unseen market conditions.

**Data handling** emerged as a crucial and demanding aspect of the project. The sheer size of financial datasets requires substantial engineering effort. Our journey from initial exploration with Dolt, to processing with efficient libraries like Polars, and finally implementing a partitioned Parquet strategy for the dashboard, underscored the importance of scalable data management for both development and deployment.

Finally, the **Model vs. Market comparison**, facilitated by the basic trading simulation, was instructive. It demonstrated how discrepancies between a model's predicted RV (translated into a theoretical option price via BSM) and market IV (also translated into a market option price) could hypothetically identify mispriced options. However, this also served as a reminder that such simulations are simplified and that actual profitability depends on the model's predictive accuracy and real market movements, not just the perceived difference.

### 3.2. Choices Made

Several key decisions shaped the direction and outcome of our project. For **modeling**, we chose LightGBM primarily for its recognized efficiency and strong performance on large, tabular datasets, prioritizing it over potentially more computationally expensive GARCH models or more complex deep learning architectures which posed development time constraints.

Regarding **data storage** for the dashboard, we adopted a strategy of partitioned Parquet files hosted within our GitHub repository. This choice was driven by the need for dashboard compatibility, performance, and to overcome limitations encountered with direct DoltHub access or handling very large single CSV files in the Streamlit Cloud environment.

For the **dashboard framework**, Streamlit was selected due to its rapid Python-based development cycle and ease of integration with our existing Python codebase, making it a more straightforward choice for our team compared to alternatives like Dash or R Shiny.

In defining our **target variable**, we opted to model conditional future Realized Volatility, $\sigma(k,T)$, which separates the task of volatility forecasting from the assumptions inherent in specific option pricing models like Black-Scholes-Merton.

Lastly, our **feature engineering** efforts were concentrated on using standard financial technical indicators supplemented with explicit interaction terms involving log-moneyness ($k$), aiming to help the model capture the nuanced effects often seen in volatility surfaces.

### 3.3. Model Support for Goals

The LightGBM modeling approach was instrumental in achieving our project goals. It directly enabled the prediction of Realized Volatility surfaces, which formed the core of the "LGBM Model RV" visualization. These predictions then facilitated a direct comparison with market-derived Implied Volatility surfaces, allowing for the identification of divergences between our model's forecasts and market expectations. Furthermore, the predicted volatility values served as a crucial input for the Black-Scholes-Merton pricing model in the "Execution Price" view of the dashboard, which in turn powered the trading simulation. While there is always room for improvement in predictive accuracy, the chosen LightGBM framework provided a robust and computationally feasible method for demonstrating the core concepts of volatility modeling, surface visualization, and model-versus-market analysis that we set out to explore.

## 4. Discussion

### 4.1. Project Summary

This project successfully culminated in the development of a comprehensive data pipeline and an interactive Streamlit dashboard designed for exploring stock volatility surfaces. Our process involved sourcing and meticulously processing large-scale financial datasets, which included OHLCV records, options data, stock split information, and CBOE volatility indices. We engineered a wide array of relevant features from this data and subsequently trained LightGBM models to predict conditional future realized volatility. The resulting dashboard allows users to visualize these model predictions, compare them against market-implied volatility, and extend the analysis to theoretical option pricing via the Black-Scholes-Merton model, complete with a basic trading simulation. Key technical achievements of this project include the implementation of an efficient data handling strategy using partitioned Parquet files suitable for web deployment, the successful integration of LightGBM models for predicting volatility conditioned on moneyness and time, and the creation of intuitive 3-D visualizations of volatility surfaces using Plotly.

### 4.2. Next Steps

Looking ahead, several avenues could be explored for future improvements and expansions of this project. Regarding **more sophisticated models**, we could investigate deep learning architectures such as Transformers, Temporal Fusion Transformers (TFT), or NBEATS, which are designed for time-series forecasting. Additionally, incorporating advanced financial volatility models like Heston or SABR, or even developing ticker-specific time-series models like GARCH (perhaps as features or as alternatives for specific stocks), could yield more nuanced predictions.

**Data enhancements** could also be pursued. This might involve incorporating higher-frequency (intraday) data to refine realized volatility calculations or integrating alternative data sources such as market sentiment indicators, macroeconomic news releases, or analyses of news articles related to specific equities, which could provide additional predictive signals.

From a **user experience** perspective, the dashboard could be enhanced by adding more interactive elements, such as allowing users to click on specific points on a surface to see detailed underlying data or potential model explanations (e.g., SHAP values for LightGBM, if feasible). Side-by-side comparison tools for different stocks or dates, and more realistic and configurable trading simulation options, would also enrich the user's analytical capabilities. We envision creating additional specialized views, perhaps tailored for portfolio management applications or exploring statistical arbitrage trading styles.

Finally, **refining the surface generation** techniques themselves is another area for development. This could involve investigating advanced interpolation or fitting methods to create smoother and more robust surfaces, particularly when dealing with options data that may be sparse for certain strikes or maturities.

### 4.3. Unique Aspects

This project incorporated several unique aspects that distinguish it. A primary one was the significant effort dedicated to **large data handling**, specifically developing and implementing the partitioned Parquet strategy. This was crucial for managing multi-gigabyte datasets within the practical constraints of an academic project and a web deployment environment, enabling both efficient processing and responsive dashboard performance.

Another distinctive feature is our approach to **conditional volatility modeling**. We utilized machine learning techniques to model volatility conditional on future log-moneyness ($k$) and time to maturity, deriving these predictions primarily from widely accessible OHLCV data. This contrasts with many traditional approaches that rely more heavily on often expensive, proprietary data from options markets to infer volatility.

Lastly, we aimed for a **comprehensive user experience** by providing a multifaceted dashboard. Users are not limited to viewing a single type of volatility surface; instead, they can visualize and interact with multiple representations (model-predicted RV, market IV), explore theoretical derivative pricing estimations based on these volatilities, and even engage with a basic trading simulation to understand potential implications of model-market discrepancies.

### 4.4. Limitations

Despite our efforts, the project has several limitations that are important to acknowledge.
A significant constraint was related to **forward testing**. Due to the historical cutoff date of our primary dataset, we were unable to evaluate the out-of-sample performance of our models on truly "future" data that occurred after the dataset was compiled. This limits our ability to definitively validate the model's robustness and predictive power under subsequent, unseen market conditions.

The **Implied Volatility (IV) data** itself, derived from options market prices, is inherently noisy and can be model-dependent (i.e., different IVs might be calculated using different assumptions or pricing models). This introduces a degree of ambiguity when comparing our model's Realized Volatility predictions against market IV, as we are comparing a direct prediction of future volatility with a market-inferred expectation.

Our **model simplifications** also represent a limitation. The current approach does not explicitly incorporate the impact of key forward-looking events, such as scheduled earnings announcements, major macroeconomic data releases, or significant geopolitical risk factors. These events can substantially influence realized volatility, and their absence in the model means it may not fully capture all drivers of future price movements.

Finally, the **trading simulation** included in the dashboard is basic and does not account for critical real-world factors. Specifically, it ignores transaction costs (brokerage fees, etc.), liquidity constraints (the ability to execute trades at desired prices without impacting the market), or execution slippage (the difference between the expected trade price and the actual price at which the trade is executed). These factors would be critical in any practical implementation of a trading strategy and would likely reduce the profitability of the simulated trades.

### 4.5. Availability

- **Code:** The complete source code for the data pipeline, modeling, and dashboard can be found in this GitHub repository: [GitHub Repository](https://github.com/nrhoads02/ds401-project)
- **Dashboard:** The live, interactive dashboard is hosted on Streamlit Community Cloud: [Volatility Dashboard](https://ds401-dern-volatility-dashboard.streamlit.app/)
- **Maintenance:** As a capstone project, long-term maintenance is not guaranteed. While all of the necessary code and data files should remain hosted within our repository, the dashboard and its corresponding code are ultimately provided as-is.
