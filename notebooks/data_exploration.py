#!/usr/bin/env python
# coding: utf-8

import importlib
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.data_transformation import transformation_pipeline
from src.data_modeling import xgboost_modeling
from src.data_modeling import model_visualization
from src.data_modeling import volatility_clustering
from src.data_modeling import tft_modeling, nbeats_modeling, deepar_modeling


# Import ohlcv data as polar dataframe
ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv").with_columns(
    pl.col("date").str.to_date("%Y-%m-%d")
)


# Apply transformations (~1 minute)
ohlcv = transformation_pipeline.transformation_pipeline(ohlcv)


# Verify columns appended properly
print("Columns after transformation:", ohlcv.columns)
ohlcv.head(10)


importlib.reload(deepar_modeling)

# Train paired DeepAR models
deepar_results = deepar_modeling.train_paired_deepar_models(
    df=ohlcv,
    target_cols=['YZVol_10_future', 'YZVol_30_future', 'YZVol_90_future'],
    train_ratio=0.5,
    distribution="lognormal",
)

# Generate and visualize volatility surfaces
surface = deepar_modeling.create_volatility_surface_for_stock(
    df=ohlcv,
    stock="AAPL",
    model_results=deepar_results,
    target_cols=['YZVol_10_future', 'YZVol_30_future', 'YZVol_90_future'],
    prediction_horizons=[10, 30, 60, 90, 180]
)

# Save the trained models
deepar_modeling.save_deepar_models(deepar_results)


def predict_volatility_for_date(df, stock, model_results, target_date, 
                           target_cols=['YZVol_10_future', 'YZVol_30_future', 'YZVol_90_future'],
                           context_length=60, device="cpu"):
    """
    Generate volatility predictions for a specific date.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with stock data
    stock : str
        Stock symbol (e.g., 'AAPL')
    model_results : Dict
        Loaded DeepAR model results
    target_date : pl.Date or datetime.date
        The date for which to generate predictions
    target_cols : List[str]
        Target columns to predict
    context_length : int
        Number of days to use as context
    device : str
        Device to use for prediction
        
    Returns:
    --------
    Dict with volatility predictions
    """
    import torch
    import numpy as np
    
    # Determine which model to use
    stocks_A = model_results.get('stocks_A', [])
    
    if stock in stocks_A:
        print(f"{stock} is in Stock Group A - using Model 2 for prediction")
        model = model_results.get('model_2')
        dataset = model_results.get('model_2_dataset')
    else:
        print(f"{stock} is in Stock Group B - using Model 1 for prediction")
        model = model_results.get('model_1')
        dataset = model_results.get('model_1_dataset')
    
    # Move model to the correct device
    model = model.to(device)
    
    # Get feature columns
    feature_cols = model_results.get('feature_cols', [])
    
    # Filter data for this stock up to the target date
    stock_data = df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") <= target_date)
    ).sort("date")
    
    # Check if we have enough data
    if stock_data.height < context_length:
        raise ValueError(f"Not enough data for {stock} up to {target_date}")
    
    # Use the last context_length days as context
    stock_data = stock_data.tail(context_length)
    
    # Extract target(s)
    targets = []
    for target_col in target_cols:
        if target_col in stock_data.columns:
            values = stock_data.select(target_col).to_numpy().flatten()
            targets.append(values)
        else:
            raise ValueError(f"Target column {target_col} not found in data")
    
    # Stack targets into a 2D array [time_steps, n_targets]
    target = np.column_stack(targets)
    
    # Extract features
    features = []
    for col in feature_cols:
        if col in stock_data.columns:
            values = stock_data.select(col).to_numpy().flatten()
            
            # Replace any NaN/infinite values with 0
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
            
            features.append(values)
        else:
            # Create a dummy feature of zeros if column doesn't exist
            features.append(np.zeros(context_length))
    
    # Stack features into a 2D array [time_steps, n_features]
    features = np.column_stack(features)
    
    # Get dates
    dates = stock_data["date"].to_list()
    
    # Prepare context data
    context_data = {
        "context_target": torch.tensor(target, dtype=torch.float32).unsqueeze(0),  # Add batch dimension
        "context_features": torch.tensor(features, dtype=torch.float32).unsqueeze(0),
        "context_dates": dates
    }
    
    # Normalize data
    if dataset is not None:
        if hasattr(dataset, 'target_means') and hasattr(dataset, 'target_stds'):
            # Multi-target normalization
            normalized_target = np.zeros_like(target)
            
            for dim in range(len(target_cols)):
                normalized_target[:, dim] = ((target[:, dim] - dataset.target_means[dim]) 
                                           / dataset.target_stds[dim])
            
            context_data["context_target"] = torch.tensor(normalized_target, dtype=torch.float32).unsqueeze(0)
        
        # Normalize features
        if hasattr(dataset, 'feature_means') and hasattr(dataset, 'feature_stds'):
            normalized_features = (features - dataset.feature_means) / dataset.feature_stds
            context_data["context_features"] = torch.tensor(normalized_features, dtype=torch.float32).unsqueeze(0)
    
    # Ensure tensors are on the correct device
    context_data["context_target"] = context_data["context_target"].to(device)
    context_data["context_features"] = context_data["context_features"].to(device)
    
    # Generate predictions
    model.eval()
    with torch.no_grad():
        # Single step prediction for all horizons
        samples, params = model.predict(
            context_data["context_target"], 
            context_data["context_features"], 
            prediction_length=1,
            num_samples=100,
            return_params=True
        )
        
        # Extract mean predictions
        if model.distribution == "student-t":
            mu, sigma, nu = params
        else:
            mu, sigma = params
        
        # De-normalize predictions
        if hasattr(dataset, 'target_means') and hasattr(dataset, 'target_stds'):
            # Multi-target normalization
            predictions = {}
            for dim, col in enumerate(target_cols):
                # Extract the horizon from the column name (assuming format like 'YZVol_10_future')
                horizon = int(col.split('_')[1])
                
                # De-normalize
                mean_vol = mu[0, 0, dim].item() * dataset.target_stds[dim] + dataset.target_means[dim]
                samples_vol = samples[0, 0, dim, :].cpu().numpy() * dataset.target_stds[dim] + dataset.target_means[dim]
                
                # Calculate prediction intervals
                lower_50 = np.quantile(samples_vol, 0.25)
                upper_50 = np.quantile(samples_vol, 0.75)
                lower_90 = np.quantile(samples_vol, 0.05)
                upper_90 = np.quantile(samples_vol, 0.95)
                
                predictions[horizon] = {
                    "mean": mean_vol,
                    "50%_interval": (lower_50, upper_50),
                    "90%_interval": (lower_90, upper_90),
                    "samples": samples_vol
                }
        else:
            # Single target normalization
            predictions = {}
    
    return {
        "date": target_date,
        "stock": stock,
        "predictions": predictions
    }

# Example usage:
import polars as pl
from datetime import date

# Load your saved model if you haven't already
# deepar_results = deepar_modeling.load_deepar_models()

# For a specific date (replace with your target date)
target_date = pl.date(2020, 3, 15)  # Or use: date(2020, 6, 15)

# Generate predictions for that date
vol_predictions = predict_volatility_for_date(
    df=ohlcv,
    stock="AAPL",
    model_results=deepar_results,
    target_date=target_date
)

# Display the results
print(f"Volatility predictions for {vol_predictions['stock']} on {vol_predictions['date']}:")
for horizon, pred in vol_predictions["predictions"].items():
    print(f"{horizon}-day volatility: {pred['mean']:.6f}")
    print(f"  50% interval: {pred['50%_interval'][0]:.6f} to {pred['50%_interval'][1]:.6f}")
    print(f"  90% interval: {pred['90%_interval'][0]:.6f} to {pred['90%_interval'][1]:.6f}")


def plot_volatility_predictions_over_time(df, stock, model_results, 
                                 start_date, end_date, 
                                 target_cols=['YZVol_10_future', 'YZVol_30_future', 'YZVol_90_future'],
                                 interval_days=14,  # Sample every two weeks
                                 show_intervals=True,
                                 context_length=60, 
                                 device="cpu"):
    """
    Plot volatility predictions for a stock over a date range.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with stock data
    stock : str
        Stock symbol (e.g., 'AAPL')
    model_results : Dict
        Loaded DeepAR model results
    start_date : pl.Date or datetime.date
        Start date for predictions
    end_date : pl.Date or datetime.date
        End date for predictions
    target_cols : List[str]
        Target columns to predict
    interval_days : int
        Number of days between prediction points
    show_intervals : bool
        Whether to show confidence intervals
    context_length : int
        Number of days to use as context
    device : str
        Device to use for prediction
        
    Returns:
    --------
    Tuple of (fig, ax) with the matplotlib figure and axes
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import timedelta
    import polars as pl
    
    # Get all dates in the range
    all_dates = df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") >= start_date) &
        (pl.col("date") <= end_date)
    ).select("date").unique().sort("date")["date"].to_list()
    
    # Sample dates at regular intervals
    if interval_days > 1 and len(all_dates) > interval_days:
        sampled_dates = all_dates[::interval_days]
        # Always include the last date
        if sampled_dates[-1] != all_dates[-1]:
            sampled_dates.append(all_dates[-1])
    else:
        sampled_dates = all_dates
    
    print(f"Generating predictions for {len(sampled_dates)} dates from {start_date} to {end_date}")
    
    # Generate predictions for each date
    predictions = []
    for date in sampled_dates:
        try:
            result = predict_volatility_for_date(
                df=df,
                stock=stock,
                model_results=model_results,
                target_date=date,
                target_cols=target_cols,
                context_length=context_length,
                device=device
            )
            predictions.append(result)
        except Exception as e:
            print(f"Error predicting for {date}: {str(e)}")
    
    if not predictions:
        raise ValueError("No valid predictions generated")
    
    # Extract data for plotting
    dates = [p["date"] for p in predictions]
    horizons = sorted([int(col.split('_')[1]) for col in target_cols])
    
    # Prepare data by horizon
    horizon_data = {}
    for horizon in horizons:
        means = []
        lower_50 = []
        upper_50 = []
        lower_90 = []
        upper_90 = []
        
        for p in predictions:
            if horizon in p["predictions"]:
                pred = p["predictions"][horizon]
                means.append(pred["mean"])
                lower_50.append(pred["50%_interval"][0])
                upper_50.append(pred["50%_interval"][1])
                lower_90.append(pred["90%_interval"][0])
                upper_90.append(pred["90%_interval"][1])
            else:
                # If this horizon wasn't predicted, use NaN
                means.append(np.nan)
                lower_50.append(np.nan)
                upper_50.append(np.nan)
                lower_90.append(np.nan)
                upper_90.append(np.nan)
        
        horizon_data[horizon] = {
            "means": np.array(means),
            "lower_50": np.array(lower_50),
            "upper_50": np.array(upper_50),
            "lower_90": np.array(lower_90),
            "upper_90": np.array(upper_90)
        }
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color map for different horizons - from blue (short-term) to red (long-term)
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(horizons)))
    
    # Plot each horizon
    for i, horizon in enumerate(horizons):
        data = horizon_data[horizon]
        
        # Plot mean line
        line, = ax.plot(dates, data["means"], label=f"{horizon}-day Volatility", 
                color=colors[i], linewidth=2)
        
        # Plot confidence intervals if requested
        if show_intervals:
            # 50% interval (darker shade)
            ax.fill_between(dates, data["lower_50"], data["upper_50"], 
                          color=colors[i], alpha=0.3)
            
            # 90% interval (lighter shade)
            ax.fill_between(dates, data["lower_90"], data["upper_90"], 
                          color=colors[i], alpha=0.1)
    
    # Add stock price to the plot on a secondary axis
    stock_data = df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") >= start_date) &
        (pl.col("date") <= end_date)
    ).sort("date")
    
    if stock_data.height > 0:
        ax2 = ax.twinx()
        ax3 = ax2.twinx()  # Create a third axis for volatility
        price_dates = stock_data["date"].to_list()
        prices = stock_data["close"].to_list()
        vol10 = stock_data["YZVol_10_future"].to_list()
        vol30 = stock_data["YZVol_30_future"].to_list()
        vol90 = stock_data["YZVol_90_future"].to_list()
        
        ax2.plot(price_dates, prices, 'k--', alpha=0.6, linewidth=1, label='Stock Price')
        ax2.set_ylabel('Price ($)', color='black')
        ax2.tick_params(axis='y', labelcolor='black')

        # Plot volatility on the third axis
        ax3.plot(price_dates, vol10, 'g-', alpha=0.6, linewidth=1, label='10-day Volatility')
        ax3.plot(price_dates, vol30, 'b-', alpha=0.6, linewidth=1, label='30-day Volatility')
        ax3.plot(price_dates, vol90, 'r-', alpha=0.6, linewidth=1, label='90-day Volatility')
        ax3.set_ylabel('Volatility', color='green')
        ax3.tick_params(axis='y', labelcolor='green')
        
        # Add stock price legend to the plot
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines3, labels3 = ax3.get_legend_handles_labels()
        ax.legend(lines + lines2 + lines3, labels + labels2 + labels3, loc='upper left')
    else:
        ax.legend(loc='upper left')
    
    # Set up plot labels and formatting
    ax.set_xlabel('Date')
    ax.set_ylabel('Predicted Volatility')
    ax.set_title(f'{stock} Predicted Volatility Over Time')
    
    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    fig.autofmt_xdate()
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    return fig, ax

# Example usage:
import polars as pl
from datetime import date

# Load your saved model if you haven't already
# deepar_results = deepar_modeling.load_deepar_models()

# Define date range
start_date = pl.date(2020, 1, 1)
end_date = pl.date(2020, 12, 31)

# Plot volatility predictions over time
fig, ax = plot_volatility_predictions_over_time(
    df=ohlcv,
    stock="AAPL",
    model_results=deepar_results,
    start_date=start_date,
    end_date=end_date,
    interval_days=1  # Sample monthly for faster execution
)

plt.show()


importlib.reload(deepar_modeling)

# deepar_modeling.save_deepar_models(deepar_results)

deepar_results = deepar_modeling.load_deepar_models()


importlib.reload(model_visualization)

# # First, run your standard evaluation
# results = model_visualization.evaluate_volatility_predictors(
#     ohlcv=ohlcv,
#     ticker="AAPL",
#     model_results=deepar_results,
#     target_vol="YZVol_30_future",
#     model_name="DeepAR",
#     model_type="deepar"
# )

# # Then use the improved visualization with corrected intervals
# improved_results = model_visualization.update_plot_detailed_volatility_comparison(
#     ohlcv=ohlcv,
#     ticker="AAPL",
#     model_results=deepar_results,
#     existing_eval_results=results,
#     target_vol="YZVol_30_future",
#     model_name="DeepAR",
#     model_type="deepar", 
#     stride=5,  # Smaller stride for smoother intervals
#     start_date="2020-01-01",  # Adjust as needed
#     end_date="2020-12-31",  # Adjust as needed
# )

results = model_visualization.visualize_full_period_volatility(
    ohlcv=ohlcv,
    ticker="AAPL",
    model_results=deepar_results,
    start_date=pl.date(2020, 1, 1),
    end_date=pl.date(2021, 6, 30),
    target_vol="YZVol_10_future",
    model_name="DeepAR",
    model_type="deepar",
    show_intervals=True
)


# # For volatility surface visualization
# model_visualization.plot_deepar_volatility_surface(surface)

# # To compare with XGBoost
# compare_model_performance(
#     ohlcv=ohlcv,
#     xgboost_results=xgboost_results,
#     deepar_results=deepar_results,
#     tickers=["AAPL", "MSFT", "GOOG"]
# )


importlib.reload(nbeats_modeling)
# After running your data through the transformation pipeline
results = nbeats_modeling.train_multi_horizon_volatility_models(
    df=ohlcv,  
    # All three horizons in one model
    target_cols=['YZVol_10_future', 'YZVol_30_future', 'YZVol_90_future'],
    transform_target=True,
    lookback_window=60,
    batch_size=64,
    num_epochs=30,
    include_specialized_stacks=True 
)
# Results are automatically saved during training, but you can save explicitly too
nbeats_modeling.save_models(results)


# Train the TFT models
tft_results = tft_modeling.train_paired_tft_models(
    df=ohlcv,  
    target_horizons=[10,30,90],  # You can specify multiple horizons: [10, 30, 90]
    train_ratio=0.5,  # Same 50/50 split
    # TFT-specific parameters (with reasonable defaults)
    sequence_length=90,  # Use 30 days of history for each prediction
    batch_size=64,
    learning_rate=0.001,
    epochs=50,  # Start with 50, increase for better performance
    hidden_size=128,
    lstm_layers=2,
    attention_heads=4,
    dropout=0.1
)

# Save the TFT models
model_path = tft_modeling.save_tft_models(tft_results)
print(f"TFT models saved to: {model_path}")


importlib.reload(xgboost_modeling)

# After running your data through the transformation pipeline
results = xgboost_modeling.train_paired_volatility_models(
    df=ohlcv,  
    target_col='YZVol_90_future', 
    train_ratio=0.5  # 50/50 split
)

xgboost_modeling.save_models(results)


results = xgboost_modeling.load_models()


# Reload the module to get the latest changes
importlib.reload(model_visualization)

# # Analyze a single stock (PFG in this example)
# results_viz = xgboost_visualization.evaluate_volatility_predictors(
#     ohlcv=ohlcv,
#     ticker="AAPL",
#     model_results=results,  # This is your existing model results from train_paired_volatility_models
#     start_date=pl.date(2020, 1, 1),
#     end_date=pl.date(2021, 9, 1),
#     target_vol="YZVol_30_future",
#     model_name="XGBoost"
# )
# xgboost_visualization.plot_volatility_comparison(results_viz)

# To analyze multiple stocks at once (uncomment to use)
model_visualization.analyze_multiple_stocks(
    ohlcv=ohlcv,
    model_results=results,
    tickers=["AAPL", "XOM", "JPM", "TSLA", "NVDA", "PG", "UNH", "BA", "NKE", "KO"],
    start_date=pl.date(2020, 1, 1),
    end_date=pl.date(2021, 9, 1),
    target_vol="YZVol_90_future",
)


# Determine which model was used for AAPL
if "AAPL" in results['stocks_A']:
    print("AAPL is in Stock Group A - examining Model 2")
    model_to_use = results['model_2']
else:
    print("AAPL is in Stock Group B - examining Model 1")
    model_to_use = results['model_1']

feature_cols = results['feature_cols']

# 1. Extract feature importance for this stock's model
importance = model_to_use.feature_importances_
feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

print("\nTop 15 features for AAPL's prediction model:")
for feature, imp in feature_importance[:15]:
    print(f"{feature}: {imp:.6f}")

# 2. Visualize feature importance
plt.figure(figsize=(12, 8))
features = [f[0] for f in feature_importance[:15]]
importances = [f[1] for f in feature_importance[:15]]
plt.barh(features, importances)
plt.xlabel('Importance')
plt.title('Top 15 Features for AAPL Prediction')
plt.gca().invert_yaxis()  # Highest at top
plt.tight_layout()
plt.show()

# 3. Extract a sample tree from the ensemble
try:
    # Get the model's underlying booster
    if hasattr(model_to_use, 'get_booster'):
        booster = model_to_use.get_booster()
    else:
        booster = model_to_use
    
    # Extract the first tree (index 0) as text
    tree_text = booster.get_dump(dump_format='text')[0]
    
    print("\nSample decision tree from the ensemble (first tree):")
    # Print first 20 lines of the tree to avoid overwhelming output
    print('\n'.join(tree_text.split('\n')[:20]))
    print("... (tree continues)")
    
except Exception as e:
    print(f"Could not extract tree structure: {str(e)}")
    print("This might be due to XGBoost version differences.")


# This requires: pip install shap
import shap

shap.initjs()

# Filter for just AAPL stock data
aapl_data = ohlcv.filter(pl.col("act_symbol") == "AAPL")

# Create features dataframe
aapl_features = aapl_data.select(feature_cols)

# Handle infinite values
for col in feature_cols:
    aapl_features = aapl_features.with_columns([
        pl.when(pl.col(col).is_infinite())
        .then(None)
        .otherwise(pl.col(col))
        .alias(col)
    ])

# Fill nulls and convert to numpy array
X_aapl = aapl_features.fill_null(0).to_numpy()

# Create explainer
explainer = shap.TreeExplainer(model_to_use)

# Calculate SHAP values for AAPL data
shap_values = explainer.shap_values(X_aapl)

# Show summary plot
shap.summary_plot(shap_values, X_aapl, feature_names=feature_cols)

# For a single prediction point (most recent date)
shap.force_plot(
    explainer.expected_value, 
    shap_values[-1], 
    X_aapl[-1], 
    feature_names=feature_cols
)


from src.data_modeling import volatility_clustering
importlib.reload(volatility_clustering)

# Step 1: Find optimal parameters (this will generate heatmaps to visualize parameter choices)
best_eps, best_min_samples, n_clusters, noise_pct = volatility_clustering.find_optimal_eps_and_clusters(
    ohlcv,
    n_components=6,
    min_data_pct=95,  # Increase to require more data coverage per stock
    max_noise_pct=20,  # Maximum acceptable noise percentage
    min_clusters=3,  # Minimum number of clusters desired
)

# Step 2: Run clustering with optimal parameters (only if good parameters were found)
if best_eps is not None:
    cluster_df, pca_df, pca = volatility_clustering.cluster_stocks_by_timeseries(
        ohlcv,
        start_date="2012-01-01",
        end_date="2024-08-01",
        min_data_pct=95,
        n_components=6,
        algorithm='dbscan',  # Specify DBSCAN algorithm
        eps=best_eps,
        min_samples=best_min_samples
    )

else:
    # Fall back to K-means if no good DBSCAN parameters were found
    print("No good DBSCAN parameters found, using K-means instead")
    cluster_df, pca_df, pca = volatility_clustering.cluster_stocks_by_timeseries(
        ohlcv,
        start_date="2012-01-01",
        end_date="2024-08-01",
        min_data_pct=95,
        n_components=6,
        algorithm='kmeans',
        n_clusters=8  # Try different values here
    )

volatility_clustering.plot_stock_clusters(cluster_df, pca)


cluster_df, pca_df, pca = volatility_clustering.cluster_stocks_by_timeseries(
        ohlcv,
        start_date="2012-01-01",
        end_date="2024-08-01",
        min_data_pct=95,
        n_components=6,
        algorithm='kmeans',
        n_clusters=4  # Try different values here
    )

# Visualize the results
volatility_clustering.plot_stock_clusters(cluster_df, pca)


pca_df.filter(pl.col("act_symbol") == "AAPL").head(10)
cluster_df.filter(pl.col("act_symbol") == "AAPL").head(10)


# import src.data_modeling.volatility_clustering as vc
# importlib.reload(vc)

# # Load your processed data
# df = ohlcv

# # Define features for clustering
# volatility_features = [
#     'log_returns',
#     'YZVol_10',
#     'YZVol_30',
#     'YZVol_90'
# ]

# # Step 1: Find optimal eps value
# _, distance_matrix, tickers = vc.multi_feature_temporal_clustering(
#     df, 
#     features=volatility_features,
#     return_distance_matrix=True
# )

# sorted_distances = vc.find_optimal_eps(distance_matrix, k=4)

# # Step 2: Perform clustering with optimal parameters
# ticker_clusters = vc.multi_feature_temporal_clustering(
#     df,
#     features=volatility_features,
#     eps=0.5,  # Use value from k-distance plot
#     min_samples=5
# )

# # Step 3: Visualize results
# vc.visualize_clusters(df, ticker_clusters, feature='YZVol_30')

# # Optional: Perform parameter sweep to find best parameters
# parameter_results = vc.parameter_sweep(
#     df,
#     features=volatility_features,
#     eps_values=[0.3, 0.5, 0.7, 1.0],
#     min_samples_values=[3, 5, 10]
# )


## Above is an implementation of a volatility clustering approach. It takes too long to compute, don't run this.


# ## Apple Example
# Here, we are filtering our data so we are only looking at AAPL data from 2020.  
# We will construct some simple plots.

aapl = ohlcv.filter(pl.col("act_symbol") == "AAPL").sort("date")

aapl = aapl.filter(pl.col("date") >= pl.date(2012, 1, 1))

aapl.head(10)


aapl_date = aapl['date'].to_list()
aapl_close = aapl['close'].to_list()
aapl_volume = aapl['volume'].to_list()
aapl_log_returns = aapl['log_returns'].to_list()
aapl_sma_30 = aapl['SMA_30'].to_list()
aapl_ema_30 = aapl['EMA_30'].to_list()
aapl_std_30 = aapl['STD_30'].to_list()
aapl_rmi_30 = aapl['RSI_30'].to_list()
aapl_macd = aapl['vol_macd_line'].to_list()
aapl_signal_line = aapl['vol_signal_line'].to_list()
aapl_macd_histogram = aapl['vol_macd_hist'].to_list()
aapl_donchian_30 = aapl['DonchianRange_30'].to_list()
aapl_parkinson_10 = aapl['ParkinsonVol_10'].to_list()
aapl_parkinson_30 = aapl['ParkinsonVol_30'].to_list()
aapl_parkinson_90 = aapl['ParkinsonVol_90'].to_list()
aapl_yz_10 = aapl['YZVol_10'].to_list()
aapl_yz_30 = aapl['YZVol_30'].to_list()
aapl_yz_90 = aapl['YZVol_90'].to_list()
aapl_vix = aapl['vix'].to_list()
aapl_vxapl = aapl['vxapl'].to_list()

# Create BB_upper and BB_lower
aapl_bb_upper_30 = [sma + 2 * std for sma, std in zip(aapl_sma_30, aapl_std_30)]
aapl_bb_lower_30 = [sma - 2 * std for sma, std in zip(aapl_sma_30, aapl_std_30)]


# Create a figure with 6 subplots (3 rows, 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Plot 1: Close Price with SMA, EMA, and Bollinger Bands
ax = axes[0, 0]
ax.plot(aapl_date, aapl_close, label='Close Price', color='blue', linewidth=1.5)
ax.plot(aapl_date, aapl_sma_30, label='SMA 30', color='orange', linewidth=1)
ax.plot(aapl_date, aapl_ema_30, label='EMA 30', color='green', linewidth=1)
ax.fill_between(aapl_date, aapl_bb_lower_30, aapl_bb_upper_30, color='grey', alpha=0.3, label='Bollinger Bands')
ax.set_title("AAPL Price with MAs & Bollinger Bands")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
ax.grid(True)

# Plot 2: Daily Volume
ax = axes[0, 1]
ax.plot(aapl_date, aapl_volume, color='purple')
ax.set_title("AAPL Daily Volume")
ax.set_xlabel("Date")
ax.set_ylabel("Volume")
ax.grid(True)

# Plot 3: Log Returns
ax = axes[1, 0]
ax.plot(aapl_date, aapl_log_returns, color='red', linewidth=1)
ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
ax.set_title("AAPL Log Returns")
ax.set_xlabel("Date")
ax.set_ylabel("Log Return")
ax.grid(True)

# Plot 4: MACD and Signal Line
ax = axes[1, 1]
ax.plot(aapl_date, aapl_macd, label='MACD', color='blue', linewidth=1.5)
ax.plot(aapl_date, aapl_signal_line, label='Signal Line', color='orange', linewidth=1.5)
ax.bar(aapl_date, aapl_macd_histogram, label='MACD Histogram', color='grey', alpha=0.3)
ax.set_title("AAPL MACD Indicator")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)

# Plot 5: RMI (Relative Momentum Index)
ax = axes[2, 0]
ax.plot(aapl_date, aapl_rmi_30, label='RMI 21', color='magenta', linewidth=1.5)
ax.set_title("AAPL RMI (30)")
ax.set_xlabel("Date")
ax.set_ylabel("RMI")
ax.legend()
ax.grid(True)

# Plot 6: VxAPL (Volatility Indicator)
ax = axes[2, 1]
ax.plot(aapl_date, aapl_vxapl, label='VxAPL', color='brown', linewidth=1.5)
ax.plot(aapl_date, aapl_vix, label='VIX', color='grey', linewidth=1.5)
ax.plot(aapl_date, aapl_close, label='Close Price', color='blue', linewidth=1.5)
ax.set_title("AAPL VxAPL")
ax.set_xlabel("Date")
ax.set_ylabel("VxAPL")
ax.legend()
ax.grid(True)

plt.tight_layout()
plt.show()


volatility_history = pl.read_csv("data/raw/options/csv/volatility_history.csv")

# Adjust date column format
volatility_history = volatility_history.with_columns(
    pl.col("date").str.to_date("%Y-%m-%d")
)

# Filter to only include AAPL
volatility_history = volatility_history.filter(pl.col("act_symbol") == "AAPL").filter(pl.col("date") >= pl.date(2012, 1, 1))


fig, ax = plt.subplots(figsize=(15, 7))

ax.plot(aapl_date, aapl_vxapl, label='VxAPL', color='green', linewidth=1.5)
ax.plot(aapl_date, aapl_vix, label='VIX', color='darkgreen', linewidth=1.5)
ax.plot(aapl_date, aapl_close, label='Close Price', color='black', linewidth=1.5)
ax.plot(aapl_date, aapl_donchian_30, label='Donchian', color='brown', linewidth=1.5)


# Create a secondary y-axis for the HV and IV data
ax2 = ax.twinx()

# Extract hv_current and iv_current
hv = np.array(volatility_history['hv_current'].to_list())
iv = np.array(volatility_history['iv_current'].to_list())

# Get the corresponding dates for volatility_history
vol_dates = volatility_history['date'].to_list()

# Plot the HV and IV on the secondary axis
ax2.plot(vol_dates, hv, label='HV Current', color='orange', linewidth=1.5)
ax2.plot(vol_dates, iv, label='IV Current', color='purple', linewidth=1.5)
ax2.plot(aapl_date, aapl_parkinson_10, label='Parkinsons 10', color='lightgrey', linewidth=1.5)
ax2.plot(aapl_date, aapl_parkinson_30, label='Parkinsons 30', color='grey', linewidth=1.5)
ax2.plot(aapl_date, aapl_parkinson_90, label='Parkinsons 90', color='darkgrey', linewidth=1.5)
ax2.plot(aapl_date, aapl_yz_10, label='YZ 10', color='lightblue', linewidth=1.5)
ax2.plot(aapl_date, aapl_yz_30, label='YZ 30', color='blue', linewidth=1.5)
ax2.plot(aapl_date, aapl_yz_90, label='YZ 90', color='darkblue', linewidth=1.5)
ax2.set_ylabel("HV/IV")

# Combine legends from both axes
lines, labels = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines + lines2, labels + labels2, loc='upper left')

ax.set_title("AAPL VxAPL, VIX, Parkinson's and Yang-Zhang 10, 30, and 90 day, HV Current, IV Current, and Close Price")
ax.set_xlabel("Date")
ax.set_ylabel("VIX")
ax.grid(True)
plt.tight_layout()
plt.show()


# Define the window suffixes to check
windows = [10, 30, 90]
window_groups = {}

for w in windows:
    # Exclude indicators containing other window suffixes
    excluded_windows = [f"_{other_w}" for other_w in windows if other_w != w]
    
    window_groups[w] = [
        col for col in aapl.columns
        if not any(excl in col for excl in excluded_windows) 
        and not col.startswith("date") 
        and not col.startswith("act_symbol")
    ]

print("Indicator groups by window:")
for w in windows:
    print(f"Window {w}: {window_groups[w]}")

stock_symbol = aapl['act_symbol'][0]

# Compute the correlation matrix and plot the heatmap.
for w in windows:
    indicators = window_groups[w]
    if len(indicators) == 0:
        continue  # Skip if no indicators for this window.
    
    stock_data_group = aapl.drop_nulls(indicators)
    if stock_data_group.height == 0:
        print(f"No sufficient data for window {w} indicators.")
        continue

    # Compute correlation matrix (Polars returns a DataFrame; convert to NumPy for plotting)
    corr_matrix = stock_data_group.select(indicators).corr().to_numpy()

    plt.figure(figsize=(16, 14))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='RdBu_r',
                xticklabels=indicators, yticklabels=indicators,
                vmin=-1, vmax=1)
    plt.title(f"Correlation Matrix for {stock_symbol} ({w}-day Window Indicators)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Optionally, print the top 15 most correlated indicator pairs for this window.
    pairs = []
    for i in range(len(indicators)):
        for j in range(i + 1, len(indicators)):
            pairs.append((indicators[i], indicators[j], corr_matrix[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    print(f"\nTop 15 correlated pairs for window {w}:")
    for feat1, feat2, corr in pairs[:15]:
        print(f"{feat1} <-> {feat2}: {corr:.4f}")


# Get actual column names that exist in the dataframe
indicators_to_plot = [col for col in aapl.columns
                      if not col.startswith("date") 
                      and not col.startswith("act_symbol")]

# Drop rows with null values in the selected columns
stock_data = aapl.drop_nulls(indicators_to_plot)

# Extract dates for time series plotting
dates = stock_data["date"].to_numpy()

# Loop over each indicator and produce histogram and time series plot
for col in indicators_to_plot:
    # Convert the column to a NumPy array
    data_series = stock_data[col].to_numpy()
    
    # Create a figure with two subplots: histogram and time series
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    
    # Histogram: Plot the distribution
    axs[0].hist(data_series, bins=30, density=True, alpha=0.7, color='skyblue')
    axs[0].set_title(f"Histogram of {col}")
    axs[0].set_xlabel(f"{col} Value")
    axs[0].set_ylabel("Frequency")
    axs[0].grid(True, alpha=0.3)
    
    # Time Series: Plot the indicator values over time
    axs[1].plot(dates, data_series, linewidth=1.5, color='darkblue')
    axs[1].set_title(f"{col} Over Time")
    axs[1].set_xlabel("Date")
    axs[1].set_ylabel(f"{col} Value")
    axs[1].grid(True, alpha=0.3)
    
    # Format the date axis for better readability
    fig.autofmt_xdate()
    
    plt.suptitle(f"{stock_symbol}: {col} Analysis (2020)")
    plt.tight_layout()
    plt.show()

