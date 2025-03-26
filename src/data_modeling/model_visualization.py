import numpy as np
import matplotlib.pyplot as plt
import polars as pl
import datetime
from typing import List, Dict, Union, Optional, Tuple
from matplotlib import cm
import matplotlib.dates as mdates

# Import LGBM modeling functions
from src.data_modeling import lgbm_modeling

def compare_predictions_vs_actual(
    model_results: Dict,
    df: Optional[pl.DataFrame] = None,
    predictions_df: Optional[pl.DataFrame] = None,
    stock: str = "AAPL",
    start_date: Union[str, datetime.date] = "2020-01-01",
    end_date: Union[str, datetime.date] = "2020-12-31",
    metric: str = "YZVol",
    windows: Optional[List[int]] = None,
    stride: int = 1
) -> Dict:
    """
    Compare model predictions against actual values and other volatility metrics.
    
    Parameters:
    -----------
    model_results : Dict
        Results dictionary from LGBM model training
    df : pl.DataFrame, optional
        DataFrame with actual values (required if predictions_df not provided)
    predictions_df : pl.DataFrame, optional
        DataFrame with model predictions (will be generated from df if not provided)
    stock : str
        Stock symbol to analyze
    start_date : Union[str, datetime.date]
        Start date for analysis
    end_date : Union[str, datetime.date]
        End date for analysis
    metric : str
        Metric to analyze (without window numbers or _future suffix)
    windows : List[int], optional
        List of time windows to include (default: all available)
    stride : int
        Stride for prediction (1 for daily, higher for sparser predictions)
        
    Returns:
    --------
    Dict
        Results dictionary with comparisons and metrics
    """
    # Normalize date formats
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Get predictions if not provided
    if predictions_df is None:
        if df is None:
            raise ValueError("Either df or predictions_df must be provided")
        
        # Calculate lookback period (30 days before start_date for context)
        max_timepoint = max(model_results.get('timepoints', [10]))
        lookback_days = datetime.timedelta(days=max_timepoint)
        context_start_date = start_date - lookback_days
        
        print(f"Generating predictions for {stock} from {context_start_date} to {end_date}...")
        print(f"(Using data from {context_start_date} to {start_date} for context)")
        
        predictions_df = lgbm_modeling.generate_volatility_predictions(
            model_results=model_results,
            df=df,
            stocks=[stock],
            start_date=context_start_date,  # Start earlier to get context
            end_date=end_date,
            stride=stride
        )
    
    # Filter predictions for the stock and date range
    # For visualization, we only want from start_date to end_date
    stock_predictions = predictions_df.filter(
        (pl.col("act_symbol") == stock) &
        (pl.col("date") >= start_date) &  # Use requested start_date for visualization
        (pl.col("date") <= end_date)
    ).sort("date")
    
    if stock_predictions.height == 0:
        raise ValueError(f"No predictions found for {stock} in the specified date range")
    
    # Determine available windows if not specified
    if windows is None:
        # Find all columns matching the pattern {metric}_{window}_future(_pred)
        all_cols = stock_predictions.columns
        available_windows = []
        
        for col in all_cols:
            # Check for prediction columns
            if col.startswith(f"{metric}_") and col.endswith("_future_pred"):
                try:
                    # Extract window from column name
                    window_str = col.replace(f"{metric}_", "").replace("_future_pred", "")
                    window = int(window_str)
                    available_windows.append(window)
                except ValueError:
                    continue
                    
            # Also check for actual columns
            elif col.startswith(f"{metric}_") and col.endswith("_future"):
                try:
                    # Extract window from column name
                    window_str = col.replace(f"{metric}_", "").replace("_future", "")
                    window = int(window_str)
                    available_windows.append(window)
                except ValueError:
                    continue
        
        # Use unique and sorted windows
        windows = sorted(list(set(available_windows)))
    
    if not windows:
        raise ValueError(f"No available windows found for metric {metric}")
    
    print(f"Analyzing {len(windows)} windows for {metric}: {windows}")
    
    # Collect actual values, predictions, and alternative metrics for each window
    results = {
        "stock": stock,
        "metric": metric,
        "dates": stock_predictions["date"].to_numpy(),
        "windows": windows,
        "comparisons": {}
    }
    
    for window in windows:
        # Column names
        actual_col = f"{metric}_{window}_future"
        pred_col = f"{metric}_{window}_future_pred"
        
        # Skip window if either actual or predicted column is missing
        if actual_col not in stock_predictions.columns or pred_col not in stock_predictions.columns:
            print(f"Skipping window {window} - missing data")
            continue
            
        # Get actual and predicted values
        actual_values = stock_predictions[actual_col].to_numpy()
        pred_values = stock_predictions[pred_col].to_numpy()
        
        # Remove any NaN or infinite values
        valid_mask = (~np.isnan(actual_values)) & (~np.isnan(pred_values)) & \
                     (~np.isinf(actual_values)) & (~np.isinf(pred_values))
        
        if np.sum(valid_mask) < 5:  # Require at least 5 valid points
            print(f"Skipping window {window} - not enough valid data points")
            continue
            
        # Store values
        window_results = {
            "actual": actual_values[valid_mask],
            "model_pred": pred_values[valid_mask],
            "dates": results["dates"][valid_mask],
            "alternatives": {},
            "metrics": {}
        }
        
        # Find alternative volatility metrics to compare against
        alternatives = []
        
        # 1. Current volatility (same metric without _future)
        current_col = f"{metric}_{window}"
        if current_col in stock_predictions.columns:
            alternatives.append(current_col)
            
        # 2. Find common volatility metrics
        # - Parkinson volatility
        parkinson_col = f"ParkinsonVol_{window}"
        if parkinson_col in stock_predictions.columns:
            alternatives.append(parkinson_col)
            
        # 3. CBOE volatility indices (if present)
        for cboe_index in ["vix", "vxapl", "vxazn", "vvix", "gvz", "ovx"]:
            if cboe_index in stock_predictions.columns:
                alternatives.append(cboe_index)
        
        # Add unique alternatives found in the data
        for col in stock_predictions.columns:
            # Add any other YZ volatility windows
            if col.startswith("YZVol_") and col != current_col and not col.endswith("_future") and not col.endswith("_pred"):
                alternatives.append(col)
                
            # Add other relevant metrics like ATR
            if col.startswith("ATR_"):
                alternatives.append(col)
        
        # Use a set to remove duplicates, then convert back to list
        alternatives = list(set(alternatives))
        
        # Extract alternative metrics
        for alt_col in alternatives:
            alt_values = stock_predictions[alt_col].to_numpy()
            alt_values = alt_values[valid_mask]  # Apply same mask
            
            # Skip if all values are NaN or Inf
            if np.all(np.isnan(alt_values)) or np.all(np.isinf(alt_values)):
                continue
                
            # Handle CBOE indices (convert from percentage to decimal)
            if alt_col in ["vix", "vxapl", "vxazn", "vvix", "gvz", "ovx"]:
                alt_values = alt_values / 100.0
                
            # Calculate scaling factor to match target scale
            alt_mean = np.nanmean(alt_values)
            target_mean = np.nanmean(window_results["actual"])
            
            if alt_mean > 0:
                scale_factor = target_mean / alt_mean
            else:
                scale_factor = 1.0
                
            # Store scaled values
            window_results["alternatives"][alt_col] = {
                "original": alt_values.copy(),
                "scaled": alt_values * scale_factor,
                "scale_factor": scale_factor
            }
        
        # Calculate performance metrics
        window_results["metrics"] = calculate_performance_metrics(
            window_results["model_pred"],
            window_results["actual"],
            window_results["alternatives"]
        )
        
        # Store in results
        results["comparisons"][window] = window_results
    
    # Calculate overall win rate
    if results["comparisons"]:
        total_wins = 0
        total_comparisons = 0
        
        for window, window_results in results["comparisons"].items():
            metrics = window_results["metrics"]
            if "win_rate" in metrics:
                total_wins += metrics["win_rate"] * 100  # Convert to percentage points
                total_comparisons += 1
        
        if total_comparisons > 0:
            results["overall_win_rate"] = total_wins / total_comparisons
        else:
            results["overall_win_rate"] = 0.0
    
    # Visualize results
    plot_prediction_comparison(results)
    plot_win_loss_chart(results)
    
    return results

def calculate_performance_metrics(
    model_predictions: np.ndarray, 
    actual_values: np.ndarray, 
    alternatives: Dict
) -> Dict:
    """
    Calculate performance metrics for model predictions and alternatives.
    
    Parameters:
    -----------
    model_predictions : np.ndarray
        Model predictions
    actual_values : np.ndarray
        Actual values
    alternatives : Dict
        Dictionary of alternative metrics
        
    Returns:
    --------
    Dict
        Performance metrics
    """
    metrics = {}
    
    # Calculate model metrics
    model_rmse = np.sqrt(np.mean((model_predictions - actual_values) ** 2))
    model_mae = np.mean(np.abs(model_predictions - actual_values))
    model_mape = np.mean(np.abs((actual_values - model_predictions) / np.maximum(0.001, np.abs(actual_values)))) * 100
    model_corr = np.corrcoef(model_predictions, actual_values)[0, 1]
    
    metrics["rmse"] = model_rmse
    metrics["mae"] = model_mae
    metrics["mape"] = model_mape
    metrics["correlation"] = model_corr
    
    # Calculate daily errors
    model_daily_errors = np.abs(model_predictions - actual_values)
    
    # Compare against alternatives
    alt_metrics = {}
    model_win_rates = []
    
    for alt_name, alt_data in alternatives.items():
        alt_values = alt_data["scaled"]
        alt_daily_errors = np.abs(alt_values - actual_values)
        
        # Calculate metrics
        alt_rmse = np.sqrt(np.mean((alt_values - actual_values) ** 2))
        alt_mae = np.mean(np.abs(alt_values - actual_values))
        alt_mape = np.mean(np.abs((actual_values - alt_values) / np.maximum(0.001, np.abs(actual_values)))) * 100
        alt_corr = np.corrcoef(alt_values, actual_values)[0, 1]
        
        # Calculate win/loss
        model_wins = np.sum(model_daily_errors < alt_daily_errors)
        model_losses = np.sum(model_daily_errors > alt_daily_errors)
        ties = np.sum(np.isclose(model_daily_errors, alt_daily_errors))
        
        # Win rate calculation
        total_days = len(model_daily_errors)
        model_win_rate = model_wins / total_days if total_days > 0 else 0
        
        # Store metrics
        alt_metrics[alt_name] = {
            "rmse": alt_rmse,
            "mae": alt_mae,
            "mape": alt_mape,
            "correlation": alt_corr,
            "model_wins": model_wins,
            "model_losses": model_losses,
            "ties": ties,
            "win_rate": model_win_rate
        }
        
        # Collect win rates for averaging
        model_win_rates.append(model_win_rate)
    
    # Average win rate
    if model_win_rates:
        metrics["win_rate"] = np.mean(model_win_rates)
    else:
        metrics["win_rate"] = 0.5  # Neutral if no alternatives
    
    metrics["alternatives"] = alt_metrics
    
    return metrics

def plot_prediction_comparison(results: Dict):
    """
    Plot model predictions against actual values and alternatives.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from compare_predictions_vs_actual
    """
    # Get basic info
    stock = results["stock"]
    metric = results["metric"]
    windows = results["windows"]
    
    # Create a figure with subplots for each window
    n_windows = len(results["comparisons"])
    if n_windows == 0:
        print("No valid windows to plot")
        return
    
    # Determine grid size based on number of windows
    if n_windows <= 3:
        fig, axes = plt.subplots(n_windows, 1, figsize=(12, 4 * n_windows))
        if n_windows == 1:
            axes = [axes]  # Make it iterable for consistency
    else:
        n_cols = min(2, n_windows)
        n_rows = (n_windows + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
        # Flatten axes array for easy iteration
        axes = axes.flatten() if n_windows > 1 else [axes]
    
    # Plot each window's data
    plot_idx = 0
    for window, window_results in results["comparisons"].items():
        if plot_idx >= len(axes):
            print(f"Warning: More windows than plot axes, skipping window {window}")
            continue
            
        ax = axes[plot_idx]
        
        # Plot actual values
        ax.plot(window_results["dates"], window_results["actual"], 'b-', 
                label=f'Actual {metric}_{window}_future', linewidth=2)
        
        # Plot model predictions
        ax.plot(window_results["dates"], window_results["model_pred"], 'r--', 
                label=f'Model Prediction', linewidth=1.5)
        
        # Plot top alternatives (limit to 3 to avoid cluttering)
        alt_metrics = window_results["metrics"]["alternatives"]
        sorted_alts = sorted(alt_metrics.items(), key=lambda x: x[1]["rmse"])
        
        colors = ['g', 'c', 'm']
        for i, (alt_name, alt_metric) in enumerate(sorted_alts[:3]):
            alt_data = window_results["alternatives"][alt_name]
            label = f"{alt_name} (scaled)" if "scale_factor" in alt_data else alt_name
            ax.plot(window_results["dates"], alt_data["scaled"], 
                    color=colors[i % len(colors)], linestyle='-', 
                    alpha=0.6, linewidth=1, label=label)
        
        # Set title and labels
        ax.set_title(f'{stock} {metric}_{window} - Prediction vs Actual')
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        
        # Add metrics annotation
        model_rmse = window_results["metrics"]["rmse"]
        model_mae = window_results["metrics"]["mae"]
        model_win_rate = window_results["metrics"]["win_rate"] * 100 if "win_rate" in window_results["metrics"] else 0
        
        metrics_text = f"Model RMSE: {model_rmse:.4f}\nModel MAE: {model_mae:.4f}\nWin Rate: {model_win_rate:.1f}%"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add legend
        ax.legend(loc='lower right')
        
        # Format dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        fig.autofmt_xdate()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plot_idx += 1
    
    # Hide any unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].axis('off')
    
    # Overall title
    fig.suptitle(f'{stock} {metric} Volatility Predictions - Windows: {windows}', fontsize=16)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_win_loss_chart(results: Dict):
    """
    Plot win/loss charts for model vs alternative metrics, showing per-window details.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from compare_predictions_vs_actual
    """
    # Get basic info
    stock = results["stock"]
    metric = results["metric"]
    windows = sorted(results["comparisons"].keys())
    
    if not windows:
        print("No valid windows to plot")
        return
    
    # Create one figure per window to show detailed win rates
    for window in windows:
        window_results = results["comparisons"][window]
        alt_metrics = window_results["metrics"]["alternatives"]
        
        # Skip if no alternatives
        if not alt_metrics:
            print(f"No alternatives to compare for window {window}")
            continue
        
        # Sort alternatives by win rate
        sorted_alts = sorted(alt_metrics.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        alt_names = [alt for alt, _ in sorted_alts]
        alt_rates = [metrics["win_rate"] * 100 for _, metrics in sorted_alts]  # Convert to percentage
        
        # Choose colors based on win rate
        alt_colors = []
        for rate in alt_rates:
            if rate >= 60:  # Strong win
                alt_colors.append('green')
            elif rate >= 50:  # Marginal win
                alt_colors.append('lightgreen')
            elif rate >= 40:  # Marginal loss
                alt_colors.append('salmon')
            else:  # Strong loss
                alt_colors.append('red')
        
        # Create horizontal bars
        fig, ax = plt.subplots(figsize=(10, max(6, len(alt_names) * 0.4)))
        y_pos = np.arange(len(alt_names))
        bar_container = ax.barh(y_pos, alt_rates, color=alt_colors)
        
        # Add vertical line at 50%
        ax.axvline(x=50, linestyle='--', color='gray', alpha=0.7)
        
        # Annotate each bar with its value
        for bar, rate in zip(bar_container, alt_rates):
            width = bar.get_width()
            ax.text(min(width + 1, 95), bar.get_y() + bar.get_height()/2.,
                   f'{rate:.1f}%', ha='left' if width < 90 else 'right', va='center')
        
        # Set title and labels
        window_rmse = window_results["metrics"]["rmse"]
        window_mae = window_results["metrics"]["mae"]
        window_win_rate = window_results["metrics"]["win_rate"] * 100
        
        title = f'{stock} {metric}_{window} Model Performance\n'
        title += f'RMSE: {window_rmse:.4f}, MAE: {window_mae:.4f}, Avg Win Rate: {window_win_rate:.1f}%'
        ax.set_title(title)
        
        ax.set_xlabel('Win Rate (%)')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(alt_names)
        ax.set_xlim(0, 100)
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    # Create summary figure with win rates by window
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Collect win rates for each window
    win_rates = []
    for window in windows:
        if window in results["comparisons"]:
            win_rates.append(results["comparisons"][window]["metrics"]["win_rate"] * 100)
        else:
            win_rates.append(0)
    
    # Color bars by win rate
    colors = []
    for rate in win_rates:
        if rate >= 60:  # Strong win
            colors.append('green')
        elif rate >= 50:  # Marginal win
            colors.append('lightgreen')
        elif rate >= 40:  # Marginal loss
            colors.append('salmon')
        else:  # Strong loss
            colors.append('red')
    
    # Plot bars
    bar_container = ax.bar(windows, win_rates, color=colors)
    
    # Add a horizontal line at 50%
    ax.axhline(y=50, linestyle='--', color='gray', alpha=0.7)
    
    # Annotate each bar with its value
    for bar, rate in zip(bar_container, win_rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
               f'{rate:.1f}%', ha='center', va='bottom')
    
    # Set title and labels
    overall_win_rate = results.get("overall_win_rate", 0.0)
    ax.set_title(f'{stock} {metric} Overall Model Performance - Avg Win Rate: {overall_win_rate:.1f}%')
    ax.set_xlabel('Window (Days)')
    ax.set_ylabel('Win Rate (%)')
    ax.set_ylim(0, 100)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_error_distribution(results: Dict):
    """
    Plot the distribution of prediction errors.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from compare_predictions_vs_actual
    """
    # Get basic info
    stock = results["stock"]
    metric = results["metric"]
    
    # Collect all errors
    all_errors = []
    window_labels = []
    
    for window, window_results in results["comparisons"].items():
        # Calculate errors
        errors = window_results["model_pred"] - window_results["actual"]
        all_errors.append(errors)
        window_labels.append(f"{window}-day")
    
    if not all_errors:
        print("No error data to plot")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. Error histograms
    n_windows = len(all_errors)
    colors = plt.cm.viridis(np.linspace(0, 0.9, n_windows))
    
    for i, (errors, label) in enumerate(zip(all_errors, window_labels)):
        # Plot histogram
        n, bins, patches = ax1.hist(errors, bins=20, alpha=0.7, 
                                    color=colors[i], label=label, density=True)
        
        # Add vertical line at 0 (perfect prediction)
        if i == 0:  # Only add once
            ax1.axvline(x=0, color='red', linestyle='--', alpha=0.8)
    
    # Set title and labels
    ax1.set_title('Error Distribution by Window')
    ax1.set_xlabel('Prediction Error (Model - Actual)')
    ax1.set_ylabel('Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Box plots
    box_data = all_errors
    ax2.boxplot(box_data, labels=window_labels, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.8))
    
    # Add horizontal line at 0
    ax2.axhline(y=0, color='red', linestyle='--', alpha=0.8)
    
    # Set title and labels
    ax2.set_title('Error Distribution Summary')
    ax2.set_xlabel('Window')
    ax2.set_ylabel('Prediction Error')
    ax2.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle(f'{stock} {metric} Prediction Error Analysis', fontsize=16)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
    
    # Also plot error time series
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for i, (errors, label, window_results) in enumerate(zip(all_errors, window_labels, results["comparisons"].values())):
        # Plot error time series
        ax.plot(window_results["dates"], errors, '-', color=colors[i], label=label, alpha=0.8)
    
    # Add horizontal line at 0
    ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Set title and labels
    ax.set_title(f'{stock} {metric} Prediction Error Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Prediction Error (Model - Actual)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Format dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    fig.autofmt_xdate()
    
    plt.tight_layout()
    plt.show()

def generate_performance_summary_table(
    model_results: Dict,
    df: pl.DataFrame,
    stocks: List[str],
    metrics: List[str] = ["YZVol"],
    windows: Optional[List[int]] = None,
    start_date: Union[str, datetime.date] = "2020-01-01",
    end_date: Union[str, datetime.date] = "2020-12-31",
    stride: int = 10
) -> Tuple[pl.DataFrame, pl.DataFrame]:
    """
    Generate a comprehensive performance summary table across multiple stocks, metrics, and windows.
    
    Parameters:
    -----------
    model_results : Dict
        Results dictionary from LGBM model training
    df : pl.DataFrame
        DataFrame with actual values
    stocks : List[str]
        List of stock symbols to analyze
    metrics : List[str]
        List of metrics to analyze
    windows : Optional[List[int]]
        List of time windows to include (if None, will find all available windows)
    start_date : Union[str, datetime.date]
        Start date for analysis
    end_date : Union[str, datetime.date]
        End date for analysis
    stride : int
        Stride for prediction
        
    Returns:
    --------
    Tuple[pl.DataFrame, pl.DataFrame]
        Summary table with performance metrics and aggregated statistics
    """
    # Normalize date formats
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Calculate lookback period for context
    max_timepoint = max(model_results.get('timepoints', [10]))
    lookback_days = datetime.timedelta(days=max_timepoint)
    context_start_date = start_date - lookback_days
    
    # Generate predictions for all stocks
    print(f"Generating predictions for {len(stocks)} stocks...")
    print(f"(Using data from {context_start_date} for context)")
    
    predictions_df = lgbm_modeling.generate_volatility_predictions(
        model_results=model_results,
        df=df,
        stocks=stocks,
        start_date=context_start_date,  # Start earlier to get context
        end_date=end_date,
        stride=stride
    )
    
    # If windows not specified, find all available windows from the prediction data
    if windows is None:
        all_windows = set()
        
        for metric in metrics:
            for col in predictions_df.columns:
                # Check for prediction columns for this metric
                if col.startswith(f"{metric}_") and col.endswith("_future_pred"):
                    try:
                        # Extract window from column name
                        window_str = col.replace(f"{metric}_", "").replace("_future_pred", "")
                        window = int(window_str)
                        all_windows.add(window)
                    except ValueError:
                        continue
        
        windows = sorted(list(all_windows))
        print(f"Found {len(windows)} available windows: {windows}")
    
    # Prepare summary data
    summary_data = []
    
    for stock in stocks:
        for metric in metrics:
            for window in windows:
                # Column names
                actual_col = f"{metric}_{window}_future"
                pred_col = f"{metric}_{window}_future_pred"
                
                # Skip if columns not in data
                if actual_col not in predictions_df.columns or pred_col not in predictions_df.columns:
                    print(f"Skipping {stock} {metric}_{window} - missing columns")
                    continue
                
                # Filter data for this stock - for visualization only use from start_date
                stock_data = predictions_df.filter(
                    (pl.col("act_symbol") == stock) &
                    (pl.col("date") >= start_date) &  # Use requested start_date
                    (pl.col("date") <= end_date)
                )
                
                if stock_data.height == 0:
                    print(f"Skipping {stock} - no data in date range")
                    continue
                
                # Extract actual and predicted values
                actual_values = stock_data[actual_col].to_numpy()
                pred_values = stock_data[pred_col].to_numpy()
                
                # Remove NaN and Inf values
                valid_mask = (~np.isnan(actual_values)) & (~np.isnan(pred_values)) & \
                             (~np.isinf(actual_values)) & (~np.isinf(pred_values))
                
                if np.sum(valid_mask) < 5:  # Require at least 5 valid points
                    print(f"Skipping {stock} {metric}_{window} - not enough valid data")
                    continue
                
                actual = actual_values[valid_mask]
                pred = pred_values[valid_mask]
                
                # Calculate metrics
                rmse = np.sqrt(np.mean((pred - actual) ** 2))
                mae = np.mean(np.abs(pred - actual))
                mape = np.mean(np.abs((actual - pred) / np.maximum(0.001, np.abs(actual)))) * 100
                corr = np.corrcoef(pred, actual)[0, 1]
                
                # Try to find alternative metrics for win rate calculation
                alternatives = {}
                
                # 1. Current volatility (same metric without _future)
                current_col = f"{metric}_{window}"
                if current_col in stock_data.columns:
                    alt_values = stock_data[current_col].to_numpy()[valid_mask]
                    alt_mean = np.mean(alt_values)
                    target_mean = np.mean(actual)
                    scale_factor = target_mean / alt_mean if alt_mean > 0 else 1.0
                    
                    alternatives[current_col] = {
                        "scaled": alt_values * scale_factor
                    }
                
                # 2. Find more alternatives such as ParkinsonVol or ATR
                for alt_prefix in ["ParkinsonVol_", "ATR_"]:
                    alt_col = f"{alt_prefix}{window}"
                    if alt_col in stock_data.columns:
                        alt_values = stock_data[alt_col].to_numpy()[valid_mask]
                        alt_mean = np.mean(alt_values)
                        target_mean = np.mean(actual)
                        scale_factor = target_mean / alt_mean if alt_mean > 0 else 1.0
                        
                        alternatives[alt_col] = {
                            "scaled": alt_values * scale_factor
                        }
                
                # 3. Check for CBOE volatility indices
                for cboe_index in ["vix", "vxapl", "vxazn", "vvix"]:
                    if cboe_index in stock_data.columns:
                        cboe_values = stock_data[cboe_index].to_numpy()[valid_mask]
                        # Convert from percentage to decimal
                        if np.median(cboe_values) > 1:
                            cboe_values = cboe_values / 100.0
                        
                        cboe_mean = np.mean(cboe_values)
                        target_mean = np.mean(actual)
                        scale_factor = target_mean / cboe_mean if cboe_mean > 0 else 1.0
                        
                        alternatives[cboe_index] = {
                            "scaled": cboe_values * scale_factor
                        }
                
                # Calculate win rate if we have alternatives
                win_rate = 0.5  # Default
                if alternatives:
                    win_rates = []
                    for alt_name, alt_data in alternatives.items():
                        alt_values = alt_data["scaled"]
                        
                        # Calculate errors
                        model_errors = np.abs(pred - actual)
                        alt_errors = np.abs(alt_values - actual)
                        
                        # Calculate win/loss
                        model_wins = np.sum(model_errors < alt_errors)
                        total = len(model_errors)
                        
                        win_rates.append(model_wins / total if total > 0 else 0.5)
                    
                    win_rate = np.mean(win_rates)
                
                # Store results
                summary_data.append({
                    "stock": stock,
                    "metric": metric,
                    "window": window,
                    "data_points": np.sum(valid_mask),
                    "rmse": rmse,
                    "mae": mae,
                    "mape": mape,
                    "correlation": corr,
                    "win_rate": win_rate * 100  # Convert to percentage
                })
    
    # Create DataFrame
    if not summary_data:
        print("No valid data for summary table")
        return pl.DataFrame(), pl.DataFrame()
    
    summary_df = pl.DataFrame(summary_data)
    
    # Calculate aggregated statistics
    agg_df = summary_df.group_by(["metric", "window"]).agg([
        pl.mean("rmse").alias("avg_rmse"),
        pl.mean("mae").alias("avg_mae"),
        pl.mean("mape").alias("avg_mape"),
        pl.mean("correlation").alias("avg_correlation"),
        pl.mean("win_rate").alias("avg_win_rate"),
        pl.count("stock").alias("stock_count")
    ])
    
    return summary_df, agg_df