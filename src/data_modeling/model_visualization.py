import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from typing import Dict, List
import time

def evaluate_volatility_predictors(
    ohlcv: pl.DataFrame,
    ticker: str,
    model_results: Dict,
    start_date: pl.Date = pl.date(2012, 1, 1),
    end_date: pl.Date = pl.date(2024, 9, 1),
    target_vol: str = "YZVol_30_future",
    model_name: str = "XGBoost"
) -> Dict:
    """
    Evaluate all available volatility measures against a target volatility metric,
    including predictions from a trained XGBoost model.
    
    Parameters:
        ohlcv: DataFrame containing stock and volatility data
        ticker: Stock ticker symbol
        model_results: Results dictionary from train_paired_volatility_models
        start_date: Start date for analysis
        end_date: End date for analysis
        target_vol: Target volatility column to use as ground truth
        model_name: Name to use for the model in output
        
    Returns:
        Dictionary with performance metrics and plotting data
    """
    print(f"Evaluating volatility predictors for {ticker}...")
    start_time = time.time()
    
    # 1. Get stock data with valid target values
    stock_data = ohlcv.filter(
        (pl.col("act_symbol") == ticker) &
        (~pl.col(target_vol).is_null()) &
        (~pl.col(target_vol).is_nan()) &
        (~pl.col(target_vol).is_infinite()) &
        (pl.col("date") >= start_date) &
        (pl.col("date") < end_date)
    ).sort("date")
    
    if stock_data.height == 0:
        raise ValueError(f"No valid data found for ticker {ticker} in the specified date range")
    
    print(f"Number of data points for {ticker}: {stock_data.height}")
    
    # 2. Determine which model to use based on which stock group contains ticker
    stocks_A = model_results.get('stocks_A', [])
    
    # For clarity on the paired model approach
    if ticker in stocks_A:
        print(f"{ticker} is in Stock Group A - using Model 2 for prediction")
        print("(Model 2 was trained on Group B stocks, so it hasn't seen this stock during training)")
        model_to_use = model_results.get('model_2')
    else:
        print(f"{ticker} is in Stock Group B - using Model 1 for prediction")
        print("(Model 1 was trained on Group A stocks, so it hasn't seen this stock during training)")
        model_to_use = model_results.get('model_1')
    
    # 3. Identify CBOE indices and other volatility columns
    # Find all CBOE index columns (lowercase names) with expanded pattern list
    cboe_pattern = ["vix", "vix9d", "vxapl", "vxazn", "vxeem", "gvz", "ovx", "vvix"]
    cboe_cols = [col for col in stock_data.columns if any(idx in col.lower() for idx in cboe_pattern)]
    
    # Identify Yang-Zhang and Parkinson volatility columns
    yz_cols = [col for col in stock_data.columns if "YZVol_" in col and "_future" not in col]
    parkinson_cols = [col for col in stock_data.columns if "ParkinsonVol_" in col]
    
    # Combine all volatility predictors
    volatility_predictors = cboe_cols + yz_cols + parkinson_cols
    
    print(f"\nFound volatility predictors:")
    print(f"CBOE indices: {cboe_cols}")
    print(f"Yang-Zhang volatility: {yz_cols}")
    print(f"Parkinson volatility: {parkinson_cols}")
    
    # 4. Check if we already have model predictions for this ticker in the results
    # If original ticker in results matches current ticker, we can use those predictions
    results_ticker = model_results.get('ticker', None)
    has_model_pred = 'model_pred' in model_results
    dates_match = False
    
    if results_ticker == ticker and has_model_pred:
        # Check if dates match too
        results_dates = model_results.get('dates', np.array([]))
        stock_dates = stock_data["date"].to_numpy()
        
        # Check if length and boundaries match
        if len(results_dates) == len(stock_dates) and results_dates[0] == stock_dates[0] and results_dates[-1] == stock_dates[-1]:
            dates_match = True
            
    if results_ticker == ticker and has_model_pred and dates_match:
        # Use pre-existing model predictions if they're for the same ticker and dates
        print(f"Using pre-existing model predictions for {ticker}")
        model_pred = model_results.get('model_pred')
    else:
        # Need to generate new predictions
        
        # Make sure we have at least one of the models
        model_1 = model_results.get('model_1')
        model_2 = model_results.get('model_2')
        
        if model_1 is None and model_2 is None:
            # If no models in results, we can't make predictions
            print("No models found in model_results. Using a simple baseline instead.")
            # Use YZ volatility as baseline model
            if len(yz_cols) > 0:
                # Use current YZ volatility as a simple prediction
                baseline_col = stock_data[yz_cols[0]].to_numpy()
                print(f"Using {yz_cols[0]} as a baseline prediction")
                model_pred = baseline_col
            else:
                # Last resort: use the target's mean value as prediction
                print("No YZ volatility found. Using target mean as prediction.")
                stock_actual = stock_data[target_vol].to_numpy()
                target_mean = np.mean(stock_actual)
                model_pred = np.full_like(stock_actual, target_mean)
        else:
            # Double-check that the selected model exists
            if model_to_use is None:
                if model_1 is not None:
                    print(f"Selected model is None. Falling back to Model 1.")
                    model_to_use = model_1
                else:
                    print(f"Selected model is None. Falling back to Model 2.")
                    model_to_use = model_2
            
            # Get feature columns for prediction
            feature_cols = model_results.get('feature_cols', None)
            
            if not feature_cols:
                # If feature_cols not found, we'll need to infer them from data
                # Exclude date, symbol, target columns and other non-feature columns
                print("No explicit feature columns found, inferring from data...")
                exclude_patterns = ["date", "act_symbol", "_future"]
                feature_cols = [
                    col for col in stock_data.columns 
                    if not any(pattern in col.lower() for pattern in exclude_patterns)
                ]
                
            print(f"Using {len(feature_cols)} features for prediction")
            
            # Prepare features for model prediction
            # Make sure all feature columns exist in the data
            available_feature_cols = [col for col in feature_cols if col in stock_data.columns]
            
            if len(available_feature_cols) < len(feature_cols):
                print(f"Warning: {len(feature_cols) - len(available_feature_cols)} feature columns not found in data")
                
            if len(available_feature_cols) == 0:
                print("No feature columns available. Using a simple baseline instead.")
                # Use simple baseline if no features available
                if len(yz_cols) > 0:
                    baseline_col = stock_data[yz_cols[0]].to_numpy() 
                    model_pred = baseline_col
                else:
                    stock_actual = stock_data[target_vol].to_numpy()
                    target_mean = np.mean(stock_actual)
                    model_pred = np.full_like(stock_actual, target_mean)
            else:
                # Process features and make predictions
                features_for_pred = stock_data.select(available_feature_cols)
                
                # Handle each feature one by one to avoid issues with infinity/null values
                for col in available_feature_cols:
                    try:
                        features_for_pred = features_for_pred.with_columns(
                            pl.when(pl.col(col).is_infinite())
                            .then(None)
                            .otherwise(pl.col(col))
                            .alias(col)
                        )
                    except Exception as e:
                        # Skip this column if there's an error
                        print(f"Skipping column {col}: {str(e)}")
                        features_for_pred = features_for_pred.drop(col)
                
                # Convert to numpy and make predictions
                X_features = features_for_pred.fill_null(0).to_numpy()
                try:
                    model_pred = model_to_use.predict(X_features)
                except Exception as e:
                    print(f"Error making predictions: {str(e)}. Using a simple baseline instead.")
                    # Use simple baseline if prediction fails
                    if len(yz_cols) > 0:
                        baseline_col = stock_data[yz_cols[0]].to_numpy() 
                        model_pred = baseline_col
                    else:
                        stock_actual = stock_data[target_vol].to_numpy()
                        target_mean = np.mean(stock_actual)
                        model_pred = np.full_like(stock_actual, target_mean)
    
    # 5. Extract target volatility (ground truth)
    dates = stock_data["date"].to_numpy()
    stock_actual = stock_data[target_vol].to_numpy()
    
    # Ensure model predictions match the length of actual values
    if len(model_pred) != len(stock_actual):
        print(f"Warning: Model predictions length ({len(model_pred)}) doesn't match actual values length ({len(stock_actual)})")
        print("Fixing the length mismatch...")
        
        # Trim to the shorter length
        min_length = min(len(model_pred), len(stock_actual))
        model_pred = model_pred[:min_length]
        stock_actual = stock_actual[:min_length]
        dates = dates[:min_length]
    
    # 6. Calculate performance metrics for each predictor
    performance = {}
    predictor_data = {}
    
    # Add model predictions
    predictor_data[model_name] = model_pred
    
    # Pre-calculate target stats for scaling
    target_mean = np.mean(stock_actual)
    target_std = np.std(stock_actual)
    print(f"Target volatility mean: {target_mean:.6f}, std: {target_std:.6f}")
    
    # Calculate model error for daily win/loss comparison
    model_daily_errors = np.abs(model_pred - stock_actual)
    
    # Process each volatility predictor
    for predictor in volatility_predictors:
        if predictor not in stock_data.columns:
            continue
            
        predictor_col = stock_data[predictor].to_numpy()[:len(stock_actual)]  # Ensure matching length
        valid_mask = ~np.isnan(predictor_col) & ~np.isinf(predictor_col)
        valid_count = sum(valid_mask)
        
        if valid_count < 10:  # Skip if very few valid points
            print(f"Skipping {predictor}: only {valid_count} valid values")
            continue
            
        # If it's a CBOE index (typically in percentage), convert to decimal
        if predictor in cboe_cols:
            predictor_decimal = np.zeros_like(predictor_col)
            predictor_decimal[valid_mask] = predictor_col[valid_mask] / 100
            
            # Calculate scaling factor to match target volatility magnitude
            pred_mean = np.mean(predictor_decimal[valid_mask])
            scale_factor = target_mean / pred_mean if pred_mean > 0 else 1.0
            
            # Apply scaling
            predictor_scaled = np.zeros_like(predictor_decimal)
            predictor_scaled[valid_mask] = predictor_decimal[valid_mask] * scale_factor
                        
            # Store scaled values for later use
            predictor_data[predictor] = {
                'original': predictor_col,
                'decimal': predictor_decimal,
                'scaled': predictor_scaled,
                'valid_mask': valid_mask,
                'scale_factor': scale_factor
            }
            
            # Calculate metrics using scaled values
            rmse = np.sqrt(np.mean((stock_actual[valid_mask] - predictor_scaled[valid_mask]) ** 2))
            mae = np.mean(np.abs(stock_actual[valid_mask] - predictor_scaled[valid_mask]))
            # Use original decimal values for correlation (scaling doesn't affect correlation)
            corr = np.corrcoef(stock_actual[valid_mask], predictor_decimal[valid_mask])[0, 1]
            
            # Calculate daily error for win/loss calculation
            predictor_daily_errors = np.zeros_like(predictor_col)
            predictor_daily_errors[valid_mask] = np.abs(stock_actual[valid_mask] - predictor_scaled[valid_mask])
            
            # Calculate win/loss against model
            pred_wins = np.sum((predictor_daily_errors < model_daily_errors) & valid_mask)
            
        else:
            # For non-CBOE predictors, use as-is (already in decimal form)
            pred_mean = np.mean(predictor_col[valid_mask])
            
            # For consistency, still calculate a scale factor even for decimal values
            scale_factor = 1.0
            if np.abs(pred_mean) > 0.0001:  # Avoid division by very small numbers
                scale_factor = target_mean / pred_mean
                
            # Store scaled values
            predictor_scaled = np.zeros_like(predictor_col)
            predictor_scaled[valid_mask] = predictor_col[valid_mask] * scale_factor
                    
            predictor_data[predictor] = {
                'original': predictor_col,
                'scaled': predictor_scaled,
                'valid_mask': valid_mask,
                'scale_factor': scale_factor
            }
            
            # Calculate metrics
            rmse = np.sqrt(np.mean((stock_actual[valid_mask] - predictor_scaled[valid_mask]) ** 2))
            mae = np.mean(np.abs(stock_actual[valid_mask] - predictor_scaled[valid_mask]))
            corr = np.corrcoef(stock_actual[valid_mask], predictor_col[valid_mask])[0, 1]
            
            # Calculate daily error for win/loss calculation
            predictor_daily_errors = np.zeros_like(predictor_col)
            predictor_daily_errors[valid_mask] = np.abs(stock_actual[valid_mask] - predictor_scaled[valid_mask])
            
            # Calculate win/loss against model
            pred_wins = np.sum((predictor_daily_errors < model_daily_errors) & valid_mask)
        
        # Store performance metrics, now including daily win/loss stats
        win_rate = (pred_wins / valid_count) * 100 if valid_count > 0 else 0
        performance[predictor] = {
            'rmse': rmse,
            'mae': mae,
            'correlation': corr,
            'valid_days': valid_count,
            'scale_factor': scale_factor,
            'wins': pred_wins,
            'win_rate': win_rate
        }
    
    # Calculate model performance
    model_rmse = np.sqrt(np.mean((stock_actual - model_pred) ** 2))
    model_mae = np.mean(np.abs(stock_actual - model_pred))
    model_corr = np.corrcoef(stock_actual, model_pred)[0, 1]
    
    # For the model, calculate wins against each predictor and use the average
    model_win_counts = []
    for predictor in volatility_predictors:
        if predictor in predictor_data:
            data = predictor_data[predictor]
            if isinstance(data, dict) and 'valid_mask' in data:
                valid_mask = data['valid_mask']
                
                if 'scaled' in data:
                    predictor_scaled = data['scaled']
                    predictor_daily_errors = np.zeros_like(predictor_scaled)
                    predictor_daily_errors[valid_mask] = np.abs(stock_actual[valid_mask] - predictor_scaled[valid_mask])
                    
                    model_wins_vs_pred = np.sum((model_daily_errors < predictor_daily_errors) & valid_mask)
                    model_win_counts.append((model_wins_vs_pred, np.sum(valid_mask)))
    
    # Calculate average model win rate
    if model_win_counts:
        # Calculate average win rate instead of summing all wins
        win_rates = [wins/days*100 for wins, days in model_win_counts]
        model_win_rate = sum(win_rates) / len(win_rates)
        
        # For display purposes, use average win percentage but show total days
        total_valid_days = len(model_pred)
        total_model_wins = int(round(model_win_rate * total_valid_days / 100))
    else:
        # No comparisons available, model vs itself
        model_win_rate = 50.0  # Neutral for model against itself
        total_model_wins = len(model_pred) // 2
    
    performance[model_name] = {
        'rmse': model_rmse,
        'mae': model_mae,
        'correlation': model_corr,
        'valid_days': len(model_pred),
        'scale_factor': 1.0,  # Model predictions already in target scale
        'wins': total_model_wins,
        'win_rate': model_win_rate
    }
    
    # 7. Rank predictors by win rate instead of correlation
    sorted_predictors = sorted(
        [(p, performance[p]['win_rate']) for p in performance], 
        key=lambda x: x[1], 
        reverse=True
    )
    
    # 8. Display performance table
    print("\nPerformance Comparison:")
    print("=" * 90)
    headers = ["Predictor", "RMSE", "MAE", "Correlation", "Scale Factor", "Win/Loss (vs Model)", "Valid Days"]
    print(f"{headers[0]:<20} {headers[1]:<10} {headers[2]:<10} {headers[3]:<12} {headers[4]:<12} {headers[5]:<20} {headers[6]:<10}")
    print("-" * 90)
    
    for predictor, _ in sorted_predictors:
        metrics = performance[predictor]
        # Format win/loss as "wins/total (percentage%)"
        win_loss_str = f"{metrics['wins']}/{metrics['valid_days']} ({metrics['win_rate']:.1f}%)"
        
        print(f"{predictor:<20} {metrics['rmse']:.6f}{'':<4} {metrics['mae']:.6f}{'':<4} "
              f"{metrics['correlation']:.6f}{'':<6} {metrics['scale_factor']:.6f}{'':<6} "
              f"{win_loss_str:<20} {metrics['valid_days']}")
    
    # 9. Select top predictors for visualization
    # Always include VIX if available, then add top 2 other predictors by correlation
    top_predictors = []
    
    # First check if VIX is in the predictors
    if 'vix' in performance:
        top_predictors.append('vix')
    
    # Add top 2 non-VIX predictors by correlation
    count = 0
    for predictor, _ in sorted_predictors:
        if predictor != 'vix' and predictor != model_name:
            top_predictors.append(predictor)
            count += 1
            if count >= 2:
                break
    
    print(f"\nTop predictors selected for visualization: {top_predictors}")
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return {
        'ticker': ticker,
        'dates': dates,
        'actual': stock_actual,
        'model_pred': model_pred,
        'close_price': stock_data["close"].to_numpy()[:len(stock_actual)],  # Ensure matching length
        'performance': performance,
        'predictor_data': predictor_data,
        'top_predictors': top_predictors,
        'model_name': model_name,
        'target_vol': target_vol
    }

def plot_volatility_comparison(results: Dict) -> None:
    """
    Plot the comparison between target volatility, model predictions, and top CBOE indices.
    
    Parameters:
        results: Results dictionary from evaluate_volatility_predictors
    """
    # Extract data from results
    ticker = results['ticker']
    dates = results['dates']
    stock_actual = results['actual']
    model_pred = results['model_pred']
    close_price = results['close_price']
    predictor_data = results['predictor_data']
    top_predictors = results['top_predictors']
    model_name = results['model_name']
    target_vol = results['target_vol']
    
    # Create figure with two panels
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top Panel - Plot actual volatility
    lns1 = ax1.plot(dates, stock_actual, 'b-', label=f'Actual {target_vol}', linewidth=2)
    
    # Add model prediction
    plot_lines = [lns1]
    model_error = np.abs(model_pred - stock_actual)
    lns2 = ax1.plot(dates, model_pred, 'r--', label=f'{model_name} Prediction', linewidth=2)
    plot_lines.append(lns2)
    
    # Add top predictors
    color_cycle = ['m', 'c', 'y', 'k']
    predictor_errors = {}
    
    for i, predictor in enumerate(top_predictors):
        if predictor in predictor_data:
            color = color_cycle[i % len(color_cycle)]
            data = predictor_data[predictor]
            
            valid_mask = data['valid_mask']
            
            # Always use scaled values for plotting
            if 'scaled' in data:
                scaled_values = data['scaled']
                
                lns = ax1.plot(
                    dates[valid_mask], 
                    scaled_values[valid_mask], 
                    f'{color}-', 
                    label=f'{predictor.upper()} (scaled)', 
                    linewidth=1.5
                )
                plot_lines.append(lns)
                
                # Calculate error for plotting
                predictor_error = np.zeros_like(scaled_values)
                predictor_error[valid_mask] = np.abs(scaled_values[valid_mask] - stock_actual[valid_mask])
                predictor_errors[predictor] = {
                    'error': predictor_error,
                    'valid_mask': valid_mask
                }
    
    # Set up primary y-axis
    ax1.set_ylabel('Volatility (scaled to target level)', color='b')
    ax1.tick_params(axis='y', labelcolor='b')
    
    # Add price on secondary y-axis
    ax1b = ax1.twinx()
    lns_price = ax1b.plot(dates, close_price, 'g-', label='Close Price', linewidth=1)
    ax1b.set_ylabel('Price ($)', color='g')
    ax1b.tick_params(axis='y', labelcolor='g')
    
    # Combine legends from all lines
    all_lines = []
    for line_group in plot_lines:
        all_lines.extend(line_group)
    all_lines.extend(lns_price)
    
    all_labels = [line.get_label() for line in all_lines]
    ax1.legend(all_lines, all_labels, loc='upper left')
    
    ax1.set_title(f'{ticker} Volatility Prediction Comparison: {target_vol} vs Predictors')
    ax1.grid(True, alpha=0.3)
    
    # Bottom Panel - Plot errors
    color_cycle = ['r', 'm', 'c', 'y', 'k']
    i = 0
    
    # Plot model error
    mean_err = np.mean(model_error)
    ax2.plot(
        dates, model_error, f'{color_cycle[i]}-',
        label=f'{model_name} Error (Avg: {mean_err:.6f})', 
        linewidth=1.5
    )
    i += 1
    
    # Plot predictor errors
    for predictor in top_predictors:
        if predictor in predictor_errors:
            color = color_cycle[i % len(color_cycle)]
            i += 1
            
            error_data = predictor_errors[predictor]
            error = error_data['error']
            valid_mask = error_data['valid_mask']
            
            mean_err = np.mean(error[valid_mask])
            ax2.plot(
                dates[valid_mask], error[valid_mask], f'{color}-',
                label=f'{predictor.upper()} Error (Avg: {mean_err:.6f})', 
                linewidth=1.5
            )
    
    ax2.set_ylabel('Absolute Error')
    ax2.set_xlabel('Date')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Function to analyze all stock tickers in data
def analyze_multiple_stocks(
    ohlcv: pl.DataFrame,
    model_results: Dict,
    tickers: List[str] = None,
    n_samples: int = 5,
    start_date: pl.Date = pl.date(2012, 1, 1),
    end_date: pl.Date = pl.date(2024, 9, 1),
    target_vol: str = "YZVol_30_future",
    model_name: str = "XGBoost"
) -> List[Dict]:
    """
    Run volatility analysis on multiple stocks and summarize results.
    
    Parameters:
        ohlcv: DataFrame containing stock and volatility data
        model_results: Results dictionary from train_paired_volatility_models
        tickers: List of tickers to analyze (if None, randomly samples from data)
        n_samples: Number of tickers to randomly sample if tickers not provided
        start_date: Start date for analysis
        end_date: End date for analysis
        target_vol: Target volatility column to use as ground truth
        model_name: Name to use for the model in output
        
    Returns:
        List of summary dictionaries for each analyzed stock
    """
    if tickers is None:
        # Get all unique tickers with sufficient data
        all_tickers = ohlcv.filter(
            (~pl.col(target_vol).is_null()) &
            (pl.col("date") >= start_date) &
            (pl.col("date") < end_date)
        )["act_symbol"].unique().to_list()
        
        # Randomly sample n_samples tickers
        np.random.seed(42)
        tickers = np.random.choice(all_tickers, min(n_samples, len(all_tickers)), replace=False)
    
    print(f"Analyzing {len(tickers)} stocks: {tickers}")
    
    # Get the two stock groups from the model results
    stocks_A = model_results.get('stocks_A', [])
    stocks_B = [ticker for ticker in tickers if ticker not in stocks_A]
    
    print(f"Model Training Groups:")
    print(f"  - Group A stocks: Used to train Model 1 ({len(stocks_A)} stocks)")
    print(f"  - Group B stocks: Used to train Model 2 ({len(stocks_B)} stocks)")
    print(f"For out-of-sample predictions:")
    print(f"  - Model 1 will predict Group B stocks")
    print(f"  - Model 2 will predict Group A stocks")
    
    # Collect performance metrics for all stocks
    summary_data = []
    
    for ticker in tickers:
        try:
            print(f"\n{'='*50}\nAnalyzing {ticker}\n{'='*50}")
            results = evaluate_volatility_predictors(
                ohlcv, ticker, model_results, 
                start_date, end_date, target_vol, model_name
            )
            
            # Plot results
            plot_volatility_comparison(results)
            
            # Collect performance metrics
            performance = results['performance']
            model_metrics = performance[model_name]
            
            # Find best non-model predictor by win rate
            best_predictor = None
            best_win_rate = -999
            
            for pred, metrics in performance.items():
                if pred != model_name and metrics['win_rate'] > best_win_rate:
                    best_predictor = pred
                    best_win_rate = metrics['win_rate']
            
            if best_predictor:
                best_metrics = performance[best_predictor]
                
                # Add to summary
                summary_data.append({
                    'ticker': ticker,
                    'group': 'A' if ticker in stocks_A else 'B',
                    'model_used': 'Model 2' if ticker in stocks_A else 'Model 1',
                    'model_rmse': model_metrics['rmse'],
                    'model_mae': model_metrics['mae'],
                    'model_corr': model_metrics['correlation'],
                    'model_win_rate': model_metrics['win_rate'],
                    'best_predictor': best_predictor,
                    'best_pred_rmse': best_metrics['rmse'],
                    'best_pred_mae': best_metrics['mae'],
                    'best_pred_corr': best_metrics['correlation'],
                    'best_pred_win_rate': best_metrics['win_rate'],
                    'model_wins': model_metrics['rmse'] < best_metrics['rmse']
                })
        
        except Exception as e:
            print(f"Error analyzing {ticker}: {str(e)}")
    
    # Print summary table
    if summary_data:
        print("\n\nSummary of Results Across Stocks:")
        print("=" * 120)
        print(f"{'Ticker':<8} {'Group':<6} {'Model Used':<10} {'Model RMSE':<12} {'Model MAE':<12} {'Model Corr':<12} {'Model Win%':<10} " +
              f"{'Best Pred':<12} {'Pred RMSE':<12} {'Pred MAE':<12} {'Pred Corr':<12} {'Pred Win%':<10} {'Winner':<8}")
        print("-" * 120)
        
        model_wins = 0
        for row in summary_data:
            winner = "Model" if row['model_wins'] else "Pred"
            if row['model_wins']:
                model_wins += 1
                
            print(f"{row['ticker']:<8} {row['group']:<6} {row['model_used']:<10} {row['model_rmse']:<12.6f} {row['model_mae']:<12.6f} " +
                  f"{row['model_corr']:<12.6f} {row['model_win_rate']:<10.1f} {row['best_predictor'][:10]:<12} " +
                  f"{row['best_pred_rmse']:<12.6f} {row['best_pred_mae']:<12.6f} " +
                  f"{row['best_pred_corr']:<12.6f} {row['best_pred_win_rate']:<10.1f} {winner:<8}")
        
        print(f"\nModel wins in {model_wins}/{len(summary_data)} cases ({model_wins/len(summary_data)*100:.1f}%)")
        
        # Additional group-based analysis
        group_A_results = [row for row in summary_data if row['group'] == 'A']
        group_B_results = [row for row in summary_data if row['group'] == 'B']
        
        if group_A_results:
            group_A_wins = sum(1 for row in group_A_results if row['model_wins'])
            print(f"Group A stocks (using Model 2): {group_A_wins}/{len(group_A_results)} wins ({group_A_wins/len(group_A_results)*100:.1f}%)")
            
        if group_B_results:
            group_B_wins = sum(1 for row in group_B_results if row['model_wins'])
            print(f"Group B stocks (using Model 1): {group_B_wins}/{len(group_B_results)} wins ({group_B_wins/len(group_B_results)*100:.1f}%)")

    return summary_data