import polars as pl
import numpy as np
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.multitest import multipletests
import warnings
from collections import Counter
warnings.filterwarnings('ignore')

def evaluate_indicators_multi_ticker(df, tickers=None, n_tickers=20, min_years=3, alpha=0.05, window_size=252):
    """
    Evaluate which financial indicators need time-aware normalization across multiple tickers.
    
    Parameters:
        df (pl.DataFrame): DataFrame containing multiple stocks' time series data
        tickers (list): Specific tickers to analyze (if None, random sample will be used)
        n_tickers (int): Number of tickers to sample if tickers not provided
        min_years (int): Minimum years of data required for a ticker to be included
        alpha (float): Significance level for statistical tests
        window_size (int): Window size for rolling statistics (default: 252 trading days = 1 year)
        
    Returns:
        dict: Dictionary with categorized indicators
    """
    # Get all available tickers
    all_tickers = df["act_symbol"].unique().to_list()
    print(f"Total unique tickers in dataset: {len(all_tickers)}")
    
    # If tickers not provided, sample random ones
    if tickers is None:
        # Filter tickers by data availability
        valid_tickers = []
        min_data_points = min_years * 252  # Approx trading days per year
        
        for ticker in all_tickers:
            ticker_data = df.filter(pl.col("act_symbol") == ticker)
            if len(ticker_data) >= min_data_points:
                valid_tickers.append(ticker)
        
        print(f"Tickers with at least {min_years} years of data: {len(valid_tickers)}")
        
        # Sample random tickers
        if len(valid_tickers) > n_tickers:
            np.random.seed(42)  # For reproducibility
            tickers = np.random.choice(valid_tickers, n_tickers, replace=False).tolist()
        else:
            tickers = valid_tickers
    
    print(f"Analyzing {len(tickers)} tickers: {', '.join(tickers)}")
    print("=" * 80)
    
    # Store results for each ticker
    ticker_results = {}
    
    # Analyze each ticker
    for ticker in tickers:
        print(f"\nAnalyzing {ticker}...")
        ticker_data = df.filter(pl.col("act_symbol") == ticker)
        
        if len(ticker_data) < window_size:
            print(f"Skipping {ticker}: Insufficient data ({len(ticker_data)} data points)")
            continue
        
        # Run indicator evaluation
        ticker_results[ticker] = evaluate_single_ticker(ticker_data, alpha, window_size, print_details=False)
    
    # Aggregate results across tickers
    return aggregate_results(ticker_results)


def evaluate_single_ticker(df, alpha=0.05, window_size=252, print_details=True):
    """
    Evaluate indicators for a single ticker (similar to original evaluate_indicators)
    
    Parameters:
        df: DataFrame for a single ticker
        alpha: Significance level
        window_size: Window size for rolling statistics
        print_details: Whether to print individual indicator results
        
    Returns:
        dict: Categorization of indicators
    """
    ticker = df['act_symbol'][0]
    data_points = len(df)
    
    if print_details:
        print(f"Evaluating indicators for {ticker} with {data_points} data points")
        print("=" * 80)
    
    # Exclude non-numeric and identifier columns
    exclude_cols = ["act_symbol", "date"]
    
    # Get all numeric columns from the DataFrame
    numeric_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Financial domain knowledge pre-classification
    domain_categories = {
        "definite_time_aware": [],  # Definitely need time-aware normalization
        "likely_time_aware": [],    # Likely need time-aware normalization
        "likely_regular": [],       # Likely fine with regular normalization
        "uncertain": []             # Need further testing
    }
    
    # Pre-classify based on financial domain knowledge
    for col in numeric_cols:
        col_lower = col.lower()
        
        # Target variables and volatility measures 
        if ("future" in col_lower or 
            any(x in col_lower for x in ["vol_", "vix", "gvz", "ovx"]) or
            col in ["open", "high", "low", "close"]):
            domain_categories["definite_time_aware"].append(col)
            
        # Trend-following and accumulative metrics
        elif (any(x in col_lower for x in ["sma_", "ema_", "obv"]) or
              "donchian" in col_lower or "vwap" in col_lower):
            domain_categories["likely_time_aware"].append(col)
            
        # Oscillators and ratio metrics
        elif (any(x in col_lower for x in ["rsi_", "cci_", "cmf_", "ratio"]) or
              "skew" in col_lower or "kurt" in col_lower):
            domain_categories["likely_regular"].append(col)
            
        # Other metrics, require testing
        else:
            domain_categories["uncertain"].append(col)
    
    # Results storage
    time_aware = []
    time_aware_optional = []
    regular = []
    cautionary = []
    
    # Print header for individual indicator results if requested
    if print_details:
        print("\nINDIVIDUAL INDICATOR RESULTS")
        print("=" * 80)
        print(f"{'INDICATOR':<25} {'ADF p-val':<10} {'KPSS p-val':<10} {'NORM DIFF':<10} {'CORR':<10} {'CATEGORY':<20}")
        print("-" * 80)
    
    # Analyze each numeric column
    indicator_results = {}
    
    for col in numeric_cols:
        # Extract values
        values = df[col].to_numpy()
        
        # Skip if too many NaN values
        if np.isnan(values).sum() > len(values) * 0.1:
            if print_details:
                print(f"{col:<25} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'N/A':<10} {'Skipped - Too many NaNs':<20}")
            indicator_results[col] = {"category": "skipped", "reason": "too_many_nans"}
            continue
        
        # Calculate rolling statistics
        rolling_mean = np.array([np.mean(values[max(0, i-window_size):i+1]) for i in range(len(values))])
        rolling_std = np.array([np.std(values[max(0, i-window_size):i+1]) for i in range(len(values))])
        
        # Calculate stability metrics
        cv_mean = np.std(rolling_mean) / np.mean(np.abs(rolling_mean) + 1e-10)
        cv_std = np.std(rolling_std) / np.mean(rolling_std + 1e-10)
        
        # Run stationarity tests
        try:
            # Augmented Dickey-Fuller test (null hypothesis: series is non-stationary)
            adf_result = adfuller(values, regression='ct')
            adf_pvalue = adf_result[1]
            
            # KPSS test (null hypothesis: series is stationary)
            kpss_result = kpss(values, regression='ct')
            kpss_pvalue = kpss_result[1]
        except Exception as e:
            adf_pvalue = np.nan
            kpss_pvalue = np.nan
        
        # Practical impact: Compare regular vs. time-aware normalization
        # Regular normalization (z-score)
        std_val = np.std(values)
        if std_val > 0:
            regular_norm = (values - np.mean(values)) / std_val
        else:
            regular_norm = np.zeros_like(values)
        
        # Time-aware normalization (expanding window)
        time_aware_norm = []
        for i in range(len(values)):
            window = values[:i+1]
            if i > 0 and np.std(window) > 0:
                time_aware_norm.append((values[i] - np.mean(window)) / np.std(window))
            else:
                time_aware_norm.append(0)  # First value or zero std
        
        time_aware_norm = np.array(time_aware_norm)
        
        # Calculate difference metrics
        mean_abs_diff = np.mean(np.abs(regular_norm - time_aware_norm))
        
        # Calculate correlation between regular and time-aware normalization
        valid_mask = ~np.isnan(regular_norm) & ~np.isnan(time_aware_norm)
        if np.sum(valid_mask) > 10 and np.std(regular_norm[valid_mask]) > 0 and np.std(time_aware_norm[valid_mask]) > 0:
            corr = np.corrcoef(regular_norm[valid_mask], time_aware_norm[valid_mask])[0, 1]
        else:
            corr = np.nan
        
        # Determine necessity of time-aware normalization
        statistical_evidence = "inconclusive"
        if not np.isnan(adf_pvalue) and not np.isnan(kpss_pvalue):
            if adf_pvalue > alpha and kpss_pvalue < alpha:
                statistical_evidence = "strong_nonstationary"
            elif adf_pvalue < alpha and kpss_pvalue > alpha:
                statistical_evidence = "strong_stationary"
        
        practical_impact = "inconclusive"
        if not np.isnan(mean_abs_diff) and not np.isnan(corr):
            if mean_abs_diff > 0.5 or corr < 0.9:
                practical_impact = "significant"
            elif mean_abs_diff > 0.2 or corr < 0.95:
                practical_impact = "moderate"
            else:
                practical_impact = "minimal"
        
        # Make final recommendation
        if col in domain_categories["definite_time_aware"]:
            recommendation = "time_aware"  # Always time-aware based on domain knowledge
            category = "TIME-AWARE REQUIRED"
            time_aware.append(col)
        elif statistical_evidence == "strong_nonstationary" or practical_impact == "significant":
            recommendation = "time_aware"  # Time-aware based on statistical evidence or practical impact
            category = "TIME-AWARE REQUIRED"
            time_aware.append(col)
        elif statistical_evidence == "strong_stationary" and practical_impact == "minimal":
            recommendation = "regular"     # Regular normalization is sufficient
            category = "REGULAR SUFFICIENT"
            regular.append(col)
        elif practical_impact == "moderate":
            recommendation = "time_aware_optional"  # Time-aware recommended but not critical
            category = "TIME-AWARE RECOMMENDED"
            time_aware_optional.append(col)
        else:
            recommendation = "cautionary_time_aware"  # Use time-aware to be safe
            category = "CAUTIONARY"
            cautionary.append(col)
        
        # Store detailed results
        indicator_results[col] = {
            "adf_pvalue": adf_pvalue,
            "kpss_pvalue": kpss_pvalue,
            "mean_abs_diff": mean_abs_diff,
            "correlation": corr,
            "statistical_evidence": statistical_evidence,
            "practical_impact": practical_impact,
            "recommendation": recommendation,
            "category": category
        }
        
        # Format values for printing if requested
        if print_details:
            adf_str = f"{adf_pvalue:.4f}" if not np.isnan(adf_pvalue) else "N/A"
            kpss_str = f"{kpss_pvalue:.4f}" if not np.isnan(kpss_pvalue) else "N/A"
            diff_str = f"{mean_abs_diff:.4f}" if not np.isnan(mean_abs_diff) else "N/A"
            corr_str = f"{corr:.4f}" if not np.isnan(corr) else "N/A"
            
            # Print results for this indicator
            print(f"{col:<25} {adf_str:<10} {kpss_str:<10} {diff_str:<10} {corr_str:<10} {category:<20}")
    
    # Return categorization for this ticker
    return {
        'time_aware_required': time_aware,
        'time_aware_recommended': time_aware_optional,
        'regular_sufficient': regular,
        'cautionary': cautionary,
        'details': indicator_results
    }


def aggregate_results(ticker_results):
    """
    Aggregate results across multiple tickers to provide final recommendations.
    
    Parameters:
        ticker_results: Dictionary of results per ticker
        
    Returns:
        dict: Aggregated indicator categorization
    """
    # Get all unique indicators across tickers
    all_indicators = set()
    for ticker, result in ticker_results.items():
        for category in ['time_aware_required', 'time_aware_recommended', 'regular_sufficient', 'cautionary']:
            all_indicators.update(result[category])
    
    # Count category occurrences for each indicator
    indicator_votes = {}
    for indicator in all_indicators:
        votes = {'time_aware_required': 0, 'time_aware_recommended': 0, 'regular_sufficient': 0, 'cautionary': 0}
        n_tickers = 0
        
        for ticker, result in ticker_results.items():
            for category in votes.keys():
                if indicator in result[category]:
                    votes[category] += 1
                    n_tickers += 1
        
        # Store votes if indicator was found in at least one ticker
        if n_tickers > 0:
            indicator_votes[indicator] = votes
    
    # Make final decisions based on majority voting with conservative rules
    final_categories = {
        'time_aware_required': [],
        'time_aware_recommended': [],
        'regular_sufficient': [],
        'cautionary': []
    }
    
    # Detailed voting results for display
    detailed_votes = {}
    
    for indicator, votes in indicator_votes.items():
        # Special rule for target variables and price data - always time-aware
        indicator_lower = indicator.lower()
        if ("future" in indicator_lower or 
            any(x in indicator_lower for x in ["yzv", "vix", "gvz", "ovx"]) or
            indicator in ["open", "high", "low", "close"]):
            final_category = 'time_aware_required'
        else:
            # Conservative approach for other indicators
            vote_count = Counter(votes)
            most_votes = max(votes.values())
            most_common = [cat for cat, count in votes.items() if count == most_votes]
            
            # If tie between time-aware and regular, go with time-aware
            if 'time_aware_required' in most_common:
                final_category = 'time_aware_required'
            elif 'time_aware_recommended' in most_common:
                final_category = 'time_aware_recommended'
            elif 'cautionary' in most_common:
                final_category = 'cautionary'
            else:
                final_category = 'regular_sufficient'
        
        final_categories[final_category].append(indicator)
        
        # Store voting details
        n_tickers = sum(votes.values())
        detailed_votes[indicator] = {
            'required_pct': votes['time_aware_required'] / n_tickers * 100,
            'recommended_pct': votes['time_aware_recommended'] / n_tickers * 100,
            'regular_pct': votes['regular_sufficient'] / n_tickers * 100,
            'cautionary_pct': votes['cautionary'] / n_tickers * 100,
            'final_category': final_category
        }
    
    # Print aggregated results
    print("\nAGGREGATED RESULTS ACROSS ALL TICKERS")
    print("=" * 80)
    print(f"{'INDICATOR':<25} {'TIME-AWARE REQ%':<15} {'TIME-AWARE REC%':<15} {'REGULAR%':<10} {'CAUTIONARY%':<15} {'FINAL CATEGORY':<20}")
    print("-" * 80)
    
    for indicator, details in sorted(detailed_votes.items()):
        print(f"{indicator:<25} {details['required_pct']:>13.1f}% {details['recommended_pct']:>13.1f}% {details['regular_pct']:>8.1f}% {details['cautionary_pct']:>13.1f}% {details['final_category'].replace('_', ' ').upper():<20}")
    
    # Print summary
    print("\nSUMMARY OF AGGREGATED RESULTS")
    print("=" * 80)
    print(f"TIME-AWARE NORMALIZATION REQUIRED ({len(final_categories['time_aware_required'])} indicators):")
    for indicator in sorted(final_categories['time_aware_required']):
        print(f"  - {indicator}")
    
    print(f"\nTIME-AWARE NORMALIZATION RECOMMENDED ({len(final_categories['time_aware_recommended'])} indicators):")
    for indicator in sorted(final_categories['time_aware_recommended']):
        print(f"  - {indicator}")
    
    print(f"\nREGULAR NORMALIZATION SUFFICIENT ({len(final_categories['regular_sufficient'])} indicators):")
    for indicator in sorted(final_categories['regular_sufficient']):
        print(f"  - {indicator}")
    
    print(f"\nCAUTIONARY TIME-AWARE ({len(final_categories['cautionary'])} indicators):")
    for indicator in sorted(final_categories['cautionary']):
        print(f"  - {indicator}")
    
    print("\nRECOMMENDATION STRATEGY:")
    print("=" * 80)
    print("1. Apply regular (non-time-aware) normalization to ALL indicators first.")
    print("2. Then apply time-aware normalization to the 'required' and 'recommended' categories.")
    print("3. For your target variables (YZVol_*_future), always use time-aware normalization.")
    print("\nNote: Time-aware normalization is computationally expensive but crucial for accurate")
    print("modeling of non-stationary financial time series, especially for target variables.")
    
    return final_categories


if __name__ == "__main__":
    print("Load your data and run evaluate_indicators_multi_ticker() to evaluate across multiple tickers")
    print("Example:")
    print("    import polars as pl")
    print("    from multi_ticker_evaluation import evaluate_indicators_multi_ticker")
    print("")
    print("    # Specify tickers or let the function sample randomly")
    print("    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META']")
    print("    # Run evaluation")
    print("    categories = evaluate_indicators_multi_ticker(ohlcv, tickers=tickers)")
    print("")
    print("    # Or let the function sample 20 random tickers")
    print("    categories = evaluate_indicators_multi_ticker(ohlcv, n_tickers=20)")