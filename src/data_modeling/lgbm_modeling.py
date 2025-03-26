"""
Memory-efficient LightGBM modeling pipeline for volatility prediction.

This module provides a PyTorch-style approach to LightGBM modeling with
efficient time series data handling, batched processing, and memory management.
"""
import os
import pickle
import datetime
import time
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm.auto import tqdm
import gc
from typing import Dict, List, Tuple, Optional, Union, Any, Callable, Iterator
import tempfile
import logging

# Set up logger
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Constants
MODEL_DIR = "data/models/lgbm_memory_efficient"
TIMEPOINTS = [1, 2, 3, 5, 10, 20, 30]  # Default time points
BATCH_SIZE = 4096  # Default batch size
STRIDE = 10  # Default stride for training instances

# Create directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

def train_lgbm_models_ts_optimized(
    df: pl.DataFrame, 
    target_cols: List[str],
    timepoints: List[int] = TIMEPOINTS,
    train_ratio: float = 0.5,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 10,
    max_batches_per_target: int = 5,
    batch_size: int = BATCH_SIZE,
    stride: int = STRIDE,
    use_disk_offload: bool = True,
    max_stocks_in_memory: int = 200,
    eval_stride_multiplier: int = 10  # New parameter to control evaluation stride
) -> Dict:
    """
    Time series optimized LGBM training pipeline using Polars for efficient data handling.
    
    Parameters:
    -----------
    df : pl.DataFrame
        Input data
    target_cols : List[str]
        List of target columns to predict
    timepoints : List[int]
        List of historical timepoints to use as features
    train_ratio : float
        Ratio of stocks to use for training (vs. testing)
    num_boost_round : int
        Number of boosting rounds
    early_stopping_rounds : int
        Number of rounds for early stopping
    max_batches_per_target : int
        Maximum number of batches to use per target
    batch_size : int
        Batch size for training
    stride : int
        Stride between instances
    use_disk_offload : bool
        Whether to use disk-based storage for stock data
    max_stocks_in_memory : int
        Maximum number of stocks to keep in memory at once
    eval_stride_multiplier : int
        Multiplier for evaluation stride (relative to training stride)
    """
    logger.info(f"Training LGBM models with {len(target_cols)} targets using time series optimized approach")
    start_time = time.time()
    
    # Extract feature columns
    exclude_patterns = ["date", "act_symbol", "_future"]
    feature_cols = [
        col for col in df.columns 
        if not any(pattern in col for pattern in exclude_patterns) 
        and col not in target_cols
    ]
    
    logger.info(f"Using {len(feature_cols)} features × {len(timepoints)} timepoints = {len(feature_cols) * len(timepoints)} total features")
    
    # Split stocks
    stocks = df["act_symbol"].unique().to_list()
    np.random.seed(42)  # Ensure consistent stock splitting
    np.random.shuffle(stocks)
    
    split_idx = int(len(stocks) * train_ratio)
    stocks_A = stocks[:split_idx]
    stocks_B = stocks[split_idx:]
    
    logger.info(f"Stock-based split: {len(stocks_A)} stocks in group A, {len(stocks_B)} in group B")
    
    # Create sets for faster lookups
    stocks_A_set = set(stocks_A)
    stocks_B_set = set(stocks_B)
    all_stocks_set = stocks_A_set | stocks_B_set
    
    # Setup temporary directory for disk offloading if enabled
    temp_dir = None
    if use_disk_offload:
        temp_dir = tempfile.mkdtemp(prefix="lgbm_stock_data_")
        logger.info(f"Using disk offload mode with temporary directory: {temp_dir}")
    
    # Prepare stock data dictionary or file paths
    stock_data = {}
    stock_metadata = {}  # Store metadata like length for each stock
    
    # Get unique stocks only
    unique_stocks = list(all_stocks_set)
    total_stocks = len(unique_stocks)
    logger.info(f"Processing {total_stocks} stocks...")
    
    # First, sort the entire dataframe for consistency
    # This is critical for time series order consistency
    df_sorted = df.sort(["act_symbol", "date"])
    
    # Process stocks in batches to limit memory usage
    for i in tqdm(range(0, total_stocks, max_stocks_in_memory), desc="Processing stock batches"):
        # Get the current batch of stocks
        current_batch_stocks = unique_stocks[i:min(i+max_stocks_in_memory, total_stocks)]
        
        # Filter just this batch
        batch_df = df_sorted.filter(pl.col("act_symbol").is_in(current_batch_stocks))
        
        # Process stocks in this batch
        for stock in tqdm(current_batch_stocks, desc=f"Processing stocks {i+1}-{min(i+max_stocks_in_memory, total_stocks)}", leave=False):
            # Filter just this stock
            stock_chunk = batch_df.filter(pl.col("act_symbol") == stock)
            
            # Skip if empty
            if stock_chunk.height == 0:
                continue
            
            # Extract features and targets directly
            features = stock_chunk.select(feature_cols).fill_null(0).to_numpy()
            targets = stock_chunk.select(target_cols).fill_null(0).to_numpy()
            dates = stock_chunk["date"].to_numpy()
            
            # Replace any remaining NaN or Inf values with zeros
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            targets = np.nan_to_num(targets, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Store metadata for this stock
            stock_metadata[stock] = {
                "length": len(features),
                "dates": dates
            }
            
            if use_disk_offload:
                # Save to disk instead of keeping in memory
                stock_path = os.path.join(temp_dir, f"{stock}.npz")
                # Ensure consistent data format when saving to disk
                np.savez_compressed(stock_path, 
                                   features=features.astype(np.float32), 
                                   targets=targets.astype(np.float32))
                stock_data[stock] = stock_path
            else:
                # Store in memory
                stock_data[stock] = {
                    "features": features,
                    "targets": targets
                }
        
        # Clear batch data to free memory
        del batch_df
        gc.collect()
    
    logger.info(f"Cached data for {len(stock_data)} stocks")
    
    # Helper function to get stock data (either from memory or disk)
    def get_stock_data(stock):
        if use_disk_offload:
            # Load from disk
            data = np.load(stock_data[stock])
            return {
                "features": data["features"],
                "targets": data["targets"],
                "dates": stock_metadata[stock]["dates"]
            }
        else:
            # Return from memory, adding dates
            return {
                "features": stock_data[stock]["features"],
                "targets": stock_data[stock]["targets"],
                "dates": stock_metadata[stock]["dates"]
            }
    
    # Create generator function for batched time series
    def generate_ts_batches(stock_list, batch_size=batch_size, target_idx=0, shuffle=True, custom_stride=None):
        """Generate training or evaluation batches.
        
        Parameters:
        -----------
        stock_list : List[str]
            List of stocks to use
        batch_size : int
            Batch size
        target_idx : int
            Index of target column
        shuffle : bool
            Whether to shuffle samples
        custom_stride : int, optional
            Custom stride to use (if None, use the global stride)
        """
        # Use provided stride or default
        actual_stride = custom_stride if custom_stride is not None else stride
        
        # Find valid indices for each stock
        all_samples = []
        max_lookback = max(timepoints)
        
        # Pre-compute all valid samples
        for stock in stock_list:
            if stock not in stock_data:
                continue
                
            stock_len = stock_metadata[stock]["length"]
            if stock_len > max_lookback:
                # Only include indices with enough history, using the provided stride
                for i in range(max_lookback, stock_len, actual_stride):
                    all_samples.append((stock, i))
        
        # Shuffle if requested
        if shuffle:
            np.random.seed(42)  # Consistent random seed
            np.random.shuffle(all_samples)
            
        # Generate batches
        for start_idx in range(0, len(all_samples), batch_size):
            end_idx = min(start_idx + batch_size, len(all_samples))
            batch_samples = all_samples[start_idx:end_idx]
            
            X_batch = np.zeros((len(batch_samples), len(timepoints) * len(feature_cols)))
            y_batch = np.zeros(len(batch_samples))
            
            # Load all unique stocks for this batch
            needed_stocks = set(stock for stock, _ in batch_samples)
            loaded_stocks = {stock: get_stock_data(stock) for stock in needed_stocks}
            
            for i, (stock, idx) in enumerate(batch_samples):
                # Extract features from timepoints
                features = []
                stock_data_loaded = loaded_stocks[stock]
                
                for t in timepoints:
                    if idx - t >= 0:
                        features.extend(stock_data_loaded["features"][idx - t])
                    else:
                        features.extend([0.0] * len(feature_cols))
                
                X_batch[i] = features
                y_batch[i] = stock_data_loaded["targets"][idx][target_idx]
            
            # Clean up
            del loaded_stocks
            
            # Make sure we have valid data
            X_batch = np.nan_to_num(X_batch, nan=0.0, posinf=0.0, neginf=0.0)
            y_batch = np.nan_to_num(y_batch, nan=0.0, posinf=0.0, neginf=0.0)
            
            yield X_batch, y_batch
    
    # Define training function for a single target
    def train_target(target_idx, train_stocks, test_stocks, target_name):
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'min_data_in_leaf': 20,
            'feature_fraction': 0.7,
            'verbose': -1
        }
        
        # Calculate normalization for the target
        all_target_values = []
        sampled_stocks = np.random.choice(train_stocks, 
                                         min(20, len(train_stocks)),  # Use more stocks
                                         replace=False)
        
        for stock in sampled_stocks:
            if stock in stock_data:
                stock_data_loaded = get_stock_data(stock)
                all_target_values.extend(stock_data_loaded["targets"][:, target_idx])
                del stock_data_loaded
                
        all_target_values = np.array(all_target_values)
        # Filter out extremes
        all_target_values = all_target_values[np.isfinite(all_target_values)]
        
        # Compute robust stats
        target_mean = np.mean(all_target_values)
        target_std = np.std(all_target_values)
        if target_std < 1e-6:
            target_std = 1.0
        
        # Clean up
        del all_target_values
        gc.collect()
        
        logger.info(f"Target {target_name} normalization: mean={target_mean:.6f}, std={target_std:.6f}")
        
        # Initialize model with first batch
        logger.info(f"Training {target_name} in batches...")
        model = None
        
        # Process batches for training
        batch_count = 0
        for X_train, y_train in generate_ts_batches(train_stocks, batch_size=batch_size, target_idx=target_idx):
            # Skip empty batches
            if len(X_train) == 0:
                continue
                
            # Normalize target
            y_train_norm = (y_train - target_mean) / target_std
            
            # Create dataset
            lgb_train = lgb.Dataset(X_train, y_train_norm)
            
            if model is None:
                # First batch: initialize model
                model = lgb.train(params, lgb_train, num_boost_round=20)
            else:
                # Continue training
                model = lgb.train(
                    params, 
                    lgb_train, 
                    num_boost_round=10, 
                    init_model=model,
                    keep_training_booster=True
                )
            
            batch_count += 1
            if batch_count >= max_batches_per_target:
                break
                
            # Free memory
            del X_train, y_train, y_train_norm, lgb_train
            gc.collect()
        
        # Evaluate with increased stride
        logger.info(f"Evaluating {target_name}...")
        all_preds = []
        all_targets = []
        
        # Calculate evaluation stride - use a multiple of the training stride
        eval_stride = stride * eval_stride_multiplier
        logger.info(f"  Using evaluation stride: {eval_stride} (training stride: {stride})")
        
        # Use our generator with the increased stride
        for X_test, y_test in generate_ts_batches(test_stocks, batch_size=batch_size, 
                                                 target_idx=target_idx, shuffle=False, 
                                                 custom_stride=eval_stride):
            if len(X_test) == 0:
                continue
                
            # Predict and denormalize
            preds_norm = model.predict(X_test)
            preds = preds_norm * target_std + target_mean
            
            all_preds.extend(preds)
            all_targets.extend(y_test)
            
            # Free memory
            del X_test, y_test, preds_norm
            gc.collect()
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Remove any invalid values
        valid_idx = np.isfinite(all_preds) & np.isfinite(all_targets)
        all_preds = all_preds[valid_idx]
        all_targets = all_targets[valid_idx]
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds)
        
        # Output some additional diagnostics
        logger.info(f"  Evaluated on {len(all_preds)} test points")
        logger.info(f"  Predictions: min={np.min(all_preds):.6f}, max={np.max(all_preds):.6f}, mean={np.mean(all_preds):.6f}")
        logger.info(f"  Targets: min={np.min(all_targets):.6f}, max={np.max(all_targets):.6f}, mean={np.mean(all_targets):.6f}")
        
        return model, {"rmse": rmse, "mae": mae, "r2": r2, "target_mean": target_mean, "target_std": target_std}
    
    # Train models for all targets
    logger.info("\n=== Training Model 1 (Train on stocks B, Test on stocks A) ===")
    models_1 = []
    metrics_1 = []
    normalization_1 = {"target_means": [], "target_stds": []}
    
    for i, target in enumerate(tqdm(target_cols, desc="Training targets")):
        model, metrics = train_target(i, stocks_B, stocks_A, target)
        models_1.append(model)
        metrics_1.append({**metrics, "target": target})
        normalization_1["target_means"].append(metrics["target_mean"])
        normalization_1["target_stds"].append(metrics["target_std"])
        logger.info(f"Target {target} - RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.6f}")
        gc.collect()
    
    logger.info("\n=== Training Model 2 (Train on stocks A, Test on stocks B) ===")
    models_2 = []
    metrics_2 = []
    normalization_2 = {"target_means": [], "target_stds": []}
    
    for i, target in enumerate(tqdm(target_cols, desc="Training targets")):
        model, metrics = train_target(i, stocks_A, stocks_B, target)
        models_2.append(model)
        metrics_2.append({**metrics, "target": target})
        normalization_2["target_means"].append(metrics["target_mean"])
        normalization_2["target_stds"].append(metrics["target_std"])
        logger.info(f"Target {target} - RMSE: {metrics['rmse']:.6f}, MAE: {metrics['mae']:.6f}, R²: {metrics['r2']:.6f}")
        gc.collect()
    
    # Calculate overall metrics
    combined_metrics = {}
    for target, m1, m2 in zip(target_cols, metrics_1, metrics_2):
        combined_metrics[target] = {
            "rmse": (m1["rmse"] + m2["rmse"]) / 2,
            "mae": (m1["mae"] + m2["mae"]) / 2,
            "r2": (m1["r2"] + m2["r2"]) / 2
        }
    
    # Calculate average metrics
    avg_metrics = {
        "rmse": np.mean([m["rmse"] for m in combined_metrics.values()]),
        "mae": np.mean([m["mae"] for m in combined_metrics.values()]),
        "r2": np.mean([m["r2"] for m in combined_metrics.values()])
    }
    
    logger.info(f"\nAverage metrics - RMSE: {avg_metrics['rmse']:.6f}, MAE: {avg_metrics['mae']:.6f}, R²: {avg_metrics['r2']:.6f}")
    logger.info(f"Total time: {(time.time() - start_time) / 60:.2f} minutes")
    
    # Clean up temporary directory if used
    if use_disk_offload and temp_dir:
        import shutil
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Could not clean up temporary directory ({e})")
    
    return {
        'model_1': models_1,
        'model_2': models_2,
        'feature_cols': feature_cols,
        'target_cols': target_cols,
        'timepoints': timepoints,
        'stocks_A': stocks_A,
        'stocks_B': stocks_B,
        'metrics': {
            'model_1': metrics_1,
            'model_2': metrics_2,
            'combined': combined_metrics,
            'average': avg_metrics
        },
        'normalization_1': normalization_1,
        'normalization_2': normalization_2
    }

# Update the run_lazy_lgbm_pipeline function to include the new parameters
def run_lazy_lgbm_pipeline(
    df: pl.DataFrame,
    target_cols: List[str] = None,
    timepoints: List[int] = TIMEPOINTS,
    train_ratio: float = 0.5,
    num_boost_round: int = 100,
    early_stopping_rounds: int = 10,
    batch_size: int = BATCH_SIZE,
    stride: int = STRIDE,
    batch_mode: str = None,  # None, 'target', or 'window'
    use_disk_offload: bool = True,  # Add new parameter
    max_stocks_in_memory: int = 200,  # Add new parameter
    eval_stride_multiplier: int = 10
) -> Dict:
    """
    Run the memory-efficient LightGBM pipeline with optional batching of targets.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with stock data
    target_cols : List[str], optional
        List of target columns to predict, if None, will use all columns ending with '_future'
    timepoints : List[int]
        List of specific timepoints to use
    train_ratio : float
        Ratio for stock splitting
    num_boost_round : int
        Number of boosting rounds
    early_stopping_rounds : int
        Early stopping rounds
    batch_size : int
        Batch size
    stride : int
        Stride between instances
    batch_mode : str, optional
        How to batch targets: None (no batching), 'target' (by metric type), or 'window' (by time horizon)
    use_disk_offload : bool
        Whether to use disk-based storage for stock data
    max_stocks_in_memory : int
        Maximum number of stocks to keep in memory at once
        
    Returns:
    --------
    Dict with models and results
    """
    # Auto-detect target columns if not specified
    if target_cols is None:
        target_cols = [col for col in df.columns if col.endswith('_future')]
        logger.info(f"Detected {len(target_cols)} target columns ending with '_future'")
    
    # Print stats for target columns
    for target_col in target_cols:
        stats = df.select([
            pl.col(target_col).is_nan().sum().alias("nan_count"),
            pl.col(target_col).is_infinite().sum().alias("inf_count")
        ])
        logger.info(f"Target column {target_col} stats:")
        logger.info(f"  NaNs: {stats[0, 'nan_count']}")
        logger.info(f"  Infs: {stats[0, 'inf_count']}")
        logger.info(df[target_col].describe())
    
    # Replace NaNs and Infs in target columns with None
    for target_col in target_cols:
        df = df.with_columns(
            pl.when(pl.col(target_col).is_nan() | pl.col(target_col).is_infinite())
            .then(None)
            .otherwise(pl.col(target_col))
            .alias(target_col)
        )
    
    # Drop any rows with null values in target columns
    df = df.drop_nulls(subset=target_cols)
    
    # If no batching requested, use the optimized training function
    if batch_mode is None:
        return train_lgbm_models_ts_optimized(
            df=df,
            target_cols=target_cols,
            timepoints=timepoints,
            train_ratio=train_ratio,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            batch_size=batch_size,
            stride=stride,
            use_disk_offload=use_disk_offload,  # Pass new parameter
            max_stocks_in_memory=max_stocks_in_memory,  # Pass new parameter
            eval_stride_multiplier=eval_stride_multiplier
        )
    
    # Handle batching modes
    batches = []
    
    if batch_mode == 'target':
        # Group targets by metric type (YZVol, VolSkew, etc.)
        target_groups = {}
        for col in target_cols:
            # Extract the base metric name (everything before the first underscore after removing "_future")
            base_name = col.replace("_future", "").split("_")[0]
            if base_name not in target_groups:
                target_groups[base_name] = []
            target_groups[base_name].append(col)
        
        # Each metric type becomes a batch
        for group_name, group_cols in target_groups.items():
            batches.append(group_cols)
            
    elif batch_mode == 'window':
        # Group targets by time window (10, 30, 90, etc.)
        window_groups = {}
        for col in target_cols:
            # Extract the window size from the column name
            parts = col.split('_')
            # Find the part that's a number (the window size)
            for part in parts:
                if part.isdigit():
                    window = int(part)
                    if window not in window_groups:
                        window_groups[window] = []
                    window_groups[window].append(col)
                    break
        
        # Each window becomes a batch
        for window in sorted(window_groups.keys()):
            batches.append(window_groups[window])
    else:
        raise ValueError(f"Unknown batch_mode: {batch_mode}. Use None, 'target', or 'window'")
    
    # Print batching information
    logger.info(f"Batching by {batch_mode}, created {len(batches)} batches:")
    for i, batch in enumerate(batches):
        if len(batch) > 3:
            logger.info(f"  Batch {i+1}: {len(batch)} targets - {batch[:3]}...")
        else:
            logger.info(f"  Batch {i+1}: {len(batch)} targets - {batch}")
    
    # Train models for each batch
    all_results = []
    for i, batch_targets in enumerate(batches):
        logger.info(f"\n{'='*50}")
        logger.info(f"Training Batch {i+1}/{len(batches)} with {len(batch_targets)} targets")
        logger.info(f"{'='*50}")
        
        batch_result = train_lgbm_models_ts_optimized(
            df=df,
            target_cols=batch_targets,
            timepoints=timepoints,
            train_ratio=train_ratio,
            num_boost_round=num_boost_round,
            early_stopping_rounds=early_stopping_rounds,
            batch_size=batch_size,  
            stride=stride,
            use_disk_offload=use_disk_offload,  # Pass new parameter
            max_stocks_in_memory=max_stocks_in_memory,  # Pass new parameter
            eval_stride_multiplier=eval_stride_multiplier

        )
        
        all_results.append(batch_result)
        
        # Force garbage collection between batches
        gc.collect()
    
    # Combine results from all batches
    combined_result = {
        'model_1': [],
        'model_2': [],
        'feature_cols': all_results[0]['feature_cols'],
        'target_cols': [],
        'timepoints': timepoints,
        'stocks_A': all_results[0]['stocks_A'],
        'stocks_B': all_results[0]['stocks_B'],
        'metrics': {
            'model_1': [],
            'model_2': [],
            'combined': {},
            'average': {'rmse': 0.0, 'mae': 0.0, 'r2': 0.0}
        },
        'normalization_1': {'target_means': [], 'target_stds': []},
        'normalization_2': {'target_means': [], 'target_stds': []}
    }
    
    # Merge all batch results
    all_rmse = []
    all_mae = []
    all_r2 = []
    
    for result in all_results:
        # Extend models
        combined_result['model_1'].extend(result['model_1'])
        combined_result['model_2'].extend(result['model_2'])
        
        # Extend target columns
        combined_result['target_cols'].extend(result['target_cols'])
        
        # Extend metrics
        combined_result['metrics']['model_1'].extend(result['metrics']['model_1'])
        combined_result['metrics']['model_2'].extend(result['metrics']['model_2'])
        
        # Update combined metrics
        for target, metrics in result['metrics']['combined'].items():
            combined_result['metrics']['combined'][target] = metrics
            all_rmse.append(metrics['rmse'])
            all_mae.append(metrics['mae'])
            all_r2.append(metrics['r2'])
        
        # Extend normalization parameters
        if 'normalization_1' in result:
            combined_result['normalization_1']['target_means'].extend(result['normalization_1']['target_means'])
            combined_result['normalization_1']['target_stds'].extend(result['normalization_1']['target_stds'])
        
        if 'normalization_2' in result:
            combined_result['normalization_2']['target_means'].extend(result['normalization_2']['target_means'])
            combined_result['normalization_2']['target_stds'].extend(result['normalization_2']['target_stds'])
    
    # Calculate average metrics
    if all_rmse:
        combined_result['metrics']['average']['rmse'] = sum(all_rmse) / len(all_rmse)
        combined_result['metrics']['average']['mae'] = sum(all_mae) / len(all_mae)
        combined_result['metrics']['average']['r2'] = sum(all_r2) / len(all_r2)
    
    return combined_result

#############################################
# Prediction Functions
#############################################

def predict_with_ts_lgbm_models(model_results: Dict, new_data: pl.DataFrame, stride: int = 1) -> pl.DataFrame:
    """
    Make predictions using the trained LightGBM models.
    
    Parameters:
    -----------
    model_results : Dict
        Results dictionary from train_lgbm_models_ts_optimized
    new_data : pl.DataFrame
        New data for prediction
    stride : int
        Stride for predictions (1 for every day, higher for fewer predictions)
        
    Returns:
    --------
    pl.DataFrame
        DataFrame with predictions
    """
    model_1 = model_results['model_1']
    model_2 = model_results['model_2']
    feature_cols = model_results['feature_cols']
    target_cols = model_results['target_cols']
    stocks_A = model_results['stocks_A']
    stocks_B = model_results['stocks_B']
    timepoints = model_results['timepoints']
    
    # For normalization
    norm_1 = {
        'target_means': model_results.get('normalization_1', {}).get('target_means'),
        'target_stds': model_results.get('normalization_1', {}).get('target_stds')
    }
    
    norm_2 = {
        'target_means': model_results.get('normalization_2', {}).get('target_means'),
        'target_stds': model_results.get('normalization_2', {}).get('target_stds')
    }
    
    # Preprocess data by stock
    logger.info("Preprocessing prediction data...")
    stock_data = {}
    
    # Convert data to columnar format first for better performance
    df_dict = new_data.to_dict(as_series=False)
    
    for stock in tqdm(list(set(stocks_A + stocks_B)), desc="Preprocessing stocks"):
        # Skip if stock not in data
        if stock not in df_dict["act_symbol"]:
            continue
            
        # Get rows for this stock
        rows = [i for i, s in enumerate(df_dict["act_symbol"]) if s == stock]
        if not rows:
            continue
            
        # Get dates and sort by date
        dates = [df_dict["date"][i] for i in rows]
        date_indices = np.argsort(dates)
        sorted_rows = [rows[i] for i in date_indices]
        
        # Extract features for this stock
        stock_features = np.array([[df_dict[feature_cols[j]][sorted_rows[i]] 
                         if feature_cols[j] in df_dict else 0.0
                         for j in range(len(feature_cols))] 
                        for i in range(len(sorted_rows))])
        
        stock_dates = np.array([df_dict["date"][i] for i in sorted_rows])
        
        # Replace NaN and Inf values with 0
        stock_features = np.nan_to_num(stock_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store in dictionary
        stock_data[stock] = {
            "features": stock_features,
            "dates": stock_dates
        }
    
    # Generate predictions for all stocks
    logger.info("Generating predictions...")
    all_predictions = []
    max_lookback = max(timepoints)
    
    for stock, data in tqdm(stock_data.items(), desc="Predicting"):
        features = data["features"]
        dates = data["dates"]
        
        # Skip if not enough data
        if len(features) <= max_lookback:
            continue
        
        # Determine which model to use based on stock group
        if stock in stocks_A:
            models = model_1
            norm = norm_1
        else:
            models = model_2
            norm = norm_2
        
        # Generate features for prediction points
        for i in range(max_lookback, len(features), stride):
            feature_vector = []
            
            # Extract features from timepoints
            for t in timepoints:
                if i - t >= 0:
                    feature_vector.extend(features[i - t])
                else:
                    feature_vector.extend([0.0] * len(feature_cols))
            
            # Make sure we have valid features
            feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Predict with all target models
            predictions = {}
            for j, (model, target) in enumerate(zip(models, target_cols)):
                # Make prediction
                pred_norm = model.predict([feature_vector])[0]
                
                # Denormalize if normalization info is available
                if norm['target_means'] is not None and norm['target_stds'] is not None:
                    pred = pred_norm * norm['target_stds'][j] + norm['target_means'][j]
                else:
                    pred = pred_norm
                
                predictions[f"{target}_pred"] = pred
            
            # Store prediction
            result = {"act_symbol": stock, "date": dates[i]}
            result.update(predictions)
            all_predictions.append(result)
    
    # Convert to DataFrame
    if not all_predictions:
        logger.warning("No predictions generated!")
        return new_data
        
    predictions_df = pl.DataFrame(all_predictions)
    
    # Join with original data
    result_df = new_data.join(
        predictions_df,
        on=["act_symbol", "date"],
        how="left"
    )
    
    return result_df

def predict_for_visualization(
    model_results: Dict, 
    df: pl.DataFrame,
    stock: str,
    date: Union[str, datetime.date],
    target_pattern: str = "YZVol_{window}_future"
) -> pl.DataFrame:
    """
    Create predictions for a specific stock and date that can be used with visualization functions.
    This produces a single row DataFrame with all required values for visualizing volatility surface.
    
    Parameters:
    -----------
    model_results : Dict
        Model results from training
    df : pl.DataFrame
        Source data containing the stock and date
    stock : str
        Stock symbol to predict for
    date : Union[str, datetime.date]
        Date to predict for
    target_pattern : str
        Pattern for target columns with {window} placeholder
        
    Returns:
    --------
    pl.DataFrame
        Single row DataFrame with stock, date and all predictions
    """
    # Normalize the date format
    if isinstance(date, str):
        date = datetime.datetime.strptime(date, "%Y-%m-%d").date()
    
    # Extract stock data for this date and surrounding context
    # We need enough context for the model's timepoints
    max_timepoint = max(model_results['timepoints'])
    
    # Get earlier dates for context 
    context_data = df.filter(
        (pl.col("act_symbol") == stock) & 
        (pl.col("date") <= date)
    ).sort("date").tail(max_timepoint + 1)
    
    if context_data.height < max_timepoint + 1:
        raise ValueError(f"Not enough context data for stock {stock} on date {date}. Need {max_timepoint + 1} days, got {context_data.height}.")
    
    # Generate predictions
    pred_df = predict_with_ts_lgbm_models(model_results, context_data, stride=1)
    
    # Extract the row for our exact date
    result_row = pred_df.filter(
        (pl.col("act_symbol") == stock) & 
        (pl.col("date") == date)
    )
    
    if result_row.height == 0:
        raise ValueError(f"No prediction generated for {stock} on {date}.")
    
    # Create a modified version of the result that makes it compatible with visualization
    # First identify different target types (YZVol, VolSkew, etc.) and their windows
    target_types = {}
    for col in model_results['target_cols']:
        # Skip any columns that don't end with _future
        if not col.endswith('_future'):
            continue
            
        # Extract base name and window
        parts = col.split('_')
        window = None
        base_name = None
        
        # Find the numeric part (window) and construct the base name
        for i, part in enumerate(parts):
            if part.isdigit():
                window = int(part)
                base_name = '_'.join(parts[:i]) + '_'
                break
        
        if base_name and window:
            if base_name not in target_types:
                target_types[base_name] = []
            target_types[base_name].append(window)
    
    # Extract prediction columns
    pred_columns = [col for col in result_row.columns if col.endswith('_pred')]
    
    # Create the visualization-ready row
    viz_data = {"act_symbol": stock, "date": date}
    
    # Copy over any existing metrics we predict
    for col in pred_columns:
        # Extract the original target name by removing _pred suffix
        orig_target = col[:-5]
        
        # Add the prediction as if it were the actual value
        viz_data[orig_target] = result_row[col][0]
    
    # Ensure we have basic columns for visualization (fill with defaults if needed)
    # All volatility metrics
    for metric_type in ['YZVol_', 'VolSkew_', 'VolCurvature_', 'MeanReversion_', 'VolOfVol_', 'PriceVolCorr_', 'VolIntensity_']:
        windows = target_types.get(metric_type, [])
        
        if not windows:
            # Try to infer windows from other metrics
            all_windows = []
            for windows_list in target_types.values():
                all_windows.extend(windows_list)
            windows = sorted(list(set(all_windows)))
        
        # Create columns for each window
        for window in windows:
            col_name = f"{metric_type}{window}_future"
            if col_name not in viz_data:
                # If we don't have a prediction for this column, check if it's in the original data
                if col_name in result_row.columns:
                    viz_data[col_name] = result_row[col_name][0]
                else:
                    # Set default values based on metric type
                    if metric_type == 'YZVol_':
                        viz_data[col_name] = 0.25  # Default volatility
                    elif metric_type == 'VolSkew_':
                        viz_data[col_name] = -0.4  # Default negative skew for equities
                    elif metric_type == 'VolCurvature_':
                        viz_data[col_name] = 0.5  # Default positive curvature
                    elif metric_type == 'MeanReversion_':
                        viz_data[col_name] = 2.0  # Default mean reversion speed
                    elif metric_type == 'VolOfVol_':
                        viz_data[col_name] = 0.3  # Default vol-of-vol
                    elif metric_type == 'PriceVolCorr_':
                        viz_data[col_name] = -0.6  # Default negative correlation for equities
                    elif metric_type == 'VolIntensity_':
                        viz_data[col_name] = 0.5  # Default medium intensity
    
    # Create DataFrame from the constructed row
    return pl.DataFrame([viz_data])

def generate_volatility_predictions(
    model_results: Dict,
    df: pl.DataFrame,
    stocks: List[str],
    start_date: Union[str, datetime.date],
    end_date: Union[str, datetime.date],
    stride: int = 5
) -> pl.DataFrame:
    """
    Generate volatility predictions for multiple stocks over a date range.
    
    Parameters:
    -----------
    model_results : Dict
        Model results from training
    df : pl.DataFrame
        Source data
    stocks : List[str]
        Stock symbols to predict for
    start_date : Union[str, datetime.date]
        Start date for predictions
    end_date : Union[str, datetime.date]
        End date for predictions
    stride : int
        Stride for date selection (1 for daily, higher for sparser selection)
        
    Returns:
    --------
    pl.DataFrame
        DataFrame with predictions for all stocks and dates
    """
    # Normalize date formats
    if isinstance(start_date, str):
        start_date = datetime.datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.datetime.strptime(end_date, "%Y-%m-%d").date()
    
    # Filter data for the given stocks and date range
    filtered_data = df.filter(
        (pl.col("act_symbol").is_in(stocks)) &
        (pl.col("date") >= start_date) &
        (pl.col("date") <= end_date)
    )
    
    # Get unique dates within range and apply stride
    all_dates = filtered_data["date"].unique().sort()
    selected_dates = all_dates[::stride]
    
    # Generate predictions
    pred_df = predict_with_ts_lgbm_models(
        model_results=model_results,
        new_data=filtered_data,
        stride=stride
    )
    
    return pred_df

#############################################
# Model Serialization Functions
#############################################

def save_lgbm_model(model_results: Dict, model_dir: str = MODEL_DIR) -> str:
    """
    Save the LGBM model and metadata efficiently.
    
    Parameters:
    -----------
    model_results : Dict
        Model results from training
    model_dir : str
        Directory to save the model
        
    Returns:
    --------
    str
        Path to the saved model
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create timestamp for directory name
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create model-specific directory
    model_path = os.path.join(model_dir, f"lgbm_model_{timestamp}")
    os.makedirs(model_path, exist_ok=True)
    
    logger.info(f"Saving models to {model_path}")
    
    # Save models separately
    for i, model in enumerate(model_results['model_1']):
        model.save_model(os.path.join(model_path, f"model_1_{i}.txt"))
    
    for i, model in enumerate(model_results['model_2']):
        model.save_model(os.path.join(model_path, f"model_2_{i}.txt"))
    
    # Save metadata separately
    metadata = {
        'feature_cols': model_results['feature_cols'],
        'target_cols': model_results['target_cols'],
        'timepoints': model_results['timepoints'],
        'stocks_A': model_results['stocks_A'],
        'stocks_B': model_results['stocks_B'],
        'metrics': model_results['metrics'],
        'normalization_1': model_results.get('normalization_1', {}),
        'normalization_2': model_results.get('normalization_2', {}),
        'timestamp': timestamp
    }
    
    with open(os.path.join(model_path, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
    
    logger.info(f"Model saved to {model_path}")
    return model_path


def load_lgbm_model(filepath: str = None, model_dir: str = MODEL_DIR) -> Dict:
    """
    Load the LGBM model and metadata.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the specific model directory to load
    model_dir : str
        Directory containing model directories
        
    Returns:
    --------
    Dict
        Model results
    """
    # If no filepath provided, find the most recent model directory
    if filepath is None:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
        # Get all model directories
        model_dirs = [
            os.path.join(model_dir, d) for d in os.listdir(model_dir) 
            if d.startswith("lgbm_model_") and os.path.isdir(os.path.join(model_dir, d))
        ]
        
        if not model_dirs:
            raise FileNotFoundError(f"No model directories found in {model_dir}")
        
        # Sort by modification time, newest first
        filepath = max(model_dirs, key=os.path.getmtime)
    
    logger.info(f"Loading model from {filepath}")
    
    # Load metadata
    with open(os.path.join(filepath, "metadata.pkl"), "rb") as f:
        metadata = pickle.load(f)
    
    # Load models
    model_1 = []
    model_2 = []
    
    # Load model 1
    i = 0
    while os.path.exists(os.path.join(filepath, f"model_1_{i}.txt")):
        model = lgb.Booster(model_file=os.path.join(filepath, f"model_1_{i}.txt"))
        model_1.append(model)
        i += 1
    
    # Load model 2
    i = 0
    while os.path.exists(os.path.join(filepath, f"model_2_{i}.txt")):
        model = lgb.Booster(model_file=os.path.join(filepath, f"model_2_{i}.txt"))
        model_2.append(model)
        i += 1
    
    # Construct result dictionary
    result = {
        'model_1': model_1,
        'model_2': model_2,
        'feature_cols': metadata['feature_cols'],
        'target_cols': metadata['target_cols'],
        'timepoints': metadata['timepoints'],
        'stocks_A': metadata['stocks_A'],
        'stocks_B': metadata['stocks_B'],
        'metrics': metadata['metrics'],
        'normalization_1': metadata.get('normalization_1', {}),
        'normalization_2': metadata.get('normalization_2', {})
    }
    
    logger.info(f"Model loaded successfully with {len(model_1)} targets")
    
    return result

#############################################
# Main Execution
#############################################

if __name__ == "__main__":
    # Example usage
    from src.data_transformation import transformation_pipeline
    
    # Load data
    logger.info("Loading data...")
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv").with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # For testing with a smaller dataset
    # Use a limited number of stocks
    test_stocks = ohlcv["act_symbol"].unique()[:50]  # Use only first 50 stocks
    ohlcv = ohlcv.filter(pl.col("act_symbol").is_in(test_stocks))
    
    # Apply transformations
    logger.info("Applying transformations...")
    data = transformation_pipeline.transformation_pipeline(ohlcv)
    
    # Define target columns for faster testing - using only a few key targets
    all_future_cols = [col for col in data.columns if col.endswith("_future")]
    target_cols = [col for col in all_future_cols if any(f"YZVol_{days}_future" == col for days in [10, 30, 90])]
    
    # Run optimized pipeline
    model_results = train_lgbm_models_ts_optimized(
        df=data,
        target_cols=target_cols,
        timepoints=[1, 2, 3, 5, 10],  # Use fewer timepoints for faster training
        train_ratio=0.5,
        num_boost_round=50,
        early_stopping_rounds=5,
        max_batches_per_target=3  # Limit batches for testing
    )
    
    # Save model
    model_path = save_lgbm_model(model_results)
    
    # Example of generating visualization-ready prediction for a single stock/date
    try:
        sample_stock = "AAPL"
        sample_date = "2020-03-15"
        pred_df = predict_for_visualization(
            model_results=model_results,
            df=data,
            stock=sample_stock,
            date=sample_date
        )
        logger.info(f"Generated visualization-ready prediction for {sample_stock} on {sample_date}")
        logger.info(pred_df)
    except Exception as e:
        logger.error(f"Error generating visualization prediction: {e}")