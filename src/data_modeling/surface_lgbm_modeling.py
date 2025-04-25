# surface_lgbm_modeling.py
# ---------------------------------------------------------------
# Global LGBM realized‐volatility‐surface trainer
# Trains one model per horizon with A/B stock split for cross-validation,
# streams data efficiently using memory-mapping and LRU cache,
# incorporates feature interactions with log-strike 'k',
# weights samples based on 'k' magnitude, allows for extended training,
# and leverages GPU if available.
# ---------------------------------------------------------------

import os
import zlib  # Added for model compression
import shutil
import joblib
import logging
import datetime as _dt
from typing import Dict, Any, List, Tuple, Union, Generator, Optional
from collections import OrderedDict
import tempfile

import numpy as np
import polars as pl
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, r2_score
import gc

# ────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────
MODEL_DIR           = "data/models/surface_lgbm"
HORIZONS            = [10, 15, 20, 25, 30, 35]
BATCH_SIZE          = 8192          # Samples per batch
MAX_STOCKS_IN_CACHE = 512           # Max stocks data (mmap objects) held in LRU cache
K_GRID_POINTS       = 101           # strike‐grid density
SEED                = 42            # Global random seed
K_RANGE             = [-1, 1]       # Range for log-moneyness 'k'

# --- Sample Weighting Configuration ---
# Increase weight for samples further from ATM (larger abs(k))
# weight = 1 + K_WEIGHT_FACTOR * abs(k)
K_WEIGHT_FACTOR     = 20.0

# --- Compression Configuration --- 
COMPRESS_MODELS     = True          # Whether to compress model files
COMPRESSION_LEVEL   = 9             # Zlib compression level (1-9)
MAX_MODEL_SIZE_MB   = 100           # Compress models larger than this size (MB)

# Features to remove (consistently low/zero importance or constant per horizon)
FEATURES_TO_REMOVE = {
    "Tcal", # Removed because it's constant within each horizon model
    "gain_10", "gain_20", "gain_35",
    "loss_10", "loss_20", "loss_35",
    "price_delta"
}

# --- Extended Training Hyperparameters ---
LGBM_PARAMS_GPU = {
    "objective":      "regression",
    "boosting_type":  "gbdt",
    "device":         "gpu",
    "metric":         "rmse",
    "learning_rate":  0.005, 
    "num_leaves":     255, 
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq":     1,
    "seed":             SEED,
    "num_threads":     -1,
    "verbose":         -1
}

N_ESTIMATORS = 30000 
EARLY_STOPPING_ROUNDS = 300

# Set seed globally
np.random.seed(SEED)

# ────────────────────────────────────────────────────────────────
# Logger
# ────────────────────────────────────────────────────────────────
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# ────────────────────────────────────────────────────────────────
# Utilities (with added compression utilities)
# ────────────────────────────────────────────────────────────────
def _ensure_dir(path: str):
    """Ensure directory exists."""
    os.makedirs(path, exist_ok=True)

def _clean_dir(path: str):
    """Remove directory tree if exists, handling potential errors."""
    if path and os.path.isdir(path):
        try:
            shutil.rmtree(path)
            logger.info(f"Removed temp dir {path}")
        except OSError as e:
            logger.error(f"Error removing directory {path}: {e}")

def _to_date(x: Union[str, _dt.date, _dt.datetime, np.datetime64]) -> _dt.date:
    """Convert various date formats to date object."""
    if isinstance(x, str): return _dt.date.fromisoformat(x)
    if isinstance(x, _dt.datetime): return x.date()
    if isinstance(x, np.datetime64): return x.astype('datetime64[D]').item().date()
    if isinstance(x, _dt.date): return x
    raise TypeError(f"Cannot convert {type(x)} to date")

# --- New compression utilities ---
def _save_model(model: lgb.Booster, path: str, compress: bool = COMPRESS_MODELS) -> Tuple[str, float]:
    """
    Save LightGBM model, with optional compression.
    
    Args:
        model: LightGBM Booster object
        path: Path to save the model
        compress: Whether to apply compression
        
    Returns:
        Tuple of (final_path, size_mb)
    """
    if not compress:
        # Standard save without compression
        model.save_model(path)
        size_mb = os.path.getsize(path) / (1024 * 1024)
        return path, size_mb
    
    # Try saving uncompressed first to a temporary file
    temp_dir = os.path.dirname(path)
    _ensure_dir(temp_dir)
    temp_path = f"{path}.temp"
    
    try:
        model.save_model(temp_path)
        size_mb = os.path.getsize(temp_path) / (1024 * 1024)
        
        # Only compress if larger than MAX_MODEL_SIZE_MB
        if size_mb <= MAX_MODEL_SIZE_MB:
            # If under size limit, just rename to final path
            os.rename(temp_path, path)
            return path, size_mb
        
        # Compress the model
        with open(temp_path, 'rb') as f:
            model_data = f.read()
        
        compressed_data = zlib.compress(model_data, level=COMPRESSION_LEVEL)
        compressed_path = f"{path}.z"
        
        with open(compressed_path, 'wb') as f:
            f.write(compressed_data)
        
        compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
        logger.info(f"Model compressed: {size_mb:.2f}MB -> {compressed_size_mb:.2f}MB "
                   f"(ratio: {size_mb/compressed_size_mb:.2f}x)")
        
        return compressed_path, compressed_size_mb
    
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)

def _load_model(path: str) -> Optional[lgb.Booster]:
    """
    Load LightGBM model, handling both compressed and uncompressed formats.
    
    Args:
        path: Path to the model file
        
    Returns:
        LightGBM Booster or None if loading fails
    """
    # Check for compressed version first
    compressed_path = f"{path}.z"
    
    if os.path.exists(compressed_path):
        try:
            with open(compressed_path, 'rb') as f:
                compressed_data = f.read()
            
            # Decompress the data
            model_data = zlib.decompress(compressed_data)
            
            # Create a temporary file to use with LightGBM's loader
            temp_path = f"{path}.temp"
            with open(temp_path, 'wb') as f:
                f.write(model_data)
            
            try:
                model = lgb.Booster(model_file=temp_path)
                return model
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                    
        except Exception as e:
            logger.error(f"Failed to load compressed model from {compressed_path}: {e}")
    
    # Fall back to uncompressed version
    if os.path.exists(path):
        try:
            return lgb.Booster(model_file=path)
        except Exception as e:
            logger.error(f"Failed to load uncompressed model from {path}: {e}")
    
    return None

# --- Updated LRU Cache to handle weights ---
class MMapLRUCache:
    """LRU Cache for NumPy memory-mapped arrays (X, y, w)."""
    def __init__(self, capacity: int):
        self.cache = OrderedDict()
        self.capacity = capacity

    def get(self, key: str) -> Optional[Dict[str, np.ndarray]]:
        """Get item from cache, move to end."""
        if key not in self.cache: return None
        self.cache.move_to_end(key)
        return self.cache[key] # Returns {'X': mmap, 'y': mmap, 'w': mmap}

    def put(self, key: str, value: Dict[str, str]):
        """Put item paths into cache, load mmaps, handle eviction."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return # Already cached

        if len(self.cache) >= self.capacity:
            self.cache.popitem(last=False) # Evict LRU

        try:
            # Load all three arrays using memory mapping
            X = np.load(value['X_path'], mmap_mode='r')
            y = np.load(value['y_path'], mmap_mode='r')
            w = np.load(value['w_path'], mmap_mode='r') # Load weights
            self.cache[key] = {'X': X, 'y': y, 'w': w} # Store dict of mmaps
            logger.debug(f"Cache loaded and stored mmap for: {key}")
        except FileNotFoundError as e:
            logger.error(f"File not found for cache key {key}: {e}")
            if key in self.cache: del self.cache[key]
        except Exception as e:
            logger.error(f"Error loading mmap for key {key}: {e}")
            if key in self.cache: del self.cache[key]

    def __len__(self) -> int:
        return len(self.cache)

# Global cache instance
mmap_cache = MMapLRUCache(capacity=MAX_STOCKS_IN_CACHE)

# ────────────────────────────────────────────────────────────────
# Prepare per‐horizon arrays (saving X, y, and weights)
# ────────────────────────────────────────────────────────────────
def _prepare_horizon_data(
    df: pl.DataFrame,
    h: int,
    temp_dir: str
) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Dict[str, int]], List[str]]:
    """
    Prepares feature (X), target (y), and sample weight (w) data for horizon `h`.
    Includes interaction terms involving 'k', removes low-importance features.
    Weights are calculated based on abs(k). Data saved to disk as `.npy`.

    Args:
        df: DataFrame with base features and target precursors.
        h: Prediction horizon (days).
        temp_dir: Directory for temporary `.npy` files.

    Returns:
        Tuple: (stock_data_paths, stock_meta, final_feature_cols)
         - stock_data_paths maps symbol to {'X_path': str, 'y_path': str, 'w_path': str}.
         - stock_meta maps symbol to {'length': int}.
         - final_feature_cols is the list of feature names used.
    """
    _ensure_dir(temp_dir)
    logger.info(f"Horizon {h}: Preparing data (incl. weights) using temp dir: {temp_dir}")

    exclude = {"date", "act_symbol", "close"}
    for hv in HORIZONS:
        exclude.add(f"rv_{hv}d_future")
        exclude.add(f"log_ret_future_{hv}")

    base_feature_cols = sorted([
        c for c in df.columns if c not in exclude and c not in FEATURES_TO_REMOVE
    ])
    interaction_bases = ["rv_35d", "rv_10d", "vix", "vvix", "ovx", "ATR_35", "intraday_range_var"]
    cols_to_select = sorted(list(set(base_feature_cols + interaction_bases)))

    rv_col = f"rv_{h}d_future"
    lr_col = f"log_ret_future_{h}"

    stock_data_paths: Dict[str, Dict[str, str]] = {}
    stock_meta: Dict[str, Dict[str, int]] = {}
    symbols = df["act_symbol"].unique().sort().to_list()
    total_symbols = len(symbols)
    processed_symbols = 0

    for sym in symbols:
        sym_df = df.filter(pl.col("act_symbol") == sym).sort("date")

        required_cols = cols_to_select + [rv_col, lr_col]
        missing_req = [col for col in required_cols if col not in sym_df.columns]
        if sym_df.is_empty() or missing_req:
            logger.warning(f"H{h}, Sym {sym}: Skipping - empty data or missing required columns: {missing_req}")
            continue

        tmp = (
            sym_df
            .select(required_cols)
            .with_columns([
                pl.col(lr_col).alias("k"),
                (pl.col(rv_col) / (h/252.0)).sqrt().alias("sigma")
            ])
            # Drop based on k, sigma, AND interaction bases now
            .drop_nulls(subset=["k", "sigma"] + interaction_bases)
        )

        if tmp.is_empty():
            logger.warning(f"H{h}, Sym {sym}: Skipping - no valid samples after target/null drop.")
            continue

        # Add interactions and sample weights
        interaction_expressions = [
            pl.col("k").pow(2).alias("k_squared"),
            pl.col("k").mul(pl.col("rv_35d")).alias("k_x_rv35d"),
            pl.col("k").mul(pl.col("rv_10d")).alias("k_x_rv10d"),
            pl.col("k").mul(pl.col("vix")).alias("k_x_vix"),
            pl.col("k").mul(pl.col("vvix")).alias("k_x_vvix"),
            pl.col("k").mul(pl.col("ovx")).alias("k_x_ovx"),
            pl.col("k").mul(pl.col("ATR_35")).alias("k_x_ATR35"),
            pl.col("k").mul(pl.col("intraday_range_var")).alias("k_x_intraday_range_var")
        ]
        interaction_names = [expr.meta.output_name() for expr in interaction_expressions]
        weight_expression = (1.0 + K_WEIGHT_FACTOR * pl.col("k").abs()).alias("sample_weight")

        try:
            # Add weights and interactions in one go
            tmp = tmp.with_columns(interaction_expressions + [weight_expression])
        except Exception as e:
             logger.error(f"H{h}, Sym {sym}: Error calculating interactions/weights: {e}. Skipping.")
             continue

        final_feature_cols = base_feature_cols + ["k"] + interaction_names

        try:
            # Select final features, target, and weights
            Xh = tmp.select(final_feature_cols).to_numpy().astype(np.float32)
            yh = tmp.select("sigma").to_numpy().flatten().astype(np.float32)
            wh = tmp.select("sample_weight").to_numpy().flatten().astype(np.float32) # Extract weights
        except pl.ColumnNotFoundError as e:
            logger.error(f"H{h}, Sym{sym}: Column not found during final select: {e}. Skipping.")
            continue

        Xh = np.nan_to_num(Xh, nan=0.0, posinf=1e6, neginf=-1e6)
        yh = np.nan_to_num(yh, nan=0.0, posinf=1e6, neginf=-1e6)
        wh = np.nan_to_num(wh, nan=1.0, posinf=1e6, neginf=1.0) # Default weight 1 if NaN occurs
        wh = np.maximum(1e-6, wh) # Ensure weights are positive

        if Xh.shape[0] == 0:
             logger.warning(f"H{h}, Sym {sym}: Skipping - zero samples after NaN handling.")
             continue

        sym_safe = sym.replace(os.path.sep, "_").replace(":","_")
        x_path = os.path.join(temp_dir, f"h{h}_{sym_safe}_X.npy")
        y_path = os.path.join(temp_dir, f"h{h}_{sym_safe}_y.npy")
        w_path = os.path.join(temp_dir, f"h{h}_{sym_safe}_w.npy") # Path for weights

        try:
            np.save(x_path, Xh)
            np.save(y_path, yh)
            np.save(w_path, wh) # Save weights
        except Exception as e:
             logger.error(f"H{h}, Sym {sym}: Failed to save .npy files (X/y/w): {e}")
             continue

        # Store paths for X, y, and w
        stock_data_paths[sym] = {"X_path": x_path, "y_path": y_path, "w_path": w_path}
        stock_meta[sym] = {"length": len(yh)}
        processed_symbols += 1

        if processed_symbols % 100 == 0:
            logger.info(f"H{h}: Prepared data for {processed_symbols}/{total_symbols} symbols.")
            gc.collect()

    final_feature_count = len(final_feature_cols)
    logger.info(f"H{h}: Finished preparing data for {processed_symbols} symbols. Final feature count: {final_feature_count}")

    return stock_data_paths, stock_meta, final_feature_cols


# ────────────────────────────────────────────────────────────────
# Batch generator (Updated to yield weights)
# ────────────────────────────────────────────────────────────────
def _generate_batches(
    symbols: List[str],
    stock_data_paths: Dict[str, Dict[str, str]],
    stock_meta: Dict[str, Dict[str, int]],
    feat_len: int,
    shuffle: bool = True,
    batch_size: int = BATCH_SIZE
) -> Generator[Tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """
    Generates batches of (X, y, w) data using memory-mapping and LRU cache.

    Args:
        symbols: List of stock symbols.
        stock_data_paths: Dict mapping symbol to paths {'X_path', 'y_path', 'w_path'}.
        stock_meta: Dict mapping symbol to metadata.
        feat_len: Expected number of features in X.
        shuffle: If True, shuffle symbol order and indices within symbols.
        batch_size: Samples per batch.

    Yields:
        Tuples of (X_batch, y_batch, w_batch).
    """
    global mmap_cache

    symbols_shuffled = symbols[:] if shuffle else symbols
    if shuffle: np.random.shuffle(symbols_shuffled)

    X_buffer = np.zeros((batch_size, feat_len), dtype=np.float32)
    y_buffer = np.zeros(batch_size, dtype=np.float32)
    w_buffer = np.zeros(batch_size, dtype=np.float32) # Buffer for weights
    buffer_idx = 0

    for sym in symbols_shuffled:
        if sym not in stock_meta or sym not in stock_data_paths: continue
        n_samples = stock_meta[sym]["length"]
        if n_samples == 0: continue

        cached_data = mmap_cache.get(sym)
        if cached_data is None:
            mmap_cache.put(sym, stock_data_paths[sym])
            cached_data = mmap_cache.get(sym)
        if cached_data is None: continue # Skip if still None after put/get

        X_sym = cached_data['X']
        y_sym = cached_data['y']
        w_sym = cached_data['w'] # Get weights from cache

        # Consistency checks (include weights)
        if not (X_sym.shape[0] == y_sym.shape[0] == w_sym.shape[0] == n_samples):
            logger.error(f"Inconsistent lengths for {sym}: meta={n_samples}, X={X_sym.shape[0]}, y={y_sym.shape[0]}, w={w_sym.shape[0]}. Skip.")
            continue
        if X_sym.shape[1] != feat_len:
             logger.error(f"Inconsistent feature length for {sym}: expected={feat_len}, got={X_sym.shape[1]}. Skip.")
             continue

        indices = np.arange(n_samples)
        if shuffle: np.random.shuffle(indices)

        for idx in indices:
            X_buffer[buffer_idx] = X_sym[idx]
            y_buffer[buffer_idx] = y_sym[idx]
            w_buffer[buffer_idx] = w_sym[idx] # Add weight to buffer
            buffer_idx += 1

            if buffer_idx == batch_size:
                yield X_buffer, y_buffer, w_buffer # Yield weights
                buffer_idx = 0

    if buffer_idx > 0:
        yield X_buffer[:buffer_idx], y_buffer[:buffer_idx], w_buffer[:buffer_idx] # Yield weights


# ────────────────────────────────────────────────────────────────
# Train single model (Updated to use weights)
# ────────────────────────────────────────────────────────────────
def _train_single_model(
    h: int,
    train_syms: List[str],
    test_syms: List[str],
    stock_data_paths: Dict[str, Dict[str, str]],
    stock_meta: Dict[str, Dict[str, int]],
    feature_names: List[str],
    model_label: str
) -> Tuple[Optional[lgb.Booster], Dict[str, float]]:
    """
    Trains a single LightGBM model for horizon `h` using sample weights.
    Loads all training/validation data into memory for a single `lgb.train` call.

    Args:
        h: Horizon in days.
        train_syms: Symbols for training.
        test_syms: Symbols for validation.
        stock_data_paths: Dict with paths to X, y, w files.
        stock_meta: Symbol metadata.
        feature_names: List of final feature names.
        model_label: Label for logging.

    Returns:
        Tuple: (trained Booster or None, evaluation metrics).
    """
    feat_len = len(feature_names)
    metrics = {"rmse": np.nan, "r2": np.nan}
    booster = None

    logger.info(f"H{h} {model_label}: Loading data (incl. weights) for training and validation.")

    # Helper to load X, y, and w data
    def load_data_for_symbols(symbols: List[str], label: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], int]:
        X_list, y_list, w_list = [], [], []
        total_samples = 0
        for sym in symbols:
            if sym not in stock_meta or sym not in stock_data_paths: continue
            n = stock_meta[sym]['length']
            paths = stock_data_paths[sym]
            if n == 0 or 'w_path' not in paths: continue # Ensure weight path exists
            try:
                X_sym = np.load(paths['X_path'])
                y_sym = np.load(paths['y_path'])
                w_sym = np.load(paths['w_path']) # Load weights
                if (X_sym.shape[0] == n and y_sym.shape[0] == n and
                    w_sym.shape[0] == n and X_sym.shape[1] == feat_len):
                    X_list.append(X_sym)
                    y_list.append(y_sym)
                    w_list.append(w_sym) # Append weights
                    total_samples += n
                else: logger.warning(f"H{h} {model_label}: Data/Weight inconsistency for {label} symbol {sym}. Skipping.")
            except Exception as e:
                logger.error(f"H{h} {model_label}: Failed to load {label} data/weights for {sym}: {e}")
        return X_list, y_list, w_list, total_samples

    X_train_list, y_train_list, w_train_list, train_samples = load_data_for_symbols(train_syms, "train")
    X_test_list, y_test_list, w_test_list, test_samples = load_data_for_symbols(test_syms, "test")

    if not X_train_list or not X_test_list:
        logger.error(f"H{h} {model_label}: Insufficient data/weights loaded. Aborting.")
        return None, metrics

    try:
        logger.info(f"H{h} {model_label}: Concatenating data/weights... Train: {train_samples}, Test: {test_samples}")
        X_train_all = np.vstack(X_train_list); y_train_all = np.concatenate(y_train_list); w_train_all = np.concatenate(w_train_list)
        X_test_all = np.vstack(X_test_list); y_test_all = np.concatenate(y_test_list); w_test_all = np.concatenate(w_test_list)
        del X_train_list, y_train_list, w_train_list, X_test_list, y_test_list, w_test_list
        gc.collect()
        logger.info(f"H{h} {model_label}: Concatenation complete.")

        logger.info(f"H{h} {model_label}: Creating LightGBM datasets with weights...")
        # Pass weights to lgb.Dataset
        dtrain = lgb.Dataset(X_train_all, label=y_train_all, weight=w_train_all, feature_name=feature_names, free_raw_data=False)
        dvalid = lgb.Dataset(X_test_all, label=y_test_all, weight=w_test_all, feature_name=feature_names, reference=dtrain, free_raw_data=False)
        logger.info(f"H{h} {model_label}: Datasets created.")

        logger.info(f"H{h} {model_label}: Starting training with {N_ESTIMATORS} rounds, LR={LGBM_PARAMS_GPU['learning_rate']}, stopping={EARLY_STOPPING_ROUNDS})...")
        evals_result = {}
        booster = lgb.train(
            LGBM_PARAMS_GPU, dtrain,
            num_boost_round=N_ESTIMATORS,
            valid_sets=[dtrain, dvalid], valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(evals_result)
            ]
        )
        logger.info(f"H{h} {model_label}: Training finished. Best iteration: {booster.best_iteration}")

        logger.info(f"H{h} {model_label}: Evaluating final model on test set ({test_samples} samples)...")
        preds_all = booster.predict(X_test_all, num_iteration=booster.best_iteration)
        trues_all = y_test_all

        valid_idx = np.isfinite(preds_all) & np.isfinite(trues_all)
        if np.sum(~valid_idx) > 0:
             logger.warning(f"H{h} {model_label}: Found {np.sum(~valid_idx)} invalid preds/targets. Excluding.")
             preds_all = preds_all[valid_idx]; trues_all = trues_all[valid_idx]

        if len(trues_all) > 0:
            # Note: Standard metrics are unweighted. Weighted metrics could also be calculated.
            rmse = np.sqrt(mean_squared_error(trues_all, preds_all))
            r2 = r2_score(trues_all, preds_all)
            metrics = {"rmse": rmse, "r2": r2}
            logger.info(f"H{h} {model_label}: Final Test Metrics - RMSE={rmse:.4f}, R²={r2:.4f}")
        else: logger.warning(f"H{h} {model_label}: No valid samples for final evaluation.")

        if booster:
            try: # Print feature importances (split and gain)
                logger.info(f"--- H{h} {model_label} Feature Importances ---")
                if len(feature_names) == booster.num_feature():
                    for imp_type in ['gain', 'split']:
                         importance = booster.feature_importance(importance_type=imp_type)
                         feat_imp = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
                         logger.info(f"--- Top 30 by {imp_type} ---")
                         for name, imp in feat_imp[:30]: logger.info(f"{name}: {imp:.4f}")
                         k_related = {n: f"{imp:.4f}" for n, imp in feat_imp if n.startswith('k')}
                         logger.info(f"K-related ({imp_type}): {k_related}")
                else: logger.warning("Feature name/count mismatch, skipping importance.")
            except Exception as imp_err: logger.error(f"Failed to print importance: {imp_err}")

    except MemoryError:
        logger.error(f"H{h} {model_label}: MemoryError during data/weight concatenation or LGBM dataset creation.", exc_info=True)
        gc.collect(); return None, metrics
    except Exception as e:
        logger.error(f"H{h} {model_label}: Unexpected error during training/evaluation: {e}", exc_info=True)
        return None, metrics
    finally:
         try: del X_train_all, y_train_all, w_train_all, X_test_all, y_test_all, w_test_all, dtrain, dvalid
         except NameError: pass
         gc.collect()

    return booster, metrics


# ────────────────────────────────────────────────────────────────
# Train models for all horizons (Updated to use compression)
# ────────────────────────────────────────────────────────────────
def train_surface_models(
    df: pl.DataFrame,
    model_name: str = None,
) -> Dict[str, Any]:
    """
    Train models for each horizon with A/B cross-validation using disk-backed data.
    Features include interactions, low-importance ones removed, samples weighted by k.

    Args:
        df: Input DataFrame with base features and target precursors.
        model_name: Optional name for the model run.

    Returns:
        Dictionary with trained models, metrics, features used, etc.
    """
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    md = model_name or f"surface_{ts}"
    model_dir = os.path.join(MODEL_DIR, md)
    _ensure_dir(model_dir)
    temp_root = tempfile.mkdtemp(prefix=f"lgbm_prep_{ts}_", dir=os.path.dirname(model_dir))
    logger.info(f"Using temporary directory: {temp_root}")

    result = { "models": {}, "metrics": {}, "feature_cols": {}, "stocks": {},
               "model_dir": model_dir, "timestamp": ts, "horizons": HORIZONS }

    all_symbols = df["act_symbol"].unique().sort().to_list()
    if not all_symbols:
        logger.error("No symbols found. Aborting."); _clean_dir(temp_root); return result

    np.random.shuffle(all_symbols)
    split_idx = len(all_symbols) // 2
    base_stocks_A, base_stocks_B = all_symbols[:split_idx], all_symbols[split_idx:]
    logger.info(f"Global stock split: {len(base_stocks_A)} A, {len(base_stocks_B)} B")

    try:
        for h in HORIZONS:
            horizon_temp_dir = os.path.join(temp_root, f"horizon_{h}")
            logger.info(f"=== Horizon {h}: Preparing data ===")
            try:
                # Prepare data (X, y, w) and get final feature names
                sd_paths, sm, final_feature_names = _prepare_horizon_data(df, h, horizon_temp_dir)
            except Exception as e:
                 logger.error(f"Failed data prep for H={h}: {e}. Skip.", exc_info=True); continue

            if not sd_paths or not sm: logger.warning(f"No usable data for H={h}. Skip."); continue

            result["feature_cols"][h] = final_feature_names
            feat_len = len(final_feature_names)
            logger.info(f"H{h}: Prepared {len(sd_paths)} symbols. Using {feat_len} features.")

            horizon_symbols = set(sd_paths.keys())
            stocks_A = [s for s in base_stocks_A if s in horizon_symbols]
            stocks_B = [s for s in base_stocks_B if s in horizon_symbols]

            if not stocks_A or not stocks_B: logger.warning(f"H{h}: Insufficient symbols in A/B splits. Skip."); continue

            result["stocks"][h] = {"A": stocks_A, "B": stocks_B}
            logger.info(f"H{h}: Valid symbols: {len(stocks_A)} A, {len(stocks_B)} B")

            result["models"][h] = {}; result["metrics"][h] = {"A": {}, "B": {}, "avg": {}}

            # Train/Eval Model A
            logger.info(f"--- H{h}: Training Model A (Train: B, Test: A) ---")
            model_A, metrics_A = _train_single_model(
                h, stocks_B, stocks_A, sd_paths, sm, final_feature_names, "Model A")
            
            # Modified to use _save_model
            if model_A:
                base_path = os.path.join(model_dir, f"h{h}_model_A.lgb")
                saved_path, size_mb = _save_model(model_A, base_path)
                result["models"][h]["A"] = model_A
                result["metrics"][h]["A"] = metrics_A
                logger.info(f"H{h} Model A saved to {saved_path} ({size_mb:.2f} MB)")
            else: 
                logger.error(f"H{h} Model A failed.")

            # Train/Eval Model B
            logger.info(f"--- H{h}: Training Model B (Train: A, Test: B) ---")
            mmap_cache.cache.clear(); gc.collect()
            model_B, metrics_B = _train_single_model(
                h, stocks_A, stocks_B, sd_paths, sm, final_feature_names, "Model B")
            
            # Modified to use _save_model
            if model_B:
                base_path = os.path.join(model_dir, f"h{h}_model_B.lgb")
                saved_path, size_mb = _save_model(model_B, base_path)
                result["models"][h]["B"] = model_B
                result["metrics"][h]["B"] = metrics_B
                logger.info(f"H{h} Model B saved to {saved_path} ({size_mb:.2f} MB)")
            else: 
                logger.error(f"H{h} Model B failed.")

            # Average Metrics
            metrics_A = result["metrics"][h].get("A", {})
            metrics_B = result["metrics"][h].get("B", {})
            valid_A = np.isfinite(metrics_A.get("rmse", np.nan))
            valid_B = np.isfinite(metrics_B.get("rmse", np.nan))
            avg_rmse, avg_r2 = np.nan, np.nan
            if valid_A and valid_B:
                avg_rmse = (metrics_A["rmse"] + metrics_B["rmse"]) / 2
                avg_r2 = (metrics_A["r2"] + metrics_B["r2"]) / 2
            elif valid_A: avg_rmse, avg_r2 = metrics_A["rmse"], metrics_A["r2"]
            elif valid_B: avg_rmse, avg_r2 = metrics_B["rmse"], metrics_B["r2"]
            result["metrics"][h]["avg"] = {"rmse": avg_rmse, "r2": avg_r2}
            logger.info(f"H{h}: Average Metrics - RMSE={avg_rmse:.4f}, R²={avg_r2:.4f}")

            mmap_cache.cache.clear(); gc.collect()
    finally:
        logger.info("Cleaning up temporary data preparation directory...")
        _clean_dir(temp_root)

    metadata = { "horizons": HORIZONS, "feature_cols_by_h": result["feature_cols"],
                 "metrics": result["metrics"], "stocks": result["stocks"], "timestamp": ts,
                 "compression": {"enabled": COMPRESS_MODELS, "level": COMPRESSION_LEVEL} }  # Add compression info
    try:
        joblib.dump(metadata, os.path.join(model_dir, "meta.joblib"))
        logger.info(f"Training complete. Models/metadata saved in {model_dir}")
    except Exception as e: logger.error(f"Failed to save metadata: {e}")

    return result

# ────────────────────────────────────────────────────────────────
# Find latest model (remains the same)
# ────────────────────────────────────────────────────────────────
def find_latest_model() -> str:
    """Find the path to the most recently created model directory."""
    _ensure_dir(MODEL_DIR)
    model_dirs = [ os.path.join(MODEL_DIR, d) for d in os.listdir(MODEL_DIR)
                   if os.path.isdir(os.path.join(MODEL_DIR, d)) and d.startswith("surface_") ]
    if not model_dirs: raise FileNotFoundError(f"No model directories matching 'surface_*' found in {MODEL_DIR}")
    try: return max(model_dirs, key=os.path.getmtime)
    except Exception as e: raise IOError(f"Could not determine latest model directory in {MODEL_DIR}: {e}")

# ────────────────────────────────────────────────────────────────
# Load models (updated to handle compression)
# ────────────────────────────────────────────────────────────────
def load_surface_models(model_dir: str = None) -> Dict[str, Any]:
    """
    Load pretrained surface models from disk using LightGBM's loader.
    Handles both compressed (.lgb.z) and uncompressed (.lgb) formats.

    Args:
        model_dir: Specific model directory path. Loads latest if None.

    Returns:
        Dictionary with models (Boosters) and metadata.
    """
    if model_dir is None: model_dir = find_latest_model()
    logger.info(f"Loading models from {model_dir}")
    
    if not os.path.isdir(model_dir): 
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    metadata_path = os.path.join(model_dir, "meta.joblib")
    if not os.path.exists(metadata_path): 
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    try: 
        metadata = joblib.load(metadata_path)
    except Exception as e: 
        raise IOError(f"Failed to load metadata file {metadata_path}: {e}")

    horizons = metadata.get("horizons", HORIZONS)
    result = { "models": {}, "metrics": metadata.get("metrics", {}),
               "feature_cols": metadata.get("feature_cols_by_h", {}),
               "stocks": metadata.get("stocks", {}), "model_dir": model_dir,
               "timestamp": metadata.get("timestamp", "unknown"), "horizons": horizons }

    for h in horizons:
        result["models"][h] = {}
        files_found = False
        
        for key in ["A", "B"]:
            # Try loading with the new helper function
            base_path = os.path.join(model_dir, f"h{h}_model_{key}.lgb")
            model = _load_model(base_path)
            
            if model:
                result["models"][h][key] = model
                files_found = True
                logger.debug(f"Loaded Model {key} for H={h}")
            else:
                logger.warning(f"Could not load model for H={h}, key={key}")
        
        if not files_found: 
            logger.warning(f"No model files loaded for H={h}")

    logger.info(f"Finished loading models for horizons: {list(result['models'].keys())}")
    return result

# ────────────────────────────────────────────────────────────────
# Predict surface (remains the same, uses features from loaded model)
# ────────────────────────────────────────────────────────────────
def predict_surface(
    model_dict: Dict[str, Any],
    df: pl.DataFrame,
    stock: str,
    trade_date: Union[str, _dt.date, _dt.datetime, np.datetime64],
    k_range: List[float] = K_RANGE,
    k_grid_points: int = K_GRID_POINTS
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Generate sigma(k,T) volatility surface for a given stock and trade date.
    Handles feature preparation including interactions defined during training.

    Args:
        model_dict: Dictionary from `load_surface_models`.
        df: DataFrame with *base* feature data for the stock.
        stock: Stock symbol.
        trade_date: Date for surface generation.
        k_grid_points: Number of points in the log-strike (k) grid.

    Returns:
        Tuple (K_mesh, T_mesh, Z) or None if prediction fails.
    """
    models_by_h = model_dict["models"]
    stocks_by_h = model_dict.get("stocks", {})
    feature_cols_by_h = model_dict.get("feature_cols", {})
    horizons = sorted(models_by_h.keys())

    if not horizons: logger.error("Predict: No valid models/horizons found"); return None

    td = _to_date(trade_date)
    today_df = df.filter( (pl.col("act_symbol") == stock) & (pl.col("date").cast(pl.Date) == td) ).head(1)
    if today_df.is_empty(): logger.error(f"Predict: No data found for '{stock}' on {td}"); return None

    k_grid = np.linspace(k_range[0], k_range[1], k_grid_points)
    n_horizons = len(horizons); n_strikes = len(k_grid)

    try: S0 = today_df.select("close").item()
    except Exception as e: logger.error(f"Predict: Failed to get 'close' price for {stock} on {td}: {e}"); return None

    Z = np.full((n_horizons, n_strikes), np.nan); T_cal_list = []

    for h_idx, h in enumerate(horizons):
        Tcal = int(round(h * 7/5)); T_cal_list.append(Tcal)

        model_key = None # Determine A or B model
        if h in stocks_by_h:
            if stock in stocks_by_h[h].get("A", []): model_key = "B"
            elif stock in stocks_by_h[h].get("B", []): model_key = "A"
        if model_key is None or model_key not in models_by_h.get(h, {}) or models_by_h[h][model_key] is None:
             valid_keys = [k for k, m in models_by_h.get(h, {}).items() if m is not None]
             if not valid_keys: logger.warning(f"Predict: No valid model for H={h}. Skip."); continue
             model_key = valid_keys[0]; logger.warning(f"Predict: Unknown group/model for {stock}/H={h}. Fallback: {model_key}.")

        model = models_by_h[h][model_key]
        feature_names = feature_cols_by_h.get(h) # Full list model expects
        if not feature_names: logger.warning(f"Predict: No feature names for H={h}. Skip."); continue
        feat_len = len(feature_names); logger.debug(f"Predict: Using Model {model_key} for H={h} ({feat_len} features).")

        X_grid = np.zeros((n_strikes, feat_len), dtype=np.float32)
        try: feature_indices = {name: i for i, name in enumerate(feature_names)}
        except Exception as e: logger.error(f"Predict: Error creating feature index map H={h}: {e}"); continue

        # Identify base features needed from input df
        base_feature_names_needed = [ f for f in feature_names if f != 'k' and not f.startswith('k_x_') and f != 'k_squared' ]
        interaction_base_cols = ["rv_35d", "rv_10d", "vix", "vvix", "ovx", "ATR_35", "intraday_range_var"]
        cols_to_extract_from_df = sorted(list(set(base_feature_names_needed + interaction_base_cols)))

        try: # Extract only needed base features from input df
            base_values_dict = today_df.select(cols_to_extract_from_df).row(0, named=True)
        except Exception as e: logger.error(f"Predict: Failed to extract base features for {stock}/{td} (H={h}): {e}"); continue

        # --- Fill Prediction Grid ---
        k_idx = feature_indices.get('k')
        if k_idx is None: logger.error(f"Predict: 'k' not in feature map H={h}. Skip."); continue

        # 1. Fill base feature columns
        for name in base_feature_names_needed:
            if (idx := feature_indices.get(name)) is not None:
                 base_val = base_values_dict.get(name, 0.0)
                 X_grid[:, idx] = np.nan_to_num(base_val, nan=0.0, posinf=1e6, neginf=-1e6)

        # 2. Fill 'k' column
        X_grid[:, k_idx] = k_grid
        k_column_data = X_grid[:, k_idx]

        # 3. Calculate and fill interaction columns
        base_values_for_interactions = { # Extract needed interaction bases once
            b_name: np.nan_to_num(base_values_dict.get(b_name, 0.0), nan=0.0, posinf=1e6, neginf=-1e6)
            for b_name in interaction_base_cols if b_name in base_values_dict }

        interaction_calcs = { # Define calculations
            "k_squared": lambda k, b: k**2, "k_x_rv35d": lambda k, b: k * b.get("rv_35d",0),
            "k_x_rv10d": lambda k, b: k * b.get("rv_10d",0), "k_x_vix": lambda k, b: k * b.get("vix",0),
            "k_x_vvix": lambda k, b: k * b.get("vvix",0), "k_x_ovx": lambda k, b: k * b.get("ovx",0),
            "k_x_ATR35": lambda k, b: k * b.get("ATR_35",0),
            "k_x_intraday_range_var": lambda k, b: k * b.get("intraday_range_var",0) }

        for name, calc_func in interaction_calcs.items():
             if (idx := feature_indices.get(name)) is not None: # If model uses this interaction
                 try:
                     col_data = calc_func(k_column_data, base_values_for_interactions)
                     X_grid[:, idx] = np.nan_to_num(col_data, nan=0.0, posinf=1e6, neginf=-1e6)
                 except Exception as calc_err:
                     logger.error(f"Predict: Error calculating interaction '{name}' H={h}: {calc_err}. Col set to 0.")
                     X_grid[:, idx] = 0.0

        # --- Make Predictions ---
        try:
            sigma_hat = model.predict(X_grid, num_iteration=model.best_iteration)
            Z[h_idx, :] = np.maximum(0.0, sigma_hat)
        except Exception as e:
            logger.error(f"Predict: Prediction failed for H={h}, Model={model_key}: {e}")
            # Z[h_idx, :] remains np.nan

    # --- Finalize Meshes ---
    K_mesh = S0 * np.exp(k_grid)[None, :].repeat(n_horizons, axis=0)
    T_mesh = np.array(T_cal_list)[:, None].repeat(n_strikes, axis=1)

    if np.isnan(Z).all(): logger.error(f"Predict: Surface for {stock}/{td} is all NaN."); return None
    elif np.isnan(Z).any(): logger.warning(f"Predict: Surface for {stock}/{td} contains NaNs.")

    return K_mesh, T_mesh, Z