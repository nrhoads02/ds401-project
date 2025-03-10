"""
XGBoost paired modeling pipeline for volatility prediction.

This module provides functionality for training complementary XGBoost models 
where each stock is predicted by a model that didn't see that stock during training.
"""
import polars as pl
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import pickle
import os
import datetime

def save_models(model_results, model_dir="data/models/xgboost"):
    """
    Save the paired XGBoost models to a pickle file with timestamp.
    
    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_paired_volatility_models
    model_dir : str
        Directory to save the model
        
    Returns:
    --------
    str
        Path to the saved model file
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Create timestamp for filename
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create filename
    filename = f"xgboost_model_{timestamp}.pkl"
    filepath = os.path.join(model_dir, filename)
    
    # Save the model
    with open(filepath, "wb") as f:
        pickle.dump(model_results, f)
    
    print(f"Models saved to {filepath}")
    return filepath

def load_models(filepath=None, model_dir="data/models/xgboost"):
    """
    Load paired XGBoost models from a pickle file.
    If no filepath is provided, loads the most recent model.
    
    Parameters:
    -----------
    filepath : str, optional
        Path to the specific model file to load
    model_dir : str
        Directory where models are stored
        
    Returns:
    --------
    dict
        The loaded model results dictionary
    """
    # If no filepath provided, find the most recent model
    if filepath is None:
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory {model_dir} does not exist")
        
        model_files = [
            os.path.join(model_dir, f) for f in os.listdir(model_dir) 
            if f.startswith("xgboost_model_") and f.endswith(".pkl")
        ]
        
        if not model_files:
            raise FileNotFoundError(f"No model files found in {model_dir}")
        
        # Sort by modification time, newest first
        filepath = max(model_files, key=os.path.getmtime)
        print(f"Loading most recent model: {filepath}")
    
    # Load and return the model
    with open(filepath, "rb") as f:
        model_results = pickle.load(f)
    
    print(f"Model loaded successfully from {filepath}")
    return model_results

def train_paired_volatility_models(
    df: pl.DataFrame, 
    target_col: str = 'YZVol_30_future',
    train_ratio: float = 0.5
):
    """
    Train two complementary XGBoost models where each stock is predicted by a model 
    that was trained on a different set of stocks.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame with stock data including technical indicators
    target_col : str
        Target column to predict (e.g., 'YZVol_30_future')
    train_ratio : float
        Portion of stocks to use for training each model (0.0-1.0)
        
    Returns:
    --------
    dict
        Dictionary with models, evaluation metrics, and feature importance
    """
    print(f"Training paired XGBoost models to predict {target_col}")
    start_time = time.time()
    
    # Clone data to avoid modifying the original
    data = df.clone()
    
    # More aggressive cleaning of the target column
    print(f"Cleaning target column: {target_col}")
    before_count = data.height
    
    # Check data statistics before filtering
    _target_stats = data.select(target_col).describe()
    print(f"Target column stats before cleaning:")
    print(_target_stats)
    
    # Filter out problematic values
    data = data.filter(
        ~pl.col(target_col).is_null() &  # Remove nulls
        ~pl.col(target_col).is_nan() &   # Remove NaN values
        ~pl.col(target_col).is_infinite() &  # Remove infinity
        (pl.col(target_col) < 100.0)       # Remove unreasonably large values
    )
    
    after_count = data.height
    print(f"Removed {before_count - after_count} rows with invalid target values")
    
    # Check data statistics after filtering
    _target_stats = data.select(target_col).describe()
    print(f"Target column stats after cleaning:")
    print(_target_stats)

    # Define patterns for columns to exclude as features
    exclude_patterns = [
        "date", "act_symbol", "_future"  # Exclude target-related columns
    ]
    
    # Select features (exclude specified patterns and target column)
    feature_cols = [
        col for col in data.columns 
        if not any(pattern in col for pattern in exclude_patterns) and col != target_col
    ]
    
    print(f"Using {len(feature_cols)} features for prediction")
    
    # Get all unique stocks and shuffle them
    stocks = data["act_symbol"].unique().to_list()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(stocks)
    
    # Split stocks into two groups (A and B)
    split_idx = int(len(stocks) * train_ratio)
    stocks_A = stocks[:split_idx]
    stocks_B = stocks[split_idx:]
    
    print(f"Stock-based split: {len(stocks_A)} stocks in group A, {len(stocks_B)} in group B")
    
    # Create the two complementary datasets
    # Model 1: Train on A, Test on B
    train_data_1 = data.filter(pl.col("act_symbol").is_in(stocks_A))
    test_data_1 = data.filter(pl.col("act_symbol").is_in(stocks_B))
    
    # Model 2: Train on B, Test on A
    train_data_2 = data.filter(pl.col("act_symbol").is_in(stocks_B))
    test_data_2 = data.filter(pl.col("act_symbol").is_in(stocks_A))
    
    print(f"Model 1 - Training: {train_data_1.height} rows, Testing: {test_data_1.height} rows")
    print(f"Model 2 - Training: {train_data_2.height} rows, Testing: {test_data_2.height} rows")
    
    # Define XGBoost parameters
    params = {
        'objective': 'reg:squarederror',
        'learning_rate': 0.05,
        'max_depth': 6,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1,
        'random_state': 42
    }
    
    # Train and evaluate Model 1 (Train on A, Test on B)
    print("\n=== Training Model 1 (Train on stocks A, Test on stocks B) ===")
    model_1_results = _train_and_evaluate_model(
        train_data_1, test_data_1, feature_cols, target_col, params
    )
    
    # Train and evaluate Model 2 (Train on B, Test on A)
    print("\n=== Training Model 2 (Train on stocks B, Test on stocks A) ===")
    model_2_results = _train_and_evaluate_model(
        train_data_2, test_data_2, feature_cols, target_col, params
    )
    
    # Compare feature importance between the two models
    print("\n=== Comparing feature importance between models ===")
    _compare_feature_importance(model_1_results['feature_importance'], 
                               model_2_results['feature_importance'])
    
    # Calculate overall metrics by combining predictions from both models
    y_test_combined = np.concatenate([
        model_1_results['test_predictions']['y_test'],
        model_2_results['test_predictions']['y_test']
    ])
    
    y_pred_combined = np.concatenate([
        model_1_results['test_predictions']['y_pred'],
        model_2_results['test_predictions']['y_pred']
    ])
    
    combined_rmse = np.sqrt(mean_squared_error(y_test_combined, y_pred_combined))
    combined_mae = mean_absolute_error(y_test_combined, y_pred_combined)
    combined_r2 = r2_score(y_test_combined, y_pred_combined)
    
    print("\n=== Combined Model Performance ===")
    print(f"RMSE: {combined_rmse:.6f}")
    print(f"MAE: {combined_mae:.6f}")
    print(f"R²: {combined_r2:.6f}")
    
    # Visualize combined results
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test_combined, y_pred_combined, alpha=0.3)
    plt.plot([min(y_test_combined), max(y_test_combined)], 
             [min(y_test_combined), max(y_test_combined)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Combined Model: Actual vs Predicted {target_col}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    total_time = time.time() - start_time
    print(f"\nTotal training and evaluation time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    
    return {
        'model_1': model_1_results['model'],
        'model_2': model_2_results['model'],
        'feature_cols': feature_cols,
        'stocks_A': stocks_A,
        'stocks_B': stocks_B,
        'metrics': {
            'model_1': model_1_results['metrics'],
            'model_2': model_2_results['metrics'],
            'combined': {
                'rmse': combined_rmse,
                'mae': combined_mae,
                'r2': combined_r2
            }
        },
        'feature_importance': {
            'model_1': model_1_results['feature_importance'],
            'model_2': model_2_results['feature_importance']
        },
        'test_predictions': {
            'y_test': y_test_combined,
            'y_pred': y_pred_combined
        }
    }

def _train_and_evaluate_model(train_data, test_data, feature_cols, target_col, params):
    """
    Helper function to train and evaluate a single model.
    """
    print("Preparing features and target for training...")
    
    # Handle each feature one by one to avoid duplicate column names
    train_features = train_data.select(feature_cols)
    for col in feature_cols:
        train_features = train_features.with_columns([
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        ])
    X_train = train_features.fill_null(0).to_numpy()
    
    # Ensure target values are clean
    y_train = train_data.select(target_col).to_numpy().ravel()
    
    # Same for test data
    test_features = test_data.select(feature_cols)
    for col in feature_cols:
        test_features = test_features.with_columns([
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        ])
    X_test = test_features.fill_null(0).to_numpy()
    
    y_test = test_data.select(target_col).to_numpy().ravel()
    
    print(f"Training shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Testing shape: X={X_test.shape}, y={y_test.shape}")
    
    # Train the model
    print("Training model...")
    start_time = time.time()
    
    try:
        # First try with newer XGBoost API
        model = xgb.XGBRegressor(**params, eval_metric='rmse')
        model.fit(
            X_train, y_train,
            verbose=100
        )
    except TypeError:
        try:
            # Try with older XGBoost API (no early stopping param)
            print("Using alternative XGBoost API...")
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train, verbose=100)
        except Exception as e:
            # Fallback to most basic approach
            print(f"Falling back to basic fit method: {str(e)}")
            model = xgb.XGBRegressor(**params)
            model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Make predictions
    print("Generating predictions...")
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Evaluation metrics:")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.6f}")
    
    # Feature importance
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 most important features:")
    for feature, imp in feature_importance[:10]:
        print(f"{feature}: {imp:.6f}")
    
    # Visualization: Scatter plot of actual vs predicted
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Actual vs Predicted {target_col}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    return {
        'model': model,
        'metrics': {
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        },
        'feature_importance': feature_importance,
        'test_predictions': {
            'y_test': y_test,
            'y_pred': y_pred
        }
    }

def _compare_feature_importance(importance_1, importance_2, top_n=15):
    """
    Compare feature importance between two models.
    """
    # Extract top features from both models
    top_features_1 = [feature for feature, _ in importance_1[:top_n]]
    top_features_2 = [feature for feature, _ in importance_2[:top_n]]
    
    # Find common and unique features
    common_features = set(top_features_1) & set(top_features_2)
    only_in_1 = set(top_features_1) - set(top_features_2)
    only_in_2 = set(top_features_2) - set(top_features_1)
    
    print(f"Common top features: {len(common_features)}")
    for feature in common_features:
        imp_1 = next(imp for feat, imp in importance_1 if feat == feature)
        imp_2 = next(imp for feat, imp in importance_2 if feat == feature)
        print(f"  {feature}: Model 1={imp_1:.6f}, Model 2={imp_2:.6f}")
    
    print(f"\nFeatures only in Model 1 top {top_n}: {len(only_in_1)}")
    for feature in only_in_1:
        imp = next(imp for feat, imp in importance_1 if feat == feature)
        print(f"  {feature}: {imp:.6f}")
    
    print(f"\nFeatures only in Model 2 top {top_n}: {len(only_in_2)}")
    for feature in only_in_2:
        imp = next(imp for feat, imp in importance_2 if feat == feature)
        print(f"  {feature}: {imp:.6f}")
    
    # Create bar chart comparing top features
    plt.figure(figsize=(12, 8))
    
    # Collect data for comparison
    all_top_features = list(set(top_features_1) | set(top_features_2))
    imp_model_1 = []
    imp_model_2 = []
    
    for feature in all_top_features:
        # Find importance in model 1 (0 if not in top features)
        imp_1 = next((imp for feat, imp in importance_1 if feat == feature), 0)
        imp_model_1.append(imp_1)
        
        # Find importance in model 2 (0 if not in top features)
        imp_2 = next((imp for feat, imp in importance_2 if feat == feature), 0)
        imp_model_2.append(imp_2)
    
    # Sort by average importance
    sorted_indices = np.argsort([(i1 + i2)/2 for i1, i2 in zip(imp_model_1, imp_model_2)])[::-1]
    all_top_features = [all_top_features[i] for i in sorted_indices]
    imp_model_1 = [imp_model_1[i] for i in sorted_indices]
    imp_model_2 = [imp_model_2[i] for i in sorted_indices]
    
    # Plot
    x = np.arange(len(all_top_features))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(14, 8))
    rects1 = ax.bar(x - width/2, imp_model_1, width, label='Model 1')
    rects2 = ax.bar(x + width/2, imp_model_2, width, label='Model 2')
    
    ax.set_ylabel('Feature Importance')
    ax.set_title('Feature Importance Comparison Between Models')
    ax.set_xticks(x)
    ax.set_xticklabels(all_top_features, rotation=45, ha='right')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def predict_with_paired_models(model_results, new_data):
    """
    Make predictions using the paired volatility models.
    
    Parameters:
    -----------
    model_results : dict
        Results dictionary from train_paired_volatility_models
    new_data : pl.DataFrame
        New data for prediction
        
    Returns:
    --------
    pl.DataFrame
        Original DataFrame with predictions added
    """
    model_1 = model_results['model_1']
    model_2 = model_results['model_2']
    feature_cols = model_results['feature_cols']
    stocks_A = model_results['stocks_A']
    stocks_B = model_results['stocks_B']
    
    # Ensure all required features are present
    missing_cols = [col for col in feature_cols if col not in new_data.columns]
    if missing_cols:
        raise ValueError(f"Missing features in new data: {missing_cols}")
    
    # Clone data to avoid modifying the original
    result_df = new_data.clone()
    
    # Split data by stock groups
    data_A = new_data.filter(pl.col("act_symbol").is_in(stocks_A))
    data_B = new_data.filter(pl.col("act_symbol").is_in(stocks_B))
    
    # Process features for group A
    features_A = data_A.select(feature_cols)
    for col in feature_cols:
        features_A = features_A.with_columns([
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        ])
    X_A = features_A.fill_null(0).to_numpy()
    
    # Process features for group B
    features_B = data_B.select(feature_cols)
    for col in feature_cols:
        features_B = features_B.with_columns([
            pl.when(pl.col(col).is_infinite())
            .then(None)
            .otherwise(pl.col(col))
            .alias(col)
        ])
    X_B = features_B.fill_null(0).to_numpy()
    
    # Make predictions with the appropriate model for each group
    # For group A stocks, use model 2 (trained on group B)
    # For group B stocks, use model 1 (trained on group A)
    pred_A = model_2.predict(X_A)
    pred_B = model_1.predict(X_B)
    
    # Add predictions back to original dataframes
    data_A_with_pred = data_A.with_columns(pl.lit(pred_A).alias("volatility_pred"))
    data_B_with_pred = data_B.with_columns(pl.lit(pred_B).alias("volatility_pred"))
    
    # Combine results
    result_df = pl.concat([data_A_with_pred, data_B_with_pred])
    
    return result_df

if __name__ == "__main__":
    # Example usage
    from src.data_transformation import transformation_pipeline
    
    # Load data
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv").with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # Apply transformations
    data = transformation_pipeline.transformation_pipeline(ohlcv)
    
    # Train paired models
    results = train_paired_volatility_models(
        df=data,
        target_col='YZVol_30_future',
        train_ratio=0.5
    )