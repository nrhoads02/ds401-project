# Pre-trained Models

This folder stores the pre-trained machine learning models used for volatility surface prediction in this project.

## Model Type

The models contained here are LightGBM (`lgb`) models trained to predict future realized volatility surfaces ($\sigma(k, T)$, where k is log-moneyness and T is time to maturity) based on historical stock data and technical indicators.

## Structure

Models are organized within subdirectories named using the convention `surface_YYYYMMDD_HHMMSS`, indicating the date and time the model training process was initiated. Typically, the subdirectory with the latest timestamp contains the most recent set of models intended for use.

Within each `surface_*` directory:

* **Model Files**: You will find model files named like `h[HORIZON]_model_[A/B].lgb.z`.
    * `[HORIZON]` refers to the prediction horizon in days (e.g., 10, 15, 20, 25, 30, 35).
    * `[A/B]` indicates the model was trained using an A/B stock splitting strategy for cross-validation (Model A was trained on stock set B and validated on set A, and vice-versa).
    * The `.lgb.z` extension signifies that the standard LightGBM model file (`.lgb`) has been compressed using zlib to reduce file size.
* **Metadata File**: A `meta.joblib` file contains essential metadata about the training run, including:
    * The exact list of features used for each horizon model.
    * The list of stock symbols included in the A and B splits.
    * Performance metrics (like RMSE, R²) for each model on its respective validation set.
    * Timestamp of the training run.

## Loading Models

The models and their associated metadata should be loaded using the `load_surface_models` function located in `src/data_modeling/surface_lgbm_modeling.py`. This function automatically handles finding the latest model directory (or loading a specific one if provided), reading the `meta.joblib` file, and loading the compressed `.lgb.z` model files into usable LightGBM Booster objects.