# Data Modeling

This folder contains modules for training volatility surface models, generating predictions, visualizing surfaces, and analyzing options chain data.

## Modules

* **`surface_lgbm_modeling.py`**: Trains LightGBM models to predict realized volatility surfaces ($\sigma(k, T)$). It trains separate models for different prediction horizons, uses an A/B stock split for cross-validation, incorporates feature interactions (especially with log-moneyness 'k'), weights samples based on 'k', and handles data streaming using memory-mapping. It also includes functionality to compress the trained model files (`.lgb`) using zlib (`.lgb.z`) and **load both compressed and uncompressed models**.
* **`compress_lgbm_models.py`**: A utility script to compress existing `.lgb` model files found in the model directory (`data/models/surface_lgbm`) into `.lgb.z` files using zlib. **Note:** For most purposes, running this script is unnecessary as the compressed model files are already included in the repository and `surface_lgbm_modeling.py` can load them directly.
* **`option_chain_modeling.py`**: Focuses on processing and visualizing implied volatility surfaces derived directly from options chain data. It includes functions to adjust option data for splits, generate the IV surface for a specific stock and date, plot the surface, compare it against the LGBM model's predicted RV surface, and calculate the pricing difference between the two.
* **`vol_surface_visualization.py`**: Provides functions to generate and visualize the predicted realized volatility surface from the trained LGBM models. It uses the `predict_surface` function from `surface_lgbm_modeling.py` and overlays actual realized volatility points (if available in the input data) onto the 3D plot for comparison.

## Usage

The primary workflow involves loading the pre-trained LGBM models using the `load_surface_models` function in `surface_lgbm_modeling.py`. Once loaded, these models can be used with `predict_surface` (also in `surface_lgbm_modeling.py`) to generate volatility predictions for specific stocks and dates. The resulting predictions can then be visualized using `vol_surface_visualization.py` or compared against market data using `option_chain_modeling.py`. Training models via `train_surface_models` is generally not required as models are pre-trained and included.
