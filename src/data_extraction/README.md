# Data Extraction

This folder contains modules responsible for extracting data from various sources and preparing it for further processing.

## Modules

* **`dataframe_loader.py`**: Provides functions to efficiently load financial data from partitioned Parquet files located in the `data/parquet` directory. It uses metadata files (`partition_metadata.json`) created by `parquet_converter.py` to optimize loading by selecting only relevant partitions based on requested stock symbols.
* **`dolt_csv_export.py`**: Exports tables from the configured Dolt databases (`stocks` and `options` located in `data/raw/`) into CSV format in the `csv` directory. **Note:** For most purposes, running this script is unnecessary as the required data is already available in the Parquet files within the `data/parquet` directory.
* **`parquet_converter.py`**: Converts the exported CSV files (specifically `option_chain.csv`, `ohlcv.csv`, and `split.csv`) into partitioned Parquet files stored in `data/parquet`. It partitions the data based on stock symbols to optimize loading speed. **Note:** For most purposes, running this script is unnecessary as the required Parquet files are already included in the repository and accessible via `dataframe_loader.py`.

## Usage

The primary module intended for general use is `dataframe_loader.py`, which allows loading specific datasets (OHLCV, options, splits) potentially filtered by stock symbols directly from the pre-generated Parquet files.
