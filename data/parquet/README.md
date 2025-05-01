# Parquet Data Files

This folder contains the project's core datasets stored in the efficient Parquet format.

## Purpose

The original data, exported from Dolt databases, resulted in CSV files exceeding 7GB in total size, making them unsuitable for direct storage in the GitHub repository. We converted the key datasets to the Parquet format, which offers significant compression and optimized read performance. This conversion reduced the total size considerably, allowing the data to be included in the repository.

## Structure

The data is organized into subfolders corresponding to the original tables:

* **`ohlcv/`**: Contains daily Open, High, Low, Close, and Volume data for stocks.
* **`option_chain/`**: Contains daily options chain data, including strike prices, expiration dates, implied volatility, greeks, etc..
* **`split/`**: Contains stock split event data.

## Partitioning

To further optimize data loading and allow for efficient filtering by stock symbol, the data within the `ohlcv` and `option_chain` folders is partitioned. This means the data for each dataset is split across multiple files (e.g., `part_0.parquet`, `part_1.parquet`, ...), where each file contains data for a specific subset of stock symbols.

Each partitioned folder (`ohlcv/` and `option_chain/`) contains a `partition_metadata.json` file. This metadata file maps stock symbols to their corresponding partition file, allowing the `dataframe_loader.py` module in `src/data_extraction/` to load only the necessary files when data for specific symbols is requested, significantly speeding up data access.

## Loading Data

Use the `load_data` function in `src/data_extraction/dataframe_loader.py` to load data from these Parquet files into Polars DataFrames. This function automatically handles the partitioned structure based on the metadata files.
