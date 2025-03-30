#!/usr/bin/env python3
"""
DataFrame loader module for partitioned Parquet files.

This module provides functions to efficiently load financial data from
partitioned Parquet files created by the parquet_converter.py script.
"""

import json
import time
from pathlib import Path
from typing import Union, List, Optional, Dict, Set
import polars as pl


# Cache for metadata to avoid repeated disk reads
_metadata_cache = {}


def normalize_dataset_name(dataset: str) -> str:
    """
    Normalize dataset name to standard form.
    
    Args:
        dataset: Raw dataset name
        
    Returns:
        Normalized dataset name
    """
    dataset = dataset.lower()
    if dataset in ['option_chain', 'option', 'options_chain']:
        return 'options'
    return dataset


def get_dataset_dir(dataset: str) -> str:
    """
    Get the directory name for a dataset.
    
    Args:
        dataset: Normalized dataset name
        
    Returns:
        Directory name
    """
    dataset_dirs = {
        'ohlcv': 'ohlcv',
        'options': 'option_chain',
        'split': 'split'
    }
    
    if dataset not in dataset_dirs:
        raise ValueError(f"Dataset must be one of: {', '.join(dataset_dirs.keys())}")
    
    return dataset_dirs[dataset]


def load_metadata(dataset: str) -> Dict:
    """
    Load the partition metadata for a dataset.
    
    Args:
        dataset: Dataset name ('ohlcv', 'options', or 'split')
    
    Returns:
        Dictionary containing partition metadata
    """
    # Normalize dataset name
    dataset = normalize_dataset_name(dataset)
    
    # Check if metadata is already in cache
    if dataset in _metadata_cache:
        return _metadata_cache[dataset]
    
    # Get directory name
    dataset_dir = get_dataset_dir(dataset)
    
    metadata_path = Path(f"data/parquet/{dataset_dir}/partition_metadata.json")
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}. "
                              f"Make sure you've run parquet_converter.py for {dataset}.")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Cache the metadata
    _metadata_cache[dataset] = metadata
    
    return metadata


def find_partitions_for_symbols(metadata: Dict, symbols: Optional[Union[str, List[str]]] = None) -> Set[int]:
    """
    Find which partitions contain the requested symbols.
    
    Args:
        metadata: The partition metadata
        symbols: Single symbol, list of symbols, or None for all symbols
    
    Returns:
        Set of partition IDs that need to be loaded
    """
    if symbols is None:
        # Return all partition IDs if no specific symbols are requested
        return {int(k) for k in metadata["partitions"].keys()}
    
    # Convert single symbol to list
    if isinstance(symbols, str):
        symbols = [symbols]
    
    # Find partitions containing any of the requested symbols
    partitions_to_load = set()
    
    # Set of symbols not found in any partition
    missing_symbols = set(symbols)
    
    for partition_id, partition_symbols in metadata["partitions"].items():
        # Check if any of the requested symbols are in this partition
        for symbol in symbols:
            if symbol in partition_symbols:
                partitions_to_load.add(int(partition_id))
                missing_symbols.discard(symbol)  # Remove from missing set
    
    if missing_symbols:
        print(f"Warning: The following symbols were not found: {', '.join(missing_symbols)}")
    
    if not partitions_to_load:
        print(f"Warning: None of the requested symbols {symbols} found in the dataset")
    
    return partitions_to_load


def load_and_process_parquet(dataset: str, partition_ids: Set[int], symbols: Optional[Union[str, List[str]]] = None) -> pl.DataFrame:
    """
    Load and process Parquet files from specified partitions.
    
    Args:
        dataset: Dataset name ('ohlcv', 'options', or 'split')
        partition_ids: Set of partition IDs to load
        symbols: Optional filter for specific symbols
    
    Returns:
        Processed polars DataFrame
    """
    # Normalize dataset name
    dataset = normalize_dataset_name(dataset)
    
    # Get directory name
    dataset_dir = get_dataset_dir(dataset)
    
    dir_path = Path(f"data/parquet/{dataset_dir}")
    
    # Load and concatenate all required partitions
    dfs = []
    for partition_id in partition_ids:
        parquet_path = dir_path / f"part_{partition_id}.parquet"
        
        if not parquet_path.exists():
            print(f"Warning: Partition file not found: {parquet_path}")
            continue
        
        # Load the parquet file
        df = pl.read_parquet(parquet_path)
        dfs.append(df)
    
    if not dfs:
        raise ValueError(f"No data found for the requested symbols in {dataset}. "
                        f"Check if the symbols exist in the dataset.")
    
    # Concatenate all dataframes
    full_df = pl.concat(dfs)
    
    # Apply symbol filter if specified
    if symbols is not None:
        if isinstance(symbols, str):
            symbols = [symbols]
        full_df = full_df.filter(pl.col("act_symbol").is_in(symbols))
    
    # No need to convert date columns - Parquet already preserves type information
    return full_df


def load_data(dataset: str, symbols: Optional[Union[str, List[str]]] = None, verbose: bool = False) -> pl.DataFrame:
    """
    Load data from partitioned Parquet files.
    
    Args:
        dataset: Dataset name ('ohlcv', 'options', 'split')
        symbols: Optional single symbol or list of symbols to filter by
        verbose: If True, print timing and partition information
    
    Returns:
        Polars DataFrame with the requested data
    
    Examples:
        # Load all OHLCV data
        ohlcv_df = load_data('ohlcv')
        
        # Load OHLCV data for AAPL
        aapl_ohlcv = load_data('ohlcv', 'AAPL')
        
        # Load options data for multiple symbols
        options_df = load_data('options', ['AAPL', 'AMZN', 'MSFT'])
    """
    start_time = time.time()
    
    # Normalize dataset name
    dataset = normalize_dataset_name(dataset)
    
    try:
        # Load metadata
        metadata = load_metadata(dataset)
        
        # Find which partitions to load
        partitions_to_load = find_partitions_for_symbols(metadata, symbols)
        
        if verbose:
            print(f"Loading {dataset} data from {len(partitions_to_load)} partitions")
            if symbols:
                symbol_list = symbols if isinstance(symbols, list) else [symbols]
                print(f"Filtering for {len(symbol_list)} symbols")
        
        # Load and process the data
        df = load_and_process_parquet(dataset, partitions_to_load, symbols)
        
        if verbose:
            elapsed = time.time() - start_time
            print(f"Loaded {len(df)} rows in {elapsed:.2f} seconds")
        
        return df
    
    except Exception as e:
        print(f"Error loading data from {dataset}: {str(e)}")
        raise


def get_available_symbols(dataset: str) -> List[str]:
    """
    Get a list of all available symbols in a dataset.
    
    Args:
        dataset: Dataset name ('ohlcv', 'options', 'split')
    
    Returns:
        List of available symbols
    """
    metadata = load_metadata(dataset)
    
    # Collect all symbols across all partitions
    all_symbols = set()
    for symbols in metadata["partitions"].values():
        all_symbols.update(symbols)
    
    return sorted(list(all_symbols))


def get_dataset_stats(dataset: str) -> Dict:
    """
    Get statistics about a dataset.
    
    Args:
        dataset: Dataset name ('ohlcv', 'options', 'split')
    
    Returns:
        Dictionary with dataset statistics
    """
    metadata = load_metadata(dataset)
    
    # Calculate total size
    total_size_mb = sum(float(size) for size in metadata.get("sizes_mb", {}).values())
    
    # Count symbols
    all_symbols = set()
    for symbols in metadata["partitions"].values():
        all_symbols.update(symbols)
    
    return {
        "num_partitions": len(metadata["partitions"]),
        "num_symbols": len(all_symbols),
        "total_size_mb": total_size_mb,
        "avg_partition_size_mb": total_size_mb / max(1, len(metadata["partitions"]))
    }