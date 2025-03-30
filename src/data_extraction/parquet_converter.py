#!/usr/bin/env python3
"""
High-performance CSV to Parquet converter with fixed-count partitioning
Optimized for speed with reasonable partition sizes
"""

import os
import time
import json
from pathlib import Path
import duckdb


def load_symbols(symbols_path):
    """Load symbols from symbols.txt into a set for fast lookups."""
    with open(symbols_path, 'r') as f:
        # Strip whitespace and remove empty lines
        symbols = {line.strip() for line in f if line.strip()}
    return symbols


def create_partitions(symbols, num_partitions):
    """
    Distribute symbols evenly across fixed number of partitions.
    Returns a dict mapping symbol to partition_id
    """
    # Sort symbols for deterministic partitioning
    sorted_symbols = sorted(list(symbols))
    
    # Calculate symbols per partition
    symbols_per_partition = len(sorted_symbols) // num_partitions
    remainder = len(sorted_symbols) % num_partitions
    
    # Distribute symbols to partitions
    symbol_to_partition = {}
    start_idx = 0
    
    for partition_id in range(num_partitions):
        # Add one extra symbol to early partitions if we can't divide evenly
        extra = 1 if partition_id < remainder else 0
        partition_size = symbols_per_partition + extra
        
        # Assign symbols to this partition
        for i in range(start_idx, start_idx + partition_size):
            if i < len(sorted_symbols):
                symbol_to_partition[sorted_symbols[i]] = partition_id
        
        start_idx += partition_size
    
    return symbol_to_partition


def convert_with_partitioning(csv_path, output_dir, symbol_to_partition, num_partitions):
    """
    Filter CSV by symbols and convert to multiple Parquet files based on partitioning.
    """
    # Create output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    
    conn = duckdb.connect(database=':memory:')
    
    # Create a temporary table for the partitioning map
    conn.execute("CREATE TABLE symbol_partitions(symbol VARCHAR, partition_id INTEGER)")
    
    # Insert partition mapping data in batches
    batch_size = 1000
    items = list(symbol_to_partition.items())
    for i in range(0, len(items), batch_size):
        batch = items[i:i+batch_size]
        placeholders = ','.join(['(?, ?)'] * len(batch))
        flat_batch = [item for pair in batch for item in pair]  # Flatten [(sym, part), ...] to [sym, part, ...]
        conn.execute(f"INSERT INTO symbol_partitions VALUES {placeholders}", flat_batch)
    
    print(f"Processing {csv_path}...")
    file_size_gb = os.path.getsize(str(csv_path)) / (1024 * 1024 * 1024)
    print(f"File size: {file_size_gb:.2f} GB")
    
    # Create metadata to track which symbols are in which partition
    metadata = {str(i): [] for i in range(num_partitions)}
    
    # Process each partition
    partition_sizes = {}
    original_size = os.path.getsize(csv_path) / (1024 * 1024 * 1024)  # Size in GB
    total_new_size = 0
    
    for partition_id in range(num_partitions):
        start_time = time.time()
        parquet_path = output_dir / f"part_{partition_id}.parquet"
        
        # Execute query to filter for symbols in this partition and write to Parquet
        conn.execute(f"""
            COPY (
                SELECT * FROM read_csv_auto(
                    '{csv_path}', 
                    all_varchar=false
                )
                WHERE act_symbol IN (
                    SELECT symbol FROM symbol_partitions 
                    WHERE partition_id = {partition_id}
                )
            ) TO '{parquet_path}' (
                FORMAT PARQUET, 
                COMPRESSION 'zstd'
            )
        """)
        
        elapsed = time.time() - start_time
        
        # Skip empty partitions
        if os.path.exists(parquet_path) and os.path.getsize(parquet_path) > 0:
            size_mb = os.path.getsize(str(parquet_path)) / (1024 * 1024)
            size_gb = size_mb / 1024
            total_new_size += size_gb
            partition_sizes[partition_id] = size_mb
            
            # Add symbols in this partition to metadata
            for symbol, part_id in symbol_to_partition.items():
                if part_id == partition_id:
                    metadata[str(partition_id)].append(symbol)
            
            print(f"Partition {partition_id}: {size_mb:.2f} MB, completed in {elapsed:.2f} seconds")
        else:
            print(f"Partition {partition_id}: Empty or skipped, completed in {elapsed:.2f} seconds")
    
    # Save metadata to JSON file for future reference
    metadata_path = output_dir / "partition_metadata.json"
    metadata_with_sizes = {
        "partitions": metadata,
        "sizes_mb": {str(k): v for k, v in partition_sizes.items()}
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_with_sizes, f, indent=2)
    
    print(f"Metadata saved to {metadata_path}")
    
    # Calculate overall compression ratio
    if total_new_size > 0:
        ratio = original_size / total_new_size
        print(f"Compression: {original_size:.2f} GB → {total_new_size:.2f} GB (ratio: {ratio:.2f}x)")
    
    return original_size, total_new_size


def main():
    # Define paths based on the folder structure
    symbols_path = Path("data/processed/symbols.txt")
    parquet_base_dir = Path("data/parquet")
    
    # Ensure base directory exists
    parquet_base_dir.mkdir(exist_ok=True, parents=True)
    
    # Load symbols
    print("Loading symbols...")
    start_time = time.time()
    symbols = load_symbols(symbols_path)
    print(f"Loaded {len(symbols)} symbols in {time.time() - start_time:.2f} seconds")
    
    # Define input files with fixed partition counts
    file_configs = [
        {
            "csv": Path("data/raw/options/csv/option_chain.csv"),
            "output_dir": parquet_base_dir / "option_chain",
            "num_partitions": 15  # Fixed at 15 partitions as proven to work
        },
        {
            "csv": Path("data/raw/stocks/csv/ohlcv.csv"),
            "output_dir": parquet_base_dir / "ohlcv",
            "num_partitions": 2   # Only need 2 as suggested
        },
        {
            "csv": Path("data/raw/stocks/csv/split.csv"),
            "output_dir": parquet_base_dir / "split",
            "num_partitions": 1    # Only need 1 as suggested
        }
    ]
    
    # Process each file
    total_start = time.time()
    total_original_size = 0
    total_new_size = 0
    
    for config in file_configs:
        csv_path = config["csv"]
        output_dir = config["output_dir"]
        num_partitions = config["num_partitions"]
        
        print(f"\n{'=' * 50}")
        print(f"Processing {csv_path.name} into {num_partitions} partitions")
        print(f"{'=' * 50}")
        
        # Create symbol to partition mapping - simple and fast
        symbol_to_partition = create_partitions(symbols, num_partitions)
        
        # Convert file with partitioning
        try:
            orig_size, new_size = convert_with_partitioning(
                str(csv_path), 
                output_dir, 
                symbol_to_partition, 
                num_partitions
            )
            
            total_original_size += orig_size
            total_new_size += new_size
            
        except Exception as e:
            print(f"Error processing {csv_path}: {e}")
            import traceback
            traceback.print_exc()
    
    total_time = (time.time() - total_start)/60
    print(f"\nTotal processing time: {total_time:.2f} minutes")
    
    if total_original_size > 0 and total_new_size > 0:
        overall_ratio = total_original_size / total_new_size
        print(f"\nOverall compression ratio: {overall_ratio:.2f}x")
        print(f"Total original size: {total_original_size:.2f} GB")
        print(f"Total compressed size: {total_new_size:.2f} GB")
        print(f"Space saved: {total_original_size - total_new_size:.2f} GB")
        print(f"Average processing speed: {total_original_size/total_time:.2f} GB/minute")


if __name__ == "__main__":
    main()