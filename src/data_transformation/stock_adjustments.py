import polars as pl
from src.data_extraction.dataframe_loader import load_data

def remove_duplicates(split_df, max_days=150, verbose=False):
    """
    Removes duplicate stock splits in a Polars dataframe, keeping only the most recent occurrence.
    
    Args:
        split_df: Polars DataFrame with columns [act_symbol, ex_date, to_factor, for_factor]
        max_days: Maximum number of days between potential duplicates (default: 150)
        verbose: If True, prints information about removed duplicates (default: False)
        
    Returns:
        Polars DataFrame with duplicates removed
    """
    # First, find all the duplicate pairs
    df_a = split_df.select([
        pl.col("act_symbol"),
        pl.col("ex_date").alias("first_split_date"),
        pl.col("to_factor"),
        pl.col("for_factor")
    ])
    
    df_b = split_df.select([
        pl.col("act_symbol"),
        pl.col("ex_date").alias("second_split_date"),
        pl.col("to_factor").alias("b_to_factor"),
        pl.col("for_factor").alias("b_for_factor")
    ])
    
    duplicates = df_a.join(
        df_b,
        on="act_symbol",
        how="inner"
    ).filter(
        (pl.col("first_split_date") < pl.col("second_split_date")) &
        ((pl.col("second_split_date") - pl.col("first_split_date")).dt.total_days() <= max_days) &
        (pl.col("to_factor") == pl.col("b_to_factor")) &
        (pl.col("for_factor") == pl.col("b_for_factor"))
    ).select([
        pl.col("act_symbol"),
        pl.col("first_split_date"),
        pl.col("second_split_date"),
        pl.col("to_factor"),
        pl.col("for_factor")
    ])
    
    # If no duplicates found, return the original dataframe
    if len(duplicates) == 0:
        if verbose:
            print("No duplicates found.")
        return split_df
    
    # Get a list of rows to keep (only the latest of each duplicate)
    keep_list = []
    
    # For each unique symbol, find all duplicates and keep only the latest one
    for symbol in duplicates["act_symbol"].unique():
        # Get all split dates for this symbol
        symbol_data = split_df.filter(pl.col("act_symbol") == symbol)
        
        # Get the duplicate dates to be removed for this symbol
        symbol_duplicates = duplicates.filter(pl.col("act_symbol") == symbol)
        remove_dates = symbol_duplicates.select("first_split_date").to_series().to_list()
        
        # Keep records that aren't in the removal list
        symbol_cleaned = symbol_data.filter(~pl.col("ex_date").is_in(remove_dates))
        keep_list.append(symbol_cleaned)
        
    # Handle symbols without duplicates - keep all their records
    all_dupes_symbols = duplicates["act_symbol"].unique().to_list()
    non_dupes = split_df.filter(~pl.col("act_symbol").is_in(all_dupes_symbols))
    keep_list.append(non_dupes)
    
    # Combine all the records to keep
    cleaned_df = pl.concat(keep_list)
    
    if verbose:
        dupe_count = duplicates.group_by("act_symbol").agg(pl.count("first_split_date").alias("duplicates"))
        print(f"Found {len(duplicates)} duplicate pairs across {len(dupe_count)} symbols")
        print(dupe_count)
        
        removed_count = len(split_df) - len(cleaned_df)
        print(f"Removed {removed_count} duplicate records, {len(cleaned_df)} records remaining")
    
    return cleaned_df


def remove_incomplete_tickers(ohlcv: pl.DataFrame) -> pl.DataFrame:
    print("Filtering tickers present on first and last dates, with no nulls in-between...")

    # Get first and last dates - fixing the duplicate column name issue
    date_info = ohlcv.select(
        pl.min("date").alias("min_date"), 
        pl.max("date").alias("max_date")
    ).row(0)
    
    first_date, last_date = date_info[0], date_info[1]

    print(f"First date: {first_date}")
    print(f"Last date: {last_date}")

    # Get tickers present on first and last days
    first_day_tickers = (
        ohlcv.filter(pl.col("date") == first_date)
             .select("act_symbol")
             .unique()
    )

    last_day_tickers = (
        ohlcv.filter(pl.col("date") == last_date)
        .select("act_symbol")
        .unique()
    )

    # Tickers present on both first and last day
    continuous_tickers = (
        first_day_tickers.join(last_day_tickers, on="act_symbol", how="inner")
    ).select("act_symbol")

    # Filter to only continuous tickers
    ohlcv_continuous = ohlcv.filter(pl.col("act_symbol").is_in(continuous_tickers["act_symbol"]))

    # Identify tickers with nulls in the period
    tickers_with_nulls = (
        ohlcv_continuous
        .filter(pl.any_horizontal(pl.exclude(["date", "act_symbol"]).is_null()))
        .filter(pl.col("date").is_between(first_date, last_date))
        .select("act_symbol")
        .unique()
        .get_column("act_symbol")
        .to_list()
    )

    # Remove tickers with nulls
    filtered_df = ohlcv_continuous.filter(~pl.col("act_symbol").is_in(tickers_with_nulls))

    # Report
    original_tickers = ohlcv["act_symbol"].n_unique()
    final_tickers = continuous_tickers.filter(~pl.col("act_symbol").is_in(tickers_with_nulls)).height

    print(f"Removed {original_tickers - final_tickers} tickers due to null values")

    return filtered_df


def adjust_splits(ohlcv: pl.DataFrame, max_duplicate_days=150, verbose=False) -> pl.DataFrame:
    """
    Correctly adjusts OHLCV data for stock splits using reverse cumulative split factors.
    Uses the partitioned Parquet files. Also removes duplicate split records.
    
    Parameters:
        ohlcv (pl.DataFrame): DataFrame with columns: act_symbol, date, open, high, low, close, volume
        max_duplicate_days (int): Maximum number of days between duplicate splits (default: 150)
        verbose (bool): Whether to print verbose output (default: True)
        
    Returns:
        pl.DataFrame: Properly adjusted OHLCV data with correct pricing and volume
    """
    print("Adjusting OHLCV data for stock splits...")
    
    # Load the split data
    splits_raw = load_data("split")
    
    # Remove duplicate split records
    if verbose:
        print("Checking for and removing duplicate split records...")
    
    splits_cleaned = remove_duplicates(splits_raw, max_days=max_duplicate_days, verbose=verbose)
    
    # Process the cleaned splits data
    splits = (
        splits_cleaned
        .with_columns(
            pl.when((pl.col("to_factor") == 0) | (pl.col("for_factor") == 0))
              .then(1.0)
              .otherwise(pl.col("to_factor") / pl.col("for_factor"))
              .alias("split_factor")
        )
        # Shift the effective ex_date by 1 day so that the factor only applies after the split day.
        .with_columns(pl.col("ex_date").dt.offset_by("-1d").alias("ex_date"))
        .filter(pl.col("split_factor") != 1.0)
        .select(["act_symbol", "ex_date", "split_factor"])
    )

    # Calculate reverse cumulative product of split factors per symbol
    splits_processed = (
        splits.sort(["act_symbol", "ex_date"])
        .group_by("act_symbol", maintain_order=True)
        .agg(
            pl.col("ex_date"),
            pl.col("split_factor")
              .reverse()
              .cum_prod()
              .reverse()
              .alias("cumulative_factor")
        )
        .explode(["ex_date", "cumulative_factor"])
    )

    print("Split data processed and cumulative factors calculated...")

    # Join splits to OHLCV and calculate adjustments
    return (
        ohlcv.sort(["act_symbol", "date"])
        .join_asof(
            splits_processed.sort(["act_symbol", "ex_date"]),
            left_on="date",
            right_on="ex_date",
            by="act_symbol",
            strategy="forward"
        )
        .with_columns(
            pl.coalesce(pl.col("cumulative_factor"), pl.lit(1.0)).alias("adjustment_factor")
        )
        .with_columns(
            (pl.col(["open", "high", "low", "close"]) / pl.col("adjustment_factor")).round(2),
            (pl.col("volume") * pl.col("adjustment_factor")).round(0).cast(pl.Int64)
        )
        .drop(["ex_date", "cumulative_factor", "adjustment_factor"])
        .sort(["date", "act_symbol"])  # Maintain original date ordering
    )

if __name__ == "__main__":
    # Load OHLCV data from Parquet files
    ohlcv = load_data("ohlcv")
    
    # Process and adjust data
    filtered_ohlcv = remove_incomplete_tickers(ohlcv)
    adjusted_ohlcv = adjust_splits(filtered_ohlcv)
    
    print(f"Final dataset shape: {adjusted_ohlcv.shape}")
    print(f"Number of symbols: {adjusted_ohlcv['act_symbol'].n_unique()}")
    print(adjusted_ohlcv)