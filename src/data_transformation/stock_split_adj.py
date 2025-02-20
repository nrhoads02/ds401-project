import polars as pl

def adjust_splits(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """
    Correctly adjusts OHLCV data for stock splits using reverse cumulative split factors.
    
    Parameters:
        ohlcv (pl.DataFrame): DataFrame with columns: act_symbol, date, open, high, low, close, volume
        
    Returns:
        pl.DataFrame: Properly adjusted OHLCV data with correct pricing and volume
    """
    # Load and preprocess split data, shifting ex_date by one day
    splits = (
        pl.read_csv("data/raw/stocks/csv/split.csv")
        .with_columns(
            pl.col("ex_date").str.to_date("%Y-%m-%d"),
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
        .drop(["ex_date", "cumulative_factor", "adjustment_factor"])  # REMOVED split_factor FROM DROP LIST
        .sort(["date", "act_symbol"])  # Maintain original date ordering
    )


if __name__ == "__main__":
    # Load and process data
    ohlcv = (
        pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    )
    
    adjusted_ohlcv = adjust_splits(ohlcv)
    print(adjusted_ohlcv)
    