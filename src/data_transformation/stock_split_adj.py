import polars as pl

def calculate_adjustment(row: dict) -> float:
    """
    Given a row (as a dict) containing:
      - 'ex_dates': list of split ex-dates (or None)
      - 'factors': list of split factors (or None)
      - 'date': the trading date
    Returns the cumulative adjustment factor for that row.
    """
    ex_dates = row.get("ex_dates")
    factors = row.get("factors")
    trade_date = row.get("date")
    
    if ex_dates is None or factors is None:
        return 1.0

    adjustment = 1.0
    for ex_date, factor in zip(ex_dates, factors):
        if ex_date > trade_date:
            adjustment *= factor
    return adjustment

def adjust_splits(ohlcv: pl.DataFrame) -> pl.DataFrame:
    """
    Adjusts stock price and volume data in the given ohlcv DataFrame based on splits.

    Parameters:
        ohlcv (pl.DataFrame): A DataFrame with stock data including 'act_symbol' and 'date' columns.
                              The 'date' column should already be parsed as a Date.

    Returns:
        pl.DataFrame: The adjusted ohlcv DataFrame with split-adjusted prices and volumes.
    """
    # Read the splits data and parse the ex_date
    splits = (
        pl.read_csv("data/raw/stocks/csv/split.csv")
          .with_columns(pl.col("ex_date").str.to_date("%Y-%m-%d"))
    )
    
    # Compute the adjustment factor for each split row
    splits = splits.with_columns(
        pl.when((pl.col("to_factor") == 0) | (pl.col("for_factor") == 0))
          .then(1.0)
          .otherwise(pl.col("to_factor") / pl.col("for_factor"))
          .alias("factor")
    )
    
    # Group splits by stock symbol and aggregate the ex_dates and factors
    splits_grouped = splits.group_by("act_symbol").agg(
        pl.col("ex_date").sort().alias("ex_dates"),
        pl.col("factor").sort_by("ex_date").alias("factors")
    )
    
    # Join the splits data onto the ohlcv data based on the stock symbol.
    ohlcv = ohlcv.join(splits_grouped, on="act_symbol", how="left")

    # Convert the ohlcv DataFrame to a list of dictionaries to calculate adjustments row by row.
    row_dicts = ohlcv.to_dicts()
    adjustment_factors = [calculate_adjustment(row) for row in row_dicts]
    ohlcv = ohlcv.with_columns(pl.Series("adjustment_factor", adjustment_factors))
    
    # Adjust the prices and volumes.
    ohlcv = ohlcv.with_columns([
        (pl.col("open")  / pl.col("adjustment_factor")).round(2).alias("open"),
        (pl.col("high")  / pl.col("adjustment_factor")).round(2).alias("high"),
        (pl.col("low")   / pl.col("adjustment_factor")).round(2).alias("low"),
        (pl.col("close") / pl.col("adjustment_factor")).round(2).alias("close"),
        (pl.col("volume") * pl.col("adjustment_factor")).round(0).cast(pl.Int64).alias("volume"),
    ])
    
    # Drop temporary columns used for the adjustment.
    ohlcv = ohlcv.drop(["ex_dates", "factors", "adjustment_factor"])
    
    return ohlcv

if __name__ == "__main__":
    # Read the stocks OHLCV CSV and convert the 'date' column to Date type.
    ohlcv = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
    ohlcv = ohlcv.with_columns(
        pl.col("date").str.to_date("%Y-%m-%d")
    )
    
    # Adjust the splits using the function.
    ohlcv_adjusted = adjust_splits(ohlcv)
    
    # Print the results
    print(ohlcv_adjusted)
