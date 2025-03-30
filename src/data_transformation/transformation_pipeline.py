from src.data_transformation import stock_adjustments, technical_indicators, cboe_index_join
import polars as pl

def transformation_pipeline(df: pl.DataFrame, calculate_future_cols: bool = True) -> pl.DataFrame:
    """
    Transform stock data using Polars' streaming capabilities to minimize memory usage.
    Focuses on optimizing the two main bottlenecks: technical indicator generation and null dropping.
    
    Args:
        df: Input DataFrame with stock data
    
    Returns:
        Transformed DataFrame
    """
    print("Starting optimized streaming pipeline...")
    
    # Initial adjustments
    df = stock_adjustments.adjust_splits(df)
    df = stock_adjustments.remove_incomplete_tickers(df)
    
    # Technical indicators with streaming
    print("Adding technical indicators using streaming mode...")
    df = (
        df.lazy()
        .pipe(technical_indicators.add_technical_indicators, calculate_future_cols=calculate_future_cols)
        # Using streaming=True for this collection to process in chunks
        .collect(streaming=True)
    )
    print("Technical indicators added!")
    
    # CBOE index join
    print("Joining CBOE indices...")
    df = cboe_index_join.join_cboe_indices(df)
    
    # Null dropping with streaming
    print("Dropping nulls using streaming mode...")
    df = (
        df.lazy()
        .drop_nulls()
        # Using streaming=True again for the null dropping operation
        .collect(streaming=True)
    )
    print("Nulls dropped!")
    
    # The data should maintain its original sort order throughout the process
    # If needed, an explicit sort could be added here, but we're trying to avoid that
    
    return df

if __name__ == "__main__":
    df = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    
    df = transformation_pipeline(df)
    
    print(df)