from src.data_transformation import stock_adjustments, technical_indicators, cboe_index_join

import polars as pl

def transformation_pipeline(df: pl.DataFrame) -> pl.DataFrame:
    df = stock_adjustments.adjust_splits(df)
    df = stock_adjustments.remove_incomplete_tickers(df)

    df = (
        df.lazy()
        .pipe(technical_indicators.add_technical_indicators)
        .collect()
    )
    print("Technical indicators added!")

    df = cboe_index_join.join_cboe_indices(df)

    print("Dropping rows with null values...")
    df = df.drop_nulls()

    return df

if __name__ == "__main__":
    df = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")

    df = df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )

    df = transformation_pipeline(df)
    print(df)
