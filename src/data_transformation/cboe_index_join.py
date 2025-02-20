import os
import glob
import polars as pl

def join_cboe_indices(stocks_df: pl.DataFrame, cboe_folder: str = "data/raw/cboe") -> pl.DataFrame:
    """
    Joins all cboe index CSVs (with _History.csv in the filename) to the provided stocks DataFrame.
    
    Parameters:
        stocks_df (pl.DataFrame): The stocks DataFrame with a "date" column.
        cboe_folder (str): Folder path where cboe CSV files reside.
        
    Returns:
        pl.DataFrame: The stocks DataFrame with additional index columns.
    """
    # Define indices that follow the DATE, OPEN, HIGH, LOW, CLOSE format.
    special_indices = {"VIX", "VIX9D", "VXAPL", "VXAZN", "VXEEM"}
    
    # Get list of all index history CSV files in the folder.
    index_files = glob.glob(os.path.join(cboe_folder, "*_History.csv"))
    
    for file_path in index_files:
        # Extract the index name from the file name (e.g., "VIX_History.csv" -> "VIX").
        base_name = os.path.basename(file_path)
        index_name = base_name.split("_")[0]
        index_lower = index_name.lower()  # Convert to lowercase for the column name.
        
        # Read the index CSV.
        df = pl.read_csv(file_path)
        
        # Parse the 'DATE' column to Date type using the format in these files (MM/DD/YYYY).
        df = df.with_columns(
            pl.col("DATE").str.strptime(pl.Date, "%m/%d/%Y")
        )
        
        if index_name in special_indices:
            # For indices with DATE, OPEN, HIGH, LOW, CLOSE, join on the closing price.
            df = df.rename({"DATE": "date", "CLOSE": index_lower})
            df = df.select(["date", index_lower])
        else:
            # For indices with a DATE column and a single data column.
            cols = df.columns
            if len(cols) < 2:
                raise ValueError(f"Expected at least 2 columns in {file_path}, got: {cols}")
            df = df.rename({cols[0]: "date", cols[1]: index_lower})
            df = df.select(["date", index_lower])
        
        # Left join the index data onto the stocks DataFrame on 'date'.
        stocks_df = stocks_df.join(df, on="date", how="left")
    
    # Optional: filter stocks_df for dates >= March 16, 2011.
    stocks_df = stocks_df.filter(pl.col("date") >= pl.date(2011, 3, 16))
    return stocks_df

if __name__ == "__main__":
    stocks_df = pl.read_csv("data/raw/stocks/csv/ohlcv.csv")
    
    stocks_df = stocks_df.with_columns(
        pl.col("date").str.strptime(pl.Date, "%Y-%m-%d")
    )
    
    stocks_df = join_cboe_indices(stocks_df, cboe_folder="data/raw/cboe")
    
    print(stocks_df)
