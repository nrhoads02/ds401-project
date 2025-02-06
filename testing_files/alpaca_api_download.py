import requests
import csv
import os

API_KEY = "PKC4KCAOY052W6OLERUO"
API_SECRET = "MYkFU3D3xlYNKaxgL80kOuC9MWo9OvEcJHZQX0m4"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}

def download_all_stock_bars(symbols, timeframe, start, end, output_file):
    """
    Download all historical stock bars from Alpaca's v2 stocks endpoint across multiple pages and save to CSV.
    
    Parameters:
      symbols (list): List of stock symbols (e.g., ["AAPL", "MSFT", "GOOGL"]).
      timeframe (str): Bar timeframe (e.g., "1T" for one-minute bars).
      start (str): ISO8601 datetime string for start.
      end (str): ISO8601 datetime string for end.
      output_file (str): File path for the output CSV.
    """
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    all_bars = {}  # Will be a dictionary keyed by symbol

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": "raw",
        "sort": "asc",
        "limit": 10000  # Maximum data points per request.
    }

    next_page_token = None
    page = 1

    while True:
        if next_page_token:
            params["page_token"] = next_page_token
        else:
            # Remove page_token from params if it exists from previous iterations.
            params.pop("page_token", None)

        print(f"Requesting page {page} for symbols: {symbols}")
        response = requests.get(base_url, headers=headers, params=params)
        if response.status_code != 200:
            print("Error downloading stock bars:", response.status_code, response.text)
            break

        data = response.json()
        bars_dict = data.get("bars", {})

        # Merge this page's data into all_bars
        for symbol, bar_list in bars_dict.items():
            if symbol not in all_bars:
                all_bars[symbol] = []
            all_bars[symbol].extend(bar_list)

        next_page_token = data.get("next_page_token")
        if not next_page_token:
            print("No more pages.")
            break
        page += 1

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Write all data to CSV
    with open(output_file, mode="w", newline="") as csv_file:
        fieldnames = ["symbol", "time", "open", "high", "low", "close", "volume"]
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for symbol, bar_list in all_bars.items():
            for bar in bar_list:
                writer.writerow({
                    "symbol": symbol,
                    "time": bar.get("t"),
                    "open": bar.get("o"),
                    "high": bar.get("h"),
                    "low": bar.get("l"),
                    "close": bar.get("c"),
                    "volume": bar.get("v")
                })

    print(f"All stock bars saved to {output_file}")

tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "JPM", "V", "UNH", "DIS"]
timeframe = "5T"
start_date = "2024-03-01T09:30:00-04:00"
end_date = "2025-02-01T16:00:00-04:00"
output_csv = "data/raw/stocks/bars_1T.csv"
download_all_stock_bars(tickers, timeframe, start_date, end_date, output_csv)
