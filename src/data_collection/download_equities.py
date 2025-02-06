"""
download_equities.py

Module for downloading historical equities data from Alpaca’s v2 stocks bars endpoint.
This module handles pagination and writes the downloaded bars data to a CSV file.
"""

import requests
import csv
import os
import pandas as pd
import time

# Alpaca API credentials (for paper trading)
API_KEY = "PKC4KCAOY052W6OLERUO"
API_SECRET = "MYkFU3D3xlYNKaxgL80kOuC9MWo9OvEcJHZQX0m4"

HEADERS = {
    "accept": "application/json",
    "APCA-API-KEY-ID": API_KEY,
    "APCA-API-SECRET-KEY": API_SECRET
}


def download_all_stock_bars(symbols, timeframe, start, end, output_file):
    """
    Downloads historical stock bars for the given list of symbols from Alpaca's v2 stocks bars endpoint,
    handling pagination if the data exceeds the per-request limit, and writes the results to a CSV file.

    Parameters:
        symbols (list of str): List of stock symbols, e.g. ["AAPL", "MSFT", "GOOGL"].
        timeframe (str): Bar timeframe (e.g., "1T" for one-minute bars, "1D" for daily bars).
        start (str): ISO8601 datetime string for the start of the interval (e.g., "2023-06-01T09:30:00-04:00").
        end (str): ISO8601 datetime string for the end of the interval (e.g., "2023-06-01T16:00:00-04:00").
        output_file (str): Path to the CSV file where the data will be saved.
    """
    base_url = "https://data.alpaca.markets/v2/stocks/bars"
    all_bars = {}  

    params = {
        "symbols": ",".join(symbols),
        "timeframe": timeframe,
        "start": start,
        "end": end,
        "adjustment": "raw",
        "sort": "asc",
        "limit": 10000  
    }

    next_page_token = None
    page = 1

    while True:
        if next_page_token:
            params["page_token"] = next_page_token
        else:
            params.pop("page_token", None)

        print(f"Requesting page {page}.")
        response = requests.get(base_url, headers=HEADERS, params=params)

        if response.status_code != 200:
            print("Error downloading stock bars:", response.status_code, response.text)
            break

        data = response.json()
        bars_dict = data.get("bars", {})
        for symbol, bar_list in bars_dict.items():
            if symbol not in all_bars:
                all_bars[symbol] = []
            all_bars[symbol].extend(bar_list)

        next_page_token = data.get("next_page_token")
        if not next_page_token:
            print("No more pages to fetch.")
            break
        page += 1

        time.sleep(0.1)

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

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


if __name__ == "__main__":
    sp500_tickers = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")[0]["Symbol"].tolist()
    tickers = sp500_tickers
    timeframe = "5T" 
    start_date = "2024-03-01T09:30:00-04:00" 
    end_date = "2025-02-01T16:00:00-04:00" 
    output_csv = os.path.join("data", "raw", "stocks", "bars.csv")

    # Download bars data for all S&P 500 stocks, 5-minute bars, from March 1, 2024 to February 1, 2025.

    download_all_stock_bars(tickers, timeframe, start_date, end_date, output_csv)
