# Raw Data README

Our project's base data is stored in a dolt db. Cloning instructions are in the project's main README.md

## Stocks

This repository contains four tables:

1. symbol : This table contains stock ticker symbols, security names, listing exchange, and other data for US equities.

Contains symbol, security name, listing exchange, market category, and other assorted data about the ticker

2. ohlcv : This table contains daily open, high, low, and close prices and volumes for US equities.

Contains date, symbol, open, high, low, close, volume

3. split : This table contains stock split ratios

Contains symbol, date, to_factor, for_factor

4. dividend : This table contains past dividend payouts

Contains symbol, date, and amount

## Options

This repository contains two tables:

1. option_chain : This table contains bids, asks, vols, and greeks for US equity options.

Contains date, symbol, expiration, strike price, call/put, bid price, ask price, volatility, delta, gamma, theta, vega, and rho

2. volatility_history: This table contains both current, week ago, month ago, and year ago high/low implied vols and historical variance. This table can be useful for computing IV rank.

Contains date, symbol, historical variance across various time frames, and implied volatilities across various time frames.

## cboe

There are a few assorted CSV files in the ./cboe/ folder, which contain some CBOE volatility indices. These include VIX, VVIX, GVX, and a few others. Information on these can be found here:

<https://www.cboe.com/tradable_products/vix/vix_historical_data/>

## DOLT DB ACCESS

To access a dolt db in the command line, you can follow this process:

1. **Open the SQL Shell:**
   Navigate to your Dolt database directory and start the SQL shell:

   ```bash
   cd path/to/dolt/db
   dolt sql
   ```

2. **View a Table:**
   In the SQL shell, display table contents (e.g., first 10 rows):

   ```sql
   SELECT * FROM table_name LIMIT 10;
   ```
