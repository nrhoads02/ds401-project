# METADATA for Stocks Data

This document describes the variables in the Stocks database. It covers the following tables: `dividend`, `ohlcv`, `split`, and `symbol`.

---

## Table: dividend

| Variable   | Type         | Possible Values             | Description                                      |
|------------|--------------|-----------------------------|--------------------------------------------------|
| act_symbol | varchar(64)  | e.g. "AAPL", "GOOG"         | Ticker symbol of the stock                       |
| ex_date    | date         | Format: YYYY-MM-DD          | Ex-dividend date                                 |
| amount     | decimal(10,5)| Numeric value (e.g. 0.25000)  | Dividend amount per share                        |

---

## Table: ohlcv

| Variable   | Type         | Possible Values             | Description                                      |
|------------|--------------|-----------------------------|--------------------------------------------------|
| date       | date         | Format: YYYY-MM-DD          | Trading date                                     |
| act_symbol | varchar(64)  | e.g. "AAPL", "GOOG"         | Ticker symbol of the stock                       |
| open       | decimal(7,2) | Numeric price               | Opening price of the day                         |
| high       | decimal(7,2) | Numeric price               | Highest price during the trading day             |
| low        | decimal(7,2) | Numeric price               | Lowest price during the trading day              |
| close      | decimal(7,2) | Numeric price               | Closing price of the day                         |
| volume     | bigint       | Numeric value               | Trading volume (number of shares traded)         |

---

## Table: split

| Variable   | Type         | Possible Values                     | Description                                      |
|------------|--------------|-------------------------------------|--------------------------------------------------|
| act_symbol | varchar(64)  | e.g. "AAPL", "GOOG"                 | Ticker symbol of the stock                       |
| ex_date    | date         | Format: YYYY-MM-DD                  | Date when the stock split occurred               |
| to_factor  | decimal(10,5)| Numeric ratio (e.g. 2.00000)          | Numerator of the split ratio (new shares)         |
| for_factor | decimal(10,5)| Numeric ratio (e.g. 1.00000)          | Denominator of the split ratio (old shares)       |

---

## Table: symbol

| Variable         | Type         | Possible Values                         | Description                                      |
|------------------|--------------|-----------------------------------------|--------------------------------------------------|
| act_symbol       | varchar(64)  | e.g. "AAPL", "GOOG"                     | Ticker symbol of the stock                       |
| security_name    | text         | Any text                                | Full name of the security                        |
| listing_exchange | text         | e.g. "NASDAQ", "NYSE"                   | Exchange where the stock is listed               |
| market_category  | text         | e.g. "Q", "G", "S"                      | Market category classification                   |
| is_etf           | tinyint      | 0 or 1                                  | Indicator if the security is an ETF (1 = Yes)      |
| round_lot_size   | int          | Numeric value                           | Standard trading lot size                        |
| is_test_issue    | tinyint      | 0 or 1                                  | Indicator if the security is a test issue (1 = Yes)|
| financial_status | text         | e.g. "D", "N", "G"                      | Financial status code of the security            |
| cqs_symbol       | text         | Any text                                | Alternative (CQS) symbol, if applicable          |
| nasdaq_symbol    | text         | Any text                                | NASDAQ-specific symbol (if applicable)           |
| is_next_shares   | tinyint      | 0 or 1                                  | Indicator if next shares are available (1 = Yes)   |
| last_seen        | date         | Format: YYYY-MM-DD                      | Date when the security was last updated          |





# METADATA for Options Data

This document describes the variables in the Options database. It covers the following tables: `option_chain` and `volatility_history`.

---

## Table: option_chain

| Variable   | Type         | Possible Values                              | Description                                      |
|------------|--------------|----------------------------------------------|--------------------------------------------------|
| date       | date         | Format: YYYY-MM-DD                           | Trading date                                     |
| act_symbol | varchar(64)  | e.g. "AAPL", "GOOG"                          | Ticker symbol of the underlying stock            |
| expiration | date         | Format: YYYY-MM-DD                           | Option expiration date                           |
| strike     | decimal(7,2) | Numeric price                                | Strike price of the option                       |
| call_put   | varchar(64)  | "call" or "put"                              | Option type: "call" for call options, "put" for put options |
| bid        | decimal(7,2) | Numeric price                                | Bid price for the option                         |
| ask        | decimal(7,2) | Numeric price                                | Ask price for the option                         |
| vol        | decimal(5,4) | Numeric value                                | Trading volume or open interest (if applicable)  |
| delta      | decimal(5,4) | Numeric value                                | Delta of the option                              |
| gamma      | decimal(5,4) | Numeric value                                | Gamma of the option                              |
| theta      | decimal(5,4) | Numeric value                                | Theta of the option                              |
| vega       | decimal(5,4) | Numeric value                                | Vega of the option                               |
| rho        | decimal(5,4) | Numeric value                                | Rho of the option                                |

---

## Table: volatility_history

| Variable             | Type         | Possible Values                              | Description                                      |
|----------------------|--------------|----------------------------------------------|--------------------------------------------------|
| date                 | date         | Format: YYYY-MM-DD                           | Trading date                                     |
| act_symbol           | varchar(64)  | e.g. "AAPL", "GOOG"                          | Ticker symbol of the underlying stock            |
| hv_current           | decimal(5,4) | Numeric value                                | Current historical volatility                    |
| hv_week_ago          | decimal(5,4) | Numeric value                                | Historical volatility from one week ago          |
| hv_month_ago         | decimal(5,4) | Numeric value                                | Historical volatility from one month ago         |
| hv_year_high         | decimal(5,4) | Numeric value                                | Highest historical volatility in the past year   |
| hv_year_high_date    | date         | Format: YYYY-MM-DD                           | Date when the highest historical volatility was recorded |
| hv_year_low          | decimal(5,4) | Numeric value                                | Lowest historical volatility in the past year    |
| hv_year_low_date     | date         | Format: YYYY-MM-DD                           | Date when the lowest historical volatility was recorded |
| iv_current           | decimal(5,4) | Numeric value                                | Current implied volatility                       |
| iv_week_ago          | decimal(5,4) | Numeric value                                | Implied volatility from one week ago             |
| iv_month_ago         | decimal(5,4) | Numeric value                                | Implied volatility from one month ago            |
| iv_year_high         | decimal(5,4) | Numeric value                                | Highest implied volatility in the past year      |
| iv_year_high_date    | date         | Format: YYYY-MM-DD                           | Date when the highest implied volatility was recorded |
| iv_year_low          | decimal(5,4) | Numeric value                                | Lowest implied volatility in the past year       |
| iv_year_low_date     | date         | Format: YYYY-MM-DD                           | Date when the lowest implied volatility was recorded |
