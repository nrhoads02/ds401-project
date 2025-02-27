# Technical Indicators Documentation

This documentation describes the technical indicators computed from OHLCV data. Each indicator is calculated per asset (grouped by `act_symbol`) using lazy operations for efficiency.

---

## 1. Log Returns  
**Description**: Natural logarithm of the ratio between consecutive closing prices.  
**Input**: `close` prices.  
**Formula**: `ln(Close_t / Close_{t-1})`.  
**Output**: Float series (captures percentage change).

---

## 2. Simple Moving Averages (SMA) & Exponential Moving Averages (EMA)  
**SMA**  
- **Description**: Rolling mean of closing prices over specified windows.  
- **Variants**: SMA calculated for windows of 3, 5, 10, 20, and 50 days.  

**EMA**  
- **Description**: Exponentially weighted average prioritizing recent prices.  
- **Variants**: EMA computed for spans of 3, 5, 10, 20, and 50 days.  

**Input**: `close` prices.  
**Output**: Smoothed trend lines (float series).

---

## 3. Relative Strength Index (RSI)  
**Description**: Momentum oscillator that measures the speed and change of price movements.  
**Calculation Steps**:
- Compute price change (`delta`).
- Separate gains and losses.
- Calculate rolling averages over windows of 7, 9, 14, 21, and 30 days.
- Compute RSI using the formula: `RSI = 100 - 100 / (1 + (avg_gain / avg_loss))`.  
**Input**: `close` prices.  
**Output**: Series bounded between 0 and 100.

---

## 4. MACD (Moving Average Convergence Divergence)  
**Components**:
- **MACD Line**: Difference between the 12-day EMA and 26-day EMA.
- **Signal Line**: 9-day EMA of the MACD Line.
- **MACD Histogram**: Difference between the MACD Line and the Signal Line.  
**Input**: `close` prices.  
**Output**: Three series reflecting trend momentum.

---

## 5. Bollinger Bands  
**Description**: Volatility indicator consisting of:
- **Middle Band**: SMA over a window.
- **Standard Deviation (Std)**: Rolling standard deviation.
- **Width**: Computed as 4 times the standard deviation.  
**Variants**: Calculated for 20-day and 50-day windows.  
**Input**: `close` prices.  
**Output**: SMA, standard deviation, and width bands.

---

## 6. Chaikin Money Flow (CMF)  
**Description**: Combines price and volume to gauge buying/selling pressure.  
**Calculation Steps**:
- Compute Money Flow Multiplier: `(close - low) / (high - low)`.
- Multiply by `volume` and aggregate over rolling windows of 10, 20, 50, and 100 days.
- Divide by total volume over the same windows.  
**Input**: `high`, `low`, `close`, `volume`.  
**Output**: Series ranging from -1 to 1.

---

## 7. Commodity Channel Index (CCI)  
**Description**: Measures deviation of the typical price from its moving average, indicating potential overbought or oversold conditions.  
**Calculation Steps**:
- Compute Typical Price: `(high + low + close) / 3`.
- Calculate rolling mean and mean absolute deviation over windows of 14 and 20 days.
- Apply formula: `CCI = (Typical Price - SMA) / (0.015 * MAD)`.  
**Input**: `high`, `low`, `close`.  
**Output**: Float series indicating price deviation.

---

## 8. Average True Range (ATR)  
**Description**: Measures market volatility by averaging the true range over a given period.  
**Calculation Steps**:
- Compute True Range as the maximum of:
  - `high - low`
  - `|high - previous close|`
  - `|low - previous close|`
- Calculate rolling mean over windows of 14 and 20 days.  
**Input**: `high`, `low`, `close`.  
**Output**: Volatility measure (float series).

---

## 9. Statistical Measures on Log Returns  
For rolling windows of 20 and 50 days, the following statistics are computed:
- **Mean** and **Standard Deviation** of log returns.
- **Skewness**: Using the third moment of deviations.
- **Kurtosis**: Using the fourth moment (adjusted by subtracting 3 for excess kurtosis).  
**Input**: `log_returns`.  
**Output**: Statistical descriptors of the return distribution.

---

## 10. Quantile Spreads  
**Description**: Measures dispersion in the log returns distribution.  
**Variants**:
- **Extreme Spread**: Difference between the 95th and 5th quantiles.
- **IQR Spread**: Difference between the 75th and 25th quantiles.  
**Calculation**: Computed over windows of 20 and 50 days.  
**Input**: `log_returns`.  
**Output**: Spread values as indicators of distribution width.

---

## 11. On-Balance Volume (OBV)  
**Description**: Volume-based indicator that accumulates volume based on price movement direction.  
**Calculation Steps**:
- Compare current close with previous close.
- Add `volume` if price increased, subtract if decreased.
- Compute cumulative sum per asset.  
**Input**: `close`, `volume`.  
**Output**: Cumulative volume indicator (OBV).

---

## 12. VWAP Deviation  
**Description**: Measures the deviation of the current closing price from the Volume Weighted Average Price (VWAP).  
**Calculation Steps**:
- Compute VWAP as the cumulative sum of `(typical price * volume)` divided by cumulative volume.
- Calculate the difference between `close` and VWAP.  
**Input**: `high`, `low`, `close`, `volume`.  
**Output**: Deviation series indicating relative price strength.

---

## 13. Volume Percentile  
**Description**: Determines the percentile ranking of current volume relative to a rolling median.  
**Calculation Steps**:
- Compute the median volume over windows of 20 and 50 days.
- Flag periods where current volume exceeds the median.
- Calculate the rolling average of this flag (scaled to 100).  
**Input**: `volume`.  
**Output**: Percentage indicator of volume strength.

---

## 14. Donchian Range  
**Description**: Measures the range between the highest high and lowest low over a specified window.  
**Variants**: Calculated for 20-day and 50-day windows.  
**Input**: `high`, `low`.  
**Output**: Range value (float series) indicating market volatility.

---

## 15. Parkinson Volatility  
**Description**: Estimates volatility using the ratio of high to low prices.  
**Calculation Steps**:
- Compute the log of the ratio `high/low` squared.
- Average over windows of 21, 26, 30, and 50 days.
- Adjust by a normalization factor.  
**Input**: `high`, `low`.  
**Output**: Volatility estimate based on price extremes.

---

## 16. Yang-Zhang Volatility  
**Description**: Combines overnight and intraday volatility measures for a robust volatility estimate.  
**Calculation Steps**:
- Compute log returns for open-to-close, close-to-open.
- Calculate rolling variances for these components and for the overall log returns.
- Combine the components using a weighting factor.  
**Variants**: Calculated for windows of 21, 26, 30, and 50 days.  
**Input**: `open`, `close`, `log_returns`.  
**Output**: Comprehensive volatility estimate (float series).

---
