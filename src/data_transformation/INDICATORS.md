# Technical Indicators Documentation

This documentation describes the technical indicators computed from OHLCV data. Each indicator is calculated per asset (grouped by `act_symbol`) using lazy operations for efficiency. The indicators are derived using rolling windows—primarily 10, 30, and 90 days—to capture trends, momentum, volatility, and volume dynamics over multiple time horizons.

---

## 1. Log Returns  
**Description**: The natural logarithm of the ratio between consecutive closing prices, capturing the continuously compounded percentage change. This measure stabilizes variance and is foundational for many volatility and risk metrics.  
**Input**: `close` prices.  
**Formula**: `ln(Close_t / Close_{t-1})`.  
**Output**: A float series that quantifies daily returns, setting the stage for further statistical analysis.

---

## 2. Trend Indicators: SMA, EMA, and Standard Deviation  
**Simple Moving Average (SMA)**  
- **Description**: The rolling mean of closing prices over a specified window (10, 30, or 90 days) that smooths out short-term fluctuations. It provides a clear view of the underlying trend by filtering out noise.  
- **Output**: A float series representing the trend component of the asset’s price.  

**Exponential Moving Average (EMA)**  
- **Description**: A weighted average that emphasizes more recent prices, making it more responsive to recent market movements. This helps in quickly identifying shifts in trends.  
- **Output**: A float series that reacts more rapidly to price changes.  

**Standard Deviation (STD)**  
- **Description**: The rolling standard deviation of closing prices over the same windows, indicating the dispersion of prices around the SMA/EMA. It acts as a proxy for volatility and is essential for risk assessment.  
- **Output**: A float series measuring price variability.  

Together, these metrics provide a robust view of both the trend and the volatility inherent in the price data.

---

## 3. Relative Strength Index (RSI)  
**Description**: A momentum oscillator that quantifies the speed and magnitude of recent price movements by comparing the average gains to average losses. Calculated over rolling windows of 10, 30, and 90 days, it helps identify overbought or oversold conditions.  
**Calculation Steps**:
- Compute the change (delta) between consecutive closing prices.
- Separate positive gains from negative losses.
- Calculate rolling averages for gains and losses.
- Apply the formula: `RSI = 100 - 100 / (1 + (avg_gain / avg_loss))`.  
**Input**: `close` prices.  
**Output**: A series bounded between 0 and 100, where higher values suggest potential overbought conditions and lower values indicate oversold levels, aiding in timing entry and exit points.

---

## 4. MACD of Historical Volatility  
**Description**: A variation of the traditional MACD, this indicator applies the concept to volatility rather than price. By using fast (12-day) and slow (26-day) rolling standard deviations of log returns, with a 9-day exponential moving average as the signal, it captures shifts in market volatility trends.  
**Components**:
- **Vol MACD Line**: The difference between the fast and slow volatility measures.
- **Signal Line**: The 9-day exponential moving average of the MACD line.
- **MACD Histogram**: The difference between the MACD line and the signal line.  
**Input**: `log_returns`.  
**Output**: Three series that reflect the momentum and direction of volatility trends, offering insights into potential regime shifts in market behavior.

---

## 5. Chaikin Money Flow (CMF)  
**Description**: An indicator that combines price and volume to assess buying and selling pressure over rolling windows of 10, 30, and 90 days. It helps determine whether volume is supporting price trends, adding context to price movements.  
**Calculation Steps**:
- Calculate the Money Flow Multiplier: `(close - low) / (high - low)`.
- Compute the rolling sum of the product of the multiplier and volume.
- Divide by the rolling sum of volume to derive the CMF.  
**Input**: `high`, `low`, `close`, `volume`.  
**Output**: A series ranging from -1 to 1, where positive values indicate buying pressure and negative values indicate selling pressure, useful for confirming trends.

---

## 6. Commodity Channel Index (CCI)  
**Description**: Measures the deviation of the typical price from its moving average, signaling potential overbought or oversold conditions. Calculated over windows of 10, 30, and 90 days, it provides early warnings of price reversals.  
**Calculation Steps**:
- Compute the Typical Price: `(high + low + close) / 3`.
- Determine the rolling mean of the typical price.
- Calculate the mean absolute deviation (MAD) over the window.
- Apply the formula: `CCI = (Typical Price - SMA) / (0.015 * MAD)`.  
**Input**: `high`, `low`, `close`.  
**Output**: A float series where extreme values (e.g., above 100 or below -100) may indicate a shift in market sentiment.

---

## 7. Average True Range (ATR)  
**Description**: A volatility measure that averages the true range over rolling windows of 10, 30, and 90 days. It reflects the degree of price movement regardless of direction and is widely used for setting stop-loss levels.  
**Calculation Steps**:
- Compute the True Range as the maximum of:
  - `high - low`
  - `|high - previous close|`
  - `|low - previous close|`
- Take the rolling mean of the True Range over the specified window.  
**Input**: `high`, `low`, `close`.  
**Output**: A float series representing market volatility, providing a benchmark for expected price movement.

---

## 8. Statistical Measures on Log Returns  
**Description**: A collection of statistical descriptors calculated on log returns over rolling windows of 10, 30, and 90 days. These metrics help summarize the distribution and detect deviations from normality.  
**Measures**:
- **Mean** and **Standard Deviation**: Assess the central tendency and dispersion.
- **Skewness**: Evaluates the asymmetry of the distribution.
- **Kurtosis**: Measures the "tailedness" (excess kurtosis calculated by subtracting 3).  
**Input**: `log_returns`.  
**Output**: Multiple series that detail the behavior of returns, assisting in risk management and model diagnostics.

---

## 9. Quantile Spreads  
**Description**: Indicators that measure the dispersion within the log returns distribution over rolling windows of 10, 30, and 90 days. They provide insight into the spread of extreme and interquartile values.  
**Variants**:
- **Extreme Spread**: The difference between the 95th and 5th percentiles.
- **IQR Spread**: The difference between the 75th and 25th percentiles.  
**Input**: `log_returns`.  
**Output**: Float series that capture the width of the distribution, offering additional perspectives on market uncertainty.

---

## 10. On-Balance Volume (OBV)  
**Description**: A volume-based indicator that accumulates volume based on the direction of price movement, thereby highlighting the strength of a trend.  
**Calculation Steps**:
- Compare the current closing price to the previous close.
- Add volume if the price increases; subtract volume if it decreases.
- Compute the cumulative sum for each asset.  
**Input**: `close`, `volume`.  
**Output**: A cumulative series that reflects buying and selling pressure, aiding in trend confirmation.

---

## 11. VWAP Deviation  
**Description**: Measures the deviation of the current closing price from the Volume Weighted Average Price (VWAP), indicating how far prices stray from the average weighted by volume. This can signal potential overbought or oversold conditions.  
**Calculation Steps**:
- Calculate VWAP as the cumulative sum of `( (high + low + close) / 3 * volume )` divided by cumulative volume.
- Compute the difference between the close and VWAP.  
**Input**: `high`, `low`, `close`, `volume`.  
**Output**: A float series that helps in identifying price strength relative to trading volume.

---

## 12. Volume Percentile  
**Description**: Determines the relative strength of current volume compared to a rolling median over windows of 10, 30, and 90 days. This indicator highlights periods of unusually high or low trading activity.  
**Calculation Steps**:
- Calculate the rolling median of volume.
- Flag instances where the current volume exceeds the median.
- Compute a rolling average of these flags and scale the result to a percentage.  
**Input**: `volume`.  
**Output**: A percentage series that reflects volume intensity, useful for detecting unusual trading patterns.

---

## 13. Donchian Range  
**Description**: Measures the range between the highest high and the lowest low over rolling windows of 10, 30, and 90 days. This indicator is often used to identify breakout opportunities and periods of consolidation.  
**Calculation Steps**:
- Determine the rolling maximum of high prices and the rolling minimum of low prices.
- Calculate the difference between these values.  
**Input**: `high`, `low`.  
**Output**: A float series representing the trading range, which can signal shifts in market conditions.

---

## 14. Parkinson Volatility  
**Description**: An estimator of volatility that uses the range between high and low prices, offering a more efficient measure compared to simple standard deviation. It is particularly sensitive to extreme price movements.  
**Calculation Steps**:
- Compute the logarithm of the ratio `high/low` and square it.
- Calculate the rolling mean of this squared log ratio.
- Normalize by dividing by `4 * log(2)` and then take the square root.  
**Input**: `high`, `low`.  
**Output**: A float series that provides a refined volatility estimate, useful for risk assessment.

---

## 15. Yang-Zhang Volatility (and Lagged Variants)  
**Description**: A comprehensive volatility estimator that incorporates both overnight and intraday price movements by combining log returns from open-to-close and close-to-open transitions with the overall log returns variance. In addition, lagged variants are provided—each computed YZ volatility is shifted by its respective window length (10, 30, or 90 days) to align the forecast horizon with the measurement period.  
**Calculation Steps**:
- Compute log returns for open-to-close and close-to-open transitions.
- Calculate the rolling variances for these components along with the variance of overall log returns.
- Combine these components using a weighting factor.
- Generate lagged series where, for example, the 10-day volatility is shifted by 10 days.  
**Input**: `open`, `close`, `log_returns`.  
**Output**: A set of float series providing robust estimates of volatility, with the lagged versions serving as aligned response variables for predictive modeling.

---
