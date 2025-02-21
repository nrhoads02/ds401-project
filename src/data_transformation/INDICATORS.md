# Technical Indicators Documentation

## 1. Log Returns  
**Description**: Measures the natural logarithm of the ratio between consecutive closing prices.  
**Input**: `close` prices.  
**Formula**: `ln(Close_t / Close_{t-1})`.  
**Output**: Float series (positive/negative values).  

---

## 2. Simple Moving Averages (SMA)  
**Description**: Rolling mean of closing prices over specified windows.  
**Variants**:  
- `SMA_3` (3-day), `SMA_5` (5-day), `SMA_10` (10-day), `SMA_20` (20-day).  
**Input**: `close` prices.  
**Output**: Float series (smoothed trend lines).  

---

## 3. Exponential Moving Averages (EMA)  
**Description**: Weighted average prioritizing recent prices.  
**Variants**:  
- `EMA_3`, `EMA_5`, `EMA_10`, `EMA_20`.  
**Input**: `close` prices.  
**Parameters**: Smoothing factor (`2/(window + 1)`).  
**Output**: Float series (more responsive than SMA).  

---

## 4. Relative Strength Index (RSI)  
**Description**: Momentum oscillator (0–100) for overbought/oversold signals.  
**Variants**:  
- `RSI_7`, `RSI_9`, `RSI_14`, `RSI_21`, `RSI_30`.  
**Input**: `close` prices.  
**Parameters**: Lookback window (e.g., 14 days).  
**Output**: 0–100 series.  
  - **Thresholds**: >70 (overbought), <30 (oversold).  

---

## 5. MACD (Moving Average Convergence Divergence)  
**Components**:  
- `MACD`: 12-day EMA − 26-day EMA.  
- `Signal_Line`: 9-day EMA of MACD.  
- `MACD_Histogram`: MACD − Signal_Line.  
**Input**: `close` prices.  
**Output**: Three series (bullish/bearish momentum).  

---

## 6. Bollinger Bands  
**Components**:  
- Middle Band (SMA), Upper Band (SMA + 2σ), Lower Band (SMA − 2σ).  
**Variants**:  
- 20-day (`BB_Middle_20`, `BB_Upper_20`, `BB_Lower_20`).  
- 50-day (`BB_Middle_50`, `BB_Upper_50`, `BB_Lower_50`).  
**Input**: `close` prices.  
**Parameters**: Window (20/50 days), multiplier (2.0).  
**Output**: Three bands (volatility indicator).  

---

## 7. Stochastic Oscillator  
**Components**:  
- `%K`: Current close vs. recent high-low range.  
- `%D`: 3-day SMA of %K.  
**Variants**:  
- 9-day (`Stochastic_%K_9`, `Stochastic_%D_9`).  
- 14-day (`Stochastic_%K_14`, `Stochastic_%D_14`).  
- 21-day (`Stochastic_%K_21`, `Stochastic_%D_21`).  
**Input**: `high`, `low`, `close`.  
**Output**: 0–100 series.  
  - **Thresholds**: >80 (overbought), <20 (oversold).  

---

## 8. Chaikin Money Flow (CMF)  
**Description**: Combines price and volume to measure buying/selling pressure.  
**Variants**:  
- `CMF_10`, `CMF_20`, `CMF_50`, `CMF_100`.  
**Input**: `high`, `low`, `close`, `volume`.  
**Formula**:  
  - `MFM = (Close - Low) / (High - Low)`.  
  - `CMF = ∑(MFM * Volume) / ∑Volume`.  
**Output**: -1 to 1 series (positive = buying pressure).  

---

## 9. Fibonacci Retracement Levels  
**Levels**:  
- `Fibonacci_0`, `Fibonacci_23.6`, `Fibonacci_38.2`, `Fibonacci_50`, `Fibonacci_61.8`, `Fibonacci_100`.  
**Input**: Historical `high` and `low` from the dataset.  
**Output**: Static price levels (support/resistance).  

---

## 10. Parabolic SAR  
**Description**: Trend-following indicator (dots below/above price).  
**Input**: `high`, `low`, `close`.  
**Parameters**: Acceleration factor (0.02).  
**Output**: Float series (rising = downtrend, falling = uptrend).  

---

## 11. Ichimoku Cloud  
**Components**:  
- `Tenkan_Sen` (9-period avg).  
- `Kijun_Sen` (26-period avg).  
- `Senkou_Span_A` (leading span A).  
- `Senkou_Span_B` (leading span B).  
- `Chikou_Span` (lagging line).  
**Input**: `high`, `low`, `close`.  
**Output**: Five series (cloud-based support/resistance).  

---

## Notes  
- **Input Data**: All indicators require OHLCV columns (`open`, `high`, `low`, `close`, `volume`).  
- **Output Types**:  
  - Time-series values (e.g., `RSI`, `EMA`).  
  - Boolean flags (e.g., `Bullish_Engulfing`).  
  - Static levels (e.g., `Fibonacci_61.8`).  
- **Usage**:  
  - Trend analysis (e.g., `SMA`, `MACD`).  
  - Momentum/volatility (e.g., `RSI`, `Bollinger Bands`).  
  - Volume-based signals (e.g., `CMF`).  
  - Reversal patterns (e.g., `Parabolic SAR`, `Candlestick Patterns`).  