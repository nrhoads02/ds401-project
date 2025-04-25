# technical_indicators.py
# ---------------------------------------------------------------------------
# Core technical‑indicator engine for the volatility‑surface project.
# All calculations are vectorised with Polars and organised so that
# they stream efficiently when called from transformation_pipeline.py.
# ---------------------------------------------------------------------------

import math
from typing import List

import polars as pl
import numpy as np

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def add_technical_indicators(
    df: pl.LazyFrame,
    *,
    calculate_future_cols: bool = True,
    heston_windows: List[int] = (10, 15, 20, 25, 30, 35),
    vol_windows:   List[int] = (10, 20, 35),
    core_windows:  List[int] = (10, 20, 35)
) -> pl.LazyFrame:
    """
    Enrich an OHLCV DataFrame with a library of technical indicators plus
    daily realised‑variance estimates suitable for local‑vol surface modelling.

    Parameters
    ----------
    df : pl.LazyFrame
        Must contain columns `date`, `act_symbol`, `open`, `high`, `low`,
        `close`, `volume`.
    calculate_future_cols : bool, default True
        Whether to compute forward‑shifted target columns.
    heston_windows / vol_windows / core_windows
        Rolling‑window lengths used throughout the indicator set.

    Returns
    -------
    pl.LazyFrame
        Original columns + indicators.  Rolling targets are added when
        `calculate_future_cols` is True.
    """
    # Polars sorts
    df = df.sort(["date", "act_symbol"])

    # -----------------------------------------------------------------------
    # 1. Log returns and positive / negative splits
    # -----------------------------------------------------------------------
    df = df.with_columns(
        pl.when((pl.col("close").shift(1).over("act_symbol") > 0) &
                (pl.col("close") > 0))
          .then((pl.col("close") / pl.col("close").shift(1).over("act_symbol")).log())
          .otherwise(pl.lit(None))
          .alias("log_returns")
    )

    df = df.with_columns([
        pl.when(pl.col("log_returns") > 0).then(pl.col("log_returns")).otherwise(0).alias("pos_returns"),
        pl.when(pl.col("log_returns") < 0).then(-pl.col("log_returns")).otherwise(0).alias("neg_returns")
    ])

    # -----------------------------------------------------------------------
    # 2. Trend indicators: SMA, EMA, rolling STD
    # -----------------------------------------------------------------------
    trend_cols = []
    for n in vol_windows:
        trend_cols.extend([
            pl.col("close").rolling_mean(n).over("act_symbol").alias(f"SMA_{n}"),
            pl.col("close").ewm_mean(span=n, adjust=False).over("act_symbol").alias(f"EMA_{n}"),
            pl.col("close").rolling_std(n).over("act_symbol").alias(f"STD_{n}")
        ])
    df = df.with_columns(trend_cols)

    # SMA / EMA ratio
    df = df.with_columns([
        pl.when(pl.col(f"EMA_{n}") > 1e-6)
          .then(pl.col(f"SMA_{n}") / pl.col(f"EMA_{n}"))
          .otherwise(1.0)
          .alias(f"SMA_EMA_ratio_{n}")
        for n in core_windows
    ])

    # -----------------------------------------------------------------------
    # 3. RSI
    # -----------------------------------------------------------------------
    df = df.with_columns(pl.col("close").diff().over("act_symbol").alias("price_delta"))
    for n in core_windows:
        df = df.with_columns([
            pl.when(pl.col("price_delta") > 0).then(pl.col("price_delta")).otherwise(0).alias(f"gain_{n}"),
            pl.when(pl.col("price_delta") < 0).then(-pl.col("price_delta")).otherwise(0).alias(f"loss_{n}")
        ])
        df = df.with_columns([
            pl.col(f"gain_{n}").rolling_mean(n).over("act_symbol").alias(f"avg_gain_{n}"),
            pl.col(f"loss_{n}").rolling_mean(n).over("act_symbol").alias(f"avg_loss_{n}")
        ])
        df = df.with_columns(
            pl.when(pl.col(f"avg_loss_{n}") > 1e-6)
              .then(100 - 100 / (1 + pl.col(f"avg_gain_{n}") / pl.col(f"avg_loss_{n}")))
              .otherwise(pl.when(pl.col(f"avg_gain_{n}") > 0).then(100.0).otherwise(50.0))
              .clip(0, 100)
              .alias(f"RSI_{n}")
        )

    # -----------------------------------------------------------------------
    # 4. ATR
    # -----------------------------------------------------------------------
    for n in vol_windows:
        df = df.with_columns(pl.col("close").shift(1).over("act_symbol").alias(f"prev_close_{n}"))
        df = df.with_columns(
            pl.max_horizontal(
                pl.col("high") - pl.col("low"),
                (pl.col("high") - pl.col(f"prev_close_{n}")).abs(),
                (pl.col("low") - pl.col(f"prev_close_{n}")).abs()
            ).alias(f"TR_{n}")
        )
        df = df.with_columns(pl.col(f"TR_{n}").rolling_mean(n).over("act_symbol").alias(f"ATR_{n}"))

    # -----------------------------------------------------------------------
    # 5. Parkinson‑plus‑jumps daily variance
    # -----------------------------------------------------------------------
    pk_const = 1.0 / (4.0 * math.log(2.0))
    df = df.with_columns([
        # Intraday Parkinson component
        pl.when((pl.col("high") > 0) & (pl.col("low") > 0) & (pl.col("high") > pl.col("low")))
          .then((pl.col("high") / pl.col("low")).log().pow(2) * pk_const)
          .otherwise(0.0)
          .alias("intraday_range_var"),

        # Open–close jump
        pl.when((pl.col("open") > 0) & (pl.col("close") > 0))
          .then((pl.col("close") / pl.col("open")).log().pow(2))
          .otherwise(0.0)
          .alias("oc_var"),

        # Overnight jump (prev close → today open)
        pl.when((pl.col("open") > 0) &
                (pl.col("close").shift(1).over("act_symbol") > 0))
          .then((pl.col("open") /
                 pl.col("close").shift(1).over("act_symbol")).log().pow(2))
          .otherwise(0.0)
          .alias("overnight_var")
    ])

    df = df.with_columns(
        (pl.col("intraday_range_var") +
         pl.col("oc_var") +
         pl.col("overnight_var")).alias("parkinson_plus_jumps_daily")
    )

    # -----------------------------------------------------------------------
    # 6. Rolling realised variance and forward targets
    # -----------------------------------------------------------------------
    for h in heston_windows:  # 10,15,20,25,30,35
        df = df.with_columns(
            pl.col("parkinson_plus_jumps_daily")
              .rolling_sum(h).over("act_symbol")
              .alias(f"rv_{h}d")
        )
        if calculate_future_cols:
            df = df.with_columns([
                pl.col(f"rv_{h}d").shift(-h).over("act_symbol").alias(f"rv_{h}d_future"),
                pl.col("log_returns").rolling_sum(h).over("act_symbol")
                  .shift(-h).alias(f"log_ret_future_{h}")
            ])

    # -----------------------------------------------------------------------
    # 7. OBV‑family, CMF, CCI, VWAP – unchanged from original
    # -----------------------------------------------------------------------
    # Typical price
    df = df.with_columns(((pl.col("high") + pl.col("low") + pl.col("close")) / 3).alias("typical_price"))

    # CMF
    df = df.with_columns(
        pl.when(pl.col("high") - pl.col("low") > 1e-6)
          .then((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low")))
          .otherwise(0.5)
          .alias("money_flow_multiplier")
    )
    for n in core_windows:
        df = df.with_columns([
            (pl.col("money_flow_multiplier") * pl.col("volume"))
              .rolling_sum(n).over("act_symbol").alias(f"money_flow_volume_{n}"),
            pl.col("volume").rolling_sum(n).over("act_symbol").alias(f"volume_sum_{n}")
        ])
        df = df.with_columns(
            pl.when(pl.col(f"volume_sum_{n}") > 1e-6)
              .then(pl.col(f"money_flow_volume_{n}") / pl.col(f"volume_sum_{n}"))
              .otherwise(0.0)
              .clip(-1, 1)
              .alias(f"CMF_{n}")
        )

    # CCI
    for n in core_windows:
        df = df.with_columns(pl.col("typical_price").rolling_mean(n).over("act_symbol").alias(f"tp_sma_{n}"))
        df = df.with_columns((pl.col("typical_price") - pl.col(f"tp_sma_{n}")).alias(f"tp_dev_{n}"))
        df = df.with_columns(
            pl.col(f"tp_dev_{n}").abs().rolling_mean(n).over("act_symbol").alias(f"tp_mad_{n}")
        )
        df = df.with_columns(
            pl.when(pl.col(f"tp_mad_{n}") > 1e-6)
              .then((pl.col(f"tp_dev_{n}") / (0.015 * pl.col(f"tp_mad_{n}"))))
              .otherwise(0.0)
              .clip(-666.667, 666.667)
              .alias(f"CCI_{n}")
        )

    # OBV and OBV change
    df = df.with_columns(pl.col("close").shift(1).over("act_symbol").alias("prev_close_obv"))
    df = df.with_columns(
        pl.when(pl.col("close") > pl.col("prev_close_obv")).then(pl.col("volume"))
          .when(pl.col("close") < pl.col("prev_close_obv")).then(-pl.col("volume"))
          .otherwise(0).alias("volume_direction")
    )
    df = df.with_columns(pl.col("volume_direction").cum_sum().over("act_symbol").alias("OBV"))
    for n in core_windows:
        df = df.with_columns((pl.col("OBV") - pl.col("OBV").shift(n).over("act_symbol")).alias(f"OBV_change_{n}"))

    # VWAP and deviation
    for n in core_windows:
        df = df.with_columns([
            ((pl.col("typical_price") * pl.col("volume")).rolling_sum(n).over("act_symbol"))
              .alias(f"tp_vol_sum_{n}"),
            pl.col("volume").rolling_sum(n).over("act_symbol").alias(f"vwap_volume_{n}")
        ])
        df = df.with_columns(
            pl.when(pl.col(f"vwap_volume_{n}") > 1e-6)
              .then(pl.col(f"tp_vol_sum_{n}") / pl.col(f"vwap_volume_{n}"))
              .otherwise(pl.col("typical_price"))
              .alias(f"VWAP_{n}")
        )
        df = df.with_columns((pl.col("close") - pl.col(f"VWAP_{n}")).alias(f"VWAP_deviation_{n}"))

    # -----------------------------------------------------------------------
    return df
