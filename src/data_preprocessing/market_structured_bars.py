from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarResult:
    """Container for sampled bar endpoints and full OHLCV aggregates.

    Attributes:
        sample: Source-trade rows at each sampled bar endpoint.
        ohlcv: Aggregated OHLCV rows for completed bars.
    """

    sample: pd.DataFrame
    ohlcv: pd.DataFrame


def save_structured_bar_result(result: BarResult, output_path: Path) -> Path:
    """Save one structured-bar OHLCV result to parquet.

    Args:
        result: Structured-bar result to save.
        output_path: Parquet destination.

    Returns:
        The parquet path written to disk.
    """
    columns = [
        "end",
        "start",
        "symbol",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "dollar_value",
        "ticks",
        "buy_volume",
        "sell_volume",
    ]
    bars = result.ohlcv.copy()
    if bars.index.name == "end":
        bars = bars.reset_index()
    else:
        bars = bars.reset_index(drop=True)
    bars = bars.reindex(columns=columns)
    bars["end"] = pd.to_datetime(bars["end"], utc=True)
    bars["start"] = pd.to_datetime(bars["start"], utc=True)

    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(destination, index=False)
    return destination


def _prepare_trade_data(
        trades: pd.DataFrame,
        *,
        timestamp_col: str = "timestamp",
        price_col: str = "price",
        volume_col: str = "size",
        symbol_col: str = "symbol",
) -> pd.DataFrame:
    """Normalize raw trade data for bar construction.

    Args:
        trades: Raw trade data.
        timestamp_col: Timestamp column name.
        price_col: Price column name.
        volume_col: Volume column name.
        symbol_col: Symbol column name.

    Returns:
        A normalized trade frame indexed by timestamp.

    Raises:
        ValueError: If timestamp information is unavailable.
    """
    df = trades.copy()
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.sort_values([timestamp_col], kind="stable")
        df = df.set_index(timestamp_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Trades must include a timestamp column or a DatetimeIndex.")

    df.index.name = "timestamp"
    df[price_col] = df[price_col].astype(float)
    df[volume_col] = df[volume_col].astype(float)
    if symbol_col not in df.columns:
        df[symbol_col] = "UNKNOWN"

    price_diff = df[price_col].diff()
    tick_sign = np.sign(price_diff).replace(0.0, np.nan).ffill().fillna(1.0)
    df["tick_sign"] = tick_sign.astype(float)
    df["dollar_value"] = df[price_col] * df[volume_col]
    return df


def _build_ohlcv_bars(
        trades: pd.DataFrame,
        bar_end_indices: list[int],
        *,
        price_col: str,
        volume_col: str,
) -> BarResult:
    """Aggregate prepared trades into OHLCV bars.

    Args:
        trades: Prepared trade data.
        bar_end_indices: Positional indices marking bar boundaries.
        price_col: Price column name.
        volume_col: Volume column name.

    Returns:
        A ``BarResult`` containing sampled bar endpoints and OHLCV rows.
    """
    if not bar_end_indices:
        empty_sample = trades.iloc[0:0].copy()
        empty_ohlcv = pd.DataFrame(
            columns=[
                "start",
                "end",
                "symbol",
                "open",
                "high",
                "low",
                "close",
                "volume",
                "dollar_value",
                "ticks",
                "buy_volume",
                "sell_volume",
            ]
        )
        return BarResult(sample=empty_sample, ohlcv=empty_ohlcv)

    sample = trades.iloc[bar_end_indices].copy()
    rows: list[dict] = []
    start_idx = 0
    for end_idx in bar_end_indices:
        window = trades.iloc[start_idx: end_idx + 1]
        rows.append(
            {
                "start": window.index[0],
                "end": window.index[-1],
                "symbol": window["symbol"].iloc[-1],
                "open": float(window[price_col].iloc[0]),
                "high": float(window[price_col].max()),
                "low": float(window[price_col].min()),
                "close": float(window[price_col].iloc[-1]),
                "volume": float(window[volume_col].sum()),
                "dollar_value": float(window["dollar_value"].sum()),
                "ticks": int(len(window)),
                "buy_volume": float(window.loc[window["tick_sign"] > 0, volume_col].sum()),
                "sell_volume": float(window.loc[window["tick_sign"] < 0, volume_col].sum()),
            }
        )
        start_idx = end_idx + 1

    ohlcv = pd.DataFrame(rows).set_index("end")
    return BarResult(sample=sample, ohlcv=ohlcv)


def _compute_threshold_bar_end_indices(values: pd.Series, threshold: float) -> list[int]:
    """Find bar boundaries whenever a cumulative threshold is reached.

    Args:
        values: Input values to accumulate.
        threshold: Positive threshold that closes a bar.

    Returns:
        Positional indices marking the end of each completed bar.

    Raises:
        ValueError: If ``threshold`` is not positive.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")
    cumulative_value = 0.0
    indices: list[int] = []
    for idx, value in enumerate(values.astype(float).to_numpy()):
        cumulative_value += value
        if cumulative_value >= threshold:
            indices.append(idx)
            cumulative_value = 0.0
    return indices


def get_tick_bars(
        trades: pd.DataFrame,
        threshold: int,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    """Build tick bars from raw trade data.

    Args:
        trades: Raw trade data.
        threshold: Number of ticks per completed bar.
        price_col: Price column name.
        volume_col: Volume column name.

    Returns:
        A ``BarResult`` with tick bars and OHLCV aggregates.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(pd.Series(1.0, index=prepared.index), threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_bars(
        trades: pd.DataFrame,
        threshold: float,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    """Build volume bars from raw trade data.

    Args:
        trades: Raw trade data.
        threshold: Volume threshold per completed bar.
        price_col: Price column name.
        volume_col: Volume column name.

    Returns:
        A ``BarResult`` with volume bars and OHLCV aggregates.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(prepared[volume_col], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_bars(
        trades: pd.DataFrame,
        threshold: float,
        *,
        price_col: str = "price",
        volume_col: str = "size",
) -> BarResult:
    """Build dollar bars from raw trade data.

    Args:
        trades: Raw trade data.
        threshold: Dollar-value threshold per completed bar.
        price_col: Price column name.
        volume_col: Volume column name.

    Returns:
        A ``BarResult`` with dollar bars and OHLCV aggregates.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(prepared["dollar_value"], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)
