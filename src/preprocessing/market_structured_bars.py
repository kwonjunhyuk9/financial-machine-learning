from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


DOLLAR_BAR_COLUMNS = [
    "start", "end", "symbol", "open", "high", "low", "close", "volume",
]


@dataclass(frozen=True)
class BarResult:
    """Container for sampled bar endpoints and full OHLCV aggregates.

    Attributes:
        sample: Source-trade rows at each sampled bar endpoint.
        ohlcv: Aggregated OHLCV rows for completed bars.
    """

    sample: pd.DataFrame
    ohlcv: pd.DataFrame


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


def get_dollar_bars(
        trades: pd.DataFrame,
        threshold: float,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        complete_timestamps: bool = False,
) -> BarResult:
    """Build dollar bars from raw trade data.

    Args:
        trades: Raw trade data.
        threshold: Dollar-value threshold per completed bar.
        price_col: Price column name.
        volume_col: Volume column name.

    Returns:
        A ``BarResult`` with sampled trades and OHLCV aggregates, including
        internal ``dollar_value`` and ``ticks`` counts. Buy/sell volumes and
        tick signs are not computed or returned.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    if complete_timestamps:
        grouped = prepared["dollar_value"].groupby(level=0, sort=True).sum()
        endpoints = _compute_threshold_bar_end_indices(grouped, threshold)
        indices = (prepared.index.searchsorted(grouped.index[endpoints], side="right") - 1).tolist()
    else:
        indices = _compute_threshold_bar_end_indices(prepared["dollar_value"], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def build_dollar_features(paths, *, manifest_path: Path,
                          start: pd.Timestamp, end: pd.Timestamp,
                          lookback_sessions: int, target_bars_per_session: int,
                          show_progress: bool = False) -> pd.DataFrame:
    """Build resumable dollar bars for the fixed universe and benchmark SPY.

    Persist only ``DOLLAR_BAR_COLUMNS``; dollar values and tick counts remain
    internal to bar construction and unfinished-trade handling.

    Args:
        show_progress: Show one overall progress bar, including cached symbols.
    """
    from src.preprocessing.market_data import load_manifest

    if lookback_sessions < 1 or target_bars_per_session < 1:
        raise ValueError("Dollar-bar lookback and target bar count must be positive")
    symbols = list(dict.fromkeys([*load_manifest(manifest_path).symbol, "SPY"]))
    report = []
    with tqdm(symbols, desc="Dollar bars", disable=not show_progress) as progress:
        for symbol in progress:
            progress.set_postfix_str(symbol)
            report.append(build_symbol_dollar_features(
                paths, symbol, manifest_path=manifest_path, start=start, end=end,
                lookback_sessions=lookback_sessions, target_bars_per_session=target_bars_per_session,
            ))
    return pd.DataFrame(report)


def build_symbol_dollar_features(paths, symbol: str, *, manifest_path: Path,
                                start: pd.Timestamp, end: pd.Timestamp,
                                lookback_sessions: int, target_bars_per_session: int) -> dict:
    """Build one symbol with the same cache identity used by the full pipeline."""
    from src.preprocessing.market_data import (
        feature_identity, raw_partitions, reusable_feature, save_feature,
    )

    if lookback_sessions < 1 or target_bars_per_session < 1:
        raise ValueError("Dollar-bar lookback and target bar count must be positive")
    output = paths.feature(symbol, "dollar_bars")
    partitions = raw_partitions(paths, symbol, "tick", start=start, end=end)
    identity = feature_identity(
        paths,
        [path.with_suffix(".json") for path in partitions],
        manifest_path=manifest_path,
        settings={"start": start, "end": end, "lookback_sessions": lookback_sessions,
                  "target_bars_per_session": target_bars_per_session,
                  "columns": DOLLAR_BAR_COLUMNS},
    )
    if reusable_feature(output, identity):
        return {"symbol": symbol, "output_rows": pq.read_metadata(output).num_rows,
                "processed_files": 0, "cached_files": 1}
    pending = pd.DataFrame()
    history, parts = [], []
    for file in partitions:
        if not file.with_suffix(".json").exists():
            raise ValueError(f"Incomplete raw partition: {file}")
        trades = pd.read_parquet(file)
        if trades.empty:
            continue
        daily_value = float((trades["price"] * trades["size"]).sum())
        if history:
            threshold = float(np.median(history[-lookback_sessions:])) / target_bars_per_session
            combined = pd.concat([pending, trades], ignore_index=True)
            result = get_dollar_bars(combined, threshold=threshold, complete_timestamps=True)
            bars = result.ohlcv.reset_index()
            if not bars.empty:
                bars = bars.groupby("end", as_index=False).agg(
                    start=("start", "min"), symbol=("symbol", "last"),
                    open=("open", "first"), high=("high", "max"), low=("low", "min"),
                    close=("close", "last"), volume=("volume", "sum"),
                    dollar_value=("dollar_value", "sum"), ticks=("ticks", "sum"),
                )
                parts.append(bars)
                pending = combined.iloc[int(bars.ticks.sum()):].copy()
            else:
                pending = combined
        history.append(daily_value)
    if not parts:
        raise ValueError(f"No dollar bars available for {symbol}")
    save_feature(pd.concat(parts, ignore_index=True)[DOLLAR_BAR_COLUMNS], output, identity)
    return {"symbol": symbol, "output_rows": sum(len(part) for part in parts),
            "processed_files": 1, "cached_files": 0, "unfinished_ticks": len(pending)}


def read_all_bars(paths, *, manifest_path: Path) -> pd.DataFrame:
    """Load dollar bars for all fixed-universe symbols."""
    from src.preprocessing.market_data import load_manifest

    return pd.concat(
        [pd.read_parquet(paths.feature(symbol, "dollar_bars"))
         for symbol in load_manifest(manifest_path).symbol],
        ignore_index=True,
    )
