from __future__ import annotations

from pathlib import Path
from math import isfinite
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm.auto import tqdm


DOLLAR_BAR_COLUMNS = [
    "start", "end", "symbol", "open", "high", "low", "close", "volume",
]


def adjust_trade_units(trades: pd.DataFrame, events: list[dict], *,
                       reference: pd.Timestamp) -> pd.DataFrame:
    """Keep p*s unchanged while expressing every trade in reference-date shares."""
    adjusted = trades.copy()
    factor = pd.Series(1.0, index=trades.index)
    timestamps = pd.to_datetime(trades["timestamp"], utc=True)
    for event in events:
        effective = pd.Timestamp(event["effective_at"])
        ratio = float(event["ratio"])
        if not isfinite(ratio) or ratio <= 0:
            raise ValueError("Split ratio must be finite and positive")
        if effective <= reference:
            factor.loc[timestamps < effective] /= ratio
        else:
            factor.loc[timestamps >= effective] *= ratio
    adjusted["price"] = trades["price"] * factor
    adjusted["size"] = trades["size"] / factor
    # Retain original dollar values to avoid floating-point changes to boundaries.
    adjusted["dollar_value"] = trades["price"] * trades["size"]
    return adjusted


def _prepare_trade_data(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize fixed-schema trades, preserving supplied dollar values.

    Args:
        trades: Raw trades with timestamp, symbol, price, and size columns.

    Returns:
        A normalized trade frame indexed by UTC timestamp.

    Raises:
        ValueError: If required trade columns are missing.
    """
    missing = {"timestamp", "symbol", "price", "size"}.difference(trades.columns)
    if missing:
        raise ValueError(f"Trades are missing required columns: {sorted(missing)}")
    df = trades.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp", kind="stable").set_index("timestamp")
    df["price"] = df["price"].astype(float)
    df["size"] = df["size"].astype(float)
    if "dollar_value" not in df:
        df["dollar_value"] = df["price"] * df["size"]
    return df


def _build_ohlcv_bars(
        trades: pd.DataFrame,
        bar_end_indices: list[int],
) -> pd.DataFrame:
    """Aggregate prepared trades into OHLCV bars.

    Args:
        trades: Prepared trade data.
        bar_end_indices: Positional indices marking bar boundaries.

    Returns:
        An OHLCV DataFrame indexed by end time when non-empty.
    """
    if not bar_end_indices:
        return pd.DataFrame(
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
    rows: list[dict] = []
    start_idx = 0
    for end_idx in bar_end_indices:
        window = trades.iloc[start_idx: end_idx + 1]
        rows.append(
            {
                "start": window.index[0],
                "end": window.index[-1],
                "symbol": window["symbol"].iloc[-1],
                "open": float(window["price"].iloc[0]),
                "high": float(window["price"].max()),
                "low": float(window["price"].min()),
                "close": float(window["price"].iloc[-1]),
                "volume": float(window["size"].sum()),
                "dollar_value": float(window["dollar_value"].sum()),
                "ticks": int(len(window)),
            }
        )
        start_idx = end_idx + 1

    return pd.DataFrame(rows).set_index("end")


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
        complete_timestamps: bool = False,
) -> pd.DataFrame:
    """Build dollar bars from raw trade data.

    Args:
        trades: Raw trades with timestamp, symbol, price, and size columns.
        threshold: Dollar-value threshold per completed bar.
        complete_timestamps: Keep all trades at the same timestamp in one bar.

    Returns:
        An OHLCV DataFrame including internal ``dollar_value`` and ``ticks``
        counts. Non-empty results are indexed by ``end``; empty results retain
        ``end`` as a column. Buy/sell volumes and tick signs are not returned.
    """
    prepared = _prepare_trade_data(trades)
    if complete_timestamps:
        grouped = prepared["dollar_value"].groupby(level=0, sort=True).sum()
        endpoints = _compute_threshold_bar_end_indices(grouped, threshold)
        indices = (prepared.index.searchsorted(grouped.index[endpoints], side="right") - 1).tolist()
    else:
        indices = _compute_threshold_bar_end_indices(prepared["dollar_value"], threshold)
    return _build_ohlcv_bars(prepared, indices)


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
        feature_identity, load_split_events,
        raw_partitions, reusable_feature, save_feature, split_path,
    )

    if lookback_sessions < 1 or target_bars_per_session < 1:
        raise ValueError("Dollar-bar lookback and target bar count must be positive")
    output = paths.feature(symbol, "dollar_bars")
    partitions = raw_partitions(paths, symbol, "tick", start=start, end=end)
    for path in partitions:
        if not path.with_suffix(".json").exists():
            raise ValueError(f"Incomplete raw partition: {path}")
    events = load_split_events(paths, symbol, start=start, end=end)
    identity = feature_identity(
        paths,
        [split_path(paths, symbol), *partitions,
         *(path.with_suffix(".json") for path in partitions)],
        manifest_path=manifest_path,
        settings={"start": start, "end": end, "lookback_sessions": lookback_sessions,
                  "target_bars_per_session": target_bars_per_session,
                  "columns": DOLLAR_BAR_COLUMNS,
                  "share_reference": paths.start,
                  "split_events": events},
    )
    if reusable_feature(output, identity):
        return {"symbol": symbol, "output_rows": pq.read_metadata(output).num_rows,
                "processed_files": 0, "cached_files": 1}
    pending = pd.DataFrame()
    history, parts = [], []
    for file in partitions:
        trades = pd.read_parquet(file)
        if trades.empty:
            continue
        trades = adjust_trade_units(trades, events, reference=paths.start)
        daily_value = float(trades["dollar_value"].sum())
        if history:
            threshold = float(np.median(history[-lookback_sessions:])) / target_bars_per_session
            combined = pd.concat([pending, trades], ignore_index=True)
            bars = get_dollar_bars(
                combined, threshold=threshold, complete_timestamps=True,
            ).reset_index()
            if not bars.empty:
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
    from src.preprocessing.market_data import load_manifest, require_feature_metadata

    symbols = load_manifest(manifest_path).symbol
    for symbol in symbols:
        require_feature_metadata(paths.feature(symbol, "dollar_bars"))

    return pd.concat(
        [pd.read_parquet(paths.feature(symbol, "dollar_bars"))
         for symbol in symbols],
        ignore_index=True,
    )
