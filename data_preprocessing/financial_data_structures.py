from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BarResult:
    """Container for sampled bar endpoints and full OHLCV aggregates."""

    sample: pd.DataFrame
    ohlcv: pd.DataFrame


def _ewma(values: list[float], span: int) -> float:
    """Return the latest exponentially weighted moving average value."""
    if not values:
        return 0.0
    return float(pd.Series(values, dtype=float).ewm(span=span, adjust=False).mean().iloc[-1])


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
        ValueError: If no timestamp information is available.
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
    df["signed_tick"] = df["tick_sign"]
    df["signed_volume"] = df["tick_sign"] * df[volume_col]
    df["signed_dollar_value"] = df["tick_sign"] * df["dollar_value"]
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
    """Build tick bars from raw trade data."""
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
    """Build volume bars from raw trade data."""
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
    """Build dollar bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_threshold_bar_end_indices(prepared["dollar_value"], threshold)
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def _compute_imbalance_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    """Find imbalance-bar boundaries with adaptive expectations."""
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    seed = values[: min(len(values), expected_num_ticks_init)]
    non_zero_seed = seed[np.abs(seed) > 0]
    expected_imbalance = float(non_zero_seed.mean()) if len(non_zero_seed) else 1.0
    expected_num_ticks = float(max(min_exp_num_ticks, min(expected_num_ticks_init, len(values))))

    indices: list[int] = []
    bar_sizes: list[int] = []
    mean_bar_imbalances: list[float] = []

    cumulative_imbalance = 0.0
    ticks_in_bar = 0
    current_values: list[float] = []

    for idx, value in enumerate(values):
        cumulative_imbalance += value
        ticks_in_bar += 1
        current_values.append(value)
        threshold = max(1e-12, expected_num_ticks * abs(expected_imbalance))

        if abs(cumulative_imbalance) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)
            mean_bar_imbalances.append(float(np.mean(current_values)))

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-expected_window:], expected_window),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_imbalance = _ewma(mean_bar_imbalances[-expected_window:], expected_window)
            if abs(expected_imbalance) < 1e-12:
                expected_imbalance = 1.0

            cumulative_imbalance = 0.0
            ticks_in_bar = 0
            current_values = []

    return indices


def get_tick_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build tick imbalance bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_tick",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build volume imbalance bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_volume",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build dollar imbalance bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_dollar_value",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def _compute_run_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    """Find run-bar boundaries with adaptive buy and sell expectations."""
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    initial = values[: min(len(values), expected_num_ticks_init)]
    initial_buy = initial[initial > 0]
    initial_sell = -initial[initial < 0]
    expected_buy = float(initial_buy.mean()) if len(initial_buy) else 1.0
    expected_sell = float(initial_sell.mean()) if len(initial_sell) else 1.0
    expected_buy_prob = float((initial > 0).mean()) if len(initial) else 0.5
    expected_num_ticks = float(max(min_exp_num_ticks, min(expected_num_ticks_init, len(values))))

    indices: list[int] = []
    bar_sizes: list[int] = []
    bar_buy_probs: list[float] = []
    bar_buy_means: list[float] = []
    bar_sell_means: list[float] = []

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    ticks_in_bar = 0
    buy_values: list[float] = []
    sell_values: list[float] = []

    for idx, value in enumerate(values):
        ticks_in_bar += 1
        if value > 0:
            cumulative_buy += value
            buy_values.append(value)
        elif value < 0:
            cumulative_sell += -value
            sell_values.append(-value)

        threshold = max(
            1e-12,
            expected_num_ticks
            * max(expected_buy_prob * expected_buy, (1.0 - expected_buy_prob) * expected_sell),
        )

        if max(cumulative_buy, cumulative_sell) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)
            bar_buy_probs.append(len(buy_values) / ticks_in_bar if ticks_in_bar else 0.5)
            bar_buy_means.append(float(np.mean(buy_values)) if buy_values else 0.0)
            bar_sell_means.append(float(np.mean(sell_values)) if sell_values else 0.0)

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-expected_window:], expected_window),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_buy_prob = _ewma(bar_buy_probs[-expected_window:], expected_window)
            expected_buy = max(_ewma(bar_buy_means[-expected_window:], expected_window), 1e-12)
            expected_sell = max(_ewma(bar_sell_means[-expected_window:], expected_window), 1e-12)

            cumulative_buy = 0.0
            cumulative_sell = 0.0
            ticks_in_bar = 0
            buy_values = []
            sell_values = []

    return indices


def get_tick_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build tick run bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_tick",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build volume run bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_volume",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        expected_num_ticks_init: int = 1_000,
        expected_window: int = 20,
) -> BarResult:
    """Build dollar run bars from raw trade data."""
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_dollar_value",
        expected_num_ticks_init=expected_num_ticks_init,
        expected_window=expected_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_etf_trick_series(
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        *,
        initial_value: float = 1.0,
) -> pd.Series:
    """Compute the ETF trick net asset value series.

    Args:
        prices: Asset price history.
        weights: Asset allocation weights aligned by date.
        initial_value: Starting portfolio value.

    Returns:
        A net asset value series.
    """
    common_columns = prices.columns.intersection(weights.columns)
    if len(common_columns) == 0:
        raise ValueError("Prices and weights must share at least one asset column.")
    if initial_value <= 0:
        raise ValueError("Initial value must be positive.")

    aligned_prices = prices.loc[:, common_columns].astype(float).sort_index()
    aligned_weights = weights.loc[:, common_columns].astype(float).reindex(aligned_prices.index).ffill().fillna(0.0)

    returns = aligned_prices.pct_change().fillna(0.0)
    lagged_weights = aligned_weights.shift(1).fillna(0.0)
    portfolio_returns = (lagged_weights * returns).sum(axis=1)

    nav = initial_value * (1.0 + portfolio_returns).cumprod()
    nav.name = "etf_trick"
    return nav


def get_pca_weights(
        cov: pd.DataFrame | np.ndarray,
        risk_dist: np.ndarray | pd.Series | None = None,
        risk_target: float = 1.0,
) -> pd.Series | np.ndarray:
    """Compute PCA portfolio weights for a target risk distribution.

    Args:
        cov: Covariance matrix.
        risk_dist: Risk allocation across principal components.
        risk_target: Total target risk scale.

    Returns:
        Portfolio weights as a series or array matching ``cov``.
    """
    cov_values = cov.to_numpy(dtype=float) if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    if cov_values.ndim != 2 or cov_values.shape[0] != cov_values.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if risk_target <= 0:
        raise ValueError("Risk target must be positive.")

    eigenvalues, eigenvectors = np.linalg.eigh(cov_values)
    indices = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[indices]
    eigenvectors = eigenvectors[:, indices]

    if np.any(eigenvalues <= 0):
        raise ValueError("Covariance matrix must be positive definite.")

    if risk_dist is None:
        risk_dist_values = np.zeros(cov_values.shape[0], dtype=float)
        risk_dist_values[-1] = 1.0
    else:
        risk_dist_values = np.asarray(risk_dist, dtype=float).reshape(-1)
        if risk_dist_values.shape[0] != cov_values.shape[0]:
            raise ValueError("Risk distribution must match covariance dimensions.")

    loads = risk_target * np.sqrt(risk_dist_values / eigenvalues)
    weights = eigenvectors @ loads

    if isinstance(cov, pd.DataFrame):
        return pd.Series(weights, index=cov.index, name="pca_weight")
    return weights


def get_cusum_events(g_raw: pd.Series, threshold: float) -> pd.DatetimeIndex:
    """Detect event timestamps with a symmetric CUSUM filter.

    Args:
        g_raw: Input series to filter.
        threshold: Positive CUSUM threshold.

    Returns:
        A datetime index of detected event times.
    """
    if threshold <= 0:
        raise ValueError("Threshold must be positive.")

    t_events: list[pd.Timestamp] = []
    s_pos = 0.0
    s_neg = 0.0
    diff = g_raw.astype(float).diff().dropna()

    for timestamp, value in diff.items():
        s_pos = max(0.0, s_pos + value)
        s_neg = min(0.0, s_neg + value)
        if s_neg < -threshold:
            s_neg = 0.0
            t_events.append(timestamp)
        elif s_pos > threshold:
            s_pos = 0.0
            t_events.append(timestamp)

    return pd.DatetimeIndex(t_events)
