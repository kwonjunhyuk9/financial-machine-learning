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


def _validate_adaptive_bar_parameters(
        exp_num_ticks_init: int,
        num_prev_bars: int,
        expected_imbalance_window: int,
) -> None:
    """Validate adaptive-bar expectation parameters."""
    parameters = (exp_num_ticks_init, num_prev_bars, expected_imbalance_window)
    if any(
            isinstance(value, bool)
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in parameters
    ):
        raise ValueError("Adaptive-bar expectation parameters must be positive integers.")


def _expected_imbalance(values: list[float], window: int) -> float:
    """Return the latest EWMA over the available raw imbalances."""
    actual_window = min(len(values), window)
    if actual_window == 0:
        return 0.0
    return _ewma(values[-actual_window:], actual_window)


def _compute_imbalance_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    """Find imbalance-bar boundaries with adaptive expectations.

    Args:
        prepared: Prepared trade data.
        imbalance_col: Signed imbalance column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks.
        expected_imbalance_window: Observation window used to update expected imbalance.
        min_exp_num_ticks: Minimum expected tick count.
        max_exp_num_ticks: Maximum expected tick count.

    Returns:
        Positional indices marking imbalance-bar endpoints.
    """
    _validate_adaptive_bar_parameters(
        exp_num_ticks_init,
        num_prev_bars,
        expected_imbalance_window,
    )
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    expected_num_ticks = float(np.clip(exp_num_ticks_init, min_exp_num_ticks, max_exp_num_ticks))
    expected_imbalance = 0.0
    raw_imbalances: list[float] = []
    indices: list[int] = []
    bar_sizes: list[int] = []

    cumulative_imbalance = 0.0
    ticks_in_bar = 0

    for idx, value in enumerate(values):
        raw_imbalances.append(float(value))
        cumulative_imbalance += value
        ticks_in_bar += 1

        if not indices:
            if len(raw_imbalances) < exp_num_ticks_init:
                continue
            expected_imbalance = _expected_imbalance(
                raw_imbalances,
                expected_imbalance_window,
            )

        threshold = max(1e-12, expected_num_ticks * abs(expected_imbalance))
        if abs(cumulative_imbalance) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-num_prev_bars:], num_prev_bars),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_imbalance = _expected_imbalance(raw_imbalances, expected_imbalance_window)

            cumulative_imbalance = 0.0
            ticks_in_bar = 0

    return indices


def get_tick_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build tick imbalance bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks.
        expected_imbalance_window: Observation window used to update expected imbalance.

    Returns:
        A ``BarResult`` with tick imbalance bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_tick",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build volume imbalance bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks.
        expected_imbalance_window: Observation window used to update expected imbalance.

    Returns:
        A ``BarResult`` with volume imbalance bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_volume",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_imbalance_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build dollar imbalance bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks.
        expected_imbalance_window: Observation window used to update expected imbalance.

    Returns:
        A ``BarResult`` with dollar imbalance bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_imbalance_bar_end_indices(
        prepared,
        "signed_dollar_value",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def _compute_run_bar_end_indices(
        prepared: pd.DataFrame,
        imbalance_col: str,
        *,
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
        min_exp_num_ticks: int = 10,
        max_exp_num_ticks: int = 100_000,
) -> list[int]:
    """Find run-bar boundaries with adaptive buy and sell expectations.

    Args:
        prepared: Prepared trade data.
        imbalance_col: Signed run-value column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks and buy probability.
        expected_imbalance_window: Observation window used to update buy and sell imbalance.
        min_exp_num_ticks: Minimum expected tick count.
        max_exp_num_ticks: Maximum expected tick count.

    Returns:
        Positional indices marking run-bar endpoints.
    """
    _validate_adaptive_bar_parameters(
        exp_num_ticks_init,
        num_prev_bars,
        expected_imbalance_window,
    )
    values = prepared[imbalance_col].astype(float).to_numpy()
    if len(values) == 0:
        return []

    tick_signs = prepared["tick_sign"].astype(float).to_numpy()
    expected_buy = 0.0
    expected_sell = 0.0
    expected_buy_prob = 0.5
    expected_num_ticks = float(np.clip(exp_num_ticks_init, min_exp_num_ticks, max_exp_num_ticks))

    indices: list[int] = []
    bar_sizes: list[int] = []
    bar_buy_probs: list[float] = []
    raw_buy_imbalances: list[float] = []
    raw_sell_imbalances: list[float] = []

    cumulative_buy = 0.0
    cumulative_sell = 0.0
    ticks_in_bar = 0
    buy_ticks_in_bar = 0

    for idx, (value, tick_sign) in enumerate(zip(values, tick_signs)):
        ticks_in_bar += 1
        if tick_sign > 0:
            magnitude = abs(float(value))
            cumulative_buy += magnitude
            raw_buy_imbalances.append(magnitude)
            buy_ticks_in_bar += 1
        else:
            magnitude = abs(float(value))
            cumulative_sell += magnitude
            raw_sell_imbalances.append(magnitude)

        if not indices:
            if idx + 1 < exp_num_ticks_init:
                continue
            expected_buy_prob = buy_ticks_in_bar / ticks_in_bar
            expected_buy = _expected_imbalance(
                raw_buy_imbalances,
                expected_imbalance_window,
            )
            expected_sell = _expected_imbalance(
                raw_sell_imbalances,
                expected_imbalance_window,
            )

        threshold = max(
            1e-12,
            expected_num_ticks
            * max(expected_buy_prob * expected_buy, (1.0 - expected_buy_prob) * expected_sell),
        )

        if max(cumulative_buy, cumulative_sell) >= threshold:
            indices.append(idx)
            bar_sizes.append(ticks_in_bar)
            bar_buy_probs.append(buy_ticks_in_bar / ticks_in_bar)

            expected_num_ticks = float(
                np.clip(
                    _ewma(bar_sizes[-num_prev_bars:], num_prev_bars),
                    min_exp_num_ticks,
                    max_exp_num_ticks,
                )
            )
            expected_buy_prob = _ewma(bar_buy_probs[-num_prev_bars:], num_prev_bars)
            expected_buy = _expected_imbalance(raw_buy_imbalances, expected_imbalance_window)
            expected_sell = _expected_imbalance(raw_sell_imbalances, expected_imbalance_window)

            cumulative_buy = 0.0
            cumulative_sell = 0.0
            ticks_in_bar = 0
            buy_ticks_in_bar = 0

    return indices


def get_tick_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build tick run bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks and buy probability.
        expected_imbalance_window: Observation window used to update buy and sell imbalance.

    Returns:
        A ``BarResult`` with tick run bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_tick",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_volume_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build volume run bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks and buy probability.
        expected_imbalance_window: Observation window used to update buy and sell imbalance.

    Returns:
        A ``BarResult`` with volume run bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_volume",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
    )
    return _build_ohlcv_bars(prepared, indices, price_col=price_col, volume_col=volume_col)


def get_dollar_run_bars(
        trades: pd.DataFrame,
        *,
        price_col: str = "price",
        volume_col: str = "size",
        exp_num_ticks_init: int = 20_000,
        num_prev_bars: int = 3,
        expected_imbalance_window: int = 10_000,
) -> BarResult:
    """Build dollar run bars from raw trade data.

    Args:
        trades: Raw trade data.
        price_col: Price column name.
        volume_col: Volume column name.
        exp_num_ticks_init: Initial expected number of ticks per bar.
        num_prev_bars: Completed-bar window used to update expected ticks and buy probability.
        expected_imbalance_window: Observation window used to update buy and sell imbalance.

    Returns:
        A ``BarResult`` with dollar run bars.
    """
    prepared = _prepare_trade_data(trades, price_col=price_col, volume_col=volume_col)
    indices = _compute_run_bar_end_indices(
        prepared,
        "signed_dollar_value",
        exp_num_ticks_init=exp_num_ticks_init,
        num_prev_bars=num_prev_bars,
        expected_imbalance_window=expected_imbalance_window,
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

    Raises:
        ValueError: If inputs have no common assets or ``initial_value`` is invalid.
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

    Raises:
        ValueError: If covariance inputs or risk targets are invalid.
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
