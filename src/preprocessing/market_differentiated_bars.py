from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


def get_weights_fixed_width(
    differencing_order: float,
    weight_cutoff: float,
) -> np.ndarray:
    """Compute fixed-width fractional differencing weights.

    Args:
        differencing_order: Fractional differencing order.
        weight_cutoff: Absolute weight cutoff threshold.

    Returns:
        A column vector of weights ordered from oldest to newest.
    """
    weights = [1.0]
    lag = 1
    while True:
        weight = (
            -weights[-1] / lag * (differencing_order - lag + 1)
        )
        if abs(weight) < weight_cutoff:
            break
        weights.append(weight)
        lag += 1
    return np.array(weights[::-1]).reshape(-1, 1)


def fractional_difference_fixed_width(
    series_frame: pd.DataFrame,
    differencing_order: float,
    weight_cutoff: float = 1e-5,
) -> pd.DataFrame:
    """Apply fixed-width fractional differencing to each column.

    Args:
        series_frame: Input time series frame.
        differencing_order: Fractional differencing order.
        weight_cutoff: Weight cutoff threshold used to set the window width.

    Returns:
        A frame of fractionally differenced series.
    """
    weights = get_weights_fixed_width(differencing_order, weight_cutoff)
    width = len(weights) - 1

    differentiated_columns = {}
    for column_name in series_frame.columns:
        filled_series = series_frame[[column_name]].ffill().dropna()
        if len(filled_series) <= width:
            differentiated_columns[column_name] = pd.Series()
            continue
        values = filled_series[column_name].to_numpy()
        differentiated_values = np.convolve(values, weights[:, 0][::-1], mode="valid")
        output_index = filled_series.index[width:]
        finite = np.isfinite(series_frame.loc[output_index, column_name].to_numpy())
        if not finite.any():
            differentiated_columns[column_name] = pd.Series()
            continue
        differentiated_columns[column_name] = pd.Series(
            differentiated_values[finite], index=pd.Index(output_index[finite].tolist()),
        )
    return pd.concat(differentiated_columns, axis=1)


def evaluate_fractional_differencing_orders(
    log_price_series: pd.Series | pd.DataFrame,
    weight_cutoff: float = 0.01,
    differencing_orders: Iterable[float] | None = None,
    *,
    adf_maxlag: int = 1,
    adf_regression: str = "c",
    adf_autolag: str | None = None,
    adf_significance: str = "5%",
    stop_at_first_stationary: bool = False,
) -> pd.DataFrame:
    """Evaluate stationarity across fractional differencing orders.

    Args:
        log_price_series: Time-indexed Series or single-column DataFrame of log
            prices.
        weight_cutoff: Weight cutoff passed to fixed-width differencing.
        differencing_orders: Optional iterable of fractional differencing orders.
        adf_maxlag: Maximum lag passed to the ADF test.
        adf_regression: Deterministic terms used by the ADF regression.
        adf_autolag: Optional ADF lag-selection method.
        adf_significance: Critical level, one of ``"1%"``, ``"5%"``, ``"10%"``.
        stop_at_first_stationary: Evaluate unique orders in ascending order and
            stop at the first order passing the ADF critical value.

    Returns:
        A DataFrame with ADF statistics and correlations by differencing order.
        Early stopping returns only evaluated orders.

    Raises:
        ValueError: If ``log_price_series`` is not a Series or single-column
            DataFrame.
    """
    from statsmodels.tsa.stattools import adfuller

    diagnostics = pd.DataFrame(
        columns=[
            'adf_statistic',
            'p_value',
            'used_lags',
            'n_observations',
            f"critical_value_{adf_significance.replace('%', 'pct')}",
            'correlation',
        ]
    )
    if isinstance(log_price_series, pd.Series):
        log_price_frame = log_price_series.to_frame(
            name=log_price_series.name or 'log_close'
        )
    else:
        log_price_frame = log_price_series.copy()
    if log_price_frame.shape[1] != 1:
        raise ValueError(
            'log_price_series must be a Series or single-column DataFrame'
        )
    column_name = log_price_frame.columns[0]
    if differencing_orders is None:
        differencing_orders = np.linspace(0, 1, 11)

    if stop_at_first_stationary:
        differencing_orders = sorted(set(differencing_orders))

    log_prices = log_price_frame[[column_name]].dropna()

    for differencing_order in differencing_orders:
        differentiated_prices = fractional_difference_fixed_width(
            log_prices,
            differencing_order,
            weight_cutoff=weight_cutoff,
        )
        correlation = np.corrcoef(
            log_prices.loc[differentiated_prices.index, column_name],
            differentiated_prices[column_name],
        )[0, 1]
        adf_result = adfuller(
            differentiated_prices[column_name],
            maxlag=adf_maxlag,
            regression=adf_regression,
            autolag=adf_autolag,
        )
        diagnostics.loc[differencing_order] = (
            list(adf_result[:4])
            + [adf_result[4][adf_significance]]
            + [correlation]
        )
        if stop_at_first_stationary and adf_result[0] < adf_result[4][adf_significance]:
            break

    return diagnostics


def build_fractional_features(paths, *, manifest_path, expected_securities: int,
                              fit_end: pd.Timestamp, weight_cutoff: float,
                              differencing_orders: Sequence[float], adf_maxlag: int,
                              adf_regression: str, adf_autolag: str | None,
                              adf_significance: str,
                              show_progress: bool = False) -> pd.DataFrame:
    """Fit warmup-only differencing orders and build every symbol feature.

    Args:
        show_progress: Show one overall progress bar, including cached symbols.
    """
    from src.preprocessing.market_data import (
        feature_identity, load_manifest, reusable_feature,
        save_feature,
    )

    report = []
    symbols = load_manifest(manifest_path, expected_securities=expected_securities).symbol
    with tqdm(symbols, desc="Fractional features", disable=not show_progress) as progress:
        for symbol in progress:
            progress.set_postfix_str(symbol)
            output = paths.feature(symbol, "fractional")
            identity = feature_identity(
                paths, [paths.feature(symbol, "dollar_bars")], manifest_path=manifest_path,
                settings={"fit_end": fit_end, "weight_cutoff": weight_cutoff,
                          "differencing_orders": list(differencing_orders), "adf_maxlag": adf_maxlag,
                          "adf_regression": adf_regression, "adf_autolag": adf_autolag,
                          "adf_significance": adf_significance},
            )
            if reusable_feature(output, identity):
                report.append({"symbol": symbol, "status": "cached"})
                continue
            bars = pd.read_parquet(paths.feature(symbol, "dollar_bars")).set_index("end")
            log_close = np.log(bars[["close"]]).rename(columns={"close": "log_close"})
            diagnostics = evaluate_fractional_differencing_orders(
                log_close.loc[log_close.index < fit_end],
                weight_cutoff=weight_cutoff,
                differencing_orders=differencing_orders,
                adf_maxlag=adf_maxlag, adf_regression=adf_regression, adf_autolag=adf_autolag,
                adf_significance=adf_significance,
                stop_at_first_stationary=True,
            )
            passing = diagnostics.index[diagnostics.adf_statistic < diagnostics[f"critical_value_{adf_significance.replace('%', 'pct')}"]]
            if passing.empty:
                raise ValueError(f"No warmup fractional order passes stationarity for {symbol}")
            order = float(passing.min())
            result = fractional_difference_fixed_width(log_close, order, weight_cutoff)
            result = result.rename(
                columns={"log_close": "fractionally_differenced_log_close"}
            ).rename_axis("end").reset_index()
            result["symbol"] = symbol
            result.attrs["differencing_order"] = order
            result.attrs["fit_end"] = str(fit_end)
            save_feature(result, output, identity)
            report.append(
                {
                    "symbol": symbol,
                    "d": order,
                    "fit_end": fit_end,
                    "rows": len(result),
                    "status": "processed",
                }
            )
    return pd.DataFrame(report)
