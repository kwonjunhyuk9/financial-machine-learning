import numpy as np
import pandas as pd
import pytest
from unittest.mock import Mock

from src.preprocessing import market_differentiated_bars as fractional

from src.preprocessing.market_differentiated_bars import (
    evaluate_fractional_differencing_orders,
    get_weights_fixed_width,
)


def test_fixed_width_weights_stop_at_threshold():
    weights = get_weights_fixed_width(
        differencing_order=0.5,
        weight_cutoff=0.1,
    )

    assert weights[-1, 0] == 1.0
    assert len(weights) > 1


def test_evaluate_orders_uses_descriptive_column_names():
    random_generator = np.random.default_rng(seed=0)
    log_prices = random_generator.normal(scale=0.01, size=200).cumsum()
    log_price_series = pd.Series(log_prices, name='log_close')

    diagnostics = evaluate_fractional_differencing_orders(
        log_price_series,
        weight_cutoff=0.01,
        differencing_orders=[0.0],
    )

    assert diagnostics.columns.tolist() == [
        'adf_statistic',
        'p_value',
        'used_lags',
        'n_observations',
        'critical_value_5pct',
        'correlation',
    ]


# Preserve the former implementation as an independent numerical reference.
def reference_difference(frame, order, cutoff=0.01):
    weights = get_weights_fixed_width(order, cutoff)
    width = len(weights) - 1
    columns = {}
    for column in frame.columns:
        filled = frame[[column]].ffill().dropna()
        output = pd.Series()
        for position in range(width, len(filled)):
            start, end = filled.index[position - width], filled.index[position]
            if not np.isfinite(frame.loc[end, column]):
                continue
            output.loc[end] = np.dot(weights.T, filled.loc[start:end])[0, 0]
        columns[column] = output.copy(deep=True)
    return pd.concat(columns, axis=1)



@pytest.mark.parametrize("order", [0.0, 0.3, 0.5, 1.0])
@pytest.mark.parametrize("case", ["regular", "missing", "infinite", "short", "empty", "all_missing", "all_infinite"])
def test_vectorized_difference_matches_reference(order, case):
    size = 2 if case == "short" else 0 if case == "empty" else 80
    frame = pd.DataFrame(np.random.default_rng(8).normal(size=(size, 2)), columns=["a", "b"],
                         index=pd.date_range("2025-01-01", periods=size, freq="min", tz="UTC", name="end"))
    if case == "missing":
        frame.iloc[:3, 0] = np.nan
        frame.iloc[20:23, 1] = np.nan
    elif case == "infinite":
        frame.iloc[15, 0] = np.inf
        frame.iloc[25, 1] = -np.inf
    elif case == "all_missing":
        frame[:] = np.nan
    elif case == "all_infinite":
        frame[:] = np.inf
    with np.errstate(invalid="ignore"):
        expected = reference_difference(frame, order)
        actual = fractional.fractional_difference_fixed_width(frame, order, 0.01)
    pd.testing.assert_frame_equal(actual, expected, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pass_at", [0, 2, None])
def test_early_stop_sorts_unique_orders_and_stops_at_first_pass(monkeypatch, pass_at):
    import statsmodels.tsa.stattools as stattools

    count = 0
    def adf(values, **kwargs):
        nonlocal count
        statistic = -4.0 if count == pass_at else -2.0
        count += 1
        return statistic, 0.1, 1, len(values), {"5%": -3.0}

    monkeypatch.setattr(stattools, "adfuller", adf)
    prices = pd.Series(np.random.default_rng(2).normal(size=100).cumsum())
    result = fractional.evaluate_fractional_differencing_orders(
        prices, differencing_orders=[0.3, 0.0, 0.2, 0.1, 0.2], stop_at_first_stationary=True,
    )
    expected_orders = [0.0, 0.1, 0.2, 0.3] if pass_at is None else [0.0, 0.1, 0.2, 0.3][:pass_at + 1]
    assert result.index.tolist() == expected_orders
    assert count == len(expected_orders)


def test_default_diagnostics_evaluates_every_candidate_in_requested_order(monkeypatch):
    import statsmodels.tsa.stattools as stattools

    adf = Mock(return_value=(-4.0, 0.1, 1, 70, {"5%": -3.0}))
    monkeypatch.setattr(stattools, "adfuller", adf)
    result = fractional.evaluate_fractional_differencing_orders(
        pd.Series(np.random.default_rng(3).normal(size=100).cumsum()),
        differencing_orders=[0.3, 0.0, 0.2],
    )
    assert result.index.tolist() == [0.3, 0.0, 0.2]
    assert adf.call_count == 3


@pytest.mark.parametrize("stationary", [False, True])
def test_early_and_full_search_choose_same_order_and_features(stationary):
    values = np.random.default_rng(4).normal(size=500)
    if not stationary:
        values = values.cumsum()
    frame = pd.DataFrame({"log_close": values})
    full = fractional.evaluate_fractional_differencing_orders(frame)
    early = fractional.evaluate_fractional_differencing_orders(frame, stop_at_first_stationary=True)
    passing = full.index[full.adf_statistic < full.critical_value_5pct]
    assert len(passing)
    order = float(passing.min())
    assert early.index[-1] == order
    pd.testing.assert_frame_equal(
        fractional.fractional_difference_fixed_width(frame, order, 0.01),
        reference_difference(frame, order), rtol=1e-12, atol=1e-12,
    )


@pytest.mark.parametrize("passes", [True, False])
def test_builder_early_stop_cache_and_failure(tmp_path, monkeypatch, passes):
    from io import StringIO
    from tqdm.auto import tqdm
    from src.preprocessing import market_data

    paths = market_data.ResearchPaths(tmp_path, period="2025-02-01_2025-12-31")
    monkeypatch.setattr(market_data, "load_manifest", lambda *a, **k: pd.DataFrame({"symbol": ["A"]}))
    monkeypatch.setattr(market_data, "feature_identity", lambda *a, **k: {"settings": {"test": True}})
    source = paths.feature("A", "dollar_bars")
    times = pd.date_range("2025-01-01", periods=120, freq="h", tz="UTC")
    market_data.save_frame(pd.DataFrame({"end": times, "close": np.exp(5 + np.random.default_rng(5).normal(0, 0.01, 120))}), source)
    evaluations = []
    def evaluate(frame, **kwargs):
        assert frame.index.max() < pd.Timestamp("2025-01-05", tz="UTC")
        assert kwargs["stop_at_first_stationary"]
        evaluations.append(frame)
        return pd.DataFrame({"adf_statistic": [-4.0 if passes else -2.0],
                             "critical_value_5pct": [-3.0]}, index=[0.3])

    monkeypatch.setattr(fractional, "evaluate_fractional_differencing_orders", evaluate)
    bars = []
    def progress(*a, **k):
        bar = tqdm(*a, file=StringIO(), **k)
        bars.append(bar)
        return bar
    monkeypatch.setattr(fractional, "tqdm", progress)
    kwargs = dict(manifest_path=tmp_path / "manifest.csv", expected_securities=1,
                  fit_end=pd.Timestamp("2025-01-05", tz="UTC"), weight_cutoff=0.01,
                  differencing_orders=[0.0, 0.3, 1.0], adf_maxlag=1, adf_regression="c",
                  adf_autolag=None, adf_significance="5%", show_progress=True)
    if not passes:
        with pytest.raises(ValueError, match="No warmup fractional order passes stationarity for A"):
            fractional.build_fractional_features(paths, **kwargs)
        assert not paths.feature("A", "fractional").exists()
    else:
        report = fractional.build_fractional_features(paths, **kwargs)
        assert report.d.tolist() == [0.3]
        saved = pd.read_parquet(paths.feature("A", "fractional"))
        assert saved.attrs["differencing_order"] == 0.3
        assert saved.end.max() >= kwargs["fit_end"]
        cached = fractional.build_fractional_features(paths, **kwargs)
        assert cached.status.tolist() == ["cached"]
        assert len(evaluations) == 1
    assert all(bar.disable for bar in bars)
