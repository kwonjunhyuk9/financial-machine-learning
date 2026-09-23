import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score
from sklearn.metrics import recall_score

from src.backtesting.backtest_statistics import (
    ClassificationScores,
    Efficiency,
    GeneralCharacteristics,
    ImplementationShortfall,
    Performance,
    Runs,
    compute_strategy_returns,
)


def test_return_concentration_handles_absent_positive_or_negative_returns():
    assert np.isnan(Runs.hhi_positive_returns(pd.Series([-0.01, -0.02, -0.03])))
    assert np.isnan(Runs.hhi_negative_returns(pd.Series([0.01, 0.02, 0.03])))


def test_strategy_returns_default_to_zero_execution_costs():
    predictions = pd.DataFrame(
        {
            "raw_return": [0.02, -0.01],
            "primary_side": [1, -1],
            "meta_action": [1, 0],
            "bet_size": [0.5, 0.8],
        }
    )

    returns = compute_strategy_returns(predictions)

    assert returns.loc[0, "primary_only_gross_return"] == pytest.approx(0.02)
    assert returns.loc[0, "primary_only_total_cost"] == 0.0
    assert returns.loc[0, "primary_only_net_return"] == pytest.approx(0.02)
    assert returns.loc[0, "meta_filtered_gross_return"] == pytest.approx(0.01)
    assert returns.loc[0, "meta_filtered_total_cost"] == 0.0
    assert returns.loc[0, "meta_filtered_net_return"] == pytest.approx(0.01)
    assert returns.loc[1, "meta_filtered_net_return"] == 0.0


def test_performance_and_aum_statistics_use_series_means():
    pnl = pd.Series([1.0, -0.5, 2.0])
    aum = pd.Series([100.0, 200.0])

    assert Performance.pnl(pnl) == 2.5
    assert GeneralCharacteristics.average_aum(aum) == 150.0


def test_performance_short_pnl_uses_aligned_negative_positions():
    index = pd.Index(["a", "b", "c", "d"])
    pnl = pd.Series([10.0, -5.0, 2.0, 100.0], index=index)
    positions = pd.Series(
        [1.0, -1.0, 0.0, -1.0, -1.0],
        index=pd.Index(["a", "b", "c", "d", "unmatched"]),
    )

    assert Performance.pnl_from_short_positions(pnl, positions) == 95.0
    assert Performance.pnl_from_short_positions(
        pnl,
        pd.Series([1.0, 0.0], index=index[:2]),
    ) == 0.0


def test_annualized_rate_of_return_accepts_explicit_elapsed_time():
    returns = pd.Series(
        [0.10, 0.0],
        index=pd.date_range("2025-01-02", periods=2, tz="UTC"),
    )
    start = pd.Timestamp("2025-01-01", tz="UTC")
    end = pd.Timestamp("2025-01-04", tz="UTC")
    expected = 1.1 ** (365.25 / 3.0) - 1.0

    assert Performance.annualized_rate_of_return(
        returns,
        start=start,
        end=end,
    ) == pytest.approx(expected)
    assert Performance.annualized_rate_of_return(returns) == pytest.approx(
        1.1 ** 365.25 - 1.0
    )

    with pytest.raises(ValueError, match="provided together"):
        Performance.annualized_rate_of_return(returns, start=start)
    with pytest.raises(ValueError, match="must not precede"):
        Performance.annualized_rate_of_return(returns, start=end, end=start)


def test_implementation_shortfall_uses_split_costs_and_net_performance():
    turnover = pd.Series([10_000.0, 20_000.0])
    broker_fees = turnover * 0.0001
    slippage = turnover * 0.0001
    gross_performance = pd.Series([120.0, -20.0])
    execution_costs = broker_fees + slippage
    net_performance = gross_performance - execution_costs

    assert ImplementationShortfall.broker_fees_per_turnover(
        broker_fees, turnover,
    ) == pytest.approx(0.0001)
    assert ImplementationShortfall.average_slippage_per_turnover(
        slippage, turnover,
    ) == pytest.approx(0.0001)
    assert ImplementationShortfall.dollar_performance_per_turnover(
        net_performance, turnover,
    ) == pytest.approx(net_performance.sum() / turnover.sum())
    assert ImplementationShortfall.return_on_execution_costs(
        net_performance, execution_costs,
    ) == pytest.approx(net_performance.sum() / execution_costs.sum())
    assert np.isnan(ImplementationShortfall.return_on_execution_costs(
        net_performance, np.zeros(2),
    ))


def test_event_characteristics_use_full_range_and_active_bets():
    event_start = pd.DatetimeIndex([
        "2025-01-03 00:00:00+00:00",
        "2025-01-01 00:00:00+00:00",
        "2025-01-02 00:00:00+00:00",
    ])
    event_end = pd.Series(
        event_start + pd.to_timedelta([1, 2, 3], unit="D"),
        index=event_start,
    )
    positions = pd.Series([0.2, 0.0, 1.0], index=event_start)

    assert GeneralCharacteristics.start(event_start) == pd.Timestamp(
        "2025-01-01 00:00:00+00:00"
    )
    assert GeneralCharacteristics.end(event_start) == pd.Timestamp(
        "2025-01-03 00:00:00+00:00"
    )
    assert GeneralCharacteristics.end(event_start, event_end) == pd.Timestamp(
        "2025-01-05 00:00:00+00:00"
    )
    assert GeneralCharacteristics.frequency_of_bets(
        positions,
        event_end,
    ) == pytest.approx(2 / (4 / 365.25))
    assert GeneralCharacteristics.average_holding_period(
        positions,
        event_end,
    ) == pytest.approx(2.0)


def test_event_characteristics_handle_no_active_bets_and_zero_span():
    event_start = pd.DatetimeIndex(["2025-01-01 00:00:00+00:00"])
    event_end = pd.Series(event_start, index=event_start)
    active = pd.Series([1.0], index=event_start)
    flat = pd.Series([0.0], index=event_start)

    assert np.isnan(GeneralCharacteristics.frequency_of_bets(
        active,
        event_end,
    ))
    assert GeneralCharacteristics.frequency_of_bets(flat, event_end) == 0.0
    assert GeneralCharacteristics.average_holding_period(
        active,
        event_end,
    ) == 0.0
    assert np.isnan(GeneralCharacteristics.average_holding_period(
        flat,
        event_end,
    ))


@pytest.mark.parametrize(
    ("event_end", "message"),
    [
        (
            pd.Series(
                pd.to_datetime(["2025-01-02", "2025-01-03"], utc=True),
                index=pd.date_range("2025-02-01", periods=2, tz="UTC"),
            ),
            "same index",
        ),
        (
            pd.Series(
                [pd.Timestamp("2025-01-02", tz="UTC"), pd.NaT],
                index=pd.date_range("2025-01-01", periods=2, tz="UTC"),
            ),
            "missing",
        ),
        (
            pd.Series(
                pd.to_datetime(["2024-12-31", "2025-01-03"], utc=True),
                index=pd.date_range("2025-01-01", periods=2, tz="UTC"),
            ),
            "must not precede",
        ),
        (
            pd.Series(
                pd.to_datetime(["2025-01-02", "2025-01-03"]),
                index=pd.date_range("2025-01-01", periods=2, tz="UTC"),
            ),
            "same timezone",
        ),
    ],
)
def test_event_characteristics_validate_event_times(event_end, message):
    event_start = pd.date_range("2025-01-01", periods=2, tz="UTC")
    positions = pd.Series([1.0, 0.0], index=event_start)
    calls = [
        lambda: GeneralCharacteristics.end(event_start, event_end),
        lambda: GeneralCharacteristics.frequency_of_bets(positions, event_end),
        lambda: GeneralCharacteristics.average_holding_period(
            positions,
            event_end,
        ),
    ]

    for call in calls:
        with pytest.raises(ValueError, match=message):
            call()


def test_runs_percentiles_default_to_95_and_handle_no_drawdown():
    index = pd.date_range("2025-01-01", periods=5, tz="UTC")
    equity = pd.Series([100.0, 90.0, 95.0, 100.0, 80.0], index=index)
    increasing_equity = pd.Series(np.arange(5.0), index=index)

    assert Runs.percentile_drawdown(equity) == pytest.approx(
        Runs.drawdown(equity).quantile(0.95)
    )
    assert Runs.percentile_time_under_water(equity) == pytest.approx(
        Runs.time_under_water(equity).quantile(0.95)
    )
    assert np.isnan(Runs.percentile_drawdown(increasing_equity))
    assert np.isnan(Runs.percentile_time_under_water(increasing_equity))


def test_efficiency_sharpe_ratio_handles_constant_returns():
    event_start = pd.date_range("2025-01-01", periods=2, tz="UTC")
    returns = pd.Series([0.01, 0.01], index=event_start)
    event_end = pd.Series(event_start + pd.Timedelta(days=1), index=event_start)
    positions = pd.Series(1.0, index=event_start)

    assert pd.isna(Efficiency.sharpe_ratio(
        returns,
        event_end,
        positions,
        annual_risk_free_rate=0.0,
    ))


def test_efficiency_sharpe_ratios_use_event_period_risk_free_returns():
    event_start = pd.DatetimeIndex([
        "2025-01-01 00:00:00+00:00",
        "2025-02-01 00:00:00+00:00",
        "2025-03-01 00:00:00+00:00",
        "2025-04-01 00:00:00+00:00",
    ])
    returns = pd.Series([0.01, 0.02, 0.0, 0.015], index=event_start)
    positions = pd.Series([1.0, 0.5, 0.0, -0.5], index=event_start)
    holding_days = pd.Series([1.0, 7.0, 30.0, 365.25], index=event_start)
    event_end = event_start.to_series() + pd.to_timedelta(holding_days, unit="D")
    risk_free_returns = (1.03 ** (holding_days / 365.25)) - 1.0
    excess_returns = returns - positions.abs() * risk_free_returns
    expected_sharpe = excess_returns.mean() / excess_returns.std(ddof=1)

    sharpe = Efficiency.sharpe_ratio(returns, event_end, positions)
    annualized = Efficiency.annualized_sharpe_ratio(
        returns,
        event_end,
        positions,
        periods_per_year=12,
    )

    assert sharpe == pytest.approx(expected_sharpe)
    assert annualized == pytest.approx(expected_sharpe * np.sqrt(12))
    assert excess_returns.iloc[2] == 0.0
    assert Efficiency.sharpe_ratio(
        returns,
        event_end,
        positions,
        annual_risk_free_rate=0.0,
    ) != pytest.approx(sharpe)


def test_probabilistic_sharpe_uses_event_period_excess_returns():
    event_start = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    returns = pd.Series([0.01, -0.005, 0.02, 0.004, -0.002, 0.015], index=event_start)
    positions = pd.Series([1.0, 0.5, 0.0, -0.5, 1.0, 0.25], index=event_start)
    event_end = pd.Series(
        event_start + pd.to_timedelta([1, 2, 7, 14, 30, 60], unit="D"),
        index=event_start,
    )
    holding_years = (event_end - event_start.to_series()).dt.total_seconds() / (
        365.25 * 24 * 60 * 60
    )
    excess_returns = returns - positions.abs() * (1.03 ** holding_years - 1.0)
    assert Efficiency.probabilistic_sharpe_ratio(
        returns,
        event_end,
        positions,
    ) == pytest.approx(Efficiency.probabilistic_sharpe_ratio(
        excess_returns,
        event_end,
        positions,
        annual_risk_free_rate=0.0,
    ))


def test_probabilistic_sharpe_ratio_defaults_to_annualized_benchmark_one():
    event_start = pd.date_range("2025-01-01", periods=6, freq="D", tz="UTC")
    returns = pd.Series([0.01, -0.005, 0.02, 0.004, -0.002, 0.015], index=event_start)
    event_end = pd.Series(event_start + pd.Timedelta(days=1), index=event_start)
    positions = pd.Series(1.0, index=event_start)

    default = Efficiency.probabilistic_sharpe_ratio(
        returns,
        event_end,
        positions,
    )
    explicit = Efficiency.probabilistic_sharpe_ratio(
        returns,
        event_end,
        positions,
        annualized_benchmark_sharpe_ratio=1.0,
        periods_per_year=252,
    )
    zero_benchmark = Efficiency.probabilistic_sharpe_ratio(
        returns,
        event_end,
        positions,
        annualized_benchmark_sharpe_ratio=0.0,
        periods_per_year=252,
    )

    assert default == pytest.approx(explicit)
    assert default < zero_benchmark

    with pytest.raises(ValueError, match="periods_per_year"):
        Efficiency.probabilistic_sharpe_ratio(
            returns,
            event_end,
            positions,
            periods_per_year=0,
        )


@pytest.mark.parametrize("annual_risk_free_rate", [-1.0, np.nan, "invalid"])
def test_efficiency_sharpe_ratio_rejects_invalid_annual_rate(
        annual_risk_free_rate,
):
    event_start = pd.date_range("2025-01-01", periods=2, tz="UTC")
    returns = pd.Series([0.01, 0.02], index=event_start)
    event_end = pd.Series(event_start + pd.Timedelta(days=1), index=event_start)
    positions = pd.Series(1.0, index=event_start)

    with pytest.raises(ValueError, match="annual_risk_free_rate"):
        Efficiency.sharpe_ratio(
            returns,
            event_end,
            positions,
            annual_risk_free_rate=annual_risk_free_rate,
        )


def test_efficiency_sharpe_ratio_validates_event_times():
    event_start = pd.date_range("2025-01-01", periods=2, tz="UTC")
    returns = pd.Series([0.01, 0.02], index=event_start)
    valid_event_end = pd.Series(
        event_start + pd.Timedelta(days=1),
        index=event_start,
    )
    positions = pd.Series(1.0, index=event_start)

    with pytest.raises(ValueError, match="pandas Series"):
        Efficiency.sharpe_ratio(returns.to_numpy(), valid_event_end, positions)
    with pytest.raises(ValueError, match="DatetimeIndex"):
        Efficiency.sharpe_ratio(
            returns.reset_index(drop=True),
            valid_event_end,
            positions,
        )
    with pytest.raises(ValueError, match="same index"):
        Efficiency.sharpe_ratio(
            returns,
            valid_event_end.reset_index(drop=True),
            positions,
        )
    with pytest.raises(ValueError, match="contain datetimes"):
        Efficiency.sharpe_ratio(
            returns,
            pd.Series([1, 2], index=event_start),
            positions,
        )
    with pytest.raises(ValueError, match="missing"):
        Efficiency.sharpe_ratio(
            returns,
            valid_event_end.mask([False, True]),
            positions,
        )
    with pytest.raises(ValueError, match="must not precede"):
        Efficiency.sharpe_ratio(
            returns,
            pd.Series(event_start - pd.Timedelta(seconds=1), index=event_start),
            positions,
        )


def test_efficiency_sharpe_ratio_validates_positions():
    event_start = pd.date_range("2025-01-01", periods=2, tz="UTC")
    returns = pd.Series([0.01, 0.02], index=event_start)
    event_end = pd.Series(event_start + pd.Timedelta(days=1), index=event_start)
    invalid_cases = [
        (np.array([1.0, 0.0]), "pandas Series"),
        (pd.Series([1.0, 0.0]), "same index"),
        (pd.Series([1.0, np.nan], index=event_start), "finite values"),
        (pd.Series([1.0, np.inf], index=event_start), "finite values"),
        (pd.Series([1.0, 1.1], index=event_start), "in \\[-1, 1\\]"),
        (pd.Series([True, False], index=event_start), "numeric values"),
    ]

    for positions, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            Efficiency.sharpe_ratio(returns, event_end, positions)


def test_classification_scores_support_weights_and_positive_label():
    y_true = pd.Series([-1, -1, 1, 1])
    y_pred = pd.Series([-1, 1, 1, -1])
    probabilities = np.array([
        [0.9, 0.1],
        [0.4, 0.6],
        [0.2, 0.8],
        [0.7, 0.3],
    ])
    sample_weight = pd.Series([4.0, 1.0, 2.0, 1.0])

    assert ClassificationScores.accuracy(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    ) == pytest.approx(accuracy_score(
        y_true,
        y_pred,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.precision(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(precision_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.recall(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(recall_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.f1_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ) == pytest.approx(f1_score(
        y_true,
        y_pred,
        pos_label=-1,
        sample_weight=sample_weight,
    ))
    assert ClassificationScores.negative_log_loss(
        y_true,
        probabilities,
        labels=[-1, 1],
        sample_weight=sample_weight,
    ) == pytest.approx(-log_loss(
        y_true,
        probabilities,
        labels=[-1, 1],
        sample_weight=sample_weight,
    ))
