import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_statistics import Efficiency, GeneralCharacteristics, Runs
from src.backtesting.portfolio import (
    daily_portfolio,
    portfolio_equity,
    portfolio_trades,
    simulate_portfolio,
    validate_event_prices,
)


@pytest.mark.parametrize("side, final_aum", [(1, 1100.0), (-1, 900.0)])
def test_long_and_short_cash_accounting(side, final_aum):
    times = pd.date_range("2025-01-01", periods=3, tz="UTC")
    prices = pd.Series([100.0, 120.0, 110.0], index=times)
    targets = pd.Series([side, 0.0], index=times[[0, 2]])
    ledger = simulate_portfolio(prices, targets, initial_aum=1000)
    assert ledger.quantity.tolist() == [side * 10, side * 10, 0]
    assert ledger.aum.iloc[-1] == pytest.approx(final_aum)
    np.testing.assert_allclose(ledger.aum, ledger.cash + ledger.quantity * prices)
    trades = portfolio_trades(ledger)
    assert len(trades) == 1
    assert trades.net_pnl.sum() == pytest.approx(final_aum - 1000)


def test_same_target_rebalances_after_price_drift_and_reinvests():
    times = pd.date_range("2025-01-01", periods=4, tz="UTC")
    prices = pd.Series([100.0, 120.0, 100.0, 110.0], index=times)
    targets = pd.Series([0.5, 0.5, 0.5, 0.0], index=times)
    ledger = simulate_portfolio(prices, targets, initial_aum=1000)
    assert ledger.quantity.iloc[1] == pytest.approx(550 / 120)
    assert ledger.traded_value.iloc[1] == pytest.approx(50)
    assert ledger.position_value.iloc[2] == pytest.approx(ledger.aum.iloc[2] / 2)
    assert len(portfolio_trades(ledger)) == 1


def test_costs_and_flip_allocation_reconcile_with_equity():
    times = pd.date_range("2025-01-01", periods=4, tz="UTC")
    prices = pd.Series([100.0, 110.0, 90.0, 95.0], index=times)
    targets = pd.Series([1.0, -1.0, -0.5, 0.0], index=times)
    ledger = simulate_portfolio(
        prices,
        targets,
        initial_aum=1000,
        broker_fee_bps=40,
        slippage_bps=60,
    )
    assert ledger.quantity.iloc[0] == pytest.approx(1000 / 1.01 / 100)
    assert ledger.cash.iloc[0] == pytest.approx(0, abs=1e-10)
    np.testing.assert_allclose(
        ledger.position_value / ledger.aum, targets, atol=1e-12,
    )
    np.testing.assert_allclose(ledger.broker_fee, ledger.traded_value * 0.004)
    np.testing.assert_allclose(ledger.slippage_cost, ledger.traded_value * 0.006)
    np.testing.assert_allclose(ledger.execution_cost, ledger.traded_value * 0.01)
    np.testing.assert_allclose(
        ledger.execution_cost,
        ledger.broker_fee + ledger.slippage_cost,
    )
    trades = portfolio_trades(ledger)
    assert trades.side.tolist() == [1.0, -1.0]
    assert trades.execution_cost.sum() == pytest.approx(ledger.execution_cost.sum())
    assert trades.net_pnl.sum() == pytest.approx(ledger.aum.iloc[-1] - 1000)
    assert ledger.aum.iloc[-1] - 1000 == pytest.approx(
        ledger.gross_pnl.sum() - ledger.execution_cost.sum()
    )


def test_daily_marks_carry_weekends_and_include_initial_cost():
    times = pd.DatetimeIndex(["2025-01-03 12:00Z", "2025-01-06 12:00Z"])
    ledger = simulate_portfolio(
        pd.Series([100.0, 110.0], index=times),
        pd.Series([1.0, 0.0], index=times),
        initial_aum=1000, broker_fee_bps=4, slippage_bps=6,
    )
    daily = daily_portfolio(ledger)
    assert len(daily) == 4
    assert daily.net_return.iloc[0] < 0
    assert daily.net_return.iloc[1:3].eq(0).all()
    assert daily.period_start.iloc[0] == times[0]
    assert daily.index[-1] == times[-1]
    assert (1 + daily.net_return).prod() == pytest.approx(ledger.aum.iloc[-1] / 1000)
    assert (1 + daily.underlying_return).prod() == pytest.approx(1.1)
    assert Runs.drawdown(portfolio_equity(ledger)).max() >= 1 - 1 / 1.001


def test_one_basis_point_cost_components_apply_to_every_traded_dollar():
    times = pd.date_range("2025-01-01", periods=4, tz="UTC")
    prices = pd.Series([100.0, 110.0, 90.0, 95.0], index=times)
    targets = pd.Series([1.0, -1.0, -0.5, 0.0], index=times)

    ledger = simulate_portfolio(
        prices,
        targets,
        initial_aum=10_000,
        broker_fee_bps=1,
        slippage_bps=1,
    )

    np.testing.assert_allclose(ledger.broker_fee, ledger.traded_value * 0.0001)
    np.testing.assert_allclose(ledger.slippage_cost, ledger.traded_value * 0.0001)
    np.testing.assert_allclose(
        ledger.execution_cost,
        ledger.broker_fee + ledger.slippage_cost,
    )
    assert ledger.aum.iloc[-1] - 10_000 == pytest.approx(
        ledger.gross_pnl.sum() - ledger.execution_cost.sum()
    )


@pytest.mark.parametrize(
    ("broker_fee_bps", "slippage_bps", "message"),
    [
        (-1, 0, "broker_fee_bps"),
        (0, np.nan, "slippage_bps"),
        (6_000, 4_000, "Total one-way execution cost"),
    ],
)
def test_portfolio_rejects_invalid_execution_costs(
        broker_fee_bps, slippage_bps, message,
):
    times = pd.date_range("2025-01-01", periods=2, tz="UTC")
    prices = pd.Series([100.0, 110.0], index=times)
    targets = pd.Series([1.0, 0.0], index=times)

    with pytest.raises(ValueError, match=message):
        simulate_portfolio(
            prices,
            targets,
            broker_fee_bps=broker_fee_bps,
            slippage_bps=slippage_bps,
        )


def test_full_span_frequency_turnover_and_daily_aum():
    times = pd.date_range("2025-01-01", periods=3, tz="UTC")
    amounts = pd.Series([100.0, 0.0, 120.0], index=times)
    aum = pd.Series([100.0, 120.0], index=times[1:])
    assert GeneralCharacteristics.average_aum(aum) == 110
    assert GeneralCharacteristics.leverage(
        pd.Series([100.0, 0.0], index=times[1:]), aum,
    ) == pytest.approx(50 / 110)
    assert GeneralCharacteristics.annualized_turnover(
        amounts, aum, time_range=(times[0], times[-1]),
    ) == pytest.approx(365.25)
    assert GeneralCharacteristics.annualized_turnover(amounts, aum) == pytest.approx(365.25)
    assert GeneralCharacteristics.maximum_dollar_position_size(amounts) == 120
    assert GeneralCharacteristics.frequency_of_bets(
        pd.Series([1.0], index=times[1:2]),
        pd.Series(times[2:], index=times[1:2]),
        time_range=(times[0], times[-1]),
    ) == pytest.approx(365.25 / 2)


def test_portfolio_sharpe_uses_full_capital_and_actual_intervals():
    ends = pd.date_range("2025-01-02", periods=4, tz="UTC")
    starts = pd.Series(ends - pd.Timedelta(days=1), index=ends)
    starts.iloc[0] = ends[0] - pd.Timedelta(hours=12)
    returns = pd.Series([0.0, 0.01, -0.02, 0.03], index=ends)
    years = (ends.to_series() - starts).dt.total_seconds() / (365.25 * 86400)
    excess = returns - (1.03 ** years - 1)
    expected = excess.mean() / excess.std(ddof=1)
    result = Efficiency.portfolio_statistics(returns, starts)
    assert result["sharpe_ratio"] == pytest.approx(expected)
    assert result["annualized_sharpe"] == pytest.approx(expected * np.sqrt(365.25))
    equivalent = Efficiency.portfolio_statistics(excess, starts, annual_risk_free_rate=0)
    assert result["probabilistic_sharpe_ratio"] == pytest.approx(
        equivalent["probabilistic_sharpe_ratio"]
    )


def test_drawdown_recovered_and_unrecovered_episodes():
    times = pd.date_range("2025-01-01", periods=6, tz="UTC")
    equity = pd.Series([100, 90, 100, 110, 88, 99], index=times)
    np.testing.assert_allclose(Runs.drawdown(equity), [0.1, 0.2])
    np.testing.assert_allclose(Runs.time_under_water(equity), [2 / 365.25, 2 / 365.25])
    assert Runs.drawdown(pd.Series([100, 110], index=times[:2])).empty


def test_flat_account_and_no_lookahead():
    times = pd.date_range("2025-01-01", periods=3, tz="UTC")
    prices = pd.Series([100.0, 105.0, 110.0], index=times)
    flat = simulate_portfolio(prices, pd.Series([0.0, 0.0], index=times[[0, 2]]))
    assert flat.aum.eq(100_000).all()
    assert portfolio_trades(flat).empty
    target = pd.Series([0.5, 0.5, 0.0], index=times)
    before = simulate_portfolio(prices, target)
    prices.iloc[-1] = 120
    after = simulate_portfolio(prices, target)
    pd.testing.assert_frame_equal(before.iloc[:-1], after.iloc[:-1])


def test_exact_event_prices_and_invalid_simulation_inputs():
    times = pd.date_range("2025-01-01", periods=3, tz="UTC")
    prices = pd.Series([100.0, 105.0, 110.0], index=times)
    events = pd.DataFrame({
        "partition": ["holdout"], "event_end": [times[-1]], "raw_return": [0.1],
    }, index=times[:1])
    validate_event_prices(events, prices)
    with pytest.raises(ValueError, match="exact"):
        validate_event_prices(events, prices.iloc[1:])
    events["raw_return"] = 0.2
    with pytest.raises(ValueError, match="reproduce"):
        validate_event_prices(events, prices)
    with pytest.raises(ValueError, match="exact"):
        simulate_portfolio(prices.iloc[1:], pd.Series([1, 0], index=times[[0, 2]]))
    with pytest.raises(ValueError, match="positive"):
        simulate_portfolio(
            pd.Series([100.0, 250.0, 110.0], index=times),
            pd.Series([-1, 0], index=times[[0, 2]]),
        )
