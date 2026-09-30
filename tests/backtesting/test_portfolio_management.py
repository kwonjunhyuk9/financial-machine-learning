import numpy as np
import pandas as pd
import pytest

from src.backtesting.backtest_statistics import Efficiency, GeneralCharacteristics, Runs
from src.backtesting.portfolio_management import (
    PortfolioSettings,
    benchmark_returns,
    candidate_snapshot,
    daily_portfolio,
    elapsed_session_minutes,
    portfolio_equity,
    portfolio_trades,
    simulate_portfolio,
    simulate_cross_sectional,
    validate_event_prices,
)
from src.preprocessing.market_data import ResearchPaths


def test_benchmark_returns_uses_last_spy_trade_each_day(tmp_path):
    paths = ResearchPaths(tmp_path)
    directory = paths.raw("SPY", "tick")
    directory.mkdir(parents=True)
    pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-02-03T20:00Z",
                    "2025-02-03T20:59Z",
                    "2025-02-04T20:59Z",
                ]
            ),
            "price": [99.0, 100.0, 110.0],
        }
    ).to_parquet(directory / "2025-02-03.parquet", index=False)

    result = benchmark_returns(
        paths,
        pd.Timestamp("2025-02-03T20:30Z"),
        pd.Timestamp("2025-02-04T21:00Z"),
    )

    assert result.tolist() == pytest.approx([0.0, 0.1])


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


def cross_sectional_scenario():
    start = pd.Timestamp("2025-01-02 14:30Z")
    calendar = pd.DataFrame({"open": [start], "close": [start + pd.Timedelta(minutes=10)]})
    symbols = [f"S{i:02}" for i in range(25)]
    events = pd.DataFrame({"symbol": symbols, "event_start": start,
        "vertical_barrier": start + pd.Timedelta(minutes=8), "target_return": .5,
        "primary_side": [1] * 5 + [-1] * 20, "meta_action": 1,
        "meta_probability": 1., "mean_sentiment_score": np.arange(25, 0, -1),
        "partition": "holdout", "event_end": start})
    calibration = pd.DataFrame({"symbol": symbols, "w": 1.0})

    def observations(price=100.):
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": symbols, "price": price,
            })

    return start, calendar, events, calibration, observations


def test_account_uses_subsequent_quotes_caps_and_final_liquidation():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    trades = result["trades"]
    assert trades.timestamp.min() > start
    assert set(trades.symbol).issubset(set(events.symbol.head(5)) | set(events.symbol.tail(5)))
    exposure = result["exposures"]
    assert exposure.groupby("timestamp").apply(lambda group: group.weight.gt(0).sum()).max() <= 5
    assert exposure.groupby("timestamp").apply(lambda group: group.weight.lt(0).sum()).max() <= 5
    ledger = result["ledger"]
    assert ledger.iloc[-1].gross_exposure == 0
    assert ledger.iloc[-1].aum == pytest.approx(
        100_000 - trades.broker_fee.sum() - trades.slippage.sum()
    )
    events["event_end"] = start + pd.Timedelta(days=200)
    again = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    pd.testing.assert_frame_equal(trades, again["trades"])


def test_development_events_do_not_enter_holdout_average():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    development = events.copy()
    development["partition"] = "development"
    development["primary_side"] *= -1
    base = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    result = simulate_cross_sectional(pd.concat([development, events]), observations(), calibration,
                                      calendar, calendar.close.iloc[0])
    pd.testing.assert_frame_equal(base["trades"], result["trades"])


def test_minute_half_life_excludes_overnight_and_early_close():
    calendar = pd.DataFrame({"open": pd.to_datetime(["2025-01-02 14:30Z", "2025-01-03 14:30Z"]),
                             "close": pd.to_datetime(["2025-01-02 18:00Z", "2025-01-03 21:00Z"])})
    assert elapsed_session_minutes(calendar.open.iloc[0],
        calendar.open.iloc[1] + pd.Timedelta(minutes=180), calendar) == 390


def test_no_tail_of_five_means_no_new_entries():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    active = events.head(24).assign(entry_price=100.)
    table = candidate_snapshot(active, dict.fromkeys(active.symbol, 100.),
        calibration.set_index("symbol").w, start, calendar, PortfolioSettings())
    assert not table.eligible.any()


def test_missing_final_quotes_are_not_fabricated():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    with pytest.raises(ValueError, match="unliquidated"):
        simulate_cross_sectional(events, list(observations())[:3], calibration,
                                 calendar, calendar.close.iloc[0])


def test_pending_entry_that_violates_limit_never_fills():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    events.loc[events.primary_side.eq(-1), "meta_action"] = 0

    def observations():
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol, "price": 100. if minute == 0 else 200.,
            })

    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    assert result["trades"].empty
    assert result["ledger"].iloc[-1].aum == 100_000


def test_observed_bar_count_not_future_stored_expiry_controls_horizon():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    events["horizon_bars"] = 2
    events["vertical_barrier"] = start + pd.Timedelta(days=99)

    def marked():
        for time, quotes in observations():
            yield time, pd.concat([quotes.assign(bar_end=False),
                                   quotes.assign(price=np.nan, bar_end=True)], ignore_index=True)

    result = simulate_cross_sectional(events, marked(), calibration, calendar, calendar.close.iloc[0])
    exits = result["trades"].loc[result["trades"].reason.eq("signal_reduction")]
    assert not exits.empty
    assert exits.timestamp.min() == start + pd.Timedelta(minutes=3)


def test_missing_calibration_excludes_trading_but_reports_symbol():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    calibration.loc[calibration.symbol.eq("S00"), "w"] = np.nan
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    assert "S00" in set(result["exclusions"].symbol)
    assert "S00" not in set(result["trades"].symbol)


def test_replacement_waits_for_both_legs_after_decision():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    events.loc[events.symbol.eq("S05"), ["meta_action", "primary_side"]] = [0, 1]
    newcomer = events.loc[events.symbol.eq("S05")].copy()
    newcomer["event_start"] = start + pd.Timedelta(minutes=2)
    newcomer["meta_action"] = 1
    newcomer["mean_sentiment_score"] = 100
    events = pd.concat([events, newcomer], ignore_index=True)

    def observations():
        symbols = list(calibration.symbol)
        for minute in range(11):
            available = ["S05"] if minute == 3 else symbols
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": available, "price": 100.,
            })

    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0])
    replacement = result["trades"].loc[result["trades"].reason.str.startswith("replacement")]
    assert not replacement.empty
    assert replacement.timestamp.min() == start + pd.Timedelta(minutes=4)
    assert set(replacement.loc[
        replacement.timestamp.eq(replacement.timestamp.min()), "reason"
    ]) == {"replacement_entry", "replacement_exit"}
