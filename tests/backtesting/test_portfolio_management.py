import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


import src.backtesting.portfolio_management as portfolio_management
from src.backtesting.bet_sizing import event_bet_signals
from src.backtesting import backtesting_workflow as workflow
from src.backtesting.backtesting_workflow import _position_values, holding_statistics, summarize_account
from src.backtesting.portfolio_management import (
    BarStream,
    CandidateBook,
    PortfolioSettings,
    SessionClock,
    SynchronousPriceIndex,
    SynchronousScheduler,
    _weight_to_position,
    benchmark_returns,
    candidate_snapshot,
    elapsed_session_minutes,
    entry_schedule,
    load_event_entry_prices,
    load_bar_prices,
    select_synchronous,
    simulate_cross_sectional,
    simulate_synchronous,
)
from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions,
    combinatorial_purged_cross_validation,
)
from src.preprocessing.market_data import ResearchPaths


@pytest.mark.parametrize("benchmark_symbol", ["SPY", "QQQ"])
def test_benchmark_returns_uses_last_selected_symbol_bar_each_day(tmp_path, benchmark_symbol):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    file = paths.feature(benchmark_symbol, "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": pd.to_datetime(["2025-02-03T20:00Z", "2025-02-03T20:59Z", "2025-02-04T20:59Z"]),
                  "close": [99., 100., 110.]}).to_parquet(file, index=False)

    result = benchmark_returns(
        paths,
        pd.Timestamp("2025-02-03T20:30Z"),
        pd.Timestamp("2025-02-04T21:00Z"),
        data_start=pd.Timestamp("2025-01-01", tz="UTC"), benchmark_symbol=benchmark_symbol,
    )

    assert result.tolist() == pytest.approx([0.0, 0.1])


def test_load_event_entry_prices_preserves_composite_index(tmp_path):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    starts = pd.date_range("2025-02-03T14:30Z", periods=2, freq="min")
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": starts, "close": [100.0, 101.0]}).to_parquet(file)
    events = pd.DataFrame(
        {"horizon_volatility": [0.01, 0.02]},
        index=pd.MultiIndex.from_arrays(
            [["A", "A"], starts], names=["symbol", "event_start"]
        ),
    )

    result = load_event_entry_prices(paths, events)

    assert result.index.equals(events.index)
    assert result["entry_price"].tolist() == [100.0, 101.0]


@pytest.mark.parametrize("risk_free, annual_periods, benchmark_sharpe, benchmark_symbol", [
    (.03, 365.25, 1.0, "SPY"), (.07, 252., .5, "QQQ"),
])
def test_summarize_account_reports_current_result_contract(risk_free, annual_periods, benchmark_sharpe, benchmark_symbol):
    timestamps = pd.date_range("2025-01-01", periods=2, tz="UTC")
    result = {
        "ledger": pd.DataFrame({
            "timestamp": timestamps,
            "aum": [100.0, 110.0],
            "gross_exposure": [0.5, 0.0],
            "net_exposure": [0.5, 0.0],
            "broker_fee": [0.5, 1.0],
            "slippage": [0.5, 1.0],
            "traded_value": [25.0, 50.0],
        }),
        "closed_trades": pd.DataFrame({
            "event_start": [timestamps[0]],
            "event_end": [timestamps[1]],
            "timestamp": [timestamps[1]],
            "symbol": ["AAPL"],
            "side": [1.0],
            "entry_aum": [100.0],
            "gross_pnl": [12.0],
            "execution_cost": [2.0],
            "net_pnl": [10.0],
            "net_return": [0.10],
        }),
        "exposures": pd.DataFrame(),
        "exclusions": pd.DataFrame({"symbol": ["MISSING"]}),
    }
    settings = PortfolioSettings(initial_aum=100.0)
    benchmark_series = pd.Series([0.0, 0.05], index=timestamps)

    statistics = summarize_account(result, settings, benchmark_series, benchmark_symbol=benchmark_symbol, annual_risk_free_rate=risk_free, periods_per_year=annual_periods, annualized_benchmark_sharpe_ratio=benchmark_sharpe)
    values = statistics.set_index(["section", "metric"])["value"]

    assert list(statistics.columns) == [
        "section", "metric", "entity", "value", "timestamp", "unit",
    ]
    assert {
        "general_characteristics", "performance", "runs",
        "implementation_shortfall", "efficiency", "benchmark",
        "assumption", "coverage",
    }.issubset(set(statistics["section"]))
    assert values["performance", "pnl"] == pytest.approx(10.0)
    assert values["general_characteristics", "annualized_turnover"] == pytest.approx(
        50.0 * 365.25 / 105.0
    )
    assert np.isnan(values["runs", "percentile_drawdown"])
    assert values["implementation_shortfall", "broker_fees_per_turnover"] == pytest.approx(0.02)
    assert values["implementation_shortfall", "return_on_execution_costs"] == pytest.approx(5.0)
    assert values["benchmark", "net_return"] == pytest.approx(0.05)
    assert statistics.loc[statistics.section.eq("benchmark"), "entity"].tolist() == [benchmark_symbol]
    assert values["coverage", "excluded_price_calibrations"] == 1
    assert values["coverage", "closed_positions"] == 1


    daily = workflow._daily_account(result["ledger"].set_index("timestamp"), settings.initial_aum)
    expected = workflow.Efficiency.portfolio_statistics(
        daily.net_return, daily.period_start, annual_risk_free_rate=risk_free,
        periods_per_year=annual_periods, annualized_benchmark_sharpe_ratio=benchmark_sharpe,
    )
    assert values["efficiency", "annualized_sharpe_ratio"] == pytest.approx(expected["annualized_sharpe"])


def cross_sectional_scenario():
    start = pd.Timestamp("2025-01-02 14:30Z")
    calendar = pd.DataFrame({"open": [start], "close": [start + pd.Timedelta(minutes=10)]})
    symbols = [f"S{i:02}" for i in range(25)]
    events = pd.DataFrame({"symbol": symbols, "event_start": start,
        "vertical_barrier": start + pd.Timedelta(minutes=8), "horizon_volatility": .5,
        "primary_side": [1] * 5 + [-1] * 20, "meta_action": 1,
        "meta_probability": 1., "mean_sentiment_score": np.arange(25, 0, -1),
        "partition": "holdout", "event_end": start})
    calibration = pd.DataFrame({
        "symbol": symbols,
        "event_start": start,
        "w": 1.0,
    })

    def observations(price=100.):
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": symbols, "price": price, "is_last": minute == 10,
            })

    return start, calendar, events, calibration, observations


def add_event(events, calibration, start, symbol, probability):
    newcomer = events.loc[events.symbol.eq(symbol)].copy()
    newcomer["event_start"] = start + pd.Timedelta(minutes=2)
    newcomer["meta_probability"] = probability
    newcomer["mean_sentiment_score"] = 100 if newcomer.primary_side.item() > 0 else -100
    events = pd.concat([events, newcomer], ignore_index=True)
    calibration = pd.concat([
        calibration,
        pd.DataFrame({"symbol": [symbol], "event_start": [newcomer.event_start.item()], "w": [1.0]}),
    ], ignore_index=True)
    return events, calibration


def test_weight_to_position_maps_capped_weights():
    weights = [0.0, 0.04, -0.07, 0.10, -0.10]
    assert [_weight_to_position(weight, max_positions_per_side=5) for weight in weights] == [0, 40, -70, 99, -99]


@pytest.mark.parametrize(
    ("side", "resize_limit", "expected_entries"),
    [(1, 200.0, 2), (1, 0.0, 1), (-1, 0.0, 2), (-1, 200.0, 1)],
)
def test_same_symbol_increase_uses_incremental_limit(
    monkeypatch, side, resize_limit, expected_entries
):
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    symbol = "S00" if side > 0 else "S05"
    events.loc[events.symbol.eq(symbol), "meta_probability"] = 0.55
    events, calibration = add_event(events, calibration, start, symbol, 1.0)

    original_limit_price = portfolio_management.limit_price
    position_changes = []

    def tracked_limit_price(target, current, forecast, w, maximum):
        position_changes.append((target, current))
        if current:
            return resize_limit
        return original_limit_price(target, current, forecast, w, maximum)

    monkeypatch.setattr(portfolio_management, "limit_price", tracked_limit_price)
    result = simulate_cross_sectional(
        events, observations(), calibration, calendar, calendar.close.iloc[0],
        settings=PortfolioSettings(max_positions_per_side=5),
    )

    assert (side * 50, side * 10) in position_changes
    entries = result["trades"].loc[
        result["trades"].symbol.eq(symbol)
        & result["trades"].reason.eq("signal_entry")
    ]
    assert len(entries) == expected_entries


def test_same_symbol_weaker_event_reduces_without_limit():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    events, calibration = add_event(events, calibration, start, "S00", 0.55)

    result = simulate_cross_sectional(
        events, observations(), calibration, calendar, calendar.close.iloc[0],
        settings=PortfolioSettings(max_positions_per_side=5),
    )

    reductions = result["trades"].loc[
        result["trades"].symbol.eq("S00")
        & result["trades"].reason.eq("signal_reduction")
    ]
    assert reductions.timestamp.tolist() == [start + pd.Timedelta(minutes=3), start + pd.Timedelta(minutes=9)]


def test_price_changes_alone_do_not_resize_positions():
    start, calendar, events, calibration, _ = cross_sectional_scenario()

    def observations():
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol, "is_last": minute == 10,
                "price": 100.0 + minute / 10,
            })

    result = simulate_cross_sectional(
        events, observations(), calibration, calendar, calendar.close.iloc[0],
        settings=PortfolioSettings(max_positions_per_side=5),
    )

    assert not result["trades"].loc[result["trades"].timestamp.lt(start + pd.Timedelta(minutes=8))].reason.eq("signal_reduction").any()
    entries = result["trades"].loc[result["trades"].reason.eq("signal_entry")]
    assert not entries.duplicated("symbol").any()


def test_account_uses_subsequent_quotes_caps_and_final_liquidation():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    trades = result["trades"]
    assert trades.timestamp.min() > start
    assert set(trades.symbol) == set(events.symbol.head(10))
    exposure = result["exposures"]
    assert exposure.groupby("timestamp").apply(lambda group: group.weight.gt(0).sum()).max() <= 5
    assert exposure.groupby("timestamp").apply(lambda group: group.weight.lt(0).sum()).max() <= 5
    ledger = result["ledger"]
    assert ledger.iloc[-1].gross_exposure == 0
    assert ledger.iloc[-1].aum == pytest.approx(
        100_000 - trades.broker_fee.sum() - trades.slippage.sum()
    )
    closed = result["closed_trades"]
    assert set(closed.columns) == {
        "event_start", "event_end", "timestamp", "symbol", "side", "entry_aum",
        "gross_pnl", "execution_cost", "net_pnl", "net_return",
    }
    assert (closed.event_end > closed.event_start).all()
    assert closed.net_pnl.sum() == pytest.approx(ledger.iloc[-1].aum - 100_000)
    assert closed.execution_cost.sum() == pytest.approx(
        trades.broker_fee.sum() + trades.slippage.sum()
    )
    assert closed.gross_pnl.sum() == pytest.approx(
        closed.net_pnl.sum() + closed.execution_cost.sum()
    )
    events["event_end"] = start + pd.Timedelta(days=200)
    again = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    pd.testing.assert_frame_equal(trades, again["trades"])


def test_development_events_do_not_enter_holdout_average():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    development = events.copy()
    development["partition"] = "development"
    development["primary_side"] *= -1
    base = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    result = simulate_cross_sectional(pd.concat([development, events]), observations(), calibration,
                                      calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    pd.testing.assert_frame_equal(base["trades"], result["trades"])


def test_development_partition_runs_full_account_statistics():
    start, calendar, events, _, observations = cross_sectional_scenario()
    development = events.copy()
    development["partition"] = "development"
    later = development.copy()
    later["event_start"] = start + pd.Timedelta(minutes=2)
    later["event_end"] = later["event_start"]
    development = pd.concat([development, later], ignore_index=True).sort_values(
        ["event_start", "symbol"]
    )
    indexed = development.set_index(["symbol", "event_start"])
    information_sets = pd.Series(indexed["event_end"].to_numpy(), index=indexed.index)
    splits = combinatorial_purged_cross_validation(information_sets, 2, 1)
    predictions = []
    for split in splits.itertuples():
        frame = indexed.iloc[list(split.test_indices)].copy()
        frame["split_num"] = split.split_num
        frame["observation_position"] = list(split.test_indices)
        predictions.append(frame)
    path = assemble_cpcv_path_predictions(
        pd.concat(predictions), splits, indexed.index, 2
    )["path_0"].reset_index()
    calibration = path[["symbol", "event_start"]].assign(w=1.0)

    result = simulate_cross_sectional(
        path,
        observations(),
        calibration,
        calendar,
        calendar.close.iloc[0],
        evaluation_partition="development",
        settings=PortfolioSettings(max_positions_per_side=5),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        statistics = summarize_account(
            result,
            PortfolioSettings(max_positions_per_side=5),
            pd.Series([0.0], index=pd.DatetimeIndex([start])),
            benchmark_symbol="SPY", annual_risk_free_rate=0.03, periods_per_year=365.25, annualized_benchmark_sharpe_ratio=1.0,
        )

    assert not result["ledger"].empty
    assert (
        (statistics["section"] == "efficiency")
        & (statistics["metric"] == "annualized_sharpe_ratio")
    ).any()


def test_minute_half_life_excludes_overnight_and_early_close():
    calendar = pd.DataFrame({"open": pd.to_datetime(["2025-01-02 14:30Z", "2025-01-03 14:30Z"]),
                             "close": pd.to_datetime(["2025-01-02 18:00Z", "2025-01-03 21:00Z"])})
    assert elapsed_session_minutes(calendar.open.iloc[0],
        calendar.open.iloc[1] + pd.Timedelta(minutes=180), calendar) == 390


@pytest.mark.parametrize("count", [1, 49])
def test_candidates_enter_without_minimum_count_or_sentiment_filter(count):
    start, calendar, events, _, _ = cross_sectional_scenario()
    active = pd.concat([events.iloc[[0]]] * count, ignore_index=True)
    active["symbol"] = [f"S{i:02}" for i in range(count)]
    active["entry_price"] = 100.
    active["mean_sentiment_score"] = np.arange(count)
    calibration = active[["symbol", "event_start"]].assign(w=1.)
    settings = PortfolioSettings(max_positions_per_side=10)
    quotes = dict.fromkeys(active.symbol, 100.)
    table = candidate_snapshot(active, quotes,
        calibration.set_index(["symbol", "event_start"]).w, start, calendar, settings)
    assert table.eligible.all()
    assert (table.target_weight == .05).all()
    reversed_sentiment = candidate_snapshot(active.assign(mean_sentiment_score=-active.mean_sentiment_score),
        quotes, calibration.set_index(["symbol", "event_start"]).w, start, calendar, settings)
    pd.testing.assert_frame_equal(table, reversed_sentiment)
    observations = [(start + pd.Timedelta(minutes=i),
                     pd.DataFrame({"symbol": active.symbol, "price": 100., "is_last": i == 10})) for i in range(11)]
    result = simulate_cross_sectional(active, observations, calibration, calendar,
        calendar.close.iloc[0], settings)
    entered = result["trades"].query("reason == 'signal_entry'")
    assert set(entered.symbol) == set(sorted(active.symbol)[:10])


@pytest.mark.parametrize("action, probability, w", [(0, 1., 1.), (1, .5, 1.), (1, 1., np.nan)])
def test_unfiltered_candidates_still_require_nonzero_signal_and_calibration(action, probability, w):
    start, calendar, events, _, _ = cross_sectional_scenario()
    active = events.head(1).assign(entry_price=100., meta_action=action, meta_probability=probability)
    calibration = active.set_index(["symbol", "event_start"]).assign(w=w).w
    table = candidate_snapshot(active, {"S00": 100.}, calibration, start, calendar, PortfolioSettings())
    assert not table.eligible.any()


@pytest.mark.parametrize("side", [1, -1])
def test_pending_new_entries_reserve_slots_until_multiple_liquidations(side):
    start, calendar, events, _, _ = cross_sectional_scenario()
    events = events.head(4).copy().assign(primary_side=side)
    events.loc[events.index[:2], "vertical_barrier"] = start + pd.Timedelta(minutes=3)
    calibration = events[["symbol", "event_start"]].assign(w=1.)
    account = portfolio_management._CrossSectionalAccount(
        events, calibration, calendar, calendar.close.iloc[0],
        PortfolioSettings(max_positions_per_side=2), "holdout")
    account.observe(start, dict.fromkeys(events.symbol, 100.), set())
    assert set(account.pending) == {"S00", "S01"}
    account.observe(start + pd.Timedelta(seconds=1), {"S02": 100., "S03": 100.}, set())
    assert not account.fills
    account.observe(start + pd.Timedelta(minutes=1), {"S00": 100.}, set())
    assert set(account.pending) == {"S01"}
    account.observe(start + pd.Timedelta(minutes=2), {"S01": 100.}, set())
    account.observe(start + pd.Timedelta(minutes=3), {"S02": 100., "S03": 100.}, set())
    assert set(account.pending) == {"S00", "S01"}
    account.observe(start + pd.Timedelta(minutes=4), {"S00": 100., "S01": 100.}, set())
    assert set(account.pending) == {"S02", "S03"}
    account.observe(start + pd.Timedelta(minutes=5), {"S02": 100.}, set())
    account.observe(start + pd.Timedelta(minutes=6), {"S03": 100.}, set())
    assert {s for s, q in account.quantity.items() if q} == {"S02", "S03"}
    counts = pd.DataFrame(account.exposures).groupby("timestamp").size()
    assert counts.max() == 2


def test_missing_final_quotes_are_not_fabricated():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    with pytest.raises(ValueError, match="unliquidated"):
        simulate_cross_sectional(events, list(observations())[:3], calibration,
                                 calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))


def test_pending_entry_that_violates_limit_never_fills():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    events.loc[events.primary_side.eq(-1), "meta_action"] = 0

    def observations():
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol, "is_last": minute == 10, "price": 100. if minute == 0 else 200.,
            })

    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    assert result["trades"].empty
    assert result["ledger"].iloc[-1].aum == 100_000


def test_observed_bar_count_not_future_stored_expiry_controls_horizon():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    events["horizon_bars"] = 2
    events["vertical_barrier"] = start + pd.Timedelta(days=99)

    def marked():
        for time, quotes in observations():
            yield time, quotes

    result = simulate_cross_sectional(events, marked(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    exits = result["trades"].loc[result["trades"].reason.eq("signal_reduction")]
    assert not exits.empty
    assert exits.timestamp.min() == start + pd.Timedelta(minutes=3)


@pytest.mark.parametrize(
    ("multipliers", "final_price", "expired"),
    [
        ((5.0, 5.0), 208.0, False),
        ((5.0, 5.0), 48.0, False),
        ((2.0, 5.0), 160.0, False),
        ((5.0, 2.0), 96.0, False),
        ((5.0, 5.0), 209.0, True),
        ((5.0, 5.0), 47.0, True),
        ((2.0, 5.0), 161.0, True),
        ((5.0, 2.0), 95.0, True),
        ((0.0, 5.0), 208.0, False),
        ((5.0, 0.0), 48.0, False),
        ((-1.0, 5.0), 208.0, False),
        ((5.0, -1.0), 48.0, False),
    ],
)
def test_horizontal_multipliers_control_event_expiry(
    monkeypatch, multipliers, final_price, expired
):
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    events["horizon_volatility"] = 0.125
    snapshots = {}
    original_observe = portfolio_management._CrossSectionalAccount.observe

    def tracked_observe(self, now, *args, **kwargs):
        result = original_observe(self, now, *args, **kwargs)
        snapshots[now] = {event["symbol"] for event in self.active.values()}
        return result

    def observations():
        intermediate = 152.0 if final_price > 128.0 else 104.0
        for minute in range(11):
            price = 128.0 if minute == 0 else intermediate if minute == 1 else final_price
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol, "is_last": minute == 10, "price": price,
            })

    monkeypatch.setattr(portfolio_management._CrossSectionalAccount, "observe", tracked_observe)
    simulate_cross_sectional(
        events, observations(), calibration, calendar, calendar.close.iloc[0],
        settings=PortfolioSettings(horizontal_multipliers=multipliers),
    )

    # Both predicted directions survive a move beyond 1 sigma but below the barrier.
    assert snapshots[start + pd.Timedelta(minutes=1)] == set(events.symbol)
    expected = set() if expired else set(events.symbol)
    assert snapshots[start + pd.Timedelta(minutes=2)] == expected


def test_missing_calibration_excludes_trading_but_reports_symbol():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    calibration.loc[calibration.symbol.eq("S00"), "w"] = np.nan
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings(max_positions_per_side=5))
    assert "S00" in set(result["exclusions"].symbol)
    assert "S00" not in set(result["trades"].symbol)


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("remaining_signal", [False, True])
def test_slots_wait_for_full_liquidation_before_selecting_better_candidate(side, remaining_signal):
    start, calendar, events, _, _ = cross_sectional_scenario()
    events = events.head(2).copy()
    events["symbol"] = ["A", "B"]
    events["primary_side"] = side
    events.loc[events.symbol.eq("A"), "vertical_barrier"] = start + pd.Timedelta(minutes=3)
    events.loc[events.symbol.eq("B"), "event_start"] = start + pd.Timedelta(minutes=2)
    if remaining_signal:
        extra = events.loc[events.symbol.eq("A")].copy()
        extra["event_start"] = start + pd.Timedelta(minutes=2)
        extra["vertical_barrier"] = start + pd.Timedelta(minutes=8)
        extra["meta_probability"] = .55
        events = pd.concat([events, extra], ignore_index=True)
    calibration = events[["symbol", "event_start"]].assign(w=1.)
    account = portfolio_management._CrossSectionalAccount(
        events, calibration, calendar, calendar.close.iloc[0],
        PortfolioSettings(max_positions_per_side=1), "holdout")
    account.observe(start, {"A": 100.}, set())
    account.observe(start + pd.Timedelta(minutes=1), {"A": 100.}, set())
    # B has a strictly better directional price advantage, but cannot evict A.
    candidate_price = 200.
    account.observe(start + pd.Timedelta(minutes=2), {"A": 100., "B": candidate_price}, set())
    table = candidate_snapshot(pd.DataFrame(account.active.values()), account.prices,
        account.w, start + pd.Timedelta(minutes=2), calendar, account.settings)
    assert table.loc["B", "score"] > table.loc["A", "score"]
    assert "B" not in account.pending
    account.observe(start + pd.Timedelta(minutes=3), {"B": candidate_price}, set())
    assert "B" not in account.pending
    assert account.quantity["A"] != 0
    account.observe(start + pd.Timedelta(minutes=4), {"A": 100.}, set())
    assert ("B" in account.pending) == (not remaining_signal)
    assert bool(account.quantity.get("A", 0)) == remaining_signal
    account.observe(start + pd.Timedelta(minutes=5), {"B": candidate_price}, set())
    assert bool(account.quantity.get("B", 0.)) == (not remaining_signal)
    assert not any(fill["reason"].startswith("replacement") for fill in account.fills)
    if not remaining_signal:
        assert account.fills[-2]["timestamp"] == start + pd.Timedelta(minutes=4)
        assert account.fills[-1]["timestamp"] == start + pd.Timedelta(minutes=5)


def calendar():
    opens = pd.to_datetime(["2025-01-03 14:30Z", "2025-01-06 14:30Z", "2025-01-07 14:30Z"])
    return pd.DataFrame({"open": opens, "close": opens + pd.Timedelta(hours=6, minutes=30)})


def events(rows):
    return pd.DataFrame(rows, columns=["symbol", "event_start", "primary_side", "meta_probability", "meta_action"]).assign(
        event_start=lambda x: pd.to_datetime(x.event_start, format="mixed", utc=True), partition="holdout")


def prices(cal, symbols=("A", "B")):
    return pd.DataFrame([{"timestamp": t, "symbol": s, "price": 100.0}
                         for t in sorted([*(cal.open + pd.Timedelta(minutes=1)),
                                          *(cal.open + pd.Timedelta(minutes=2)), *cal.close]) for s in symbols])


def test_cutoff_latest_rejection_and_one_time_assignment():
    cal = calendar()
    data = events([
        ("A", "2025-01-03 14:28Z", 1, .99, 1),
        ("A", "2025-01-03 14:29Z", 1, .8, 0),
        ("B", "2025-01-03 14:29Z", -1, .9, 1),
        ("C", "2025-01-03 14:29:01Z", 1, .9, 1),
        ("D", "2025-01-03 20:59Z", 1, .8, 1),
        ("E", "2025-01-03 20:59:01Z", 1, .8, 1),
    ])
    result = select_synchronous(data, entry_schedule(cal, cal.close.max()))
    assert "A" not in result.symbol.tolist()
    assert result.set_index("symbol").group.to_dict() == {"B": "open", "C": "close", "D": "close", "E": "open"}
    assert result.set_index("symbol").loc["E", "entry"] == cal.open.iloc[1]
    assert not result.duplicated(["symbol", "event_start"]).any()


def test_meta_probability_ranks_with_alphabetical_ties_and_equal_direction_weights():
    data = events([(s, "2025-01-03 14:00Z", side, p, 1)
                   for s, side, p in [("Z", 1, .9), ("A", 1, .9), ("C", 1, .8), ("S", -1, .7)]])
    cal = calendar()
    selected = select_synchronous(data, entry_schedule(cal, cal.close.max()), max_positions_per_side=4)
    assert selected.symbol.tolist() == ["A", "Z", "S"]
    assert selected.target_weight.tolist() == [.25, .25, -.5]


def test_books_overlap_without_netting_and_full_reentry_costs_reconcile():
    cal = calendar()
    data = events([
        ("A", "2025-01-03 16:00Z", -1, .9, 1),
        ("A", "2025-01-06 14:00Z", 1, .9, 1),
        ("A", "2025-01-06 16:00Z", -1, .9, 1),
    ])
    settings = PortfolioSettings(initial_aum=10000)
    result = simulate_synchronous(data, prices(cal), cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(), settings)
    concurrent = result['exposures'].loc[lambda x: x.timestamp.eq(cal.open.iloc[1] + pd.Timedelta(minutes=1))]
    assert len(concurrent) == 2
    assert set(concurrent.group) == {"open", "close"}
    assert concurrent.quantity.prod() < 0
    values = _position_values(result, pd.DatetimeIndex(result['ledger'].timestamp))
    assert values.shape[1] == 2
    assert len(result['trades']) == 6
    assert not result['trades'].loc[lambda x: x.timestamp.eq(cal.close.iloc[1])].shape[0]
    assert result['closed_trades'].net_pnl.sum() == pytest.approx(result['ledger'].iloc[-1].aum - 10000)
    assert result['ledger'].iloc[-1].aum < 10000
    groups = result['group_ledger'].pivot(index='timestamp',columns='group',values='aum')
    assert groups.iloc[0].tolist() == [5000,5000]
    assert groups.loc[cal.open.iloc[1] + pd.Timedelta(minutes=1), 'open'] != groups.loc[cal.open.iloc[1] + pd.Timedelta(minutes=1), 'close']
    stats = summarize_account(result, settings, pd.Series([0.,0.,0.], index=cal.close.dt.normalize()), benchmark_symbol="SPY", annual_risk_free_rate=0.03, periods_per_year=365.25, annualized_benchmark_sharpe_ratio=1.0)
    assert not stats.empty
    holdings = holding_statistics(result).set_index('metric').value
    assert holdings.maximum_positions == 2
    assert holdings.maximum_unique_symbols == 1


def test_unfilled_entry_expires_and_keeps_its_allocation():
    cal = calendar()
    data = events([("A", "2025-01-03 14:00Z", 1, .9, 1), ("B", "2025-01-03 14:00Z", 1, .8, 1)])
    quotes = prices(cal)
    quotes = quotes.loc[~(quotes.timestamp.lt(cal.close.iloc[0]) & quotes.symbol.eq("B"))]
    result = simulate_synchronous(data, quotes, cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(),
                                 PortfolioSettings(initial_aum=10000, broker_fee_bps=0, slippage_bps=0))
    assert len(result["exclusions"]) == 1
    assert result["trades"].iloc[0].traded_value == 1250
    assert result["trades"].symbol.eq("A").all()


@pytest.mark.parametrize("total_limit", [2, 10])
def test_total_directional_limit_is_split_between_books_without_final_overnight_entry(total_limit):
    cal=calendar(); rows=[]
    for time in ["2025-01-03 16:00Z", "2025-01-06 14:00Z", "2025-01-07 16:00Z"]:
        rows.extend((f'{side}_{i}',time,side,.99-i*.01,1) for side in [-1,1] for i in range(8))
    data=events(rows)
    settings = PortfolioSettings(max_positions_per_side=total_limit)
    selected = select_synchronous(data, entry_schedule(cal, cal.close.max()), total_limit)
    assert selected.groupby(["round", "primary_side"]).size().eq(total_limit // 2).all()
    assert set(selected.symbol) == {f"{side}_{i}" for side in [-1, 1] for i in range(total_limit // 2)}
    assert selected.target_weight.abs().eq(1 / total_limit).all()
    result=simulate_synchronous(data,prices(cal,data.symbol.unique()),cal,pd.Timestamp("2025-01-03 13:00Z"),cal.close.max(), settings=settings)
    exposure = result["exposures"].assign(side=lambda x: np.sign(x.quantity))
    assert exposure.groupby(["timestamp", "group", "side"]).size().max() == total_limit // 2
    assert exposure.groupby(["timestamp", "side"]).size().max() == total_limit
    assert exposure.groupby("timestamp").size().max() == 2 * total_limit
    assert not result['trades'].loc[lambda x:x.timestamp.eq(cal.close.max()) & x.reason.eq('scheduled_entry')].shape[0]


@pytest.mark.parametrize("limit", [1, 3, 0, -2, 2.0, True])
def test_synchronous_rejects_invalid_total_limits_even_with_preselected_events(limit):
    cal = calendar()
    data = events([("A", "2025-01-03 14:00Z", 1, .9, 1)])
    with pytest.raises(ValueError, match="positive even integer"):
        select_synchronous(data, entry_schedule(cal, cal.close.max()), limit)
    for selected in (None, pd.DataFrame()):
        with pytest.raises(ValueError, match="positive even integer"):
            simulate_synchronous(
                data, prices(cal), cal, pd.Timestamp("2025-01-03 13:00Z"),
                cal.close.max(), PortfolioSettings(max_positions_per_side=limit),
                selected=selected,
            )


def test_asynchronous_accepts_odd_total_limit_without_splitting():
    _, cal, data, calibration, observations = cross_sectional_scenario()
    result = simulate_cross_sectional(
        data, observations(), calibration, cal, cal.close.iloc[0],
        settings=PortfolioSettings(max_positions_per_side=3, broker_fee_bps=0, slippage_bps=0),
    )
    exposure = result["exposures"].assign(side=lambda x: np.sign(x.quantity))
    assert exposure.groupby(["timestamp", "side"]).size().max() == 3
    assert exposure.weight.abs().max() == pytest.approx(1 / 6)


def test_early_close_cutoff_and_late_feature_completion():
    cal=calendar();cal.loc[0,'close']=pd.Timestamp('2025-01-03 18:00Z')
    data=events([('A','2025-01-03 17:59Z',1,.9,1),('B','2025-01-03 17:59:01Z',1,.9,1)])
    selected=select_synchronous(data,entry_schedule(cal,cal.close.max())).set_index('symbol')
    assert selected.loc['A','entry']==cal.close.iloc[0]
    assert selected.loc['B','entry']==cal.open.iloc[1]


def test_bar_prices_use_completion_not_start_and_never_read_ticks(tmp_path):
    cal = calendar().iloc[:1]
    paths = ResearchPaths(tmp_path, start=cal.open.min().normalize(), end=cal.close.max().normalize() + pd.Timedelta(days=1))
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"start": [cal.open.iloc[0] - pd.Timedelta(days=1)],
                  "end": [cal.open.iloc[0] + pd.Timedelta(minutes=2)],
                  "open": [1.], "close": [100.]}).to_parquet(file)
    result = load_bar_prices(paths, cal.open.min(), cal.close.max(), ["A"])
    assert result.price.tolist() == [100.]
    assert result.timestamp.item() == cal.open.iloc[0] + pd.Timedelta(minutes=2)
    assert result.is_last.item()


def test_rejected_signals_leave_both_books_in_cash():
    cal = calendar()
    data = events([("A", "2025-01-03 14:00Z", 1, .1, 0)])
    result = simulate_synchronous(data, prices(cal), cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(), settings=PortfolioSettings())
    assert result["trades"].empty
    assert result["ledger"].aum.eq(100000).all()
    assert result["ledger"].gross_exposure.eq(0).all()
    assert holding_statistics(result).set_index("metric").loc["average_unallocated_capital_ratio", "value"] == 1


def test_future_prices_do_not_change_prior_selection_or_entry():
    cal = calendar()
    data = events([("A", "2025-01-03 14:00Z", 1, .9, 1)])
    original = prices(cal)
    changed = original.copy()
    changed.loc[changed.timestamp.gt(cal.open.iloc[0] + pd.Timedelta(minutes=1)), "price"] = 110
    args = (cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(), PortfolioSettings())
    first = simulate_synchronous(data, original, *args)
    second = simulate_synchronous(data, changed, *args)
    pd.testing.assert_frame_equal(first["trades"].head(1), second["trades"].head(1))
    assert second["group_ledger"].loc[lambda x: x.group.eq("close"), "aum"].eq(50000).all()
    assert second["ledger"].iloc[-1].aum > first["ledger"].iloc[-1].aum


def test_model_artifact_paths_share_only_within_family(tmp_path):
    market = ResearchPaths(tmp_path, 'market', start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    sentiment = ResearchPaths(tmp_path, 'sentiment', start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    assert market.artifacts != sentiment.artifacts
    assert market.event('prepared') == sentiment.event('prepared')
    assert market.raw('A','tick') == sentiment.raw('A','tick')


def test_public_and_compact_bar_streams_agree(tmp_path, monkeypatch):
    start, cal, data, calibration, observations = cross_sectional_scenario()
    paths = ResearchPaths(tmp_path, "sentiment", start=start.normalize(), end=start.normalize() + pd.Timedelta(days=1))
    for symbol in data.symbol:
        file = paths.feature(symbol, "dollar_bars")
        file.parent.mkdir(parents=True)
        pd.DataFrame({"end": [t for t, _ in observations()], "close": 100.}).to_parquet(file)
    prepared = {(family, "path_0"): {"events": data.assign(meta_action=action), "calibration": calibration}
                for family, action in [("boosting", 1), ("bagging", 0)]}
    settings = PortfolioSettings(max_positions_per_side=5)
    reads = []
    original_read = portfolio_management.pq.ParquetFile
    def read(file, *args, **kwargs):
        reads.append(file)
        return original_read(file, *args, **kwargs)
    monkeypatch.setattr(portfolio_management.pq, "ParquetFile", read)
    actual = portfolio_management._simulate_asynchronous_accounts(
        paths, prepared, cal, start, cal.close.iloc[0], settings, "holdout", symbols=data.symbol)
    assert len(reads) == 2 * len(set(reads)) == 2 * len(data.symbol)
    for key, item in prepared.items():
        expected = simulate_cross_sectional(item["events"], portfolio_management.observation_stream(
            paths, start, cal.close.iloc[0], symbols=data.symbol, calendar=cal),
            calibration, cal, cal.close.iloc[0], settings)
        for table in expected:
            pd.testing.assert_frame_equal(actual[key][table], expected[table])
    assert actual[("bagging", "path_0")]["trades"].empty


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("synchronous", [False, True])
def test_shared_progress_closes_and_counts_bars(monkeypatch, failure, synchronous):
    from io import StringIO
    from tqdm import tqdm
    start, cal, data, calibration, observations = cross_sectional_scenario()
    prices = pd.concat([q.assign(timestamp=t) for t, q in observations()], ignore_index=True)
    bars = []
    def progress(*args, **kwargs):
        bar = tqdm(*args, file=StringIO(), **kwargs)
        bars.append(bar)
        return bar
    monkeypatch.setattr(portfolio_management, "tqdm", progress)
    original = portfolio_management._bar_batches
    def batches(*args, **kwargs):
        for i, row in enumerate(original(*args, **kwargs)):
            if failure and i == 1:
                raise ValueError("broken input")
            yield row
    monkeypatch.setattr(portfolio_management, "_bar_batches", batches)
    if synchronous:
        account = portfolio_management._SynchronousAccount(data, cal, start, cal.close.iloc[0], PortfolioSettings(), "holdout")
    else:
        account = portfolio_management._CrossSectionalAccount(data, calibration, cal, cal.close.iloc[0], PortfolioSettings(), "holdout")
    def run():
        return portfolio_management._run_accounts({0: account}, prices, start, cal.close.iloc[0], PortfolioSettings(), show_progress=True)
    if failure:
        with pytest.raises(ValueError, match="broken input"):
            run()
    else:
        run()
        assert bars[0].n == bars[0].total == len(prices)
    assert len(bars) == 1 and bars[0].disable


# Completed dollar-bar execution

BAR_START = pd.Timestamp("2025-02-03T14:30Z")


def bar_time(minute):
    return BAR_START + pd.Timedelta(minutes=minute)


def bar_calendar():
    return pd.DataFrame({"open": [bar_time(0), bar_time(10), bar_time(20)],
                         "close": [bar_time(5), bar_time(15), bar_time(25)]})


def bar_synchronous_account(rows, costs=0):
    selected = pd.DataFrame(rows, columns=["symbol", "entry", "exit", "target_weight"])
    selected["group"] = "open"
    events = pd.DataFrame(columns=["partition", "event_start"])
    return portfolio_management._SynchronousAccount(events, bar_calendar(), bar_time(-1), bar_time(30),
                                  portfolio_management.PortfolioSettings(initial_aum=1000, broker_fee_bps=costs,
                                                       slippage_bps=costs), "holdout", selected)


def bar_observe(account, minute, fresh=None, terminal=()):
    fresh = fresh or {}
    account.observe(bar_time(minute), fresh, set(fresh), set(terminal))


def test_sync_waits_for_whole_book_then_next_own_bar_and_fixes_notional():
    account = bar_synchronous_account([
        ("A", bar_time(0), bar_time(5), .25), ("B", bar_time(0), bar_time(5), .25),
        ("C", bar_time(10), bar_time(15), .25), ("D", bar_time(10), bar_time(15), .25),
    ], costs=10)
    bar_observe(account, 0, {"A": 100., "B": 100.})
    assert not account.trades
    bar_observe(account, 1, {"A": 100.})
    bar_observe(account, 2, {"B": 100.})
    assert [t["traded_value"] for t in account.trades] == [125., 125.]
    bar_observe(account, 5, {"A": 110., "B": 110.})
    assert len(account.trades) == 2  # No fill at the exit decision itself.
    bar_observe(account, 6, {"A": 110.})
    bar_observe(account, 10, {"C": 100., "D": 100.})
    assert not account.books["open"]["pending"]
    bar_observe(account, 11, {"B": 110., "C": 100.})
    assert not any(t["symbol"] == "C" for t in account.trades)
    wealth = account.books["open"]["cash"]
    bar_observe(account, 12, {"C": 100.})
    bar_observe(account, 13, {"D": 200.})
    assert account.trades[-2]["traded_value"] == pytest.approx(wealth * .25)
    assert account.trades[-1]["traded_value"] == pytest.approx(wealth * .25)
    bar_observe(account, 14, {"C": 101.}, ["C"])
    bar_observe(account, 15, {"D": 201.}, ["D"])
    result = account.finish()
    assert result["ledger"].iloc[-1].gross_exposure == 0
    assert result["closed_trades"].net_pnl.sum() == pytest.approx(result["ledger"].iloc[-1].aum - 1000)


def test_sync_expired_waiting_round_replaced_and_final_bar_cannot_enter():
    account = bar_synchronous_account([
        ("A", bar_time(0), bar_time(5), .5), ("B", bar_time(10), bar_time(15), .5),
        ("C", bar_time(20), bar_time(25), .5),
    ])
    bar_observe(account, 0)
    bar_observe(account, 1, {"A": 100.})
    bar_observe(account, 10, {"B": 100.})
    bar_observe(account, 15)
    bar_observe(account, 20, {"C": 100.})
    bar_observe(account, 21, {"A": 100., "B": 100., "C": 100.}, ["A", "B"])
    bar_observe(account, 22, {"C": 100.}, ["C"])
    assert [t["symbol"] for t in account.trades] == ["A", "A"]
    assert not account.books["open"]["pending"]
    assert not account.books["open"]["waiting"]
    account.finish()


def bar_asynchronous_account(horizon=780):
    events = pd.DataFrame({"symbol": ["A"], "event_start": [bar_time(0)], "partition": ["holdout"],
                           "vertical_barrier": [bar_time(1000)], "horizon_bars": [horizon],
                           "horizon_volatility": [.1], "primary_side": [1], "meta_action": [1],
                           "meta_probability": [1.]})
    calibration = events[["symbol", "event_start"]].assign(w=1.)
    return portfolio_management._CrossSectionalAccount(events, calibration, bar_calendar(), bar_time(1000),
                                     portfolio_management.PortfolioSettings(broker_fee_bps=0, slippage_bps=0), "holdout")


def test_async_780_bars_excludes_registration_and_other_symbols():
    account = bar_asynchronous_account()
    bar_observe(account, 0, {"A": 100.})
    bar_observe(account, .5, {"B": 100.})
    assert account.active[("A", bar_time(0))]["remaining_bars"] == 780
    for minute in range(1, 780):
        bar_observe(account, minute, {"A": 100.})
    assert account.active[("A", bar_time(0))]["remaining_bars"] == 1
    bar_observe(account, 780, {"A": 100.})
    assert not account.active
    assert len(account.fills) == 1
    bar_observe(account, 781, {"A": 100.})
    assert account.fills[-1]["reason"] == "signal_reduction"
    account.finish()


def test_async_terminal_overrides_pending_and_new_event():
    account = bar_asynchronous_account()
    bar_observe(account, 0, {"A": 100.}, ["A"])
    assert not account.pending and not account.active and not account.fills
    account.finish()
    account = bar_asynchronous_account()
    bar_observe(account, 0, {"A": 100.})
    bar_observe(account, 1, {"A": 100.})
    bar_observe(account, 2, {"A": 105.}, ["A"])
    assert account.fills[-1]["reason"] == "research_end"
    assert account.fills[-1]["timestamp"] == bar_time(2)
    assert not account.pending and not account.active
    account.finish()


@pytest.mark.parametrize("price", [np.nan, np.inf, 0., -1.])
def test_invalid_bar_prices_rejected(price):
    with pytest.raises(ValueError, match="finite and positive"):
        portfolio_management._prepare_bar_prices(pd.DataFrame({"timestamp": [bar_time(0)], "symbol": ["A"], "price": [price]}), bar_time(0), bar_time(1))


def test_duplicate_bar_end_rejected():
    prices = pd.DataFrame({"timestamp": [bar_time(0)] * 2, "symbol": ["A"] * 2, "price": [100., 101.]})
    with pytest.raises(ValueError, match="Unique"):
        portfolio_management._prepare_bar_prices(prices, bar_time(0), bar_time(1))


def test_bar_batches_are_order_independent_and_have_no_minute_clocks():
    prices = pd.DataFrame({"timestamp": [bar_time(0), bar_time(0), bar_time(3)], "symbol": ["B", "A", "A"], "price": [20., 10., 11.]})
    one = list(portfolio_management._bar_batches(portfolio_management._prepare_bar_prices(prices, bar_time(0), bar_time(5))))
    two = list(portfolio_management._bar_batches(portfolio_management._prepare_bar_prices(prices.iloc[::-1], bar_time(0), bar_time(5))))
    assert one == two == [(bar_time(0), {"A": 10., "B": 20.}, {"B"}), (bar_time(3), {"A": 11.}, {"A"})]


def test_missing_bar_file_and_benchmark_anchor_are_explicit(tmp_path):
    from src.preprocessing.market_data import ResearchPaths
    paths = ResearchPaths(tmp_path, start=BAR_START.normalize(), end=BAR_START.normalize() + pd.Timedelta(days=1))
    with pytest.raises(ValueError, match="run dollar-bar preprocessing"):
        portfolio_management.benchmark_returns(paths, bar_time(0), bar_time(10), data_start=BAR_START.normalize(), benchmark_symbol="SPY")
    file = paths.feature("SPY", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": [bar_time(1), bar_time(2)], "close": [100., 101.]}).to_parquet(file)
    with pytest.raises(ValueError, match="does not cover"):
        portfolio_management.benchmark_returns(paths, bar_time(0), bar_time(10), data_start=BAR_START.normalize(), benchmark_symbol="SPY")


def test_sync_shared_and_single_accounts_match_without_raw_data(tmp_path):
    from src.preprocessing.market_data import ResearchPaths
    paths = ResearchPaths(tmp_path, "market", start=BAR_START.normalize(), end=BAR_START.normalize() + pd.Timedelta(days=1))
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": [bar_time(1), bar_time(2), bar_time(6), bar_time(24)], "close": [100., 101., 102., 103.]}).to_parquet(file)
    events = pd.DataFrame({"symbol": ["A"], "event_start": [bar_time(-1)], "partition": ["holdout"],
                           "primary_side": [1], "meta_probability": [.9], "meta_action": [1]})
    settings = portfolio_management.PortfolioSettings()
    prices = portfolio_management.load_bar_prices(paths, bar_time(-2), bar_time(25), ["A"])
    expected = portfolio_management.simulate_synchronous(events, prices, bar_calendar(), bar_time(-2), bar_time(25), settings)
    actual = portfolio_management._simulate_synchronous_accounts(paths, {0: {"events": events}}, bar_calendar(), bar_time(-2), bar_time(25),
                                                settings, "holdout", symbols=["A"])[0]
    for table in expected:
        pd.testing.assert_frame_equal(actual[table], expected[table])


# Candidate state and recording

PERFORMANCE_START = pd.Timestamp("2025-02-03T14:30Z")


def performance_calendar():
    return pd.DataFrame({"open": [PERFORMANCE_START, PERFORMANCE_START + pd.Timedelta(days=1)],
                         "close": [PERFORMANCE_START + pd.Timedelta(minutes=210), PERFORMANCE_START + pd.Timedelta(days=1, minutes=390)]})


def performance_active_events():
    rng = np.random.default_rng(42)
    count = 80
    events = pd.DataFrame({"symbol": [f"S{i // 4:02}" for i in range(count)],
                           "event_start": [PERFORMANCE_START + pd.Timedelta(seconds=i) for i in range(count)],
                           "entry_price": rng.uniform(90, 110, count), "horizon_volatility": .05,
                           "primary_side": rng.choice([-1, 1], count), "meta_action": rng.choice([0, 1], count),
                           "meta_probability": rng.uniform(.5, 1, count), "partition": "holdout",
                           "vertical_barrier": PERFORMANCE_START + pd.Timedelta(days=1), "horizon_bars": 780})
    events["event_signal"] = event_bet_signals(events)
    return events


@pytest.mark.parametrize("minutes", [-30, 0, 60, 210, 300, 1440, 1800, 3000])
def test_session_coordinate_matches_reference_at_holidays_and_early_close(minutes):
    cal = performance_calendar()
    clock = SessionClock(cal)
    begin = PERFORMANCE_START - pd.Timedelta(days=3)
    now = PERFORMANCE_START + pd.Timedelta(minutes=minutes)
    assert clock.at(now) - clock.at(begin) == pytest.approx(portfolio_management.elapsed_session_minutes(begin, now, cal))


@pytest.mark.parametrize("half_life", [30., 390.])
def test_cached_candidates_match_reference_through_updates_expiry_and_missing_calibration(half_life):
    settings = portfolio_management.PortfolioSettings(half_life_minutes=half_life)
    cal = performance_calendar()
    book = CandidateBook(settings, SessionClock(cal), cal.close.max())
    active = performance_active_events()
    calibration = active.set_index(["symbol", "event_start"]).assign(w=1.).w
    calibration.iloc[-1] = np.nan
    prices = dict.fromkeys(active.symbol, 100.)
    for phase in range(4):
        if phase == 1:
            prices["S00"] = 120.
        elif phase == 2:
            active = active.loc[~active.index.isin([3, 7, 8, 9, 10, 11])]
        elif phase == 3:
            active = active.iloc[:0]
        for symbol in prices:
            events = active.loc[active.symbol.eq(symbol)].to_dict("records")
            for event in events:
                event["price_w"] = calibration[(symbol, event["event_start"])]
                event["forecast_price"] = event["entry_price"] * (1 + event["primary_side"] * event["horizon_volatility"])
                event["ranking_decay"] = 2 ** ((book.clock.at(event["event_start"]) - book.origin) / half_life)
            book.aggregate(symbol, events)
            book.refresh(symbol, prices[symbol])
        now = PERFORMANCE_START + pd.Timedelta(days=1, minutes=phase + 1)
        reference = portfolio_management.candidate_snapshot(active, prices, calibration, now, cal, settings)
        factor = 2 ** ((book.origin - book.clock.at(now)) / half_life)
        assert set(book.rows) == set(reference.index)
        for symbol, row in book.rows.items():
            expected = reference.loc[symbol]
            assert row.target_weight == expected.target_weight
            assert row.eligible == expected.eligible
            assert row.event_start == expected.event_start
            assert row.limit == pytest.approx(expected.limit, nan_ok=True)
            if math.isfinite(row.score):
                assert row.score * factor == pytest.approx(expected.score, rel=1e-12, abs=1e-15)
        expected = reference.loc[reference.eligible].reset_index().sort_values(
            ["score", "event_start", "symbol"], ascending=[False, False, True]).symbol.tolist() if not reference.empty else []
        actual = sorted([*book.top(-1, 100, set()), *book.top(1, 100, set())], key=lambda row: row.rank)
        assert [row.symbol for row in actual] == expected


@pytest.mark.parametrize("signals", [[.05], [.15], [0., .1], [0., -.1], [.2, .3, .4], [-.3, -.2, -.4]])
def test_incremental_means_preserve_zero_signals_and_rounding_boundaries(signals):
    settings = portfolio_management.PortfolioSettings()
    book = CandidateBook(settings, SessionClock(performance_calendar()), PERFORMANCE_START)
    events = [{"event_signal": signal, "event_start": PERFORMANCE_START + pd.Timedelta(seconds=i)}
              for i, signal in enumerate(signals)]
    book.aggregate("A", events)
    active = pd.DataFrame(events).assign(symbol="A")
    from src.backtesting.bet_sizing import average_symbol_targets
    assert book.targets["A"] == average_symbol_targets(active).loc["A"]


def test_candidate_heaps_keep_ties_stable_and_stale_entries_bounded():
    book = CandidateBook(portfolio_management.PortfolioSettings(), SessionClock(performance_calendar()), PERFORMANCE_START)
    for symbol in ["B", "A"]:
        book.aggregate(symbol, [{"event_start": PERFORMANCE_START, "event_signal": 1., "price_w": 1.,
                                 "forecast_price": 110., "ranking_decay": 1., "primary_side": 1, "meta_action": 1}])
        book.refresh(symbol, 100.)
    assert [row.symbol for row in book.top(1, 2, set())] == ["A", "B"]
    for _ in range(1000):
        book.refresh("A", 100.)
    assert sum(map(len, book.heaps.values())) <= 4 * len(book.rows) + 64
    assert [row.symbol for row in book.top(1, 2, {"A"})] == ["B"]


def performance_make_account():
    events = performance_active_events().iloc[[0]].assign(symbol="A", event_start=PERFORMANCE_START, primary_side=1,
                                             meta_action=1, meta_probability=1., event_signal=1.)
    calibration = events[["symbol", "event_start"]].assign(w=1.)
    return portfolio_management._CrossSectionalAccount(events, calibration, performance_calendar(), performance_calendar().close.max(),
                                     portfolio_management.PortfolioSettings(broker_fee_bps=0, slippage_bps=0), "holdout")


def test_minute_last_records_flush_before_mutation_and_keep_all_fills_and_closes():
    account = performance_make_account()
    for seconds, fresh, last in [(0, {"A": 100.}, set()), (1, {"A": 100.}, set()),
                                  (2, {"A": 101.}, set()), (10, {"B": 100.}, set()),
                                  (61, {"B": 100.}, set()), (62, {"A": 102.}, {"A"})]:
        account.observe(PERFORMANCE_START + pd.Timedelta(seconds=seconds), fresh, set(fresh), last)
    boundary = performance_calendar().close.iloc[0]
    account.observe(boundary, {}, set())
    result = account.finish()
    times = result["ledger"].timestamp.tolist()
    assert times == [PERFORMANCE_START + pd.Timedelta(seconds=s) for s in [1, 10, 62]] + [boundary]
    at_last = result["ledger"].set_index("timestamp").loc[PERFORMANCE_START + pd.Timedelta(seconds=10)]
    assert at_last.aum == pytest.approx(100000 + account.fills[0]["quantity_change"])
    assert result["trades"].timestamp.isin(times).all()
    assert not result["exposures"].timestamp.eq(PERFORMANCE_START + pd.Timedelta(seconds=62)).any()
    assert len(result["trades"]) == 2


def test_record_only_close_cannot_expire_signal_or_fill_order():
    account = performance_make_account()
    account.stream[0]["vertical_barrier"] = PERFORMANCE_START + pd.Timedelta(seconds=1)
    account.stream[0].pop("horizon_bars")
    account.observe(PERFORMANCE_START, {"A": 100.}, {"A"})
    account.observe(PERFORMANCE_START + pd.Timedelta(seconds=2), {}, set())
    assert account.active and account.pending and not account.fills


def test_stream_matches_dataframe_across_arrow_batches_and_bounds(tmp_path):
    paths = ResearchPaths(tmp_path, start=PERFORMANCE_START.normalize(), end=PERFORMANCE_START.normalize() + pd.Timedelta(days=1))
    for symbol in ["A", "B"]:
        file = paths.feature(symbol, "dollar_bars")
        file.parent.mkdir(parents=True)
        pd.DataFrame({"end": pd.date_range(PERFORMANCE_START, periods=9000, freq="s"), "close": 100.}).to_parquet(file)
    begin, end = PERFORMANCE_START + pd.Timedelta(seconds=100), PERFORMANCE_START + pd.Timedelta(seconds=8500)
    stream = BarStream(paths, begin, end, ["B", "A"])
    frame = portfolio_management.load_bar_prices(paths, begin, end, ["A", "B"])
    clocks = {begin, end, end + pd.Timedelta(seconds=1)}
    assert len(stream) == len(frame) == 16802
    assert list(portfolio_management._bar_batches(stream, clocks)) == list(portfolio_management._bar_batches(frame, clocks))


@pytest.mark.parametrize("bad", ["duplicate", "missing_time", "price", "out_of_order"])
def test_stream_rejects_invalid_prices_and_times(tmp_path, bad):
    paths = ResearchPaths(tmp_path, start=PERFORMANCE_START.normalize(), end=PERFORMANCE_START.normalize() + pd.Timedelta(days=1))
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    times = [PERFORMANCE_START, PERFORMANCE_START + pd.Timedelta(seconds=1)]
    prices = [100., 101.]
    if bad == "duplicate": times[1] = times[0]
    if bad == "missing_time": times[1] = pd.NaT
    if bad == "out_of_order": times.reverse()
    if bad == "price": prices[1] = np.nan
    pd.DataFrame({"end": times, "close": prices}).to_parquet(file)
    with pytest.raises(ValueError):
        list(BarStream(paths, PERFORMANCE_START, PERFORMANCE_START + pd.Timedelta(seconds=10), ["A"]).batches())


def test_multiple_fills_in_same_minute_are_all_recorded():
    account = performance_make_account()
    account.stream[0]["horizon_bars"] = 2
    for seconds in range(5):
        account.observe(PERFORMANCE_START + pd.Timedelta(seconds=seconds), {"A": 100.}, {"A"})
    result = account.finish()
    assert result["trades"].timestamp.tolist() == [PERFORMANCE_START + pd.Timedelta(seconds=i) for i in (1, 3)]
    assert result["trades"].timestamp.isin(result["ledger"].timestamp).all()
    assert result["ledger"].iloc[-1].gross_exposure == 0


def test_temporary_arrow_batches_are_removed_on_early_close(tmp_path, monkeypatch):
    from tempfile import TemporaryDirectory
    paths = ResearchPaths(tmp_path, start=PERFORMANCE_START.normalize(), end=PERFORMANCE_START.normalize() + pd.Timedelta(days=1))
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": [PERFORMANCE_START, PERFORMANCE_START + pd.Timedelta(seconds=1)], "close": [100., 101.]}).to_parquet(file)
    directories = []
    def temporary(*args, **kwargs):
        from pathlib import Path
        result = TemporaryDirectory(*args, **kwargs)
        directories.append(Path(result.name))
        return result
    monkeypatch.setattr(portfolio_management, "TemporaryDirectory", temporary)
    batches = BarStream(paths, PERFORMANCE_START, PERFORMANCE_START + pd.Timedelta(seconds=2), ["A"]).batches()
    next(batches)
    assert directories[0].exists()
    batches.close()
    assert not directories[0].exists()


def test_daily_equity_survives_reduced_records_and_session_close_without_bar():
    from src.backtesting.backtesting_workflow import _daily_account
    account = performance_make_account()
    account.stream[0]["horizon_volatility"] = .5
    first_close, last_close = performance_calendar().close
    points = [
        (PERFORMANCE_START, 100., False),
        (PERFORMANCE_START + pd.Timedelta(minutes=1), 100., False),
        (PERFORMANCE_START + pd.Timedelta(minutes=1, seconds=20), 105., False),
        (first_close - pd.Timedelta(minutes=1), 110., False),
        (first_close, None, False),
        (performance_calendar().open.iloc[1], 115., False),
        (last_close - pd.Timedelta(minutes=1), 120., True),
        (last_close, None, False),
    ]
    for now, price, terminal in points:
        fresh = {} if price is None else {"A": price}
        account.observe(now, fresh, set(fresh), {"A"} if terminal else set())
    result = account.finish()
    daily = _daily_account(result["ledger"].set_index("timestamp"), 100000)
    assert daily.aum.tolist() == pytest.approx([100500., 101000.])
    assert daily.net_return.tolist() == pytest.approx([.005, 101000 / 100500 - 1])
    assert result["trades"].timestamp.tolist() == [PERFORMANCE_START + pd.Timedelta(minutes=1), last_close - pd.Timedelta(minutes=1)]


# Synchronous scheduling

SCHEDULED_START = pd.Timestamp("2025-02-03T14:30Z")


def scheduled_time(minute):
    return SCHEDULED_START + pd.Timedelta(minutes=minute)


def scheduled_calendar():
    return pd.DataFrame({"open": [scheduled_time(0), scheduled_time(10), scheduled_time(20)], "close": [scheduled_time(5), scheduled_time(15), scheduled_time(25)]})


def scheduled_account(rows, costs=0):
    selected = pd.DataFrame(rows, columns=["symbol", "entry", "exit", "target_weight"]).assign(group="open")
    events = pd.DataFrame(columns=["partition", "event_start"])
    return portfolio_management._SynchronousAccount(events, scheduled_calendar(), scheduled_time(0), scheduled_time(30),
                                  portfolio_management.PortfolioSettings(initial_aum=1000, broker_fee_bps=costs,
                                                       slippage_bps=costs), "holdout", selected)


def scheduled_prices(rows):
    return pd.DataFrame(rows, columns=["symbol", "timestamp", "price"])


def scheduled_run(rows, quotes):
    item = scheduled_account(rows)
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), prices=quotes) as index:
        scheduler = SynchronousScheduler({0: item}, index, set(scheduled_calendar().open) | set(scheduled_calendar().close), scheduled_time(0), scheduled_time(30))
        result = scheduler.run()[0]
        return result, scheduler.calls, dict(index.counts)


def test_price_queries_are_strict_for_fills_and_inclusive_for_marks():
    frame = scheduled_prices([("A", scheduled_time(0), 100.), ("A", scheduled_time(1), 101.), ("A", scheduled_time(3), 103.)])
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), prices=frame) as index:
        directory = Path(index.directory.name)
        assert index.next_after("A", scheduled_time(0)) == (scheduled_time(1), 101.)
        assert index.last_at_or_before("A", scheduled_time(0)) == (scheduled_time(0), 100.)
        assert index.last_at_or_before("A", scheduled_time(2)) == (scheduled_time(1), 101.)
        assert index.last_at_or_before("A", scheduled_time(-1)) is None
        assert index.next_after("A", scheduled_time(3)) is None
        assert index.terminal("A") == scheduled_time(3)
        assert index.terminal("MISSING") is None
    assert not directory.exists()


def test_orders_cannot_fill_at_decision_or_exit_boundary():
    quotes = scheduled_prices([("A", scheduled_time(i), 100. + i) for i in (0, 1, 5, 6, 24)])
    result, _, _ = scheduled_run([("A", scheduled_time(0), scheduled_time(5), .5)], quotes)
    assert result["trades"].timestamp.tolist() == [scheduled_time(1), scheduled_time(6)]
    assert result["trades"].reason.tolist() == ["scheduled_entry", "scheduled_exit"]


def test_unrelated_bars_do_not_add_account_calls_or_price_loads():
    rows = [("A", scheduled_time(0), scheduled_time(5), .5)]
    quotes = scheduled_prices([("A", scheduled_time(i), 100.) for i in (1, 3, 6, 24)])
    unrelated = pd.DataFrame({"symbol": "UNRELATED", "timestamp": pd.date_range(scheduled_time(0), periods=10000, freq="100ms"), "price": 500.})
    one, calls_one, counts_one = scheduled_run(rows, quotes)
    two, calls_two, counts_two = scheduled_run(rows, pd.concat([quotes, unrelated], ignore_index=True))
    assert calls_one == calls_two
    assert counts_one == counts_two
    assert counts_one["loads"] == 1
    for table in one:
        pd.testing.assert_frame_equal(one[table], two[table])


def test_whole_book_waits_and_freezes_notional_across_delayed_entries():
    rows = [("A", scheduled_time(0), scheduled_time(5), .25), ("B", scheduled_time(0), scheduled_time(5), .25),
            ("C", scheduled_time(10), scheduled_time(15), .25), ("D", scheduled_time(10), scheduled_time(15), .25)]
    quotes = scheduled_prices([("A", scheduled_time(1), 100.), ("A", scheduled_time(6), 110.), ("A", scheduled_time(29), 110.),
                     ("B", scheduled_time(2), 100.), ("B", scheduled_time(11), 110.), ("B", scheduled_time(29), 110.),
                     ("C", scheduled_time(11), 100.), ("C", scheduled_time(12), 100.), ("C", scheduled_time(14), 101.),
                     ("D", scheduled_time(13), 200.), ("D", scheduled_time(15), 201.)])
    result, _, _ = scheduled_run(rows, quotes)
    entries = result["trades"].loc[result["trades"].reason.eq("scheduled_entry")].set_index("symbol")
    assert entries.loc["C", "timestamp"] == scheduled_time(12)
    assert entries.loc["D", "timestamp"] == scheduled_time(13)
    assert entries.loc["C", "traded_value"] == entries.loc["D", "traded_value"] == 131.25
    assert result["closed_trades"].net_pnl.sum() == pytest.approx(result["ledger"].iloc[-1].aum - 1000)


def test_expired_waiting_round_is_replaced_and_terminal_entry_cancelled():
    rows = [("A", scheduled_time(0), scheduled_time(5), .5), ("B", scheduled_time(10), scheduled_time(15), .5), ("C", scheduled_time(20), scheduled_time(25), .5)]
    quotes = scheduled_prices([("A", scheduled_time(1), 100.), ("A", scheduled_time(21), 100.),
                     ("B", scheduled_time(10), 100.), ("B", scheduled_time(21), 100.),
                     ("C", scheduled_time(20), 100.), ("C", scheduled_time(21), 100.), ("C", scheduled_time(22), 100.)])
    result, _, _ = scheduled_run(rows, quotes)
    assert result["trades"].symbol.tolist() == ["A", "A"]
    assert result["trades"].reason.tolist() == ["scheduled_entry", "research_end"]
    assert result["trades"].timestamp.tolist() == [scheduled_time(1), scheduled_time(21)]


def test_no_bars_and_already_terminal_symbols_keep_allocations_in_cash():
    rows = [("A", scheduled_time(0), scheduled_time(5), .25), ("B", scheduled_time(0), scheduled_time(5), .25),
            ("A", scheduled_time(10), scheduled_time(15), .5)]
    quotes = scheduled_prices([("A", scheduled_time(0), 100.)])
    result, _, _ = scheduled_run(rows, quotes)
    assert result["trades"].empty
    assert result["ledger"].aum.eq(1000).all()
    assert "no dollar bars in evaluation interval" in result["exclusions"].reason.tolist()
    assert len(result["exclusions"]) == 3


def test_marks_only_held_prices_without_execution_or_minute_records():
    quotes = scheduled_prices([("A", scheduled_time(0), 100.), ("A", scheduled_time(1), 100.),
                     ("A", scheduled_time(3), 110.), ("A", scheduled_time(24), 120.)])
    result, _, _ = scheduled_run([("A", scheduled_time(0), scheduled_time(5), .5)], quotes)
    assert result["trades"].timestamp.tolist() == [scheduled_time(1), scheduled_time(24)]
    ledger = result["ledger"].set_index("timestamp")
    assert scheduled_time(3) not in ledger.index
    assert ledger.loc[scheduled_time(5), "aum"] == pytest.approx(1025.)
    assert ledger.iloc[-1].aum == pytest.approx(1050.)
    assert result["trades"].iloc[-1].reason == "research_end"


def test_symbol_cache_is_bounded_and_does_not_reload_original_parquet(tmp_path):
    paths = ResearchPaths(tmp_path, start=SCHEDULED_START.normalize(), end=SCHEDULED_START.normalize() + pd.Timedelta(days=1))
    for i in range(35):
        file = paths.feature(f"S{i}", "dollar_bars")
        file.parent.mkdir(parents=True)
        pd.DataFrame({"end": [scheduled_time(1), scheduled_time(2)], "close": [100., 101.]}).to_parquet(file)
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), paths=paths) as index:
        for i in range(35): assert index.terminal(f"S{i}") == scheduled_time(2)
        assert len(index.cache) == 32
        assert index.next_after("S0", scheduled_time(0)) == (scheduled_time(1), 100.)
        assert index.counts["loads"] == 35


@pytest.mark.parametrize("bad", ["missing", "duplicate", "reverse", "zero", "nan"])
def test_invalid_selected_symbol_prices_fail_and_release_temporary_files(bad):
    frame = scheduled_prices([("A", scheduled_time(1), 100.), ("A", scheduled_time(2), 101.)])
    if bad == "missing": frame.loc[1, "timestamp"] = pd.NaT
    if bad == "duplicate": frame.loc[1, "timestamp"] = scheduled_time(1)
    if bad == "reverse": frame = frame.iloc[::-1]
    if bad == "zero": frame.loc[1, "price"] = 0.
    if bad == "nan": frame.loc[1, "price"] = float("nan")
    with pytest.raises(ValueError):
        with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), prices=frame) as index:
            directory = Path(index.directory.name)
            index.terminal("A")
    assert not directory.exists()


def test_missing_selected_symbol_file_has_no_tick_fallback(tmp_path):
    paths = ResearchPaths(tmp_path, start=SCHEDULED_START.normalize(), end=SCHEDULED_START.normalize() + pd.Timedelta(days=1))
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), paths=paths) as index:
        with pytest.raises(ValueError, match="run dollar-bar preprocessing"):
            index.terminal("MISSING")


@pytest.mark.parametrize("failure", [False, True])
def test_decision_progress_closes_on_success_and_price_failure(monkeypatch, failure):
    from io import StringIO
    from tqdm import tqdm
    bars = []
    def progress(*args, **kwargs):
        bar = tqdm(*args, file=StringIO(), **kwargs)
        bars.append(bar)
        return bar
    monkeypatch.setattr(portfolio_management, "tqdm", progress)
    quotes = scheduled_prices([("A", scheduled_time(1), 0. if failure else 100.), ("A", scheduled_time(6), 101.)])
    item = scheduled_account([("A", scheduled_time(0), scheduled_time(5), .5)])
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), prices=quotes) as index:
        decisions = set(scheduled_calendar().open) | set(scheduled_calendar().close)
        scheduler = SynchronousScheduler({0: item}, index, decisions, scheduled_time(0), scheduled_time(30))
        if failure:
            with pytest.raises(ValueError, match="finite and positive"):
                scheduler.run(show_progress=True)
        else:
            scheduler.run(show_progress=True)
            assert bars[0].n == bars[0].total == len(decisions)
    assert len(bars) == 1 and bars[0].disable


def test_selected_parquet_with_null_end_is_not_silently_filtered_out(tmp_path):
    paths = ResearchPaths(tmp_path, start=SCHEDULED_START.normalize(), end=SCHEDULED_START.normalize() + pd.Timedelta(days=1))
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": [scheduled_time(1), pd.NaT], "close": [100., 101.]}).to_parquet(file)
    with SynchronousPriceIndex(scheduled_time(0), scheduled_time(30), paths=paths) as index:
        with pytest.raises(ValueError, match="Nonmissing"):
            index.terminal("A")
