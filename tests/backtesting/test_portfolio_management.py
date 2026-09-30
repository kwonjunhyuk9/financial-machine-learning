import warnings

import numpy as np
import pandas as pd
import pytest

from src.backtesting.portfolio_management import (
    PortfolioSettings,
    benchmark_returns,
    candidate_snapshot,
    elapsed_session_minutes,
    load_event_entry_prices,
    simulate_cross_sectional,
    summarize_account,
)
from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions,
    combinatorial_purged_cross_validation,
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


def test_load_event_entry_prices_preserves_composite_index(tmp_path):
    paths = ResearchPaths(tmp_path)
    starts = pd.date_range("2025-02-03T14:30Z", periods=2, freq="min")
    file = paths.feature("A", "dollar_bars")
    file.parent.mkdir(parents=True)
    pd.DataFrame({"end": starts, "close": [100.0, 101.0]}).to_parquet(file)
    events = pd.DataFrame(
        {"target_return": [0.01, 0.02]},
        index=pd.MultiIndex.from_arrays(
            [["A", "A"], starts], names=["symbol", "event_start"]
        ),
    )

    result = load_event_entry_prices(paths, events)

    assert result.index.equals(events.index)
    assert result["entry_price"].tolist() == [100.0, 101.0]


def test_summarize_account_reports_current_result_contract():
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
    spy_returns = pd.Series([0.0, 0.05], index=timestamps)

    statistics = summarize_account(result, settings, spy_returns)
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
    assert values["coverage", "excluded_price_calibrations"] == 1
    assert values["coverage", "closed_positions"] == 1


def cross_sectional_scenario():
    start = pd.Timestamp("2025-01-02 14:30Z")
    calendar = pd.DataFrame({"open": [start], "close": [start + pd.Timedelta(minutes=10)]})
    symbols = [f"S{i:02}" for i in range(25)]
    events = pd.DataFrame({"symbol": symbols, "event_start": start,
        "vertical_barrier": start + pd.Timedelta(minutes=8), "target_return": .5,
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
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        statistics = summarize_account(
            result,
            PortfolioSettings(),
            pd.Series([0.0], index=pd.DatetimeIndex([start])),
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


def test_no_tail_of_five_means_no_new_entries():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    active = events.head(24).assign(entry_price=100.)
    table = candidate_snapshot(active, dict.fromkeys(active.symbol, 100.),
        calibration.set_index(["symbol", "event_start"]).w,
        start, calendar, PortfolioSettings())
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
    calibration = pd.concat([
        calibration,
        pd.DataFrame({
            "symbol": ["S05"],
            "event_start": [newcomer.event_start.item()],
            "w": [1.0],
        }),
    ], ignore_index=True)

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
