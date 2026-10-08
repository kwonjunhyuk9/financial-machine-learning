import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.tree import DecisionTreeClassifier

import src.backtesting.portfolio_management as portfolio_management
from src.backtesting.portfolio_management import (
    PortfolioSettings,
    STRATEGIES,
    _position_values,
    _weight_to_position,
    benchmark_returns,
    candidate_snapshot,
    elapsed_session_minutes,
    entry_schedule,
    holding_statistics,
    load_event_entry_prices,
    load_session_prices,
    load_strategy_comparison,
    select_synchronous,
    simulate_cross_sectional,
    simulate_synchronous,
    summarize_account,
)
from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions,
    combinatorial_purged_cross_validation,
)
from src.modeling.model_workflow import (
    build_primary_model_frame, build_meta_model_frame, generate_oof_predictions,
    get_primary_feature_columns, get_meta_feature_columns,
)
from src.modeling.purged_validation import PurgedKFold
from src.preprocessing.market_technical_indicators import MODEL_FEATURES, TECHNICAL_FEATURES
from src.preprocessing.market_data import ResearchPaths
import src.preprocessing.market_data as market_data


def test_benchmark_returns_uses_last_spy_trade_each_day(tmp_path):
    paths = ResearchPaths(tmp_path, period="2025-02-01_2025-12-31")
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
        data_start=pd.Timestamp("2025-01-01", tz="UTC"),
    )

    assert result.tolist() == pytest.approx([0.0, 0.1])


def test_load_event_entry_prices_preserves_composite_index(tmp_path):
    paths = ResearchPaths(tmp_path, period="2025-02-01_2025-12-31")
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


@pytest.mark.parametrize("risk_free, annual_periods, benchmark_sharpe", [
    (.03, 365.25, 1.0), (.07, 252., .5),
])
def test_summarize_account_reports_current_result_contract(risk_free, annual_periods, benchmark_sharpe):
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

    statistics = summarize_account(result, settings, spy_returns, annual_risk_free_rate=risk_free, periods_per_year=annual_periods, annualized_benchmark_sharpe_ratio=benchmark_sharpe)
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


    daily = portfolio_management._daily_account(result["ledger"].set_index("timestamp"), settings.initial_aum)
    expected = portfolio_management.Efficiency.portfolio_statistics(
        daily.net_return, daily.period_start, annual_risk_free_rate=risk_free,
        periods_per_year=annual_periods, annualized_benchmark_sharpe_ratio=benchmark_sharpe,
    )
    assert values["efficiency", "annualized_sharpe_ratio"] == pytest.approx(expected["annualized_sharpe"])


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
    assert [_weight_to_position(weight, k=5) for weight in weights] == [0, 40, -70, 99, -99]


@pytest.mark.parametrize(
    ("side", "resize_limit", "expected_entries"),
    [(1, 200.0, 2), (1, 0.0, 1), (-1, 0.0, 2), (-1, 200.0, 1)],
)
def test_same_symbol_increase_uses_incremental_limit(
    monkeypatch, side, resize_limit, expected_entries
):
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    symbol = "S00" if side > 0 else "S24"
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
        settings=PortfolioSettings(),
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
        settings=PortfolioSettings(),
    )

    reductions = result["trades"].loc[
        result["trades"].symbol.eq("S00")
        & result["trades"].reason.eq("signal_reduction")
    ]
    assert len(reductions) == 1
    assert reductions.timestamp.item() == start + pd.Timedelta(minutes=3)


def test_price_changes_alone_do_not_resize_positions():
    start, calendar, events, calibration, _ = cross_sectional_scenario()

    def observations():
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol,
                "price": 100.0 + minute / 10,
            })

    result = simulate_cross_sectional(
        events, observations(), calibration, calendar, calendar.close.iloc[0],
        settings=PortfolioSettings(),
    )

    assert not result["trades"].reason.eq("signal_reduction").any()
    entries = result["trades"].loc[result["trades"].reason.eq("signal_entry")]
    assert not entries.duplicated("symbol").any()


def test_account_uses_subsequent_quotes_caps_and_final_liquidation():
    start, calendar, events, calibration, observations = cross_sectional_scenario()
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
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
    again = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
    pd.testing.assert_frame_equal(trades, again["trades"])


def test_development_events_do_not_enter_holdout_average():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    development = events.copy()
    development["partition"] = "development"
    development["primary_side"] *= -1
    base = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
    result = simulate_cross_sectional(pd.concat([development, events]), observations(), calibration,
                                      calendar, calendar.close.iloc[0], settings=PortfolioSettings())
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
        settings=PortfolioSettings(),
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        statistics = summarize_account(
            result,
            PortfolioSettings(),
            pd.Series([0.0], index=pd.DatetimeIndex([start])),
            annual_risk_free_rate=0.03, periods_per_year=365.25, annualized_benchmark_sharpe_ratio=1.0,
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
                                 calendar, calendar.close.iloc[0], settings=PortfolioSettings())


def test_pending_entry_that_violates_limit_never_fills():
    start, calendar, events, calibration, _ = cross_sectional_scenario()
    events.loc[events.primary_side.eq(-1), "meta_action"] = 0

    def observations():
        for minute in range(11):
            yield start + pd.Timedelta(minutes=minute), pd.DataFrame({
                "symbol": events.symbol, "price": 100. if minute == 0 else 200.,
            })

    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
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

    result = simulate_cross_sectional(events, marked(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
    exits = result["trades"].loc[result["trades"].reason.eq("signal_reduction")]
    assert not exits.empty
    assert exits.timestamp.min() == start + pd.Timedelta(minutes=3)


def test_missing_calibration_excludes_trading_but_reports_symbol():
    _, calendar, events, calibration, observations = cross_sectional_scenario()
    calibration.loc[calibration.symbol.eq("S00"), "w"] = np.nan
    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
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

    result = simulate_cross_sectional(events, observations(), calibration, calendar, calendar.close.iloc[0], settings=PortfolioSettings())
    replacement = result["trades"].loc[result["trades"].reason.str.startswith("replacement")]
    assert not replacement.empty
    assert replacement.timestamp.min() == start + pd.Timedelta(minutes=4)
    assert set(replacement.loc[
        replacement.timestamp.eq(replacement.timestamp.min()), "reason"
    ]) == {"replacement_entry", "replacement_exit"}


def calendar():
    opens = pd.to_datetime(["2025-01-03 14:30Z", "2025-01-06 14:30Z", "2025-01-07 14:30Z"])
    return pd.DataFrame({"open": opens, "close": opens + pd.Timedelta(hours=6, minutes=30)})


def events(rows):
    return pd.DataFrame(rows, columns=["symbol", "event_start", "primary_side", "meta_probability", "meta_action"]).assign(
        event_start=lambda x: pd.to_datetime(x.event_start, format="mixed", utc=True), partition="holdout")


def prices(cal, symbols=("A", "B")):
    return pd.DataFrame([{"timestamp": t, "symbol": s, "price": 100.0}
                         for t in sorted([*cal.open, *cal.close]) for s in symbols])


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
    selected = select_synchronous(data, entry_schedule(cal, cal.close.max()), k=2)
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
    concurrent = result['exposures'].loc[lambda x: x.timestamp.eq(cal.open.iloc[1])]
    assert len(concurrent) == 2
    assert set(concurrent.group) == {"open", "close"}
    assert concurrent.quantity.prod() < 0
    values = _position_values(result, pd.DatetimeIndex(result['ledger'].timestamp))
    assert values.shape[1] == 2
    assert len(result['trades']) == 6
    assert len(result['trades'].loc[lambda x: x.timestamp.eq(cal.close.iloc[1])]) == 3
    assert result['closed_trades'].net_pnl.sum() == pytest.approx(result['ledger'].iloc[-1].aum - 10000)
    assert result['ledger'].iloc[-1].aum < 10000
    groups = result['group_ledger'].pivot(index='timestamp',columns='group',values='aum')
    assert groups.iloc[0].tolist() == [5000,5000]
    assert groups.loc[cal.open.iloc[1], 'open'] != groups.loc[cal.open.iloc[1], 'close']
    stats = summarize_account(result, settings, pd.Series([0.,0.,0.], index=cal.close.dt.normalize()), annual_risk_free_rate=0.03, periods_per_year=365.25, annualized_benchmark_sharpe_ratio=1.0)
    assert not stats.empty
    holdings = holding_statistics(result).set_index('metric').value
    assert holdings.maximum_positions == 2
    assert holdings.maximum_unique_symbols == 1


def test_missing_entry_keeps_its_allocation_and_missing_exit_fails():
    cal = calendar()
    data = events([("A", "2025-01-03 14:00Z",1,.9,1),("B", "2025-01-03 14:00Z",1,.8,1)])
    quotes = prices(cal)
    quotes = quotes.loc[~(quotes.timestamp.eq(cal.open.iloc[0]) & quotes.symbol.eq("B"))]
    settings = PortfolioSettings(initial_aum=10000, broker_fee_bps=0, slippage_bps=0)
    result = simulate_synchronous(data, quotes, cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(),settings)
    assert len(result['exclusions']) == 1
    assert result['trades'].iloc[0].traded_value == 1250
    quotes = quotes.loc[~(quotes.timestamp.eq(cal.close.iloc[0]) & quotes.symbol.eq("A"))]
    with pytest.raises(ValueError,match='liquidation price'):
        simulate_synchronous(data, quotes, cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(), settings)


def test_maximum_twenty_positions_and_final_round_has_no_new_overnight_entry():
    cal=calendar(); rows=[]
    for time in ["2025-01-03 16:00Z", "2025-01-06 14:00Z", "2025-01-07 16:00Z"]:
        rows.extend((f'{side}_{i}',time,side,.9,1) for side in [-1,1] for i in range(8))
    data=events(rows)
    result=simulate_synchronous(data,prices(cal,data.symbol.unique()),cal,pd.Timestamp("2025-01-03 13:00Z"),cal.close.max(), settings=PortfolioSettings())
    assert result['exposures'].groupby('timestamp').size().max()==20
    assert not result['trades'].loc[lambda x:x.timestamp.eq(cal.close.max()) & x.reason.eq('scheduled_entry')].shape[0]


def test_early_close_cutoff_and_late_feature_completion():
    cal=calendar();cal.loc[0,'close']=pd.Timestamp('2025-01-03 18:00Z')
    data=events([('A','2025-01-03 17:59Z',1,.9,1),('B','2025-01-03 17:59:01Z',1,.9,1)])
    selected=select_synchronous(data,entry_schedule(cal,cal.close.max())).set_index('symbol')
    assert selected.loc['A','entry']==cal.close.iloc[0]
    assert selected.loc['B','entry']==cal.open.iloc[1]


def test_tick_proxies_ignore_extended_hours(tmp_path):
    cal=calendar().iloc[:1];paths=ResearchPaths(tmp_path, period="2025-02-01_2025-12-31")
    file=paths.raw('A','tick')/'2025-01-03.parquet';file.parent.mkdir(parents=True)
    pd.DataFrame({'timestamp':pd.to_datetime(['2025-01-03 12:00Z','2025-01-03 14:31Z','2025-01-03 20:59Z','2025-01-03 22:00Z']),
                  'price':[1.,100.,110.,999.]}).to_parquet(file)
    file.with_suffix('.json').write_text('{}')
    result=load_session_prices(paths,cal,['A'])
    assert result.price.tolist()==[100.,110.]


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
    changed.loc[changed.timestamp.gt(cal.open.iloc[0]), "price"] = 110
    args = (cal, pd.Timestamp("2025-01-03 13:00Z"), cal.close.max(), PortfolioSettings())
    first = simulate_synchronous(data, original, *args)
    second = simulate_synchronous(data, changed, *args)
    pd.testing.assert_frame_equal(first["trades"].head(1), second["trades"].head(1))
    assert second["group_ledger"].loc[lambda x: x.group.eq("close"), "aum"].eq(50000).all()
    assert second["ledger"].iloc[-1].aum > first["ledger"].iloc[-1].aum


@pytest.fixture
def strategy_workspace(tmp_path, monkeypatch):
    symbols = [f"S{i:02}" for i in range(50)]
    opens = pd.to_datetime(["2025-01-03 14:30Z", "2025-01-06 14:30Z", "2025-01-07 14:30Z"])
    calendar = pd.DataFrame({"open": opens, "close": opens + pd.Timedelta(minutes=10)})
    common = ResearchPaths(tmp_path, period="2025-02-01_2025-12-31")
    common.universe.mkdir(parents=True)
    pd.DataFrame({"symbol": symbols}).to_csv(common.universe / "sp500_2025.csv", index=False)
    calendar.to_parquet(common.universe / "sessions.parquet")
    (common.universe / "sessions.json").write_text(json.dumps({
        "start": "2025-01-01T00:00Z", "end": "2026-01-01T00:00Z",
    }))
    monkeypatch.setattr(market_data, "load_manifest", lambda paths, **kwargs: pd.DataFrame({"symbol": symbols}))
    rows = []
    for start, partition in [(opens[0] - pd.Timedelta(hours=hours), "development") for hours in (48, 36, 24, 12)] + [
                             (opens[0], "holdout"),
                             (opens[0] + pd.Timedelta(minutes=2), "holdout")]:
        for i, symbol in enumerate(symbols):
            side = 1 if i < 25 else -1
            rows.append({"symbol": symbol, "event_start": start,
                         "event_end": start + pd.Timedelta(minutes=5),
                         "vertical_barrier": start + pd.Timedelta(minutes=6), "target_return": .5,
                         "raw_return": .01 * side * (1 if i % 3 else -1), "direction_label": side,
                         "sample_weight": 1., "mean_sentiment_score": 50 - i,
                         "partition": partition, "holdout_boundary": opens[0] - pd.Timedelta(minutes=3),
                         "primary_side": side, "primary_probability": .9 if side == 1 else .1,
                         "meta_probability": 1., "meta_action": 1,
                         "meta_label": int((i % 2 == 1) == (side == 1))})
    combined = pd.DataFrame(rows)
    model_events = combined.drop(columns=["primary_side", "primary_probability", "meta_probability", "meta_action", "meta_label"])
    for feature in MODEL_FEATURES:
        if feature not in model_events:
            model_events[feature] = 0.0
    model_events["fractionally_differenced_log_close"] = combined.primary_side
    model_events[TECHNICAL_FEATURES[0]] = [
        float(int(symbol[1:]) % 3 != 0) if partition == "development" else 1.0
        for symbol, partition in zip(model_events.symbol, model_events.partition)
    ]
    common.event("model").parent.mkdir(parents=True)
    model_events.to_parquet(common.event("model"), index=False)
    for kind in ("market", "sentiment"):
        paths = ResearchPaths(tmp_path, kind, period="2025-02-01_2025-12-31")
        paths.artifacts.mkdir(parents=True)
        frame = build_primary_model_frame(model_events, model_events[["symbol", "event_start"]], kind)
        train = frame.loc[frame.partition.eq("development")]
        primary_features = get_primary_feature_columns(frame, kind)
        primary = DecisionTreeClassifier(max_depth=3, random_state=42)
        oof = generate_oof_predictions(
            primary, train[primary_features], train.direction_label, train.sample_weight,
            PurgedKFold(2, t1=train.event_end), positive_label=1,
        )
        meta_train = build_meta_model_frame(train, oof[["prediction", "probability", "prediction_source"]])
        primary.fit(train[primary_features], train.direction_label)
        predictions = frame.copy()
        predictions["primary_side"] = primary.predict(frame[primary_features])
        predictions["primary_probability"] = primary.predict_proba(frame[primary_features])[:, list(primary.classes_).index(1)]
        predictions["primary_confidence"] = np.maximum(predictions.primary_probability, 1 - predictions.primary_probability)
        meta_features = get_meta_feature_columns(meta_train, kind)
        meta = DecisionTreeClassifier(max_depth=3, random_state=42).fit(meta_train[meta_features], meta_train.meta_label)
        predictions["meta_action"] = meta.predict(predictions[meta_features])
        predictions["meta_probability"] = meta.predict_proba(predictions[meta_features])[:, list(meta.classes_).index(1)]
        predictions["meta_label"] = (predictions.primary_side * predictions.raw_return > 0).astype(int)
        predictions.reset_index().to_parquet(paths.artifacts / "meta_predictions.parquet", index=False)
        pd.DataFrame({"symbol": symbols, "w": 1., "reason": ""}).to_parquet(paths.artifacts / "price_calibration.parquet")
    for session in calendar.itertuples():
        for symbol in [*symbols, "SPY"]:
            file = common.raw(symbol, "tick") / f"{session.open.date()}.parquet"
            file.parent.mkdir(parents=True, exist_ok=True)
            times = pd.date_range(session.open, session.close, freq="min")
            pd.DataFrame({"timestamp": times, "symbol": symbol, "price": 100. + np.arange(len(times)) * .001}).to_parquet(file,index=False)
            file.with_suffix(".json").write_text("{}")
    pd.DataFrame({"timestamp": [opens[0] - pd.Timedelta(days=1)], "symbol": ["SPY"], "price": [100.]}).to_parquet(
        common.raw("SPY", "tick") / "2025-01-02.parquet", index=False)
    return tmp_path


def test_three_saved_strategies_and_notebook_statistics_execute(strategy_workspace, monkeypatch):
    root = strategy_workspace
    source_root = Path(__file__).resolve().parents[2]
    accounts = {}
    for strategy in STRATEGIES:
        directory = root / "notebooks/backtesting" / strategy
        directory.mkdir(parents=True)
        monkeypatch.chdir(directory)
        notebook = json.loads((source_root / "notebooks/backtesting" / strategy / "backtest_statistics.ipynb").read_text())
        namespace = {}
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                exec(compile(''.join(cell['source']), f'{strategy}:cell{i}', 'exec'), namespace)
        result = namespace['result']
        accounts[strategy] = result
        assert result['ledger'].iloc[0].aum == 100000
        assert result['ledger'].iloc[-1].gross_exposure == 0
        assert len(result['trades']) > 0
        assert result['closed_trades'].net_pnl.sum() == pytest.approx(result['ledger'].iloc[-1].aum - 100000)
    async_exposure = accounts['sentiment_asynchronous']['exposures']
    assert async_exposure.groupby('timestamp').size().max() == 20
    assert async_exposure.groupby('timestamp').weight.apply(lambda x: (x > 0).sum()).max() == 10
    assert async_exposure.groupby('timestamp').weight.apply(lambda x: (x < 0).sum()).max() == 10
    monkeypatch.chdir(root / 'notebooks/backtesting')
    comparison = json.loads((source_root / 'notebooks/backtesting/strategy_comparison.ipynb').read_text())
    namespace = {}
    for i, cell in enumerate(comparison['cells']):
        if cell['cell_type'] == 'code':
            exec(compile(''.join(cell['source']), f'comparison:cell{i}', 'exec'), namespace)
    assert set(namespace['statistics'].strategy) == set(STRATEGIES)
    assert namespace['cumulative_returns'].notna().all().all()
    file = root / 'data/backtest_results/sentiment_synchronous/comparison.json'
    metadata = json.loads(file.read_text()); metadata['initial_aum'] = 123
    file.write_text(json.dumps(metadata))
    with pytest.raises(ValueError,match='different evaluation'):
        load_strategy_comparison(root)


def test_model_artifact_paths_share_only_within_family(tmp_path):
    market = ResearchPaths(tmp_path, 'market', period="2025-02-01_2025-12-31")
    sentiment = ResearchPaths(tmp_path, 'sentiment', period="2025-02-01_2025-12-31")
    assert market.artifacts != sentiment.artifacts
    assert market.event('model') == sentiment.event('model')
    assert market.raw('A','tick') == sentiment.raw('A','tick')
