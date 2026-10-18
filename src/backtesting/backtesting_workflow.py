"""Stage-by-stage development and holdout evaluation of tuned meta families."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.backtesting.backtest_statistics import (
    ClassificationScores, Efficiency, GeneralCharacteristics,
    ImplementationShortfall, Performance, Runs,
)
from src.backtesting.bet_sizing import event_bet_signals
from src.backtesting.portfolio_management import (
    PortfolioSettings, STRATEGIES, benchmark_returns, calibrate_price_sizing,
    entry_schedule, load_event_entry_prices, load_session_prices,
    select_synchronous, simulate_strategy, simulate_synchronous,
)
from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions, combinatorial_purged_cross_validation,
    generate_cpcv_predictions, get_cpcv_price_calibrations,
)
from src.modeling.modeling_workflow import build_primary_model_frame, model_artifact_hash
from src.modeling.purged_validation import index_events
from src.preprocessing.market_data import load_manifest, manifest_hash, save_frame, sessions

META_FAMILIES = ("boosting", "bagging", "random_forest")


def _daily_account(ledger: pd.DataFrame, initial_aum: float) -> pd.DataFrame:
    """Sample calendar-day account closes with explicit return intervals."""
    daily = ledger[["aum"]].resample("D").last().ffill()
    ends = daily.index + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    daily.index = pd.DatetimeIndex(
        [min(time, ledger.index[-1]) for time in ends],
        name="timestamp",
    )
    daily["period_start"] = pd.Series(
        [ledger.index[0], *daily.index[:-1]],
        index=daily.index,
    )
    previous_aum = daily["aum"].shift(1)
    previous_aum.iloc[0] = initial_aum
    daily["net_return"] = daily["aum"] / previous_aum - 1.0
    return daily


def _position_values(result: dict[str, pd.DataFrame], index: pd.DatetimeIndex) -> pd.DataFrame:
    """Align per-symbol dollar positions to the account observation clock."""
    exposures = result["exposures"]
    if exposures.empty:
        return pd.DataFrame({"_flat": 0.0}, index=index)
    return exposures.pivot(
        index="timestamp",
        columns=["group", "symbol"] if "group" in exposures else "symbol",
        values="position_value",
    ).reindex(index).fillna(0.0)


def _closed_trade_inputs(
    closed_trades: pd.DataFrame,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return aligned position, end-time, and return series for closed episodes."""
    starts = pd.DatetimeIndex(closed_trades["event_start"])
    positions = pd.Series(closed_trades["side"].to_numpy(), index=starts)
    event_end = pd.Series(
        pd.DatetimeIndex(closed_trades["event_end"]),
        index=starts,
    )
    returns = pd.Series(closed_trades["net_return"].to_numpy(), index=starts)
    return positions, event_end, returns


def summarize_account(result: dict[str, pd.DataFrame], settings: PortfolioSettings,
                      spy_returns: pd.Series, *, annual_risk_free_rate: float,
                      periods_per_year: float, annualized_benchmark_sharpe_ratio: float) -> pd.DataFrame:
    """Compute the complete normalized statistics contract for one account."""
    ledger = result["ledger"].set_index("timestamp")
    if ledger.empty:
        raise ValueError("No account observations")
    daily = _daily_account(ledger, settings.initial_aum)
    position_values = _position_values(result, ledger.index)
    closed = result["closed_trades"]
    start = GeneralCharacteristics.start(ledger.index)
    end = GeneralCharacteristics.end(ledger.index)
    final = ledger.iloc[-1]
    increments = ledger[["broker_fee", "slippage", "traded_value"]].diff()
    increments.iloc[0] = ledger[["broker_fee", "slippage", "traded_value"]].iloc[0]
    net_dollar_performance = pd.Series(
        [final.aum - settings.initial_aum],
        dtype="float64",
    )
    equity = pd.concat([
        pd.Series([settings.initial_aum], index=pd.DatetimeIndex([start])),
        ledger["aum"],
    ])
    rows = []

    def add(section, metric, value=np.nan, unit="ratio", entity="final_account",
            timestamp=pd.NaT):
        rows.append({
            "section": section,
            "metric": metric,
            "entity": entity,
            "value": float(value),
            "timestamp": timestamp,
            "unit": unit,
        })

    add("general_characteristics", "start", unit="timestamp", timestamp=start)
    add("general_characteristics", "end", unit="timestamp", timestamp=end)
    general = {
        "average_aum": GeneralCharacteristics.average_aum(daily["aum"]),
        "leverage": GeneralCharacteristics.leverage(position_values, ledger["aum"]),
        "maximum_dollar_position_size": (
            GeneralCharacteristics.maximum_dollar_position_size(position_values)
        ),
        "annualized_turnover": GeneralCharacteristics.annualized_turnover(
            increments["traded_value"], daily["aum"], time_range=(start, end),
        ),
    }
    account_returns = daily["net_return"].copy()
    account_returns.index = account_returns.index.normalize().as_unit("ns")
    benchmark = spy_returns.copy()
    benchmark.index = benchmark.index.normalize().as_unit("ns")
    general["correlation_to_underlying"] = (
        GeneralCharacteristics.correlation_to_underlying(account_returns, benchmark)
    )
    if closed.empty:
        general.update({
            "ratio_of_longs": np.nan,
            "frequency_of_bets": 0.0,
            "average_holding_period": np.nan,
        })
        positions = event_end = trade_returns = None
    else:
        positions, event_end, trade_returns = _closed_trade_inputs(closed)
        general.update({
            "ratio_of_longs": GeneralCharacteristics.ratio_of_longs(positions),
            "frequency_of_bets": GeneralCharacteristics.frequency_of_bets(
                positions, event_end, time_range=(start, end),
            ),
            "average_holding_period": GeneralCharacteristics.average_holding_period(
                positions, event_end,
            ),
        })
    general_units = {
        "average_aum": "USD",
        "maximum_dollar_position_size": "USD",
        "frequency_of_bets": "count/year",
        "average_holding_period": "days",
    }
    for metric in [
        "average_aum", "leverage", "maximum_dollar_position_size",
        "ratio_of_longs", "frequency_of_bets", "average_holding_period",
        "annualized_turnover", "correlation_to_underlying",
    ]:
        add("general_characteristics", metric, general[metric], general_units.get(metric, "ratio"))

    net_pnl = final.aum - settings.initial_aum
    performance = {
        "pnl": Performance.pnl(net_dollar_performance),
        "annualized_rate_of_return": Performance.annualized_rate_of_return(
            daily["net_return"], start=start, end=end,
        ),
    }
    if closed.empty:
        performance.update({
            "pnl_from_long_positions": 0.0,
            "pnl_from_short_positions": 0.0,
            "hit_ratio": np.nan,
            "average_return_from_hits": np.nan,
            "average_return_from_misses": np.nan,
        })
    else:
        performance.update({
            "pnl_from_long_positions": Performance.pnl_from_long_positions(
                closed["net_pnl"], closed["side"],
            ),
            "pnl_from_short_positions": Performance.pnl_from_short_positions(
                closed["net_pnl"], closed["side"],
            ),
            "hit_ratio": Performance.hit_ratio(trade_returns),
            "average_return_from_hits": Performance.average_return_from_hits(trade_returns),
            "average_return_from_misses": Performance.average_return_from_misses(trade_returns),
        })
    for metric, value in performance.items():
        add("performance", metric, value, "USD" if "pnl" in metric else "ratio")

    runs = {
        "hhi_positive_returns": Runs.hhi_positive_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_negative_returns": Runs.hhi_negative_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_time_between_bets": Runs.hhi_time_between_bets(trade_returns)
        if trade_returns is not None else np.nan,
        "percentile_drawdown": Runs.percentile_drawdown(equity),
        "percentile_time_under_water": Runs.percentile_time_under_water(equity),
    }
    for metric, value in runs.items():
        add("runs", metric, value, "years" if "time_under_water" in metric else "ratio")

    shortfall = {
        "broker_fees_per_turnover": ImplementationShortfall.broker_fees_per_turnover(
            increments["broker_fee"], increments["traded_value"],
        ),
        "average_slippage_per_turnover": (
            ImplementationShortfall.average_slippage_per_turnover(
                increments["slippage"], increments["traded_value"],
            )
        ),
        "dollar_performance_per_turnover": (
            ImplementationShortfall.dollar_performance_per_turnover(
                net_dollar_performance, increments["traded_value"],
            )
        ),
        "return_on_execution_costs": ImplementationShortfall.return_on_execution_costs(
            net_dollar_performance,
            increments["broker_fee"] + increments["slippage"],
        ),
    }
    for metric, value in shortfall.items():
        add("implementation_shortfall", metric, value)

    efficiency = Efficiency.portfolio_statistics(
        daily["net_return"],
        daily["period_start"],
        annual_risk_free_rate=annual_risk_free_rate,
        periods_per_year=periods_per_year,
        annualized_benchmark_sharpe_ratio=annualized_benchmark_sharpe_ratio,
    )
    for metric, value in efficiency.items():
        name = "annualized_sharpe_ratio" if metric == "annualized_sharpe" else metric
        add("efficiency", name, value)

    if not result["exposures"].empty:
        exposure = result["exposures"].pivot(
            index="timestamp", columns=["group", "symbol"] if "group" in result["exposures"] else "symbol", values="weight"
        ).reindex(ledger.index).fillna(0)
        durations = ledger.index.to_series().shift(-1).sub(
            ledger.index.to_series()
        ).dt.total_seconds().fillna(0)
        for symbol in exposure:
            add("exposure", "mean_absolute_weight",
                (exposure[symbol].abs() * durations).sum() / durations.sum()
                if durations.sum() else np.nan, entity=":".join(symbol) if isinstance(symbol, tuple) else symbol)
            add("exposure", "max_absolute_weight", exposure[symbol].abs().max(), entity=":".join(symbol) if isinstance(symbol, tuple) else symbol)
    add("benchmark", "net_return", (1 + spy_returns).prod() - 1, entity="SPY")
    for metric, value in {
        "initial_aum": settings.initial_aum, "K": settings.k,
        "one_way_broker_fee_bps": settings.broker_fee_bps,
        "one_way_slippage_bps": settings.slippage_bps,
        "cash_interest": 0, "borrow_fee": 0,
    }.items():
        add("assumption", metric, value, "setting")
    exclusion_metric = "excluded_entry_prices" if "group_ledger" in result else "excluded_price_calibrations"
    add("coverage", exclusion_metric, len(result["exclusions"]), "count")
    add("coverage", "closed_positions", len(result["closed_trades"]), "count")
    statistics = pd.DataFrame(rows)
    statistics["timestamp"] = pd.to_datetime(statistics["timestamp"], utc=True)
    if not np.isclose(closed["net_pnl"].sum() if not closed.empty else 0.0, net_pnl):
        raise ValueError("Closed-position PnL does not reconcile with final account AUM")
    return statistics


def holding_statistics(result: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Time-weight holdings and exposure; keep book positions distinct."""
    ledger = result["ledger"].set_index("timestamp")
    exposures = result["exposures"]
    if exposures.empty:
        counts = unique = pd.Series(0.0, index=ledger.index)
    else:
        counts = exposures.groupby("timestamp").size().reindex(ledger.index, fill_value=0)
        unique = exposures.groupby("timestamp").symbol.nunique().reindex(ledger.index, fill_value=0)
    durations = ledger.index.to_series().shift(-1).sub(ledger.index.to_series()).dt.total_seconds().fillna(0)
    def average(values):
        return float((values * durations).sum() / durations.sum()) if durations.sum() else np.nan
    rows = []
    for name, values in (("positions", counts), ("unique_symbols", unique),
                         ("gross_exposure", ledger.gross_exposure)):
        rows.extend([{"metric": f"average_{name}", "value": average(values)},
                     {"metric": f"maximum_{name}", "value": float(values.max())}])
    rows.extend([
        {"metric": "average_cash_balance_ratio", "value": average(ledger.cash / ledger.aum)},
        {"metric": "average_unallocated_capital_ratio", "value": average((1 - ledger.gross_exposure).clip(lower=0))},
    ])
    result = pd.DataFrame(rows).assign(section="holdings", entity="final_account", timestamp=pd.NaT, unit="ratio")
    result.loc[result.metric.str.endswith(("positions", "unique_symbols")), "unit"] = "count"
    return result


def classification_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Evaluate primary/meta classifiers on the supplied OOS event predictions."""
    classification = {
        "primary": {
            "y_true": predictions["direction_label"],
            "y_pred": predictions["primary_side"],
            "probabilities": np.column_stack([
                1.0 - predictions["primary_probability"],
                predictions["primary_probability"],
            ]),
            "labels": [-1, 1],
        },
        "meta": {
            "y_true": predictions["meta_label"],
            "y_pred": predictions["meta_action"],
            "probabilities": np.column_stack([
                1.0 - predictions["meta_probability"],
                predictions["meta_probability"],
            ]),
            "labels": [0, 1],
        },
    }
    rows = []
    for entity, inputs in classification.items():
        values = {
            "accuracy": ClassificationScores.accuracy(
                inputs["y_true"], inputs["y_pred"], predictions["sample_weight"],
            ),
            "precision": ClassificationScores.precision(
                inputs["y_true"], inputs["y_pred"], sample_weight=predictions["sample_weight"],
            ),
            "recall": ClassificationScores.recall(
                inputs["y_true"], inputs["y_pred"], sample_weight=predictions["sample_weight"],
            ),
            "f1_score": ClassificationScores.f1_score(
                inputs["y_true"], inputs["y_pred"], sample_weight=predictions["sample_weight"],
            ),
            "negative_log_loss": ClassificationScores.negative_log_loss(
                inputs["y_true"], inputs["probabilities"], labels=inputs["labels"],
                sample_weight=predictions["sample_weight"],
            ),
        }
        rows.extend({
            "section": "classification_scores",
            "metric": metric,
            "entity": entity,
            "value": float(value),
            "timestamp": pd.NaT,
            "unit": "score",
        } for metric, value in values.items())
    extra = pd.DataFrame(rows)
    extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
    return extra


def load_backtest_inputs(
    paths, settings: PortfolioSettings, *, strategy: str, partition: str,
    manifest_path: Path, expected_securities: int, data_start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Load common inputs once with shared portfolio settings and a modeling bundle."""
    if strategy not in STRATEGIES or paths.model_kind != strategy.split('_')[0]:
        raise ValueError("Strategy and model artifact family must agree")
    if partition not in ("development", "holdout"):
        raise ValueError("Partition must be development or holdout")
    bundle_path = paths.artifacts / "meta_models.joblib"
    if not bundle_path.exists():
        raise ValueError("Missing meta_models.joblib; rerun the meta modeling notebook")
    primary = joblib.load(paths.artifacts / "primary_model.joblib")
    bundle = joblib.load(bundle_path)
    if set(bundle["models"]) != set(META_FAMILIES):
        raise ValueError("Three tuned meta families are required; rerun meta modeling")
    if bundle["primary_model_hash"] != model_artifact_hash(paths.artifacts / "primary_model.joblib"):
        raise ValueError("Primary model changed; rerun meta modeling")
    events = index_events(pd.read_parquet(paths.event("model")))
    manifest = index_events(pd.read_parquet(paths.artifacts / "split_manifest.parquet"))
    if set(events.index) != set(manifest.index) or not events.partition.equals(manifest.partition.reindex(events.index)):
        raise ValueError("Events must match the fixed split manifest")
    boundaries = events.holdout_boundary.drop_duplicates()
    if len(boundaries) != 1:
        raise ValueError("One common holdout boundary is required")
    boundary = pd.Timestamp(boundaries.iloc[0])
    starts = events.index.get_level_values("event_start")
    if (starts[events.partition.eq("development")] >= boundary).any() or (starts[events.partition.eq("holdout")] < boundary).any():
        raise ValueError("Events cross their declared partition boundary")
    calendar = sessions(paths, start=data_start, end=end)
    evaluation_end = min(end, calendar.close.max())
    if partition == "development":
        evaluation_end = calendar.loc[calendar.close.lt(boundary), "close"].max()
        start = starts[events.partition.eq("development")].min()
    else:
        start = boundary
    if pd.isna(evaluation_end) or start >= evaluation_end:
        raise ValueError("Evaluation requires ordered nonempty bounds")
    return dict(
        paths=paths, settings=settings, strategy=strategy, partition=partition,
        events=events, primary=primary, models=bundle["models"], bundle=bundle,
        calendar=calendar, start=start, end=evaluation_end, data_start=data_start,
        manifest_path=manifest_path,
        symbols=load_manifest(manifest_path, expected_securities=expected_securities).symbol.tolist(),
    )


def development_predictions(
    inputs: dict, *, num_groups: int = 6, num_test_groups: int = 2,
    pct_embargo: float = 0.01, inner_cv_splits: int = 5,
    inner_pct_embargo: float = 0.01,
) -> tuple[dict, pd.DataFrame]:
    """Generate OOS paths for all meta families without saving intermediate files."""
    if inputs["partition"] != "development":
        raise ValueError("CPCV requires development inputs")
    events = inputs["events"]
    development = build_primary_model_frame(
        events, events.loc[events.partition.eq("development")].index,
        model_kind=inputs["paths"].model_kind,
    )
    splits = combinatorial_purged_cross_validation(
        development.event_end, num_groups, num_test_groups, pct_embargo,
    )
    predictions = generate_cpcv_predictions(
        development, splits, inputs["primary"], inputs["models"],
        inner_cv_splits=inner_cv_splits, inner_pct_embargo=inner_pct_embargo,
    )
    paths = {}
    for family in META_FAMILIES:
        assembled = assemble_cpcv_path_predictions(
            predictions.loc[predictions.model_family.eq(family)], splits,
            development.index, num_groups,
        )
        paths.update({(family, name): frame for name, frame in assembled.items()})
    return paths, splits


def holdout_predictions(inputs: dict) -> dict:
    """Validate saved family predictions against canonical events and primary OOS output."""
    if inputs["partition"] != "holdout":
        raise ValueError("Holdout predictions require holdout inputs")
    prediction_path = inputs["paths"].artifacts / "meta_predictions.parquet"
    if not prediction_path.exists():
        raise ValueError("Missing meta_predictions.parquet; rerun the meta modeling notebook")
    saved = pd.read_parquet(prediction_path)
    if "model_family" not in saved or set(saved.model_family) != set(META_FAMILIES):
        raise ValueError("Three-family predictions are required; rerun meta modeling")
    events = inputs["events"]
    primary = index_events(pd.read_parquet(inputs["paths"].artifacts / "primary_predictions.parquet"))
    if set(primary.index) != set(events.index) or not primary.partition.reindex(events.index).equals(events.partition):
        raise ValueError("Primary predictions must match the fixed event partitions")
    outputs = {}
    common = ["primary_side", "primary_probability", "primary_confidence"]
    for family in META_FAMILIES:
        predictions = index_events(saved.loc[saved.model_family.eq(family)])
        if set(predictions.index) != set(events.index):
            raise ValueError("Meta predictions contain missing or extra event keys")
        predictions = predictions.reindex(events.index)
        if not predictions.partition.equals(events.partition):
            raise ValueError("Meta predictions mix development and holdout partitions")
        mask = events.partition.eq("holdout")
        selected = predictions.loc[mask]
        if not selected.prediction_source.eq("holdout").all():
            raise ValueError("Holdout predictions require holdout provenance")
        if not np.array_equal(selected[common].to_numpy(), primary.loc[selected.index, common].to_numpy()):
            raise ValueError("Meta families must use the common primary predictions")
        for column in ("primary_probability", "meta_probability"):
            if not selected[column].between(0, 1).all():
                raise ValueError("Prediction probabilities must be finite and within [0, 1]")
        if not selected.primary_side.isin([-1, 1]).all() or not selected.meta_action.isin([0, 1]).all():
            raise ValueError("Invalid predicted actions")
        frame = events.loc[mask].copy()
        frame[common + ["meta_probability", "meta_action", "prediction_source"]] = selected[common + ["meta_probability", "meta_action", "prediction_source"]]
        frame["meta_label"] = (frame.primary_side * frame.raw_return > 0).astype("int8")
        frame["model_family"] = family
        outputs[(family, "holdout")] = frame
    return outputs


def prepare_bet_sizing(inputs: dict, predictions: dict, splits: pd.DataFrame | None = None) -> dict:
    """Prepare selections or event signals; calibrate from training observations only."""
    asynchronous = inputs["strategy"].endswith("asynchronous")
    calibrations = None
    if asynchronous:
        development = inputs["events"].loc[inputs["events"].partition.eq("development")]
        development = load_event_entry_prices(inputs["paths"], development)
        calibrations = (get_cpcv_price_calibrations(development, splits)
                        if inputs["partition"] == "development"
                        else calibrate_price_sizing(development.reset_index()))
    prepared = {}
    schedule = entry_schedule(inputs["calendar"], inputs["end"])
    for key, prediction in predictions.items():
        events = prediction.reset_index().copy()
        events["horizon_bars"] = inputs["settings"].horizon_bars
        calibration, selected = None, None
        if asynchronous:
            events["event_signal"] = event_bet_signals(events)
            if inputs["partition"] == "development":
                calibration = events[["symbol", "event_start", "split_num"]].merge(
                    calibrations, on=["split_num", "symbol"], how="left", validate="many_to_one",
                )
            else:
                calibration = events[["symbol", "event_start"]].merge(
                    calibrations, on="symbol", how="left", validate="many_to_one",
                )
        else:
            eligible = events.loc[events.event_start.between(inputs["start"], inputs["end"])]
            selected = select_synchronous(eligible, schedule, inputs["settings"].k)
            selected = selected.loc[selected.exit.notna() & selected.exit.le(inputs["end"])]
        prepared[key] = dict(events=events, calibration=calibration, selected=selected)
    return prepared


def simulate_portfolios(inputs: dict, prepared: dict) -> dict:
    """Execute each prepared family/path once, retaining detailed results in memory."""
    asynchronous = inputs["strategy"].endswith("asynchronous")
    prices = None
    if not asynchronous:
        calendar = inputs["calendar"]
        relevant = calendar.loc[calendar.close.ge(inputs["start"]) & calendar.open.le(inputs["end"])]
        symbols = sorted(set().union(*(set(item["events"].symbol) for item in prepared.values())))
        prices = load_session_prices(inputs["paths"], relevant, symbols)
    accounts = {}
    for key, item in prepared.items():
        if asynchronous:
            result = simulate_strategy(
                inputs["paths"], item["events"], inputs["calendar"], inputs["start"],
                inputs["end"], inputs["strategy"], inputs["settings"], item["calibration"],
                evaluation_partition=inputs["partition"], symbols=inputs["symbols"],
            )
        else:
            result = simulate_synchronous(
                item["events"], prices, inputs["calendar"], inputs["start"], inputs["end"],
                inputs["settings"], inputs["partition"], selected=item["selected"],
            )
        if result["ledger"].empty:
            raise ValueError(f"{key} produced no account ledger")
        accounts[key] = result
    return accounts


def summarize_backtests(
    inputs: dict, predictions: dict, accounts: dict, *, annual_risk_free_rate: float,
    periods_per_year: float, annualized_benchmark_sharpe_ratio: float,
) -> pd.DataFrame:
    """Aggregate existing accounts and OOS classification scores without simulation."""
    tables = []
    for (family, path), result in accounts.items():
        statistics = summarize_account(
            result, inputs["settings"],
            benchmark_returns(inputs["paths"], result["ledger"].timestamp.min(), inputs["end"], data_start=inputs["data_start"]),
            annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year,
            annualized_benchmark_sharpe_ratio=annualized_benchmark_sharpe_ratio,
        )
        tables.append(pd.concat([
            statistics, classification_statistics(predictions[(family, path)]), holding_statistics(result),
        ], ignore_index=True).assign(model_family=family, path=path))
    output = pd.concat(tables, ignore_index=True)
    output["timestamp"] = pd.to_datetime(output.timestamp, utc=True)
    return output


def save_backtest_results(
    inputs: dict, statistics: pd.DataFrame, accounts: dict, *,
    evaluation_settings: dict, validation_settings: dict | None = None,
) -> Path:
    """Save only combined statistics, account curves and reproducible settings."""
    directory = inputs["paths"].root / "data/backtest_results" / inputs["strategy"] / inputs["partition"]
    curves = []
    for (family, path), result in accounts.items():
        curves.append(result["ledger"].assign(model_family=family, path=path, entity="final_account"))
        if "group_ledger" in result:
            curves.append(result["group_ledger"].rename(columns={"group": "entity"}).assign(model_family=family, path=path))
    save_frame(statistics, directory / "statistics.parquet")
    save_frame(pd.concat(curves, ignore_index=True), directory / "account.parquet")
    metadata = {
        "strategy": inputs["strategy"], "partition": inputs["partition"],
        "period": inputs["paths"].period, "start": inputs["start"].isoformat(), "end": inputs["end"].isoformat(),
        "data_start": inputs["data_start"].isoformat(),
        "portfolio": asdict(inputs["settings"]), "evaluation": evaluation_settings,
        "validation": validation_settings, "benchmark": "SPY", "cash_interest": 0, "borrow_fee": 0,
        "universe": manifest_hash(inputs["manifest_path"]),
        "primary_model_hash": inputs["bundle"]["primary_model_hash"],
        "primary_configuration": inputs["primary"]["best_configuration"],
        "primary_random_state": inputs["primary"]["random_state"],
        "primary_features": list(inputs["primary"]["feature_columns"]),
        "primary_candidate": inputs["primary"]["selected_candidate"],
        "selected_meta_candidate": inputs["bundle"]["selected_candidate"],
        "meta_configurations": {name: artifact["best_configuration"] for name, artifact in inputs["models"].items()},
        "meta_models_hash": model_artifact_hash(inputs["paths"].artifacts / "meta_models.joblib"),
        "meta_random_states": {name: artifact["random_state"] for name, artifact in inputs["models"].items()},
        "meta_features": {name: list(artifact["feature_columns"]) for name, artifact in inputs["models"].items()},
    }
    (directory / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def load_backtest_results(root: Path, strategy: str, partition: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read the new result contract without refitting or resimulating."""
    if strategy not in STRATEGIES or partition not in ("development", "holdout"):
        raise ValueError("Explicit strategy and evaluation partition are required")
    directory = root / "data/backtest_results" / strategy / partition
    metadata = json.loads((directory / "settings.json").read_text())
    if metadata["strategy"] != strategy or metadata["partition"] != partition:
        raise ValueError("Saved results do not match the requested evaluation")
    return pd.read_parquet(directory / "statistics.parquet"), pd.read_parquet(directory / "account.parquet"), metadata
