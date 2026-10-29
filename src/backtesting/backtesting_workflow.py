"""Stage-by-stage development and holdout evaluation of a manually selected meta model."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from src.backtesting.backtest_statistics import (
    ClassificationScores, Efficiency, GeneralCharacteristics,
    ImplementationShortfall, Performance, Runs, _drawdown_time_under_water,
)
from src.backtesting.bet_sizing import event_bet_signals
from src.backtesting.portfolio_management import (
    PortfolioSettings, STRATEGIES, benchmark_returns, calibrate_price_sizing,
    entry_schedule, load_event_entry_prices,
    select_synchronous, _simulate_synchronous_accounts, _simulate_asynchronous_accounts,
)
from src.backtesting.strategy_validation import (
    assemble_cpcv_path_predictions, combinatorial_purged_cross_validation,
    generate_cpcv_predictions, get_cpcv_price_calibrations,
)
from src.modeling.modeling_workflow import (
    build_primary_model_frame, model_artifact_hash, model_evaluation_columns,
    validate_prediction_partitions,
)
from src.modeling.purged_validation import index_events
from src.preprocessing.market_data import (
    load_manifest, manifest_hash, require_feature_metadata, save_frame, sessions,
)

STATISTICS_SECTIONS = (
    "general_characteristics", "performance", "runs",
    "implementation_shortfall", "efficiency", "classification_scores",
)


def format_backtest_statistics(statistics: pd.DataFrame) -> pd.DataFrame:
    """Format backtest statistics for display and storage.

    Args:
        statistics: Backtest statistics.

    Returns:
        Statistics table with a column for each backtest path.
    """
    selected = statistics.loc[statistics.section.isin(STATISTICS_SECTIONS)].copy()
    selected["metric"] = selected.metric.where(
        selected.section.ne("classification_scores"), selected.entity + "_" + selected.metric,
    )
    selected = pd.concat([
        selected.loc[selected.model_family.eq(family)].sort_values(
            "section", key=lambda values: values.map(STATISTICS_SECTIONS.index), kind="stable",
        ) for family in selected.model_family.drop_duplicates()
    ], ignore_index=True)
    keys = ["model_family", "section", "metric", "unit"]
    rows = pd.MultiIndex.from_frame(selected[keys].drop_duplicates())
    paths = list(selected.path.drop_duplicates())
    dates = selected.timestamp.map(lambda value: value.isoformat() if pd.notna(value) else pd.NA)
    selected["display_value"] = selected.value.astype("string").where(selected.unit.ne("timestamp"), dates)
    output = selected.pivot(index=keys, columns="path", values="display_value").reindex(index=rows, columns=paths)
    return output.rename_axis(columns=None).reset_index()


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
                      benchmark_series: pd.Series, *, annual_risk_free_rate: float,
                      periods_per_year: float, annualized_benchmark_sharpe_ratio: float) -> pd.DataFrame:
    """Compute the five account report sections in normalized form.

    Args:
        result: Portfolio simulation results.
        settings: Portfolio configuration.
        benchmark_series: Benchmark returns.
        annual_risk_free_rate: Annual risk-free return.
        periods_per_year: Number of return periods per year.
        annualized_benchmark_sharpe_ratio: Annualized benchmark Sharpe ratio.

    Returns:
        Account statistics grouped by report section.
    """
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
    benchmark = benchmark_series.copy()
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

    drawdown, time_under_water = _drawdown_time_under_water(equity)
    runs = {
        "hhi_positive_returns": Runs.hhi_positive_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_negative_returns": Runs.hhi_negative_returns(trade_returns)
        if trade_returns is not None else np.nan,
        "hhi_time_between_bets": Runs.hhi_time_between_bets(trade_returns)
        if trade_returns is not None else np.nan,
        "percentile_drawdown": drawdown.quantile(0.95),
        "percentile_time_under_water": time_under_water.quantile(0.95),
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

    statistics = pd.DataFrame(rows)
    statistics["timestamp"] = pd.to_datetime(statistics["timestamp"], utc=True)
    if not np.isclose(closed["net_pnl"].sum() if not closed.empty else 0.0, net_pnl):
        raise ValueError("Closed-position PnL does not reconcile with final account AUM")
    return statistics


def classification_statistics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Summarize classification metrics for each out-of-sample path.

    Args:
        predictions: Model predictions.

    Returns:
        Classification statistics by model family and path.
    """
    rows = []
    for entity, class_labels, probability_column in (
        ("primary", [-1, 1], "primary_positive_probability"),
        ("meta", [0, 1], "meta_pass_probability"),
    ):
        scores = ClassificationScores.score_binary_predictions(
            predictions[f"{entity}_label"], predictions[f"{entity}_prediction"],
            predictions[probability_column], predictions["sample_weight"],
            class_labels=class_labels,
        )
        for metric in model_evaluation_columns(class_labels):
            display_metric = (
                f"{metric}_1"
                if entity == "primary" and metric in {"precision", "recall", "f1"}
                else metric
            )
            rows.append({
                "section": "classification_scores", "metric": display_metric,
                "entity": entity, "value": scores[metric],
                "timestamp": pd.NaT, "unit": "score",
            })
    extra = pd.DataFrame(rows)
    extra["timestamp"] = pd.to_datetime(extra["timestamp"], utc=True)
    return extra


def load_backtest_inputs(
    paths, settings: PortfolioSettings, *, strategy: str, partition: str,
    manifest_path: Path, data_start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict:
    """Load portfolio evaluation inputs.

    Args:
        paths: Research data and artifact paths.
        settings: Portfolio configuration.
        strategy: Strategy name.
        partition: Evaluation partition name.
        manifest_path: Universe manifest path.
        data_start: Market-data start timestamp.
        end: Interval end timestamp.

    Returns:
        Evaluation inputs, model artifacts, and portfolio settings.
    """
    if strategy not in STRATEGIES or paths.model_kind != strategy.split('_')[0]:
        raise ValueError("Strategy and model artifact family must agree")
    if partition not in ("development", "holdout"):
        raise ValueError("Partition must be development or holdout")
    bundle_path = paths.artifacts / "meta_models.joblib"
    if not bundle_path.exists():
        raise ValueError("Missing meta_models.joblib; rerun the meta modeling notebook")
    primary = joblib.load(paths.artifacts / "primary_model.joblib")
    bundle = joblib.load(bundle_path)
    symbols = load_manifest(manifest_path).symbol.tolist()
    for symbol in dict.fromkeys([*symbols, "SPY"]):
        require_feature_metadata(paths.feature(symbol, "dollar_bars"))
    selected_family = bundle["selected_candidate"]
    if (not primary.get("configuration_id") or not bundle.get("configuration_id")
            or selected_family not in ("boosting", "bagging", "random_forest")
            or set(bundle["models"]) != {selected_family}
            or bundle["models"][selected_family].get("configuration_id") != bundle["configuration_id"]):
        raise ValueError("Explicit Primary and Meta configuration IDs are required; rerun modeling")
    if bundle["primary_model_hash"] != model_artifact_hash(paths.artifacts / "primary_model.joblib"):
        raise ValueError("Primary model changed; rerun meta modeling")
    events = index_events(pd.read_parquet(paths.event("prepared")))
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
        symbols=symbols,
    )


def development_predictions(
    inputs: dict, *, outer_num_total_groups: int = 6, outer_num_test_groups: int = 2,
    outer_pct_embargo: float = 0.01, inner_num_splits: int = 5,
    inner_pct_embargo: float = 0.01,
) -> tuple[dict, pd.DataFrame]:
    """Generate development predictions using combinatorial validation.

    Args:
        inputs: Loaded development inputs and frozen model artifacts.
        outer_num_total_groups: Number of contiguous outer CPCV groups.
        outer_num_test_groups: Number of outer groups held out per combination.
        outer_pct_embargo: Embargo fraction for the outer CPCV splits.
        inner_num_splits: Purged fold count for primary OOF meta-training inputs.
        inner_pct_embargo: Embargo fraction for the inner primary OOF folds.

    Returns:
        Predictions by model family and path, and cross-validation split metadata.
    """
    if inputs["partition"] != "development":
        raise ValueError("CPCV requires development inputs")
    events = inputs["events"]
    development = build_primary_model_frame(
        events, events.loc[events.partition.eq("development")].index,
        model_kind=inputs["paths"].model_kind,
    )
    splits = combinatorial_purged_cross_validation(
        development.event_end, outer_num_total_groups, outer_num_test_groups, outer_pct_embargo,
    )
    family = inputs["bundle"]["selected_candidate"]
    predictions = generate_cpcv_predictions(
        development, splits, inputs["primary"], inputs["models"][family],
        model_family=family, inner_num_splits=inner_num_splits, inner_pct_embargo=inner_pct_embargo,
    )
    assembled = assemble_cpcv_path_predictions(
        predictions, splits, development.index, outer_num_total_groups,
    )
    return {(family, name): frame for name, frame in assembled.items()}, splits


def holdout_predictions(inputs: dict) -> dict:
    """Load and validate holdout predictions.

    Args:
        inputs: Loaded evaluation inputs.

    Returns:
        Holdout predictions keyed by model family and path.
    """
    if inputs["partition"] != "holdout":
        raise ValueError("Holdout predictions require holdout inputs")
    prediction_path = inputs["paths"].artifacts / "meta_predictions.parquet"
    if not prediction_path.exists():
        raise ValueError("Missing meta_predictions.parquet; rerun the meta modeling notebook")
    saved = pd.read_parquet(prediction_path)
    if "model_family" not in saved or set(saved.model_family) != set(inputs["models"]):
        raise ValueError("Predictions must match the selected Meta model; rerun meta modeling")
    events = inputs["events"]
    primary = index_events(pd.read_parquet(inputs["paths"].artifacts / "primary_predictions.parquet"))
    if set(primary.index) != set(events.index) or not primary.partition.reindex(events.index).equals(events.partition):
        raise ValueError("Primary predictions must match the fixed event partitions")
    validate_prediction_partitions(primary, ("primary_cv_fold",))
    family = inputs["bundle"]["selected_candidate"]
    common = ["primary_prediction", "primary_positive_probability", "primary_class_probability"]
    predictions = index_events(saved.loc[saved.model_family.eq(family)])
    if set(predictions.index) != set(events.index):
        raise ValueError("Meta predictions contain missing or extra event keys")
    predictions = predictions.reindex(events.index)
    if not predictions.partition.equals(events.partition):
        raise ValueError("Meta predictions mix development and holdout partitions")
    mask = events.partition.eq("holdout")
    selected = predictions.loc[mask]
    validate_prediction_partitions(predictions, ("primary_cv_fold", "meta_cv_fold"))
    if not predictions[list(primary.columns)].equals(primary.reindex(events.index)):
        raise ValueError("Meta predictions must use the saved primary predictions")
    for column in ("primary_positive_probability", "meta_pass_probability"):
        if not selected[column].between(0, 1).all():
            raise ValueError("Prediction probabilities must be finite and within [0, 1]")
    if not selected.primary_prediction.isin([-1, 1]).all() or not selected.meta_prediction.isin([0, 1]).all():
        raise ValueError("Invalid predicted actions")
    frame = events.loc[mask].copy()
    frame[common + ["meta_pass_probability", "meta_prediction"]] = selected[common + ["meta_pass_probability", "meta_prediction"]]
    frame["meta_label"] = (frame.primary_prediction * frame.raw_return > 0).astype("int8")
    frame["model_family"] = family
    return {(family, "holdout"): frame}


def prepare_bet_sizing(inputs: dict, predictions: dict, splits: pd.DataFrame | None = None) -> dict:
    """Prepare portfolio selections and event signals.

    Args:
        inputs: Loaded evaluation inputs.
        predictions: Model predictions.
        splits: Cross-validation split metadata.

    Returns:
        Portfolio selections and calibrations keyed by model family and path.
    """
    asynchronous = inputs["strategy"].endswith("asynchronous")
    calibrations = None
    if asynchronous:
        development = inputs["events"].loc[inputs["events"].partition.eq("development")]
        development = load_event_entry_prices(inputs["paths"], development)
        calibrations = (get_cpcv_price_calibrations(development, splits)
                        if inputs["partition"] == "development"
                        else calibrate_price_sizing(development.reset_index()))
    prepared = {}
    if not asynchronous:
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
            selected = select_synchronous(eligible, schedule, inputs["settings"].max_positions_per_side)
            selected = selected.loc[selected.exit.notna() & selected.exit.le(inputs["end"])]
        prepared[key] = dict(events=events, calibration=calibration, selected=selected)
    return prepared


def simulate_portfolios(inputs: dict, prepared: dict, *, show_progress: bool = False) -> dict:
    """Simulate portfolios for each model family and path.

    Args:
        inputs: Common evaluation inputs and portfolio settings.
        prepared: Event signals and calibration keyed by model family and path.
        show_progress: Whether to display progress.

    Returns:
        Simulation results keyed by model family and path.
    """
    if inputs["strategy"] not in STRATEGIES or inputs["paths"].model_kind != inputs["strategy"].split("_")[0]:
        raise ValueError("Strategy and model artifact family must agree")
    simulate = (_simulate_asynchronous_accounts if inputs["strategy"].endswith("asynchronous")
                else _simulate_synchronous_accounts)
    return simulate(inputs["paths"], prepared, inputs["calendar"], inputs["start"], inputs["end"],
                    inputs["settings"], inputs["partition"], symbols=inputs["symbols"],
                    show_progress=show_progress)


def summarize_backtests(
    inputs: dict, predictions: dict, accounts: dict, *, benchmark_symbol: str, annual_risk_free_rate: float,
    periods_per_year: float, annualized_benchmark_sharpe_ratio: float,
) -> pd.DataFrame:
    """Summarize paths on common evaluation bounds with one shared benchmark.

    Args:
        inputs: Loaded evaluation inputs.
        predictions: Model predictions.
        accounts: Portfolio accounts keyed by identifier.
        benchmark_symbol: Benchmark ticker symbol.
        annual_risk_free_rate: Annual risk-free return.
        periods_per_year: Number of return periods per year.
        annualized_benchmark_sharpe_ratio: Annualized benchmark Sharpe ratio.

    Returns:
        Combined classification and account statistics.
    """
    benchmark = benchmark_returns(
        inputs["paths"], inputs["start"], inputs["end"],
        data_start=inputs["data_start"], benchmark_symbol=benchmark_symbol,
    )
    tables = []
    for (family, path), result in accounts.items():
        statistics = summarize_account(
            result, inputs["settings"], benchmark,
            annual_risk_free_rate=annual_risk_free_rate, periods_per_year=periods_per_year,
            annualized_benchmark_sharpe_ratio=annualized_benchmark_sharpe_ratio,
        )
        tables.append(pd.concat([
            statistics, classification_statistics(predictions[(family, path)]),
        ], ignore_index=True).assign(model_family=family, path=path))
    output = pd.concat(tables, ignore_index=True)
    output["timestamp"] = pd.to_datetime(output.timestamp, utc=True)
    return format_backtest_statistics(output)


def save_backtest_results(
    inputs: dict, statistics: pd.DataFrame, *,
    evaluation_settings: dict, validation_settings: dict | None = None,
) -> Path:
    """Save backtest statistics and settings.

    Args:
        inputs: Loaded evaluation inputs.
        statistics: Backtest statistics.
        evaluation_settings: Evaluation configuration.
        validation_settings: Validation configuration.

    Returns:
        Directory containing the saved statistics and settings.
    """
    directory = inputs["paths"].root / "data/backtesting" / inputs["strategy"] / inputs["partition"]
    save_frame(statistics, directory / "statistics.parquet")
    metadata = {
        "strategy": inputs["strategy"], "partition": inputs["partition"],
        "period": inputs["paths"].period, "start": inputs["start"].isoformat(), "end": inputs["end"].isoformat(),
        "data_start": inputs["data_start"].isoformat(),
        "portfolio": asdict(inputs["settings"]), "evaluation": evaluation_settings,
        "execution": {
            "observation": "dollar_bar", "fill": "next_bar_close",
            "recording": ("minute_last_and_execution_boundaries" if inputs["strategy"].endswith("asynchronous")
                          else "session_and_execution_boundaries"),
            "scheduler": "bar_stream" if inputs["strategy"].endswith("asynchronous") else "synchronous_event_scheduler",
            "decision": "initial_or_event_exit" if inputs["strategy"].endswith("asynchronous") else "session_boundary",
            **({"pending_entry_policy": "update_without_reselection_until_exit",
                "allocation": "directional_equal_weights_capped_at_inverse_capacity",
                "signal_update": "latest_event_supersedes_previous",
                "ranking": "meta_probability_time_decay",
                "candidate_queue": "persistent_bounded_per_side",
                "candidate_queue_capacity_per_side": inputs["settings"].max_positions_per_side,
                "entry_limit_policy": "discard_on_violation",
                "limit_refresh": "new_event"}
               if inputs["strategy"].endswith("asynchronous") else {}),
            "barrier": "close" if inputs["strategy"].endswith("asynchronous") else None,
            "entry_limit": inputs["strategy"].endswith("asynchronous"),
            "terminal": "last_bar_close", "benchmark": "daily_last_bar_close",
            "synchronous_rollover": "flat_book_before_entry",
        },
        "validation": validation_settings, "benchmark": evaluation_settings["benchmark_symbol"], "cash_interest": 0, "borrow_fee": 0,
        "universe": manifest_hash(inputs["manifest_path"]),
        "primary_model_hash": inputs["bundle"]["primary_model_hash"],
        "primary_configuration": inputs["primary"]["best_configuration"],
        "primary_random_state": inputs["primary"]["random_state"],
        "primary_features": list(inputs["primary"]["feature_columns"]),
        "primary_candidate": inputs["primary"]["selected_candidate"],
        "selected_meta_candidate": inputs["bundle"]["selected_candidate"],
        "primary_configuration_id": inputs["primary"]["configuration_id"],
        "meta_configuration_id": inputs["bundle"]["configuration_id"],
        "meta_configurations": {name: artifact["best_configuration"] for name, artifact in inputs["models"].items()},
        "meta_models_hash": model_artifact_hash(inputs["paths"].artifacts / "meta_models.joblib"),
        "meta_random_states": {name: artifact["random_state"] for name, artifact in inputs["models"].items()},
        "meta_features": {name: list(artifact["feature_columns"]) for name, artifact in inputs["models"].items()},
    }
    (directory / "settings.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return directory


def load_backtest_results(root: Path, strategy: str, partition: str) -> tuple[pd.DataFrame, dict]:
    """Load saved backtest statistics and settings.

    Args:
        root: Project root directory.
        strategy: Strategy name.
        partition: Evaluation partition name.

    Returns:
        Saved statistics and settings.
    """
    if strategy not in STRATEGIES or partition not in ("development", "holdout"):
        raise ValueError("Explicit strategy and evaluation partition are required")
    directory = root / "data/backtesting" / strategy / partition
    metadata = json.loads((directory / "settings.json").read_text())
    if metadata["strategy"] != strategy or metadata["partition"] != partition:
        raise ValueError("Saved results do not match the requested evaluation")
    return pd.read_parquet(directory / "statistics.parquet"), metadata
