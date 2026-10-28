import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.backtesting import backtesting_workflow as workflow
from src.backtesting.portfolio_management import PortfolioSettings, simulate_strategy
from src.modeling.modeling_workflow import (
    build_primary_model_frame, build_meta_model_frame, generate_oof_predictions,
    get_primary_feature_columns, get_meta_feature_columns, finalize_meta_models,
)
from src.modeling.purged_validation import PurgedKFold
from src.preprocessing.market_technical_indicators import MODEL_FEATURES, TECHNICAL_FEATURES
from src.preprocessing.market_data import ResearchPaths, VERSION


@pytest.fixture
def strategy_workspace(tmp_path, monkeypatch):
    symbols = [f"S{i:02}" for i in range(50)]
    opens = pd.date_range("2025-02-03 14:30Z", periods=35, freq="B")
    calendar = pd.DataFrame({"open": opens, "close": opens + pd.Timedelta(minutes=10)})
    common = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    common.universe.mkdir(parents=True)
    pd.DataFrame({"symbol": symbols}).to_csv(common.universe / "sp500_2025.csv", index=False)
    calendar.to_parquet(common.universe / "sessions.parquet")
    (common.universe / "sessions.json").write_text(json.dumps({
        "start": "2025-01-01T00:00Z", "end": "2026-01-01T00:00Z",
    }))
    monkeypatch.setattr(workflow, "load_manifest", lambda paths, **kwargs: pd.DataFrame({"symbol": symbols}))
    rows = []
    for start, partition in [(time, "development") for time in opens[:-3]] + [
                             (opens[-3], "holdout"),
                             (opens[-3] + pd.Timedelta(minutes=2), "holdout")]:
        for i, symbol in enumerate(symbols):
            side = 1 if i < 25 else -1
            rows.append({"symbol": symbol, "event_start": start,
                         "event_end": start + pd.Timedelta(minutes=5),
                         "vertical_barrier": start + pd.Timedelta(minutes=6), "horizon_volatility": .5,
                         "raw_return": .01 * side * (1 if i % 3 else -1), "primary_label": side,
                         "sample_weight": 1., "mean_sentiment_score": 50 - i,
                         "partition": partition, "holdout_boundary": opens[-3] - pd.Timedelta(minutes=3),
                         "primary_prediction": side, "primary_positive_probability": .9 if side == 1 else .1,
                         "meta_pass_probability": 1., "meta_prediction": 1,
                         "meta_label": int((i % 2 == 1) == (side == 1))})
    combined = pd.DataFrame(rows)
    model_events = combined.drop(columns=["primary_prediction", "primary_positive_probability", "meta_pass_probability", "meta_prediction", "meta_label"])
    for feature in MODEL_FEATURES:
        if feature not in model_events:
            model_events[feature] = 0.0
    model_events["fractionally_differenced_log_close"] = combined.primary_prediction
    model_events[TECHNICAL_FEATURES[0]] = [
        float(int(symbol[1:]) % 3 != 0) if partition == "development" else 1.0
        for symbol, partition in zip(model_events.symbol, model_events.partition)
    ]
    common.event("prepared").parent.mkdir(parents=True)
    model_events.to_parquet(common.event("prepared"), index=False)
    for kind in ("market", "sentiment"):
        paths = ResearchPaths(tmp_path, kind, start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
        paths.artifacts.mkdir(parents=True)
        frame = build_primary_model_frame(model_events, model_events[["symbol", "event_start"]], kind)
        train = frame.loc[frame.partition.eq("development")]
        primary_features = get_primary_feature_columns(frame, kind)
        primary = DecisionTreeClassifier(max_depth=3, random_state=42)
        oof = generate_oof_predictions(
            primary, train[primary_features], train.primary_label, train.sample_weight,
            PurgedKFold(2, t1=train.event_end), positive_label=1,
        )
        meta_train = build_meta_model_frame(train, oof[["prediction", "probability", "prediction_source"]])
        primary.fit(train[primary_features], train.primary_label)
        predictions = frame.copy()
        predictions["primary_prediction"] = primary.predict(frame[primary_features])
        predictions["primary_positive_probability"] = primary.predict_proba(frame[primary_features])[:, list(primary.classes_).index(1)]
        predictions["primary_class_probability"] = np.maximum(predictions.primary_positive_probability, 1 - predictions.primary_positive_probability)
        meta_features = get_meta_feature_columns(meta_train, kind)
        meta = DecisionTreeClassifier(max_depth=3, random_state=42).fit(meta_train[meta_features], meta_train.meta_label)
        predictions["meta_prediction"] = meta.predict(predictions[meta_features])
        predictions["meta_pass_probability"] = meta.predict_proba(predictions[meta_features])[:, list(meta.classes_).index(1)]
        predictions["meta_label"] = (predictions.primary_prediction * predictions.raw_return > 0).astype(int)
        artifact = dict(estimator=primary, feature_columns=primary_features,
                        market_data_version=VERSION,
                        selected_candidate="tree", best_configuration={"max_depth": 3}, random_state=42,
                        configuration_id="primary-manual")
        joblib.dump(artifact, paths.artifacts / "primary_model.joblib")
        primary_predictions = predictions.drop(columns=["meta_prediction", "meta_pass_probability", "meta_label"]).copy()
        primary_predictions["primary_negative_probability"] = 1 - primary_predictions.primary_positive_probability
        primary_predictions["primary_cv_fold"] = np.where(primary_predictions.partition.eq("development"), 0, np.nan)
        primary_predictions = primary_predictions.drop(columns=list(MODEL_FEATURES), errors="ignore")
        primary_predictions.reset_index().to_parquet(paths.artifacts / "primary_predictions.parquet", index=False)
        meta_oof = generate_oof_predictions(
            DecisionTreeClassifier(max_depth=3, random_state=42), meta_train[meta_features],
            meta_train.meta_label, meta_train.sample_weight, PurgedKFold(5, meta_train.event_end),
        )
        finalize_meta_models(
            meta_train, frame, primary_predictions,
            estimators={name: DecisionTreeClassifier(max_depth=3, random_state=42) for name in ("boosting",)},
            oof_predictions={name: meta_oof for name in ("boosting",)},
            configurations={name: {"max_depth": 3} for name in ("boosting",)},
            feature_columns=meta_features, artifact_dir=paths.artifacts,
            selected_candidate="boosting", random_state=42, configuration_id="meta-manual",
        )
    for symbol in [*symbols, "SPY"]:
        file = common.feature(symbol, "dollar_bars")
        file.parent.mkdir(parents=True, exist_ok=True)
        times = pd.DatetimeIndex([opens[0] - pd.Timedelta(days=1)])
        for session in calendar.itertuples():
            times = times.union(pd.date_range(session.open, session.close, freq="min"))
        pd.DataFrame({"end": times, "close": 100. + np.arange(len(times)) * .001}).to_parquet(file, index=False)
        file.with_suffix(".json").write_text(json.dumps({"version": VERSION}))
    return tmp_path



@pytest.mark.parametrize("strategy", ["sentiment_asynchronous", "market_synchronous", "sentiment_synchronous"])
@pytest.mark.parametrize("partition, benchmark_symbol", [("holdout", "S00"), ("development", "SPY")])
def test_notebook_runs_independently_and_saves_only_final_results(strategy_workspace, monkeypatch, strategy, partition, benchmark_symbol):
    import IPython.display

    root = strategy_workspace
    directory = root / "notebooks/backtesting" / strategy
    directory.mkdir(parents=True)
    monkeypatch.chdir(directory)
    displayed_tables = []
    monkeypatch.setattr(IPython.display, "display", lambda value: displayed_tables.append(value)
                        if isinstance(value, pd.DataFrame)
                        and value.index.names == ["section", "metric", "unit"] else None)
    monkeypatch.setattr(plt, "show", lambda: plt.close("all"))
    calls = []
    original_sync, original_dispatch = workflow._simulate_synchronous_accounts, workflow._simulate_asynchronous_accounts

    def sync(*args, **kwargs):
        calls.append("sync")
        return original_sync(*args, **kwargs)

    def dispatch(*args, **kwargs):
        calls.append("async")
        return original_dispatch(*args, **kwargs)

    monkeypatch.setattr(workflow, "_simulate_synchronous_accounts", sync)
    monkeypatch.setattr(workflow, "_simulate_asynchronous_accounts", dispatch)
    path = Path(__file__).resolve().parents[2] / "notebooks/backtesting" / strategy / f"{partition} backtesting.ipynb"
    notebook = json.loads(path.read_text())
    namespace = {}
    for i, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "evaluation_settings = dict(" in source:
                assert 'benchmark_symbol="SPY"' in source
                source = source.replace('benchmark_symbol="SPY"', f'benchmark_symbol="{benchmark_symbol}"')
            exec(compile(source, f"{path}:cell{i}", "exec"), namespace)
    expected = 5 if partition == "development" else 1
    assert len(namespace["accounts"]) == expected
    assert calls == (["async"] if strategy.endswith("asynchronous") else ["sync"])
    output = root / "data/backtesting" / strategy / partition
    assert {p.name for p in output.iterdir()} == {"statistics.parquet", "settings.json"}
    assert not list(root.rglob("cpcv_predictions.parquet"))
    assert not list(root.rglob("price_calibration.parquet"))
    stats, metadata = workflow.load_backtest_results(root, strategy, partition)
    asynchronous = strategy.endswith("asynchronous")
    assert metadata["execution"]["engine_version"] == (10 if asynchronous else 3)
    assert metadata["execution"]["recording"] == ("minute_last_and_execution_boundaries" if asynchronous
                                                   else "session_and_execution_boundaries")
    assert metadata["execution"]["scheduler"] == ("bar_stream" if asynchronous else "synchronous_event_scheduler_v1")
    assert metadata["execution"]["observation"] == "dollar_bar"
    assert metadata["execution"]["benchmark"] == "daily_last_bar_close"
    assert metadata["execution"]["decision"] == ("initial_or_event_exit" if asynchronous else "session_boundary")
    assert metadata["benchmark"] == metadata["evaluation"]["benchmark_symbol"] == benchmark_symbol
    assert not stats.section.isin(["benchmark", "assumption", "coverage", "exposure", "holdings"]).any()
    asynchronous = strategy == "sentiment_asynchronous"
    assert metadata["portfolio"]["horizon_bars"] == (1560 if asynchronous else 1000)
    assert metadata["portfolio"]["horizontal_multipliers"] == ([5.0, 5.0] if asynchronous else [1.0, 1.0])
    if asynchronous:
        assert metadata["execution"]["ranking"] == "meta_probability_time_decay"
        assert metadata["execution"]["candidate_queue"] == "persistent_bounded_per_side"
        assert metadata["execution"]["candidate_queue_capacity_per_side"] == metadata["portfolio"]["max_positions_per_side"]
        assert metadata["execution"]["entry_limit_policy"] == "discard_on_violation"
        assert metadata["execution"]["limit_refresh"] == "new_event"
        assert metadata["execution"]["allocation"] == "directional_equal_weights_capped_at_inverse_capacity"
        assert metadata["execution"]["signal_update"] == "latest_event_supersedes_previous"
        assert namespace["settings"].horizon_bars == 1560
        assert tuple(namespace["settings"].horizontal_multipliers) == (5.0, 5.0)
        assert all(item["events"].horizon_bars.eq(1560).all() for item in namespace["prepared"].values())
    expected_limit = 10
    assert metadata["portfolio"]["max_positions_per_side"] == expected_limit
    assert "k" not in metadata["portfolio"]
    assert not stats.metric.eq("K").any()
    assert "horizon_bars" not in metadata
    pd.testing.assert_frame_equal(stats, namespace["statistics"])
    assert set(stats.model_family) == {"boosting"}
    assert metadata["primary_configuration_id"] == "primary-manual"
    assert metadata["meta_configuration_id"] == "meta-manual"
    assert set(stats.section) == set(workflow.STATISTICS_SECTIONS)
    paths = [f"path_{i}" for i in range(5)] if partition == "development" else ["holdout"]
    assert list(stats) == ["model_family", "section", "metric", "unit", *paths]
    for path in paths:
        assert isinstance(stats[path].dtype, pd.StringDtype)
        assert stats.loc[stats.unit.eq("timestamp"), path].notna().all()
    assert {
        "primary_accuracy", "primary_precision_-1", "primary_recall_-1", "primary_f1_-1",
        "primary_precision_1", "primary_recall_1", "primary_f1_1", "primary_log_loss",
        "primary_pr_auc", "primary_roc_auc", "meta_accuracy", "meta_precision",
        "meta_recall", "meta_f1", "meta_log_loss", "meta_pr_auc", "meta_roc_auc",
    } <= set(stats.metric)
    assert not stats.metric.str.contains("f1_score|negative_log_loss").any()
    for family, displayed in zip(("boosting",), displayed_tables, strict=True):
        expected_table = stats.loc[stats.model_family.eq(family)].set_index(["section", "metric", "unit"])
        pd.testing.assert_frame_equal(displayed, expected_table[paths])
    if partition == "development":
        assert all(result["ledger"].timestamp.max() < namespace["inputs"]["events"].holdout_boundary.iloc[0]
                   for result in namespace["accounts"].values())
    else:
        assert "path" not in stats
        assert metadata["validation"] is None
    for key, result in namespace["accounts"].items():
        assert result["ledger"].timestamp.is_unique
        assert result["ledger"].iloc[0].aum == 100000
        assert result["ledger"].iloc[-1].gross_exposure == 0
        assert len(result["trades"]) > 0
        assert result["closed_trades"].net_pnl.sum() == pytest.approx(result["ledger"].iloc[-1].aum - 100000)
        if not result["exposures"].empty:
            assert result["exposures"].groupby("timestamp").size().max() <= 20
            if strategy == "sentiment_asynchronous" and partition == "holdout":
                assert result["exposures"].groupby("timestamp").size().max() == 20
                assert result["exposures"].groupby("timestamp").weight.apply(lambda x: (x > 0).sum()).max() == 10
                assert result["exposures"].groupby("timestamp").weight.apply(lambda x: (x < 0).sum()).max() == 10
        if partition == "development":
            expected_keys = namespace["inputs"]["events"].loc[lambda x: x.partition.eq("development")].index
            assert namespace["predictions"][key].index.equals(expected_keys)
    # The staged path consumes prepared sizing and agrees with the existing engine.
    key = next(iter(namespace["accounts"]))
    inputs, item = namespace["inputs"], namespace["prepared"][key]
    direct = simulate_strategy(
        inputs["paths"], item["events"].drop(columns="event_signal", errors="ignore"), inputs["calendar"], inputs["start"], inputs["end"],
        strategy, inputs["settings"], item["calibration"], evaluation_partition=partition,
        symbols=inputs["symbols"],
    )
    pd.testing.assert_frame_equal(direct["ledger"], namespace["accounts"][key]["ledger"])
    pd.testing.assert_frame_equal(direct["trades"], namespace["accounts"][key]["trades"])


def holdout_inputs(root, settings=None):
    paths = ResearchPaths(root, "sentiment", start=pd.Timestamp("2025-02-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"))
    return workflow.load_backtest_inputs(
        paths, settings if settings is not None else PortfolioSettings(max_positions_per_side=10),
        strategy="sentiment_asynchronous", partition="holdout",
        manifest_path=paths.universe / "sp500_2025.csv",
        data_start=pd.Timestamp("2025-01-01", tz="UTC"), end=pd.Timestamp("2026-01-01", tz="UTC"),
    )


@pytest.mark.parametrize("horizon_bars", [None, 2], ids=["default", "custom"])
def test_portfolio_horizon_reaches_prepared_events(strategy_workspace, horizon_bars):
    settings = (PortfolioSettings(max_positions_per_side=10) if horizon_bars is None
                else PortfolioSettings(max_positions_per_side=10, horizon_bars=horizon_bars))
    expected = 1000 if horizon_bars is None else horizon_bars
    assert settings.horizon_bars == expected
    inputs = holdout_inputs(strategy_workspace, settings=settings)
    assert "horizon_bars" not in inputs
    predictions = workflow.holdout_predictions(inputs)
    prepared = workflow.prepare_bet_sizing(inputs, predictions)
    assert prepared
    for item in prepared.values():
        assert not item["events"].empty
        assert item["events"].horizon_bars.eq(expected).all()


@pytest.mark.parametrize("problem", ["missing", "duplicate", "partition", "primary", "probability", "fold"])
def test_holdout_predictions_reject_invalid_inputs(strategy_workspace, problem):
    inputs = holdout_inputs(strategy_workspace)
    path = inputs["paths"].artifacts / "meta_predictions.parquet"
    predictions = pd.read_parquet(path)
    position = predictions.index[predictions.model_family.eq("boosting") & predictions.partition.eq("holdout")][0]
    if problem == "missing":
        predictions = predictions.drop(position)
    elif problem == "duplicate":
        predictions = pd.concat([predictions, predictions.loc[[position]]])
    elif problem == "partition":
        predictions.loc[position, "partition"] = "development"
    elif problem == "primary":
        predictions.loc[position, "primary_prediction"] *= -1
    elif problem == "probability":
        predictions.loc[position, "meta_pass_probability"] = np.nan
    else:
        predictions.loc[position, "meta_cv_fold"] = 0
    predictions.to_parquet(path, index=False)
    with pytest.raises(ValueError):
        workflow.holdout_predictions(inputs)


def test_holdout_does_not_fit_and_calibration_ignores_holdout(strategy_workspace, monkeypatch):
    inputs = holdout_inputs(strategy_workspace)
    monkeypatch.setattr(DecisionTreeClassifier, "fit", lambda *args, **kwargs: pytest.fail("Holdout must not fit"))
    predictions = workflow.holdout_predictions(inputs)
    baseline = workflow.prepare_bet_sizing(inputs, predictions)
    changed = dict(inputs, events=inputs["events"].copy())
    changed["events"].loc[lambda x: x.partition.eq("holdout"), "horizon_volatility"] = 999.0
    updated = workflow.prepare_bet_sizing(changed, predictions)
    for key in baseline:
        pd.testing.assert_frame_equal(baseline[key]["calibration"], updated[key]["calibration"])


@pytest.mark.parametrize("filename", ["meta_models.joblib", "meta_predictions.parquet"])
def test_old_artifacts_require_modeling_rerun(strategy_workspace, filename):
    inputs = holdout_inputs(strategy_workspace)
    assert holdout_inputs(strategy_workspace)["bundle"]["primary_model_hash"] == inputs["bundle"]["primary_model_hash"]
    (inputs["paths"].artifacts / filename).unlink()
    with pytest.raises(ValueError, match="rerun the meta modeling"):
        workflow.holdout_predictions(holdout_inputs(strategy_workspace))


def test_result_partitions_and_families_do_not_overwrite(strategy_workspace):
    inputs = holdout_inputs(strategy_workspace, settings=PortfolioSettings(max_positions_per_side=10, horizon_bars=7))
    statistics = pd.DataFrame({"model_family": list(workflow.META_FAMILIES),
                               "section": "performance", "metric": "pnl", "unit": "USD",
                               "holdout": pd.Series(["1.0", "2.0", "3.0"], dtype="string")})
    holdout = workflow.save_backtest_results(inputs, statistics, evaluation_settings={"benchmark_symbol": "SPY"})
    original = (holdout / "statistics.parquet").read_bytes()
    development = workflow.save_backtest_results(dict(inputs, partition="development"), statistics, evaluation_settings={"benchmark_symbol": "SPY"})
    assert development != holdout
    assert (holdout / "statistics.parquet").read_bytes() == original
    assert {p.name for p in holdout.iterdir()} == {"statistics.parquet", "settings.json"}
    saved_statistics, metadata = workflow.load_backtest_results(
        strategy_workspace, "sentiment_asynchronous", "holdout",
    )
    pd.testing.assert_frame_equal(saved_statistics, statistics)
    # Legacy account files are ignored, even when they are not readable Parquet.
    legacy_account = holdout / "account.parquet"
    legacy_account.write_bytes(b"unused legacy account")
    workflow.save_backtest_results(inputs, statistics, evaluation_settings={"benchmark_symbol": "SPY"})
    assert legacy_account.read_bytes() == b"unused legacy account"
    with_account, with_account_metadata = workflow.load_backtest_results(
        strategy_workspace, "sentiment_asynchronous", "holdout",
    )
    pd.testing.assert_frame_equal(with_account, saved_statistics)
    assert with_account_metadata == metadata
    assert metadata["portfolio"]["horizon_bars"] == 7
    assert "horizon_bars" not in metadata
    # Existing result files retain their original metadata shape when loaded.
    metadata["horizon_bars"] = metadata["portfolio"].pop("horizon_bars")
    (holdout / "settings.json").write_text(json.dumps(metadata))
    legacy_statistics, legacy_metadata = workflow.load_backtest_results(
        strategy_workspace, "sentiment_asynchronous", "holdout",
    )
    pd.testing.assert_frame_equal(legacy_statistics, saved_statistics)
    assert legacy_metadata == metadata


@pytest.mark.parametrize("paths", [["holdout"], [f"path_{i}" for i in range(5)]])
def test_statistics_conversion_preserves_report_values_and_dates(tmp_path, paths):
    rows = []
    for family_num, family in enumerate(workflow.META_FAMILIES):
        for path_num, path in enumerate(paths):
            start = pd.Timestamp("2025-02-03 14:30:00.123456789Z") + pd.Timedelta(
                days=family_num + path_num,
            )
            for metric, timestamp in (("start", start), ("end", start + pd.Timedelta(days=3))):
                rows.append(dict(model_family=family, path=path, section="general_characteristics",
                                 metric=metric, entity="final_account", unit="timestamp",
                                 value=np.nan, timestamp=timestamp))
            for section in workflow.STATISTICS_SECTIONS:
                entities = ("primary", "meta") if section == "classification_scores" else ("final_account",)
                for entity in entities:
                    rows.append(dict(model_family=family, path=path, section=section,
                                     metric="accuracy" if section == "classification_scores" else "test_metric",
                                     entity=entity, unit="ratio", value=family_num + path_num / 10 + 1.2345678901234567,
                                     timestamp=pd.NaT))
            rows.append(dict(model_family=family, path=path, section="efficiency",
                             metric="undefined", entity="final_account", unit="ratio",
                             value=np.nan, timestamp=pd.NaT))
            rows.append(dict(model_family=family, path=path, section="exposure",
                             metric="mean_absolute_weight", entity="S00", unit="ratio",
                             value=.5, timestamp=pd.NaT))
    original = pd.DataFrame(rows)
    legacy = tmp_path / "statistics.parquet"
    original.to_parquet(legacy, index=False)
    backup = tmp_path / "backup.parquet"
    backup.write_bytes(legacy.read_bytes())
    converted = workflow.format_backtest_statistics(pd.read_parquet(legacy))
    pending = tmp_path / "statistics.pending.parquet"
    converted.to_parquet(pending, index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(pending), converted)
    pending.replace(legacy)
    pd.testing.assert_frame_equal(pd.read_parquet(backup), original)
    assert list(converted) == ["model_family", "section", "metric", "unit", *paths]
    assert list(converted.model_family.drop_duplicates()) == list(workflow.META_FAMILIES)
    assert list(converted.section.drop_duplicates()) == list(workflow.STATISTICS_SECTIONS)
    assert set(converted.section) == set(workflow.STATISTICS_SECTIONS)
    for family in workflow.META_FAMILIES:
        actual = converted.loc[converted.model_family.eq(family)].set_index(["section", "metric", "unit"])
        for path in paths:
            expected = original.loc[original.model_family.eq(family) & original.path.eq(path)
                                    & original.section.isin(workflow.STATISTICS_SECTIONS)].copy()
            expected["metric"] = expected.metric.where(expected.section.ne("classification_scores"),
                                                        expected.entity + "_" + expected.metric)
            expected = expected.set_index(["section", "metric", "unit"]).reindex(actual.index)
            assert isinstance(actual[path].dtype, pd.StringDtype)
            dates = expected.index.get_level_values("unit") == "timestamp"
            for index, value in expected.loc[dates, "timestamp"].items():
                assert pd.Timestamp(actual.loc[index, path]) == value
            numeric = actual.loc[~dates, path]
            parsed = [float(value) if pd.notna(value) else np.nan for value in numeric]
            np.testing.assert_array_equal(parsed, expected.loc[~dates, "value"].to_numpy())
            assert pd.isna(actual.loc[("efficiency", "undefined", "ratio"), path])


@pytest.mark.parametrize("other_timestamp", [pd.Timestamp("2025-02-04T00:00Z"), pd.NaT])
def test_statistics_preserves_different_path_timestamps(other_timestamp):
    start = pd.Timestamp("2025-02-03T00:00Z")
    statistics = pd.DataFrame({
        "model_family": "bagging", "section": "general_characteristics",
        "metric": "start", "unit": "timestamp", "entity": "final_account",
        "value": np.nan, "path": ["path_0", "path_1"],
        "timestamp": [start, other_timestamp],
    })
    converted = workflow.format_backtest_statistics(statistics)
    assert converted.loc[0, "path_0"] == start.isoformat()
    if pd.isna(other_timestamp):
        assert pd.isna(converted.loc[0, "path_1"])
    else:
        assert pd.Timestamp(converted.loc[0, "path_1"]) == other_timestamp


@pytest.mark.parametrize("artifact_name", ["primary_model.joblib", "meta_models.joblib"])
def test_backtesting_rejects_legacy_market_data_models(strategy_workspace, artifact_name):
    inputs = holdout_inputs(strategy_workspace)
    path = inputs["paths"].artifacts / artifact_name
    artifact = joblib.load(path)
    artifact.pop("market_data_version")
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="Market-data rules changed"):
        holdout_inputs(strategy_workspace)


def test_backtesting_rejects_legacy_market_data_results(tmp_path):
    directory = tmp_path / "data/backtesting/market_synchronous/holdout"
    directory.mkdir(parents=True)
    (directory / "settings.json").write_text(json.dumps({
        "strategy": "market_synchronous", "partition": "holdout",
    }))
    with pytest.raises(ValueError, match="Market-data rules changed"):
        workflow.load_backtest_results(tmp_path, "market_synchronous", "holdout")


@pytest.mark.parametrize("artifact_name", ["primary_model.joblib", "meta_models.joblib"])
def test_backtesting_requires_explicit_selection_ids(strategy_workspace, artifact_name):
    inputs = holdout_inputs(strategy_workspace)
    path = inputs["paths"].artifacts / artifact_name
    artifact = joblib.load(path)
    artifact.pop("configuration_id")
    joblib.dump(artifact, path)
    with pytest.raises(ValueError, match="Explicit Primary and Meta configuration IDs"):
        holdout_inputs(strategy_workspace)


@pytest.mark.parametrize("partition", ["development", "holdout"])
def test_classification_report_matches_modeling_and_round_trips(tmp_path, partition):
    from src.modeling.modeling_workflow import build_model_evaluation_table

    predictions = pd.DataFrame({
        "primary_label": [-1, 1, 1, -1], "primary_prediction": [-1, 1, -1, 1],
        "primary_positive_probability": [.2, .8, .4, .6],
        "meta_label": [0, 1, 1, 0], "meta_prediction": [1, 1, 0, 0],
        "meta_pass_probability": [.7, .8, .3, .2], "sample_weight": [4., 2., 1., .5],
    })
    expected = {}
    for entity, labels, probability in (
        ("primary", [-1, 1], "primary_positive_probability"),
        ("meta", [0, 1], "meta_pass_probability"),
    ):
        table = build_model_evaluation_table(
            {"test": predictions[[f"{entity}_prediction", probability]].rename(
                columns={f"{entity}_prediction": "prediction", probability: "probability"},
            )}, predictions[f"{entity}_label"], predictions.sample_weight, class_labels=labels,
        )
        expected.update({f"{entity}_{metric}": value for metric, value in table.loc["Test"].items()})
    path_names = ["path_0", "path_1"] if partition == "development" else ["holdout"]
    classification = workflow.classification_statistics(predictions)
    tables = []
    for path in path_names:
        account = pd.DataFrame([dict(
            section="performance", metric="test_return", entity="final_account",
            value=.125, timestamp=pd.NaT, unit="ratio",
        )])
        tables.append(pd.concat([account, classification], ignore_index=True).assign(
            model_family="boosting", path=path,
        ))
    report = workflow.format_backtest_statistics(pd.concat(tables, ignore_index=True))
    rows = report.loc[report.section.eq("classification_scores")]
    assert rows.metric.tolist() == list(expected)
    assert not rows.metric.str.contains("f1_score|negative_log_loss").any()
    for path in path_names:
        assert rows.set_index("metric")[path].astype(float).to_dict() == pytest.approx(expected)
        assert float(report.loc[report.metric.eq("test_return"), path].iloc[0]) == .125
    directory = tmp_path / partition
    directory.mkdir()
    report.to_parquet(directory / "statistics.parquet", index=False)
    pd.testing.assert_frame_equal(report, pd.read_parquet(directory / "statistics.parquet"))
