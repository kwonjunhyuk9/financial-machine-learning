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
from src.preprocessing.market_data import ResearchPaths


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
                        selected_candidate="tree", best_configuration={"max_depth": 3}, random_state=42)
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
            estimators={name: DecisionTreeClassifier(max_depth=3, random_state=42) for name in workflow.META_FAMILIES},
            oof_predictions={name: meta_oof for name in workflow.META_FAMILIES},
            configurations={name: {"max_depth": 3} for name in workflow.META_FAMILIES},
            feature_columns=meta_features, artifact_dir=paths.artifacts,
            selected_candidate="boosting", random_state=42,
        )
    for symbol in [*symbols, "SPY"]:
        file = common.feature(symbol, "dollar_bars")
        file.parent.mkdir(parents=True, exist_ok=True)
        times = pd.DatetimeIndex([opens[0] - pd.Timedelta(days=1)])
        for session in calendar.itertuples():
            times = times.union(pd.date_range(session.open, session.close, freq="min"))
        pd.DataFrame({"end": times, "close": 100. + np.arange(len(times)) * .001}).to_parquet(file, index=False)
    return tmp_path



@pytest.mark.parametrize("strategy", ["sentiment_asynchronous", "market_synchronous", "sentiment_synchronous"])
@pytest.mark.parametrize("partition, benchmark_symbol", [("holdout", "S00"), ("development", "SPY")])
def test_notebook_runs_independently_and_saves_only_final_results(strategy_workspace, monkeypatch, strategy, partition, benchmark_symbol):
    import IPython.display

    root = strategy_workspace
    directory = root / "notebooks/backtesting" / strategy
    directory.mkdir(parents=True)
    monkeypatch.chdir(directory)
    monkeypatch.setattr(IPython.display, "display", lambda *args: None)
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
    expected = 15 if partition == "development" else 3
    assert len(namespace["accounts"]) == expected
    assert calls == (["async"] if strategy.endswith("asynchronous") else ["sync"])
    output = root / "data/backtesting" / strategy / partition
    assert {p.name for p in output.iterdir()} == {"statistics.parquet", "account.parquet", "settings.json"}
    assert not list(root.rglob("cpcv_predictions.parquet"))
    assert not list(root.rglob("price_calibration.parquet"))
    stats, curves, metadata = workflow.load_backtest_results(root, strategy, partition)
    asynchronous = strategy.endswith("asynchronous")
    assert metadata["execution"]["engine_version"] == (9 if asynchronous else 3)
    assert metadata["execution"]["recording"] == ("minute_last_and_execution_boundaries" if asynchronous
                                                   else "session_and_execution_boundaries")
    assert metadata["execution"]["scheduler"] == ("bar_stream" if asynchronous else "synchronous_event_scheduler_v1")
    assert metadata["execution"]["observation"] == "dollar_bar"
    assert metadata["execution"]["benchmark"] == "daily_last_bar_close"
    assert metadata["execution"]["decision"] == ("initial_or_event_exit" if asynchronous else "session_boundary")
    assert metadata["benchmark"] == metadata["evaluation"]["benchmark_symbol"] == benchmark_symbol
    assert set(stats.loc[stats.section.eq("benchmark"), "entity"]) == {benchmark_symbol}
    asynchronous = strategy == "sentiment_asynchronous"
    assert metadata["portfolio"]["horizon_bars"] == (780 if asynchronous else 1000)
    assert metadata["portfolio"]["horizontal_multipliers"] == ([5.0, 5.0] if asynchronous else [1.0, 1.0])
    if asynchronous:
        assert metadata["execution"]["ranking"] == "meta_probability_time_decay"
        assert metadata["execution"]["candidate_queue"] == "persistent_bounded_per_side"
        assert metadata["execution"]["candidate_queue_capacity_per_side"] == metadata["portfolio"]["max_positions_per_side"]
        assert metadata["execution"]["entry_limit_policy"] == "discard_on_violation"
        assert metadata["execution"]["limit_refresh"] == "new_event"
        assert metadata["execution"]["allocation"] == "directional_meta_probability_normalized"
        assert namespace["settings"].horizon_bars == 780
        assert tuple(namespace["settings"].horizontal_multipliers) == (5.0, 5.0)
        assert all(item["events"].horizon_bars.eq(780).all() for item in namespace["prepared"].values())
    expected_limit = 10
    assert metadata["portfolio"]["max_positions_per_side"] == expected_limit
    assert "k" not in metadata["portfolio"]
    limits = stats.loc[stats.section.eq("assumption") & stats.metric.eq("max_positions_per_side")]
    assert len(limits) == expected
    assert limits.value.eq(expected_limit).all()
    assert not stats.metric.eq("K").any()
    assert "horizon_bars" not in metadata
    pd.testing.assert_frame_equal(stats, namespace["statistics"])
    assert set(stats.model_family) == set(workflow.META_FAMILIES)
    assert curves.groupby(["model_family", "path", "entity", "timestamp"]).size().max() == 1
    assert set(stats.section) >= {"general_characteristics", "performance", "runs", "implementation_shortfall", "efficiency", "classification_scores"}
    if partition == "development":
        assert curves.timestamp.max() < namespace["inputs"]["events"].holdout_boundary.iloc[0]
    else:
        assert stats.path.eq("holdout").all()
        assert metadata["validation"] is None
    for key, result in namespace["accounts"].items():
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
    account = pd.DataFrame({"timestamp": [inputs["start"], inputs["end"]], "aum": [100000., 101000.]})
    accounts = {(family, "holdout"): {"ledger": account} for family in workflow.META_FAMILIES}
    statistics = pd.DataFrame({"model_family": list(workflow.META_FAMILIES), "path": "holdout", "value": [1., 2., 3.]})
    holdout = workflow.save_backtest_results(inputs, statistics, accounts, evaluation_settings={"benchmark_symbol": "SPY"})
    original = (holdout / "statistics.parquet").read_bytes()
    development = workflow.save_backtest_results(dict(inputs, partition="development"), statistics, accounts, evaluation_settings={"benchmark_symbol": "SPY"})
    assert development != holdout
    assert (holdout / "statistics.parquet").read_bytes() == original
    assert set(pd.read_parquet(holdout / "account.parquet").model_family) == set(workflow.META_FAMILIES)
    saved_statistics, saved_accounts, metadata = workflow.load_backtest_results(
        strategy_workspace, "sentiment_asynchronous", "holdout",
    )
    assert metadata["portfolio"]["horizon_bars"] == 7
    assert "horizon_bars" not in metadata
    # Existing result files retain their original metadata shape when loaded.
    metadata["horizon_bars"] = metadata["portfolio"].pop("horizon_bars")
    (holdout / "settings.json").write_text(json.dumps(metadata))
    legacy_statistics, legacy_accounts, legacy_metadata = workflow.load_backtest_results(
        strategy_workspace, "sentiment_asynchronous", "holdout",
    )
    pd.testing.assert_frame_equal(legacy_statistics, saved_statistics)
    pd.testing.assert_frame_equal(legacy_accounts, saved_accounts)
    assert legacy_metadata == metadata
