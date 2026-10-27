from src.preprocessing.market_technical_indicators import MODEL_FEATURES, TECHNICAL_FEATURES
from src.modeling.purged_validation import index_events
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier

import src.modeling.modeling_workflow as modeling_workflow
from src.modeling.purged_validation import PurgedKFold
from src.modeling.modeling_workflow import (
    build_candidate_classifiers,
    build_meta_development_frame,
    build_meta_model_frame,
    build_model_evaluation_table,
    build_primary_model_frame,
    compute_stage_importance,
    generate_oof_predictions,
    get_meta_feature_columns,
    get_primary_feature_columns,
    get_weighted_learning_curve,
    plot_learning_curves,
    run_model_selection_workflow,
    score_binary_predictions,
)


CANDIDATE_SETTINGS = {
    "boosting": {"n_estimators": 100, "learning_rate": .10},
    "bagging": {"n_estimators": 120, "max_samples": .80},
    "random_forest": {"n_estimators": 120},
}
PARAMETER_GRIDS = {
    "boosting": {"model__learning_rate": [.03, .10, .30]},
    "bagging": {"model__max_samples": [.60, .80, 1.00]},
    "random_forest": {"model__max_features": ["sqrt", .50, 1.00]},
}


def _events(num_rows: int = 10) -> pd.DataFrame:
    starts = pd.date_range("2025-01-02", periods=num_rows, freq="D", tz="UTC")
    ends = starts + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "event_start": starts,
            "symbol": "AAPL",
            "event_end": ends,
            "vertical_barrier": ends + pd.Timedelta(hours=1),
            "horizon_volatility": 0.01,
            "raw_return": np.where(np.arange(num_rows) % 2 == 0, 0.02, -0.01),
            "primary_label": np.where(np.arange(num_rows) % 2 == 0, 1, -1),
            "partition": ["development"] * (num_rows - 2) + ["holdout"] * 2,
            "holdout_boundary": starts[-2],
            "sample_weight": 1.0,
            "mean_sentiment_score": np.linspace(-1.0, 1.0, num_rows),
            "fractionally_differenced_log_close": np.linspace(0.0, 0.5, num_rows),
            **{name: np.linspace(40.0, 60.0, num_rows) for name in TECHNICAL_FEATURES},
        }
    )


def test_primary_feature_columns_exclude_outcomes_and_identifiers():
    columns = get_primary_feature_columns(_events())

    assert columns == list(MODEL_FEATURES)


@pytest.mark.parametrize("indexed", [False, True])
def test_primary_model_frame_selects_and_sorts_requested_events(indexed):
    events = _events()
    requested = events.loc[[3, 1], ["symbol", "event_start"]]
    input_events = (
        events.set_index("event_start")
        if indexed
        else events
    )
    original = input_events.copy(deep=True)

    frame = build_primary_model_frame(input_events, requested)

    assert frame.index.names == ["symbol", "event_start"]
    assert frame.index.tolist() == list(index_events(requested).index)
    assert "event_start" not in frame.columns
    pd.testing.assert_frame_equal(input_events, original)


def test_primary_model_frame_rejects_invalid_event_contracts():
    events = _events()
    duplicated = pd.concat([events, events.iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="unique valid"):
        build_primary_model_frame(duplicated, events[["symbol", "event_start"]])
    with pytest.raises(ValueError, match="unique valid"):
        build_primary_model_frame(events, events.loc[[0, 0], ["symbol", "event_start"]])
    with pytest.raises(ValueError, match="absent"):
        build_primary_model_frame(
            events,
            pd.DataFrame({"symbol": ["AAPL"], "event_start": [events["event_start"].max() + pd.Timedelta(days=1)]}),
        )
    with pytest.raises(ValueError, match="required primary model columns"):
        build_primary_model_frame(
            events.drop(columns="event_end"),
            events[["symbol", "event_start"]],
        )


def test_candidate_classifiers_use_required_model_families():
    candidates = build_candidate_classifiers(random_state=42, n_jobs=1, candidate_settings=CANDIDATE_SETTINGS)

    assert list(candidates) == ["boosting", "bagging", "random_forest"]
    assert all(
        candidate.steps == [("model", candidate["model"])]
        for candidate in candidates.values()
    )


def test_candidate_classifiers_require_explicit_settings():
    with pytest.raises(TypeError, match="candidate_settings"):
        build_candidate_classifiers(random_state=42, n_jobs=1)


@pytest.mark.parametrize(
    ("label_values", "scoring", "expected_name"),
    [
        ([-1, 1], "precision", "bagging"),
        ([0, 1], "neg_log_loss", "boosting"),
        ([0, 1], "f1", "bagging"),
    ],
)
def test_model_selection_workflow_uses_stage_objective(
    monkeypatch,
    label_values,
    scoring,
    expected_name,
):
    starts = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
    features = pd.DataFrame({"feature": np.arange(10)}, index=starts)
    labels = pd.Series(np.tile(label_values, 5), index=starts)
    weights = pd.Series(1.0, index=starts)
    information_sets = pd.Series(starts + pd.Timedelta(hours=1), index=starts)
    candidates = {
        name: modeling_workflow.MyPipeline([
            ("model", DecisionTreeClassifier(max_depth=depth, random_state=42))
        ])
        for name, depth in {
            "boosting": 1,
            "bagging": 2,
            "random_forest": 3,
        }.items()
    }
    scores_by_marker = {
        0.1: {"log_loss": 0.2, "f1": 0.5},
        0.2: {"log_loss": 0.3, "f1": 0.8},
        0.3: {"log_loss": 0.4, "f1": 0.7},
        0.4: {"log_loss": 0.1, "f1": 0.6},
        0.5: {"log_loss": 0.25, "f1": 0.9},
        0.6: {"log_loss": 0.3, "f1": 0.7},
    }

    monkeypatch.setattr(
        modeling_workflow,
        "build_candidate_classifiers",
        lambda **kwargs: candidates,
    )

    def fake_oof(estimator, features, labels, sample_weight, cv, positive_label, **kwargs):
        marker = estimator["model"].max_depth / 10
        return pd.DataFrame({
            "prediction": labels,
            "probability": marker,
            "fold": 0,
            "prediction_source": "oof",
        }, index=features.index)

    def fake_scores(labels, predictions, probabilities, sample_weight, **kwargs):
        selected = scores_by_marker[float(probabilities.iloc[0])]
        return {
            **selected,
            "accuracy": 1.0,
            "precision": selected["f1"],
            "recall": 1.0,
        }

    monkeypatch.setattr(modeling_workflow, "generate_oof_predictions", fake_oof)
    monkeypatch.setattr(modeling_workflow, "score_binary_predictions", fake_scores)

    result = run_model_selection_workflow(
        features,
        labels,
        weights,
        information_sets,
        scoring=scoring,
        cv=2,
        random_state=42,
        n_jobs=1,
        candidate_settings=CANDIDATE_SETTINGS, parameter_grids={name: {"model__max_depth": [i, i + 3]}
                           for i, name in enumerate(candidates, start=1)}, pct_embargo=0.01,
    )

    assert result.selected_name == expected_name
    assert result.final_configuration == {"model__max_depth": 4 if scoring == "neg_log_loss" else 5}
    assert result.final_oof["prediction_source"].eq("oof").all()
    assert len(result.tuning) == 6
    assert list(result.estimators) == list(candidates)
    assert result.configurations == {
        "boosting": {"model__max_depth": 4},
        "bagging": {"model__max_depth": 5},
        "random_forest": {"model__max_depth": 6 if scoring == "neg_log_loss" else 3},
    }
    assert all(base["model"].max_depth == i
               for i, base in enumerate(candidates.values(), start=1))


def test_model_selection_workflow_rejects_unsupported_scoring():
    starts = pd.date_range("2026-01-01", periods=2, freq="D", tz="UTC")
    values = pd.Series([-1, 1], index=starts)
    with pytest.raises(ValueError, match="scoring"):
        run_model_selection_workflow(
            pd.DataFrame({"feature": [0, 1]}, index=starts),
            values,
            pd.Series(1.0, index=starts),
            pd.Series(starts, index=starts),
            scoring="accuracy",
            cv=2,
            candidate_settings=CANDIDATE_SETTINGS, parameter_grids=PARAMETER_GRIDS, pct_embargo=0.01, random_state=42, n_jobs=1,
        )


def test_candidate_classifiers_fit_with_weights_and_predict_probabilities():
    features = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, 20),
            "feature_b": np.tile([0.0, 1.0], 10),
        }
    )
    labels = pd.Series(np.tile([-1, 1], 10))
    weights = pd.Series(np.linspace(0.5, 1.5, 20))

    for candidate in build_candidate_classifiers(
        random_state=42,
        n_jobs=1,
        candidate_settings=CANDIDATE_SETTINGS,
    ).values():
        candidate.fit(features, labels, sample_weight=weights)
        probabilities = candidate.predict_proba(features)

        assert probabilities.shape == (20, 2)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_generate_oof_predictions_marks_every_prediction_as_oof():
    events = _events(20)
    features = events.set_index("event_start")[["mean_sentiment_score"]]
    labels = pd.Series(
        np.tile([-1, 1], 10),
        index=features.index,
        name="primary_label",
    )
    weights = pd.Series(1.0, index=features.index)
    information_sets = events.set_index("event_start")["event_end"]
    cv = PurgedKFold(n_splits=5, t1=information_sets, pct_embargo=0.01)

    predictions = generate_oof_predictions(
        estimator=DecisionTreeClassifier(random_state=42),
        features=features,
        labels=labels,
        sample_weight=weights,
        cv=cv,
        positive_label=1,
    )

    assert predictions.index.equals(features.index)
    assert predictions["prediction_source"].eq("oof").all()
    assert predictions["fold"].nunique() == 5
    assert predictions["probability"].between(0.0, 1.0).all()


@pytest.mark.parametrize(
    ("class_labels", "labels"),
    [
        ([-1, 1], [-1, 1, 1, -1]),
        ([0, 1], [0, 1, 1, 0]),
    ],
)
def test_score_binary_predictions_supports_project_label_spaces(
    class_labels,
    labels,
):
    index = pd.RangeIndex(4)
    observed = pd.Series(labels, index=index)
    predicted = pd.Series([class_labels[0], 1, class_labels[0], 1], index=index)
    probabilities = pd.Series([0.2, 0.8, 0.4, 0.6], index=index)
    weights = pd.Series([1.0, 2.0, 1.5, 0.5], index=index)

    scores = score_binary_predictions(
        observed,
        predicted,
        probabilities,
        weights,
        class_labels=class_labels,
        positive_label=1,
    )
    probability_matrix = np.column_stack([1.0 - probabilities, probabilities])

    assert scores["log_loss"] == pytest.approx(log_loss(
        observed,
        probability_matrix,
        labels=class_labels,
        sample_weight=weights,
    ))
    assert scores["accuracy"] == pytest.approx(accuracy_score(
        observed,
        predicted,
        sample_weight=weights,
    ))
    assert scores["f1"] == pytest.approx(f1_score(
        observed,
        predicted,
        pos_label=1,
        sample_weight=weights,
    ))
    assert scores["precision"] == pytest.approx(precision_score(
        observed,
        predicted,
        pos_label=1,
        sample_weight=weights,
        zero_division=0,
    ))
    assert scores["recall"] == pytest.approx(recall_score(
        observed,
        predicted,
        pos_label=1,
        sample_weight=weights,
        zero_division=0,
    ))


@pytest.mark.parametrize(
    ("class_labels", "labels"),
    [
        ([-1, 1], [-1, 1, 1, -1]),
        ([0, 1], [0, 1, 1, 0]),
    ],
)
def test_model_evaluation_table_supports_project_label_spaces(
    class_labels,
    labels,
):
    index = pd.RangeIndex(4)
    observed = pd.Series(labels, index=index)
    predictions = pd.DataFrame({
        "prediction": [class_labels[0], 1, class_labels[0], 1],
        "probability": [0.2, 0.8, 0.4, 0.6],
    }, index=index)
    weights = pd.Series([1.0, 2.0, 1.5, 0.5], index=index)

    table = build_model_evaluation_table(
        {"test_model": predictions},
        observed,
        weights,
        class_labels=class_labels,
        positive_label=1,
    )
    expected = score_binary_predictions(
        observed,
        predictions["prediction"],
        predictions["probability"],
        weights,
        class_labels=class_labels,
        positive_label=1,
    )

    assert table.index.tolist() == ["Test Model"]
    columns = ["accuracy", "precision", "recall", "f1", "log_loss"]
    if class_labels == [-1, 1]:
        columns = ["accuracy", "precision_-1", "recall_-1", "f1_-1",
                   "precision_1", "recall_1", "f1_1", "log_loss"]
        expected = {f"{key}_1" if key in {"precision", "recall", "f1"} else key: value
                    for key, value in expected.items()}
    assert table.columns.tolist() == columns
    assert table.loc["Test Model"].to_dict() == pytest.approx(expected)


def test_plot_learning_curves_returns_each_estimator_result(monkeypatch):
    from unittest.mock import Mock

    curve = pd.DataFrame({
        "train_size": [10],
        "train_error_mean": [0.2],
        "train_error_std": [0.01],
        "validation_error_mean": [0.3],
        "validation_error_std": [0.02],
    })
    calls = []

    def fake_learning_curve(estimator, *args, **kwargs):
        calls.append((estimator, args, kwargs))
        return curve

    monkeypatch.setattr(modeling_workflow, "get_weighted_learning_curve", fake_learning_curve)
    monkeypatch.setattr(modeling_workflow.plt, "show", lambda: None)

    results = plot_learning_curves(
        {"first": object(), "second": object(), "third": object()},
        "Learning Curves",
        features=pd.DataFrame(),
        labels=pd.Series(dtype=int),
        sample_weight=pd.Series(dtype=float),
        cv=Mock(get_n_splits=lambda *args: 2),
        train_sizes=[1.0],
        class_labels=[0, 1],
    )
    figure = modeling_workflow.plt.gcf()
    assert len(figure.axes) == 3
    assert all(axis.get_subplotspec().get_gridspec().get_geometry() == (1, 3)
               for axis in figure.axes)
    modeling_workflow.plt.close("all")

    assert list(results) == ["first", "second", "third"]
    assert len(calls) == 3
    assert all(result is curve for result in results.values())


def test_compute_stage_importance_returns_all_methods(monkeypatch):
    calls = []

    def fake_importance(*args, method, **kwargs):
        calls.append((args, method, kwargs))
        return pd.DataFrame({"mean": [1.0], "std": [0.0]}, index=["feature"]), 0.5

    monkeypatch.setattr(
        "src.modeling.feature_importance.get_estimator_feature_importance",
        fake_importance,
    )

    results, scores = compute_stage_importance(
        object(),
        features=pd.DataFrame(),
        labels=pd.Series(dtype=int),
        sample_weight=pd.Series(dtype=float),
        information_sets=pd.Series(dtype="datetime64[ns, UTC]"),
        scoring="f1",
        cv=2,
        pct_embargo=0.01,
        random_state=42,
    )

    assert list(results) == ["MDI", "MDA", "SFI"]
    assert [method for _, method, _ in calls] == ["MDI", "MDA", "SFI"]
    assert scores.to_dict() == {"MDI": 0.5, "MDA": 0.5, "SFI": 0.5}


@pytest.mark.parametrize(
    ("labels", "class_labels", "scoring"),
    [
        (np.tile([-1, 1], 20), [-1, 1], "neg_log_loss"),
        (np.tile([0, 1], 20), [0, 1], "f1"),
        (np.tile([0, 1], 20), [0, 1], "precision"),
    ],
)
def test_weighted_learning_curve_is_deterministic_and_increases_train_size(
    labels,
    class_labels,
    scoring,
):
    index = pd.date_range("2025-01-01", periods=40, freq="D", tz="UTC")
    features = pd.DataFrame(
        {
            "feature_a": np.linspace(-1.0, 1.0, 40),
            "feature_b": np.tile([0.0, 1.0], 20),
        },
        index=index,
    )
    observed = pd.Series(labels, index=index)
    weights = pd.Series(np.linspace(0.5, 1.5, 40), index=index)
    information_sets = pd.Series(index, index=index)
    cv = PurgedKFold(2, information_sets)
    kwargs = {
        "estimator": DecisionTreeClassifier(max_depth=2, random_state=42),
        "features": features,
        "labels": observed,
        "sample_weight": weights,
        "cv": cv,
        "train_sizes": [0.5, 1.0],
        "class_labels": class_labels,
        "scoring": scoring,
        "random_state": 42,
    }

    first = get_weighted_learning_curve(**kwargs)
    second = get_weighted_learning_curve(**kwargs)

    pd.testing.assert_frame_equal(first, second)
    assert first.columns.tolist() == [
        "train_fraction",
        "train_size",
        "train_error_mean",
        "train_error_std",
        "validation_error_mean",
        "validation_error_std",
    ]
    assert first["train_size"].is_monotonic_increasing
    assert np.isfinite(first.select_dtypes("number")).all().all()


def test_score_binary_predictions_validates_alignment_and_classes():
    labels = pd.Series([0, 1], index=["a", "b"])
    predictions = pd.Series([0, 1], index=labels.index)
    probabilities = pd.Series([0.2, 0.8], index=labels.index)
    weights = pd.Series(1.0, index=labels.index)

    with pytest.raises(ValueError, match="same index"):
        score_binary_predictions(
            labels,
            predictions.reset_index(drop=True),
            probabilities,
            weights,
            class_labels=[0, 1],
        )
    with pytest.raises(ValueError, match="exactly two"):
        score_binary_predictions(
            labels,
            predictions,
            probabilities,
            weights,
            class_labels=[-1, 0, 1],
        )
    with pytest.raises(ValueError, match="positive_label"):
        score_binary_predictions(
            labels,
            predictions,
            probabilities,
            weights,
            class_labels=[-1, 0],
        )


def test_meta_model_frame_requires_primary_oof_predictions():
    events = _events().set_index("event_start")
    primary_oof = pd.DataFrame(
        {
            "prediction": events["primary_label"],
            "probability": 0.75,
            "prediction_source": "oof",
        },
        index=events.index,
    )

    meta = build_meta_model_frame(events, primary_oof)

    expected = (meta["primary_prediction"] * meta["raw_return"] > 0).astype("int8")
    pd.testing.assert_series_equal(meta["meta_label"], expected, check_names=False)
    assert get_meta_feature_columns(meta) == [
        *MODEL_FEATURES,
        "primary_prediction",
        "primary_class_probability",
    ]

    primary_oof.loc[primary_oof.index[0], "prediction_source"] = "in_sample"
    with pytest.raises(ValueError, match="OOF"):
        build_meta_model_frame(events, primary_oof)


def test_meta_feature_columns_require_generated_features():
    with pytest.raises(ValueError, match="required meta features"):
        get_meta_feature_columns(_events())


def test_market_features_exclude_sentiment_in_both_stages():
    events = _events()
    market = get_primary_feature_columns(events, model_kind="market")
    sentiment = get_primary_feature_columns(events, model_kind="sentiment")
    assert set(sentiment) - set(market) == {"mean_sentiment_score"}
    keys = events[["symbol", "event_start"]]
    market_frame = build_primary_model_frame(events, keys, model_kind="market")
    sentiment_frame = build_primary_model_frame(events, keys, model_kind="sentiment")
    assert market_frame.index.equals(sentiment_frame.index)
    oof = pd.DataFrame({"prediction": 1, "probability": .7, "prediction_source": "oof"}, index=market_frame.index)
    meta = build_meta_model_frame(market_frame, oof)
    assert "mean_sentiment_score" not in get_meta_feature_columns(meta, model_kind="market")
    assert set(get_meta_feature_columns(meta, model_kind="market")) == set(market) | {"primary_prediction", "primary_class_probability"}
    assert get_primary_feature_columns(events.drop(columns="mean_sentiment_score"), "market") == market


@pytest.mark.parametrize("model_kind", ["market", "sentiment"])
def test_final_model_notebook_cells_preserve_artifacts_and_development_fits(
    tmp_path, monkeypatch, model_kind,
):
    import json
    from pathlib import Path

    import joblib

    events = _events(24)
    events = index_events(pd.concat([
        events.assign(symbol="A"), events.assign(symbol="B"),
    ], ignore_index=True))
    events["sample_weight"] = np.linspace(0.5, 1.5, len(events))
    original_events = events.copy(deep=True)
    development = events.loc[events.partition.eq("development")]
    holdout = events.loc[events.partition.eq("holdout")]
    features = get_primary_feature_columns(development, model_kind)
    estimator = DecisionTreeClassifier(max_depth=2, random_state=42)
    cv = PurgedKFold(3, development.event_end, 0.01)
    primary_oof = generate_oof_predictions(
        estimator, development[features], development.primary_label.astype("int8"),
        development.sample_weight, cv,
    )
    meta_development = build_meta_model_frame(development, primary_oof)
    meta_features = get_meta_feature_columns(meta_development, model_kind)
    meta_oof = generate_oof_predictions(
        estimator, meta_development[meta_features], meta_development.meta_label,
        meta_development.sample_weight, cv,
    )
    fits = []
    original_fit = DecisionTreeClassifier.fit

    def record_fit(self, X, y, sample_weight=None, **kwargs):
        fits.append((X.copy(), y.copy(), sample_weight.copy()))
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", record_fit)
    evaluations = []
    context = dict(
        pd=pd, events=events, development=development, holdout=holdout,
        meta_development=meta_development, feature_columns=features,
        meta_features=meta_features, final_estimator=estimator,
        final_name="tree", selected_name="tree", final_configuration={"max_depth": 2},
        RANDOM_STATE=42, artifact_dir=tmp_path,
        w_development=development.sample_weight,
        finalize_primary_model=modeling_workflow.finalize_primary_model,
        finalize_meta_models=modeling_workflow.finalize_meta_models,
        candidates={"tree": estimator},
        selection_result=__import__("types").SimpleNamespace(configurations={"tree": {"max_depth": 2}}),
        score_binary_predictions=score_binary_predictions,
        build_model_evaluation_table=build_model_evaluation_table,
        plot_model_evaluation=lambda *args, **kwargs: evaluations.append(args),
        display=lambda *args: None,
    )
    source_root = Path(__file__).resolve().parents[2]
    for stage, frame, oof, columns, labels, target, side, probability in [
        ("primary", development, primary_oof, features, [-1, 1],
         "primary_label", "primary_prediction", "primary_positive_probability"),
        ("meta", meta_development, meta_oof, meta_features, [0, 1],
         "meta_label", "meta_prediction", "meta_pass_probability"),
    ]:
        context.update(
            final_oof=oof, candidate_oof={"tree": oof},
            y_development=frame[target], CLASS_LABELS=labels,
        )
        path = source_root / f"notebooks/modeling/{model_kind}/{stage}_model.ipynb"
        notebook = json.loads(path.read_text())
        cell = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and f"{stage}_predictions = finalize_" in "".join(cell["source"]))
        exec(compile("".join(cell["source"]), str(path), "exec"), context)
        fitted_features, fitted_labels, fitted_weights = fits[-1]
        pd.testing.assert_frame_equal(fitted_features, frame[columns])
        pd.testing.assert_series_equal(fitted_labels, frame[target].astype("int8"))
        np.testing.assert_array_equal(fitted_weights, frame.sample_weight.to_numpy())
        predictions = context[f"{stage}_predictions"]
        saved = pd.read_parquet(tmp_path / f"{stage}_predictions.parquet")
        expected_saved = predictions.reset_index()
        for fold_column in ("primary_cv_fold", "meta_cv_fold"):
            if fold_column in expected_saved:
                expected_saved[fold_column] = pd.to_numeric(expected_saved[fold_column])
        pd.testing.assert_frame_equal(saved, expected_saved)
        assert predictions.index.equals(events.index)
        assert predictions[side].dtype == np.dtype("int8")
        dev_output = predictions.loc[predictions.partition.eq("development")]
        test_output = predictions.loc[predictions.partition.eq("holdout")]
        np.testing.assert_array_equal(dev_output[side], oof.prediction)
        np.testing.assert_array_equal(dev_output[probability], oof.probability)
        np.testing.assert_array_equal(dev_output[f"{stage}_cv_fold"], oof.fold)
        assert f"{stage}_prediction_source" not in predictions
        assert test_output[f"{stage}_cv_fold"].isna().all()
        if stage == "primary":
            artifact = joblib.load(tmp_path / "primary_model.joblib")
        else:
            bundle = joblib.load(tmp_path / "meta_models.joblib")
            assert bundle["primary_model_hash"] == modeling_workflow.model_artifact_hash(tmp_path / "primary_model.joblib")
            assert predictions.model_family.eq("tree").all()
            artifact = dict(bundle["models"]["tree"], selected_candidate=bundle["selected_candidate"])
        expected_keys = {"estimator", "feature_columns", "selected_candidate",
                         "best_configuration", "random_state"}
        if stage == "primary":
            expected_keys.add("holdout_boundary")
            assert artifact["holdout_boundary"] == holdout.index.get_level_values("event_start").min()
            np.testing.assert_allclose(predictions.primary_negative_probability, 1 - predictions.primary_positive_probability)
            np.testing.assert_allclose(predictions.primary_class_probability, np.maximum(
                predictions.primary_positive_probability,
                1 - predictions.primary_positive_probability,
            ))
        else:
            primary_holdout = context["primary_predictions"].loc[test_output.index]
            np.testing.assert_array_equal(test_output.meta_label, (
                primary_holdout.primary_prediction * holdout.raw_return > 0
            ).astype("int8"))
        assert set(artifact) == expected_keys
        assert artifact["feature_columns"] == columns
        assert artifact["selected_candidate"] == "tree"
        assert artifact["best_configuration"] == {"max_depth": 2}
        assert artifact["random_state"] == 42
        holdout_inputs = holdout.copy()
        if stage == "meta":
            for name in ("primary_prediction", "primary_positive_probability", "primary_class_probability"):
                holdout_inputs[name] = primary_holdout[name]
        np.testing.assert_array_equal(
            artifact["estimator"].predict(holdout_inputs[columns]), test_output[side],
        )
        np.testing.assert_array_equal(
            artifact["estimator"].predict_proba(holdout_inputs[columns])[:, 1],
            test_output[probability],
        )
    assert len(fits) == 2
    assert len(evaluations) == 2
    assert not hasattr(estimator, "tree_")
    pd.testing.assert_frame_equal(events, original_events)


@pytest.mark.parametrize(
    ("scoring", "family_scores", "expected_name"),
    [
        ("neg_log_loss", [(0.2, 0.4), (0.2, 0.8), (0.2, 0.8)], "boosting"),
        ("f1", [(0.4, 0.8), (0.2, 0.8), (0.2, 0.8)], "boosting"),
        ("neg_log_loss", [(0.2, 0.8)] * 3, "boosting"),
        ("f1", [(0.2, 0.8)] * 3, "boosting"),
        ("precision", [(0.4, 0.8), (0.2, 0.8), (0.2, 0.8)], "boosting"),
        ("precision", [(0.2, 0.8)] * 3, "boosting"),
    ],
)
def test_full_search_uses_only_stage_objective_and_keeps_first_ties(
    monkeypatch, scoring, family_scores, expected_name,
):
    index = pd.date_range("2025-01-01", periods=6, tz="UTC")
    X = pd.DataFrame({"feature": range(6)}, index=index)
    y = pd.Series([0, 1] * 3, index=index)
    w = pd.Series([1, 2] * 3, index=index)
    t1 = pd.Series(index + pd.Timedelta(hours=1), index=index)
    names = ["boosting", "bagging", "random_forest"]
    bases = {name: modeling_workflow.MyPipeline([("model", DecisionTreeClassifier())])
             for name in names}
    monkeypatch.setattr(modeling_workflow, "build_candidate_classifiers", lambda **kwargs: bases)

    def oof(estimator, *args, **kwargs):
        return pd.DataFrame({"prediction": y, "probability": estimator["model"].max_depth / 10,
                             "prediction_source": "oof", "fold": 0}, index=index)

    def scores(labels, predictions, probabilities, weights, **kwargs):
        family = (round(probabilities.iloc[0] * 10) - 1) // 2
        loss, f1 = family_scores[family]
        if round(probabilities.iloc[0] * 10) % 2:
            if scoring == "neg_log_loss":
                f1 += 0.1
            else:
                loss -= 0.05
        pd.testing.assert_series_equal(weights, w)
        return {"log_loss": loss, "f1": f1, "precision": f1}

    monkeypatch.setattr(modeling_workflow, "generate_oof_predictions", oof)
    monkeypatch.setattr(modeling_workflow, "score_binary_predictions", scores)
    result = run_model_selection_workflow(
        X, y, w, t1, scoring=scoring, cv=2, pct_embargo=0.01,
        random_state=42, n_jobs=1, candidate_settings={},
        parameter_grids={name: {"model__max_depth": [2 * i + 2, 2 * i + 1]}
                         for i, name in enumerate(names)},
    )
    assert result.selected_name == expected_name
    assert result.configurations == {name: {"model__max_depth": 2 * i + 2}
                                     for i, name in enumerate(names)}
    assert len(result.tuning) == 6


def test_meta_family_bundle_fits_only_development_and_preserves_each_oof(tmp_path, monkeypatch):
    import joblib

    events = index_events(_events(30).assign(symbol="A"))
    development = events.loc[events.partition.eq("development")]
    holdout = events.loc[events.partition.eq("holdout")]
    features = get_primary_feature_columns(events, "market")
    primary = DecisionTreeClassifier(max_depth=2, random_state=42)
    cv = PurgedKFold(3, development.event_end, 0.01)
    primary_oof = generate_oof_predictions(primary, development[features], development.primary_label,
                                           development.sample_weight, cv)
    primary_predictions = modeling_workflow.finalize_primary_model(
        development, holdout, estimator=primary, final_oof=primary_oof, feature_columns=features,
        artifact_dir=tmp_path, selected_candidate="tree", best_configuration={"max_depth": 2}, random_state=42,
    )
    meta = build_meta_model_frame(development, primary_oof)
    columns = get_meta_feature_columns(meta, "market")
    estimators = {name: DecisionTreeClassifier(max_depth=depth, random_state=42)
                  for name, depth in zip(("boosting", "bagging", "random_forest"), (1, 2, 3))}
    oof = {name: generate_oof_predictions(estimator, meta[columns], meta.meta_label, meta.sample_weight, cv)
           for name, estimator in estimators.items()}
    for frame in oof.values():
        frame["fold"] += 10
    fits = []
    original_fit = DecisionTreeClassifier.fit

    def fit(self, X, y, sample_weight=None, **kwargs):
        fits.append(X.index)
        return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", fit)
    result = modeling_workflow.finalize_meta_models(
        meta, events, primary_predictions, estimators=estimators, oof_predictions=oof,
        configurations={name: {"max_depth": estimator.max_depth} for name, estimator in estimators.items()},
        feature_columns=columns, artifact_dir=tmp_path, selected_candidate="bagging", random_state=42,
    )
    assert len(fits) == 3 and all(index.equals(development.index) for index in fits)
    bundle = joblib.load(tmp_path / "meta_models.joblib")
    assert bundle["selected_candidate"] == "bagging"
    assert set(bundle["models"]) == set(estimators)
    saved = pd.read_parquet(tmp_path / "meta_predictions.parquet")
    expected_primary_columns = [
        "symbol", "event_start", "event_end", "vertical_barrier", "horizon_volatility",
        "raw_return", "primary_label", "sample_weight", "holdout_boundary", "partition",
        "primary_prediction", "primary_positive_probability", "primary_negative_probability",
        "primary_class_probability", "primary_cv_fold",
    ]
    assert primary_predictions.reset_index().columns.tolist() == expected_primary_columns
    assert saved.columns.tolist() == expected_primary_columns + [
        "meta_label", "meta_prediction", "meta_pass_probability",
        "meta_cv_fold", "model_family",
    ]
    assert not set(MODEL_FEATURES).intersection(saved.columns)
    pd.testing.assert_frame_equal(
        primary_predictions[list(modeling_workflow.PRIMARY_EVENT_COLUMNS)],
        events[list(modeling_workflow.PRIMARY_EVENT_COLUMNS)],
    )
    connected = build_meta_development_frame(events, primary_predictions)
    assert get_meta_feature_columns(connected, "market") == columns
    for family in estimators:
        output = result.loc[result.model_family.eq(family)]
        pd.testing.assert_frame_equal(output[primary_predictions.columns], primary_predictions)
        np.testing.assert_array_equal(output.loc[development.index, "meta_cv_fold"], oof[family].fold)
        assert output.loc[holdout.index, "meta_cv_fold"].isna().all()
    assert not saved.duplicated(["model_family", "symbol", "event_start"]).any()
    assert not (tmp_path / "meta_model.joblib").exists()
    for family in estimators:
        family_predictions = result.loc[result.model_family.eq(family)]
        assert family_predictions.index.equals(events.index)
        np.testing.assert_array_equal(family_predictions.loc[development.index, "meta_pass_probability"], oof[family].probability)
        assert bundle["models"][family]["best_configuration"]["max_depth"] == estimators[family].max_depth
        assert not hasattr(estimators[family], "tree_")


@pytest.fixture
def modeling_connection_inputs():
    events = pd.concat([
        _events(12).assign(symbol="A"),
        _events(12).assign(symbol="B"),
    ], ignore_index=True).sort_values(["event_start", "symbol"], ignore_index=True)
    indexed = index_events(events)
    predictions = pd.DataFrame({
        "partition": indexed.partition,
        "primary_prediction": indexed.primary_label.astype("int8"),
        "primary_positive_probability": np.linspace(0.1, 0.9, len(indexed)),
        "primary_class_probability": np.maximum(np.linspace(0.1, 0.9, len(indexed)), 1 - np.linspace(0.1, 0.9, len(indexed))),
        "primary_cv_fold": np.where(indexed.partition.eq("development"), 0, np.nan),
    }, index=indexed.index)
    return events, predictions


@pytest.mark.parametrize("reordered", [False, True])
def test_meta_development_frame_matches_previous_connection(modeling_connection_inputs, reordered):
    raw_events, predictions = modeling_connection_inputs
    events = index_events(raw_events)
    if reordered:
        predictions = predictions.iloc[::-1]
    originals = [frame.copy(deep=True) for frame in (events, predictions)]
    development_predictions = predictions.loc[predictions.partition.eq("development")]
    expected = build_meta_model_frame(
        events.loc[development_predictions.index],
        development_predictions.rename(columns={
            "primary_prediction": "prediction", "primary_positive_probability": "probability",
        }).assign(prediction_source="oof")[["prediction", "probability", "prediction_source"]],
    )

    expected[development_predictions.columns] = development_predictions
    actual = build_meta_development_frame(events, predictions)

    pd.testing.assert_frame_equal(actual, expected[actual.columns])
    assert actual.index.equals(development_predictions.index)
    assert actual.partition.eq("development").all()
    for frame, original in zip((events, predictions), originals):
        pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("change", ["partition", "missing_event"])
def test_meta_development_frame_preserves_connection_errors(modeling_connection_inputs, change):
    raw_events, predictions = modeling_connection_inputs
    events = index_events(raw_events)
    development_key = predictions.index[predictions.partition.eq("development")][0]
    exception, message = ValueError, "Primary predictions must match event partitions"
    if change == "partition":
        predictions.loc[development_key, "partition"] = "holdout"
    else:
        events = events.drop(index=development_key)
        exception, message = KeyError, None

    with pytest.raises(exception, match=message):
        build_meta_development_frame(events, predictions)


@pytest.mark.parametrize("model_kind", ["market", "sentiment"])
def test_modeling_notebook_preparation_cells_connect_with_synthetic_inputs(
    tmp_path, monkeypatch, modeling_connection_inputs, model_kind,
):
    import json
    from pathlib import Path

    events, predictions = modeling_connection_inputs
    read_parquet = pd.read_parquet

    def read_inputs(path):
        if path.name == "model.parquet":
            return events.copy(deep=True)
        if path.name == "primary_predictions.parquet":
            return predictions.reset_index()
        return read_parquet(path)

    monkeypatch.setattr(pd, "read_parquet", read_inputs)
    environment = dict(
        pd=pd, event_path=tmp_path / "model.parquet", artifact_dir=tmp_path,
        CV_SPLITS=3, PCT_EMBARGO=0.01, index_events=index_events, PurgedKFold=PurgedKFold,
        build_primary_model_frame=build_primary_model_frame,
        get_primary_feature_columns=get_primary_feature_columns,
        build_meta_development_frame=build_meta_development_frame,
        get_meta_feature_columns=get_meta_feature_columns,
    )
    root = Path(__file__).resolve().parents[2]
    for stage in ("primary", "meta"):
        notebook_path = root / f"notebooks/modeling/{model_kind}/{stage}_model.ipynb"
        notebook = json.loads(notebook_path.read_text())
        displayed = []
        environment["display"] = displayed.append
        for prefix in ("inputs", "validation"):
            cell_id = f"{prefix}-{stage}-{model_kind}"
            cell = next(cell for cell in notebook["cells"] if cell.get("id") == cell_id)
            exec(compile("".join(cell["source"]), str(notebook_path), "exec"), environment)
        summary = next(cell for cell in notebook["cells"] if cell["cell_type"] == "code" and "".join(cell["source"]).startswith("display(pd.DataFrame(fold_rows)"))
        exec(compile("".join(summary["source"]), str(notebook_path), "exec"), environment)
        assert len(displayed) == 1
        assert displayed[0].test_events.sum() == len(environment["X_development"])
        assert environment["X_development"].index.equals(environment["y_development"].index)
        assert environment["X_development"].index.equals(environment["w_development"].index)
        assert environment["X_development"].index.equals(environment["t1_development"].index)
        assert len(environment["X_development"].columns) == (
            (49 if model_kind == "market" else 50) + (2 if stage == "meta" else 0)
        )
        assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("model_kind", ["market", "sentiment"])
def test_primary_class_probability_is_maximum_even_when_prediction_disagrees(tmp_path, model_kind):
    events = index_events(_events(12))
    development = events.loc[events.partition.eq("development")]
    holdout = events.loc[events.partition.eq("holdout")]
    probabilities = np.where(np.arange(len(development)) % 2, 0.8, 0.2)
    oof = pd.DataFrame({
        "prediction": np.where(probabilities > 0.5, -1, 1),
        "probability": probabilities, "prediction_source": "oof", "fold": 0,
    }, index=development.index)
    output = modeling_workflow.finalize_primary_model(
        development, holdout, estimator=DecisionTreeClassifier(max_depth=2),
        final_oof=oof, feature_columns=get_primary_feature_columns(events, model_kind),
        artifact_dir=tmp_path, selected_candidate="tree", best_configuration={}, random_state=42,
    )
    np.testing.assert_allclose(output.loc[development.index, "primary_class_probability"], 0.8)
    assert not set(MODEL_FEATURES).intersection(output.columns)


@pytest.mark.parametrize("duplicate", ["events", "predictions"])
def test_meta_development_rejects_duplicate_event_keys(modeling_connection_inputs, duplicate):
    raw_events, predictions = modeling_connection_inputs
    events = index_events(raw_events)
    if duplicate == "events":
        events = pd.concat([events, events.iloc[[0]]])
    else:
        predictions = pd.concat([predictions, predictions.iloc[[0]]])
    with pytest.raises(ValueError, match="unique composite event keys"):
        build_meta_development_frame(events, predictions)


@pytest.mark.parametrize("column", ["primary_cv_fold", "meta_cv_fold"])
@pytest.mark.parametrize("bad_fold", [None, -1, 0.5, float("inf"), "invalid"])
def test_saved_development_requires_nonnegative_integer_folds(column, bad_fold):
    frame = pd.DataFrame({"partition": ["development"], column: [bad_fold]})
    with pytest.raises(ValueError, match="nonnegative integer folds"):
        modeling_workflow.validate_prediction_partitions(frame, (column,))


@pytest.mark.parametrize("column", ["primary_cv_fold", "meta_cv_fold"])
def test_saved_holdout_requires_missing_folds(column):
    frame = pd.DataFrame({"partition": ["holdout"], column: [0]})
    with pytest.raises(ValueError, match="must be missing"):
        modeling_workflow.validate_prediction_partitions(frame, (column,))


@pytest.mark.parametrize("stage", ["primary", "meta"])
@pytest.mark.parametrize("problem", ["source", "index", "fold"])
def test_finalize_rejects_invalid_oof_before_fitting(tmp_path, monkeypatch, stage, problem):
    events = index_events(_events(12))
    development = events.loc[events.partition.eq("development")]
    oof = pd.DataFrame({"prediction": 1, "probability": 0.7,
                        "prediction_source": "oof", "fold": 0}, index=development.index)
    if problem == "source":
        oof["prediction_source"] = "in_sample"
    elif problem == "index":
        oof = oof.iloc[::-1]
    else:
        oof["fold"] = -1

    def unexpected_fit(*args, **kwargs):
        raise AssertionError("Invalid OOF must be rejected before fitting")

    monkeypatch.setattr(DecisionTreeClassifier, "fit", unexpected_fit)
    with pytest.raises(ValueError):
        if stage == "primary":
            modeling_workflow.finalize_primary_model(
                development, events.loc[events.partition.eq("holdout")],
                estimator=DecisionTreeClassifier(), final_oof=oof,
                feature_columns=get_primary_feature_columns(events), artifact_dir=tmp_path,
                selected_candidate="tree", best_configuration={}, random_state=42,
            )
        else:
            modeling_workflow.finalize_meta_models(
                development, events, pd.DataFrame(), estimators={"tree": DecisionTreeClassifier()},
                oof_predictions={"tree": oof}, configurations={"tree": {}},
                feature_columns=[], artifact_dir=tmp_path, selected_candidate="tree", random_state=42,
            )
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("scoring,expected_name", [
    ("f1", "boosting"), ("precision", "bagging"),
])
def test_meta_selection_rewards_discrimination_over_constant_positive(
    monkeypatch, scoring, expected_name,
):
    index = pd.date_range("2025-01-01", periods=10, tz="UTC")
    features = pd.DataFrame({"feature": range(10)}, index=index)
    labels = pd.Series([1] * 6 + [0] * 4, index=index)
    weights = pd.Series(1.0, index=index)
    names = ["boosting", "bagging", "random_forest"]
    candidates = {name: modeling_workflow.MyPipeline([
        ("model", DecisionTreeClassifier(max_depth=i + 1))
    ]) for i, name in enumerate(names)}
    monkeypatch.setattr(modeling_workflow, "build_candidate_classifiers", lambda **kwargs: candidates)

    def oof(estimator, *args, **kwargs):
        predictions = {
            1: [1] * 10,
            2: [1, 1, 1, 1, 0, 0, 1, 0, 0, 0],
            3: [0] * 10,
        }[estimator["model"].max_depth]
        return pd.DataFrame({
            "prediction": predictions,
            "probability": np.where(np.asarray(predictions) == 1, 0.8, 0.2),
            "fold": 0, "prediction_source": "oof",
        }, index=index)

    monkeypatch.setattr(modeling_workflow, "generate_oof_predictions", oof)
    result = run_model_selection_workflow(
        features, labels, weights, pd.Series(index, index=index),
        scoring=scoring, cv=2, pct_embargo=0.0, random_state=42, n_jobs=1,
        candidate_settings={}, parameter_grids={name: {} for name in names},
    )
    assert result.selected_name == expected_name
    assert result.comparison.loc["boosting", "f1"] == pytest.approx(0.75)
    assert result.comparison.loc["boosting", "precision"] == pytest.approx(0.6)
    assert result.comparison.loc["random_forest", "precision"] == pytest.approx(0.0)


@pytest.mark.parametrize("negative_label", [-1, 0])
@pytest.mark.parametrize("predicted", [[-1, 1, 1, -1], [1, 1, 1, 1], [-1, -1, -1, -1]])
def test_class_metrics_match_weighted_reference(negative_label, predicted):
    observed = pd.Series([negative_label, negative_label, 1, 1])
    predictions = pd.Series([negative_label if value == -1 else 1 for value in predicted])
    weights = pd.Series([3.0, 1.0, 2.0, 3.0])
    scores = score_binary_predictions(
        observed, predictions, pd.Series([0.2, 0.8, 0.8, 0.2]), weights,
        class_labels=[negative_label, 1],
    )
    for metric, scorer in [("precision", precision_score), ("recall", recall_score), ("f1", f1_score)]:
        assert scores[metric] == pytest.approx(scorer(
            observed, predictions, pos_label=1, sample_weight=weights, zero_division=0,
        ))
        if negative_label == -1:
            assert scores[f"{metric}_-1"] == pytest.approx(scorer(
                observed, predictions, pos_label=-1, sample_weight=weights, zero_division=0,
            ))
    assert set(scores) == ({"accuracy", "precision", "recall", "f1", "log_loss"}
                          | ({"precision_-1", "recall_-1", "f1_-1"} if negative_label == -1 else set()))


@pytest.mark.parametrize("constant", [0, 1])
def test_precision_learning_curve_matches_weighted_reference(constant):
    from sklearn.dummy import DummyClassifier

    index = pd.date_range("2025-01-01", periods=20, tz="UTC")
    features = pd.DataFrame({"feature": range(20)}, index=index)
    labels = pd.Series([1, 1, 1, 0, 0] * 4, index=index)
    weights = pd.Series(np.arange(1, 21), index=index)
    cv = PurgedKFold(2, pd.Series(index, index=index))
    curve = get_weighted_learning_curve(
        DummyClassifier(strategy="constant", constant=constant), features, labels,
        weights, cv, train_sizes=[1.0], class_labels=[0, 1], scoring="precision",
    )
    for split, position in [("train", 0), ("validation", 1)]:
        errors = [1 - precision_score(labels.iloc[parts[position]],
                  np.full(len(parts[position]), constant),
                  sample_weight=weights.iloc[parts[position]], zero_division=0)
                  for parts in cv.split(features)]
        assert curve[f"{split}_error_mean"].tolist() == pytest.approx([np.mean(errors)])


@pytest.mark.parametrize("model_kind", ["market", "sentiment"])
@pytest.mark.parametrize("stage", ["primary", "meta"])
def test_notebook_search_settings_and_display_order(model_kind, stage):
    import json
    from pathlib import Path
    from types import SimpleNamespace

    primary = stage == "primary"
    class_labels = [-1, 1] if primary else [0, 1]
    scores = score_binary_predictions(
        pd.Series(class_labels * 2), pd.Series(class_labels * 2),
        pd.Series([0.2, 0.8] * 2), pd.Series([1., 2.] * 2),
        class_labels=class_labels,
    )
    expected_columns = (["accuracy", "precision_-1", "recall_-1", "f1_-1",
                         "precision_1", "recall_1", "f1_1", "log_loss"] if primary
                        else ["accuracy", "precision", "recall", "f1", "log_loss"])
    displayed = []

    def select(*args, **kwargs):
        assert kwargs["scoring"] == ("neg_log_loss" if primary else "precision")
        candidates = build_candidate_classifiers(
            random_state=42, n_jobs=1, candidate_settings=kwargs["candidate_settings"],
        )
        assert candidates["random_forest"]["model"].class_weight == (
            None if primary else "balanced_subsample"
        )
        for family in ("boosting", "bagging"):
            assert candidates[family]["model"].estimator.class_weight is None
        table = pd.DataFrame([{"candidate": "boosting", "configuration": {}, **scores}])
        return SimpleNamespace(
            estimators=candidates, oof_predictions={}, selected_name="boosting",
            final_configuration={}, final_estimator=candidates["boosting"], final_oof=None,
            tuning=table, comparison=table.set_index("candidate"),
        )

    context = dict(
        X_development=None, y_development=None, w_development=None, t1_development=None,
        CLASS_LABELS=class_labels, CV_SPLITS=2, PCT_EMBARGO=0.01, RANDOM_STATE=42,
        run_model_selection_workflow=select, display=displayed.append,
        model_evaluation_columns=modeling_workflow.model_evaluation_columns,
    )
    path = Path(__file__).resolve().parents[2] / f"notebooks/modeling/{model_kind}/{stage}_model.ipynb"
    notebook = json.loads(path.read_text())
    importance_source = next("".join(c["source"]) for c in notebook["cells"]
                             if c["cell_type"] == "code" and 'results["SFI"]' in "".join(c["source"]))
    assert ('"sfi_neg_log_loss"' if primary else '"sfi_precision"') in importance_source
    for marker in ('STAGE_SCORING =', 'comparison_columns ='):
        source = next(''.join(c['source']) for c in notebook['cells']
                      if c['cell_type'] == 'code' and marker in ''.join(c['source']))
        exec(compile(source, str(path), "exec"), context)
    for table in displayed:
        assert table.columns[-len(expected_columns):].tolist() == expected_columns
        assert len(table.columns) == len(expected_columns) + (2 if "candidate" in table.columns else 1)
