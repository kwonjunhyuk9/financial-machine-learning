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

import src.modeling.model_workflow as model_workflow
from src.modeling.purged_validation import PurgedKFold
from src.modeling.model_workflow import (
    build_candidate_classifiers,
    build_meta_model_frame,
    build_model_evaluation_table,
    build_primary_model_frame,
    candidate_parameter_grids,
    compute_stage_importance,
    generate_oof_predictions,
    get_meta_feature_columns,
    get_primary_feature_columns,
    get_weighted_learning_curve,
    plot_learning_curves,
    run_model_selection_workflow,
    score_binary_predictions,
)


def _events(num_rows: int = 10) -> pd.DataFrame:
    starts = pd.date_range("2025-01-02", periods=num_rows, freq="D", tz="UTC")
    ends = starts + pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "event_start": starts,
            "symbol": "AAPL",
            "event_end": ends,
            "vertical_barrier": ends + pd.Timedelta(hours=1),
            "target_return": 0.01,
            "raw_return": np.where(np.arange(num_rows) % 2 == 0, 0.02, -0.01),
            "direction_label": np.where(np.arange(num_rows) % 2 == 0, 1, -1),
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
    candidates = build_candidate_classifiers(random_state=42, n_jobs=1)

    assert list(candidates) == ["boosting", "bagging", "random_forest"]
    assert all(
        candidate.steps == [("model", candidate["model"])]
        for candidate in candidates.values()
    )


def test_candidate_parameter_grids_cover_all_tree_families():
    grids = candidate_parameter_grids()

    assert list(grids) == ["boosting", "bagging", "random_forest"]
    expected_learning_rates = {"model__learning_rate": [0.03, 0.10, 0.30]}
    assert grids["boosting"] == expected_learning_rates


@pytest.mark.parametrize(
    ("label_values", "scoring", "expected_name"),
    [
        ([-1, 1], "neg_log_loss", "boosting"),
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
        name: model_workflow.MyPipeline([
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
        0.4: {"log_loss": 0.1, "f1": 0.9},
    }

    monkeypatch.setattr(
        model_workflow,
        "build_candidate_classifiers",
        lambda **kwargs: candidates,
    )
    monkeypatch.setattr(
        model_workflow,
        "candidate_parameter_grids",
        lambda: {name: {"model__max_depth": [1, 4]} for name in candidates},
    )
    monkeypatch.setattr(
        model_workflow,
        "fit_classifier_with_hyperparameter_search",
        lambda *args, **kwargs: model_workflow.MyPipeline([
            ("model", DecisionTreeClassifier(max_depth=4, random_state=42))
        ]),
    )

    def fake_oof(estimator, features, labels, sample_weight, cv, positive_label):
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
            "precision": 1.0,
            "recall": 1.0,
        }

    monkeypatch.setattr(model_workflow, "generate_oof_predictions", fake_oof)
    monkeypatch.setattr(model_workflow, "score_binary_predictions", fake_scores)

    result = run_model_selection_workflow(
        features,
        labels,
        weights,
        information_sets,
        scoring=scoring,
        cv=2,
        random_state=42,
        n_jobs=1,
    )

    assert result.selected_name == expected_name
    assert result.final_configuration == {"model__max_depth": 4}
    assert result.final_oof["prediction_source"].eq("oof").all()
    assert result.tuning.loc[0, "candidate"] == expected_name


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
        name="direction_label",
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
    assert table.columns.tolist() == [
        "accuracy", "precision", "recall", "f1", "log_loss",
    ]
    assert table.loc["Test Model"].to_dict() == pytest.approx(expected)


def test_plot_learning_curves_returns_each_estimator_result(monkeypatch):
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

    monkeypatch.setattr(model_workflow, "get_weighted_learning_curve", fake_learning_curve)
    monkeypatch.setattr(model_workflow.plt, "show", lambda: None)

    results = plot_learning_curves(
        {"first": object(), "second": object()},
        "Learning Curves",
        features=pd.DataFrame(),
        labels=pd.Series(dtype=int),
        sample_weight=pd.Series(dtype=float),
        cv=object(),
        train_sizes=[1.0],
        class_labels=[0, 1],
    )
    model_workflow.plt.close("all")

    assert list(results) == ["first", "second"]
    assert len(calls) == 2
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
            "prediction": events["direction_label"],
            "probability": 0.75,
            "prediction_source": "oof",
        },
        index=events.index,
    )

    meta = build_meta_model_frame(events, primary_oof)

    expected = (meta["primary_side"] * meta["raw_return"] > 0).astype("int8")
    pd.testing.assert_series_equal(meta["meta_label"], expected, check_names=False)
    assert get_meta_feature_columns(meta) == [
        *MODEL_FEATURES,
        "primary_side",
        "primary_confidence",
    ]

    primary_oof.loc[primary_oof.index[0], "prediction_source"] = "in_sample"
    with pytest.raises(ValueError, match="OOF"):
        build_meta_model_frame(events, primary_oof)


def test_meta_feature_columns_require_generated_features():
    with pytest.raises(ValueError, match="required meta features"):
        get_meta_feature_columns(_events())
