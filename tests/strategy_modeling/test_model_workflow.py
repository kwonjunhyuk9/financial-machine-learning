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

from src.strategy_modeling.cross_validation import PurgedKFold
from src.strategy_modeling.model_workflow import (
    build_candidate_classifiers,
    build_meta_training_frame,
    candidate_parameter_grids,
    compute_strategy_returns,
    generate_oof_predictions,
    get_primary_feature_columns,
    get_weighted_learning_curve,
    probability_bet_size,
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
            "mean_sentiment_score": np.linspace(-1.0, 1.0, num_rows),
            "fractionally_differenced_log_close": np.linspace(0.0, 0.5, num_rows),
            "Relative Strength Index": np.linspace(40.0, 60.0, num_rows),
        }
    )


def test_primary_feature_columns_exclude_outcomes_and_identifiers():
    columns = get_primary_feature_columns(_events())

    assert columns == [
        "mean_sentiment_score",
        "fractionally_differenced_log_close",
        "Relative Strength Index",
    ]


def test_candidate_classifiers_use_required_model_families():
    candidates = build_candidate_classifiers(random_state=42, n_jobs=1)

    assert set(candidates) == {
        "bagging",
        "random_forest",
        "adaboost",
        "gradient_boosting",
    }
    assert all(
        candidate.steps == [("model", candidate["model"])]
        for candidate in candidates.values()
    )


def test_candidate_parameter_grids_cover_all_tree_families():
    grids = candidate_parameter_grids()

    assert set(grids) == {
        "bagging",
        "random_forest",
        "adaboost",
        "gradient_boosting",
    }
    expected_learning_rates = [
        {"model__learning_rate": 0.03},
        {"model__learning_rate": 0.10},
        {"model__learning_rate": 0.30},
    ]
    assert grids["adaboost"] == expected_learning_rates
    assert grids["gradient_boosting"] == expected_learning_rates


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
    index = pd.date_range("2025-01-01", periods=40, freq="D")
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


def test_meta_training_frame_requires_primary_oof_predictions():
    events = _events().set_index("event_start")
    primary_oof = pd.DataFrame(
        {
            "prediction": events["direction_label"],
            "probability": 0.75,
            "prediction_source": "oof",
        },
        index=events.index,
    )

    meta = build_meta_training_frame(events, primary_oof)

    expected = (meta["primary_side"] * meta["raw_return"] > 0).astype("int8")
    pd.testing.assert_series_equal(meta["meta_label"], expected, check_names=False)

    primary_oof.loc[primary_oof.index[0], "prediction_source"] = "in_sample"
    with pytest.raises(ValueError, match="OOF"):
        build_meta_training_frame(events, primary_oof)


def test_probability_bet_size_is_bounded_and_respects_pass_decision():
    probability = pd.Series([0.50, 0.60, 0.90])
    action = pd.Series([1, 0, 1])

    sizes = probability_bet_size(probability, action, step_size=0.1)

    assert sizes.between(0.0, 1.0).all()
    assert sizes.iloc[0] == 0.0
    assert sizes.iloc[1] == 0.0
    assert sizes.iloc[2] > 0.0


def test_strategy_returns_default_to_zero_execution_costs():
    predictions = pd.DataFrame(
        {
            "raw_return": [0.02, -0.01],
            "primary_side": [1, -1],
            "meta_action": [1, 0],
            "bet_size": [0.5, 0.8],
        }
    )

    returns = compute_strategy_returns(predictions)

    assert returns.loc[0, "primary_only_gross_return"] == pytest.approx(0.02)
    assert returns.loc[0, "primary_only_total_cost"] == 0.0
    assert returns.loc[0, "primary_only_net_return"] == pytest.approx(0.02)
    assert returns.loc[0, "meta_filtered_gross_return"] == pytest.approx(0.01)
    assert returns.loc[0, "meta_filtered_total_cost"] == 0.0
    assert returns.loc[0, "meta_filtered_net_return"] == pytest.approx(0.01)
    assert returns.loc[1, "meta_filtered_net_return"] == 0.0
