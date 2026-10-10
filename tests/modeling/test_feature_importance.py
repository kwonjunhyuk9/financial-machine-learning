from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from src.modeling.feature_importance import (
    get_estimator_feature_importance,
    get_mean_decrease_accuracy,
    get_mean_decrease_impurity,
)
from src.modeling.model_workflow import build_candidate_classifiers
from src.modeling.purged_validation import PurgedKFold, _purge_train_indices


CANDIDATE_SETTINGS = {
    "boosting": {"n_estimators": 100, "learning_rate": .10},
    "bagging": {"n_estimators": 120, "max_samples": .80},
    "random_forest": {"n_estimators": 120},
}


def _make_test_data(
    n_samples: int = 40,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    values, labels = make_classification(
        n_samples=n_samples,
        n_features=4,
        n_informative=2,
        n_redundant=1,
        n_repeated=0,
        n_classes=2,
        shuffle=False,
        random_state=random_state,
    )
    dates = pd.date_range("2000-01-01", periods=n_samples, freq="D", tz="UTC")
    features = pd.DataFrame(
        values,
        index=dates,
        columns=["I_0", "I_1", "R_0", "N_0"],
    )
    t1 = pd.Series([*dates[1:], dates[-1]], index=dates)
    container = pd.DataFrame(
        {"bin": labels, "w": 1.0, "t1": t1},
        index=dates,
    )
    return features, container


def test_mean_decrease_accuracy_is_reproducible_with_a_seed():
    features, container = _make_test_data(n_samples=30, random_state=11)
    kwargs = {
        "clf": DecisionTreeClassifier(random_state=11),
        "X": features,
        "y": container["bin"],
        "cv": 3,
        "sample_weight": container["w"],
        "t1": container["t1"],
        "pct_embargo": 0.0,
        "scoring": "accuracy",
        "random_state": 11,
    }

    first_importance, first_score = get_mean_decrease_accuracy(**kwargs)
    second_importance, second_score = get_mean_decrease_accuracy(**kwargs)

    pd.testing.assert_frame_equal(first_importance, second_importance)
    assert first_score == pytest.approx(second_score)
    assert np.isfinite(first_score)


@pytest.mark.parametrize("negative_label,scoring", [(-1, "neg_log_loss"), (0, "f1")])
def test_sfi_reuses_purged_splits_without_changing_results(
    monkeypatch, negative_label, scoring
):
    features, container = _make_test_data(random_state=19)
    calls = []

    def counted_purge(*args, **kwargs):
        calls.append(1)
        return _purge_train_indices(*args, **kwargs)

    monkeypatch.setattr(
        "src.modeling.purged_validation._purge_train_indices", counted_purge
    )
    kwargs = dict(
        estimator=DecisionTreeClassifier(random_state=19), features=features,
        labels=container["bin"].replace({0: negative_label}),
        sample_weight=container["w"], t1=container["t1"],
        method="SFI", scoring=scoring, cv=3, pct_embargo=.05,
    )
    cached_importance, cached_score = get_estimator_feature_importance(**kwargs)
    assert len(calls) == 3

    class UncachedPurgedKFold(PurgedKFold):
        def split(self, *args, **kwargs):
            self._cached_splits = None
            yield from super().split(*args, **kwargs)

    monkeypatch.setattr(
        "src.modeling.feature_importance.PurgedKFold", UncachedPurgedKFold
    )
    calls.clear()
    original_importance, original_score = get_estimator_feature_importance(**kwargs)
    assert len(calls) == 3 * (len(features.columns) + 1)
    pd.testing.assert_frame_equal(cached_importance, original_importance)
    assert cached_score == original_score


def test_mean_decrease_accuracy_rejects_unsupported_scoring():
    with pytest.raises(ValueError, match="neg_log_loss.*accuracy.*f1"):
        get_mean_decrease_accuracy(
            DecisionTreeClassifier(),
            X=pd.DataFrame(),
            y=pd.Series(dtype=int),
            cv=2,
            sample_weight=pd.Series(dtype=float),
            t1=pd.Series(dtype="datetime64[ns]"),
            pct_embargo=0.0,
            scoring="precision",
        )


def test_mean_decrease_accuracy_fits_one_cloned_estimator_per_fold(monkeypatch):
    features, container = _make_test_data(n_samples=30, random_state=12)
    original_fit = DecisionTreeClassifier.fit
    fit_calls = 0

    def counting_fit(estimator, *args, **kwargs):
        nonlocal fit_calls
        fit_calls += 1
        return original_fit(estimator, *args, **kwargs)

    monkeypatch.setattr(DecisionTreeClassifier, "fit", counting_fit)
    classifier = DecisionTreeClassifier(random_state=12)

    get_mean_decrease_accuracy(
        classifier,
        X=features,
        y=container["bin"],
        cv=3,
        sample_weight=container["w"],
        t1=container["t1"],
        pct_embargo=0.0,
        scoring="neg_log_loss",
        random_state=12,
    )

    assert fit_calls == 3
    assert not hasattr(classifier, "classes_")


@pytest.mark.parametrize(
    "candidate_name",
    ["boosting", "bagging", "random_forest"],
)
def test_selected_tree_ensembles_support_mdi(candidate_name):
    features, container = _make_test_data(random_state=18)
    estimator = build_candidate_classifiers(
        random_state=18,
        n_jobs=1,
        candidate_settings=CANDIDATE_SETTINGS,
    )[candidate_name].set_params(model__n_estimators=5)
    if candidate_name == "bagging":
        estimator.set_params(model__max_features=0.5)

    importance, oos_score = get_estimator_feature_importance(
        estimator,
        features,
        container["bin"],
        container["w"],
        container["t1"],
        method="MDI",
        scoring="neg_log_loss",
        cv=2,
        pct_embargo=0.0,
        random_state=18,
    )

    assert importance.index.tolist() == features.columns.tolist()
    assert importance.columns.tolist() == ["mean", "std"]
    assert importance["mean"].sum() == pytest.approx(1.0)
    assert np.isfinite(importance).all().all()
    assert np.isfinite(oos_score)


def test_mdi_maps_tree_features_to_original_columns():
    fitted = SimpleNamespace(
        estimators_=[
            SimpleNamespace(feature_importances_=np.array([0.2, 0.8])),
            SimpleNamespace(feature_importances_=np.array([0.6, 0.4])),
        ],
        estimators_features_=[np.array([2, 0]), np.array([1, 2])],
    )

    importance = get_mean_decrease_impurity(fitted, ["a", "b", "c"])

    np.testing.assert_allclose(importance["mean"], [0.4, 0.3, 0.3])


def test_mdi_sums_repeated_feature_selections():
    fitted = SimpleNamespace(
        estimators_=[SimpleNamespace(feature_importances_=np.array([0.2, 0.8]))],
        estimators_features_=[np.array([1, 1])],
    )

    importance = get_mean_decrease_impurity(fitted, ["a", "b"])

    np.testing.assert_allclose(importance["mean"], [0.0, 1.0])


def test_mdi_returns_zeros_when_trees_cannot_split():
    features = np.zeros((30, 3))
    labels = np.tile([0, 1], 15)
    fitted = RandomForestClassifier(n_estimators=3, random_state=1).fit(
        features, labels
    )

    with np.errstate(divide="raise", invalid="raise"):
        importance = get_mean_decrease_impurity(fitted, ["a", "b", "c"])

    np.testing.assert_array_equal(importance.to_numpy(), np.zeros((3, 2)))


def test_mdi_does_not_mutate_adaboost_weights():
    features, labels = make_classification(
        n_samples=100, n_features=4, n_informative=2, n_redundant=0,
        flip_y=0.15, random_state=1,
    )
    fitted = AdaBoostClassifier(n_estimators=5, random_state=1).fit(features, labels)
    original_weights = fitted.estimator_weights_.copy()

    get_mean_decrease_impurity(fitted, ["a", "b", "c", "d"])

    np.testing.assert_array_equal(fitted.estimator_weights_, original_weights)


@pytest.mark.parametrize("scoring", ["accuracy", "f1"])
def test_mda_returns_zero_for_unused_feature_with_perfect_predictions(scoring):
    dates = pd.date_range("2000-01-01", periods=30, tz="UTC")
    labels = pd.Series(np.tile([0, 1], 15), index=dates)
    features = pd.DataFrame({"signal": labels, "unused": 0.0}, index=dates)

    importance, score = get_mean_decrease_accuracy(
        DecisionTreeClassifier(max_depth=1, random_state=1),
        features, labels, 3, pd.Series(1.0, index=dates),
        pd.Series(dates, index=dates), 0.0, scoring=scoring, random_state=1,
    )

    assert score == pytest.approx(1.0)
    assert importance.loc["unused", "mean"] == pytest.approx(0.0)
    assert importance.loc["unused", "std"] == pytest.approx(0.0)
    assert np.isfinite(importance).all().all()


@pytest.mark.parametrize(
    "scoring,baseline,perfect", [("f1", 0.5, 1.0), ("neg_log_loss", -0.3, 0.0)]
)
def test_mda_preserves_negative_difference_at_zero_denominator(
    monkeypatch, scoring, baseline, perfect
):
    features, container = _make_test_data(n_samples=30)
    scores = iter([baseline, perfect] * 3)
    monkeypatch.setattr(
        "src.modeling.feature_importance._select_feature_importance_score",
        lambda *args: next(scores),
    )

    importance, score = get_mean_decrease_accuracy(
        DecisionTreeClassifier(random_state=1), features[["I_0"]],
        container["bin"], 3, container["w"], container["t1"], 0.0,
        scoring=scoring, random_state=1,
    )

    assert score == pytest.approx(baseline)
    assert importance.loc["I_0", "mean"] == pytest.approx(baseline - perfect)
    assert np.isfinite(importance).all().all()


@pytest.mark.parametrize("method", ["MDA", "SFI"])
@pytest.mark.parametrize(
    ("negative_label", "scoring"),
    [(-1, "neg_log_loss"), (0, "f1")],
)
def test_selected_estimator_importance_supports_project_label_spaces(
    method,
    negative_label,
    scoring,
):
    features, container = _make_test_data(random_state=19)
    labels = container["bin"].replace({0: negative_label})
    estimator = build_candidate_classifiers(
        random_state=19,
        n_jobs=1,
        candidate_settings=CANDIDATE_SETTINGS,
    )["bagging"].set_params(model__n_estimators=5)

    importance, oos_score = get_estimator_feature_importance(
        estimator,
        features,
        labels,
        container["w"],
        container["t1"],
        method=method,
        scoring=scoring,
        cv=2,
        pct_embargo=0.0,
        random_state=19,
    )

    assert importance.index.tolist() == features.columns.tolist()
    assert np.isfinite(
        importance.to_numpy()[~np.isnan(importance.to_numpy())]
    ).all()
    assert np.isfinite(oos_score)
