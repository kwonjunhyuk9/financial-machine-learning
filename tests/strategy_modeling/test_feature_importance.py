import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier

from src.strategy_modeling.feature_importance import (
    get_estimator_feature_importance,
    get_feature_importance,
    get_mean_decrease_accuracy,
)
from src.strategy_modeling.model_workflow import build_candidate_classifiers


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
    dates = pd.date_range("2000-01-01", periods=n_samples, freq="D")
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


@pytest.mark.parametrize("method", ["MDI", "MDA", "SFI"])
def test_feature_importance_methods_return_stable_shapes(method):
    features, container = _make_test_data(random_state=14)

    importance, oob_score, oos_score = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method=method,
        scoring="neg_log_loss",
        random_state=14,
    )

    assert importance.index.tolist() == features.columns.tolist()
    assert importance.columns.tolist() == ["mean", "std"]
    assert np.isfinite(
        importance.to_numpy()[~np.isnan(importance.to_numpy())]
    ).all()
    assert np.isfinite(oob_score)
    assert np.isfinite(oos_score)


@pytest.mark.parametrize("method", ["MDI", "MDA", "SFI"])
def test_feature_importance_methods_support_f1_for_meta_labels(method):
    features, container = _make_test_data(random_state=15)

    importance, oob_score, oos_score = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method=method,
        scoring="f1",
        random_state=15,
    )

    assert importance.index.tolist() == features.columns.tolist()
    assert np.isfinite(
        importance.to_numpy()[~np.isnan(importance.to_numpy())]
    ).all()
    assert np.isfinite(oob_score)
    assert np.isfinite(oos_score)


def test_feature_importance_forwards_seed_to_mean_decrease_accuracy():
    features, container = _make_test_data(random_state=13)

    first = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method="MDA",
        random_state=13,
    )
    second = get_feature_importance(
        features,
        container,
        n_estimators=20,
        cv=2,
        max_samples=0.8,
        num_threads=1,
        method="MDA",
        random_state=13,
    )

    pd.testing.assert_frame_equal(first[0], second[0])
    assert first[1:] == pytest.approx(second[1:])


def test_feature_importance_rejects_ignored_compatibility_options():
    features, container = _make_test_data(n_samples=30, random_state=17)

    with pytest.raises(TypeError, match="unexpected keyword"):
        get_feature_importance(features, container, ignored_option=True)


@pytest.mark.parametrize(
    "candidate_name",
    ["bagging", "random_forest", "adaboost", "gradient_boosting"],
)
def test_selected_tree_ensembles_support_mdi(candidate_name):
    features, container = _make_test_data(random_state=18)
    estimator = build_candidate_classifiers(
        random_state=18,
        n_jobs=1,
    )[candidate_name].set_params(model__n_estimators=5)

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
