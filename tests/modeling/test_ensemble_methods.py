import numpy as np
import pytest
from sklearn.base import clone
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight

from src.modeling.ensemble_methods import (
    build_bagging_classifier,
    build_boosting_classifier,
    build_random_forest_classifier,
)


def test_ensemble_factories_apply_requested_estimator_counts():
    assert build_bagging_classifier(n_estimators=3).n_estimators == 3
    assert build_random_forest_classifier(n_estimators=4).n_estimators == 4
    assert build_boosting_classifier(n_estimators=5).n_estimators == 5


def test_ensemble_factories_preserve_random_state():
    assert build_bagging_classifier(random_state=7).random_state == 7
    assert build_random_forest_classifier(random_state=7).random_state == 7
    assert build_boosting_classifier(random_state=7).random_state == 7


def test_boosting_factory_uses_entropy_based_stumps():
    classifier = build_boosting_classifier(max_depth=1)
    estimator = getattr(classifier, "estimator", None)
    if estimator is None:
        estimator = classifier.base_estimator

    assert estimator.criterion == "entropy"
    assert estimator.max_depth == 1


def test_ensemble_factories_preserve_library_leaf_defaults():
    bagging = build_bagging_classifier()
    forest = build_random_forest_classifier()
    boosting = build_boosting_classifier()

    for tree in (bagging.estimator, forest, boosting.estimator):
        assert tree.min_samples_leaf == 1
        assert tree.min_weight_fraction_leaf == 0.0


@pytest.mark.parametrize("depth", [2, 5, 10])
def test_ensemble_factories_apply_requested_depth(depth):
    bagging = build_bagging_classifier(max_depth=depth)
    forest = build_random_forest_classifier(max_depth=depth)
    boosting = build_boosting_classifier(max_depth=depth)

    for tree in (bagging.estimator, forest, boosting.estimator):
        assert tree.max_depth == depth


@pytest.mark.parametrize("negative_label", [-1, 0])
def test_boosting_uses_event_weights_without_class_balancing(negative_label):
    features = np.arange(12).reshape(-1, 1)
    labels = np.array([negative_label] * 9 + [1] * 3)
    sample_weight = np.array([1.0] * 9 + [2.0] * 3)
    classifier = build_boosting_classifier(n_estimators=1, random_state=7)

    classifier.fit(features, labels, sample_weight=sample_weight)

    root_weights = classifier.estimators_[0].tree_.value[0, 0]
    np.testing.assert_allclose(root_weights / root_weights.sum(), [3 / 5, 2 / 5])
    np.testing.assert_array_equal(classifier.predict(features), labels)


def test_random_forest_class_weight_is_configurable_without_changing_default():
    assert build_random_forest_classifier().class_weight == "balanced_subsample"
    assert build_random_forest_classifier(class_weight=None).class_weight is None


@pytest.mark.parametrize("negative_label", [-1, 0])
@pytest.mark.parametrize("weighted", [False, True])
def test_balanced_boosting_uses_class_times_event_weights(negative_label, weighted):
    features = np.arange(12).reshape(-1, 1)
    labels = np.array([negative_label] * 9 + [1] * 3)
    weights = np.array([1.] * 9 + [2.] * 3) if weighted else None
    original_weights = weights.copy() if weighted else None
    classifier = build_boosting_classifier(
        n_estimators=1, max_depth=2, random_state=7, class_weight="balanced",
    ).fit(features, labels, sample_weight=weights)
    root_weights = classifier.estimators_[0].tree_.value[0, 0]
    expected = [1 / 3, 2 / 3] if weighted else [.5, .5]
    np.testing.assert_allclose(root_weights / root_weights.sum(), expected)
    assert classifier.estimator.class_weight is None
    if weighted:
        np.testing.assert_array_equal(weights, original_weights)


def test_balanced_boosting_avoids_conflicting_tree_weights():
    features = np.zeros((10, 1))
    labels = np.array([-1] * 9 + [1])
    weights = np.array([1.] * 9 + [2.])
    conflicting = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(class_weight="balanced", max_depth=2),
        n_estimators=1,
    )
    with pytest.raises(ValueError, match="worse than random"):
        conflicting.fit(features, labels, sample_weight=weights)
    corrected = build_boosting_classifier(
        class_weight="balanced", n_estimators=1, max_depth=2,
    ).fit(features, labels, sample_weight=weights)
    np.testing.assert_array_equal(corrected.predict(features), np.ones(10))


def test_balanced_boosting_keeps_normal_failure_for_unlearnable_data():
    classifier = build_boosting_classifier(class_weight="balanced")
    with pytest.raises(ValueError, match="worse than random"):
        classifier.fit(np.zeros((4, 1)), [0, 0, 1, 1])


def test_balanced_boosting_clone_parameters_importances_and_joblib(tmp_path):
    import joblib

    classifier = build_boosting_classifier(
        n_estimators=3, max_depth=1, class_weight="balanced", random_state=7,
    )
    copied = clone(classifier).set_params(estimator__max_depth=2)
    assert copied.class_weight == "balanced"
    assert copied.estimator.max_depth == 2
    assert classifier.estimator.max_depth == 1
    X = np.arange(12).reshape(-1, 1)
    y = np.array([0] * 9 + [1] * 3)
    copied.fit(X, y, sample_weight=np.arange(1., 13.))
    path = tmp_path / "boosting.joblib"
    joblib.dump(copied, path)
    restored = joblib.load(path)
    assert restored.class_weight == "balanced"
    np.testing.assert_array_equal(restored.predict(X), copied.predict(X))
    np.testing.assert_allclose(restored.predict_proba(X), copied.predict_proba(X))
    np.testing.assert_allclose(restored.feature_importances_, [1.])


def test_native_balanced_factories_do_not_modify_supplied_event_weights(monkeypatch):
    features = np.arange(12).reshape(-1, 1)
    labels = np.array([0] * 9 + [1] * 3)
    weights = np.arange(1., 13.)
    for factory, parent in ((build_bagging_classifier, "bagging"),
                            (build_random_forest_classifier, "forest")):
        classifier = factory(n_estimators=1, class_weight="balanced", random_state=7)
        if parent == "bagging":
            assert classifier.estimator.class_weight == "balanced"
        else:
            assert classifier.class_weight == "balanced"
        # Inspect weights at the ensemble boundary, before native bootstrap/class handling.
        captured = []
        original_fit = type(classifier).fit

        def record_fit(self, X, y, sample_weight=None, **kwargs):
            captured.append(np.array(sample_weight, copy=True))
            return original_fit(self, X, y, sample_weight=sample_weight, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(type(classifier), "fit", record_fit)
            classifier.fit(features, labels, sample_weight=weights)
        np.testing.assert_array_equal(captured[0], weights)


def test_balanced_boosting_initial_weights_match_manual_adaboost():
    features = np.arange(16).reshape(-1, 1)
    labels = np.array([0, 0, 0, 1] * 4)
    weights = np.arange(1., 17.)
    balanced = build_boosting_classifier(
        n_estimators=4, max_depth=2, random_state=42, class_weight="balanced",
    ).fit(features, labels, sample_weight=weights)
    reference = build_boosting_classifier(
        n_estimators=4, max_depth=2, random_state=42,
    ).fit(features, labels, sample_weight=weights * compute_sample_weight("balanced", labels))
    np.testing.assert_allclose(balanced.predict_proba(features), reference.predict_proba(features))
    np.testing.assert_allclose(balanced.estimator_errors_, reference.estimator_errors_)
