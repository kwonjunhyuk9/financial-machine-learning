from __future__ import annotations

import numpy as np

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    RandomForestClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight


class ClassWeightedAdaBoostClassifier(AdaBoostClassifier):
    """Fit an AdaBoost classifier with class weights."""

    def __init__(
        self,
        estimator=None,
        *,
        n_estimators: int = 50,
        learning_rate: float = 1.0,
        random_state: int | None = None,
        class_weight: str | None = None,
    ):
        """Initialize the class-weighted boosting classifier.

        Args:
            estimator: Base classifier.
            n_estimators: Number of estimators.
            learning_rate: Boosting learning rate.
            random_state: Random seed.
            class_weight: Class-weighting configuration.
        """
        self.class_weight = class_weight
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            random_state=random_state,
        )

    def fit(self, X, y, sample_weight=None):
        """Fit the classifier with class and observation weights.

        Args:
            X: Training features.
            y: Training labels used to calculate class frequencies.
            sample_weight: Training observation weights.

        Returns:
            Fitted classifier.
        """
        if self.class_weight is not None:
            class_weights = compute_sample_weight(self.class_weight, y)
            sample_weight = (class_weights if sample_weight is None else
                             np.asarray(sample_weight) * class_weights)
        return super().fit(X, y, sample_weight=sample_weight)


def build_bagging_classifier(
    n_estimators: int = 1000,
    max_samples: float = 1.0,
    max_features: float = 1.0,
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int | None = None,
    class_weight: dict | str | None = None,
) -> BaggingClassifier:
    """Build a bagging classifier with entropy-based decision trees.

    Args:
        n_estimators: Number of trees in the ensemble.
        max_samples: Fraction of samples drawn for each base estimator.
        max_features: Fraction of features drawn for each base estimator.
        max_depth: Maximum depth of each decision-tree base estimator.
        n_jobs: Number of parallel workers.
        random_state: Random seed.
        class_weight: Class-weighting configuration.

    Returns:
        Configured bagging classifier.
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        class_weight=class_weight,
        max_depth=max_depth,
        random_state=random_state
    )

    return BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        oob_score=False,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_random_forest_classifier(
    n_estimators: int = 1000,
    max_features: str | int | float | None = "sqrt",
    max_depth: int | None = None,
    n_jobs: int = -1,
    random_state: int | None = None,
    max_samples: float | None = None,
    class_weight: dict | str | None = "balanced_subsample",
) -> RandomForestClassifier:
    """Build an entropy-based random forest with configurable class weights.

    Args:
        n_estimators: Number of trees in the forest.
        max_features: Feature-subsampling rule for each split.
        max_depth: Maximum depth of each tree.
        n_jobs: Number of parallel workers.
        random_state: Random seed.
        max_samples: Fraction of samples drawn for each tree.
        class_weight: Class-weighting configuration.

    Returns:
        Configured random forest classifier.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        criterion="entropy",
        bootstrap=True,
        max_samples=max_samples,
        class_weight=class_weight,
        max_features=max_features,
        max_depth=max_depth,
        n_jobs=n_jobs,
        random_state=random_state
    )


def build_boosting_classifier(
    n_estimators: int = 100,
    learning_rate: float = 1.0,
    max_depth: int | None = 1,
    random_state: int | None = None,
    class_weight: str | None = None,
) -> AdaBoostClassifier:
    """Build a boosting classifier with entropy-based decision trees.

    Args:
        n_estimators: Number of boosting rounds.
        learning_rate: Shrinkage applied to each estimator.
        max_depth: Maximum depth of the decision-tree base estimator.
        random_state: Random seed.
        class_weight: Class-weighting configuration.

    Returns:
        Configured boosting classifier.
    """
    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_depth=max_depth,
        random_state=random_state
    )

    boosting_class = ClassWeightedAdaBoostClassifier if class_weight is not None else AdaBoostClassifier
    return boosting_class(
        estimator=clf,
        **({"class_weight": class_weight} if class_weight is not None else {}),
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        random_state=random_state
    )
