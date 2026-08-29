from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.pipeline import Pipeline

from src.strategy_modeling.cross_validation import PurgedKFold
from src.strategy_modeling.model_workflow import (
    generate_oof_predictions,
    score_binary_predictions,
)


def _unwrap_estimator(estimator: BaseEstimator) -> BaseEstimator:
    if isinstance(estimator, Pipeline):
        return estimator.steps[-1][1]
    return estimator


def get_mean_decrease_impurity(
    fit: BaseEstimator,
    feat_names: Sequence[str],
) -> pd.DataFrame:
    """Compute mean decrease impurity feature importances.

    Args:
        fit: Fitted tree-ensemble estimator or fitted pipeline ending in one.
        feat_names: Feature names aligned with the estimator input.

    Returns:
        A frame with mean and standard-error importance estimates.
    """
    fitted = _unwrap_estimator(fit)
    if not hasattr(fitted, "estimators_"):
        raise ValueError("estimator must expose fitted tree estimators")

    trees = np.asarray(fitted.estimators_, dtype=object).reshape(-1)
    values = np.vstack([tree.feature_importances_ for tree in trees])
    weights = getattr(fitted, "estimator_weights_", None)
    if weights is None:
        mean = values.mean(axis=0)
        std = values.std(axis=0) * values.shape[0] ** -0.5
    else:
        normalized_weights = np.asarray(weights, dtype="float64")[:len(trees)]
        normalized_weights /= normalized_weights.sum()
        mean = np.average(values, axis=0, weights=normalized_weights)
        variance = np.average(
            (values - mean) ** 2,
            axis=0,
            weights=normalized_weights,
        )
        effective_estimators = 1.0 / np.square(normalized_weights).sum()
        std = np.sqrt(variance / effective_estimators)

    total = mean.sum()
    imp = pd.DataFrame(
        {"mean": mean / total, "std": std / total},
        index=feat_names,
    )

    return imp


def _get_class_labels(labels: pd.Series) -> list[int]:
    return np.sort(pd.unique(labels)).tolist()


def _predict_binary(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    positive_label: int = 1,
) -> pd.DataFrame:
    class_positions = np.flatnonzero(estimator.classes_ == positive_label)
    if class_positions.size != 1:
        raise ValueError(f"positive_label {positive_label!r} is absent from a fold")

    return pd.DataFrame(
        {
            "prediction": estimator.predict(features),
            "probability": estimator.predict_proba(features)[:, class_positions[0]],
        },
        index=features.index,
    )


def _select_feature_importance_score(
    scores: dict[str, float],
    scoring: str,
) -> float:
    if scoring == "neg_log_loss":
        return -scores["log_loss"]
    if scoring == "accuracy":
        return scores["accuracy"]
    if scoring == "f1":
        return scores["f1"]
    raise ValueError("scoring must be 'neg_log_loss', 'accuracy', or 'f1'.")


def _score_oof_folds(
    predictions: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    scoring: str,
) -> np.ndarray:
    class_labels = _get_class_labels(labels)
    fold_scores = []

    for fold in sorted(predictions["fold"].unique()):
        fold_predictions = predictions[predictions["fold"].eq(fold)]
        fold_index = fold_predictions.index
        scores = score_binary_predictions(
            labels.loc[fold_index],
            fold_predictions["prediction"],
            fold_predictions["probability"],
            sample_weight.loc[fold_index],
            class_labels=class_labels,
            positive_label=1,
        )
        fold_scores.append(_select_feature_importance_score(scores, scoring))

    return np.asarray(fold_scores)


def get_mean_decrease_accuracy(
    clf: BaseEstimator,
    X: pd.DataFrame,
    y: pd.Series,
    cv: int,
    sample_weight: pd.Series,
    t1: pd.Series,
    pct_embargo: float,
    scoring: str = "neg_log_loss",
    random_state: int | np.random.Generator | None = None,
) -> tuple[pd.DataFrame, float]:
    """Compute mean decrease accuracy feature importances.

    Args:
        clf: Classifier to evaluate.
        X: Feature matrix.
        y: Target values.
        cv: Number of cross-validation folds.
        sample_weight: Sample weights aligned with ``X``.
        t1: Label end times for purged cross-validation.
        pct_embargo: Embargo fraction applied to each fold.
        scoring: Scoring metric, one of ``"neg_log_loss"``, ``"accuracy"``,
            or ``"f1"``.
        random_state: Seed or generator used for feature permutations.

    Returns:
        A tuple of the importance frame and the mean baseline score.

    Raises:
        ValueError: If ``scoring`` is not supported.
    """
    if scoring not in ["neg_log_loss", "accuracy", "f1"]:
        raise ValueError("scoring must be 'neg_log_loss', 'accuracy', or 'f1'.")

    rng = np.random.default_rng(random_state)
    class_labels = _get_class_labels(y)
    cv_gen = PurgedKFold(
        n_splits=cv,
        t1=t1,
        pct_embargo=pct_embargo
    )

    scr0 = pd.Series(dtype="float64")
    scr1 = pd.DataFrame(columns=X.columns, dtype="float64")

    for i, (train, test) in enumerate(cv_gen.split(X=X)):
        X0 = X.iloc[train, :]
        y0 = y.iloc[train]
        w0 = sample_weight.iloc[train]

        X1 = X.iloc[test, :]
        y1 = y.iloc[test]
        w1 = sample_weight.iloc[test]

        fit = clone(clf).fit(
            X=X0,
            y=y0,
            sample_weight=w0.values
        )
        baseline_predictions = _predict_binary(fit, X1)
        baseline_scores = score_binary_predictions(
            y1,
            baseline_predictions["prediction"],
            baseline_predictions["probability"],
            w1,
            class_labels=class_labels,
            positive_label=1,
        )
        scr0.loc[i] = _select_feature_importance_score(
            baseline_scores,
            scoring,
        )

        for j in X.columns:
            X1_ = X1.copy(deep=True)
            X1_[j] = rng.permutation(X1_[j].to_numpy())

            permuted_predictions = _predict_binary(fit, X1_)
            permuted_scores = score_binary_predictions(
                y1,
                permuted_predictions["prediction"],
                permuted_predictions["probability"],
                w1,
                class_labels=class_labels,
                positive_label=1,
            )
            scr1.loc[i, j] = _select_feature_importance_score(
                permuted_scores,
                scoring,
            )

    imp = (-scr1).add(scr0, axis=0)

    if scoring == "neg_log_loss":
        imp = imp / -scr1
    else:
        imp = imp / (1.0 - scr1)

    imp = pd.concat(
        {
            "mean": imp.mean(),
            "std": imp.std() * imp.shape[0] ** -0.5
        },
        axis=1
    )

    return imp, scr0.mean()


def get_single_feature_importance(
    feat_names: Sequence[str],
    clf: BaseEstimator,
    trns_x: pd.DataFrame,
    cont: pd.DataFrame,
    scoring: str,
    cv_gen: PurgedKFold,
) -> pd.DataFrame:
    """Compute single-feature importances by isolated cross-validation.

    Args:
        feat_names: Feature names to score individually.
        clf: Classifier to evaluate.
        trns_x: Training feature matrix.
        cont: Container with ``bin`` labels and ``w`` sample weights.
        scoring: Scoring metric, one of ``"neg_log_loss"``, ``"accuracy"``,
            or ``"f1"``.
        cv_gen: Cross-validation generator.

    Returns:
        A frame with mean and standard-error scores for each feature.
    """
    imp = pd.DataFrame(columns=["mean", "std"], dtype="float64")

    for feat_name in feat_names:
        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x[[feat_name]],
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        fold_scores = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        )

        imp.loc[feat_name, "mean"] = fold_scores.mean()
        imp.loc[feat_name, "std"] = (
            fold_scores.std() * fold_scores.shape[0] ** -0.5
        )

    return imp


def get_estimator_feature_importance(
    estimator: BaseEstimator,
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    t1: pd.Series,
    *,
    method: str,
    scoring: str,
    cv: int = 5,
    pct_embargo: float = 0.01,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, float]:
    """Measure a selected estimator with MDI, MDA, or SFI.

    Args:
        estimator: Selected tree-ensemble estimator or pipeline.
        features: Development feature matrix.
        labels: Binary labels aligned with ``features``.
        sample_weight: Sample weights aligned with ``features``.
        t1: Label end times used by purged cross-validation.
        method: Importance method, one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
        scoring: ``"neg_log_loss"``, ``"accuracy"``, or ``"f1"``.
        cv: Number of purged folds.
        pct_embargo: Embargo fraction applied to each fold.
        random_state: Seed used for feature permutations.

    Returns:
        The feature-importance frame and the selected estimator's mean
        out-of-fold score under ``scoring``.

    Raises:
        ValueError: If ``method`` is unsupported.
    """
    if method not in {"MDI", "MDA", "SFI"}:
        raise ValueError("method must be one of: 'MDI', 'MDA', 'SFI'")

    container = pd.DataFrame(
        {"bin": labels, "w": sample_weight, "t1": t1},
        index=features.index,
    )
    cv_gen = PurgedKFold(
        n_splits=cv,
        t1=t1,
        pct_embargo=pct_embargo,
    )

    if method == "MDI":
        fitted = clone(estimator).fit(
            features,
            labels,
            sample_weight=sample_weight.to_numpy(),
        )
        importance = get_mean_decrease_impurity(
            fitted,
            feat_names=features.columns,
        )
    elif method == "MDA":
        return get_mean_decrease_accuracy(
            estimator,
            X=features,
            y=labels,
            cv=cv,
            sample_weight=sample_weight,
            t1=t1,
            pct_embargo=pct_embargo,
            scoring=scoring,
            random_state=random_state,
        )
    else:
        importance = get_single_feature_importance(
            feat_names=features.columns,
            clf=estimator,
            trns_x=features,
            cont=container,
            scoring=scoring,
            cv_gen=cv_gen,
        )

    predictions = generate_oof_predictions(
        estimator=estimator,
        features=features,
        labels=labels,
        sample_weight=sample_weight,
        cv=cv_gen,
        positive_label=1,
    )
    oos = _score_oof_folds(
        predictions,
        labels=labels,
        sample_weight=sample_weight,
        scoring=scoring,
    ).mean()

    return importance, float(oos)


def get_feature_importance(
    trns_x: pd.DataFrame,
    cont: pd.DataFrame,
    n_estimators: int = 1000,
    cv: int = 10,
    max_samples: float = 1.0,
    num_threads: int = 24,
    pct_embargo: float = 0.0,
    scoring: str = "accuracy",
    method: str = "SFI",
    min_w_leaf: float = 0.0,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, float, float]:
    """Estimate feature importance with MDI, MDA, or SFI.

    Args:
        trns_x: Training feature matrix.
        cont: Container with ``bin``, ``w``, and ``t1`` fields.
        n_estimators: Number of trees in the bagging ensemble.
        cv: Number of cross-validation folds.
        max_samples: Fraction of samples drawn for each bagging estimator.
        num_threads: Number of parallel workers.
        pct_embargo: Embargo fraction applied to each fold.
        scoring: Scoring metric, one of ``"neg_log_loss"``, ``"accuracy"``,
            or ``"f1"``.
        method: Importance method, one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
        min_w_leaf: Minimum weighted fraction required at a leaf.
        random_state: Seed used by the estimator and feature permutations.

    Returns:
        A tuple of the importance frame, out-of-bag score, and out-of-sample score.

    Raises:
        ValueError: If ``method`` is not one of ``"MDI"``, ``"MDA"``, or ``"SFI"``.
    """
    n_jobs = -1 if num_threads > 1 else 1

    clf = DecisionTreeClassifier(
        criterion="entropy",
        max_features=1,
        class_weight="balanced",
        min_weight_fraction_leaf=min_w_leaf,
        random_state=random_state
    )

    clf = BaggingClassifier(
        estimator=clf,
        n_estimators=n_estimators,
        max_features=1.0,
        max_samples=max_samples,
        oob_score=True,
        n_jobs=n_jobs,
        random_state=random_state
    )

    fit = clf.fit(
        X=trns_x,
        y=cont["bin"],
        sample_weight=cont["w"].values
    )

    oob = fit.oob_score_

    if method == "MDI":
        imp = get_mean_decrease_impurity(
            fit,
            feat_names=trns_x.columns
        )

        cv_gen = PurgedKFold(
            n_splits=cv,
            t1=cont["t1"],
            pct_embargo=pct_embargo,
        )
        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x,
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        oos = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        ).mean()

    elif method == "MDA":
        imp, oos = get_mean_decrease_accuracy(
            clf,
            X=trns_x,
            y=cont["bin"],
            cv=cv,
            sample_weight=cont["w"],
            t1=cont["t1"],
            pct_embargo=pct_embargo,
            scoring=scoring,
            random_state=random_state,
        )

    elif method == "SFI":
        cv_gen = PurgedKFold(
            n_splits=cv,
            t1=cont["t1"],
            pct_embargo=pct_embargo
        )

        predictions = generate_oof_predictions(
            estimator=clf,
            features=trns_x,
            labels=cont["bin"],
            sample_weight=cont["w"],
            cv=cv_gen,
            positive_label=1,
        )
        oos = _score_oof_folds(
            predictions,
            labels=cont["bin"],
            sample_weight=cont["w"],
            scoring=scoring,
        ).mean()

        imp = get_single_feature_importance(
            feat_names=trns_x.columns,
            clf=clf,
            trns_x=trns_x,
            cont=cont,
            scoring=scoring,
            cv_gen=cv_gen
        )

    else:
        raise ValueError("method must be one of: 'MDI', 'MDA', 'SFI'")

    return imp, oob, oos
