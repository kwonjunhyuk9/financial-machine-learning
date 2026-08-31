"""Shared safeguards for the notebook modeling and backtesting workflow."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator, StratifiedShuffleSplit

from src.data_preprocessing.prepare_the_data import EVENT_METADATA_COLUMNS
from src.model_backtesting.backtest_statistics import ClassificationScores
from src.strategy_modeling.ensemble_methods import (
    build_bagging_classifier,
    build_boosting_classifier,
    build_gradient_boosting_classifier,
    build_random_forest_classifier,
)
from src.strategy_modeling.hyperparameter_tuning import MyPipeline

PRIMARY_REQUIRED_FEATURES = {
    "mean_sentiment_score",
    "fractionally_differenced_log_close",
}
PRIMARY_REQUIRED_MODEL_COLUMNS = {
    "event_end",
    "direction_label",
    "sample_weight",
}
META_FEATURE_COLUMNS = (
    "primary_side",
    "primary_confidence",
)
META_GENERATED_COLUMNS = {
    *META_FEATURE_COLUMNS,
    "primary_probability",
    "meta_label",
}


def build_candidate_classifiers(
        random_state: int = 42,
        n_jobs: int = 1,
) -> dict[str, MyPipeline]:
    """Build the four tree-classifier families shared by primary and meta modeling.

    Args:
        random_state: Seed used by every stochastic classifier.
        n_jobs: Parallel workers used by bagging and random forest.

    Returns:
        Bagging, random forest, AdaBoost, and gradient-boosting pipelines.
    """
    estimators = {
        "bagging": build_bagging_classifier(
            n_estimators=120,
            max_samples=0.80,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "random_forest": build_random_forest_classifier(
            n_estimators=120,
            n_jobs=n_jobs,
            random_state=random_state,
        ),
        "adaboost": build_boosting_classifier(
            n_estimators=100,
            learning_rate=0.10,
            random_state=random_state,
        ),
        "gradient_boosting": build_gradient_boosting_classifier(
            n_estimators=100,
            learning_rate=0.10,
            max_depth=3,
            random_state=random_state,
        ),
    }

    return {
        name: MyPipeline([("model", estimator)])
        for name, estimator in estimators.items()
    }


def candidate_parameter_grids() -> dict[str, list[dict]]:
    """Return compact tuning grids for the shared classifier families."""
    return {
        "bagging": [
            {"model__max_samples": max_samples}
            for max_samples in [0.60, 0.80, 1.00]
        ],
        "random_forest": [
            {"model__max_features": max_features}
            for max_features in ["sqrt", 0.50, 1.00]
        ],
        "adaboost": [
            {"model__learning_rate": learning_rate}
            for learning_rate in [0.03, 0.10, 0.30]
        ],
        "gradient_boosting": [
            {"model__learning_rate": learning_rate}
            for learning_rate in [0.03, 0.10, 0.30]
        ],
    }

def build_primary_model_frame(
        events: pd.DataFrame,
        event_starts: Sequence[pd.Timestamp],
) -> pd.DataFrame:
    """Build a chronologically indexed frame for primary modeling.

    Args:
        events: Prepared events containing ``event_start`` as a column or index.
        event_starts: Event timestamps assigned to the requested model partition.

    Returns:
        A copy containing exactly the requested events, indexed and sorted by
        ``event_start``.

    Raises:
        ValueError: If timestamps are invalid, duplicated, or absent, or if
            required primary-model columns are missing.
    """
    if "event_start" in events.columns:
        indexed_events = events.copy()
        starts = pd.to_datetime(
            indexed_events["event_start"],
            utc=True,
            errors="coerce",
        )
        if starts.isna().any() or starts.duplicated().any():
            raise ValueError("Event starts must be unique valid timestamps")
        indexed_events["event_start"] = starts
        indexed_events = indexed_events.set_index("event_start")
    elif events.index.name == "event_start":
        indexed_events = events.copy()
        starts = pd.to_datetime(
            indexed_events.index,
            utc=True,
            errors="coerce",
        )
        if starts.isna().any() or starts.duplicated().any():
            raise ValueError("Event starts must be unique valid timestamps")
        indexed_events.index = pd.DatetimeIndex(starts, name="event_start")
    else:
        raise ValueError("events must contain event_start as a column or index")

    requested_starts = pd.DatetimeIndex(pd.to_datetime(
        event_starts,
        utc=True,
        errors="coerce",
    ), name="event_start")
    if (
        requested_starts.empty
        or requested_starts.isna().any()
        or requested_starts.duplicated().any()
    ):
        raise ValueError("event_starts must contain unique valid timestamps")

    missing_columns = (
        PRIMARY_REQUIRED_MODEL_COLUMNS | PRIMARY_REQUIRED_FEATURES
    ).difference(indexed_events.columns)
    if missing_columns:
        raise ValueError(
            f"Missing required primary model columns: {sorted(missing_columns)}"
        )

    missing_starts = requested_starts.difference(indexed_events.index)
    if not missing_starts.empty:
        raise ValueError("event_starts contain timestamps absent from events")

    return indexed_events.loc[requested_starts].sort_index().copy()


def get_primary_feature_columns(events: pd.DataFrame) -> list[str]:
    """Return event-start features while excluding outcomes and identifiers.

    Args:
        events: Event table containing metadata, labels, and candidate features.

    Returns:
        Candidate feature names in their input-column order.

    Raises:
        ValueError: If required event-start features are missing or no features remain.
    """
    missing = PRIMARY_REQUIRED_FEATURES.difference(events.columns)
    if missing:
        raise ValueError(f"Missing required primary features: {sorted(missing)}")

    feature_columns = [
        column
        for column in events.columns
        if column not in EVENT_METADATA_COLUMNS
    ]
    if not feature_columns:
        raise ValueError("No primary features remain after excluding metadata")

    return feature_columns


def build_meta_model_frame(
        events: pd.DataFrame,
        primary_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Create meta-labels exclusively from primary out-of-fold predictions.

    Args:
        events: Development events indexed by ``event_start`` or containing that column.
        primary_oof: Primary predictions with OOF provenance.

    Returns:
        Events augmented with primary direction, confidence, and binary meta-labels.

    Raises:
        ValueError: If primary predictions are not complete OOF predictions.
    """
    if "event_start" in events.columns:
        indexed_events = events.set_index("event_start")
    else:
        indexed_events = events.copy()

    required = {"prediction", "probability", "prediction_source"}
    missing = required.difference(primary_oof.columns)
    if missing:
        raise ValueError(f"Missing primary prediction columns: {sorted(missing)}")
    if not primary_oof["prediction_source"].eq("oof").all():
        raise ValueError("Meta-labels require primary OOF predictions")
    if not indexed_events.index.equals(primary_oof.index):
        raise ValueError("Events and primary OOF predictions must have the same index")

    out = indexed_events.copy()
    out["primary_side"] = primary_oof["prediction"].astype("int8")
    out["primary_probability"] = primary_oof["probability"].astype("float64")
    out["primary_confidence"] = np.maximum(
        out["primary_probability"],
        1.0 - out["primary_probability"],
    )
    out["meta_label"] = (
            out["primary_side"] * out["raw_return"] > 0
    ).astype("int8")

    return out


def get_meta_feature_columns(meta_frame: pd.DataFrame) -> list[str]:
    """Return primary features plus the approved meta-model features.

    Args:
        meta_frame: Frame produced by ``build_meta_model_frame``.

    Returns:
        Primary feature names followed by primary side and confidence.

    Raises:
        ValueError: If required meta-model features are missing.
    """
    missing = set(META_FEATURE_COLUMNS).difference(meta_frame.columns)
    if missing:
        raise ValueError(f"Missing required meta features: {sorted(missing)}")

    primary_frame = meta_frame.drop(
        columns=META_GENERATED_COLUMNS.intersection(meta_frame.columns),
    )
    return [
        *get_primary_feature_columns(primary_frame),
        *META_FEATURE_COLUMNS,
    ]


def generate_oof_predictions(
        estimator: BaseEstimator,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series,
        cv: BaseCrossValidator,
        positive_label: int = 1,
) -> pd.DataFrame:
    """Generate one prediction per row from purged out-of-fold estimators.

    Args:
        estimator: Classifier supporting ``fit``, ``predict``, and ``predict_proba``.
        features: Time-indexed feature matrix.
        labels: Labels aligned with ``features``.
        sample_weight: Training weights aligned with ``features``.
        cv: Cross-validator yielding train and test positional indices.
        positive_label: Class whose probability is stored in the output.

    Returns:
        Predictions, positive-class probabilities, fold ids, and OOF provenance.

    Raises:
        ValueError: If inputs are misaligned or a fold does not expose the positive class.
    """
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must have the same index")
    if not features.index.equals(sample_weight.index):
        raise ValueError("features and sample_weight must have the same index")

    predictions = pd.DataFrame(
        index=features.index,
        columns=["prediction", "probability", "fold"],
    )

    for fold, (train, test) in enumerate(cv.split(features)):
        fitted = clone(estimator).fit(
            features.iloc[train],
            labels.iloc[train],
            sample_weight=sample_weight.iloc[train].to_numpy(),
        )
        class_positions = np.flatnonzero(fitted.classes_ == positive_label)
        if class_positions.size != 1:
            raise ValueError(f"positive_label {positive_label!r} is absent from a fold")

        predictions.iloc[test, predictions.columns.get_loc("prediction")] = (
            fitted.predict(features.iloc[test])
        )
        predictions.iloc[test, predictions.columns.get_loc("probability")] = (
            fitted.predict_proba(features.iloc[test])[:, class_positions[0]]
        )
        predictions.iloc[test, predictions.columns.get_loc("fold")] = fold

    if predictions.isna().any().any():
        raise ValueError("cross-validation did not produce exactly one prediction per row")

    predictions["prediction"] = predictions["prediction"].astype(labels.dtype)
    predictions["probability"] = predictions["probability"].astype("float64")
    predictions["fold"] = predictions["fold"].astype("int64")
    predictions["prediction_source"] = "oof"

    return predictions


def score_binary_predictions(
        labels: pd.Series,
        predictions: pd.Series,
        positive_probabilities: pd.Series,
        sample_weight: pd.Series,
        *,
        class_labels: Sequence[int],
        positive_label: int = 1,
) -> dict[str, float]:
    """Score aligned binary-classification predictions with sample weights.

    Args:
        labels: Observed binary labels.
        predictions: Predicted class labels.
        positive_probabilities: Predicted probabilities for ``positive_label``.
        sample_weight: Evaluation weights aligned with ``labels``.
        class_labels: Ordered pair of labels used for log-loss probabilities.
        positive_label: Label represented by ``positive_probabilities``.

    Returns:
        Weighted log loss, accuracy, F1, precision, and recall scores.

    Raises:
        ValueError: If inputs are misaligned or the class definition is invalid.
    """
    for name, values in {
        "predictions": predictions,
        "positive_probabilities": positive_probabilities,
        "sample_weight": sample_weight,
    }.items():
        if not labels.index.equals(values.index):
            raise ValueError(f"labels and {name} must have the same index")

    if len(class_labels) != 2 or len(set(class_labels)) != 2:
        raise ValueError("class_labels must contain exactly two distinct labels")
    if positive_label not in class_labels:
        raise ValueError("positive_label must be present in class_labels")
    if not set(pd.unique(labels)).issubset(class_labels):
        raise ValueError("labels contain values outside class_labels")

    negative_label = next(
        class_label
        for class_label in class_labels
        if class_label != positive_label
    )
    probabilities_by_label = {
        negative_label: 1.0 - positive_probabilities.to_numpy(),
        positive_label: positive_probabilities.to_numpy(),
    }
    probabilities = np.column_stack([
        probabilities_by_label[class_label]
        for class_label in class_labels
    ])
    weights = sample_weight.to_numpy()

    return {
        "log_loss": float(-ClassificationScores.negative_log_loss(
            labels,
            probabilities,
            labels=class_labels,
            sample_weight=weights,
        )),
        "accuracy": float(ClassificationScores.accuracy(
            labels,
            predictions,
            sample_weight=weights,
        )),
        "f1": float(ClassificationScores.f1_score(
            labels,
            predictions,
            pos_label=positive_label,
            sample_weight=weights,
            zero_division=0,
        )),
        "precision": float(ClassificationScores.precision(
            labels,
            predictions,
            pos_label=positive_label,
            sample_weight=weights,
            zero_division=0,
        )),
        "recall": float(ClassificationScores.recall(
            labels,
            predictions,
            pos_label=positive_label,
            sample_weight=weights,
            zero_division=0,
        )),
    }


def get_weighted_learning_curve(
        estimator: BaseEstimator,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series,
        cv: BaseCrossValidator,
        *,
        train_sizes: Sequence[float] = (0.20, 0.40, 0.60, 0.80, 1.00),
        class_labels: Sequence[int] | None = None,
        positive_label: int = 1,
        scoring: str = "neg_log_loss",
        random_state: int = 42,
) -> pd.DataFrame:
    """Compute sample-weighted train and validation learning-curve errors.

    Args:
        estimator: Binary classifier supporting probabilities and sample weights.
        features: Time-indexed feature matrix.
        labels: Binary labels aligned with ``features``.
        sample_weight: Evaluation weights aligned with ``features``.
        cv: Cross-validator defining the purged train and validation folds.
        train_sizes: Fractions of each purged training fold to fit.
        class_labels: Ordered binary label pair. Inferred when omitted.
        positive_label: Label represented by the positive probability.
        scoring: ``"neg_log_loss"`` or ``"f1"``.
        random_state: Seed used for stratified training subsets.

    Returns:
        One row per training fraction with mean and standard-error train and
        validation errors. Log loss is returned directly; F1 is returned as
        ``1 - F1`` so lower values consistently indicate better performance.

    Raises:
        ValueError: If inputs, train sizes, or scoring are invalid.
    """
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must have the same index")
    if not features.index.equals(sample_weight.index):
        raise ValueError("features and sample_weight must have the same index")
    if scoring not in {"neg_log_loss", "f1"}:
        raise ValueError("scoring must be 'neg_log_loss' or 'f1'.")
    if not train_sizes or any(size <= 0.0 or size > 1.0 for size in train_sizes):
        raise ValueError("train_sizes must contain fractions in (0, 1].")

    ordered_labels = (
        np.sort(pd.unique(labels)).tolist()
        if class_labels is None
        else list(class_labels)
    )
    rows = []

    for fold, (train, validation) in enumerate(cv.split(features)):
        fold_labels = labels.iloc[train]
        for size_position, train_fraction in enumerate(train_sizes):
            if train_fraction == 1.0:
                selected = np.arange(train.shape[0])
            else:
                subset_size = max(
                    len(ordered_labels),
                    int(np.floor(train.shape[0] * train_fraction)),
                )
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    train_size=subset_size,
                    random_state=random_state + fold * len(train_sizes) + size_position,
                )
                selected, _ = next(splitter.split(
                    np.zeros(train.shape[0]),
                    fold_labels,
                ))

            selected_train = train[selected]
            fitted = clone(estimator).fit(
                features.iloc[selected_train],
                labels.iloc[selected_train],
                sample_weight=sample_weight.iloc[selected_train].to_numpy(),
            )
            class_positions = np.flatnonzero(fitted.classes_ == positive_label)
            if class_positions.size != 1:
                raise ValueError(
                    f"positive_label {positive_label!r} is absent from a fold"
                )

            split_errors = {}
            for split_name, positions in {
                "train": selected_train,
                "validation": validation,
            }.items():
                split_features = features.iloc[positions]
                split_labels = labels.iloc[positions]
                split_weights = sample_weight.iloc[positions]
                predictions = pd.Series(
                    fitted.predict(split_features),
                    index=split_features.index,
                )
                probabilities = pd.Series(
                    fitted.predict_proba(split_features)[:, class_positions[0]],
                    index=split_features.index,
                )
                scores = score_binary_predictions(
                    split_labels,
                    predictions,
                    probabilities,
                    split_weights,
                    class_labels=ordered_labels,
                    positive_label=positive_label,
                )
                split_errors[split_name] = (
                    scores["log_loss"]
                    if scoring == "neg_log_loss"
                    else 1.0 - scores["f1"]
                )

            rows.append({
                "fold": fold,
                "train_fraction": float(train_fraction),
                "train_size": int(selected_train.shape[0]),
                "train_error": split_errors["train"],
                "validation_error": split_errors["validation"],
            })

    fold_results = pd.DataFrame(rows)
    summary = fold_results.groupby("train_fraction", sort=False).agg(
        train_size=("train_size", "mean"),
        train_error_mean=("train_error", "mean"),
        train_error_std=("train_error", "sem"),
        validation_error_mean=("validation_error", "mean"),
        validation_error_std=("validation_error", "sem"),
    )
    summary["train_size"] = summary["train_size"].round().astype("int64")

    return summary.reset_index()
