"""Shared safeguards for the notebook modeling and backtesting workflow."""

import numpy as np
import pandas as pd

from scipy.stats import norm
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import BaseCrossValidator

from src.data_preprocessing.prepare_the_data import EVENT_METADATA_COLUMNS
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


def build_meta_training_frame(
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


def probability_bet_size(
        probability: pd.Series,
        action: pd.Series,
        step_size: float = 0.10,
) -> pd.Series:
    """Convert meta probabilities and act/pass decisions into bounded bet sizes.

    Args:
        probability: Probability that the primary direction will be profitable.
        action: Binary act/pass decision, where one means act.
        step_size: Discretization interval in ``(0, 1]``.

    Returns:
        Non-negative position magnitudes aligned with ``probability``.
    """
    if not probability.index.equals(action.index):
        raise ValueError("probability and action must have the same index")
    if not 0.0 < step_size <= 1.0:
        raise ValueError("step_size must be in (0, 1]")
    if not probability.between(0.0, 1.0).all():
        raise ValueError("probability must be in [0, 1]")
    if not action.isin([0, 1]).all():
        raise ValueError("action must contain only 0 and 1")

    clipped = probability.clip(1e-12, 1.0 - 1e-12)
    z_value = (clipped - 0.5) / np.sqrt(clipped * (1.0 - clipped))
    continuous = pd.Series(
        2.0 * norm.cdf(z_value) - 1.0,
        index=probability.index,
        dtype="float64",
    ).clip(lower=0.0)
    sized = continuous * action.astype("float64")

    return ((sized / step_size).round() * step_size).clip(0.0, 1.0)


def compute_strategy_returns(
        predictions: pd.DataFrame,
        one_way_cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Compute primary-only and meta-filtered event returns with explicit costs.

    A completed event trade pays the one-way cost once on entry and once on exit.

    Args:
        predictions: Event outcomes, primary sides, meta actions, and bet sizes.
        one_way_cost_bps: Slippage charged for each entry or exit in basis points.

    Returns:
        Positions plus gross, entry, exit, total-cost, and net returns for both strategies.

    Raises:
        ValueError: If required columns or bounded decision values are invalid.
    """
    required = {"raw_return", "primary_side", "meta_action", "bet_size"}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing return columns: {sorted(missing)}")
    if one_way_cost_bps < 0:
        raise ValueError("one_way_cost_bps must be non-negative")
    if not predictions["primary_side"].isin([-1, 1]).all():
        raise ValueError("primary_side must contain only -1 and 1")
    if not predictions["meta_action"].isin([0, 1]).all():
        raise ValueError("meta_action must contain only 0 and 1")
    if not predictions["bet_size"].between(0.0, 1.0).all():
        raise ValueError("bet_size must be in [0, 1]")

    out = predictions.copy()
    out["primary_only_position"] = out["primary_side"].astype("float64")
    out["meta_filtered_position"] = (
        out["primary_side"]
        * out["meta_action"]
        * out["bet_size"]
    ).astype("float64")
    one_way_rate = one_way_cost_bps / 10_000.0

    for strategy in ["primary_only", "meta_filtered"]:
        position = out[f"{strategy}_position"]
        out[f"{strategy}_gross_return"] = position * out["raw_return"]
        out[f"{strategy}_entry_cost"] = position.abs() * one_way_rate
        out[f"{strategy}_exit_cost"] = position.abs() * one_way_rate
        out[f"{strategy}_total_cost"] = (
            out[f"{strategy}_entry_cost"] + out[f"{strategy}_exit_cost"]
        )
        out[f"{strategy}_net_return"] = (
            out[f"{strategy}_gross_return"] - out[f"{strategy}_total_cost"]
        )

    return out
