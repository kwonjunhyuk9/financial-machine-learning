"""Shared safeguards for the notebook modeling and backtesting workflow."""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, StratifiedShuffleSplit

from src.preprocessing.prepare_the_data import EVENT_METADATA_COLUMNS
from src.preprocessing.market_technical_indicators import MODEL_FEATURES, require_features
from src.backtesting.backtest_statistics import ClassificationScores
from src.modeling.purged_validation import PurgedKFold, index_events
from src.modeling.ensemble_methods import (
    build_bagging_classifier,
    build_boosting_classifier,
    build_random_forest_classifier,
)
from src.modeling.hyperparameter_tuning import MyPipeline

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


@dataclass
class ModelSelectionResult:
    """Full OOF search and each family's best configuration for one stage."""

    estimators: dict[str, MyPipeline]
    oof_predictions: dict[str, pd.DataFrame]
    comparison: pd.DataFrame
    selected_name: str
    selected_estimator: MyPipeline
    final_estimator: BaseEstimator
    final_configuration: dict[str, object]
    final_oof: pd.DataFrame
    tuning: pd.DataFrame
    configurations: dict[str, dict[str, object]]


def build_candidate_classifiers(
        *,
        random_state: int,
        n_jobs: int,
        candidate_settings: dict[str, dict],
) -> dict[str, MyPipeline]:
    """Build the three tree-classifier families shared by primary and meta modeling.

    Args:
        random_state: Seed used by every stochastic classifier.
        n_jobs: Parallel workers used by bagging and random forest.
        candidate_settings: Notebook-owned constructor parameters by model family.

    Returns:
        Boosting, bagging, and random-forest pipelines in display order.
    """
    builders = {"boosting": build_boosting_classifier,
                "bagging": build_bagging_classifier,
                "random_forest": build_random_forest_classifier}
    estimators = {
        name: builders[name](**parameters, random_state=random_state,
                             **({"n_jobs": n_jobs} if name != "boosting" else {}))
        for name, parameters in candidate_settings.items()
    }

    return {
        name: MyPipeline([("model", estimator)])
        for name, estimator in estimators.items()
    }


def build_primary_model_frame(
        events: pd.DataFrame,
        event_starts: Sequence[pd.Timestamp],
        model_kind: str = "sentiment",
) -> pd.DataFrame:
    """Build a chronologically indexed frame for primary modeling.

    Args:
        events: Prepared events containing ``event_start`` as a column or index.
        event_starts: Event timestamps assigned to the requested model partition.
        model_kind: Market-only or market-plus-sentiment input family.

    Returns:
        A copy containing exactly the requested events, indexed and sorted by
        ``event_start``.

    Raises:
        ValueError: If timestamps are invalid, duplicated, or absent, or if
            required primary-model columns are missing.
    """
    indexed_events = index_events(events)
    requested = (pd.MultiIndex.from_frame(event_starts[["symbol", "event_start"]])
                 if isinstance(event_starts, pd.DataFrame) else event_starts)
    if not isinstance(requested, pd.MultiIndex) or requested.has_duplicates:
        raise ValueError("Requested events must have unique valid composite (symbol, event_start) keys")
    required_features = {"fractionally_differenced_log_close"}
    if model_kind == "sentiment":
        required_features |= {"mean_sentiment_score"}
    if model_kind not in ("market", "sentiment"):
        raise ValueError("model_kind must be market or sentiment")
    missing = (PRIMARY_REQUIRED_MODEL_COLUMNS | required_features).difference(indexed_events.columns)
    if missing:
        raise ValueError(f"Missing required primary model columns: {missing}")
    if not requested.isin(indexed_events.index).all():
        raise ValueError("Requested keys are absent from events")
    return index_events(indexed_events.loc[requested])


def get_primary_feature_columns(events: pd.DataFrame, model_kind: str = "sentiment") -> list[str]:
    """Return event-start features while excluding outcomes and identifiers.

    Args:
        events: Event table containing metadata, labels, and candidate features.
        model_kind: Market-only or market-plus-sentiment input family.

    Returns:
        Candidate feature names in their input-column order.

    Raises:
        ValueError: If required event-start features are missing or no features remain.
    """
    if model_kind not in ("market", "sentiment"):
        raise ValueError("model_kind must be market or sentiment")
    expected = [name for name in MODEL_FEATURES
                if model_kind == "sentiment" or name != "mean_sentiment_score"]
    feature_columns = [name for name in events.columns
                       if name not in EVENT_METADATA_COLUMNS
                       and (model_kind == "sentiment" or name != "mean_sentiment_score")]
    require_features(feature_columns, expected)
    return expected


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
        indexed_events = index_events(events)
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


def get_meta_feature_columns(meta_frame: pd.DataFrame, model_kind: str = "sentiment") -> list[str]:
    """Return primary features plus the approved meta-model features.

    Args:
        meta_frame: Frame produced by ``build_meta_model_frame``.
        model_kind: Same input family as the primary model.

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
        *get_primary_feature_columns(primary_frame, model_kind),
        *META_FEATURE_COLUMNS,
    ]


def generate_oof_predictions(
        estimator: BaseEstimator,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series,
        cv: BaseCrossValidator,
        positive_label: int = 1,
        *,
        show_progress: bool = False,
        _progress: tqdm | None = None,
        _progress_label: str = "",
) -> pd.DataFrame:
    """Generate one prediction per row from purged out-of-fold estimators.

    Args:
        estimator: Classifier supporting ``fit``, ``predict``, and ``predict_proba``.
        features: Time-indexed feature matrix.
        labels: Labels aligned with ``features``.
        sample_weight: Training weights aligned with ``features``.
        cv: Cross-validator yielding train and test positional indices.
        positive_label: Class whose probability is stored in the output.
        show_progress: Show completed validation folds.

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

    with (nullcontext(_progress) if _progress is not None else
          tqdm(total=cv.get_n_splits(features, labels), desc="OOF predictions",
               disable=not show_progress)) as progress:
        for fold, (train, test) in enumerate(cv.split(features)):
            progress.set_postfix_str(f"{_progress_label} fold {fold + 1}".strip())
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
            progress.update(1)

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


def run_model_selection_workflow(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    information_sets: pd.Series,
    *,
    scoring: str,
    cv: int,
    pct_embargo: float,
    random_state: int,
    n_jobs: int,
    candidate_settings: dict[str, dict],
    parameter_grids: dict[str, dict[str, list]],
    show_progress: bool = False,
) -> ModelSelectionResult:
    """Search every configuration using weighted pooled development OOF scores.

    Args:
        features: Development features, excluding holdout rows.
        labels: Primary {-1, 1} or meta {0, 1} labels.
        sample_weight: Aligned training and evaluation weights.
        information_sets: Aligned event end times for purging.
        scoring: Primary negative log loss or meta F1 objective.
        cv: Number of purged folds.
        pct_embargo: Embargo fraction.
        random_state: Classifier seed.
        n_jobs: Ensemble workers.
        candidate_settings: Notebook-owned constructor settings.
        parameter_grids: Notebook-owned grids for every family.
        show_progress: Show completed candidate folds across the full search.

    Returns:
        Family winners, their OOF predictions and configurations, the full
        search table, and the overall winner. Exact metric ties keep grid order.
    """
    if scoring not in {"neg_log_loss", "f1"}:
        raise ValueError("scoring must be 'neg_log_loss' or 'f1'")
    class_labels = np.sort(pd.unique(labels)).tolist()
    if class_labels not in ([-1, 1], [0, 1]):
        raise ValueError("labels must use {-1, 1} or {0, 1}")
    bases = build_candidate_classifiers(
        random_state=random_state, n_jobs=n_jobs, candidate_settings=candidate_settings,
    )
    splitter = PurgedKFold(cv, t1=information_sets, pct_embargo=pct_embargo)
    estimators, oof_predictions, configurations = {}, {}, {}
    rows, best_rows = [], []

    def rank(row):
        return ((row["log_loss"], -row["f1"]) if scoring == "neg_log_loss"
                else (-row["f1"], row["log_loss"]))

    total = sum(len(ParameterGrid(parameter_grids[name])) for name in bases) * cv
    with tqdm(total=total, desc="Ensemble search", disable=not show_progress) as progress:
        for name, base in bases.items():
            best_row = None
            for configuration in ParameterGrid(parameter_grids[name]):
                estimator = clone(base).set_params(**configuration)
                predictions = generate_oof_predictions(
                    estimator, features, labels, sample_weight, splitter,
                    positive_label=1, _progress=progress,
                    _progress_label=f"{name}: {configuration}",
                )
                scores = score_binary_predictions(
                    labels, predictions["prediction"], predictions["probability"],
                    sample_weight, class_labels=class_labels, positive_label=1,
                )
                row = {"candidate": name, "configuration": dict(configuration),
                       **configuration, **scores}
                rows.append(row)
                if best_row is None or rank(row) < rank(best_row):
                    best_row = row
                    estimators[name] = estimator
                    oof_predictions[name] = predictions
                    configurations[name] = dict(configuration)
            best_rows.append(best_row)
    comparison = pd.DataFrame(best_rows).set_index("candidate")
    selected_name = min(best_rows, key=rank)["candidate"]
    selected_estimator = estimators[selected_name]
    return ModelSelectionResult(
        estimators=estimators, oof_predictions=oof_predictions,
        comparison=comparison, selected_name=selected_name,
        selected_estimator=selected_estimator, final_estimator=selected_estimator,
        final_configuration=configurations[selected_name],
        final_oof=oof_predictions[selected_name], tuning=pd.DataFrame(rows),
        configurations=configurations,
    )


def finalize_primary_model(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    estimator: BaseEstimator,
    final_oof: pd.DataFrame,
    feature_columns: Sequence[str],
    artifact_dir: Path,
    selected_candidate: str,
    best_configuration: dict[str, object],
    random_state: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Fit and save the final primary model and its OOF/holdout predictions.

    Args:
        development: Prepared development events.
        holdout: Prepared holdout events.
        estimator: Selected configuration, cloned before fitting on development.
        final_oof: Development OOF predictions aligned with the development frame.
        feature_columns: Ordered features used for fitting and holdout inference.
        artifact_dir: Existing directory for the model and prediction files.
        selected_candidate: Candidate name persisted with the fitted model.
        best_configuration: Tuned parameters persisted with the fitted model.
        random_state: Research seed recorded in the model artifact.
        show_progress: Show fitting, holdout prediction, and artifact saves.

    Returns:
        Combined development and holdout predictions with the existing composite
        index, schema, and provenance. The same rows are saved as Parquet with
        symbol and event_start columns, alongside the fitted Joblib artifact.
    """
    with tqdm(total=4, desc="Final primary model", disable=not show_progress) as progress:
        progress.set_postfix_str("Fit development data")
        final_primary = clone(estimator).fit(
            development[feature_columns],
            development["direction_label"].astype("int8"),
            sample_weight=development["sample_weight"].astype("float64").to_numpy(),
        )
        progress.update(1)
        progress.set_postfix_str("Predict holdout")
        holdout_probability = final_primary.predict_proba(
            holdout[feature_columns]
        )[:, list(final_primary.classes_).index(1)]
        holdout_side = final_primary.predict(holdout[feature_columns]).astype("int8")
        progress.update(1)

        primary_oof_output = development[[
            "event_end", "raw_return", "direction_label", "sample_weight"
        ]].copy()
        primary_oof_output["partition"] = "development"
        primary_oof_output["primary_side"] = final_oof["prediction"].astype("int8")
        primary_oof_output["primary_probability"] = final_oof["probability"]
        primary_oof_output["primary_probability_negative"] = 1.0 - final_oof["probability"]
        primary_oof_output["primary_probability_positive"] = final_oof["probability"]
        primary_oof_output["primary_class_probability"] = np.where(
            primary_oof_output["primary_side"].eq(1),
            primary_oof_output["primary_probability_positive"],
            primary_oof_output["primary_probability_negative"],
        )
        primary_oof_output["primary_confidence"] = np.maximum(
            final_oof["probability"],
            1.0 - final_oof["probability"],
        )
        primary_oof_output["prediction_source"] = final_oof["prediction_source"]
        primary_oof_output["cv_fold"] = final_oof["fold"]

        primary_holdout_output = holdout[[
            "event_end", "raw_return", "direction_label", "sample_weight"
        ]].copy()
        primary_holdout_output["partition"] = "holdout"
        primary_holdout_output["primary_side"] = holdout_side
        primary_holdout_output["primary_probability"] = holdout_probability
        primary_holdout_output["primary_probability_negative"] = 1.0 - holdout_probability
        primary_holdout_output["primary_probability_positive"] = holdout_probability
        primary_holdout_output["primary_class_probability"] = np.where(
            primary_holdout_output["primary_side"].eq(1),
            primary_holdout_output["primary_probability_positive"],
            primary_holdout_output["primary_probability_negative"],
        )
        primary_holdout_output["primary_confidence"] = np.maximum(
            holdout_probability,
            1.0 - holdout_probability,
        )
        primary_holdout_output["prediction_source"] = "holdout"
        primary_holdout_output["cv_fold"] = pd.NA

        primary_predictions = pd.concat([
            primary_oof_output,
            primary_holdout_output,
        ]).sort_index(level=["event_start", "symbol"])
        progress.set_postfix_str("Save predictions")
        primary_predictions.reset_index().to_parquet(artifact_dir / "primary_predictions.parquet", index=False)
        progress.update(1)
        progress.set_postfix_str("Save model")
        joblib.dump({
            "estimator": final_primary,
            "feature_columns": feature_columns,
            "selected_candidate": selected_candidate,
            "best_configuration": best_configuration,
            "holdout_boundary": holdout.index.get_level_values("event_start").min(),
            "random_state": random_state,
        }, artifact_dir / "primary_model.joblib")
        progress.update(1)

        return primary_predictions


def finalize_meta_model(
    meta_development: pd.DataFrame,
    events: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    *,
    estimator: BaseEstimator,
    final_oof: pd.DataFrame,
    feature_columns: Sequence[str],
    artifact_dir: Path,
    selected_candidate: str,
    best_configuration: dict[str, object],
    random_state: int,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Fit and save the final meta model and its OOF/holdout predictions.

    Args:
        meta_development: Development events augmented with primary OOF predictions.
        events: Prepared events indexed by (symbol, event_start).
        primary_predictions: Primary predictions with development/holdout partitions.
        estimator: Selected configuration, cloned before fitting on development.
        final_oof: Development OOF predictions aligned with the development frame.
        feature_columns: Ordered features used for fitting and holdout inference.
        artifact_dir: Existing directory for the model and prediction files.
        selected_candidate: Candidate name persisted with the fitted model.
        best_configuration: Tuned parameters persisted with the fitted model.
        random_state: Research seed recorded in the model artifact.
        show_progress: Show fitting, holdout prediction, and artifact saves.

    Returns:
        Combined development and holdout predictions with the existing composite
        index, schema, and provenance. The same rows are saved as Parquet with
        symbol and event_start columns, alongside the fitted Joblib artifact.
    """
    with tqdm(total=4, desc="Final meta model", disable=not show_progress) as progress:
        progress.set_postfix_str("Fit development data")
        final_meta = clone(estimator).fit(
            meta_development[feature_columns],
            meta_development["meta_label"].astype("int8"),
            sample_weight=meta_development["sample_weight"].astype("float64").to_numpy(),
        )
        progress.update(1)
        progress.set_postfix_str("Predict holdout")

        holdout_primary = primary_predictions[
            primary_predictions["partition"].eq("holdout")
        ]
        holdout_events = events.loc[holdout_primary.index].copy()
        holdout_events["primary_side"] = holdout_primary["primary_side"].astype("int8")
        holdout_events["primary_probability"] = holdout_primary["primary_probability"]
        holdout_events["primary_confidence"] = holdout_primary["primary_confidence"]
        holdout_events["meta_label"] = (
            holdout_events["primary_side"] * holdout_events["raw_return"] > 0
        ).astype("int8")

        holdout_probability = final_meta.predict_proba(
            holdout_events[feature_columns]
        )[:, list(final_meta.classes_).index(1)]
        holdout_action = final_meta.predict(holdout_events[feature_columns]).astype("int8")
        progress.update(1)

        meta_oof_output = meta_development[[
            "event_end", "raw_return", "direction_label", "sample_weight",
            "primary_side", "primary_probability", "primary_confidence", "meta_label",
        ]].copy()
        meta_oof_output["partition"] = "development"
        meta_oof_output["meta_action"] = final_oof["prediction"].astype("int8")
        meta_oof_output["meta_probability"] = final_oof["probability"]
        meta_oof_output["prediction_source"] = final_oof["prediction_source"]
        meta_oof_output["cv_fold"] = final_oof["fold"]

        meta_holdout_output = holdout_events[[
            "event_end", "raw_return", "direction_label", "sample_weight",
            "primary_side", "primary_probability", "primary_confidence", "meta_label",
        ]].copy()
        meta_holdout_output["partition"] = "holdout"
        meta_holdout_output["meta_action"] = holdout_action
        meta_holdout_output["meta_probability"] = holdout_probability
        meta_holdout_output["prediction_source"] = "holdout"
        meta_holdout_output["cv_fold"] = pd.NA

        meta_predictions = pd.concat([
            meta_oof_output,
            meta_holdout_output,
        ]).sort_index(level=["event_start", "symbol"])
        progress.set_postfix_str("Save predictions")
        meta_predictions.reset_index().to_parquet(artifact_dir / "meta_predictions.parquet", index=False)
        progress.update(1)
        progress.set_postfix_str("Save model")
        joblib.dump({
            "estimator": final_meta,
            "feature_columns": feature_columns,
            "selected_candidate": selected_candidate,
            "best_configuration": best_configuration,
            "random_state": random_state,
        }, artifact_dir / "meta_model.joblib")
        progress.update(1)

        return meta_predictions


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
        show_progress: bool = False,
        _progress: tqdm | None = None,
        _progress_label: str = "",
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
        show_progress: Show completed fold and training-size evaluations.

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

    with (nullcontext(_progress) if _progress is not None else
          tqdm(total=cv.get_n_splits(features, labels) * len(train_sizes),
               desc="Learning curve", disable=not show_progress)) as progress:
        for fold, (train, validation) in enumerate(cv.split(features)):
            fold_labels = labels.iloc[train]
            for size_position, train_fraction in enumerate(train_sizes):
                progress.set_postfix_str(
                    f"{_progress_label} fold {fold + 1}, size {train_fraction:.0%}"
                )
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
                progress.update(1)

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


def plot_learning_curves(
        estimators: dict[str, BaseEstimator],
        heading: str,
        *,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series,
        cv: BaseCrossValidator,
        train_sizes: Sequence[float],
        class_labels: Sequence[int],
        positive_label: int = 1,
        scoring: str = "neg_log_loss",
        random_state: int = 42,
        show_progress: bool = False,
) -> dict[str, pd.DataFrame]:
    """Plot weighted learning curves for a collection of estimators.

    Args:
        estimators: Named estimators to compare.
        heading: Figure title.
        features: Development feature matrix.
        labels: Binary labels aligned with ``features``.
        sample_weight: Evaluation weights aligned with ``features``.
        cv: Cross-validator shared by each estimator.
        train_sizes: Fractions passed to ``get_weighted_learning_curve``.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.
        scoring: Learning-curve scoring rule.
        random_state: Seed used for stratified training subsets.
        show_progress: Show one bar shared by all model learning curves.

    Returns:
        Learning-curve frames keyed by estimator name.
    """
    results = {}
    model_count = len(estimators)
    fig, axes = plt.subplots(
        1, model_count, figsize=(7 * model_count, 5.5),
        sharex=False, sharey=False,
    )
    axes = np.atleast_1d(axes).ravel()
    with tqdm(total=model_count * cv.get_n_splits(features, labels) * len(train_sizes),
              desc="Learning curves", disable=not show_progress) as progress:
        for axis, (name, estimator) in zip(axes, estimators.items()):
            curve = get_weighted_learning_curve(
                estimator,
                features,
                labels,
                sample_weight,
                cv,
                train_sizes=train_sizes,
                class_labels=class_labels,
                positive_label=positive_label,
                scoring=scoring,
                random_state=random_state,
                _progress=progress, _progress_label=name,
            )
            results[name] = curve
            x = curve["train_size"].to_numpy()
            for split, color in [("train", "tab:red"), ("validation", "tab:blue")]:
                mean = curve[f"{split}_error_mean"].to_numpy()
                error = curve[f"{split}_error_std"].fillna(0.0).to_numpy()
                axis.plot(x, mean, marker="o", color=color, label=split)
                axis.fill_between(
                    x,
                    mean - error,
                    mean + error,
                    color=color,
                    alpha=0.15,
                )
            axis.set_title(name.replace("_", " ").title())
            axis.set_xlabel("Training set size")
            axis.set_ylabel(
                "Weighted log loss" if scoring == "neg_log_loss" else "1 − weighted F1"
            )
            axis.grid(alpha=0.25)
            axis.legend()
    for unused_axis in axes[model_count:]:
        unused_axis.remove()
    fig.suptitle(heading, fontsize=14)
    fig.tight_layout()
    plt.show()
    return results


def compute_stage_importance(
        estimator: BaseEstimator,
        *,
        features: pd.DataFrame,
        labels: pd.Series,
        sample_weight: pd.Series,
        information_sets: pd.Series,
        scoring: str,
        cv: int,
        pct_embargo: float,
        random_state: int,
        show_progress: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    """Compute MDI, MDA, and SFI results for one modeling stage.

    Args:
        estimator: Selected classifier pipeline.
        features: Development feature matrix.
        labels: Binary labels aligned with ``features``.
        sample_weight: Training and evaluation weights.
        information_sets: Event end times used by purged validation.
        scoring: Importance scoring rule.
        cv: Number of purged validation folds.
        pct_embargo: Embargo fraction applied to each fold.
        random_state: Seed used by feature permutations.
        show_progress: Show one bar for each importance method.

    Returns:
        Importance frames keyed by method and their OOS scores.
    """
    from src.modeling.feature_importance import get_estimator_feature_importance

    results = {}
    scores = {}
    for method in ["MDI", "MDA", "SFI"]:
        importance, oos = get_estimator_feature_importance(
            estimator,
            features,
            labels,
            sample_weight,
            information_sets,
            method=method,
            scoring=scoring,
            cv=cv,
            pct_embargo=pct_embargo,
            random_state=random_state,
            show_progress=show_progress,
        )
        results[method] = importance
        scores[method] = oos
    return results, pd.Series(scores, name="oos_score")


def plot_feature_importance(
        results: dict[str, pd.DataFrame],
        heading: str,
) -> None:
    """Plot the leading MDI, MDA, and SFI results.

    Args:
        results: Importance frames keyed by ``MDI``, ``MDA``, and ``SFI``.
        heading: Figure title.

    Returns:
        None.
    """
    fig, axes = plt.subplots(1, 3, figsize=(19, 8))
    for axis, method in zip(axes, ["MDI", "MDA", "SFI"]):
        importance = results[method].nlargest(15, "mean").sort_values("mean")
        axis.barh(
            importance.index,
            importance["mean"],
            xerr=importance["std"].fillna(0.0),
            alpha=0.75,
            color="tab:blue",
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_title(method)
        axis.set_xlabel("Mean importance or score")
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle(heading, fontsize=14)
    fig.tight_layout()
    plt.show()


def build_model_evaluation_table(
        predictions_by_model: dict[str, pd.DataFrame],
        observed: pd.Series,
        sample_weight: pd.Series,
        *,
        class_labels: Sequence[int],
        positive_label: int = 1,
) -> pd.DataFrame:
    """Build the weighted metric table shared by both modeling stages.

    Args:
        predictions_by_model: Prediction and probability frames keyed by model.
        observed: Observed binary labels.
        sample_weight: Evaluation weights aligned with ``observed``.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.

    Returns:
        Weighted metrics indexed by display-formatted model name.
    """
    rows = []
    for name, predictions in predictions_by_model.items():
        scores = score_binary_predictions(
            observed,
            predictions["prediction"],
            predictions["probability"],
            sample_weight,
            class_labels=class_labels,
            positive_label=positive_label,
        )
        rows.append({"model": name, **scores})
    return pd.DataFrame(rows).set_index("model")[
        ["accuracy", "precision", "recall", "f1", "log_loss"]
    ].rename(
        index=lambda name: name.replace("_", " ").title(),
    )


def plot_model_evaluation(
        predictions_by_model: dict[str, pd.DataFrame],
        observed: pd.Series,
        sample_weight: pd.Series,
        heading: str,
        *,
        class_labels: Sequence[int],
        positive_label: int = 1,
) -> None:
    """Plot weighted confusion matrices, precision-recall curves, and ROC curves.

    Args:
        predictions_by_model: Prediction and probability frames keyed by model.
        observed: Observed binary labels.
        sample_weight: Evaluation weights aligned with ``observed``.
        heading: Figure title.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.

    Returns:
        None.
    """
    model_count = len(predictions_by_model)
    fig, axes = plt.subplots(1, model_count, figsize=(5 * model_count, 5.5))
    axes = np.atleast_1d(axes).ravel()
    for axis, (name, predictions) in zip(axes, predictions_by_model.items()):
        matrix = confusion_matrix(
            observed,
            predictions["prediction"],
            labels=class_labels,
            sample_weight=sample_weight,
            normalize="true",
        )
        ConfusionMatrixDisplay(matrix, display_labels=class_labels).plot(
            ax=axis,
            colorbar=False,
            values_format=".2f",
        )
        axis.set_title(name.replace("_", " ").title())
    for unused_axis in axes[model_count:]:
        unused_axis.remove()
    fig.suptitle(f"{heading}: Weighted Confusion Matrices", fontsize=14)
    fig.tight_layout()
    plt.show()

    fig, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(14, 5))
    for name, predictions in predictions_by_model.items():
        probability = predictions["probability"]
        precision, recall, _ = precision_recall_curve(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        average_precision = average_precision_score(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        display_name = name.replace("_", " ").title()
        pr_axis.plot(
            recall,
            precision,
            label=f"{display_name} (AP={average_precision:.3f})",
        )

        false_positive_rate, true_positive_rate, _ = roc_curve(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        auc = roc_auc_score(
            observed,
            probability,
            sample_weight=sample_weight,
        )
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{display_name} (AUC={auc:.3f})",
        )

    pr_axis.set(xlabel="Recall", ylabel="Precision", title="Precision-Recall")
    pr_axis.grid(alpha=0.25)
    pr_axis.legend(fontsize=8)
    roc_axis.plot([0, 1], [0, 1], "k--", label="Random")
    roc_axis.set(
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="ROC",
    )
    roc_axis.grid(alpha=0.25)
    roc_axis.legend(fontsize=8)
    fig.suptitle(heading, fontsize=14)
    fig.tight_layout()
    plt.show()
