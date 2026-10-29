"""Shared safeguards for the notebook modeling and backtesting workflow."""

from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import joblib

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tqdm.auto import tqdm

from sklearn.base import BaseEstimator, clone
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import BaseCrossValidator, ParameterGrid, StratifiedShuffleSplit

from src.preprocessing.prepare_the_data import EVENT_METADATA_COLUMNS
from src.backtesting.backtest_statistics import ClassificationScores
from src.modeling.purged_validation import PurgedKFold, index_events
from src.modeling.ensemble_methods import (
    build_bagging_classifier,
    build_boosting_classifier,
    build_random_forest_classifier,
)
from src.modeling.hyperparameter_tuning import MyPipeline

META_FEATURE_COLUMNS = (
    "primary_prediction",
    "primary_class_probability",
)
PRIMARY_EVENT_COLUMNS = (
    "event_end", "vertical_barrier", "horizon_volatility", "raw_return",
    "primary_label", "sample_weight", "holdout_boundary", "partition",
)


@dataclass
class ModelSearchResult:
    """All configuration scores and the directory containing their fitted models."""

    tuning: pd.DataFrame
    artifact_dir: Path
    data_hash: str
    primary_model_hash: str | None


def _modeling_data_hash(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    information_sets: pd.Series,
) -> str:
    """Identify the development inputs used to fit and validate saved models."""
    digest = hashlib.sha256()
    digest.update(repr(list(features.columns)).encode())
    for values in (features, labels, sample_weight, information_sets):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    return digest.hexdigest()


def load_model_search(artifact_dir: Path) -> ModelSearchResult:
    """Load a saved model search.

    Args:
        artifact_dir: Model search artifact directory.

    Returns:
        Search table and metadata for the saved configuration models.
    """
    result = joblib.load(Path(artifact_dir) / "search.joblib")
    result.artifact_dir = Path(artifact_dir).resolve()
    return result


def load_tuning_configuration(
    result: ModelSearchResult,
    configuration_id: str | None,
    *,
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    information_sets: pd.Series,
    primary_model_hash: str | None = None,
    candidate_settings: dict[str, dict] | None = None,
) -> dict:
    """Load a selected model configuration.

    Args:
        result: Completed search to select from.
        configuration_id: Selected configuration identifier.
        features: Current development features in training order.
        labels: Current development labels.
        sample_weight: Observation weights.
        information_sets: Current event ends used for purging.
        primary_model_hash: Current primary artifact hash for a meta search.
        candidate_settings: Classifier constructor settings by model family.

    Returns:
        Fitted estimator, OOF predictions, configuration, and training metadata.
    """
    if not configuration_id or configuration_id not in set(result.tuning.configuration_id):
        raise ValueError("Set SELECTED_CONFIGURATION_ID to an ID from the tuning table")
    if result.primary_model_hash != primary_model_hash:
        raise ValueError("Primary model changed; rerun meta tuning")
    if result.data_hash != _modeling_data_hash(features, labels, sample_weight, information_sets):
        raise ValueError("Development inputs changed; rerun tuning")
    selected = joblib.load(result.artifact_dir / f"{configuration_id}.joblib")
    if (candidate_settings is not None
            and selected.get("candidate_settings") != candidate_settings.get(selected["candidate"])):
        raise ValueError("Candidate settings changed; rerun tuning with the current class weighting")
    return selected


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
    return {
        name: MyPipeline([("model", builders[name](
            **parameters, random_state=random_state,
            **({"n_jobs": n_jobs} if name != "boosting" else {}),
        ))])
        for name, parameters in candidate_settings.items()
    }


def build_primary_model_frame(
        events: pd.DataFrame,
        event_starts: Sequence[pd.Timestamp],
        model_kind: str = "sentiment",
) -> pd.DataFrame:
    """Prepare events for primary modeling.

    Args:
        events: Prepared events containing ``event_start`` as a column or index.
        event_starts: Event timestamps assigned to the requested model partition.
        model_kind: Market-only or market-plus-sentiment input family.

    Returns:
        Primary-model events indexed by event identity.
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
    missing = {"event_end", "primary_label", "sample_weight", *required_features}.difference(indexed_events.columns)
    if missing:
        raise ValueError(f"Missing required primary model columns: {missing}")
    if not requested.isin(indexed_events.index).all():
        raise ValueError("Requested keys are absent from events")
    return indexed_events.loc[requested].sort_index(level=["event_start", "symbol"])


def get_primary_feature_columns(events: pd.DataFrame, model_kind: str = "sentiment") -> list[str]:
    """Identify primary-model feature columns.

    Args:
        events: Event table containing metadata, labels, and candidate features.
        model_kind: Market-only or market-plus-sentiment input family.

    Returns:
        Candidate feature names in their input-column order.
    """
    if model_kind not in ("market", "sentiment"):
        raise ValueError("model_kind must be market or sentiment")
    feature_columns = [name for name in events.columns
                       if name not in EVENT_METADATA_COLUMNS
                       and (model_kind == "sentiment" or name != "mean_sentiment_score")]
    if not feature_columns:
        raise ValueError("No primary model features remain")
    return feature_columns


def build_meta_model_frame(
        events: pd.DataFrame,
        primary_oof: pd.DataFrame,
) -> pd.DataFrame:
    """Prepare meta-model events from primary out-of-fold predictions.

    Args:
        events: Development events indexed by ``event_start`` or containing that column.
        primary_oof: Primary predictions with OOF provenance.

    Returns:
        Events augmented with primary direction, confidence, and binary meta-labels.
    """
    if "event_start" in events.columns:
        indexed_events = index_events(events)
    else:
        indexed_events = events

    required = {"prediction", "probability", "prediction_source"}
    missing = required.difference(primary_oof.columns)
    if missing:
        raise ValueError(f"Missing primary prediction columns: {sorted(missing)}")
    if not primary_oof["prediction_source"].eq("oof").all():
        raise ValueError("Meta-labels require primary OOF predictions")
    if not indexed_events.index.equals(primary_oof.index):
        raise ValueError("Events and primary OOF predictions must have the same index")

    out = indexed_events.copy()
    out["primary_prediction"] = primary_oof["prediction"].astype("int8")
    out["primary_positive_probability"] = primary_oof["probability"].astype("float64")
    out["primary_class_probability"] = np.maximum(
        out["primary_positive_probability"],
        1.0 - out["primary_positive_probability"],
    )
    out["meta_label"] = (
            out["primary_prediction"] * out["raw_return"] > 0
    ).astype("int8")

    return out


def validate_prediction_partitions(frame: pd.DataFrame, fold_columns: Sequence[str]) -> None:
    """Validate prediction partition and fold metadata.

    Args:
        frame: Predictions with partition and fold metadata.
        fold_columns: Cross-validation fold column names.
    """
    if not frame["partition"].isin(["development", "holdout"]).all():
        raise ValueError("Predictions require development or holdout partitions")
    development = frame["partition"].eq("development")
    for column in fold_columns:
        folds = pd.to_numeric(frame.loc[development, column], errors="coerce")
        if not (folds.notna() & np.isfinite(folds) & folds.ge(0) & folds.eq(folds.round())).all():
            raise ValueError(f"Development {column} must contain nonnegative integer folds")
        if not frame.loc[~development, column].isna().all():
            raise ValueError(f"Holdout {column} must be missing")


def build_meta_development_frame(
        events: pd.DataFrame,
        primary_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Connect saved primary development OOF predictions to their events.

    Args:
        events: Prepared events indexed by symbol and event_start.
        primary_predictions: Indexed primary predictions with partition metadata.

    Returns:
        Meta development events in primary prediction order, with side, probability,
        confidence, and action labels from primary OOF predictions.
    """
    if events.index.has_duplicates or primary_predictions.index.has_duplicates:
        raise ValueError("Meta inputs require unique composite event keys")
    matched_events = events.loc[primary_predictions.index]
    if set(events.index) != set(primary_predictions.index):
        raise ValueError("Primary predictions must match event keys")
    if not primary_predictions.partition.equals(matched_events.partition):
        raise ValueError("Primary predictions must match event partitions")
    validate_prediction_partitions(primary_predictions, ("primary_cv_fold",))
    development_primary = primary_predictions[
        primary_predictions["partition"].eq("development")
    ]
    output = matched_events.loc[development_primary.index].copy()
    output[development_primary.columns] = development_primary
    output["meta_label"] = (
        output["primary_prediction"] * output["raw_return"] > 0
    ).astype("int8")
    return output


def get_meta_feature_columns(meta_frame: pd.DataFrame, model_kind: str = "sentiment") -> list[str]:
    """Identify meta-model feature columns.

    Args:
        meta_frame: Frame produced by ``build_meta_model_frame``.
        model_kind: Same input family as the primary model.

    Returns:
        Primary feature names followed by primary side and confidence.
    """
    missing = set(META_FEATURE_COLUMNS).difference(meta_frame.columns)
    if missing:
        raise ValueError(f"Missing required meta features: {sorted(missing)}")

    generated_columns = {
        *META_FEATURE_COLUMNS, "primary_positive_probability", "primary_negative_probability",
        "primary_cv_fold", "meta_label", "meta_prediction", "meta_pass_probability",
        "meta_cv_fold", "model_family",
    }
    primary_frame = meta_frame.drop(
        columns=generated_columns.intersection(meta_frame.columns),
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
        estimator: Classifier for out-of-fold predictions.
        features: Time-indexed feature matrix.
        labels: Labels aligned with ``features``.
        sample_weight: Observation weights.
        cv: Cross-validator yielding train and test positional indices.
        positive_label: Class whose probability is stored in the output.
        show_progress: Whether to display progress.
        _progress: Shared progress reporter.
        _progress_label: Label for the shared progress reporter.

    Returns:
        Predictions, positive-class probabilities, fold ids, and OOF provenance.
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
        sample_weight: Observation weights.
        class_labels: Ordered pair of labels used for log-loss probabilities.
        positive_label: Label represented by ``positive_probabilities``.

    Returns:
        Weighted classification metrics.
    """
    return ClassificationScores.score_binary_predictions(
        labels, predictions, positive_probabilities, sample_weight,
        class_labels=class_labels, positive_label=positive_label,
    )


def model_evaluation_columns(class_labels: Sequence[int]) -> list[str]:
    """Return metric keys in stage display order.

    Args:
        class_labels: Primary {-1, 1} or meta {0, 1} labels.

    Returns:
        Classification metric names in display order.
    """
    if set(class_labels) == {-1, 1}:
        return ["accuracy", "precision_-1", "recall_-1", "f1_-1",
                "precision", "recall", "f1", "log_loss", "pr_auc", "roc_auc"]
    return ["accuracy", "precision", "recall", "f1", "log_loss", "pr_auc", "roc_auc"]


def run_model_selection_workflow(
    features: pd.DataFrame,
    labels: pd.Series,
    sample_weight: pd.Series,
    information_sets: pd.Series,
    *,
    cv: int,
    pct_embargo: float,
    random_state: int,
    n_jobs: int,
    candidate_settings: dict[str, dict],
    parameter_grids: dict[str, dict[str, list]],
    artifact_dir: Path,
    primary_model_hash: str | None = None,
    show_progress: bool = False,
) -> ModelSearchResult:
    """Evaluate and save model configurations.

    Args:
        features: Development features, excluding holdout rows.
        labels: Primary {-1, 1} or meta {0, 1} labels.
        sample_weight: Observation weights.
        information_sets: Aligned event end times for purging.
        cv: Number of purged folds.
        pct_embargo: Embargo fraction.
        random_state: Classifier seed.
        n_jobs: Ensemble workers.
        candidate_settings: Notebook-owned constructor settings.
        parameter_grids: Notebook-owned grids for every family.
        artifact_dir: Model search artifact directory.
        primary_model_hash: Primary provenance for meta configurations.
        show_progress: Whether to display progress.

    Returns:
        Configuration scores, input metadata, and artifact directory.
    """
    class_labels = np.sort(pd.unique(labels)).tolist()
    if class_labels not in ([-1, 1], [0, 1]):
        raise ValueError("labels must use {-1, 1} or {0, 1}")
    bases = build_candidate_classifiers(
        random_state=random_state, n_jobs=n_jobs, candidate_settings=candidate_settings,
    )
    artifact_dir = Path(artifact_dir).resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    splitter = PurgedKFold(cv, t1=information_sets, pct_embargo=pct_embargo)
    rows = []
    total = sum(len(ParameterGrid(parameter_grids[name])) for name in bases) * (cv + 1)
    with tqdm(total=total, desc="Ensemble search", disable=not show_progress) as progress:
        for name, base in bases.items():
            for configuration in ParameterGrid(parameter_grids[name]):
                estimator = clone(base).set_params(**configuration)
                identity = json.dumps({
                    "candidate": name, "configuration": configuration,
                    "settings": candidate_settings[name], "random_state": random_state,
                }, sort_keys=True)
                configuration_id = name + "_" + hashlib.sha256(identity.encode()).hexdigest()[:16]
                predictions = generate_oof_predictions(
                    estimator, features, labels, sample_weight, splitter,
                    positive_label=1, _progress=progress,
                    _progress_label=f"{configuration_id}: {configuration}",
                )
                scores = score_binary_predictions(
                    labels, predictions["prediction"], predictions["probability"],
                    sample_weight, class_labels=class_labels, positive_label=1,
                )
                fitted = clone(estimator).fit(
                    features, labels, sample_weight=sample_weight.to_numpy(),
                )
                joblib.dump({
                    "configuration_id": configuration_id, "candidate": name,
                    "configuration": dict(configuration), "estimator": fitted,
                    "oof_predictions": predictions, "scores": scores,
                    "feature_columns": list(features.columns), "random_state": random_state,
                    "cv": cv, "pct_embargo": pct_embargo,
                    "candidate_settings": candidate_settings[name],
                    "primary_model_hash": primary_model_hash,
                }, artifact_dir / f"{configuration_id}.joblib")
                progress.update(1)
                rows.append({"configuration_id": configuration_id,
                             "candidate": name, "configuration": dict(configuration),
                             **configuration, **scores})
    result = ModelSearchResult(
        tuning=pd.DataFrame(rows), artifact_dir=artifact_dir,
        data_hash=_modeling_data_hash(features, labels, sample_weight, information_sets),
        primary_model_hash=primary_model_hash,
    )
    joblib.dump(result, artifact_dir / "search.joblib")
    return result


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
    configuration_id: str | None = None,
    fitted_estimator: BaseEstimator | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Fit and save the final primary model and its OOF/holdout predictions.

    Args:
        development: Prepared development events.
        holdout: Prepared holdout events.
        estimator: Selected classifier configuration.
        final_oof: Development OOF predictions aligned with the development frame.
        feature_columns: Ordered features used for fitting and holdout inference.
        artifact_dir: Existing directory for the model and prediction files.
        selected_candidate: Candidate name persisted with the fitted model.
        best_configuration: Tuned parameters persisted with the fitted model.
        random_state: Research seed recorded in the model artifact.
        configuration_id: Explicit selection ID persisted for downstream workflows.
        fitted_estimator: Fitted development classifier.
        show_progress: Whether to display progress.

    Returns:
        Combined development and holdout predictions.
    """
    if not final_oof.index.equals(development.index) or not final_oof["prediction_source"].eq("oof").all():
        raise ValueError("Primary OOF predictions must match development events")
    if not development.partition.eq("development").all() or not holdout.partition.eq("holdout").all():
        raise ValueError("Primary fitting requires fixed development and holdout partitions")
    validate_prediction_partitions(final_oof.assign(partition="development"), ("fold",))
    with tqdm(total=4, desc="Final primary model", disable=not show_progress) as progress:
        progress.set_postfix_str("Reuse saved model" if fitted_estimator is not None else "Fit development data")
        final_primary = fitted_estimator if fitted_estimator is not None else clone(estimator).fit(
            development[feature_columns],
            development["primary_label"].astype("int8"),
            sample_weight=development["sample_weight"].astype("float64").to_numpy(),
        )
        progress.update(1)
        progress.set_postfix_str("Predict holdout")
        holdout_probability = final_primary.predict_proba(
            holdout[feature_columns]
        )[:, list(final_primary.classes_).index(1)]
        holdout_side = final_primary.predict(holdout[feature_columns]).astype("int8")
        progress.update(1)

        primary_oof_output = development.loc[:, list(PRIMARY_EVENT_COLUMNS)].copy()
        primary_oof_output["primary_prediction"] = final_oof["prediction"].astype("int8")
        primary_oof_output["primary_positive_probability"] = final_oof["probability"]
        primary_oof_output["primary_negative_probability"] = 1.0 - final_oof["probability"]
        primary_oof_output["primary_class_probability"] = np.maximum(
            final_oof["probability"], 1.0 - final_oof["probability"],
        )
        primary_oof_output["primary_cv_fold"] = final_oof["fold"]

        primary_holdout_output = holdout.loc[:, list(PRIMARY_EVENT_COLUMNS)].copy()
        primary_holdout_output["primary_prediction"] = holdout_side
        primary_holdout_output["primary_positive_probability"] = holdout_probability
        primary_holdout_output["primary_negative_probability"] = 1.0 - holdout_probability
        primary_holdout_output["primary_class_probability"] = np.maximum(
            holdout_probability, 1.0 - holdout_probability,
        )
        primary_holdout_output["primary_cv_fold"] = pd.NA

        primary_predictions = pd.concat([
            primary_oof_output,
            primary_holdout_output,
        ]).sort_index(level=["event_start", "symbol"])
        progress.set_postfix_str("Save predictions")
        primary_predictions.reset_index().to_parquet(artifact_dir / "primary_predictions.parquet", index=False)
        progress.update(1)
        progress.set_postfix_str("Save model")
        joblib.dump({
            **({"configuration_id": configuration_id} if configuration_id is not None else {}),
            "estimator": final_primary,
            "feature_columns": feature_columns,
            "selected_candidate": selected_candidate,
            "best_configuration": best_configuration,
            "holdout_boundary": holdout.index.get_level_values("event_start").min(),
            "random_state": random_state,
        }, artifact_dir / "primary_model.joblib")
        progress.update(1)

        return primary_predictions


def model_artifact_hash(path: Path) -> str:
    """Compute the hash of a saved model artifact.

    Args:
        path: Artifact path.

    Returns:
        SHA-256 digest of the model artifact.
    """
    with path.open("rb") as artifact:
        return hashlib.file_digest(artifact, "sha256").hexdigest()


def finalize_meta_model(
    meta_development: pd.DataFrame,
    events: pd.DataFrame,
    primary_predictions: pd.DataFrame,
    *,
    estimator: BaseEstimator,
    final_oof: pd.DataFrame,
    best_configuration: dict[str, object],
    feature_columns: Sequence[str],
    artifact_dir: Path,
    selected_candidate: str,
    random_state: int,
    configuration_id: str | None = None,
    fitted_estimator: BaseEstimator | None = None,
    show_progress: bool = False,
) -> pd.DataFrame:
    """Save the explicitly selected meta model with its primary provenance.

    Args:
        meta_development: Development events with primary OOF features and meta-labels.
        events: Prepared development and holdout events.
        primary_predictions: Primary OOF and holdout predictions with partition metadata.
        estimator: Selected classifier configuration.
        final_oof: Meta OOF predictions aligned with the development frame.
        best_configuration: Selected parameters persisted with the model.
        feature_columns: Ordered features for training and holdout inference.
        artifact_dir: Existing directory containing the primary model artifact.
        selected_candidate: Selected family name persisted with the model and predictions.
        random_state: Research seed recorded in the model artifact.
        configuration_id: Explicit selection ID persisted for downstream workflows.
        fitted_estimator: Fitted development classifier.
        show_progress: Whether to display progress.

    Returns:
        Combined development and holdout meta predictions.
    """
    if not meta_development.partition.eq("development").all():
        raise ValueError("Meta fitting requires development observations only")
    if not final_oof.index.equals(meta_development.index) or not final_oof.prediction_source.eq("oof").all():
        raise ValueError("Meta OOF predictions must match development events")
    validate_prediction_partitions(final_oof.assign(partition="development"), ("fold",))
    with tqdm(total=2, desc=f"Final meta model: {selected_candidate}", disable=not show_progress) as progress:
        progress.set_postfix_str("Reuse saved model" if fitted_estimator is not None else "Fit development data")
        final_meta = fitted_estimator if fitted_estimator is not None else clone(estimator).fit(
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
        holdout_events["primary_prediction"] = holdout_primary["primary_prediction"].astype("int8")
        holdout_events["primary_positive_probability"] = holdout_primary["primary_positive_probability"]
        holdout_events["primary_class_probability"] = holdout_primary["primary_class_probability"]
        holdout_events["meta_label"] = (
            holdout_events["primary_prediction"] * holdout_events["raw_return"] > 0
        ).astype("int8")

        holdout_probability = final_meta.predict_proba(
            holdout_events[feature_columns]
        )[:, list(final_meta.classes_).index(1)]
        holdout_action = final_meta.predict(holdout_events[feature_columns]).astype("int8")
        progress.update(1)

        meta_oof_output = primary_predictions.loc[meta_development.index].copy()
        meta_oof_output["meta_label"] = meta_development["meta_label"]
        meta_oof_output["meta_prediction"] = final_oof["prediction"].astype("int8")
        meta_oof_output["meta_pass_probability"] = final_oof["probability"]
        meta_oof_output["meta_cv_fold"] = final_oof["fold"]

        meta_holdout_output = holdout_primary.copy()
        meta_holdout_output["meta_label"] = holdout_events["meta_label"]
        meta_holdout_output["meta_prediction"] = holdout_action
        meta_holdout_output["meta_pass_probability"] = holdout_probability
        meta_holdout_output["meta_cv_fold"] = pd.NA

        meta_predictions = pd.concat([
            meta_oof_output,
            meta_holdout_output,
        ]).sort_index(level=["event_start", "symbol"])

    output = meta_predictions.assign(model_family=selected_candidate)
    joblib.dump({
        **({"configuration_id": configuration_id} if configuration_id is not None else {}),
        "models": {selected_candidate: {
            **({"configuration_id": configuration_id} if configuration_id is not None else {}),
            "estimator": final_meta, "feature_columns": list(feature_columns),
            "best_configuration": best_configuration, "random_state": random_state,
        }},
        "selected_candidate": selected_candidate,
        "primary_model_hash": model_artifact_hash(artifact_dir / "primary_model.joblib"),
    }, artifact_dir / "meta_models.joblib")
    output.reset_index().to_parquet(artifact_dir / "meta_predictions.parquet", index=False)
    return output


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
        estimator: Classifier for learning-curve evaluation.
        features: Time-indexed feature matrix.
        labels: Binary labels aligned with ``features``.
        sample_weight: Observation weights.
        cv: Cross-validator defining the purged train and validation folds.
        train_sizes: Fractions of each purged training fold to fit.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by the positive probability.
        scoring: Learning-curve scoring metric.
        random_state: Seed used for stratified training subsets.
        show_progress: Whether to display progress.
        _progress: Shared progress reporter.
        _progress_label: Label for the shared progress reporter.

    Returns:
        Training and validation error estimates by training fraction.
    """
    if not features.index.equals(labels.index):
        raise ValueError("features and labels must have the same index")
    if not features.index.equals(sample_weight.index):
        raise ValueError("features and sample_weight must have the same index")
    if scoring not in {"neg_log_loss", "f1", "precision"}:
        raise ValueError("scoring must be 'neg_log_loss', 'f1', or 'precision'.")
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
                        else 1.0 - scores[scoring]
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
        sample_weight: Observation weights.
        cv: Cross-validator shared by each estimator.
        train_sizes: Fractions passed to ``get_weighted_learning_curve``.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.
        scoring: Learning-curve scoring rule.
        random_state: Seed used for stratified training subsets.
        show_progress: Whether to display progress.

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
                "Weighted log loss" if scoring == "neg_log_loss" else
                "1 − weighted precision" if scoring == "precision" else
                "1 − weighted F1"
            )
            axis.grid(alpha=0.25)
            axis.legend()
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
        sample_weight: Observation weights.
        information_sets: Event end times used by purged validation.
        scoring: Importance scoring rule.
        cv: Number of purged validation folds.
        pct_embargo: Embargo fraction applied to each fold.
        random_state: Seed used by feature permutations.
        show_progress: Whether to display progress.

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
        sample_weight: Observation weights.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.

    Returns:
        Classification metrics indexed by model name.
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
    columns = model_evaluation_columns(class_labels)
    return pd.DataFrame(rows).set_index("model")[columns].rename(
        index=lambda name: name.replace("_", " ").title(),
        columns={metric: f"{metric}_1" for metric in ("precision", "recall", "f1")}
        if set(class_labels) == {-1, 1} else {},
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
        sample_weight: Observation weights.
        heading: Figure title.
        class_labels: Ordered binary label pair.
        positive_label: Label represented by predicted probabilities.
    """
    model_count = len(predictions_by_model)
    fig, axes = plt.subplots(1, model_count, figsize=(5 * model_count, 5.5))
    axes = np.atleast_1d(axes).ravel()
    for axis, (name, predictions) in zip(axes, predictions_by_model.items()):
        matrix = ClassificationScores.confusion_matrix(
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
    fig.suptitle(f"{heading}: Weighted Confusion Matrices", fontsize=14)
    fig.tight_layout()
    plt.show()

    fig, (pr_axis, roc_axis) = plt.subplots(1, 2, figsize=(14, 5))
    for name, predictions in predictions_by_model.items():
        probability = predictions["probability"]
        precision, recall, _ = ClassificationScores.precision_recall_curve(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        average_precision = ClassificationScores.average_precision(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        pr_auc = ClassificationScores.pr_auc(
            observed, probability, pos_label=positive_label, sample_weight=sample_weight,
        )
        display_name = name.replace("_", " ").title()
        pr_axis.plot(
            recall,
            precision,
            label=f"{display_name} (PR-AUC={pr_auc:.3f}; AP={average_precision:.3f})",
        )

        false_positive_rate, true_positive_rate, _ = ClassificationScores.roc_curve(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        roc_auc = ClassificationScores.roc_auc(
            observed,
            probability,
            pos_label=positive_label,
            sample_weight=sample_weight,
        )
        roc_axis.plot(
            false_positive_rate,
            true_positive_rate,
            label=f"{display_name} (ROC-AUC={roc_auc:.3f})",
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
