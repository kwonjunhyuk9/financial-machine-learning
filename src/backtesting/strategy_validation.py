from __future__ import annotations

from itertools import combinations
from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone

from src.modeling.model_workflow import build_meta_model_frame, generate_oof_predictions

from src.modeling.purged_validation import (
    PurgedKFold,
    _embargo_train_indices,
    _purge_train_indices,
    _validate_event_intervals,
    time_groups,
)


def combinatorial_purged_cross_validation(
    samples_info_sets: pd.Series,
    num_groups: int,
    num_test_groups: int,
    pct_embargo: float = 0.0,
) -> pd.DataFrame:
    """Generate combinatorial purged cross-validation split metadata.

    Args:
        samples_info_sets: Series indexed by observation start time and valued by
            observation end time.
        num_groups: Number of contiguous groups used to partition the sorted
            observations.
        num_test_groups: Number of groups used as the test set in each split.
        pct_embargo: Finite fraction in [0, 1) of all observations to embargo,
            rounded up, strictly after each contiguous test run's latest event
            end. Adjacent test groups form one run.

    Returns:
        A frame with one row per test-group combination and split, group, and
        positional-index metadata.

    Raises:
        ValueError: If inputs are invalid or cannot form the requested CPCV splits.
    """
    samples_info_sets = _validate_samples_info_sets(samples_info_sets)

    if num_groups < 2:
        raise ValueError("num_groups must be greater than 1")

    if num_groups > samples_info_sets.shape[0]:
        raise ValueError("num_groups cannot exceed the number of observations")

    if num_test_groups < 1 or num_test_groups >= num_groups:
        raise ValueError("num_test_groups must be in [1, num_groups)")

    groups = time_groups(samples_info_sets.index, num_groups)
    out = []

    for split_num, test_groups in enumerate(
            combinations(range(num_groups), num_test_groups)
    ):
        test_indices = np.concatenate([
            groups[group]
            for group in test_groups
        ])
        train_indices = np.setdiff1d(
            np.arange(samples_info_sets.shape[0]),
            test_indices,
            assume_unique=True
        )
        train_indices = _purge_train_indices(
            samples_info_sets=samples_info_sets,
            train_indices=train_indices,
            test_indices=test_indices
        )
        train_indices = _embargo_train_indices(
            samples_info_sets=samples_info_sets,
            train_indices=train_indices,
            test_indices=test_indices,
            pct_embargo=pct_embargo
        )

        out.append({
            "split_num": split_num,
            "train_groups": tuple(
                group
                for group in range(num_groups)
                if group not in test_groups
            ),
            "test_groups": test_groups,
            "train_indices": tuple(train_indices.tolist()),
            "test_indices": tuple(test_indices.tolist())
        })

    return pd.DataFrame(out)


def get_combinatorial_backtest_paths(
    splits: pd.DataFrame,
    num_groups: int,
) -> pd.DataFrame:
    """Arrange CPCV split identifiers into combinatorial backtest paths.

    Args:
        splits: Output from ``combinatorial_purged_cross_validation``.
        num_groups: Number of contiguous observation groups used to make the
            supplied CPCV splits.

    Returns:
        A frame indexed by group number with one column per CPCV backtest path.

    Raises:
        ValueError: If split metadata cannot form complete paths.
    """
    splits = _validate_splits(splits=splits, num_groups=num_groups)
    num_test_groups = len(splits.iloc[0]["test_groups"])
    num_paths = comb(num_groups - 1, num_test_groups - 1)

    path_counts = pd.Series(0, index=range(num_groups))
    paths = pd.DataFrame(
        np.nan,
        index=pd.Index(range(num_groups), name="group"),
        columns=[f"path_{path}" for path in range(num_paths)]
    )

    for _, split in splits.iterrows():
        for group in split["test_groups"]:
            path_num = path_counts.loc[group]

            if path_num >= num_paths:
                raise ValueError("splits cannot form complete backtest paths")

            paths.loc[group, f"path_{path_num}"] = split["split_num"]
            path_counts.loc[group] += 1

    if paths.shape[1] != num_paths or paths.isna().any().any():
        raise ValueError("splits cannot form complete backtest paths")

    return paths.astype("int64")


def generate_cpcv_predictions(
    development: pd.DataFrame,
    splits: pd.DataFrame,
    primary_artifact: dict[str, object],
    meta_artifact: dict[str, object],
    split_calibrations: pd.DataFrame,
    *,
    prediction_path: Path,
    inner_cv_splits: int,
    inner_pct_embargo: float,
) -> pd.DataFrame:
    """Refit primary/meta models within CPCV splits and save test predictions.

    Args:
        development: Development-only events with a composite event index and
            observed entry prices; positional indices match the supplied splits.
        splits: CPCV split metadata containing train_indices and test_indices.
        primary_artifact: Frozen primary estimator and ordered feature_columns.
        meta_artifact: Frozen meta estimator and ordered feature_columns.
        split_calibrations: Training-only price sizing keyed by split_num/symbol.
        prediction_path: Destination Parquet file, preserving the composite index.
        inner_cv_splits: Purged fold count for primary OOF meta-training inputs.
        inner_pct_embargo: Embargo fraction for the inner purged folds.

    Returns:
        Test-only predictions ordered by split_num and observation_position,
        including per-split calibration and the existing prediction schema.

    Raises:
        ValueError: If events include a partition other than development.
    """
    if not development["partition"].eq("development").all():
        raise ValueError("CPCV predictions require development observations only")
    primary_features = primary_artifact["feature_columns"]
    meta_features = meta_artifact["feature_columns"]
    split_predictions = []
    for split in splits.itertuples(index=False):
        train = development.iloc[list(split.train_indices)].copy()
        test = development.iloc[list(split.test_indices)].copy()

        primary_train_estimator = clone(primary_artifact["estimator"])
        inner_cv = PurgedKFold(n_splits=inner_cv_splits, t1=train["event_end"], pct_embargo=inner_pct_embargo)
        primary_train_oof = generate_oof_predictions(
            primary_train_estimator,
            train[primary_features],
            train["direction_label"].astype("int8"),
            train["sample_weight"],
            inner_cv,
            positive_label=1,
        )
        meta_train = build_meta_model_frame(
            train,
            primary_train_oof[["prediction", "probability", "prediction_source"]],
        )

        fitted_primary = clone(primary_artifact["estimator"]).fit(
            train[primary_features],
            train["direction_label"].astype("int8"),
            sample_weight=train["sample_weight"].to_numpy(),
        )
        primary_probability = fitted_primary.predict_proba(test[primary_features])[:, list(fitted_primary.classes_).index(1)]
        primary_side = fitted_primary.predict(test[primary_features]).astype("int8")

        test_meta = test.copy()
        test_meta["primary_side"] = primary_side
        test_meta["primary_probability"] = primary_probability
        test_meta["primary_confidence"] = np.maximum(primary_probability, 1.0 - primary_probability)

        fitted_meta = clone(meta_artifact["estimator"]).fit(
            meta_train[meta_features],
            meta_train["meta_label"].astype("int8"),
            sample_weight=meta_train["sample_weight"].to_numpy(),
        )
        meta_probability = fitted_meta.predict_proba(test_meta[meta_features])[:, list(fitted_meta.classes_).index(1)]
        meta_action = fitted_meta.predict(test_meta[meta_features]).astype("int8")

        result = test_meta[[
            "event_end", "vertical_barrier", "target_return", "raw_return",
            "direction_label", "sample_weight", "partition",
            "holdout_boundary", "mean_sentiment_score",
        ]].copy()
        result["primary_side"] = primary_side
        result["primary_probability"] = primary_probability
        result["meta_probability"] = meta_probability
        result["meta_action"] = meta_action
        result["split_num"] = split.split_num
        result["observation_position"] = list(split.test_indices)
        calibration = split_calibrations.loc[
            split_calibrations["split_num"].eq(split.split_num)
        ].set_index("symbol")
        symbols = result.index.get_level_values("symbol")
        result["w"] = symbols.map(calibration["w"])
        result["calibration_reason"] = symbols.map(calibration["reason"])
        split_predictions.append(result)

    cpcv_predictions = pd.concat(split_predictions).sort_values(["split_num", "observation_position"])
    cpcv_predictions.to_parquet(prediction_path)
    return cpcv_predictions


def get_cpcv_price_calibrations(
    development: pd.DataFrame,
    splits: pd.DataFrame,
) -> pd.DataFrame:
    """Calibrate every symbol from each CPCV split's training observations.

    Returns one row per split and symbol with ``w`` and an exclusion reason.
    Test observations never contribute to their split's calibration.
    """
    from src.backtesting.portfolio_management import calibrate_price_sizing

    frame = (
        development.reset_index()
        if "symbol" not in development.columns
        else development
    )
    required = {"symbol", "entry_price", "target_return", "partition"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"development is missing columns: {sorted(missing)}")
    if not frame["partition"].eq("development").all():
        raise ValueError("CPCV calibration requires development observations only")
    if not {"split_num", "train_indices"}.issubset(splits.columns):
        raise ValueError("splits must contain split_num and train_indices columns")

    symbols = pd.Index(sorted(frame["symbol"].unique()), name="symbol")
    tables = []
    for split in splits.itertuples(index=False):
        training = frame.iloc[list(split.train_indices)]
        calibration = calibrate_price_sizing(training).set_index("symbol").reindex(symbols)
        calibration["reason"] = calibration["reason"].fillna("no split training events")
        calibration["split_num"] = split.split_num
        tables.append(calibration.reset_index())

    return pd.concat(tables, ignore_index=True)[
        ["split_num", "symbol", "w", "reason"]
    ]


def assemble_cpcv_path_predictions(
    split_predictions: pd.DataFrame,
    splits: pd.DataFrame,
    observation_index: pd.Index,
    num_groups: int,
) -> dict[str, pd.DataFrame]:
    """Assemble one complete OOS prediction frame per CPCV path.

    Every returned path contains each observation position exactly once, using
    the split assigned to that path and observation group.
    """
    required = {"split_num", "observation_position"}
    if not required.issubset(split_predictions.columns):
        raise ValueError(
            "split_predictions must contain split_num and observation_position columns"
        )
    if len(observation_index) == 0:
        raise ValueError("observation_index must not be empty")

    assignments = get_combinatorial_backtest_paths(splits, num_groups)
    groups = time_groups(observation_index, num_groups)
    expected = set(range(len(observation_index)))
    paths = {}

    for path_name in assignments.columns:
        segments = []
        for group, positions in enumerate(groups):
            split_num = assignments.loc[group, path_name]
            segment = split_predictions.loc[
                split_predictions["split_num"].eq(split_num)
                & split_predictions["observation_position"].isin(positions)
            ]
            if set(segment["observation_position"]) != set(positions):
                raise ValueError("split predictions do not cover a CPCV path group")
            segments.append(segment)

        path = pd.concat(segments).sort_values("observation_position", kind="stable")
        positions = path["observation_position"]
        if positions.duplicated().any() or set(positions) != expected:
            raise ValueError("CPCV path must contain every observation exactly once")
        paths[path_name] = path

    return paths


def _validate_samples_info_sets(samples_info_sets):
    """Validate and sort event information intervals.

    Args:
        samples_info_sets: Candidate event interval series.

    Returns:
        The same series sorted by start time.

    Raises:
        ValueError: If event intervals are missing, duplicated, or inconsistent.
    """
    _validate_event_intervals(samples_info_sets)
    return samples_info_sets


def _validate_splits(splits, num_groups):
    """Validate CPCV split metadata before constructing backtest paths.

    Args:
        splits: Candidate CPCV split metadata frame.
        num_groups: Number of valid group identifiers, covering ``0`` through ``num_groups -
        1``.

    Returns:
        The original ``splits`` frame when it contains valid path metadata.

    Raises:
        ValueError: If split metadata is missing or references invalid groups.
    """
    required_columns = {"split_num", "test_groups"}

    if not isinstance(splits, pd.DataFrame):
        raise ValueError("splits must be a pd.DataFrame")

    if not required_columns.issubset(splits.columns):
        raise ValueError("splits must contain split_num and test_groups columns")

    if splits.empty:
        raise ValueError("splits must not be empty")

    test_group_sizes = splits["test_groups"].map(len)

    if test_group_sizes.nunique() != 1:
        raise ValueError("all splits must use the same number of test groups")

    for test_groups in splits["test_groups"]:
        if len(set(test_groups)) != len(test_groups):
            raise ValueError("test_groups must not contain duplicate groups")

        if min(test_groups) < 0 or max(test_groups) >= num_groups:
            raise ValueError("test_groups contains a group outside [0, num_groups)")

    return splits
