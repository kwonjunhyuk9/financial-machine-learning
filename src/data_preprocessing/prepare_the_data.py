"""Clean weighted model events without transforming tree-model features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data_preprocessing.event_weights import (
    WEIGHT_COLUMNS,
    build_partitioned_event_weights,
)


SENTIMENT_FEATURE_COLUMNS = ["mean_sentiment_score"]
FRACTIONAL_FEATURE_COLUMNS = ["fractionally_differenced_log_close"]
EVENT_METADATA_COLUMNS = {
    "event_start",
    "symbol",
    "event_end",
    "vertical_barrier",
    "target_return",
    "raw_return",
    "direction_label",
    *WEIGHT_COLUMNS,
}


def get_event_feature_groups(events: pd.DataFrame) -> dict[str, list[str]]:
    """Return the fixed sentiment, fractional-price, and technical groups."""
    required = set(SENTIMENT_FEATURE_COLUMNS + FRACTIONAL_FEATURE_COLUMNS)
    missing = required.difference(events.columns)
    if missing:
        raise ValueError(f"Model data is missing features: {sorted(missing)}")

    feature_columns = [
        column for column in events.columns if column not in EVENT_METADATA_COLUMNS
    ]
    technical_columns = [
        column
        for column in feature_columns
        if column not in required
    ]
    if len(feature_columns) != 53 or len(technical_columns) != 51:
        raise ValueError(
            "Model data must contain one sentiment, one fractional-price, "
            "and 51 technical features."
        )
    return {
        "sentiment": SENTIMENT_FEATURE_COLUMNS.copy(),
        "fractional_price": FRACTIONAL_FEATURE_COLUMNS.copy(),
        "technical": technical_columns,
    }


def prepare_weighted_event_data(
    weighted_events: pd.DataFrame,
    partition_manifest: pd.DataFrame,
    close: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove events with invalid features and refresh weights when necessary.

    The same predeclared completeness rule is applied to development and final
    holdout rows. No feature is imputed or removed from the persisted schema.
    """
    missing_weights = set(WEIGHT_COLUMNS).difference(weighted_events.columns)
    if missing_weights:
        raise ValueError(
            f"Weighted events are missing columns: {sorted(missing_weights)}"
        )
    required_manifest = {"event_start", "partition", "holdout_boundary"}
    missing_manifest = required_manifest.difference(partition_manifest.columns)
    if missing_manifest:
        raise ValueError(
            f"Partition manifest is missing columns: {sorted(missing_manifest)}"
        )

    events = weighted_events.copy()
    events["event_start"] = pd.to_datetime(
        events["event_start"], utc=True, errors="coerce"
    )
    manifest = partition_manifest.copy()
    manifest["event_start"] = pd.to_datetime(
        manifest["event_start"], utc=True, errors="coerce"
    )
    manifest["holdout_boundary"] = pd.to_datetime(
        manifest["holdout_boundary"], utc=True, errors="coerce"
    )
    invalid_event_starts = (
        events["event_start"].isna().any()
        or events["event_start"].duplicated().any()
    )
    if invalid_event_starts:
        raise ValueError("Event starts must be unique valid timestamps.")
    invalid_manifest_starts = (
        manifest["event_start"].isna().any()
        or manifest["event_start"].duplicated().any()
    )
    if invalid_manifest_starts:
        raise ValueError("Manifest event starts must be unique valid timestamps.")
    if manifest["holdout_boundary"].isna().any():
        raise ValueError("Holdout boundary must contain valid timestamps.")
    if not set(events["event_start"]).issubset(set(manifest["event_start"])):
        raise ValueError("Every weighted event must appear in the manifest.")

    feature_groups = get_event_feature_groups(events)
    feature_columns = [
        *feature_groups["sentiment"],
        *feature_groups["fractional_price"],
        *feature_groups["technical"],
    ]
    numeric_features = events[feature_columns].apply(pd.to_numeric, errors="coerce")
    missing_mask = numeric_features.isna()
    infinite_mask = pd.DataFrame(
        np.isinf(numeric_features.to_numpy(dtype=float)),
        index=events.index,
        columns=feature_columns,
    )
    invalid_mask = missing_mask | infinite_mask
    invalid_rows = invalid_mask.any(axis=1)

    group_by_feature = {
        feature: group
        for group, columns in feature_groups.items()
        for feature in columns
    }
    cleaning_report = pd.DataFrame(
        {
            "feature": feature_columns,
            "feature_group": [group_by_feature[column] for column in feature_columns],
            "missing_values": [
                int(missing_mask[column].sum()) for column in feature_columns
            ],
            "infinite_values": [
                int(infinite_mask[column].sum()) for column in feature_columns
            ],
            "invalid_rows": [
                int(invalid_mask[column].sum()) for column in feature_columns
            ],
            "action": "drop_row",
        }
    )

    invalid_starts = set(events.loc[invalid_rows, "event_start"])
    prepared_manifest = manifest.copy()
    if invalid_starts:
        invalid_manifest_rows = prepared_manifest["event_start"].isin(invalid_starts)
        source_partitions = prepared_manifest.loc[invalid_manifest_rows, "partition"]
        if not source_partitions.isin(["development", "holdout"]).all():
            raise ValueError(
                "Only active development or holdout events may be cleaned."
            )
        prepared_manifest.loc[invalid_manifest_rows, "partition"] = (
            source_partitions.map(
                {
                    "development": "development_feature_missing",
                    "holdout": "holdout_feature_missing",
                }
            )
        )

    prepared = events.loc[~invalid_rows].copy()
    active_manifest_starts = set(
        prepared_manifest.loc[
            prepared_manifest["partition"].isin(["development", "holdout"]),
            "event_start",
        ]
    )
    if set(prepared["event_start"]) != active_manifest_starts:
        raise ValueError("Prepared events must match active manifest partitions.")

    if invalid_starts:
        prepared = build_partitioned_event_weights(
            prepared.drop(columns=WEIGHT_COLUMNS),
            prepared_manifest,
            close,
        )
    else:
        prepared = prepared.sort_values("event_start", ignore_index=True)

    if not np.isfinite(prepared[feature_columns].to_numpy(dtype=float)).all():
        raise ValueError("Prepared model features must be finite.")
    return (
        prepared,
        prepared_manifest.sort_values("event_start", ignore_index=True),
        cleaning_report,
    )
