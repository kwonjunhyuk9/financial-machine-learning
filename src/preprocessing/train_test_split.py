from __future__ import annotations

import numpy as np
import pandas as pd


_CANDIDATE_COLUMNS = ["event_start", "mean_sentiment_score"]
_MANIFEST_COLUMNS = ["event_start", "partition", "holdout_boundary"]


def build_event_candidates(
        news: pd.DataFrame,
        market_bar_ends: pd.Series | pd.DatetimeIndex,
) -> pd.DataFrame:
    """Align deduplicated news to completed market bars and aggregate events.

    Args:
        news: News rows with creation time, headline, and sentiment score.
        market_bar_ends: Completed market-bar timestamps.

    Returns:
        Candidate events ordered by ``event_start``.

    Raises:
        ValueError: If required inputs are missing or no news aligns to a bar.
    """
    required_columns = {"created_at", "headline", "sentiment_score"}
    missing_columns = required_columns.difference(news.columns)
    if missing_columns:
        raise ValueError(
            f"News data is missing columns: {sorted(missing_columns)}"
        )

    bar_ends = pd.DatetimeIndex(pd.to_datetime(market_bar_ends, utc=True))
    bar_ends = bar_ends.dropna().drop_duplicates().sort_values()
    if bar_ends.empty:
        raise ValueError("market_bar_ends must contain at least one timestamp.")

    aligned_news = news.copy()
    aligned_news["created_at"] = pd.to_datetime(
        aligned_news["created_at"],
        utc=True,
        errors="coerce",
    )
    aligned_news = (
        aligned_news.dropna(subset=["created_at"])
        .drop_duplicates(subset=["created_at", "headline"], keep="first")
        .sort_values("created_at", kind="stable")
    )

    positions = bar_ends.searchsorted(aligned_news["created_at"], side="left")
    valid = positions < len(bar_ends)
    aligned_news = aligned_news.loc[valid].copy()
    if aligned_news.empty:
        raise ValueError("No news rows align to a completed market bar.")
    aligned_news["event_start"] = bar_ends[positions[valid]]

    candidates = (
        aligned_news.groupby("event_start", sort=True)
        .agg(
            mean_sentiment_score=("sentiment_score", "mean"),
        )
        .reset_index()
    )
    return candidates.loc[:, _CANDIDATE_COLUMNS]


def build_event_feature_schema(
        candidates: pd.DataFrame,
        fractional_features: pd.DataFrame,
        technical_features: pd.DataFrame,
) -> pd.DataFrame:
    """Combine point-in-time event features without dropping missing values.

    Args:
        candidates: Aggregated news candidates keyed by ``event_start``.
        fractional_features: Fractionally differentiated price keyed by ``end``.
        technical_features: Technical indicators keyed by ``end``.

    Returns:
        Candidate rows with symbol and all 53 model features.

    Raises:
        ValueError: If required columns, unique timestamps, or one symbol are absent.
    """
    candidate_columns = {
        "event_start",
        "mean_sentiment_score",
    }
    fractional_columns = {"end", "fractionally_differenced_log_close"}
    technical_identifier_columns = {"start", "end", "symbol"}

    missing_candidates = candidate_columns.difference(candidates.columns)
    missing_fractional = fractional_columns.difference(fractional_features.columns)
    missing_technical = technical_identifier_columns.difference(
        technical_features.columns
    )
    if missing_candidates:
        raise ValueError(
            f"Candidate data is missing columns: {sorted(missing_candidates)}"
        )
    if missing_fractional:
        raise ValueError(
            f"Fractional features are missing columns: {sorted(missing_fractional)}"
        )
    if missing_technical:
        raise ValueError(
            f"Technical features are missing columns: {sorted(missing_technical)}"
        )

    candidate_frame = candidates.loc[:, _CANDIDATE_COLUMNS].copy()
    candidate_frame["event_start"] = pd.to_datetime(
        candidate_frame["event_start"],
        utc=True,
        errors="coerce",
    )
    fractional_frame = fractional_features.loc[:, sorted(fractional_columns)].copy()
    fractional_frame["end"] = pd.to_datetime(
        fractional_frame["end"],
        utc=True,
        errors="coerce",
    )
    technical_frame = technical_features.copy()
    technical_frame["end"] = pd.to_datetime(
        technical_frame["end"],
        utc=True,
        errors="coerce",
    )

    if candidate_frame["event_start"].isna().any():
        raise ValueError("event_start must contain valid timestamps.")
    if fractional_frame["end"].isna().any() or technical_frame["end"].isna().any():
        raise ValueError("Feature end times must contain valid timestamps.")
    if candidate_frame["event_start"].duplicated().any():
        raise ValueError("Candidate event_start must be unique.")
    if fractional_frame["end"].duplicated().any():
        raise ValueError("Fractional feature end times must be unique.")
    if technical_frame["end"].duplicated().any():
        raise ValueError("Technical feature end times must be unique.")

    symbols = technical_frame["symbol"].dropna().astype(str).unique()
    if len(symbols) != 1:
        raise ValueError("Technical features must contain exactly one symbol.")
    technical_columns = [
        column
        for column in technical_frame.columns
        if column not in technical_identifier_columns
    ]
    if len(technical_columns) != 51:
        raise ValueError("Technical features must contain exactly 51 indicators.")

    fractional_frame = fractional_frame.rename(columns={"end": "event_start"})
    technical_frame = technical_frame.rename(columns={"end": "event_start"})
    schema = candidate_frame.merge(
        fractional_frame,
        on="event_start",
        how="left",
        validate="one_to_one",
    ).merge(
        technical_frame.loc[:, ["event_start", *technical_columns]],
        on="event_start",
        how="left",
        validate="one_to_one",
    )
    schema.insert(1, "symbol", symbols.item())
    feature_columns = [
        "mean_sentiment_score",
        "fractionally_differenced_log_close",
        *technical_columns,
    ]
    schema = schema.loc[
        :, ["event_start", "symbol", *feature_columns]
    ]
    return schema.sort_values("event_start", kind="stable").reset_index(drop=True)


def chronological_train_test_split(
    candidate_events: pd.DataFrame,
    test_size: float = 0.20,
    holdout_boundary: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split ordered candidate events into development and final holdout sets.

    Args:
        candidate_events: Candidate rows with one unique ``event_start`` each.
        test_size: Fraction of the final chronological rows assigned to holdout.
        holdout_boundary: Immutable cutoff. Rows before it are development and
            rows at or after it are holdout. If omitted, ``test_size`` is used.

    Returns:
        Development candidates, holdout candidates, and the partition manifest.

    Raises:
        ValueError: If inputs cannot form two non-empty chronological partitions.
    """
    if holdout_boundary is None and not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if "event_start" not in candidate_events.columns:
        raise ValueError("candidate_events must contain event_start.")

    ordered = candidate_events.copy()
    ordered["event_start"] = pd.to_datetime(
        ordered["event_start"],
        utc=True,
        errors="coerce",
    )
    if ordered["event_start"].isna().any():
        raise ValueError("event_start must contain valid timestamps.")
    if ordered["event_start"].duplicated().any():
        raise ValueError("event_start must be unique.")
    ordered = ordered.sort_values("event_start", kind="stable").reset_index(drop=True)

    if holdout_boundary is None:
        split_position = int(np.floor(len(ordered) * (1 - test_size)))
        if split_position == 0 or split_position == len(ordered):
            raise ValueError("test_size must leave both partitions non-empty.")
        development = ordered.iloc[:split_position].reset_index(drop=True)
        holdout = ordered.iloc[split_position:].reset_index(drop=True)
        boundary = holdout.loc[0, "event_start"]
    else:
        boundary = pd.to_datetime(holdout_boundary, utc=True, errors="coerce")
        if pd.isna(boundary):
            raise ValueError("holdout_boundary must be a valid timestamp.")
        development = ordered.loc[
            ordered["event_start"].lt(boundary)
        ].reset_index(drop=True)
        holdout = ordered.loc[
            ordered["event_start"].ge(boundary)
        ].reset_index(drop=True)
        if development.empty or holdout.empty:
            raise ValueError("holdout_boundary must leave both partitions non-empty.")

    manifest = pd.DataFrame(
        {
            "event_start": ordered["event_start"],
            "partition": ["development"] * len(development)
            + ["holdout"] * len(holdout),
            "holdout_boundary": boundary,
        }
    )
    return development, holdout, manifest.loc[:, _MANIFEST_COLUMNS]
