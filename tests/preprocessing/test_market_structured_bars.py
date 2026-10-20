import json

import pandas as pd
import pytest

from src.preprocessing import market_data

from src.preprocessing.market_data import ResearchPaths, save_frame
from src.preprocessing.market_structured_bars import (
    DOLLAR_BAR_COLUMNS, build_dollar_features, get_dollar_bars,
)


def _make_trades(num_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="min"),
            "price": [100.0 + idx for idx in range(num_rows)],
            "size": [1.0] * num_rows,
        }
    )


def test_get_dollar_bars_aggregates_each_threshold_window():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "price": [100.0, 100.0, 100.0, 100.0],
            "size": [1.0, 2.0, 1.0, 2.0],
        }
    )

    result = get_dollar_bars(trades, threshold=300.0)

    assert result.ohlcv["dollar_value"].tolist() == [300.0, 300.0]
    assert result.ohlcv["ticks"].tolist() == [2, 2]
    assert not {"buy_volume", "sell_volume"} & set(result.ohlcv.columns)
    assert "tick_sign" not in result.sample.columns


def test_threshold_bars_exclude_incomplete_trailing_window():
    result = get_dollar_bars(_make_trades(4), threshold=303.0)

    assert result.ohlcv["ticks"].tolist() == [3]


@pytest.mark.parametrize("threshold", [0, -1])
def test_dollar_bars_require_positive_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold must be positive"):
        get_dollar_bars(_make_trades(4), threshold=threshold)


def test_incomplete_bar_has_no_direction_columns():
    result = get_dollar_bars(_make_trades(1), threshold=1000.0)

    assert result.ohlcv.empty
    assert not {"buy_volume", "sell_volume"} & set(result.ohlcv.columns)
    assert "tick_sign" not in result.sample.columns


def test_dollar_bars_keep_ohlcv_for_varying_prices():
    trades = _make_trades(4)
    trades["price"] = [102.0, 98.0, 105.0, 101.0]
    trades["size"] = [1.0, 2.0, 3.0, 1.0]

    result = get_dollar_bars(trades, threshold=600.0)

    assert result.ohlcv[["open", "high", "low", "close", "volume"]].values.tolist() == [
        [102.0, 105.0, 98.0, 105.0, 6.0],
    ]
    assert result.ohlcv.start.tolist() == [pd.Timestamp("2026-01-01", tz="UTC")]
    assert result.ohlcv.index.tolist() == [pd.Timestamp("2026-01-01 00:02:00", tz="UTC")]


def test_saved_bars_preserve_timestamps_and_carry_unfinished_trades(tmp_path):
    paths = ResearchPaths(
        tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
        end=pd.Timestamp("2025-02-04", tz="UTC"),
    )
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    for day, offsets, sizes in [
        ("2025-02-01", [0, 1, 2], [1.0, 1.0, 0.5]),
        ("2025-02-02", [0, 0, 1, 2], [2.5, 0.5, 1.0, 1.0]),
        ("2025-02-03", [0, 1, 2], [1.0, 1.0, 1.0]),
    ]:
        raw = paths.raw("A", "tick") / f"{day}.parquet"
        save_frame(pd.DataFrame({
            "timestamp": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(offsets, unit="min"),
            "symbol": "A", "price": 100.0, "size": sizes,
        }), raw)
        raw.with_suffix(".json").write_text("{}")
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)

    report = build_dollar_features(paths, **kwargs)
    output = paths.feature("A", "dollar_bars")
    bars = pd.read_parquet(output)
    expected = pd.DataFrame({
        "start": pd.to_datetime(["2025-02-02", "2025-02-02 00:01:00"], utc=True, format="mixed"),
        "end": pd.to_datetime(["2025-02-02", "2025-02-03 00:02:00"], utc=True, format="mixed"),
        "symbol": ["A", "A"], "open": [100.0, 100.0], "high": [100.0, 100.0],
        "low": [100.0, 100.0], "close": [100.0, 100.0], "volume": [3.0, 5.0],
    })
    assert bars.columns.tolist() == DOLLAR_BAR_COLUMNS
    pd.testing.assert_frame_equal(bars, expected)
    assert report.unfinished_ticks.tolist() == [0]
    modified = output.stat().st_mtime_ns
    assert build_dollar_features(paths, **kwargs).cached_files.tolist() == [1]
    assert output.stat().st_mtime_ns == modified

    metadata = output.with_suffix(".json")
    identity = json.loads(metadata.read_text())
    assert identity["settings"].pop("columns") == DOLLAR_BAR_COLUMNS
    metadata.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="deliberately rebuild"):
        build_dollar_features(paths, **kwargs)

@pytest.mark.parametrize("row_count", [0, 3])
def test_cached_feature_report_counts_saved_rows(tmp_path, monkeypatch, row_count):
    paths = market_data.ResearchPaths(
        tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
        end=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    monkeypatch.setattr(market_data, "feature_identity", lambda *args, **kwargs: {"settings": {}})
    market_data.save_feature(
        pd.DataFrame({"value": range(row_count)}), paths.feature("A", "dollar_bars"), {"settings": {}},
    )
    report = build_dollar_features(
        paths, manifest_path=manifest, start=paths.start, end=paths.end,
        lookback_sessions=20, target_bars_per_session=390,
    )

    assert report.columns[:4].tolist() == ["symbol", "output_rows", "processed_files", "cached_files"]
    assert report.processed_files.tolist() == [0]
    assert report.cached_files.tolist() == [1]
    assert report.output_rows.tolist() == [row_count]


def test_dollar_report_rows_match_after_cache_reuse(tmp_path):
    paths = market_data.ResearchPaths(
        tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
        end=pd.Timestamp("2025-02-03", tz="UTC"),
    )
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    for day in ("2025-02-01", "2025-02-02"):
        raw = paths.raw("A", "tick") / f"{day}.parquet"
        market_data.save_frame(pd.DataFrame({
            "timestamp": pd.date_range(day, periods=3, freq="min", tz="UTC"),
            "symbol": "A", "price": 100.0, "size": 1.0,
        }), raw)
        raw.with_suffix(".json").write_text("{}")
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)

    processed = build_dollar_features(paths, **kwargs)
    cached = build_dollar_features(paths, **kwargs)

    assert processed.processed_files.tolist() == [1]
    assert processed.cached_files.tolist() == [0]
    assert cached.processed_files.tolist() == [0]
    assert cached.cached_files.tolist() == [1]
    assert processed.output_rows.tolist() == cached.output_rows.tolist() == [1]
