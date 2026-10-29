import json
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest

from src.preprocessing import market_data, market_structured_bars
from src.preprocessing.event_labeling import get_bins

from src.preprocessing.market_data import ResearchPaths, save_frame
from src.preprocessing.market_structured_bars import (
    DOLLAR_BAR_COLUMNS, build_dollar_features, build_symbol_dollar_features, get_dollar_bars,
)


def write_empty_splits(paths, symbol):
    from src.preprocessing.market_data import split_path

    path = split_path(paths, symbol)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"symbol": symbol,
                                "start": str(paths.start), "end": str(paths.end),
                                "reference": str(paths.start), "records": [], "events": []}))


def save_ticks(frame, path):
    save_frame(frame, path)
    path.with_suffix(".json").write_text(json.dumps({}))


def _make_trades(num_rows: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=num_rows, freq="min"),
            "symbol": "A",
            "price": [100.0 + idx for idx in range(num_rows)],
            "size": [1.0] * num_rows,
        }
    )


def test_get_dollar_bars_aggregates_each_threshold_window():
    trades = pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-01", periods=4, freq="min"),
            "symbol": "A",
            "price": [100.0, 100.0, 100.0, 100.0],
            "size": [1.0, 2.0, 1.0, 2.0],
        }
    )

    result = get_dollar_bars(trades, threshold=300.0)

    assert result["dollar_value"].tolist() == [300.0, 300.0]
    assert result["ticks"].tolist() == [2, 2]
    assert not {"buy_volume", "sell_volume"} & set(result.columns)


def test_threshold_bars_exclude_incomplete_trailing_window():
    result = get_dollar_bars(_make_trades(4), threshold=303.0)

    assert result["ticks"].tolist() == [3]


@pytest.mark.parametrize("threshold", [0, -1])
def test_dollar_bars_require_positive_threshold(threshold):
    with pytest.raises(ValueError, match="Threshold must be positive"):
        get_dollar_bars(_make_trades(4), threshold=threshold)


def test_incomplete_bar_has_no_direction_columns():
    result = get_dollar_bars(_make_trades(1), threshold=1000.0)

    assert result.empty
    pd.testing.assert_frame_equal(result, pd.DataFrame(
        columns=[*DOLLAR_BAR_COLUMNS, "dollar_value", "ticks"],
    ))
    assert not {"buy_volume", "sell_volume"} & set(result.columns)


def test_dollar_bars_keep_ohlcv_for_varying_prices():
    trades = _make_trades(4)
    trades["price"] = [102.0, 98.0, 105.0, 101.0]
    trades["size"] = [1.0, 2.0, 3.0, 1.0]

    result = get_dollar_bars(trades, threshold=600.0)

    assert result[["open", "high", "low", "close", "volume"]].values.tolist() == [
        [102.0, 105.0, 98.0, 105.0, 6.0],
    ]
    assert result.start.tolist() == [pd.Timestamp("2026-01-01", tz="UTC")]
    assert result.index.tolist() == [pd.Timestamp("2026-01-01 00:02:00", tz="UTC")]


def test_complete_timestamps_keeps_oversized_trade_groups_in_one_bar():
    trades = _make_trades(5)
    trades["timestamp"] = pd.to_datetime(
        ["2026-01-01T00:00Z"] * 3 + ["2026-01-01T00:01Z", "2026-01-01T00:02Z"],
        utc=True,
    )
    trades["price"] = [100., 110., 90., 105., 95.]

    bars = get_dollar_bars(trades, threshold=200., complete_timestamps=True)

    assert bars.index.is_unique
    assert bars.index.tolist() == trades.timestamp.iloc[[0, 4]].tolist()
    assert bars[["open", "high", "low", "close", "volume", "dollar_value", "ticks"]].values.tolist() == [
        [100., 110., 90., 90., 3., 300., 3],
        [105., 105., 95., 95., 2., 200., 2],
    ]


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
        save_ticks(pd.DataFrame({
            "timestamp": pd.Timestamp(day, tz="UTC") + pd.to_timedelta(offsets, unit="min"),
            "symbol": "A", "price": 100.0, "size": sizes,
        }), raw)
    write_empty_splits(paths, "A")
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)

    report = pd.DataFrame([build_symbol_dollar_features(paths, "A", **kwargs)])
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
    assert pd.DataFrame([build_symbol_dollar_features(paths, "A", **kwargs)]).cached_files.tolist() == [1]
    assert output.stat().st_mtime_ns == modified

    metadata = output.with_suffix(".json")
    identity = json.loads(metadata.read_text())
    assert identity["settings"].pop("columns") == DOLLAR_BAR_COLUMNS
    metadata.write_text(json.dumps(identity))
    with pytest.raises(ValueError, match="deliberately rebuild"):
        pd.DataFrame([build_symbol_dollar_features(paths, "A", **kwargs)])

@pytest.mark.parametrize("row_count", [0, 3])
def test_cached_feature_report_counts_saved_rows(tmp_path, monkeypatch, row_count):
    paths = market_data.ResearchPaths(
        tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
        end=pd.Timestamp("2026-01-01", tz="UTC"),
    )
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    monkeypatch.setattr(market_data, "feature_identity", lambda *args, **kwargs: {"settings": {}})
    write_empty_splits(paths, "A")
    market_data.save_feature(
        pd.DataFrame({"value": range(row_count)}), paths.feature("A", "dollar_bars"), {"settings": {}},
    )
    report = pd.DataFrame([build_symbol_dollar_features(
        paths, "A", manifest_path=manifest, start=paths.start, end=paths.end,
        lookback_sessions=20, target_bars_per_session=390,
    )])

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
        save_ticks(pd.DataFrame({
            "timestamp": pd.date_range(day, periods=3, freq="min", tz="UTC"),
            "symbol": "A", "price": 100.0, "size": 1.0,
        }), raw)
    write_empty_splits(paths, "A")
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)

    processed = pd.DataFrame([build_symbol_dollar_features(paths, "A", **kwargs)])
    cached = pd.DataFrame([build_symbol_dollar_features(paths, "A", **kwargs)])

    assert processed.processed_files.tolist() == [1]
    assert processed.cached_files.tolist() == [0]
    assert cached.processed_files.tolist() == [0]
    assert cached.cached_files.tolist() == [1]
    assert processed.output_rows.tolist() == cached.output_rows.tolist() == [1]


@pytest.mark.parametrize("listed_spy", [False, True])
def test_full_pipeline_adds_spy_once_and_reuses_standalone_cache(tmp_path, listed_spy):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
                          end=pd.Timestamp("2025-02-03", tz="UTC"))
    manifest = tmp_path / "universe.csv"
    pd.DataFrame({"symbol": ["A", "SPY"] if listed_spy else ["A"]}).to_csv(manifest, index=False)
    original_manifest = manifest.read_bytes()
    for symbol in ("A", "SPY"):
        write_empty_splits(paths, symbol)
        for day in ("2025-02-01", "2025-02-02"):
            raw = paths.raw(symbol, "tick") / f"{day}.parquet"
            save_ticks(pd.DataFrame({"timestamp": pd.date_range(day, periods=3, freq="min", tz="UTC"),
                                     "symbol": symbol, "price": 100., "size": 1.}), raw)
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)
    standalone = build_symbol_dollar_features(paths, "SPY", **kwargs)
    assert standalone["processed_files"] == 1
    spy_file = paths.feature("SPY", "dollar_bars")
    modified = spy_file.stat().st_mtime_ns
    report = build_dollar_features(paths, **kwargs)
    assert report.symbol.tolist() == ["A", "SPY"]
    assert report.cached_files.tolist() == [0, 1]
    assert spy_file.stat().st_mtime_ns == modified
    assert spy_file.parts[-4:-1] == ("etf", "SPY", "features")
    assert pd.read_parquet(spy_file).symbol.unique().tolist() == ["SPY"]
    assert manifest.read_bytes() == original_manifest
    assert build_dollar_features(paths, **kwargs).cached_files.tolist() == [1, 1]


@pytest.fixture
def paths(tmp_path):
    return market_data.ResearchPaths(tmp_path, start=pd.Timestamp("2025-02-01", tz="UTC"),
                                     end=pd.Timestamp("2025-04-01", tz="UTC"))


def trades(times, prices, sizes):
    return pd.DataFrame({"timestamp": pd.to_datetime(times, utc=True),
                         "symbol": "A", "price": prices, "size": sizes})


def test_split_adjustment_keeps_raw_values_dollars_and_boundaries():
    raw = trades(["2025-02-03T20:59Z", "2025-02-04T14:30Z", "2025-02-04T14:31Z"],
                 [1000., 100., 101.], [1., 10., 10.])
    original = raw.copy()
    adjusted = market_structured_bars.adjust_trade_units(
        raw, [{"effective_at": "2025-02-04T14:30Z", "ratio": 10}],
        reference=pd.Timestamp("2025-02-01", tz="UTC"),
    )
    pd.testing.assert_frame_equal(raw, original)
    assert adjusted.price.tolist() == [1000., 1000., 1010.]
    assert adjusted["size"].tolist() == [1., 1., 1.]
    np.testing.assert_allclose(adjusted.price * adjusted["size"], raw.price * raw["size"])
    before, after = [get_dollar_bars(frame, 2000) for frame in (raw, adjusted)]
    pd.testing.assert_index_equal(before.index, after.index)
    assert before.dollar_value.tolist() == after.dollar_value.tolist() == [2000.]
    assert after[["open", "high", "low", "close", "volume"]].values.tolist() == [
        [1000., 1000., 1000., 1000., 2.],
    ]
    # The split is zero return, followed by a real +1% move, not a -90% label.
    assert adjusted.price.pct_change().iloc[1] == 0
    assert adjusted.price.pct_change().iloc[2] == pytest.approx(.01)
    closes = adjusted.set_index("timestamp").price
    labels = get_bins(pd.DataFrame({"event_end": [closes.index[2]]}, index=closes.index[:1]), closes)
    assert labels.realized_return.iloc[0] == pytest.approx(.01)
    assert labels.label.iloc[0] == 1
    # A reference-unit holding keeps its value without another quantity change.
    assert adjusted.price.iloc[0] * 3 == adjusted.price.iloc[1] * 3


def test_multiple_splits_reverse_split_and_warmup_reference():
    raw = trades(["2025-01-02T14:30Z", "2025-01-03T14:30Z", "2025-02-03T14:30Z",
                  "2025-03-03T14:30Z", "2025-03-04T14:30Z"],
                 [100., 50., 10., 100., 25.], [1., 2., 10., 1., 4.])
    events = [
        {"effective_at": "2025-01-03T14:30Z", "ratio": 2},
        {"effective_at": "2025-02-03T14:30Z", "ratio": 5},
        {"effective_at": "2025-03-03T14:30Z", "ratio": .1},
        {"effective_at": "2025-03-04T14:30Z", "ratio": 4},
    ]
    result = market_structured_bars.adjust_trade_units(raw, events, reference=pd.Timestamp("2025-02-01", tz="UTC"))
    np.testing.assert_allclose(result.price, [50.] * 5)
    np.testing.assert_allclose(result["size"], [2.] * 5)
    # Later splits do not change already-emitted research observations.
    prefix = market_structured_bars.adjust_trade_units(raw.iloc[:3], events[:2],
                                             reference=pd.Timestamp("2025-02-01", tz="UTC"))
    pd.testing.assert_frame_equal(result.iloc[:3], prefix)


def test_split_exactly_at_reference_and_no_split():
    raw = trades(["2025-02-03T14:29:59Z", "2025-02-03T14:30:00Z"], [100., 50.], [1., 2.])
    result = market_structured_bars.adjust_trade_units(raw,
        [{"effective_at": "2025-02-03T14:30Z", "ratio": 2}],
        reference=pd.Timestamp("2025-02-03T14:30Z"))
    assert result.price.tolist() == [50., 50.]
    unchanged = market_structured_bars.adjust_trade_units(raw, [], reference=pd.Timestamp("2025-02-01", tz="UTC"))
    pd.testing.assert_frame_equal(unchanged[raw.columns], raw)


@pytest.mark.parametrize("ratio", [0, -1, float("nan"), float("inf")])
def test_invalid_adjustment_ratio_fails(ratio):
    with pytest.raises(ValueError, match="finite and positive"):
        market_structured_bars.adjust_trade_units(
            trades(["2025-02-03T14:30Z"], [100.], [1.]),
            [{"effective_at": "2025-02-03T14:30Z", "ratio": ratio}],
            reference=pd.Timestamp("2025-02-01", tz="UTC"),
        )


def test_bar_builder_adjusts_before_carrying_pending_trades_and_rejects_changed_splits(paths):
    manifest = paths.root / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    for day, prices, sizes in [
        ("2025-02-03", [100.], [2.]),
        ("2025-02-04", [100., 100.], [2., 1.]),
        ("2025-02-05", [10.], [20.]),
    ]:
        path = paths.raw("A", "tick") / f"{day}.parquet"
        frame = trades(pd.date_range(f"{day}T14:30Z", periods=len(prices), freq="min"), prices, sizes)
        market_data.save_frame(frame, path)
        path.with_suffix(".json").write_text(json.dumps({}))
    split_file = market_data.split_path(paths, "A")
    split_file.parent.mkdir(parents=True)
    snapshot = {"symbol": "A",
                "start": str(paths.start), "end": str(paths.end), "reference": str(paths.start),
                "events": [{"effective_at": "2025-02-05T14:30Z", "ratio": 10}]}
    split_file.write_text(json.dumps(snapshot))
    kwargs = dict(manifest_path=manifest, start=paths.start, end=paths.end,
                  lookback_sessions=1, target_bars_per_session=1)
    build_symbol_dollar_features(paths, "A", **kwargs)
    bars = pd.read_parquet(paths.feature("A", "dollar_bars"))
    assert bars.close.tolist() == [100., 100.]
    assert bars.volume.tolist() == [2., 3.]
    assert bars.iloc[-1].start.date() != bars.iloc[-1].end.date()
    assert pd.read_parquet(paths.raw("A", "tick") / "2025-02-05.parquet").price.tolist() == [10.]
    assert build_symbol_dollar_features(paths, "A", **kwargs)["cached_files"] == 1
    snapshot["events"][0]["ratio"] = 5
    split_file.write_text(json.dumps(snapshot))
    with pytest.raises(ValueError, match="deliberately rebuild"):
        build_symbol_dollar_features(paths, "A", **kwargs)


@pytest.mark.parametrize("column", ["timestamp", "symbol", "price", "size"])
def test_dollar_bars_require_fixed_trade_columns(column):
    trades = _make_trades(4).drop(columns=column)
    with pytest.raises(ValueError, match=column):
        get_dollar_bars(trades, threshold=200.)


def test_dollar_bars_require_timestamp_column_even_with_datetime_index():
    trades = _make_trades(4).set_index("timestamp")
    with pytest.raises(ValueError, match="timestamp"):
        get_dollar_bars(trades, threshold=200.)


def test_dollar_bars_keep_supplied_dollar_values_and_stable_trade_order():
    trades = _make_trades(3)
    trades["timestamp"] = pd.to_datetime(["2026-01-01T00:01Z", "2026-01-01T00:00Z", "2026-01-01T00:00Z"])
    trades["price"] = ["103", "101", "102"]
    trades["size"] = ["1", "2", "3"]
    trades["dollar_value"] = [300., 100., 200.]
    original = trades.copy()

    bars = get_dollar_bars(trades, threshold=300.)

    assert bars[["open", "close", "volume", "dollar_value", "ticks"]].values.tolist() == [
        [101., 102., 5., 300., 2], [103., 103., 1., 300., 1],
    ]
    assert str(bars.index.tz) == "UTC"
    pd.testing.assert_frame_equal(trades, original)


@pytest.mark.parametrize("invalid_feature", [False, True])
def test_read_all_bars_loads_manifest_once_and_validates_before_reading(tmp_path, monkeypatch, invalid_feature):
    paths = ResearchPaths(tmp_path, start=pd.Timestamp("2025-01-01", tz="UTC"),
                          end=pd.Timestamp("2026-01-01", tz="UTC"))
    manifest_path = tmp_path / "universe.csv"
    manifest = Mock(return_value=pd.DataFrame({"symbol": ["B", "A"]}))
    monkeypatch.setattr(market_data, "load_manifest", manifest)
    trace = []

    def validate(path):
        trace.append(("validate", path))
        if invalid_feature and path == paths.feature("A", "dollar_bars"):
            raise ValueError("Feature metadata unavailable")

    def read(path):
        trace.append(("read", path))
        return pd.DataFrame({"symbol": [path.parent.parent.name], "close": [100.]})

    monkeypatch.setattr(market_data, "require_feature_metadata", validate)
    reader = Mock(side_effect=read)
    monkeypatch.setattr(market_structured_bars.pd, "read_parquet", reader)
    if invalid_feature:
        with pytest.raises(ValueError, match="Feature metadata unavailable"):
            market_structured_bars.read_all_bars(paths, manifest_path=manifest_path)
        reader.assert_not_called()
    else:
        result = market_structured_bars.read_all_bars(paths, manifest_path=manifest_path)
        pd.testing.assert_frame_equal(result, pd.DataFrame({"symbol": ["B", "A"], "close": [100., 100.]}))
    manifest.assert_called_once_with(manifest_path)
    expected = [("validate", paths.feature(symbol, "dollar_bars")) for symbol in ("B", "A")]
    if not invalid_feature:
        expected += [("read", paths.feature(symbol, "dollar_bars")) for symbol in ("B", "A")]
    assert trace == expected


def test_dollar_builder_checks_raw_completion_before_cache_reuse(paths, monkeypatch):
    manifest = paths.root / "universe.csv"
    pd.DataFrame({"symbol": ["A"]}).to_csv(manifest, index=False)
    raw = paths.raw("A", "tick") / "2025-02-03.parquet"
    save_frame(pd.DataFrame(), raw)
    reuse = Mock(return_value=True)
    monkeypatch.setattr(market_data, "reusable_feature", reuse)

    with pytest.raises(ValueError, match="Incomplete raw partition"):
        build_symbol_dollar_features(
            paths, "A", manifest_path=manifest, start=paths.start, end=paths.end,
            lookback_sessions=1, target_bars_per_session=1,
        )
    reuse.assert_not_called()
