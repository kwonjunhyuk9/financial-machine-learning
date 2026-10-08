# `preprocessing.market_data`

`collect_raw` collects SIP stock trades page by page into daily Parquet partitions and reuses completed
partitions. `read_raw` loads completed partitions for a requested interval.

Paths follow the [asset-class layout](../../architecture.md#12-research-data-layout).
`ResearchPaths.raw` and `feature`, and the `raw_partitions` and `read_raw` readers accept an optional
`asset_class` (`stock`, `etf`, or `crypto`). Omitted values select ETF for SPY and stock otherwise.
`ResearchPaths.news_source` locates the daily stock-universe news responses. Migration and recovery
commands are documented in the [README](../../../README.md#directory-structure).

Feature cache metadata must contain `settings`; old `computation_settings` metadata requires an explicit stage rebuild.

::: preprocessing.market_data
