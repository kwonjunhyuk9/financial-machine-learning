# `preprocessing.market_technical_indicators`

`build_technical_features(...)` builds cached features from local dollar bars through
`save_market_technical_indicators(...)`, which uses FinanceToolkit's `Technicals` class directly and preserves the 48 native FinanceToolkit indicators that are valid on single-security dollar bars.

::: preprocessing.market_technical_indicators
