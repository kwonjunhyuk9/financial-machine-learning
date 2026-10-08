# `preprocessing.event_weights`

`build_partitioned_event_weights` accepts a close-price Series indexed by `(symbol, end)` for both single-security and multi-security data; a time-only price index is not supported. Attribution is computed within symbol/partition; flooring and mean-one normalization apply across the entire partition.

::: preprocessing.event_weights
