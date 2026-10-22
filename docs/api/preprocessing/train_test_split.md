# `preprocessing.train_test_split`

Combined candidate schemas begin with `symbol` followed by `event_start`; rows remain sorted chronologically.

Partition metadata is ordered as `holdout_boundary` followed by `partition`: the cutoff precedes the assignment it determines.
Subsequent preprocessing stages preserve this order.

::: preprocessing.train_test_split
