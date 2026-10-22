# `preprocessing.event_weights`

The weighted-event output adds only `sample_weight` immediately after `direction_label`
(or at the end if no direction label is supplied), normalized to mean one
within each partition across symbols after flooring at the smallest positive
return-attribution value. `return_attribution_weight` is an internal intermediate
and is not returned or persisted. Data preparation retains the same output contract
when it recomputes weights after removing invalid feature rows.

Existing weighted and prepared event artifacts must have the former
`return_attribution_weight` column removed before using the new schema; their
remaining columns and `sample_weight` values do not need to change.

::: preprocessing.event_weights
