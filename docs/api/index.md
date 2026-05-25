# API Overview

This site exposes the current Python modules through `mkdocstrings`.

The API section is intentionally aligned with the two implemented packages in this repository:

- `data_preprocessing`: dataset ingestion, bar construction, labeling, weighting, and feature preparation.
- `feature_analysis`: validation, tuning, ensemble methods, and feature evaluation utilities.

If you add public modules later under `strategy_research`, `model_backtesting`, or `live_trading`, extend the API nav only when those modules contain real importable code. That keeps the docs close to the current repository state instead of documenting placeholders.
