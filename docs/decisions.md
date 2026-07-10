# Decision Record

## 1. Decision Log

### 1.1 Choosing the Data Source

Decision:

- Use Finnhub

Reason:

- Finnhub provides historical and realtime data that fit the current research workflow.
- Finnhub covers all market data, fundamental data, analytic data, and alternative data at a reasonable price.

### 1.2 Choosing the Data Storage

Decision:

- Use Parquet instead of a database for research data storage.
- Use a Trading State Store only for live execution state.

Reason:

- It is simpler and more lightweight for the current research workflow.
- It reduces operational overhead compared with managing a database for research datasets.
- Live trading still needs a dedicated state store for orders, executions, and positions.

### 1.3 Choosing the Execution Platform

Decision:

- Use Kraken

Reason:

- Kraken provides reliable infrastructure.
- Kraken provides broad market support for equity and cryptocurrency execution.

### 1.4 Choosing the Documentation Stack

Decision:

- Use MkDocs for the documentation site.
- Use the Material for MkDocs theme for navigation and presentation.
- Use mkdocstrings to generate API reference pages from Python docstrings.

Reason:

- The project uses Markdown-oriented project documents, so MkDocs fits the existing writing workflow.
- MkDocs works well with the Google-style docstrings used in the Python modules and keeps API documentation close
  to the code.
- The current codebase is relatively small and research-oriented, so a lightweight documentation stack is a better fit
  than a heavier Sphinx setup.

### 1.5 Choosing the Logging Tool

Decision:

- Use loguru for application and workflow diagnostics.

Reason:

- loguru keeps logging setup lightweight while still supporting levels, formatting, and file sinks.
- It fits the project's research-oriented Python workflow without requiring a larger logging configuration layer.

### 1.6 Choosing the Testing Tool

Decision:

- Use pytest for automated tests.

Reason:

- pytest provides a simple test authoring workflow with clear assertions and fixture support.
- It is widely supported by Python tooling and fits the project's lightweight package structure.
