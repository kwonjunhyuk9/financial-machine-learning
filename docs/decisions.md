# Decision Record

## 1. Decision Log

### 1.1 Choosing the Data Source

Decision:

- Use Finnhub

Reason:

- Finnhub provides historical and realtime data that fit the current research workflow.
- Finnhub covers all market data, fundamental data, analytic data, and alternative data at a reasonable price.

### 1.2 Choosing the Execution Platform

Decision:

- Use Kraken

Reason:

- Kraken provides reliable infrastructure and broad market support for equity and cryptocurrency execution.

### 1.3 Data Storage

Decision:

- Use Parquet instead of a database for data storage.

Reason:

- It is simpler and more lightweight for the current research workflow.
- It reduces operational overhead compared with managing a database.

### 1.4 Choosing the Documentation Stack

Decision:

- Use MkDocs for the documentation site.
- Use the Material for MkDocs theme for navigation and presentation.
- Use mkdocstrings to generate API reference pages from Python docstrings.

Reason:

- The project already uses Markdown-oriented project documents, so MkDocs fits the existing writing workflow better than
  a reStructuredText-first stack.
- mkdocstrings works well with the Google-style docstrings used in the Python modules and keeps API documentation close
  to the code.
- The current codebase is relatively small and research-oriented, so a lightweight documentation stack is a better fit
  than a heavier Sphinx setup.
