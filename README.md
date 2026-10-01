# Financial Machine Learning

## Project Description

This project is a system for financial machine learning research. It is heavily inspired by Marcos de Prado's Advances
in Financial Machine Learning, and it tries to apply those ideas in a practical software system.

Unlike existing projects that mainly provide core market data functions inspired by Advances in Financial Machine
Learning, this project aims to provide a comprehensive framework for the entire investment research workflow. It
supports both market data and alternative data to enable a broader range of financial research applications.

## Directory Structure

The project is organized as follows:

```text
  |-- src/                          # Production code for financial research
  |-- notebooks/                    # Reproducible research experiments
  |-- tests/                        # Automated testing and validation
  |-- data/                         # Datasets, model artifacts, and results
  `-- docs/                         # Project documentation
```

## Installation

Create a virtual environment, activate it, and install the project in editable mode:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Strategies

The project compares the following synchronous and asynchronous cross-sectional sentiment long-short strategies:

<table>
  <thead>
    <tr>
      <th width="10%">Aspect</th>
      <th width="45%">Synchronous Cross-Sectional Sentiment Long-Short</th>
      <th width="45%">Asynchronous Cross-Sectional Sentiment Long-Short</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Selection</th>
      <td>
        <ul>
          <li>Go long the top 20% of securities by sentiment each day.</li>
          <li>Go short the bottom 20% of securities by sentiment each day.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Maintain up to <code>K</code> eligible long positions.</li>
          <li>Maintain up to <code>K</code> eligible short positions.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <th>Entry</th>
      <td>
        <ul>
          <li>Before market open: enter at the same-day open.</li>
          <li>During market hours: enter at the same-day close.</li>
          <li>After market close: enter at the next-day open.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Rank eligible symbols in long and short queues by directional price advantage multiplied by time decay.</li>
          <li>When a slot opens, select the highest-scoring candidate from the corresponding queue.</li>
          <li>Enter only when the limit-price condition is satisfied.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <th>Exit</th>
      <td>
        <ul>
          <li>Before market open: liquidate at the same-day close.</li>
          <li>During market hours: liquidate at the next-day close.</li>
          <li>After market close: liquidate at the next-day close.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>At an event's first barrier, remove its signal and resize or close the position using the remaining active signals.</li>
        </ul>
      </td>
    </tr>
  </tbody>
</table>

## Notebook Execution Order

Run each from its containing directory in a fresh kernel:

| Order | Notebook                                                     |
|------:|--------------------------------------------------------------|
|     1 | `notebooks/preprocessing/market_data.ipynb`                  |
|     2 | `notebooks/preprocessing/market_structured_bars.ipynb`       |
|     3 | `notebooks/preprocessing/market_differentiated_bars.ipynb`   |
|     4 | `notebooks/preprocessing/market_technical_indicators.ipynb`  |
|     5 | `notebooks/preprocessing/alternative_data.ipynb`             |
|     6 | `notebooks/preprocessing/alternative_sentiment_scores.ipynb` |
|     7 | `notebooks/preprocessing/train_test_split.ipynb`             |
|     8 | `notebooks/preprocessing/event_labeling.ipynb`               |
|     9 | `notebooks/preprocessing/event_weights.ipynb`                |
|    10 | `notebooks/preprocessing/prepare_the_data.ipynb`             |
|    11 | `notebooks/modeling/primary_model.ipynb`                     |
|    12 | `notebooks/modeling/meta_model.ipynb`                        |
|    13 | `notebooks/backtesting/strategy_validation.ipynb`            |
|    14 | `notebooks/backtesting/bet_sizing.ipynb`                     |
|    15 | `notebooks/backtesting/portfolio_management.ipynb`           |
|    16 | `notebooks/backtesting/backtest_statistics.ipynb`            |
