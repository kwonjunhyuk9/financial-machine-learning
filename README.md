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

The project compares the following synchronous and asynchronous cross-sectional market and sentiment long-short strategies:

<table>
  <thead>
    <tr>
      <th width="10%">Aspect</th>
      <th width="30%">Synchronous Cross-Sectional Market Long-Short</th>
      <th width="30%">Synchronous Cross-Sectional Sentiment Long-Short</th>
      <th width="30%">Asynchronous Cross-Sectional Sentiment Long-Short</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Features</th>
      <td>
        <ul>
          <li>Market data.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Market data.</li>
          <li>News data.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Market data.</li>
          <li>News data.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <th>Selection</th>
      <td>
        <ul>
          <li>Each day, go long the top N% of securities predicted to rise, ranked by confidence by market data.</li>
          <li>Each day, go short the top N% of securities predicted to fall, ranked by confidence by market data.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Each day, go long the top N% of securities predicted to rise, ranked by confidence by market data and sentiment data.</li>
          <li>Each day, go short the top N% of securities predicted to fall, ranked by confidence by market data and sentiment data.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Maintain up to K eligible long positions.</li>
          <li>Maintain up to K eligible short positions.</li>
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
          <li>Before market open: enter at the same-day open.</li>
          <li>During market hours: enter at the same-day close.</li>
          <li>After market close: enter at the next-day open.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Candidate Priority Queue: Keep up to K unselected candidates per side, ranked by latest meta probability × time decay and requiring positive price advantage, separate from positions.</li>
          <li>Candidate Registration: On new events, if the score is higher than the worst score from the queue, replace it with the new event.</li>
          <li>Candidate Removal: On each own-symbol bar, discard limit violations from queue.</li>
          <li>Order Reselection: Select initially and on event exits or completed liquidation; otherwise leave vacant slots unfilled.</li>
          <li>Order Update: A new event replaces the symbol's previous signal and updates its direction, eligibility, and limit.</li>
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
          <li>Before market open: liquidate at the same-day close.</li>
          <li>During market hours: liquidate at the next-day close.</li>
          <li>After market close: liquidate at the next-day close.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Close a position when its latest signal expires at its first barrier or loses its direction; superseded signals never return.</li>
        </ul>
      </td>
    </tr>
    <tr>
      <th>Portfolio</th>
      <td>
        <ul>
          <li>Initially split capital equally between the open and close books; track each book's PnL independently.</li>
          <li>Allocate half of each book's equity to each direction, equally weighted across selected positions.</li>
          <li>Open book: up to 5 longs and 5 shorts.</li>
          <li>Close book: up to 5 longs and 5 shorts.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Initially split capital equally between the open and close books; track each book's PnL independently.</li>
          <li>Allocate half of each book's equity to each direction, equally weighted across selected positions.</li>
          <li>Open book: up to 5 longs and 5 shorts.</li>
          <li>Close book: up to 5 longs and 5 shorts.</li>
        </ul>
      </td>
      <td>
        <ul>
          <li>Share up to 50% of AUM equally per side, capped at 1/K per selected symbol; leave unused capacity in cash.</li>
          <li>Resize when selected membership changes, not when probabilities alone change.</li>
          <li>Discretize the latest signed signal only to determine direction and eligibility.</li>
          <li>Hold up to 10 longs and 10 shorts.</li>
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
|     3 | `notebooks/preprocessing/market_technical_indicators.ipynb`  |
|     4 | `notebooks/preprocessing/market_differentiated_bars.ipynb`   |
|     5 | `notebooks/preprocessing/alternative_data.ipynb`             |
|     6 | `notebooks/preprocessing/alternative_sentiment_scores.ipynb` |
|     7 | `notebooks/preprocessing/train_test_split.ipynb`             |
|     8 | `notebooks/preprocessing/event_labeling.ipynb`               |
|     9 | `notebooks/preprocessing/event_weights.ipynb`                |
|    10 | `notebooks/preprocessing/prepare_the_data.ipynb`             |
|    11 | `notebooks/modeling/{market,sentiment}/primary_model.ipynb` |
|    12 | `notebooks/modeling/{market,sentiment}/meta_model.ipynb` |
|    13 | `notebooks/backtesting/{strategy}/development backtesting.ipynb` |
|    14 | `notebooks/backtesting/{strategy}/holdout backtesting.ipynb` |
