# Requirements

## 1. Users and Environment

### 1.1 User Groups

| User Type    | Primary Goal                         |
|--------------|--------------------------------------|
| Preprocessor | Prepare financial data and signals   |
| Modeler      | Develop predictive investment models |
| Backtester   | Assess strategy behavior             |

## 2. Functional Requirements

### 2.1 Preprocessing

- Market Data: Ticks.
- Market Features: Market Structured Bars, Market Differentiated Bars, Breadth, Momentum, Overlap, Volatility.
- Alternative Data: News.
- Alternative Features: Sentiment Scores.
- Event Labeling:
- Event Weights:

### 2.2 Modeling

- Ensemble Methods: Build Boosting, bagging, and Random Forest classifiers without scaling or imputing prepared
  features.
- Hyperparameter Tuning: Tune the selected ensemble classifier with grid search and weighted purged cross-validation.
- Purged Validation: Reuse the fixed event partition and score development folds while purging overlapping labels and
  embargoing test periods.
- Feature Importance: Measure relevance with impurity, permutation, and single-feature methods.
- Primary Model: Select the candidate family and hyperparameters by minimizing weighted purged OOF log loss, with
  weighted F1 used only for exact ties; Predict event side in `{-1, 1}` from market data and alternative data.
- Meta Model: Select the candidate family and hyperparameters by maximizing weighted purged OOF F1, with weighted log
  loss used for ties; Predict event size in `{0, 1}` from market data and alternative data.

### 2.3 Backtesting

- Bet Sizing: Convert model probabilities and price forecasts into bounded target positions and limit prices.
- Portfolio Management:
- Strategy Validation: Generate combinatorial purged cross-validation splits and backtest paths.
- Backtest Statistics: Compute performance, drawdown, execution-cost, efficiency, and classification metrics.