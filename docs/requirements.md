# Requirements Specification

## 1. Users and Environment

### 1.1 User Groups

| User Type         | Primary Goal                               |
|-------------------|--------------------------------------------|
| Preprocessor | Prepare financial data and signals         |
| Modeler      | Develop predictive investment models       |
| Backtester   | Assess strategy behavior before deployment |

## 2. Functional Requirements

### 2.1 Preprocessing

- Market Data: Tick, 1min
- Market Features: Market Structured Bars, Market Differentiated Bars, Breadth, Momentum, Overlap, Volatility
- Alternative Data: News
- News Eligibility: Keep Benzinga articles tagged only with AAPL under `/news`, `/analyst-ratings`, or
  `/analyst-stock-ratings`, including all descendant article paths. Do not add category columns or features;
  aggregate article sentiment into the existing mean sentiment score per completed market bar.
- Alternative Features: Sentiment Scores
- Event Processing: Combine point-in-time features without dropping missing values, establish the unlabeled candidate
  event holdout boundary with an initial chronological 80/20 split, reuse that fixed boundary after upstream eligibility
  changes, learn labeling rules from development only, compute event weights within each partition, explore development
  only, and remove events with invalid model features before modeling while preserving the complete feature schema
- Event Weights: Persist concurrency-adjusted return attribution and sample weights only. Within each partition,
  floor zero attribution at the smallest positive attribution and normalize to mean one, without time decay or an
  average-uniqueness multiplier. Reject partitions whose attribution weights are all zero.

### 2.2 Modeling

- Ensemble Methods: Build boosting, bagging, and random-forest classifiers without scaling or
  imputing prepared features
- Hyperparameter Tuning: Tune only the selected ensemble classifier with grid search and weighted purged cross-validation
- Purged Validation: Reuse the fixed event partition and score development folds while purging overlapping labels and
  embargoing test periods
- Feature Importance: Measure relevance with impurity, permutation, and single-feature methods
- Primary Model: Predict event direction in `{-1, 1}` from event-start sentiment, fractionally differentiated price, and
  technical features; compare and tune classifiers with weighted purged cross-validation; produce development OOF and
  final holdout sides, probabilities, and confidence
- Meta Model: Use event-start model features with primary OOF side and confidence to learn whether to act and how
  confidently to size the trade without changing primary direction; keep triple-barrier target returns out of model inputs

### 2.3 Backtesting

- Bet Sizing: Convert model probabilities and price forecasts into bounded target positions and limit prices
- Strategy Validation: Generate combinatorial purged cross-validation splits and backtest paths
- Backtest Statistics: Compute performance, drawdown, execution-cost, efficiency, and classification metrics
- Return Concentration: Report undefined HHI as `NaN` when a positive or negative return subset has fewer than
  three observations, including an empty subset.
- Backtest Synthetic: Simulate synthetic trading-rule outcomes across profit-taking and stop-loss settings

## 3. Problem Framing

### 3.1 Machine Learning Systems

- Supervised: Learn from historical events labeled with `direction_label` and OOF-derived `meta_label`.
- Classification: Predict primary direction in `{-1, 1}` and meta action in `{0, 1}`.
- Batch: Train and predict offline from locally stored historical data without online or incremental learning.
- Model-Based: Learn models from development data and use them to predict previously unseen events.

### 3.2 Internal Training Criteria

- Bagging: Train entropy-based decision trees on bootstrap samples and aggregate their predictions.
- Random Forest: Train bootstrapped, feature-subsampled decision trees using entropy-based splits.
- Boosting: Increase the weight of misclassified observations according to weighted classification error;
  use entropy to split each shallow decision-tree base estimator.

### 3.3 Model Selection Evaluation Measures

- Primary Model: Select the candidate family and hyperparameters by minimizing weighted purged OOF log loss, with
  weighted F1 used only for exact ties.
- Meta Model: Select the candidate family and hyperparameters by maximizing weighted purged OOF F1, with weighted
  log loss used for ties.
- Both Models: Report accuracy, precision, recall, F1, and log loss; visualize
  true-class-normalized confusion matrices and sample-weighted precision-recall and ROC curves on development OOF
  predictions, then evaluate the fixed final estimator on the chronological holdout once.
- Display candidate comparisons in Boosting, Bagging, Random Forest order, independently of performance-based selection.

### 3.4 Investment Strategy Evaluation Measures

- Evaluate `primary_only` and `meta_filtered` together using net return after execution costs, compound net return,
  annualized Sharpe ratio, maximum drawdown, hit ratio, average hit, average miss, total execution costs, and return on
  execution costs.
- Report the measures together without optimizing or declaring a single aggregate strategy measure.
