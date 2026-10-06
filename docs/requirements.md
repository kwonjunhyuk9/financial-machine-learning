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

- Market Data: Retrieve tick data.
- Market Features: Calculate structured bars, differentiated bars and technical indicators.
- Alternative Data: Retrieve news data.
- Alternative Features: Calculate sentiment scores.
- Train Test Split: Split event data chronologically into development and holdout sets.
- Event Labeling: Assign direction labels using the triple-barrier method.
- Event Weights: Calculate concurrency-adjusted sample weights.
- Prepare the Data: Inspect and clean the model-ready dataset.

### 2.2 Modeling

- Ensemble Methods: Build Boosting, Bagging, and Random Forest Classifiers.
- Hyperparameter Tuning: Tune the selected ensemble classifier with grid search.
- Purged Validation: Evaluate models while removing overlapping events with purging and embargoing.
- Feature Importance: Measure each feature’s contribution using MDI, MDA, and SFI.
- Primary Model: Select the candidate family and hyperparameters by minimizing weighted purged OOF log loss, with
  weighted F1 used only for exact ties; Predict event side in `{-1, 1}` from market data and alternative data.
- Meta Model: Select the candidate family and hyperparameters by maximizing weighted purged OOF F1, with weighted log
  loss used for ties; Predict event size in `{0, 1}` from market data and alternative data.

### 2.3 Backtesting

- Strategy Validation: Evaluate strategies with CPCV paths while removing overlapping events with purging and
  embargoing.
- Bet Sizing: Convert model probabilities into target position sizes.
- Portfolio Management: Simulate portfolio positions, trades, and account value.
- Backtest Statistics: Measure characteristics, performance, risk, costs, and classification results.

## 3. Future Ideas

### 3.1. Cryptocurrency Research Workflow

- Extend the research workflow to cryptocurrency markets.

### 3.2. ChatGPT-Based Features

- Explore ChatGPT as an alternative to FinBERT for text-based feature generation.

### 3.3. Kraken Integration

- Integrate Kraken into the cryptocurrency research workflow.