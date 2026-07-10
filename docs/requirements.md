# Requirements Specification

## 1. Users and Environment

### 1.1 User Groups

| User Type         | Primary Goal                                          |
|-------------------|-------------------------------------------------------|
| Data Preprocessor | Prepare financial data and signals                    |
| Strategy Modeler  | Develop predictive investment models                  |
| Model Backtester  | Assess strategy behavior before deployment            |
| Live Trader       | Operate trading workflows in live market environments |

## 2. Functional Requirements

### 2.1 Data Preprocessing

- Fetch Fundamental Data: Value, Quality, Profitability, Growth, Investment, Leverage, Efficiency, Cash Flow,
  Shareholder Yield, Size
- Preprocess Fundamental Data: PBR, ROE, Operating Margin, Sales Growth, Asset Growth, Debt-to-Equity Ratio, Asset
  Turnover, FCF, Dividend Yield, Market Capitalization
- Fetch Market Data: Price, Volume
- Preprocess Market Data: Fractionally Differentiated Bars, Structural Breaks, Entropy Features
- Fetch Alternative Data: News
- Preprocess Alternative Data: Sentiment Scoring

### 2.2 Strategy Modeling

- Ensemble Methods: Build bagging, random forest, and boosting classifiers
- Hyperparameter Tuning: Tune classifiers with grid or randomized purged cross-validation
- Cross Validation: Split and score models while purging overlapping labels and embargoing test periods
- Feature Importance: Measure relevance with impurity, permutation, single-feature, and orthogonal-importance methods

### 2.3 Model Backtesting

- Bet Sizing: Convert model probabilities and price forecasts into bounded target positions and limit prices
- Backtest Overfitting: Estimate backtest overfitting risk with combinatorially symmetric cross-validation
- Backtest Validation: Generate combinatorial purged cross-validation splits and backtest paths
- Backtest Synthetic: Simulate synthetic trading-rule outcomes across profit-taking and stop-loss settings
- Backtest Statistics: Compute performance, drawdown, execution-cost, efficiency, and classification metrics
- Strategy Risk: Estimate precision, betting frequency, and failure probability needed to reach a target Sharpe ratio

### 2.4 Live Trading

- Configuration: Load Kraken API credentials, trading pair, order size, and live/dry-run mode
- Kraken Client: Fetch Kraken account balances, market prices, and order status
- Order Manager: Submit, cancel, and track Kraken market or limit orders
- Risk Manager: Block orders that exceed cash, position, or maximum order-size limits
- Live Runner: Run the live trading loop from signal generation to risk check to Kraken order submission
