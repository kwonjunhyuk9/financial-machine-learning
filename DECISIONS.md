# Decision Record

## 1. Decision Log

### 1.1 Choosing the Market Data

Decision:

- Use Toss API for Equities
- Use Binance API for Cryptocurrencies

Reason:

- Toss has a sufficiently large user base to reflect trends in both U.S. and Korean equities.
- Toss provides both historical and live raw market data, which is essential for creating dollar bars.
- Binance is the largest cryptocurrency exchange globally, making it a representative source for crypto market data.
- Binance also provides both historical and live raw market data, which is essential for creating dollar bars.

### 1.2 Choosing the Fundamental Data

Decision:
Reason:

### 1.3 Choosing the Analytic Data

Decision:
Reason:

### 1.4 Choosing the Alternative Data

Decision:
Reason:

### 1.5 Choosing the Execution Platform

Decision:

- Use Toss API for Equities
- Use Binance API for Cryptocurrencies

Reason:

- Toss supports both U.S. and Korean equities.
- Binance provides reliable infrastructure and broad market support for cryptocurrency execution.

### 1.6 Data Storage

Decision:

- Use Parquet instead of a database for data storage.

Reason:

- It is simpler and more lightweight for the current research workflow.
- It reduces operational overhead compared with managing a database.
