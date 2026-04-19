# WSC_MS — Momentum Strategy (CatBoost)

A cross-sectional momentum trading strategy for Indian equity markets using CatBoost, evaluated on the Nifty 50, Nifty 100, and Nifty 500 universes.

## Strategy Overview

- **Signal**: Cross-sectional momentum (1M, 3M, 6M, 12M lookbacks)
- **Model**: CatBoost classifier predicting whether a stock will beat the cross-sectional median return next period
- **Regime Filter**: Macro regime detection (Bull / Neutral / Bear) using Nifty 50 SMA stack
- **Risk**: Daily path stop-loss validation, turnover-based transaction costs, cash allocation for unfilled slots
- **Evaluation**: Walk-forward out-of-sample, 2021–2025

---

## Repository Structure

```
.
├── config.py                  # Global parameters (lookbacks, costs, risk-free rate)
├── data_fetcher.py            # Price fetching, caching, forward return computation
├── features.py                # Momentum feature computation + Rank IC
├── engine.py                  # Core backtesting engine + HMM Regime Logic
├── main.py                    # Unified CLI runner for all indices
├── prepare_nifty500.py        # Builds daily/monthly cache from raw 5-min tick CSVs
└── data/
    ├── historical_composition.csv       # Point-in-time Nifty 50 composition
    └── nifty_next_50_composition.csv    # Point-in-time Nifty Next 50 composition
```

---

## Running the Strategy

The entire suite has been consolidated into a single runner which applies the Hidden Markov Model (HMM) regime logic across all universes.

### Run All Indexes
```bash
python3 main.py
```

### Run a Specific Index
```bash
python3 main.py --index nifty100
python3 main.py --index nifty500
```

---

## Key Results (2021–2025, Out-of-Sample)

| Universe | Frequency | Strategy | CAGR | Sharpe | Max DD | Calmar |
|:---|:---|:---|:---:|:---:|:---:|:---:|
| **Nifty 50** | Monthly | Long Only | 12.76% | 0.461 | -12.88% | 0.991 |
| **Nifty 100** | Monthly | Long Only | 30.19% | 1.088 | -13.00% | 2.323 |
| **Nifty 100** | Monthly | Long + Short | 16.91% | 0.896 | -8.26% | 2.046 |
| Nifty 250** | Weekly | Long Only | 24.22% | 1.162 | -9.11% | 2.660 |
| Nifty 500* | Monthly | Long Only | 50.90% | 1.531 | -13.37% | 3.807 |
| Nifty 500* | Monthly | Long + Short | 23.21% | 0.965 | -9.23% | 2.513 |
| Nifty 500* | Weekly | Long Only | 25.59% | 1.007 | -9.74% | 2.629 |
| Nifty 500* | Weekly | Long + Short | 19.87% | 0.996 | -7.69% | 2.582 |

> **Note**: Nifty 50 and Nifty 100 runs are **survivorship-bias free**, using historical point-in-time composition data to strictly limit stock selection to names that were actively in the index on each historical date. 
> *Nifty 500 results use a static snapshot universe (2025 composition). A random baseline on the same universe generates ~23% CAGR, indicating survivorship bias inflates the Nifty 500 results.
> **Nifty 250 proxy: Uses the top 250 stocks by historical observation count (oldest, large/mid-caps). Note the severe drop in monthly performance because all recent high-growth small-caps and IPOs are excluded in this subset.

---

## Dependencies

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy hmmlearn
```

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `TRANSACTION_COST_BPS` | 10 | Cost per side (bps) |
| `SHORT_BORROW_COST_ANNUAL` | 8% | Annual cost to borrow for shorts |
| `RISK_FREE_ANNUAL` | 7% | India 10Y govt bond proxy |
| `DATA_START` | 2008-01-01 | Start of price history |
| `DATA_END` | 2025-04-01 | End of evaluation period |
