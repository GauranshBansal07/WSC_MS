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
| **Nifty 50** | Monthly | Long Only | 20.09% | 0.748 | -14.98% | 1.341 |
| **Nifty 50** | Monthly | Long + Short | 5.56% | -0.127 | -17.32% | 0.321 |
| **Nifty 100** | Monthly | Long Only | 31.57% | 1.031 | -17.90% | 1.764 |
| **Nifty 100** | Monthly | Long + Short | 14.03% | 0.647 | -11.91% | 1.178 |
| Nifty 250** | Weekly | Long Only | 29.67% | 1.127 | -9.96% | 2.978 |
| Nifty 250** | Weekly | Long + Short | 11.29% | 0.443 | -10.16% | 1.111 |
| Nifty 250** | Monthly | Long Only | 4.15% | -0.126 | -23.68% | 0.175 |
| Nifty 250** | Monthly | Long + Short | -2.21% | -0.871 | -26.63% | -0.083 |
| Nifty 500* | Monthly | Long Only | 56.17% | 1.363 | -15.28% | 3.676 |
| Nifty 500* | Monthly | Long + Short | 20.15% | 0.743 | -14.31% | 1.408 |
| Nifty 500* | Weekly | Long Only | 31.21% | 1.109 | -18.64% | 1.674 |
| Nifty 500* | Weekly | Long + Short | 19.46% | 1.020 | -7.86% | 2.474 |

> **Note**: Nifty 50 and Nifty 100 runs are **survivorship-bias free**, using historical point-in-time composition data to strictly limit stock selection to names that were actively in the index on each historical date. 
> *Nifty 500 results use a static snapshot universe (2025 composition). A random baseline on the same universe generates ~23% CAGR, indicating survivorship bias inflates the Nifty 500 results.
> **Nifty 250 proxy: Uses the top 250 stocks by historical observation count (oldest, large/mid-caps). Note the severe drop in monthly performance because all recent high-growth small-caps and IPOs are excluded in this subset.

---

## Dependencies

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy
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
