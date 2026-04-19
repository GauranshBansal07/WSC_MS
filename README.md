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
| **Nifty 50** | Monthly | Long Only | 12.93% | 0.512 | -12.88% | 1.004 |
| **Nifty 50** | Monthly | Long + Short | 4.71% | -0.254 | -12.40% | 0.380 |
| **Nifty 100** | Monthly | Long Only | 29.26% | 1.039 | -12.96% | 2.257 |
| **Nifty 100** | Monthly | Long + Short | 17.33% | 0.922 | -7.08% | 2.447 |
| Nifty 250** | Weekly | Long Only | 26.66% | 1.191 | -11.04% | 2.414 |
| Nifty 250** | Weekly | Long + Short | 13.21% | 0.663 | -6.88% | 1.920 |
| Nifty 250** | Monthly | Long Only | 9.92% | 0.254 | -21.68% | 0.457 |
| Nifty 250** | Monthly | Long + Short | 5.04% | -0.136 | -12.66% | 0.398 |
| Nifty 500* | Monthly | Long Only | 42.50% | 1.431 | -12.75% | 3.334 |
| Nifty 500* | Monthly | Long + Short | 23.77% | 1.078 | -7.03% | 3.383 |
| Nifty 500* | Weekly | Long Only | 31.17% | 1.146 | -9.13% | 3.414 |
| Nifty 500* | Weekly | Long + Short | 22.34% | 1.210 | -6.52% | 3.424 |

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
