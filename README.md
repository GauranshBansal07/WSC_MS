# WSC_MS — Momentum Strategy (CatBoost + Learned HMM)

A cross-sectional momentum trading strategy for Indian equity markets.
Uses CatBoost to rank stocks by relative momentum, then a walk-forward
bivariate Gaussian HMM to detect the prevailing volatility regime and
scale position count accordingly.

---

## Strategy Overview

| Component | Detail |
|:---|:---|
| **Universe** | Nifty 100 (Nifty 50 + Nifty Next 50), point-in-time composition |
| **Signal** | Cross-sectional momentum — 1M, 3M, 6M, 12M lookbacks |
| **Classifier** | CatBoost (depth=4, lr=0.05, 150 iters) — predicts whether stock beats median next month |
| **Walk-forward** | Expanding window, minimum 60 months training before first prediction |
| **Regime filter** | Bivariate Gaussian HMM on (Nifty 50 daily log-return, 20d realized vol), 3 states, refit every 12 months with 5 random restarts |
| **Sizing** | States sorted by vol mean: LowVol → Bull (10 names), MedVol → Neutral (4), HighVol → Bear (3) |
| **Weighting** | Equal-weight within the selected book |
| **Risk** | Daily path stop-loss: −10% (Bull), −7% (Neutral), −5% (Bear) |
| **Costs** | 10 bps per side transaction cost; 5% p.a. leverage cost on exposure > 1× |

**Data period:** Historical PiT composition from January 2008; constituent
prices fetched from yfinance and cached locally. The non-survivorship-biased
PiT CSVs ensure no future index entrants contaminate historical signals.

**Out-of-sample period:** January 2018 – March 2025 (~7 years, 86 monthly
rebalances) after the mandatory 60-month training warm-up.

---

## Key Results — Nifty 100 Monthly Long-Only

### Primary variant: HMM Directional (current default)

| Metric | Value |
|:---|:---:|
| **CAGR** | **30.19%** |
| **Volatility** | 19.90% |
| **Sharpe Ratio** | **1.088** |
| **Max Drawdown** | −13.00% |
| **Calmar Ratio** | **2.323** |
| Win Rate | 69.77% |
| Avg positions held | 6.2 / month |

### Comparison: Barroso-Santa-Clara vol scaling (BSC 2015)

Running `--sizing volscale` replaces the discrete {10,4,3} book with
continuous gross-exposure scaling: `weight = min(1.25, max(0.30, σ*/ĥ_t))`
where `σ* = 0.20` (median 126-day realized vol of unscaled strategy)
and `ĥ_t` is the trailing 126-day realized vol.

| Metric | HMM Directional | VolScale 126d |
|:---|:---:|:---:|
| CAGR | **30.19%** | 22.08% |
| Volatility | 19.90% | 13.50% |
| Sharpe | **1.088** | 1.053 |
| Max DD | −13.00% | **−11.17%** |
| Calmar | **2.323** | 1.977 |

### Rolling validation — is the HMM advantage structural?

Rolling 2-year Calmar across all 63 monthly windows (Jan 2020 – Mar 2025):

| Stat | HMM Directional | VolScale 126d |
|:---|:---:|:---:|
| Median | 3.95 | 3.14 |
| P25 | 1.82 | 1.71 |
| P75 | 7.81 | 5.46 |
| Windows where HMM wins | **54/63 (86%)** | — |

HMM outperforms in **6/7 calendar years** (2018–2024). The advantage is
**structural**, not concentrated in a single tail-risk episode.

---

## Sizing Schemes

### `--sizing directional` (default)

Regime label from walk-forward HMM controls position count:

| HMM State | Vol Feature Mean | Positions | Stop-Loss |
|:---|:---:|:---:|:---:|
| LowVol → **Bull** | lowest | 10 | −10% |
| MedVol → **Neutral** | middle | 4 | −7% |
| HighVol → **Bear** | highest | 3 | −5% |

### `--sizing volscale`

Barroso-Santa-Clara (JFE 2015) continuous scaling — 10 names always selected,
gross exposure scaled between `[0.30×, 1.25×]` based on trailing realized vol.

---

## Repository Structure

```
.
├── config.py                  — Global parameters (lookbacks, costs, rates)
├── data_fetcher.py            — Price fetching, caching, forward returns
├── features.py                — Momentum feature computation
├── engine.py                  — Backtest engine: walk-forward CatBoost, BSC vol scaling,
│                                HMM sizing, stop-loss, transaction costs
├── regime.py                  — HMM regime detection (learned walk-forward, vol-poster
│                                variants)
├── main.py                    — CLI runner for all indices/sizing schemes
├── export_results.py          — Exports Date / Period_Return / Period_Turnover CSV for
│                                external evaluators
├── prepare_nifty500.py        — Builds daily/monthly cache from raw 5-min tick CSVs
└── data/
    ├── historical_composition.csv       — Point-in-time Nifty 50 composition (Jan 2008 →)
    └── nifty_next_50_composition.csv    — Point-in-time Nifty Next 50 composition (Jan 2008 →)
```

---

## Running the Strategy

### Installation

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy hmmlearn
```

### Run (Nifty 100, HMM directional — recommended)

```bash
python3 main.py --index nifty100 --sizing directional
```

### Run pure vol scaling baseline

```bash
python3 main.py --index nifty100 --sizing volscale
```

### Export CSV for external evaluator

```bash
python3 export_results.py --sizing directional --output results.csv
# python3 utils/nexus_evaluator.py results.csv
```

### All available CLI flags

```
--index   nifty50 | nifty100 | nifty250 | nifty500 | all
--regime  learned_hmm | fixed_hmm | none   (used with --sizing directional)
--sizing  directional | volscale
```

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `TRANSACTION_COST_BPS` | 10 | Cost per side (bps) |
| `LEVERAGE_COST_ANNUAL` | 5% | Annual drag on leveraged gross exposure > 1× |
| `RISK_FREE_ANNUAL` | 7% | India 10Y govt bond proxy |
| `DATA_START` | 2008-01-01 | Start of historical composition data |
| `DATA_END` | 2025-04-01 | End of evaluation period |
| `LOOKBACK_WINDOWS` | [1, 3, 6, 12] | Momentum lookback months |

---

## Notes on Survivorship Bias

- **Nifty 50 / Nifty 100**: Zero survivorship bias. The PiT composition CSVs
  enumerate exact index membership for every month from January 2008. Stocks
  are only eligible for selection if they were actually in the index that month.
- **Nifty 500**: Uses a static 2025 composition snapshot → significant
  survivorship bias. Results for this universe should not be used for
  performance evaluation.
