# WSC_MS — CatBoost + HMM Cross-Sectional Momentum Strategy

Cross-sectional momentum strategy for Indian equity markets. Stocks are ranked monthly by a CatBoost classifier trained on momentum signals; position sizing and stop-losses are scaled dynamically by a walk-forward HMM volatility regime.

---

## How It Works

### 1 — Signal Generation

At each monthly rebalance, every Nifty constituent receives a **predicted probability** of outperforming the cross-sectional median in the coming month. The CatBoost model is trained on an expanding window — no future data ever touches the training set.

**Features** (10 total, lookbacks in months):

| Type | Features |
|:---|:---|
| Raw momentum | `mom_1m`, `mom_6m`, `mom_12m`, `mom_36m`, `mom_60m` |
| Cross-sectional Z-score | `zscore_1m`, `zscore_6m`, `zscore_12m`, `zscore_36m`, `zscore_60m` |

The 3M window was ablated out — it adds mean-reversion noise on Indian markets. 36M and 60M windows capture structural multi-year trends orthogonal to the short-term reversal.

### 2 — Regime Detection

A **bivariate Gaussian HMM** is fitted walk-forward on Nifty 50 daily `(log-return, 20-day realised vol)`. States are sorted by fitted **volatility mean** — not returns — giving three regimes that reflect the actual market stress level:

| Regime | Realised Vol | Max Positions | Daily Stop |
|:---|:---:|:---:|:---:|
| **LowVol** | Lowest | 10 | −7% |
| **MedVol** | Middle | 4 | −6% |
| **HighVol** | Highest | 3 | −4% |

The model is refit every 12 months (expanding window, 5 random restarts, best log-likelihood kept).

### 3 — Portfolio Construction

From the ranked candidates with predicted probability ≥ 0.55, the top N (regime-gated) are selected and weighted by:

```
weight_i  ∝  pred_prob_i / σ_i(60d)
```

Weights are normalised to sum to 1. Cash (uninvested fraction) earns the risk-free rate.

**Stop-loss** uses a dual-tranche approach: the held portion (carried over from last month) is referenced to last month-end close; the new portion (added this month) is referenced to current month-end close. Each is monitored independently against intra-month daily closes.

**Costs**: 10 bps per side transaction cost.

---

## Key Results

### Nifty 100 Monthly — Primary Benchmark (survivorship-bias free)

> Out-of-sample: January 2018 – January 2025 (84 monthly rebalances)

| Metric | Value |
|:---|:---:|
| **CAGR** | **31.24%** |
| Annualised Volatility | 18.21% |
| **Sharpe Ratio** | **1.219** |
| **Max Drawdown** | **−10.48%** |
| **Calmar Ratio** | **2.979** |
| Win Rate | 71.43% |
| Avg positions / month | ~6 |

Execution: signal at month-end close, entry and exit at month-end close.

### Lookback Ablation (leave-one-out on [1, 3, 6, 12, 36, 60])

| Config | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Drop 1M → `[3, 6, 12, 36, 60]` | 25.98% | 1.140 | −9.74% | 2.668 |
| **Drop 3M → `[1, 6, 12, 36, 60]`** ✓ | **29.29%** | **1.164** | **−10.00%** | **2.929** |
| Drop 6M → `[1, 3, 12, 36, 60]` | 25.82% | 1.035 | −15.43% | 1.673 |
| Drop 12M → `[1, 3, 6, 36, 60]` | 22.80% | 1.045 | −9.07% | 2.513 |
| Drop 36M → `[1, 3, 6, 12, 60]` | 26.99% | 1.048 | −11.77% | 2.294 |
| Drop 60M → `[1, 3, 6, 12, 36]` | 31.25% | 1.282 | −13.88% | 2.251 |

**Critical finding**: the 6M window is the single most important lookback — removing it causes the worst risk-adjusted outcome. The 3M window is detrimental and its removal gives the global Calmar optimum.

### Adjusted-Open Robustness Test

Full pipeline re-run on last-trading-day-of-month adjusted open prices to confirm alpha is not an artefact of the close series:

| Metric | Adj Close | Adj Open | Delta |
|:---|:---:|:---:|:---:|
| CAGR (%) | 31.24 | 30.68 | −0.56 |
| Sharpe | 1.219 | 1.143 | −0.076 |
| Max DD (%) | −10.48 | −14.25 | −3.77 |

CAGR drops only 0.56% and Sharpe only 0.076 — the strategy is not biased towards the closing price series.

---

## Validation & Robustness

`validate_strategy.py` runs a comprehensive suite and outputs charts to `output/`:

| Test | What it answers |
|:---|:---|
| **Monte Carlo Bootstrap** (10k resamples) | How stable are Sharpe / CAGR / MaxDD point estimates? |
| **Monte Carlo Path Simulation** (1k paths) | Is performance path-dependent or robust to return ordering? |
| **Sensitivity: Prob Threshold** (0.50–0.70) | Does alpha hold across entry thresholds? |
| **Sensitivity: Stop-Loss** (−3% to −15%) | Where is the true risk/return optimum? |
| **Sensitivity: Regime Sizing** (heatmap) | Are our 10/4/3 sizes at the right local maximum? |
| **Rolling Sharpe / Calmar** | Any regime of degradation? |
| **Regime Performance Decomposition** | How does each vol regime contribute to returns? |
| **Adj-Open Stress Test** | Is alpha robust to close-price methodology? |

---

## Repository Structure

```
├── config.py                    — Global parameters
├── data_fetcher.py              — Price fetching (monthly/daily), caching, forward returns
├── features.py                  — Momentum + Z-score feature computation
├── engine.py                    — Core engine: walk-forward CatBoost, HMM directional sizing,
│                                  prob_invvol weighting, dual-tranche stop-loss, turnover tracking
├── regime.py                    — Walk-forward HMM regime detection
├── main.py                      — CLI runner
├── execution_realism.py         — Per-position trading log generator (CC execution)
├── validate_strategy.py         — Full validation suite (Monte Carlo + sensitivity + charts)
├── diagnostics.py               — Post-hoc analysis: --mode annual | --mode lookback
├── live_portfolio.py            — Month-start production signal generator
├── open_vs_close.py             — Adj-open vs adj-close robustness test
│
└── data/
    ├── historical_composition.csv      — PiT Nifty 50 composition (Jan 2008 →)
    └── nifty_next_50_composition.csv   — PiT Nifty Next 50 composition (Jan 2008 →)
```

---

## Quick Start

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy hmmlearn matplotlib seaborn

# Primary strategy — Nifty 100 monthly
python3 main.py --index nifty100

# Full validation suite (charts → output/)
python3 validate_strategy.py --index nifty100

# Adjusted-open stress test
python3 open_vs_close.py

# Month-start live portfolio
python3 live_portfolio.py --index nifty100

# Per-position trading log
python3 execution_realism.py --log

# Post-hoc diagnostics
python3 diagnostics.py --mode annual
python3 diagnostics.py --mode lookback
```

### CLI Flags

```
main.py:
  --index     nifty50 | nifty100 | all   (default: all)
  --regime    learned_hmm | fixed_hmm | none   (default: learned_hmm)

validate_strategy.py / live_portfolio.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none
```

---

## Configuration

**`config.py`**

| Parameter | Value | Description |
|:---|:---:|:---|
| `LOOKBACK_WINDOWS` | [1, 6, 12, 36, 60] | Momentum formation periods (months) |
| `DATA_START` | 2008-01-01 | Start of PiT composition data |
| `DATA_END` | 2025-01-31 | End of evaluation period |
| `TRANSACTION_COST_BPS` | 10 | Cost per side |
| `RISK_FREE_ANNUAL` | 7% | India 10Y govt bond proxy |

**`engine.py` portfolio constants**

| Parameter | Value | Description |
|:---|:---:|:---|
| `PROB_THRESHOLD` | 0.55 | Minimum predicted probability to enter |
| `REGIME_SIZE` | LowVol:10 / MedVol:4 / HighVol:3 | Max holdings per HMM volatility state |
| `REGIME_STOP` | LowVol:−7% / MedVol:−6% / HighVol:−4% | Daily stop-loss per regime |

---

## Notes on Survivorship Bias

- **Nifty 50 / Nifty 100**: zero survivorship bias — PiT composition CSVs cover every month from Jan 2008.
- All headline results are on Nifty 100 with PiT composition.
