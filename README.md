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

The model is refit every 12 months (expanding window, 5 random restarts, best log-likelihood kept). Using the volatility dimension for ordering — rather than mean return — gives more stable state assignments across refits.

### 3 — Portfolio Construction

From the ranked candidates with predicted probability ≥ 0.55, the top N (regime-gated) are selected and weighted by:

```
weight_i  ∝  pred_prob_i / σ_i(60d)
```

This is the `prob_invvol` scheme — it simultaneously rewards high-conviction picks and penalises high idiosyncratic risk. Weights are normalised to sum to 1. Cash (uninvested fraction) earns the risk-free rate.

**Costs**: 10 bps per side transaction cost; 5% p.a. leverage cost on gross exposure above 1×.

---

## Key Results

### Nifty 100 Monthly — Primary Benchmark (survivorship-bias free)

> Out-of-sample: January 2018 – January 2025 (84 monthly rebalances)

| Metric | Value |
|:---|:---:|
| **CAGR** | **29.29%** |
| Annualised Volatility | 17.30% |
| **Sharpe Ratio** | **1.164** |
| **Max Drawdown** | **−10.00%** |
| **Calmar Ratio** | **2.929** |
| Win Rate | 69.77% |
| Avg positions / month | ~6 |

This is the close-to-close baseline — signal at month-end T close, execute at that close, exit at month-end T+1 close. See [Execution Realism](#execution-realism) for the fillable-alternative variants.

### Sizing Scheme Comparison

| Scheme | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| **HMM LowVol/MedVol/HighVol** ✓ | **29.29%** | **1.164** | **−10.00%** | **2.929** |
| VolScale 126d (BSC 2015) | 22.08% | 1.053 | −11.17% | 1.977 |
| HMM 1/σ dynamic | 28.92% | 1.136 | −11.35% | 2.548 |

### Weighting Method Comparison

| Method | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Equal Weight | 30.5% | 1.143 | −11.1% | 2.74 |
| Probability-Weighted | 30.5% | 1.145 | −11.2% | 2.72 |
| Inverse Volatility | 29.3% | 1.161 | −10.1% | 2.90 |
| **Prob × Inv-Vol** ✓ | **29.3%** | **1.164** | **−10.0%** | **2.93** |
| Half-Kelly | 30.3% | 1.149 | −11.6% | 2.61 |

`prob_invvol` wins on every risk-adjusted metric. Equal/Prob weighting gives marginally higher raw CAGR but ~10% larger drawdowns. Half-Kelly underperforms on Calmar due to over-concentration at wrong moments.

### Lookback Ablation (leave-one-out on [1, 3, 6, 12, 36, 60])

| Config | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Drop 1M → `[3, 6, 12, 36, 60]` | 25.98% | 1.140 | −9.74% | 2.668 |
| **Drop 3M → `[1, 6, 12, 36, 60]`** ✓ | **29.29%** | **1.164** | **−10.00%** | **2.929** |
| Drop 6M → `[1, 3, 12, 36, 60]` | 25.82% | 1.035 | −15.43% | 1.673 |
| Drop 12M → `[1, 3, 6, 36, 60]` | 22.80% | 1.045 | −9.07% | 2.513 |
| Drop 36M → `[1, 3, 6, 12, 60]` | 26.99% | 1.048 | −11.77% | 2.294 |
| Drop 60M → `[1, 3, 6, 12, 36]` | 31.25% | 1.282 | −13.88% | 2.251 |

**Critical finding**: the 6M window is the single most important lookback — removing it causes the worst risk-adjusted outcome. The 3M window is detrimental (mean-reversion noise on Indian markets) and its removal gives the global Calmar optimum.

---

## Execution Realism

The headline 29% CAGR assumes instant fills at the month-end close, which isn't achievable at retail size. [`execution_realism.py`](execution_realism.py) quantifies the cost of realistic fills across four variants:

| Execution | CAGR | DD | Sharpe | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Close-to-close month-end (baseline, unfillable) | 29.29% | −10.00% | 1.164 | 2.929 |
| **First-open entry → last-close exit** ✓ | **18.05%** | **−8.51%** | **0.828** | **2.122** |
| Full open-to-open (within-month) | 13.44% | −10.07% | 0.622 | 1.334 |

Roughly ~11pp of the headline CAGR is execution slippage — the momentum premium is [concentrated in the overnight segment](https://doi.org/10.1016/j.jfineco.2019.03.011) (Lou, Polk & Skouras 2019), so any open-to-open fill gives up most of the edge. **Entering at the next day's open and exiting at the final day's close** (bolded above) recovers the most value while remaining realistically fillable.

Per-position trading logs (one CSV row per holding, suitable for manual yfinance verification) are produced with the `--log` flag, e.g. `python3 execution_realism.py --variant oc --log`.

---

## Validation & Robustness

`validate_strategy.py` runs a comprehensive suite and outputs 14 charts to `output/`:

| Test | What it answers |
|:---|:---|
| **Monte Carlo Bootstrap** (10k resamples) | How stable are Sharpe / CAGR / MaxDD point estimates? |
| **Monte Carlo Path Simulation** (1k paths) | Is performance path-dependent or robust to return ordering? |
| **Sensitivity: Prob Threshold** (0.50–0.70) | Does alpha hold across entry thresholds? |
| **Sensitivity: Stop-Loss** (−3% to −15%) | Where is the true risk/return optimum? |
| **Sensitivity: Regime Sizing** (heatmap) | Are our 10/4/3 sizes at the right local maximum? |
| **Rolling Sharpe / Calmar** | Any regime of degradation? |
| **Regime Performance Decomposition** | How does each vol regime contribute to returns? |

Supporting script:
- [`diagnostics.py`](diagnostics.py) — `--mode annual` (per-year PnL/DD/Sharpe) and `--mode lookback` (leave-one-out window ablation).

---

## Repository Structure

```
├── config.py                    — Global parameters
├── data_fetcher.py              — Price fetching (close/daily/open matrices), caching, forward returns
├── features.py                  — Momentum + Z-score feature computation
├── engine.py                    — Core engine: walk-forward CatBoost, HMM directional /
│                                  volscale / hmm_vol_size sizing, prob_invvol weighting,
│                                  daily stop-loss, turnover tracking
├── regime.py                    — Walk-forward HMM regime detection
├── main.py                      — CLI runner
├── export_results.py            — CSV exporter for external evaluators
├── live_portfolio.py            — Month-start production signal generator
├── validate_strategy.py         — Full validation suite (Monte Carlo + sensitivity + charts)
├── compare_weights.py           — Head-to-head benchmark of all 5 weighting methods
├── diagnostics.py               — Post-hoc analysis: --mode annual | --mode lookback
├── execution_realism.py         — Execution-slippage suite: cc / oc / oo / four variants + --log
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
python3 main.py --index nifty100 --sizing directional

# Full validation suite (14 charts → output/)
python3 validate_strategy.py --index nifty100

# Month-start live portfolio
python3 live_portfolio.py --index nifty100

# Weighting method benchmark
python3 compare_weights.py --index nifty100

# Post-hoc diagnostics
python3 diagnostics.py --mode annual
python3 diagnostics.py --mode lookback

# Execution realism — unified suite
python3 execution_realism.py --variant cc   --log   # 29% close-to-close baseline
python3 execution_realism.py --variant oc   --log   # fillable open-entry / close-exit
python3 execution_realism.py --variant oo   --log   # open-to-open (within-month)
python3 execution_realism.py --variant four         # 4-way entry/exit × training comparison
```

### CLI Flags

```
main.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none
  --sizing    directional | volscale

compare_weights.py / validate_strategy.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none

live_portfolio.py:
  --index     nifty50 | nifty100
```

---

## Configuration

**`config.py`**

| Parameter | Value | Description |
|:---|:---:|:---|
| `LOOKBACK_WINDOWS` | [1, 6, 12, 36, 60] | Momentum formation periods (months) |
| `DATA_START` | 2008-01-01 | Start of PiT composition data |
| `DATA_END` | 2025-04-01 | End of evaluation period |
| `TRANSACTION_COST_BPS` | 10 | Cost per side |
| `LEVERAGE_COST_ANNUAL` | 5% | Drag on gross exposure > 1× |
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
- All headline results are on Nifty 100 with PiT composition. Earlier Nifty 250 / 500 extensions used a 2025 static snapshot and were removed to avoid confusion.
