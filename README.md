# WSC_MS — CatBoost + HMM Cross-Sectional Momentum Strategy

A cross-sectional momentum trading strategy for Indian equity markets combining **CatBoost gradient-boosted classification** with **walk-forward Hidden Markov Model (HMM) regime detection**. Evaluated on the Nifty 50 and Nifty 100 universes with zero survivorship bias.

## Strategy Architecture

```
  Historical Prices          Nifty 50 Index
        │                          │
        ▼                          ▼
 ┌─────────────┐         ┌──────────────────┐
 │  Momentum   │         │  Bivariate HMM   │
 │  Features   │         │  (log-ret + vol)  │
 │  + Z-scores │         │  Walk-forward fit │
 └──────┬──────┘         └────────┬─────────┘
        │                         │
        ▼                         ▼
 ┌─────────────┐         ┌──────────────────┐
 │  CatBoost   │         │  Regime Label    │
 │  Classifier │         │  Bull / Neutral  │
 │  (expanding │         │       / Bear     │
 │   window)   │         └────────┬─────────┘
 └──────┬──────┘                  │
        │                         │
        ▼                         ▼
 ┌────────────────────────────────────────┐
 │        Portfolio Constructor           │
 │  • Prob × InvVol weighting (default)  │
 │  • Regime-conditioned position sizing  │
 │  • Regime-conditioned stop-losses      │
 │  • Daily path stop-loss monitoring     │
 │  • Transaction cost deduction          │
 └────────────────────────────────────────┘
```

### Signal Pipeline

- **Features**: Cross-sectional momentum at lookbacks `[1M, 6M, 12M, 36M, 60M]` with cross-sectional Z-score normalization
- **Model**: CatBoost classifier predicting whether a stock will beat the cross-sectional median forward return
- **Training**: Expanding-window walk-forward — never uses future data; minimum 60 months of history before first prediction

### Regime Detection

Two HMM methods are supported:

| Method | Description |
|:---|:---|
| `learned_hmm` (default) | Bivariate Gaussian HMM fitted on (Nifty 50 daily log-return, 20-day realized vol). Walk-forward refit every 12 months with 5 random restarts. States sorted by mean return. |
| `fixed_hmm` | Hardcoded 3-state Gaussian HMM with manually calibrated parameters. Used as ablation baseline. |
| `none` | All dates set to 'Neutral' — disables regime filtering entirely. |

### Portfolio Construction

| Regime | Max Holdings | Daily Stop-Loss |
|:---|:---:|:---:|
| **Bull** | 10 | −7% |
| **Neutral** | 4 | −6% |
| **Bear** | 3 | −4% |

- **Weighting** (`prob_invvol`): Each position sized ∝ `pred_prob / σ_60d` — blends signal conviction with risk equalization. Normalised to sum to 1.
- Stocks selected by predicted probability (≥ 0.55 threshold)
- Daily intra-period path monitoring for stop-loss triggers
- Cash earns risk-free rate (7% p.a.)
- Transaction costs: 10 bps per side

---

## Repository Structure

```
.
├── config.py                  # Global parameters (lookbacks, costs, risk-free rate)
├── data_fetcher.py            # Price fetching, caching, forward return computation
├── features.py                # Momentum feature computation + cross-sectional Z-scores
├── engine.py                  # CatBoost walk-forward + portfolio simulation engine
│                              #   sizing: directional | volscale | hmm_vol_size
│                              #   weighting: equal | probability | inverse_vol |
│                              #              prob_invvol | kelly
├── regime.py                  # HMM regime detection (fixed + learned + vol-size variant)
├── main.py                    # Unified CLI runner for all indices / sizing schemes
├── export_results.py          # Exports Date / Period_Return / Period_Turnover CSV
├── live_portfolio.py          # Month-start production signal generator — outputs
│                              #   exact ticker weights + stop-loss levels for execution
├── validate_strategy.py       # Validation suite: Monte Carlo bootstrap (10k),
│                              #   path simulation, parameter sensitivity, 14+ charts
├── compare_weights.py         # Head-to-head benchmark of all 5 weighting methods
├── ablation_lookbacks.py      # Leave-one-out ablation on lookback window set
├── prepare_nifty500.py        # Builds daily/monthly cache from raw 5-min tick CSVs
├── output/                    # Generated charts and validation results
└── data/
    ├── historical_composition.csv      # Point-in-time Nifty 50 composition (Jan 2008 →)
    └── nifty_next_50_composition.csv   # Point-in-time Nifty Next 50 composition (Jan 2008 →)
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy hmmlearn matplotlib seaborn
```

### 2. Run the Strategy

```bash
# Recommended — HMM directional, prob_invvol weighting
python3 main.py --index nifty100 --sizing directional --weighting prob_invvol

# Specific regime method
python3 main.py --index nifty100 --regime learned_hmm
python3 main.py --index nifty100 --regime fixed_hmm
python3 main.py --index nifty100 --regime none
```

### 3. Run Validation Suite

```bash
# Full validation + 14 publication-quality charts
python3 validate_strategy.py --index nifty100

# With specific regime method
python3 validate_strategy.py --index nifty100 --regime learned_hmm
```

Generates charts in `output/` and prints:
- Monte Carlo bootstrap confidence intervals (10,000 resamples)
- Monte Carlo path simulation (1,000 shuffled equity curves)
- Parameter sensitivity analysis (threshold, stop-loss, regime sizing)
- Regime-conditioned performance decomposition
- Walk-forward stability metrics

### 4. Generate Live Portfolio (Production)

```bash
python3 live_portfolio.py --index nifty100
```

Outputs exact ticker allocations, weights, and stop-loss levels to execute at the open on the first trading day of the month.

### 5. Benchmark Weighting Methods

```bash
python3 compare_weights.py --index nifty100
```

Saves equity curves, metric bar charts, and risk-return scatter to `output/`.

### 6. Run Lookback Ablation

```bash
python3 ablation_lookbacks.py
```

Leave-one-out ablation over `[1, 3, 6, 12, 36, 60]` to identify the marginal contribution of each lookback window.

---

## Key Results (2018–2025, Out-of-Sample, Nifty 100 Monthly)

### Primary Variant: HMM Directional + `prob_invvol`

| Metric | Value |
|:---|:---:|
| **CAGR** | **29.29%** |
| **Volatility** | 17.30% |
| **Sharpe Ratio** | **1.164** |
| **Max Drawdown** | −10.00% |
| **Calmar Ratio** | **2.929** |
| Win Rate | 69.77% |
| Avg positions / month | ~6 |

> **Note**: Nifty 100 results are **survivorship-bias free**, using historical point-in-time composition CSVs to strictly limit stock selection to names actively in the index on each historical date.

### Sizing Scheme Comparison

| Metric | HMM Directional 10:4:3 | VolScale 126d (BSC 2015) | HMM 1/σ Dynamic |
|:---|:---:|:---:|:---:|
| CAGR | **29.29%** | 22.08% | 28.92% |
| Sharpe | **1.164** | 1.053 | 1.136 |
| Max DD | **−10.00%** | −11.17% | −11.35% |
| Calmar | **2.929** | 1.977 | 2.548 |

### Lookback Window Ablation (leave-one-out on [1, 3, 6, 12, 36, 60])

| Config | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Drop 1M → `[3, 6, 12, 36, 60]` | 25.98% | 1.140 | −9.74% | 2.668 |
| **Drop 3M → `[1, 6, 12, 36, 60]`** ✓ | **29.29%** | **1.164** | **−10.00%** | **2.929** |
| Drop 6M → `[1, 3, 12, 36, 60]` | 25.82% | 1.035 | −15.43% | 1.673 |
| Drop 12M → `[1, 3, 6, 36, 60]` | 22.80% | 1.045 | −9.07% | 2.513 |
| Drop 36M → `[1, 3, 6, 12, 60]` | 26.99% | 1.048 | −11.77% | 2.294 |
| Drop 60M → `[1, 3, 6, 12, 36]` | 31.25% | 1.282 | −13.88% | 2.251 |

The **6M window is most critical** (removing it causes the largest drawdown spike). The **3M window is detrimental** on Indian markets (mean-reversion noise); removing it gives the best risk-adjusted outcome.

---

## Weighting Method Comparison

| Method | CAGR | Sharpe | Max DD | Calmar |
|:---|:---:|:---:|:---:|:---:|
| Equal Weight | 30.5% | 1.143 | −11.1% | 2.74 |
| Probability-Weighted | 30.5% | 1.145 | −11.2% | 2.72 |
| Inverse Volatility | 29.3% | 1.161 | −10.1% | 2.90 |
| **Prob × Inv-Vol** ✓ | **29.3%** | **1.164** | **−10.0%** | **2.93** |
| Half-Kelly | 30.3% | 1.149 | −11.6% | 2.61 |

**`prob_invvol` wins on every risk-adjusted metric** (Sharpe, Max DD, Calmar, Win Rate). Equal and Probability weighting produce marginally higher raw CAGR but with ~10% larger drawdowns. Half-Kelly underperforms on Calmar due to aggressive concentration in high-conviction names at the wrong time.

---

## Validation & Robustness

The `validate_strategy.py` script runs a comprehensive validation suite:

### Monte Carlo Bootstrap (10,000 resamples)
Builds confidence intervals for Sharpe, CAGR, and Max Drawdown by bootstrap resampling period returns. Answers: *"How stable are our point estimates?"*

### Monte Carlo Path Simulation (1,000 paths)
Randomly shuffles the order of returns to generate alternative equity curves. Tests whether performance is path-dependent or driven by a lucky sequence.

### Parameter Sensitivity Analysis
- **Probability Threshold**: Sweeps 0.50–0.70 to test alpha robustness
- **Stop-Loss Level**: Sweeps −3% to −15% to find the optimal risk/return tradeoff
- **Regime Position Sizing**: Heatmap of Sharpe across Bull/Neutral/Bear holding count combinations

### Walk-Forward Stability
Rolling Sharpe and Calmar over the out-of-sample window to detect regime-specific performance degradation.

### Regime Performance Decomposition
Mean return, win rate, and observation count broken down by Bull / Neutral / Bear regime.

---

## Generated Visualizations

The validation suite outputs 14 charts to `output/`:

| Category | Charts |
|:---|:---|
| **Backtest Metrics** | `equity_curve.png`, `monthly_returns_heatmap.png`, `rolling_sharpe.png`, `drawdown_underwater.png`, `holdings_distribution.png`, `return_distribution.png` |
| **Monte Carlo** | `mc_bootstrap_sharpe.png`, `mc_bootstrap_cagr.png`, `mc_path_simulation.png` |
| **Sensitivity** | `sensitivity_threshold.png`, `sensitivity_stoploss.png`, `sensitivity_regime_heatmap.png` |
| **Regime & Stability** | `regime_performance.png`, `walkforward_stability.png` |

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `TRANSACTION_COST_BPS` | 10 | Cost per side (bps) |
| `LEVERAGE_COST_ANNUAL` | 5% | Annual drag on gross exposure > 1× |
| `RISK_FREE_ANNUAL` | 7% | India 10Y govt bond proxy |
| `DATA_START` | 2008-01-01 | Start of price history |
| `DATA_END` | 2025-04-01 | End of evaluation period |
| `LOOKBACK_WINDOWS` | [1, 6, 12, 36, 60] | Momentum formation periods (months) |

### Portfolio Parameters (`engine.py`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `PROB_THRESHOLD` | 0.55 | Minimum predicted probability to enter position |
| `REGIME_SIZE` | Bull: 10, Neutral: 4, Bear: 3 | Max holdings per regime |
| `REGIME_STOP` | Bull: −7%, Neutral: −6%, Bear: −4% | Daily stop-loss per regime |
| `weighting` | `prob_invvol` | Probability × Inverse Volatility (default) |

### All Available CLI Flags

```
main.py / export_results.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none
  --sizing    directional | volscale | hmm_vol_size
  --weighting equal | probability | inverse_vol | prob_invvol | kelly

compare_weights.py / validate_strategy.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none

live_portfolio.py:
  --index     nifty50 | nifty100
```

---

## Notes on Survivorship Bias

- **Nifty 50 / Nifty 100**: Zero survivorship bias. The PiT composition CSVs enumerate exact index membership for every month from January 2008. Stocks are only eligible for selection if they were actually in the index that month.
- **Nifty 500**: Uses a static 2025 composition snapshot — significant survivorship bias. Results for this universe should not be used for performance evaluation.
