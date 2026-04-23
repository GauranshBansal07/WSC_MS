# WSC_MS — Momentum Strategy (CatBoost + Learned HMM)

A cross-sectional momentum trading strategy for Indian equity markets.
Uses CatBoost to rank stocks by predicted relative momentum, then a
walk-forward bivariate Gaussian HMM to detect the prevailing volatility
regime and scale position count and stop-losses accordingly.

---

## Strategy Overview

| Component | Detail |
|:---|:---|
| **Universe** | Nifty 100 (Nifty 50 + Nifty Next 50), point-in-time composition |
| **Signal** | Cross-sectional momentum — 1M, 6M, 12M, 36M, 60M lookbacks |
| **Classifier** | CatBoost (depth=4, lr=0.05, 150 iters) — predicts if stock beats cross-sectional median next month |
| **Walk-forward** | Expanding window, minimum 60 months training before first prediction |
| **Regime filter** | Bivariate Gaussian HMM on (Nifty 50 daily log-return, 20d realized vol), 3 states, refit every 12 months with 5 random restarts |
| **Sizing** | States sorted by return mean: Bull (10 names), Neutral (4), Bear (3) |
| **Weighting** | `prob_invvol` — probability × inverse-volatility hybrid (normalized) |
| **Stop-losses** | Daily path stop: −7% (Bull), −6% (Neutral), −4% (Bear) |
| **Costs** | 10 bps per side transaction cost; 5% p.a. leverage cost on gross exposure > 1× |

**Data period:** Historical PiT composition from January 2008; constituent
prices fetched from yfinance and cached locally. Non-survivorship-biased
PiT CSVs ensure no future index entrants contaminate historical signals.

**Out-of-sample period:** January 2018 – March 2025 (~7 years, 86 monthly
rebalances) after the mandatory 60-month training warm-up.

---

## Key Results — Nifty 100 Monthly Long-Only

### Primary variant: HMM Directional + prob_invvol (current default)

| Metric | Value |
|:---|:---:|
| **CAGR** | **29.29%** |
| **Volatility** | 17.30% |
| **Sharpe Ratio** | **1.164** |
| **Max Drawdown** | −10.00% |
| **Calmar Ratio** | **2.929** |
| Win Rate | 69.77% |
| Avg positions held | ~6 / month |

### Comparison: Sizing Scheme Head-to-Head

| Metric | HMM Directional (10:4:3) | VolScale 126d | HMM 1/σ Dynamic |
|:---|:---:|:---:|:---:|
| CAGR | **29.29%** | 22.08% | 28.92% |
| Sharpe | **1.164** | 1.053 | 1.136 |
| Max DD | **−10.00%** | −11.17% | −11.35% |
| Calmar | **2.929** | 1.977 | 2.548 |

**Verdict:** Static HMM 10:4:3 regime-gating is decisively superior to both
continuous vol-scaling and smooth inverse-volatility sizing. The large
Bull→Neutral step (10→4 names) is the primary risk-management mechanism and
cannot be replicated by a continuous function.

---

## Sizing Schemes

### `--sizing directional` *(default)*

Regime label from the walk-forward HMM controls position count and stop-loss:

| HMM State | Regime Label | Positions | Daily Stop |
|:---|:---:|:---:|:---:|
| Highest mean log-ret | **Bull** | 10 | −7% |
| Middle mean log-ret | **Neutral** | 4 | −6% |
| Lowest mean log-ret | **Bear** | 3 | −4% |

### `--sizing volscale`

Barroso-Santa-Clara (JFE 2015) continuous scaling — 10 names always selected,
gross exposure scaled between `[0.30×, 1.25×]` based on trailing 126-day
realized vol. Target vol = 20% (median of unscaled strategy's rolling vol).

### `--sizing hmm_vol_size`

Experimental. Uses per-state fitted log-return standard deviations from the
HMM covariance matrix to derive position counts via `size = round(1/σ_state / max(1/σ) × 10)`.
Marginally outperforms volscale but underperforms the static 10:4:3 gating.

---

## Weighting Methods

Position weights within the selected book can be configured with `--weighting`:

| Method | Key | Description |
|:---|:---:|:---|
| Equal weight | `equal` | 1/N all positions |
| Probability | `probability` | Weights ∝ CatBoost pred_prob |
| Inverse volatility | `inverse_vol` | Weights ∝ 1/σ_60d |
| **Prob × Inv-Vol** | **`prob_invvol`** | **Weights ∝ pred_prob / σ_60d (default)** |
| Half-Kelly | `kelly` | Fractional Kelly from pred_prob |

Use `compare_weights.py` to benchmark all five methods head-to-head.

---

## Repository Structure

```
.
├── config.py                  — Global parameters (lookbacks, costs, rates)
├── data_fetcher.py            — Price fetching, caching, forward returns
├── features.py                — Momentum feature computation
├── engine.py                  — Core backtest engine: walk-forward CatBoost,
│                                HMM directional/volscale/hmm_vol_size sizing,
│                                prob_invvol weighting, daily stop-loss, tx costs
├── regime.py                  — Walk-forward HMM regime detection
│                                (get_regimes, get_regimes_and_vol_sizes)
├── main.py                    — CLI runner for all indices/sizing schemes
├── export_results.py          — Exports Date / Period_Return / Period_Turnover CSV
│                                for external evaluators (nexus_evaluator.py)
├── live_portfolio.py          — Production month-start signal generator:
│                                downloads live prices, trains CatBoost, evaluates
│                                HMM regime, outputs ticker weights + stop-losses
├── validate_strategy.py       — Full validation suite: Monte Carlo bootstrap (10k),
│                                path simulation, parameter sensitivity analysis,
│                                regime-conditional statistics, 18-panel chart output
├── compare_weights.py         — Head-to-head benchmark of all 5 weighting methods
│                                with equity curve, metric bar chart, risk-return scatter
├── prepare_nifty500.py        — Builds daily/monthly cache from raw 5-min tick CSVs
└── data/
    ├── historical_composition.csv      — Point-in-time Nifty 50 composition (Jan 2008 →)
    └── nifty_next_50_composition.csv   — Point-in-time Nifty Next 50 composition (Jan 2008 →)
```

---

## Running the Strategy

### Installation

```bash
pip install catboost scikit-learn pandas numpy yfinance scipy hmmlearn seaborn matplotlib
```

### Backtest (recommended)

```bash
python3 main.py --index nifty100 --sizing directional --weighting prob_invvol
```

### Export CSV for external evaluator

```bash
python3 export_results.py --sizing directional --output results.csv
python3 utils/nexus_evaluator.py results.csv
```

### Generate live month-start portfolio (production use)

```bash
python3 live_portfolio.py --index nifty100
```

Outputs exact ticker allocations, weights, and stop-loss levels to execute
at the open on the first trading day of the month.

### Run full validation suite

```bash
python3 validate_strategy.py
```

Generates 18 diagnostic charts including:
- Monte Carlo bootstrap distribution (10k iterations)
- Simulated equity path fan chart
- Parameter sensitivity heatmaps (stop-loss, position count)
- Rolling Sharpe / Calmar over time
- Regime-conditional return distributions

### Benchmark weighting methods

```bash
python3 compare_weights.py --index nifty100
```

Saves equity curve, metric bar chart, and risk-return scatter to `output/`.

### All available CLI flags

```
main.py / export_results.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none   (with --sizing directional)
  --sizing    directional | volscale | hmm_vol_size
  --weighting equal | probability | inverse_vol | prob_invvol | kelly

compare_weights.py:
  --index     nifty50 | nifty100
  --regime    learned_hmm | fixed_hmm | none

live_portfolio.py:
  --index     nifty50 | nifty100
```

---

## Configuration (`config.py`)

| Parameter | Value | Description |
|:---|:---:|:---|
| `TRANSACTION_COST_BPS` | 10 | Cost per side (bps) |
| `LEVERAGE_COST_ANNUAL` | 5% | Annual drag on gross exposure > 1× |
| `RISK_FREE_ANNUAL` | 7% | India 10Y govt bond proxy |
| `DATA_START` | 2008-01-01 | Start of historical composition data |
| `DATA_END` | 2025-04-01 | End of evaluation period |
| `LOOKBACK_WINDOWS` | [1, 6, 12, 36, 60] | Momentum lookback months |

---

## Design Decisions & Research Notes

### Why `[1, 6, 12, 36, 60]` lookbacks?

The 3M lookback was removed because short Nifty momentum is dominated by
mean-reversion, not trend persistence. The 36M and 60M windows capture
structural multi-year trends that are orthogonal to the 1M reversal signal.
This single change accounts for the largest share of the performance uplift
over simpler implementations.

### Why `prob_invvol` weighting?

Uniform (1/N) weighting overweights high-volatility names in the book that
the model is also moderately confident on. `prob_invvol` simultaneously
rewards high-conviction picks and penalizes high idiosyncratic risk, producing
a better risk-adjusted allocation without adding a separate optimization layer.

### Why static 10:4:3 regime sizing?

Empirically verified to outperform both continuous vol-scaling (BSC 2015) and
smooth HMM 1/σ-derived sizes. The sharp Bull→Neutral cut (10→4 names) is the
key protective mechanism — it aggressively reduces exposure precisely when the
HMM detects deteriorating return expectations, not merely rising volatility.

### Stop-loss calibration

Stop-losses are set per-regime: −7% (Bull), −6% (Neutral), −4% (Bear).
These were validated via parameter sensitivity sweeps; they represent the
robust peak of the Calmar surface, deliberately avoiding the steep cliff at
tighter thresholds (≤ −5% across all regimes) where whipsaw losses dominate.

---

## Notes on Survivorship Bias

- **Nifty 50 / Nifty 100**: Zero survivorship bias. The PiT composition CSVs
  enumerate exact index membership for every month from January 2008. Stocks
  are only eligible for selection if they were actually in the index that month.
- **Nifty 500**: Uses a static 2025 composition snapshot → significant
  survivorship bias. Results for this universe should not be used for
  performance evaluation.
