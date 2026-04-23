#!/usr/bin/env python3
"""
ablation_lookbacks.py
=====================
Leave-one-out ablation on LOOKBACK_WINDOWS = [1, 3, 6, 12, 36, 60].

For each variant (drop one window at a time), runs the full
walk-forward pipeline and reports CAGR, Sharpe, Max DD, Calmar.

Usage:
    source venv/bin/activate
    python3 ablation_lookbacks.py
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import (
    DATA_START, DATA_END,
    HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats
)
from regime import get_regimes

FULL_WINDOWS = [1, 3, 6, 12, 36, 60]

# Pre-load data once — expensive; reuse across all ablation runs
print("Loading Nifty 100 PiT data (from cache)...")
csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
fwd_returns  = compute_forward_returns(monthly_prices)

print("\n" + "=" * 72)
print("  LOOKBACK ABLATION  —  [1, 3, 6, 12, 36, 60]:  leave-one-out")
print("=" * 72)
print(f"  {'Config':<25s} {'CAGR':>7s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s}  {'Vol':>7s}")
print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*8}  {'-'*7}")

results_rows = []

for drop in FULL_WINDOWS:
    windows = [w for w in FULL_WINDOWS if w != drop]
    label   = f"drop {drop}M → {windows}"

    momentum_dict = compute_all_momentum(monthly_prices, windows)
    stacked       = build_stacked_dataset(monthly_prices, mask, fwd_returns,
                                          momentum_dict, windows)
    res_df        = run_expanding_window(stacked, min_train_months=60)
    if res_df is None:
        print(f"  {label:<25s}  (insufficient data — skipping)")
        continue

    rebal_dates   = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes       = get_regimes(rebal_dates, padding_start, DATA_END, method='learned_hmm')
    port, _, _    = simulate_portfolio(res_df, regimes, daily_prices,
                                       sizing_scheme='directional',
                                       weighting='prob_invvol')
    s = performance_stats(port, periods_per_year=12)
    print(f"  {str(windows):<25s} {s['ann']:>6.2f}% {s['sharpe']:>8.3f} "
          f"{s['dd']:>7.2f}% {s['calmar']:>8.3f}  {s['vol']:>6.2f}%")
    results_rows.append({'dropped': drop, 'windows': str(windows), **s})

print("=" * 72)

# Find best by Calmar
if results_rows:
    best = max(results_rows, key=lambda r: r['calmar'])
    print(f"\n  ✓ Best Calmar: drop {best['dropped']}M  →  "
          f"CAGR {best['ann']:.2f}%  Sharpe {best['sharpe']:.3f}  Calmar {best['calmar']:.3f}")
    best_s = min(results_rows, key=lambda r: abs(r['calmar'] - best['calmar']))

    # Best by Sharpe
    best_sr = max(results_rows, key=lambda r: r['sharpe'])
    print(f"  ✓ Best Sharpe: drop {best_sr['dropped']}M  →  "
          f"CAGR {best_sr['ann']:.2f}%  Sharpe {best_sr['sharpe']:.3f}  Calmar {best_sr['calmar']:.3f}")
