#!/usr/bin/env python3
"""
run_nifty_extended.py
=====================
Runs the strategy on Nifty 500 (monthly) and Nifty 250 (weekly) universes
using the static nifty500 cache. Evaluates OOS from 2020-01-01 onward only
to limit survivorship-bias exposure (the cache is a 2025 static snapshot).

WARNING: These are NOT survivorship-bias-free results. The Nifty 100 monthly
results remain the clean benchmark. These figures are directional only.

Usage:
    source venv/bin/activate
    python3 run_nifty_extended.py
"""

import warnings
import sys
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import RISK_FREE_ANNUAL, LOOKBACK_WINDOWS
from features import compute_all_momentum
from data_fetcher import compute_forward_returns
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats, print_stats
)
from regime import get_regimes

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_PATH   = 'daily_cache_nifty500.csv'
OOS_START    = '2020-01-01'   # Only show results from here — limits survivorship bias
NIFTY50_TICKER = '^NSEI'

# Weekly lookbacks in weeks ≈ [1m, 3m, 6m, 12m, 24m]
# NOTE: 552 total weeks in cache (2015-2025); keep lookbacks under ~100 to avoid NaN-out
WEEKLY_LOOKBACKS = [4, 13, 26, 52, 104]


def load_cache():
    print(f"Loading {CACHE_PATH}...")
    df = pd.read_csv(CACHE_PATH, index_col=0, parse_dates=True)
    df = df.sort_index()
    print(f"  Shape: {df.shape}  |  {df.index[0].date()} → {df.index[-1].date()}")
    return df


def run_universe(name, prices_daily, freq, lookbacks, min_train, top_n=None):
    """
    Core runner for a single universe/frequency combo.

    Args:
        name        : display name
        prices_daily: daily price DataFrame (all tickers)
        freq        : 'ME' (monthly end) or 'W-FRI' (weekly Friday)
        lookbacks   : list of int lookback periods (in freq units)
        min_train   : minimum number of periods before first OOS prediction
        top_n       : if set, use top_n tickers by observation count
    """
    print(f"\n{'='*72}")
    print(f"  {name}")
    print(f"{'='*72}")

    if top_n is not None:
        selected = prices_daily.count().nlargest(top_n).index.tolist()
        prices_daily = prices_daily[selected]
        print(f"  Using top {top_n} tickers by observation count")

    # Resample to target frequency
    prices = prices_daily.resample(freq).last().dropna(how='all')
    print(f"  Resampled to {freq}: {prices.shape[0]} periods × {prices.shape[1]} tickers")

    mask = prices.notna()
    fwd_returns   = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df  = run_expanding_window(stacked, min_train_months=min_train)

    if res_df is None:
        print("  ERROR: Not enough data. Skipping.")
        return None

    all_dates   = sorted(res_df['date'].unique())
    oos_dates   = [d for d in all_dates if pd.Timestamp(d) >= pd.Timestamp(OOS_START)]
    print(f"  Walk-forward dates: {len(all_dates)} total  |  {len(oos_dates)} OOS (from {OOS_START})")

    if len(oos_dates) == 0:
        print("  No OOS dates — skipping.")
        return None

    # Regime detection on Nifty 50 (still using monthly HMM, applied per rebalance date)
    rebal_dates   = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes       = get_regimes(rebal_dates, padding_start, '2025-04-01', method='learned_hmm')

    # Full simulation (includes warm-up)
    port, counts, extra = simulate_portfolio(
        res_df, regimes, prices_daily,
        sizing_scheme='directional', weighting='prob_invvol'
    )

    # Trim to OOS only
    port_oos   = port[port.index >= OOS_START]
    counts_oos = [c for d, c in zip(rebal_dates, counts) if pd.Timestamp(d) >= pd.Timestamp(OOS_START)]

    if len(port_oos) == 0:
        print("  No OOS returns to evaluate.")
        return None

    periods_per_year = 52 if freq == 'W-FRI' else 12
    stats = performance_stats(port_oos, periods_per_year=periods_per_year)
    print_stats(
        stats,
        f"{name} [OOS {OOS_START} → 2025]",
        counts_oos,
        freq_label='wk' if freq == 'W-FRI' else 'mo'
    )
    return stats


def main():
    daily = load_cache()

    # ── Run 1: Nifty 500 Monthly ───────────────────────────────────────────────
    stats_500m = run_universe(
        name        = 'Nifty 500 — Monthly — Long Only',
        prices_daily= daily,
        freq        = 'ME',
        lookbacks   = LOOKBACK_WINDOWS,      # [1, 6, 12, 36, 60] months
        min_train   = 36,                    # 3 years of monthly training (cache starts 2015)
        top_n       = None,                  # use all ~497 tickers
    )

    # ── Run 2: Nifty 250 Weekly ────────────────────────────────────────────────
    stats_250w = run_universe(
        name        = 'Nifty 250 — Weekly — Long Only',
        prices_daily= daily,
        freq        = 'W-FRI',
        lookbacks   = WEEKLY_LOOKBACKS,      # [4, 13, 26, 52, 104] weeks ≈ 1/3/6/12/24m
        min_train   = 104,                   # 2 years × 52 weeks training
        top_n       = 250,                   # top 250 by observation count
    )

    # ── Summary ────────────────────────────────────────────────────────────────
    print(f"\n{'='*72}")
    print(f"  EXTENDED UNIVERSE SUMMARY  (OOS from {OOS_START} — ⚠ survivorship bias)")
    print(f"{'='*72}")
    print(f"  {'Universe':<30s} {'CAGR':>7s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s}")
    print(f"  {'-'*30} {'-'*7} {'-'*8} {'-'*8} {'-'*8}")

    rows = [
        ('Nifty 100 Monthly (clean)',   29.29, 1.164, -10.00, 2.929),
    ]
    if stats_500m:
        rows.append(('Nifty 500 Monthly ⚠', stats_500m['ann'], stats_500m['sharpe'],
                     stats_500m['dd'], stats_500m['calmar']))
    if stats_250w:
        rows.append(('Nifty 250 Weekly ⚠', stats_250w['ann'], stats_250w['sharpe'],
                     stats_250w['dd'], stats_250w['calmar']))

    for label, cagr, sr, dd, cal in rows:
        print(f"  {label:<30s} {cagr:>6.2f}% {sr:>8.3f} {dd:>7.2f}% {cal:>8.3f}")


if __name__ == '__main__':
    main()
