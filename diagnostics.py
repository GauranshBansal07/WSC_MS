#!/usr/bin/env python3
"""
diagnostics.py — post-hoc analysis of the main strategy run.

Modes:
  annual    Year-by-year CAGR / DD / Sharpe / Calmar decomposition.
  lookback  Leave-one-out ablation on LOOKBACK_WINDOWS = [1, 3, 6, 12, 36, 60].

Usage:
  python3 diagnostics.py --mode annual
  python3 diagnostics.py --mode lookback
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config import (DATA_START, DATA_END,
                    HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (build_stacked_dataset, run_expanding_window,
                    simulate_portfolio, performance_stats)
from regime import get_regimes


CSV_PATHS = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
FULL_WINDOWS = [1, 3, 6, 12, 36, 60]


def _load():
    monthly_prices, mask = fetch_monthly_prices(CSV_PATHS, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    fwd = compute_forward_returns(monthly_prices)
    return monthly_prices, mask, daily_prices, fwd


# ---------------------------------------------------------------------------
# Mode: annual
# ---------------------------------------------------------------------------

def run_annual():
    print("Loading data...")
    monthly_prices, mask, daily_prices, fwd = _load()
    mom = compute_all_momentum(monthly_prices, FULL_WINDOWS)

    print("Building stacked dataset and running expanding window...")
    stacked = build_stacked_dataset(monthly_prices, mask, fwd, mom, FULL_WINDOWS)
    res_df = run_expanding_window(stacked, min_train_months=60)

    rebal_dates = sorted(res_df['date'].unique())
    padding = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding, DATA_END, method='learned_hmm')

    print("Simulating portfolio...")
    port, _, _ = simulate_portfolio(res_df, regimes, daily_prices,
                                    sizing_scheme='directional', weighting='prob_invvol')

    print("\n--- ANNUAL BREAKDOWN ---")
    rows = []
    for y in port.index.year.unique():
        port_y = port[port.index.year == y]
        if len(port_y) < 10:
            continue
        cum = (1 + port_y).cumprod()
        ret = cum.iloc[-1] - 1
        dd_min = ((cum - cum.cummax()) / cum.cummax()).min()
        days = (port_y.index[-1] - port_y.index[0]).days
        if days < 30:
            continue
        ann = (1 + ret) ** (365.25 / days) - 1
        vol = port_y.std() * np.sqrt(252)
        rows.append({
            'Year': y,
            'Return': ret,
            'Ann Return': ann,
            'MaxDD': dd_min,
            'Calmar': ann / abs(dd_min) if abs(dd_min) > 0 else np.nan,
            'Sharpe': ann / vol if vol > 0 else np.nan,
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("------------------------")


# ---------------------------------------------------------------------------
# Mode: lookback
# ---------------------------------------------------------------------------

def run_lookback():
    print("Loading Nifty 100 PiT data (from cache)...")
    monthly_prices, mask, daily_prices, fwd = _load()

    print("\n" + "=" * 72)
    print(f"  LOOKBACK ABLATION  —  {FULL_WINDOWS}:  leave-one-out")
    print("=" * 72)
    print(f"  {'Config':<25s} {'CAGR':>7s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s}  {'Vol':>7s}")
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*8}  {'-'*7}")

    results = []
    for drop in FULL_WINDOWS:
        windows = [w for w in FULL_WINDOWS if w != drop]
        mom = compute_all_momentum(monthly_prices, windows)
        stacked = build_stacked_dataset(monthly_prices, mask, fwd, mom, windows)
        res_df = run_expanding_window(stacked, min_train_months=60)
        if res_df is None:
            print(f"  {'drop ' + str(drop):<25s}  (insufficient data — skipping)")
            continue
        rebal = sorted(res_df['date'].unique())
        padding = (pd.to_datetime(rebal[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        regimes = get_regimes(rebal, padding, DATA_END, method='learned_hmm')
        port, _, _ = simulate_portfolio(res_df, regimes, daily_prices,
                                        sizing_scheme='directional', weighting='prob_invvol')
        s = performance_stats(port, periods_per_year=12)
        print(f"  {str(windows):<25s} {s['ann']:>6.2f}% {s['sharpe']:>8.3f} "
              f"{s['dd']:>7.2f}% {s['calmar']:>8.3f}  {s['vol']:>6.2f}%")
        results.append({'dropped': drop, **s})
    print("=" * 72)

    if results:
        best_c  = max(results, key=lambda r: r['calmar'])
        best_sr = max(results, key=lambda r: r['sharpe'])
        print(f"\n  ✓ Best Calmar: drop {best_c['dropped']}M  →  "
              f"CAGR {best_c['ann']:.2f}%  Sharpe {best_c['sharpe']:.3f}  Calmar {best_c['calmar']:.3f}")
        print(f"  ✓ Best Sharpe: drop {best_sr['dropped']}M  →  "
              f"CAGR {best_sr['ann']:.2f}%  Sharpe {best_sr['sharpe']:.3f}  Calmar {best_sr['calmar']:.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Post-hoc diagnostics for the momentum strategy")
    p.add_argument('--mode', choices=['annual', 'lookback'], required=True,
                   help="annual=per-year breakdown | lookback=leave-one-out window ablation")
    args = p.parse_args()

    if   args.mode == 'annual':   run_annual()
    elif args.mode == 'lookback': run_lookback()


if __name__ == '__main__':
    main()
