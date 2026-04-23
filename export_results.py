#!/usr/bin/env python3
"""
export_results.py
=================
Runs the HMM directional strategy (Nifty 100 monthly, 2015-2025) and
outputs a CSV compatible with nexus_evaluator.py.

Output columns:
  Date            — rebalance date (YYYY-MM-DD)
  Period_Return   — net strategy return for that period (decimal, e.g. 0.032)
  Period_Turnover — two-sided portfolio turnover that period (decimal)

Usage:
    source venv/bin/activate
    python3 export_results.py [--output results.csv] [--sizing directional|volscale]
"""

import warnings
import argparse
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import (
    DATA_START, DATA_END,
    HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
    LOOKBACK_WINDOWS
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, compute_target_vol, performance_stats, print_stats
)
from regime import get_regimes


def main():
    parser = argparse.ArgumentParser(description="Export strategy results to CSV for nexus_evaluator")
    parser.add_argument('--output',  type=str, default='results.csv',
                        help="Output CSV path (default: results.csv)")
    parser.add_argument('--sizing',  type=str, default='directional',
                        choices=['directional', 'volscale'],
                        help="Sizing scheme (default: directional)")
    parser.add_argument('--regime',  type=str, default='learned_hmm',
                        choices=['learned_hmm', 'fixed_hmm', 'none'],
                        help="HMM regime method (default: learned_hmm)")
    parser.add_argument('--weighting', type=str, default='prob_invvol',
                        choices=['equal', 'probability', 'inverse_vol', 'prob_invvol', 'kelly'],
                        help="Cross-sectional position weighting method (default: prob_invvol)")
    args = parser.parse_args()

    # ---- 1) Load data (uses local cache — no downloads for price data) -----
    print("Loading Nifty 100 PiT data (from cache)...")
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    fwd_returns   = compute_forward_returns(monthly_prices)
    momentum_dict = compute_all_momentum(monthly_prices, LOOKBACK_WINDOWS)
    stacked       = build_stacked_dataset(monthly_prices, mask, fwd_returns,
                                          momentum_dict, LOOKBACK_WINDOWS)
    res_df        = run_expanding_window(stacked, min_train_months=60)
    if res_df is None:
        print("ERROR: walk-forward returned None. Exiting.")
        return

    rebal_dates   = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')

    # ---- 2) Run strategy ---------------------------------------------------
    if args.sizing == 'directional':
        print(f"Running HMM directional ({args.regime}) with {args.weighting} weighting...")
        regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=args.regime)
        port, counts, extra = simulate_portfolio(
            res_df, regimes, daily_prices, sizing_scheme='directional', weighting=args.weighting
        )
        label = f"HMM directional ({args.regime}) - {args.weighting}"

    else:  # volscale
        print(f"Running pure vol scaling (126d, target_vol=0.20) with {args.weighting} weighting...")
        target_vol, daily_rets = compute_target_vol(res_df, daily_prices)
        target_vol = 0.20  # fixed spec
        vp = {'target_vol': target_vol, 'daily_rets': daily_rets}
        port, counts, extra = simulate_portfolio(
            res_df, {}, daily_prices,
            sizing_scheme='volscale', volscale_params=vp, weighting=args.weighting
        )
        label = f"Pure vol scaling (126d) - {args.weighting}"

    # ---- 3) Print full performance summary ---------------------------------
    stats = performance_stats(port, periods_per_year=12)
    print_stats(stats, f"Nifty 100 Monthly — {label}", counts, freq_label="mo")

    # ---- 4) Build output CSV -----------------------------------------------
    turnover_track = extra['turnover_track']

    # Align lengths (simulate_portfolio may have one fewer if last date has no next period)
    n = min(len(port), len(turnover_track))
    out_df = pd.DataFrame({
        'Date':             port.index[:n].strftime('%Y-%m-%d'),
        'Period_Return':    port.values[:n].round(6),
        'Period_Turnover':  np.array(turnover_track[:n]).round(6),
    })

    out_df.to_csv(args.output, index=False)
    print(f"\n✓ Saved {len(out_df)} rows → {args.output}")
    print(f"  Date range  : {out_df['Date'].iloc[0]}  →  {out_df['Date'].iloc[-1]}")
    print(f"  Avg return  : {out_df['Period_Return'].mean():.4f}  "
          f"(std {out_df['Period_Return'].std():.4f})")
    print(f"  Avg turnover: {out_df['Period_Turnover'].mean():.4f}  "
          f"(std {out_df['Period_Turnover'].std():.4f})")
    print(f"\nRun:  python3 utils/nexus_evaluator.py {args.output}")


if __name__ == '__main__':
    main()
