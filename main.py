import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
import logging

warnings.filterwarnings('ignore')

from config import (
    DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
    LOOKBACK_WINDOWS
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats, print_stats,
)
from regime import get_regimes


def run_pit_universe(universe_name, csv_paths, is_weekly=False, regime_method='learned_hmm'):
    """Run point-in-time universe evaluation (Nifty 50 / Nifty 100)."""
    banner = "=" * 80
    freq_label = "WEEKLY" if is_weekly else "MONTHLY"
    print(f"\n{banner}\nCATBOOST MOMENTUM — {universe_name} (PiT) {freq_label}\n{banner}")

    monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    if is_weekly:
        prices = daily_prices.resample('W-FRI').last().dropna(how='all')
        mask = mask.resample('W-FRI').ffill().reindex(prices.index).ffill().fillna(False)
        periods_per_year = 52
        lookbacks = [1, 4, 12, 24, 52]
        min_train = 156
        entry_prices = None
    else:
        prices = monthly_prices
        periods_per_year = 12
        lookbacks = LOOKBACK_WINDOWS
        min_train = 48
        entry_prices = monthly_prices

    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)
    if res_df is None:
        return

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    freq_label_short = "wk" if is_weekly else "mo"

    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)
    port, counts, _ = simulate_portfolio(res_df, regimes, daily_prices,
                                         monthly_prices=entry_prices)
    stats = performance_stats(port, periods_per_year)
    print_stats(stats, f"{universe_name} {freq_label} — LONG ONLY", counts,
                freq_label=freq_label_short)


def main():
    parser = argparse.ArgumentParser(description="CatBoost Momentum Strategy")
    parser.add_argument(
        '--index', type=str, default='all',
        choices=['nifty50', 'nifty100', 'all'],
        help="Universe to backtest (default: all)"
    )
    parser.add_argument(
        '--regime', type=str, default='learned_hmm',
        choices=['fixed_hmm', 'learned_hmm', 'none'],
        help="HMM regime method (default: learned_hmm)"
    )
    args = parser.parse_args()

    run_50 = run_100 = False
    if args.index == 'all':
        run_50 = run_100 = True
    elif args.index == 'nifty50':  run_50  = True
    elif args.index == 'nifty100': run_100 = True

    if run_50:
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV],
                         is_weekly=False, regime_method=args.regime)
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV],
                         is_weekly=True,  regime_method=args.regime)
    if run_100:
        run_pit_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV],
                         is_weekly=False, regime_method=args.regime)


if __name__ == '__main__':
    main()
