import os
import sys
import argparse
import warnings
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
    simulate_portfolio, performance_stats, print_stats
)
from regime import get_regimes


def run_pit_universe(universe_name, csv_paths, is_weekly=False, regime_method='fixed_hmm'):
    """Run point-in-time universe evaluation (Nifty 50 / Nifty 100)."""
    banner = "=" * 80
    freq_label = "WEEKLY" if is_weekly else "MONTHLY"
    print(f"\n{banner}\nCATBOOST HMM STRATEGY — {universe_name} (PiT) {freq_label}\n{banner}")

    monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    
    if is_weekly:
        prices = daily_prices.resample('W-FRI').last().dropna(how='all')
        mask = mask.resample('W-FRI').ffill().reindex(prices.index).ffill().fillna(False)
        periods_per_year = 52
        lookbacks = [1, 4, 12, 24, 52]
        min_train = 156
    else:
        prices = monthly_prices
        periods_per_year = 12
        lookbacks = LOOKBACK_WINDOWS
        min_train = 60

    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)

    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)
    if res_df is None: return

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)
    
    port_long, c_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = performance_stats(port_long, periods_per_year)
    
    port_ls, c_ls, s_ls = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = performance_stats(port_ls, periods_per_year)

    short_freq = "wk" if is_weekly else "mo"
    print_stats(stats_long, f"{universe_name} {freq_label} — LONG ONLY", c_long, freq_label=short_freq)
    print_stats(stats_ls, f"{universe_name} {freq_label} — LONG + SHORT", c_ls, s_ls, freq_label=short_freq)


def run_static_universe(universe_name, ticker_selection, is_weekly=False, regime_method='fixed_hmm'):
    """Run static cached universe evaluation (Nifty 250 / Nifty 500)."""
    banner = "=" * 80
    freq_label = "WEEKLY" if is_weekly else "MONTHLY"
    print(f"\n{banner}\nCATBOOST HMM STRATEGY — {universe_name} {freq_label}\n{banner}")

    if not os.path.exists("daily_cache_nifty500.csv"):
        print("Missing daily_cache_nifty500.csv — run prepare_nifty500.py first.")
        return

    daily_full = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
    
    if ticker_selection == 'nifty500':
        daily_prices = daily_full
    elif ticker_selection == 'nifty250':
        top250_tickers = daily_full.notna().sum().nlargest(250).index.tolist()
        daily_prices = daily_full[top250_tickers]
    else:
        return
        
    prices = daily_prices.resample('W-FRI' if is_weekly else 'ME').last().dropna(how='all')
    periods_per_year = 52 if is_weekly else 12
    lookbacks = [1, 4, 12, 24, 52] if is_weekly else [1, 3, 6, 12]
    min_train = 156 if is_weekly else 36

    mask = prices.notna()
    fwd_returns  = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)

    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)
    if res_df is None: return

    eval_start = pd.Timestamp('2021-01-01')
    res_df = res_df[res_df['date'] >= eval_start]

    rebal_dates = sorted(res_df['date'].unique())
    if not rebal_dates: return
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)
    
    port_long, c_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = performance_stats(port_long, periods_per_year)
    
    port_ls, c_ls, s_ls = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = performance_stats(port_ls, periods_per_year)

    short_freq = "wk" if is_weekly else "mo"
    print_stats(stats_long, f"{universe_name} {freq_label} — LONG ONLY", c_long, freq_label=short_freq)
    print_stats(stats_ls, f"{universe_name} {freq_label} — LONG + SHORT", c_ls, s_ls, freq_label=short_freq)


def main():
    parser = argparse.ArgumentParser(description="Run Calmar Momentum Strategy")
    parser.add_argument(
        '--index', type=str, default='all',
        choices=['nifty50', 'nifty100', 'nifty250', 'nifty500', 'all'],
        help="Index to run backtest on"
    )
    parser.add_argument(
        '--regime', type=str, default='fixed_hmm',
        choices=['fixed_hmm', 'learned_hmm', 'none'],
        help="Regime switching methodology to apply"
    )
    args = parser.parse_args()

    run_50 = False
    run_100 = False
    run_250 = False
    run_500 = False

    if args.index == 'all':
        run_50 = run_100 = run_250 = run_500 = True
    else:
        if args.index == 'nifty50': run_50 = True
        elif args.index == 'nifty100': run_100 = True
        elif args.index == 'nifty250': run_250 = True
        elif args.index == 'nifty500': run_500 = True

    if run_50:
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV], is_weekly=False, regime_method=args.regime)
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV], is_weekly=True, regime_method=args.regime)
    if run_100:
        run_pit_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV], is_weekly=False, regime_method=args.regime)
        run_pit_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV], is_weekly=True, regime_method=args.regime)
    if run_250:
        run_static_universe("NIFTY 250 (Proxy)", 'nifty250', is_weekly=False, regime_method=args.regime)
        run_static_universe("NIFTY 250 (Proxy)", 'nifty250', is_weekly=True, regime_method=args.regime)
    if run_500:
        run_static_universe("NIFTY 500", 'nifty500', is_weekly=False, regime_method=args.regime)
        run_static_universe("NIFTY 500", 'nifty500', is_weekly=True, regime_method=args.regime)


if __name__ == '__main__':
    main()
