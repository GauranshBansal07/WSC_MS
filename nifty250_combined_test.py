"""
Nifty 250 — Monthly + Weekly Long Only and Long/Short Strategy
==============================================================
Uses the Nifty 500 price cache but selects the top 250 stocks
by data completeness (most consecutive price observations), which
acts as a reasonable proxy for the Nifty 100 + Midcap 150 universe.

Runs 4 strategies in one go:
  1. Monthly — Long Only
  2. Monthly — Long + Short
  3. Weekly  — Long Only
  4. Weekly  — Long + Short
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from config import DATA_END, RISK_FREE_ANNUAL, TRANSACTION_COST_BPS
from data_fetcher import compute_forward_returns
from features import compute_all_momentum
from catboost_test import (
    build_stacked_dataset, run_expanding_window, get_macro_regimes,
    simulate_portfolio
)


# ---- Performance stats helpers --------------------------------------------
def perf_stats_monthly(port_ret):
    rf = (1 + RISK_FREE_ANNUAL) ** (1.0 / 12.0) - 1.0
    total = (np.prod(1 + port_ret) - 1) * 100
    ann   = (np.prod(1 + port_ret) ** (12 / len(port_ret)) - 1) * 100 if len(port_ret) > 0 else 0
    vol   = port_ret.std() * np.sqrt(12) * 100
    sharpe = ((port_ret.mean() - rf) / port_ret.std() * np.sqrt(12)) if port_ret.std() > 0 else 0
    cum = (1 + port_ret).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = ann / abs(dd) if dd != 0 else 0
    win = (port_ret > 0).mean() * 100
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, dd=dd, calmar=calmar, win=win)


def perf_stats_weekly(port_ret):
    rf = (1 + RISK_FREE_ANNUAL) ** (1.0 / 52.0) - 1.0
    total = (np.prod(1 + port_ret) - 1) * 100
    ann   = (np.prod(1 + port_ret) ** (52 / len(port_ret)) - 1) * 100 if len(port_ret) > 0 else 0
    vol   = port_ret.std() * np.sqrt(52) * 100
    sharpe = ((port_ret.mean() - rf) / port_ret.std() * np.sqrt(52)) if port_ret.std() > 0 else 0
    cum = (1 + port_ret).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = ann / abs(dd) if dd != 0 else 0
    win = (port_ret > 0).mean() * 100
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, dd=dd, calmar=calmar, win=win)


def print_stats(stats, label, holdings_long, holdings_short=None, freq='mo'):
    banner = "=" * 80
    print(f"\n{banner}\nPORTFOLIO RESULT — {label}\n{banner}")
    print(f"  Avg long  holdings / {freq}: {np.mean(holdings_long):.1f}")
    if holdings_short is not None:
        print(f"  Avg short holdings / {freq}: {np.mean(holdings_short):.1f}")
    print(f"  Total Return (%):       {stats['total']:.2f}")
    print(f"  Annualized Return (%):  {stats['ann']:.2f}")
    print(f"  Volatility (%):         {stats['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats['calmar']:.3f}")
    print(f"  Win Rate (%):           {stats['win']:.1f}")


def run_universe(prices_monthly, daily_prices, lookbacks, min_train_months,
                 universe_name, freq_label, perf_fn, freq_short):
    banner = "=" * 80
    print(f"\n{banner}\nCATBOOST STRATEGY — {universe_name}\n{banner}")

    mask = prices_monthly.notna()
    fwd_returns  = compute_forward_returns(prices_monthly)
    momentum_dict = compute_all_momentum(prices_monthly, lookbacks)

    print(f"\n[1] Building feature panel...")
    stacked = build_stacked_dataset(prices_monthly, mask, fwd_returns, momentum_dict, lookbacks)
    print(f"  Observations: {len(stacked)}  |  Avg {freq_label} universe: "
          f"{stacked.groupby(level=0).size().mean():.1f}")

    print(f"\n[2] Walk-forward CatBoost ({min_train_months}-period warm-up)...")
    res_df = run_expanding_window(stacked, min_train_months=min_train_months)
    if res_df is None:
        return

    eval_start = pd.Timestamp('2021-01-01')
    res_df = res_df[res_df['date'] >= eval_start]
    if res_df.empty:
        print("  No predictions in 2021-2025 range.")
        return

    acc  = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    print(f"\n[3] Classifying macro regimes...")
    rebal_dates   = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_macro_regimes(rebal_dates, padding_start, DATA_END)

    print(f"\n[4] Simulating Long Only...")
    port_long, cnt_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = perf_fn(port_long)

    print(f"\n[5] Simulating Long + Short...")
    port_ls, cnt_ls, cnt_s = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = perf_fn(port_ls)

    print_stats(stats_long, f"NIFTY 250 — {universe_name} — LONG ONLY", cnt_long, freq=freq_short)
    print_stats(stats_ls, f"NIFTY 250 — {universe_name} — LONG + SHORT", cnt_ls, cnt_s, freq=freq_short)

    return stats_long, stats_ls


def main():
    banner2 = "=" * 80
    print(f"\n{banner2}")
    print(f"NIFTY 250 — MONTHLY & WEEKLY COMPARISON (2021-2025)")
    print(f"{banner2}")

    # --- Load the Nifty 500 cache and carve out the Top 250 ----------------
    # "Top 250" = stocks with most complete price history (largest/most liquid)
    print("\n[0] Loading Nifty 500 cache and selecting top-250 by data completeness...")
    if not os.path.exists("daily_cache_nifty500.csv"):
        print("Missing daily_cache_nifty500.csv — run prepare_nifty500.py first.")
        return

    daily_full = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
    obs_counts = daily_full.notna().sum()
    top250_tickers = obs_counts.nlargest(250).index.tolist()
    daily_prices = daily_full[top250_tickers]

    # Resample daily → monthly
    monthly_prices = daily_prices.resample('ME').last().dropna(how='all')
    # Resample daily → weekly (Friday close)
    weekly_prices  = daily_prices.resample('W-FRI').last().dropna(how='all')

    print(f"  Daily universe: {daily_prices.shape[1]} stocks")
    print(f"  Monthly index:  {len(monthly_prices)} months")
    print(f"  Weekly index:   {len(weekly_prices)} weeks")

    # --- Monthly run (lookbacks in months) ----------------------------------
    MONTHLY_LOOKBACKS = [1, 3, 6, 12]
    run_universe(
        prices_monthly  = monthly_prices,
        daily_prices    = daily_prices,
        lookbacks       = MONTHLY_LOOKBACKS,
        min_train_months= 36,
        universe_name   = "NIFTY 250 MONTHLY (2021-2025)",
        freq_label      = "monthly",
        perf_fn         = perf_stats_monthly,
        freq_short      = "mo",
    )

    # --- Weekly run (lookbacks re-used numerically as weeks) ----------------
    # 1W, 4W, 12W, 24W, 52W
    WEEKLY_LOOKBACKS = [1, 4, 12, 24, 52]
    run_universe(
        prices_monthly  = weekly_prices,
        daily_prices    = daily_prices,
        lookbacks       = WEEKLY_LOOKBACKS,
        min_train_months= 156,          # 3 years in weeks
        universe_name   = "NIFTY 250 WEEKLY (2021-2025)",
        freq_label      = "weekly",
        perf_fn         = perf_stats_weekly,
        freq_short      = "wk",
    )


if __name__ == '__main__':
    main()
