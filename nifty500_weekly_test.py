"""
CatBoost Long/Short strategy applied to the Nifty 500 universe.
Using WEEKLY frequency resampling of the 5-minute ticks.
Evaluation explicitly focused on 2021-2025.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from config import DATA_END, RISK_FREE_ANNUAL
from data_fetcher import compute_forward_returns
from features import compute_all_momentum
from catboost_test import (
    build_stacked_dataset, run_expanding_window, get_macro_regimes,
    simulate_portfolio
)

def performance_stats_weekly(port_ret):
    rf_weekly = (1 + RISK_FREE_ANNUAL)**(1.0/52.0) - 1.0
    
    total = (np.prod(1 + port_ret) - 1) * 100
    ann = (np.prod(1 + port_ret) ** (52 / len(port_ret)) - 1) * 100 if len(port_ret) > 0 else 0
    vol = port_ret.std() * np.sqrt(52) * 100
    
    mean_excess = port_ret.mean() - rf_weekly
    sharpe = (mean_excess / port_ret.std() * np.sqrt(52)) if port_ret.std() > 0 else 0.0
    
    cum = (1 + port_ret).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = ann / abs(dd) if dd != 0 else 0.0
    win = (port_ret > 0).mean() * 100
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, dd=dd, calmar=calmar, win=win)

def evaluate_nifty500_weekly():
    universe_name = "NIFTY 500 (WEEKLY - 2021-2025)"
    banner = "=" * 80
    print(f"\n{banner}\nCATBOOST STRATEGY — {universe_name}\n{banner}")

    print("\n[1] Loading local Nifty 500 caches and Resampling to WEEKLY...")
    if not os.path.exists("daily_cache_nifty500.csv"):
        print("Missing daily cache! Please run prepare_nifty500.py first.")
        return
        
    daily_prices = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
    
    # Resample to weekly
    weekly_prices = daily_prices.resample('W-FRI').last().dropna(how='all')
    
    mask = weekly_prices.notna()

    # Lookbacks: 1W, 4W, 12W, 24W, 52W
    WEEKLY_LOOKBACKS = [1, 4, 12, 24, 52]
    fwd_returns = compute_forward_returns(weekly_prices)
    momentum_dict = compute_all_momentum(weekly_prices, WEEKLY_LOOKBACKS)

    print("\n[2] Building feature panel...")
    stacked = build_stacked_dataset(weekly_prices, mask, fwd_returns, momentum_dict, WEEKLY_LOOKBACKS)
    print(f"  Observations: {len(stacked)}  |  Avg weekly universe: "
          f"{stacked.groupby(level=0).size().mean():.1f}")

    print("\n[3] Walk-forward CatBoost classification (Requires 3-Year / 156-Week warm-up)...")
    res_df = run_expanding_window(stacked, min_train_months=156)
    if res_df is None:
        return

    # Evaluate strictly 2021 to 2025
    eval_start = pd.Timestamp('2021-01-01')
    res_df = res_df[res_df['date'] >= eval_start]

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy (1-Week Fwd): {acc:.3f}  |  precision: {prec:.3f}")
    
    print("\n[4] Classifying macro regimes (Based on Nifty 50 SMAs)...")
    rebal_dates = sorted(res_df['date'].unique())
    if len(rebal_dates) == 0:
        print("No valid predictions in the 2021-2025 range.")
        return
        
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_macro_regimes(rebal_dates, padding_start, DATA_END)

    print(f"\n[5] Simulating portfolio with strict INTRA-WEEK daily path validation...")
    
    port_ret_long, counts_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = performance_stats_weekly(port_ret_long)
    
    port_ret_ls, counts_ls, short_counts_ls = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = performance_stats_weekly(port_ret_ls)

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG ONLY\n{banner}")
    print(f"  Avg long holdings / wk: {np.mean(counts_long):.1f}")
    print(f"  Total Return (%):       {stats_long['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_long['ann']:.2f}")
    print(f"  Volatility (%):         {stats_long['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_long['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_long['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_long['calmar']:.3f}")

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG + SHORT\n{banner}")
    print(f"  Avg long holdings / wk: {np.mean(counts_ls):.1f}")
    print(f"  Avg short holdings / wk:{np.mean(short_counts_ls):.1f}")
    print(f"  Total Return (%):       {stats_ls['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_ls['ann']:.2f}")
    print(f"  Volatility (%):         {stats_ls['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_ls['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_ls['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_ls['calmar']:.3f}")

if __name__ == '__main__':
    evaluate_nifty500_weekly()
