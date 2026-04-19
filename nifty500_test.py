"""
CatBoost Long/Short strategy applied to the Nifty 500 universe.
Using the consolidated 5-minute ticks flattened to Daily and Monthly CSVs.
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

from config import (
    DATA_START, DATA_END, TRANSACTION_COST_BPS
)
from data_fetcher import compute_forward_returns
from features import compute_all_momentum
from catboost_test import (
    build_stacked_dataset, run_expanding_window, get_macro_regimes,
    simulate_portfolio, performance_stats
)


def evaluate_nifty500():
    universe_name = "NIFTY 500 (2021-2025)"
    banner = "=" * 80
    print(f"\n{banner}\nCATBOOST STRATEGY — {universe_name}\n{banner}")

    print("\n[1] Loading local Nifty 500 caches...")
    if not os.path.exists("monthly_cache_nifty500.csv") or not os.path.exists("daily_cache_nifty500.csv"):
        print("Missing cache files! Please run prepare_nifty500.py first.")
        return
        
    prices = pd.read_csv("monthly_cache_nifty500.csv", index_col=0, parse_dates=True)
    daily_prices = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
    
    # In lieu of a PiT index, we assume activity if the ticker has a price.
    mask = prices.notna()

    # Reduce lookbacks because our history only starts in 2015. 
    # If we use 60M, we burn 5 years just building the feature.
    NIFTY_500_LOOKBACKS = [1, 3, 6, 12]
    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, NIFTY_500_LOOKBACKS)

    print("\n[2] Building feature panel...")
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, NIFTY_500_LOOKBACKS)
    print(f"  Observations: {len(stacked)}  |  Avg monthly universe: "
          f"{stacked.groupby(level=0).size().mean():.1f}")

    print("\n[3] Walk-forward CatBoost classification (36 month warm-up)...")
    res_df = run_expanding_window(stacked, min_train_months=36)
    if res_df is None:
        return

    # User explicitly requested 2021 to 2025 evaluation
    eval_start = pd.Timestamp('2021-01-01')
    res_df = res_df[res_df['date'] >= eval_start]

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy (2021+): {acc:.3f}  |  precision: {prec:.3f}")
    
    print("\n[4] Classifying macro regimes (Based on Nifty 50 SMAs)...")
    rebal_dates = sorted(res_df['date'].unique())
    if len(rebal_dates) == 0:
        print("No valid predictions in the 2021-2025 range.")
        return
        
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_macro_regimes(rebal_dates, padding_start, DATA_END)

    print(f"\n[5] Simulating portfolio with strict DAILY percentage stop approximations...")
    
    port_ret_long, counts_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = performance_stats(port_ret_long)
    
    port_ret_ls, counts_ls, short_counts_ls = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = performance_stats(port_ret_ls)

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG ONLY\n{banner}")
    print(f"  Avg long holdings / mo: {np.mean(counts_long):.1f}")
    print(f"  Total Return (%):       {stats_long['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_long['ann']:.2f}")
    print(f"  Volatility (%):         {stats_long['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_long['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_long['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_long['calmar']:.3f}")

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG + SHORT\n{banner}")
    print(f"  Avg long holdings / mo: {np.mean(counts_ls):.1f}")
    print(f"  Avg short holdings / mo:{np.mean(short_counts_ls):.1f}")
    print(f"  Total Return (%):       {stats_ls['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_ls['ann']:.2f}")
    print(f"  Volatility (%):         {stats_ls['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_ls['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_ls['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_ls['calmar']:.3f}")

    # Output predictions to CSV
    os.makedirs('output', exist_ok=True)
    res_df.to_csv('output/catboost_preds_nifty_500.csv', index=False)
    print(f"\n  Predictions saved -> output/catboost_preds_nifty_500.csv")

if __name__ == '__main__':
    evaluate_nifty500()
