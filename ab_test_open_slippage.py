#!/usr/bin/env python3
"""
ab_test_open_slippage.py
========================
Trains and simulates the CatBoost momentum strategy exclusively using Adjusted Open Execution.

1. Features derived from Month-End Open prices.
2. Targets are strictly: (Month T Last Day Open) / (Month T First Day Open) - 1.
3. Sidesteps the toxic EoM Close rollover perfectly, staying protected overnight in Cash.
"""

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from config import DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS
from data_fetcher import load_historical_composition, fetch_daily_prices
from features import compute_all_momentum
import engine as eng
from regime import get_regimes

def simulate_open_portfolio(res_df, regimes, first_open, last_open, daily_prices, weighting='prob_invvol'):
    returns = []
    dates = sorted(res_df['date'].unique())
    prev_weights = {}

    for i, date in enumerate(dates):
        group = res_df[res_df['date'] == date]
        if group.empty or i == len(dates) - 1:
            continue
            
        # The holding month is T+1
        next_date = dates[i+1]
        current_regime = regimes.get(date, 'Neutral')
        max_size = eng.REGIME_SIZE.get(current_regime, 4)
        stop = eng.REGIME_STOP.get(current_regime, -0.05)
        
        buys = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(max_size, 'pred_prob')
        if buys.empty:
            returns.append(0.0)
            prev_weights = {}
            continue
            
        curr_weights = eng._compute_weights(buys, daily_prices, date, method=weighting)
        invested_fraction = min(1.0, len(buys) / max_size)
        curr_weights = {t: w * invested_fraction for t, w in curr_weights.items()}
        
        invested_return = 0.0
        
        for ticker, w in curr_weights.items():
            if ticker in first_open.columns and ticker in last_open.columns and ticker in daily_prices.columns:
                p_start = first_open.loc[next_date, ticker]
                p_end = last_open.loc[next_date, ticker]
                
                if pd.isna(p_start) or pd.isna(p_end) or p_start <= 0:
                    continue
                    
                # Trace from the start of Month T+1 onward
                trace_start = date + pd.Timedelta(days=1)
                
                try:
                    path = daily_prices.loc[trace_start:next_date, ticker] / p_start - 1.0
                    if (path <= stop).any():
                        invested_return += stop * w
                    else:
                        invested_return += (p_end / p_start - 1.0) * w
                except KeyError:
                    # In case of missing tracing data just take the boundary points
                    invested_return += (p_end / p_start - 1.0) * w
                        
        cash_weight = 1.0 - sum(curr_weights.values())
        rf_period = (1 + eng.RISK_FREE_ANNUAL) ** (1.0 / 12) - 1.0
        gross_return = invested_return + (cash_weight * rf_period)
        
        # Turnover & Costs
        all_tickers = set(prev_weights.keys()).union(curr_weights.keys())
        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        tx_cost = turnover * eng.TX_COST_SIDE
        
        returns.append(gross_return - tx_cost)
        prev_weights = curr_weights
        
    return pd.Series(returns, index=pd.DatetimeIndex(dates[:-1]))

def main():
    print(f"\n=================================================================")
    print(f"  OPEN-TO-OPEN EXECUTION ENGINE (NIFTY 100)")
    print(f"=================================================================\n")

    print("[*] Loading Adjusted Open Boundaries...")
    # These CSVs contain tickers with '.NS' appended
    first_open = pd.read_csv("monthly_first_open_adj.csv", index_col=0, parse_dates=True)
    last_open = pd.read_csv("monthly_last_open_adj.csv", index_col=0, parse_dates=True)
    
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    mask_c, tickers = load_historical_composition(csv_paths)
    
    # ensure columns match (mask_c does not have .NS, first_open does)
    yf_cols = [t if t.endswith('.NS') else f"{t}.NS" for t in mask_c.columns]
    mask_c.columns = yf_cols
    
    mask = mask_c.reindex(index=last_open.index).ffill().fillna(False)
    
    daily_prices = fetch_daily_prices(tickers, DATA_START, DATA_END)
    # also ensure daily_prices has .NS columns
    dp_cols = [t if t.endswith('.NS') else f"{t}.NS" for t in daily_prices.columns]
    daily_prices.columns = dp_cols
    
    print("[*] Extracting Structural Adjusted Variables...")
    momentum_dict = compute_all_momentum(last_open, LOOKBACK_WINDOWS)
    
    # Target = Next month's (Last Day Open / First Day Open)
    fwd_returns_for_model = (last_open.shift(-1) / first_open.shift(-1)) - 1.0
    
    print("[*] Building Stacked Dataset...")
    stacked = eng.build_stacked_dataset(last_open, mask, fwd_returns_for_model, momentum_dict, LOOKBACK_WINDOWS)
    
    print("[*] Training CatBoost exclusively on Open executions...")
    res_open = eng.run_expanding_window(stacked, min_train_months=48)
    
    print("\n[*] Simulating Adjusted Open-to-Open Portfolio Path...")
    all_dates = sorted(res_open['date'].unique())
    regimes = get_regimes(all_dates, DATA_START, DATA_END, method='learned_hmm')
    
    port_rets_open = simulate_open_portfolio(
        res_open, regimes, first_open, last_open, daily_prices, weighting='prob_invvol'
    )
    
    stats_open = eng.performance_stats(port_rets_open, periods_per_year=12)
    
    print(f"\n=================================================================")
    print(f"  FINAL RESULTS: 09:15 AM SAFEHOUSE STRATEGY")
    print(f"=================================================================")
    print(f"  CAGR   : {stats_open['ann']:.2f}%")
    print(f"  Sharpe : {stats_open['sharpe']:.3f}")
    print(f"  Max DD : {stats_open['dd']:.2f}%")
    print(f"  Calmar : {stats_open['calmar']:.3f}")
    print(f"  Win RT : {stats_open['win']:.1f}%")
    print(f"=================================================================\n")

if __name__ == "__main__":
    main()
