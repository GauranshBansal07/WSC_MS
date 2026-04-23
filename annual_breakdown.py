#!/usr/bin/env python3
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings('ignore')

from config import DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
import engine as eng
from regime import get_regimes

def main():
    print("Loading data...")
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    fwd_returns = compute_forward_returns(monthly_prices)
    momentum_dict = compute_all_momentum(monthly_prices, LOOKBACK_WINDOWS)

    print("Building stacked dataset and running expanding window...")
    stacked = eng.build_stacked_dataset(monthly_prices, mask, fwd_returns, momentum_dict, LOOKBACK_WINDOWS)
    res_df = eng.run_expanding_window(stacked, min_train_months=60)
    
    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method='learned_hmm')

    print("Simulating portfolio...")
    port, _, _ = eng.simulate_portfolio(res_df, regimes, daily_prices, sizing_scheme='directional', weighting='prob_invvol')

    print("\n--- ANNUAL BREAKDOWN ---")
    years = port.index.year.unique()
    rows = []
    for y in years:
        port_y = port[port.index.year == y]
        if len(port_y) < 10: continue # ignore tiny stubs
        # Calmar = Ann Return / MaxDD
        # Calculate year's metric
        cum = (1 + port_y).cumprod()
        ret = cum.iloc[-1] - 1
        dd = (cum - cum.cummax()) / cum.cummax()
        max_dd = dd.min()
        
        # Annualized values for this year
        days = (port_y.index[-1] - port_y.index[0]).days
        if days < 30: continue
        ann = (1 + ret)**(365.25 / days) - 1
        calmar = ann / abs(max_dd) if abs(max_dd) > 0 else np.nan
        
        # Volatility
        vol = port_y.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else np.nan

        rows.append({
            'Year': y,
            'Return': ret,
            'Ann Return': ann,
            'MaxDD': max_dd,
            'Calmar': calmar,
            'Sharpe': sharpe
        })

    df = pd.DataFrame(rows)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("------------------------")

if __name__ == '__main__':
    main()
