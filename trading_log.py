#!/usr/bin/env python3
"""
Trading log for the original close-to-close monthly strategy (the 29% CAGR baseline).
Outputs a CSV with one row per position: rebalance date, ticker, entry price,
exit price, weight, realized return, stopped-out flag. Easy to spot-check against
yfinance.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config import (DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
                    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
import engine as eng
from regime import get_regimes


def main():
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    fwd = compute_forward_returns(monthly_prices)
    mom = compute_all_momentum(monthly_prices, LOOKBACK_WINDOWS)
    stacked = eng.build_stacked_dataset(monthly_prices, mask, fwd, mom, LOOKBACK_WINDOWS)
    res = eng.run_expanding_window(stacked, min_train_months=60)

    all_dates = sorted(res['date'].unique())
    regimes = get_regimes(all_dates, DATA_START, DATA_END, method='learned_hmm')

    rows = []
    for i, date in enumerate(all_dates):
        group = res[res['date'] == date]
        regime = regimes.get(date, 'Neutral')
        size   = eng.REGIME_SIZE.get(regime, 4)
        stop   = eng.REGIME_STOP.get(regime, -0.07)
        buys   = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(size, 'pred_prob')
        if buys.empty or i == len(all_dates) - 1:
            continue
        next_date = all_dates[i + 1]
        rel_w = eng._compute_weights(buys, daily_prices, date, method='prob_invvol')
        target_total = len(buys) / size
        curr_w = {t: w * target_total for t, w in rel_w.items()}

        for t, w in curr_w.items():
            if t not in daily_prices.columns:
                continue
            win = daily_prices.loc[date:next_date, t].dropna()
            if win.empty:
                continue
            entry_price = float(win.iloc[0])
            exit_price  = float(win.iloc[-1])
            path = win / entry_price - 1.0
            stopped = bool((path <= stop).any())
            realized = stop if stopped else (exit_price / entry_price - 1.0)
            rows.append({
                'rebalance_date': pd.Timestamp(date).strftime('%Y-%m-%d'),
                'exit_date':      pd.Timestamp(next_date).strftime('%Y-%m-%d'),
                'regime':         regime,
                'ticker':         t,
                'weight':         round(w, 4),
                'entry_price':    round(entry_price, 2),
                'exit_price':     round(exit_price, 2),
                'raw_return':     round(exit_price / entry_price - 1.0, 4),
                'stopped_out':    stopped,
                'realized_ret':   round(realized, 4),
                'pred_prob':      round(float(buys[buys['ticker'] == t]['pred_prob'].iloc[0]), 4),
            })

    log = pd.DataFrame(rows)
    out_path = "trading_log_original.csv"
    log.to_csv(out_path, index=False)
    print(f"[*] Wrote {len(log)} positions across {log['rebalance_date'].nunique()} rebalances")
    print(f"[*] Saved: {out_path}")
    print("\n=== First 15 rows ===")
    print(log.head(15).to_string(index=False))
    print("\n=== Last 10 rows ===")
    print(log.tail(10).to_string(index=False))
    print(f"\nColumns: rebalance_date, exit_date — both use last-trading-day closes.")
    print(f"entry_price = adj close on rebalance_date; exit_price = adj close on exit_date (or stop-out day).")


if __name__ == '__main__':
    main()
