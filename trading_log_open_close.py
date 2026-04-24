#!/usr/bin/env python3
"""
Trading log for the "month-open entry, month-close exit" variant (variant 4).
- Entry: first trading day's OPEN of month T+1
- Exit:  last trading day's CLOSE of month T+1
- Model trained on original close-to-close monthly target (no lookahead)

Outputs trading_log_open_close.csv and prints CAGR/DD/Sharpe/Calmar.
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
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    first_open = pd.read_csv("monthly_first_open_adj.csv", index_col=0, parse_dates=True)
    first_open.columns = [c if c.endswith('.NS') else f"{c}.NS" for c in first_open.columns]
    last_close = monthly_close  # last-day close = monthly_close (already 'ME'.last())

    fwd = compute_forward_returns(monthly_close)
    mom = compute_all_momentum(monthly_close, LOOKBACK_WINDOWS)
    stacked = eng.build_stacked_dataset(monthly_close, mask, fwd, mom, LOOKBACK_WINDOWS)
    res = eng.run_expanding_window(stacked, min_train_months=60)

    all_dates = sorted(res['date'].unique())
    regimes = get_regimes(all_dates, DATA_START, DATA_END, method='learned_hmm')

    rows = []
    returns = []
    prev_w = {}

    for i, date in enumerate(all_dates):
        group = res[res['date'] == date]
        if group.empty or i == len(all_dates) - 1:
            continue
        next_date = all_dates[i + 1]
        regime = regimes.get(date, 'Neutral')
        size   = eng.REGIME_SIZE.get(regime, 4)
        stop   = eng.REGIME_STOP.get(regime, -0.07)

        buys = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(size, 'pred_prob')
        if buys.empty:
            returns.append(0.0); prev_w = {}
            continue

        rel_w  = eng._compute_weights(buys, daily_prices, date, method='prob_invvol')
        inv_f  = min(1.0, len(buys) / size)
        curr_w = {t: w * inv_f for t, w in rel_w.items()}

        invested = 0.0
        for t, w in curr_w.items():
            if t not in first_open.columns or t not in last_close.columns or t not in daily_prices.columns:
                continue
            ps = first_open.loc[next_date, t] if next_date in first_open.index else np.nan
            pe = last_close.loc[next_date, t] if next_date in last_close.index else np.nan
            if pd.isna(ps) or pd.isna(pe) or ps <= 0:
                continue
            trace_start = date + pd.Timedelta(days=1)
            try:
                path = daily_prices.loc[trace_start:next_date, t] / ps - 1.0
                stopped = bool((path <= stop).any())
            except KeyError:
                stopped = False
            realized = stop if stopped else (pe / ps - 1.0)
            invested += realized * w

            rows.append({
                'signal_date':    pd.Timestamp(date).strftime('%Y-%m-%d'),
                'entry_date':     pd.Timestamp(next_date).strftime('%Y-%m-01'),  # 1st trading day of T+1
                'exit_date':      pd.Timestamp(next_date).strftime('%Y-%m-%d'),  # last trading day of T+1
                'regime':         regime,
                'ticker':         t,
                'weight':         round(w, 4),
                'entry_price_open':  round(float(ps), 2),
                'exit_price_close':  round(float(pe), 2),
                'raw_return':        round(pe / ps - 1.0, 4),
                'stopped_out':       stopped,
                'realized_ret':      round(realized, 4),
                'pred_prob':         round(float(buys[buys['ticker'] == t]['pred_prob'].iloc[0]), 4),
            })

        cash = 1.0 - sum(curr_w.values())
        rf = (1 + eng.RISK_FREE_ANNUAL) ** (1/12) - 1.0
        gross = invested + cash * rf
        all_t = set(prev_w) | set(curr_w)
        turnover = sum(abs(curr_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
        returns.append(gross - turnover * eng.TX_COST_SIDE)
        prev_w = curr_w

    log = pd.DataFrame(rows)
    out_path = "trading_log_open_close.csv"
    log.to_csv(out_path, index=False)

    port = pd.Series(returns, index=pd.DatetimeIndex([d for d in all_dates[:len(returns)]]))
    stats = eng.performance_stats(port, periods_per_year=12)

    print(f"[*] Wrote {len(log)} positions across {log['signal_date'].nunique()} rebalances")
    print(f"[*] Saved: {out_path}")
    print("\n=== First 10 rows ===")
    print(log.head(10).to_string(index=False))
    print("\n=== Last 10 rows ===")
    print(log.tail(10).to_string(index=False))
    print("\n" + "="*60)
    print("  MONTH-OPEN ENTRY / MONTH-CLOSE EXIT — PERFORMANCE")
    print("="*60)
    print(f"  CAGR   : {stats['ann']:.2f}%")
    print(f"  Sharpe : {stats['sharpe']:.3f}")
    print(f"  Max DD : {stats['dd']:.2f}%")
    print(f"  Calmar : {stats['calmar']:.3f}")
    print("="*60)
    print("\nSchema notes:")
    print("  signal_date       = last trading day of month T (when the model ranks stocks)")
    print("  entry_date        = 1st trading day of month T+1 (buy at OPEN)")
    print("  exit_date         = last trading day of month T+1 (sell at CLOSE)")
    print("  entry_price_open  = adj-open on 1st trading day of T+1")
    print("  exit_price_close  = adj-close on last trading day of T+1")


if __name__ == '__main__':
    main()
