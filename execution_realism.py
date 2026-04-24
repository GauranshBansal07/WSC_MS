#!/usr/bin/env python3
"""
execution_realism.py — CC baseline analysis & per-position trading log.

Runs the canonical close-to-close month-end execution using engine.simulate_portfolio
(with the monthly_prices entry-price fix) and optionally writes a per-position CSV
for manual verification against yfinance quotes.

Usage:
  python3 execution_realism.py            # print headline stats
  python3 execution_realism.py --log      # also write trading_log_cc.csv
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config import (DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
                    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
import engine as eng
from regime import get_regimes


def _print_stats(name, stats):
    print(f"\n--- {name} ---")
    print(f"  CAGR   : {stats['ann']:.2f}%")
    print(f"  Max DD : {stats['dd']:.2f}%")
    print(f"  Sharpe : {stats['sharpe']:.3f}")
    print(f"  Calmar : {stats['calmar']:.3f}")


def _emit_log_rows(res, regimes, monthly_close, daily_prices):
    """Per-position trading log — mirrors engine.simulate_portfolio dual-tranche logic."""
    rows = []
    dates = sorted(res['date'].unique())
    prev_w = {}

    for i, date in enumerate(dates):
        if i == len(dates) - 1:
            continue
        prev_date = dates[i - 1] if i > 0 else None
        next_date = dates[i + 1]
        group = res[res['date'] == date]
        regime = regimes.get(date, 'Neutral')
        size = eng.REGIME_SIZE.get(regime, 4)
        stop = eng.REGIME_STOP.get(regime, -0.07)

        buys = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(size, 'pred_prob')
        if buys.empty:
            prev_w = {}
            continue
        rel_w = eng._compute_weights(buys, daily_prices, date, method='prob_invvol')
        target_total = len(buys) / size
        curr_w = {t: w * target_total for t, w in rel_w.items()}

        for t, w in curr_w.items():
            if t not in daily_prices.columns:
                continue

            w_prev_t = prev_w.get(t, 0.0)
            w_held   = min(w_prev_t, w)
            w_new    = max(0.0, w - w_prev_t)

            def _mp(d):
                if d is not None and t in monthly_close.columns and d in monthly_close.index:
                    ep = monthly_close.loc[d, t]
                    if pd.notna(ep) and ep > 0:
                        return float(ep)
                return None

            curr_entry = _mp(date)
            prev_entry = _mp(prev_date)

            win = daily_prices.loc[date:next_date, t].dropna()
            if win.empty:
                continue
            fallback = float(win.iloc[0])
            if curr_entry is None:
                curr_entry = fallback
            if prev_entry is None:
                prev_entry = curr_entry
            exit_ = float(win.iloc[-1])

            def _tranche_ret(entry):
                if entry <= 0:
                    return None, False
                path = win / entry - 1.0
                if (path <= stop).any():
                    return stop, True
                return exit_ / entry - 1.0, False

            ret_held, stopped_held = _tranche_ret(prev_entry) if w_held > 0 else (None, False)
            ret_new,  stopped_new  = _tranche_ret(curr_entry) if w_new  > 0 else (None, False)

            # Composite log row (weighted average entry / return across both tranches)
            total_w = w_held + w_new
            if total_w <= 0:
                continue
            blended_entry = (
                (prev_entry * w_held + curr_entry * w_new) / total_w
                if (w_held + w_new) > 0 else curr_entry
            )
            realized_ret = (
                ((ret_held or 0.0) * w_held + (ret_new or 0.0) * w_new) / total_w
            )
            stopped = stopped_held or stopped_new

            rows.append({
                'signal_date':  pd.Timestamp(date).strftime('%Y-%m-%d'),
                'exit_date':    pd.Timestamp(next_date).strftime('%Y-%m-%d'),
                'regime':       regime,
                'ticker':       t,
                'w_held':       round(w_held, 4),
                'w_new':        round(w_new, 4),
                'entry_held':   round(prev_entry, 2),
                'entry_new':    round(curr_entry, 2),
                'exit_price':   round(exit_, 2),
                'stopped_out':  stopped,
                'realized_ret': round(realized_ret, 4),
                'pred_prob':    round(float(buys[buys['ticker'] == t]['pred_prob'].iloc[0]), 4),
            })

        prev_w = curr_w
    return rows


def run_cc(log=False):
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    fwd = compute_forward_returns(monthly_close)
    mom = compute_all_momentum(monthly_close, LOOKBACK_WINDOWS)
    stacked = eng.build_stacked_dataset(monthly_close, mask, fwd, mom, LOOKBACK_WINDOWS)
    res = eng.run_expanding_window(stacked, min_train_months=48)

    dates = sorted(res['date'].unique())
    padding = (pd.to_datetime(dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(dates, padding, DATA_END, method='learned_hmm')

    port, _, _ = eng.simulate_portfolio(res, regimes, daily_prices,
                                        sizing_scheme='directional',
                                        weighting='prob_invvol',
                                        monthly_prices=monthly_close)
    _print_stats("CC — close-to-close month-end (with entry-price fix)",
                 eng.performance_stats(port, periods_per_year=12))

    if log:
        rows = _emit_log_rows(res, regimes, monthly_close, daily_prices)
        out = "trading_log_cc.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"  [log] {len(rows)} positions -> {out}")


def main():
    p = argparse.ArgumentParser(description="CC execution baseline + optional trading log")
    p.add_argument('--log', action='store_true',
                   help="Write per-position trading log to trading_log_cc.csv")
    args = p.parse_args()
    run_cc(log=args.log)


if __name__ == '__main__':
    main()
