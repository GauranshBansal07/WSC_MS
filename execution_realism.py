#!/usr/bin/env python3
"""
execution_realism.py — Unified execution-slippage analysis suite.

Variants:
  cc    Close-to-close month-end (baseline, unfillable)   ~29% CAGR
  oc    First-open entry, last-close exit (fillable)      ~18% CAGR
  oo    First-open to last-open (within-month)            ~13% CAGR
  four  4-variant comparison (matched vs original target training)

Usage:
  python3 execution_realism.py --variant cc --log
  python3 execution_realism.py --variant oc --log
  python3 execution_realism.py --variant four

--log writes a per-position CSV (trading_log_<variant>.csv) for manual
yfinance verification. Ignored for --variant four.
"""
import argparse, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config import (DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
                    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
import engine as eng
from regime import get_regimes


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_open_matrices():
    fo = pd.read_csv("monthly_first_open_adj.csv", index_col=0, parse_dates=True)
    lo = pd.read_csv("monthly_last_open_adj.csv",  index_col=0, parse_dates=True)
    fo.columns = [c if c.endswith('.NS') else f"{c}.NS" for c in fo.columns]
    lo.columns = [c if c.endswith('.NS') else f"{c}.NS" for c in lo.columns]
    return fo, lo


def _print_stats(name, stats):
    print(f"\n--- {name} ---")
    print(f"  CAGR   : {stats['ann']:.2f}%")
    print(f"  Max DD : {stats['dd']:.2f}%")
    print(f"  Sharpe : {stats['sharpe']:.3f}")
    print(f"  Calmar : {stats['calmar']:.3f}")


def _train_close_target(monthly_close, mask, min_train=60):
    """CatBoost walk-forward on original close-to-close monthly target."""
    fwd = compute_forward_returns(monthly_close)
    mom = compute_all_momentum(monthly_close, LOOKBACK_WINDOWS)
    stacked = eng.build_stacked_dataset(monthly_close, mask, fwd, mom, LOOKBACK_WINDOWS)
    return eng.run_expanding_window(stacked, min_train_months=min_train)


def simulate(res, regimes, entry_df, exit_df, daily_prices,
             entry_at_current=False, log_rows=None):
    """Generic monthly simulator.

    entry_at_current=True  -> entry price indexed at date T (signal day close).
    entry_at_current=False -> entry price indexed at next_date T+1 (next-month open).
    Exit price always indexed at next_date.
    If log_rows is a list, per-position dicts are appended for CSV output.
    """
    returns = []
    dates = sorted(res['date'].unique())
    prev_w = {}
    for i, date in enumerate(dates):
        group = res[res['date'] == date]
        if group.empty or i == len(dates) - 1:
            continue
        next_date = dates[i + 1]
        regime = regimes.get(date, 'Neutral')
        size = eng.REGIME_SIZE.get(regime, 4)
        stop = eng.REGIME_STOP.get(regime, -0.07)

        buys = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(size, 'pred_prob')
        if buys.empty:
            returns.append(0.0); prev_w = {}
            continue

        rel_w = eng._compute_weights(buys, daily_prices, date, method='prob_invvol')
        inv_f = min(1.0, len(buys) / size)
        curr_w = {t: w * inv_f for t, w in rel_w.items()}

        invested = 0.0
        for t, w in curr_w.items():
            if t not in entry_df.columns or t not in exit_df.columns or t not in daily_prices.columns:
                continue
            entry_idx = date if entry_at_current else next_date
            ps = entry_df.loc[entry_idx, t] if entry_idx in entry_df.index else np.nan
            pe = exit_df.loc[next_date, t] if next_date in exit_df.index else np.nan
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

            if log_rows is not None:
                log_rows.append({
                    'signal_date':  pd.Timestamp(date).strftime('%Y-%m-%d'),
                    'entry_date':   pd.Timestamp(entry_idx).strftime('%Y-%m-%d'),
                    'exit_date':    pd.Timestamp(next_date).strftime('%Y-%m-%d'),
                    'regime':       regime,
                    'ticker':       t,
                    'weight':       round(w, 4),
                    'entry_price':  round(float(ps), 2),
                    'exit_price':   round(float(pe), 2),
                    'raw_return':   round(pe / ps - 1.0, 4),
                    'stopped_out':  stopped,
                    'realized_ret': round(realized, 4),
                    'pred_prob':    round(float(buys[buys['ticker'] == t]['pred_prob'].iloc[0]), 4),
                })

        cash = 1.0 - sum(curr_w.values())
        rf = (1 + eng.RISK_FREE_ANNUAL) ** (1/12) - 1.0
        gross = invested + cash * rf
        all_t = set(prev_w) | set(curr_w)
        turnover = sum(abs(curr_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
        returns.append(gross - turnover * eng.TX_COST_SIDE)
        prev_w = curr_w
    return pd.Series(returns, index=pd.DatetimeIndex(dates[:len(returns)]))


def _write_log(rows, variant):
    out = f"trading_log_{variant}.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"  [log] {len(rows)} positions -> {out}")


# ---------------------------------------------------------------------------
# Variant runners
# ---------------------------------------------------------------------------

def run_cc(log=False):
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    res = _train_close_target(monthly_close, mask)
    dates = sorted(res['date'].unique())
    regimes = get_regimes(dates, DATA_START, DATA_END, method='learned_hmm')

    rows = [] if log else None
    port = simulate(res, regimes, monthly_close, monthly_close, daily_prices,
                    entry_at_current=True, log_rows=rows)
    _print_stats("CC — close-to-close month-end (baseline, unfillable)",
                 eng.performance_stats(port, periods_per_year=12))
    if log: _write_log(rows, 'cc')


def run_oc(log=False):
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    first_open, _ = _load_open_matrices()
    res = _train_close_target(monthly_close, mask)
    dates = sorted(res['date'].unique())
    regimes = get_regimes(dates, DATA_START, DATA_END, method='learned_hmm')

    rows = [] if log else None
    port = simulate(res, regimes, first_open, monthly_close, daily_prices,
                    entry_at_current=False, log_rows=rows)
    _print_stats("OC — first-open entry, last-close exit (fillable)",
                 eng.performance_stats(port, periods_per_year=12))
    if log: _write_log(rows, 'oc')


def run_oo(log=False):
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    first_open, last_open = _load_open_matrices()

    # Matched open-to-open training target
    mom = compute_all_momentum(last_open, LOOKBACK_WINDOWS)
    fwd = (last_open.shift(-1) / first_open.shift(-1)) - 1.0
    stacked = eng.build_stacked_dataset(last_open, mask, fwd, mom, LOOKBACK_WINDOWS)
    res = eng.run_expanding_window(stacked, min_train_months=48)
    dates = sorted(res['date'].unique())
    regimes = get_regimes(dates, DATA_START, DATA_END, method='learned_hmm')

    rows = [] if log else None
    port = simulate(res, regimes, first_open, last_open, daily_prices,
                    entry_at_current=False, log_rows=rows)
    _print_stats("OO — first-open to last-open (within-month)",
                 eng.performance_stats(port, periods_per_year=12))
    if log: _write_log(rows, 'oo')


def run_four():
    """4-way: (first-close vs first-open entry) x (matched vs original-c-c training)."""
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    first_open, _ = _load_open_matrices()
    last_close = monthly_close
    first_close = daily_prices.resample('MS').first()
    first_close.index = first_close.index + pd.offsets.MonthEnd(0)

    mom = compute_all_momentum(monthly_close, LOOKBACK_WINDOWS)
    fwd_fc_lc = (last_close.shift(-1) / first_close.shift(-1)) - 1.0
    fwd_fo_lc = (last_close.shift(-1) / first_open.shift(-1))  - 1.0
    fwd_cc    =  last_close.shift(-1) / last_close - 1.0

    print("[*] Training CatBoost models...")
    res_fc = eng.run_expanding_window(eng.build_stacked_dataset(monthly_close, mask, fwd_fc_lc, mom, LOOKBACK_WINDOWS), min_train_months=48)
    res_fo = eng.run_expanding_window(eng.build_stacked_dataset(monthly_close, mask, fwd_fo_lc, mom, LOOKBACK_WINDOWS), min_train_months=48)
    res_cc = eng.run_expanding_window(eng.build_stacked_dataset(monthly_close, mask, fwd_cc,    mom, LOOKBACK_WINDOWS), min_train_months=48)

    dates = sorted(res_cc['date'].unique())
    regimes = get_regimes(dates, DATA_START, DATA_END, method='learned_hmm')

    print("\n" + "=" * 60 + "\n  FOUR-VARIANT COMPARISON\n" + "=" * 60)
    for name, res, entry, exit_ in [
        ("1) Entry=first-CLOSE, Exit=last-CLOSE, Train=matched",  res_fc, first_close, last_close),
        ("2) Entry=first-CLOSE, Exit=last-CLOSE, Train=original", res_cc, first_close, last_close),
        ("3) Entry=first-OPEN,  Exit=last-CLOSE, Train=matched",  res_fo, first_open,  last_close),
        ("4) Entry=first-OPEN,  Exit=last-CLOSE, Train=original", res_cc, first_open,  last_close),
    ]:
        port = simulate(res, regimes, entry, exit_, daily_prices, entry_at_current=False)
        _print_stats(name, eng.performance_stats(port, periods_per_year=12))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Execution-realism analysis suite")
    p.add_argument('--variant', choices=['cc', 'oc', 'oo', 'four'], required=True,
                   help="cc=close-to-close baseline | oc=open-entry/close-exit | "
                        "oo=open-to-open | four=4-way comparison")
    p.add_argument('--log', action='store_true',
                   help="Write per-position trading log CSV (cc/oc/oo only)")
    args = p.parse_args()

    if   args.variant == 'cc':   run_cc(log=args.log)
    elif args.variant == 'oc':   run_oc(log=args.log)
    elif args.variant == 'oo':   run_oo(log=args.log)
    elif args.variant == 'four': run_four()


if __name__ == '__main__':
    main()
