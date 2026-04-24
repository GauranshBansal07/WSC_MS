#!/usr/bin/env python3
"""
Four variants comparing entry-at-first-day (close or open) vs training target.
Exit always at last-day close of the target month.
"""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')

from config import (DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
                    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS)
from data_fetcher import load_historical_composition, fetch_daily_prices, fetch_monthly_prices
from features import compute_all_momentum
import engine as eng
from regime import get_regimes


def simulate(res_df, regimes, p_start_df, p_end_df, daily_prices, weighting='prob_invvol',
             entry_at_current=False):
    """Generic simulator: enter at p_start_df[entry_date], exit at p_end_df[next_date].
    entry_at_current=False: entry indexed at next_date (first-day of T+1 variants).
    entry_at_current=True:  entry indexed at current date (last-day-T variants)."""
    returns = []
    dates = sorted(res_df['date'].unique())
    prev_w = {}
    for i, date in enumerate(dates):
        group = res_df[res_df['date'] == date]
        if group.empty or i == len(dates) - 1:
            continue
        next_date = dates[i + 1]
        regime = regimes.get(date, 'Neutral')
        max_size = eng.REGIME_SIZE.get(regime, 4)
        stop = eng.REGIME_STOP.get(regime, -0.05)

        buys = group[group['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(max_size, 'pred_prob')
        if buys.empty:
            returns.append(0.0)
            prev_w = {}
            continue

        w_rel = eng._compute_weights(buys, daily_prices, date, method=weighting)
        inv_frac = min(1.0, len(buys) / max_size)
        curr_w = {t: w * inv_frac for t, w in w_rel.items()}

        invested = 0.0
        for t, w in curr_w.items():
            if t in p_start_df.columns and t in p_end_df.columns and t in daily_prices.columns:
                entry_idx = date if entry_at_current else next_date
                ps = p_start_df.loc[entry_idx, t] if entry_idx in p_start_df.index else np.nan
                pe = p_end_df.loc[next_date, t]   if next_date  in p_end_df.index   else np.nan
                if pd.isna(ps) or pd.isna(pe) or ps <= 0:
                    continue
                trace_start = date + pd.Timedelta(days=1)
                try:
                    path = daily_prices.loc[trace_start:next_date, t] / ps - 1.0
                    if (path <= stop).any():
                        invested += stop * w
                    else:
                        invested += (pe / ps - 1.0) * w
                except KeyError:
                    invested += (pe / ps - 1.0) * w

        cash = 1.0 - sum(curr_w.values())
        rf = (1 + eng.RISK_FREE_ANNUAL) ** (1 / 12) - 1.0
        gross = invested + cash * rf
        all_t = set(prev_w.keys()) | set(curr_w.keys())
        turnover = sum(abs(curr_w.get(t, 0) - prev_w.get(t, 0)) for t in all_t)
        returns.append(gross - turnover * eng.TX_COST_SIDE)
        prev_w = curr_w
    return pd.Series(returns, index=pd.DatetimeIndex(dates[:-1]))


def run_variant(name, res_df, regimes, p_start, p_end, daily_prices, entry_at_current=False):
    port = simulate(res_df, regimes, p_start, p_end, daily_prices, entry_at_current=entry_at_current)
    s = eng.performance_stats(port, periods_per_year=12)
    print(f"\n--- {name} ---")
    print(f"  CAGR   : {s['ann']:.2f}%")
    print(f"  Max DD : {s['dd']:.2f}%")
    print(f"  Sharpe : {s['sharpe']:.3f}")
    print(f"  Calmar : {s['calmar']:.3f}")
    return s


def main():
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]

    # --- Price matrices ---------------------------------------------------
    first_open = pd.read_csv("monthly_first_open_adj.csv", index_col=0, parse_dates=True)
    last_open  = pd.read_csv("monthly_last_open_adj.csv",  index_col=0, parse_dates=True)

    monthly_close, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

    # First-day close of each month from daily closes: resample 'MS' first, then align to month-end
    first_close = daily_prices.resample('MS').first()
    first_close.index = first_close.index + pd.offsets.MonthEnd(0)

    # Penultimate-day close of each month (for causally-valid features when executing at last-day OPEN).
    # shift(1) moves each value forward one trading day, so resample('ME').last() picks up
    # the value originally at the 2nd-to-last trading day of the month.
    penult_close = daily_prices.shift(1).resample('ME').last()

    # last-day close is monthly_close (already month-end indexed)
    last_close = monthly_close

    # Align column namespaces: ensure all have .NS suffix consistent with daily_prices
    def norm(df):
        df = df.copy()
        df.columns = [c if c.endswith('.NS') else f"{c}.NS" for c in df.columns]
        return df
    first_open, last_open = norm(first_open), norm(last_open)
    # daily_prices, monthly_close, first_close already .NS (from fetch_*)

    # --- Build training datasets ------------------------------------------
    mom_close  = compute_all_momentum(monthly_close, LOOKBACK_WINDOWS)
    # Lookahead-free momentum for variants that execute at last-day OPEN
    mom_penult = compute_all_momentum(penult_close, LOOKBACK_WINDOWS)

    # Target A: matched entry=first_close, exit=last_close of month T+1
    fwd_fc_lc = (last_close.shift(-1) / first_close.shift(-1)) - 1.0
    # Target B: matched entry=first_open, exit=last_close of month T+1
    fwd_fo_lc = (last_close.shift(-1) / first_open.shift(-1))  - 1.0
    # Target C: original close-to-close month-end
    fwd_cc    = last_close.shift(-1) / last_close - 1.0
    print("[*] Training CatBoost models...")
    stacked_fc = eng.build_stacked_dataset(monthly_close, mask, fwd_fc_lc, mom_close, LOOKBACK_WINDOWS)
    stacked_fo = eng.build_stacked_dataset(monthly_close, mask, fwd_fo_lc, mom_close, LOOKBACK_WINDOWS)
    stacked_cc = eng.build_stacked_dataset(monthly_close, mask, fwd_cc,    mom_close, LOOKBACK_WINDOWS)

    res_fc = eng.run_expanding_window(stacked_fc, min_train_months=48)
    res_fo = eng.run_expanding_window(stacked_fo, min_train_months=48)
    res_cc = eng.run_expanding_window(stacked_cc, min_train_months=48)

    all_dates = sorted(res_cc['date'].unique())
    regimes = get_regimes(all_dates, DATA_START, DATA_END, method='learned_hmm')

    print("\n" + "="*60)
    print("  FOUR-VARIANT COMPARISON")
    print("="*60)
    run_variant("1) Entry=first-CLOSE, Exit=last-CLOSE, Train=matched",
                res_fc, regimes, first_close, last_close, daily_prices)
    run_variant("2) Entry=first-CLOSE, Exit=last-CLOSE, Train=original(m-end c-c)",
                res_cc, regimes, first_close, last_close, daily_prices)
    run_variant("3) Entry=first-OPEN,  Exit=last-CLOSE, Train=matched",
                res_fo, regimes, first_open, last_close, daily_prices)
    run_variant("4) Entry=first-OPEN,  Exit=last-CLOSE, Train=original(m-end c-c)",
                res_cc, regimes, first_open, last_close, daily_prices)


if __name__ == '__main__':
    main()
