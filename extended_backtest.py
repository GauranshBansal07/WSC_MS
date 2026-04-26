"""
Extended backtest — two runs, three variants each.

Run 1: 2013-onset → Oct 2025   | PiT Nifty 100 composition
Run 2: 2013-onset → Apr 2026   | Fixed Oct 2025 composition (ffilled forward)
         Apr 24 2026 treated as month-end.

Variants per run:
  Baseline — no 52wk filter
  H15      — exclude stocks >15% below trailing 252-day high
  H20      — exclude stocks >20% below trailing 252-day high

min_train=1 (not 0 — that causes Python -1 index wrap, i.e. lookahead).
With DATA_START=2008 the stacked dataset's first valid date is ~2013
when mom_60m becomes non-NaN; predictions effectively start from ~2013.
"""

import warnings, logging
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings('ignore')
logging.getLogger('hmmlearn').setLevel(logging.ERROR)
logging.getLogger('root').setLevel(logging.ERROR)

from config import (HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
                    LOOKBACK_WINDOWS)
from data_fetcher import (fetch_monthly_prices, fetch_daily_prices,
                          compute_forward_returns, load_historical_composition)
from features import compute_all_momentum
from engine import (build_stacked_dataset, run_expanding_window,
                    simulate_portfolio, performance_stats, print_stats)
from regime import get_regimes

DATA_START = '2008-01-01'


# ── helpers ────────────────────────────────────────────────────────────────

def apply_52wk_filter(res_df, daily_prices, threshold):
    """Zero out pred_prob for stocks sitting >threshold below 252-day high."""
    df = res_df.copy()
    high_252 = daily_prices.rolling(252).max()
    for date in df['date'].unique():
        ts = pd.Timestamp(date)
        avail_h = high_252.loc[high_252.index <= ts]
        avail_p = daily_prices.loc[daily_prices.index <= ts]
        if avail_h.empty or avail_p.empty:
            continue
        highs  = avail_h.iloc[-1]
        prices = avail_p.iloc[-1]
        dm = df['date'] == date
        for ticker in df.loc[dm, 'ticker'].values:
            if ticker not in highs or ticker not in prices:
                continue
            h, p = highs[ticker], prices[ticker]
            if pd.isna(h) or pd.isna(p) or h <= 0:
                continue
            if p < h * (1 - threshold):
                df.loc[dm & (df['ticker'] == ticker), 'pred_prob'] = 0.0
    return df


def run_variant(label, rdf, regimes, daily_prices, monthly_prices, orig_df):
    port, counts, _ = simulate_portfolio(
        rdf, regimes, daily_prices, monthly_prices=monthly_prices)
    stats = performance_stats(port, periods_per_year=12)
    print_stats(stats, label, counts, freq_label="mo")
    filtered = (
        (orig_df['pred_prob'] >= 0.55).groupby(orig_df['date']).sum()
        - (rdf['pred_prob'] >= 0.55).groupby(rdf['date']).sum()
    )
    if filtered.mean() > 0:
        print(f"  Avg stocks filtered/month by 52wk rule: {filtered.mean():.1f}")
    return stats


def print_comparison(title, s_a, s_h15, s_h20):
    banner = "=" * 72
    print(f"\n{banner}\n{title}\n{banner}")
    rows = [("CAGR (%)", "ann", ".2f"), ("Sharpe", "sharpe", ".3f"),
            ("Max DD (%)", "dd", ".2f"), ("Calmar", "calmar", ".3f"),
            ("Win Rate (%)", "win", ".2f")]
    print(f"{'Metric':<18}  {'Base':>10}  {'H15':>10}  {'H20':>10}"
          f"  {'H15-B':>8}  {'H20-B':>8}")
    print("-" * 72)
    for lbl, key, fmt in rows:
        a, h15, h20 = s_a[key], s_h15[key], s_h20[key]
        print(f"{lbl:<18}  {a:>10{fmt}}  {h15:>10{fmt}}  {h20:>10{fmt}}"
              f"  {h15-a:>+8{fmt}}  {h20-a:>+8{fmt}}")
    print(banner)


def run_pipeline(label, data_end, monthly_prices, mask, daily_prices, tickers):
    """Full pipeline: CatBoost + regime + 3 variants."""
    print(f"\n{'#'*80}\n# {label}\n{'#'*80}")

    fwd     = compute_forward_returns(monthly_prices)
    moms    = compute_all_momentum(monthly_prices, LOOKBACK_WINDOWS)
    stacked = build_stacked_dataset(monthly_prices, mask, fwd, moms, LOOKBACK_WINDOWS)
    res_df  = run_expanding_window(stacked, min_train_months=1)

    acc  = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    rebal_dates = sorted(res_df['date'].unique())
    print(f"  CatBoost accuracy: {acc:.3f}  precision: {prec:.3f}")
    print(f"  Rebal window: {pd.Timestamp(rebal_dates[0]).strftime('%Y-%m')} "
          f"→ {pd.Timestamp(rebal_dates[-1]).strftime('%Y-%m')}  ({len(rebal_dates)} months)")

    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=24)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, data_end, method='learned_hmm')

    res_h15 = apply_52wk_filter(res_df, daily_prices, 0.15)
    res_h20 = apply_52wk_filter(res_df, daily_prices, 0.20)

    print(f"\n{'─'*60}")
    s_a   = run_variant("Baseline", res_df,  regimes, daily_prices, monthly_prices, res_df)
    s_h15 = run_variant("H15",      res_h15, regimes, daily_prices, monthly_prices, res_df)
    s_h20 = run_variant("H20",      res_h20, regimes, daily_prices, monthly_prices, res_df)

    print_comparison(f"COMPARISON — {label}", s_a, s_h15, s_h20)
    return s_a, s_h15, s_h20


# ══════════════════════════════════════════════════════════════════════════════
# RUN 1: PiT Nifty 100 | 2013-onset → Oct 2025
# ══════════════════════════════════════════════════════════════════════════════
DATA_END_1 = '2025-10-31'
csv_paths  = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]

print(f"\nDownloading data for Run 1 (through {DATA_END_1})...")
mp1, mask1 = fetch_monthly_prices(
    csv_paths, DATA_START, DATA_END_1,
    cache_path='tmp_r1_monthly.csv', force_refresh=True)
tickers1   = mask1.columns.tolist()
dp1        = fetch_daily_prices(
    tickers1, DATA_START, DATA_END_1,
    cache_path='tmp_r1_daily.csv', force_refresh=True)

r1_base, r1_h15, r1_h20 = run_pipeline(
    "RUN 1 — PiT Nifty 100 | Oct 2025", DATA_END_1, mp1, mask1, dp1, tickers1)


# ══════════════════════════════════════════════════════════════════════════════
# RUN 2: Fixed Oct 2025 composition | 2013-onset → Apr 2026 (Apr 24 = month-end)
# ══════════════════════════════════════════════════════════════════════════════
DATA_END_2 = '2026-04-25'   # exclusive → includes Apr 24 data

print(f"\nDownloading data for Run 2 (through Apr 24 2026, fixed composition)...")
mp2_raw, mask2_raw = fetch_monthly_prices(
    csv_paths, DATA_START, DATA_END_2,
    cache_path='tmp_r2_monthly.csv', force_refresh=True)

# Forward-fill mask from Oct 2025 through Apr 2026
# (assumes Oct 2025 constituents for Nov 2025 – Apr 2026)
mask2 = mask2_raw.reindex(mp2_raw.index).ffill().fillna(False)
tickers2 = mask2.columns.tolist()

dp2 = fetch_daily_prices(
    tickers2, DATA_START, DATA_END_2,
    cache_path='tmp_r2_daily.csv', force_refresh=True)

r2_base, r2_h15, r2_h20 = run_pipeline(
    "RUN 2 — Fixed Oct-2025 Composition | Apr 2026", DATA_END_2, mp2_raw, mask2, dp2, tickers2)


# ══════════════════════════════════════════════════════════════════════════════
# SIDE-BY-SIDE SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
banner = "=" * 80
print(f"\n{banner}")
print("MEGA SUMMARY — Run 1 (Oct 2025 PiT)  vs  Run 2 (Apr 2026 fixed comp)")
print(banner)
rows = [("CAGR (%)", "ann", ".2f"), ("Sharpe", "sharpe", ".3f"),
        ("Max DD (%)", "dd", ".2f"), ("Calmar", "calmar", ".3f"),
        ("Win Rate (%)", "win", ".2f")]
print(f"{'Metric':<18}  {'R1-Base':>9}  {'R1-H15':>9}  {'R1-H20':>9}"
      f"  {'R2-Base':>9}  {'R2-H15':>9}  {'R2-H20':>9}")
print("-" * 80)
for lbl, key, fmt in rows:
    vals = [r1_base[key], r1_h15[key], r1_h20[key],
            r2_base[key], r2_h15[key], r2_h20[key]]
    row = f"{lbl:<18}" + "".join(f"  {v:>9{fmt}}" for v in vals)
    print(row)
print(banner)

# cleanup temp caches
import os
for f in ['tmp_r1_monthly.csv','tmp_r1_daily.csv','tmp_r2_monthly.csv','tmp_r2_daily.csv']:
    try: os.remove(f)
    except: pass
print("\nTemp caches removed.")
