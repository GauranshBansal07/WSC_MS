"""
open_vs_close.py — stress test: adj-open vs adj-close monthly strategy

For each month we want the LAST trading day's open price, adjusted for
splits/dividends using the same factor yfinance applies to close prices:

    adj_open = raw_open  ×  (adj_close / raw_close)

Data source: daily interval (auto_adjust=False) so we get both raw and
adjusted close on the same bar.  Resample to month-end with .last() so
we always pick the last AVAILABLE trading day of that calendar month —
if month-end falls on a weekend or holiday the previous Friday is used
automatically, and we never bleed into the following month.

The full CatBoost + HMM pipeline is re-run from scratch on open prices.
Daily adjusted closes are still used for intra-month stop-loss monitoring
(you watch daily closes to decide if a stop is hit — opening prices are
not relevant there).
"""

import os
import logging
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings('ignore')
logging.getLogger('hmmlearn').setLevel(logging.ERROR)
logging.getLogger('root').setLevel(logging.ERROR)

from config import (
    DATA_START, DATA_END,
    HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
    LOOKBACK_WINDOWS,
)
from data_fetcher import (
    load_historical_composition, fetch_monthly_prices, fetch_daily_prices,
)
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats, print_stats,
)
from regime import get_regimes

ADJ_OPEN_CACHE = os.path.join(os.path.dirname(__file__), 'open_adj_cache.csv')


# ---------------------------------------------------------------------------
# Adjusted-open fetcher
# ---------------------------------------------------------------------------

def fetch_adj_open_monthly(tickers, start, end,
                           cache_path=ADJ_OPEN_CACHE, force_refresh=False):
    """
    Download daily OHLCV (unadjusted) + adj_close, compute adjusted open,
    then resample to the last trading day of each calendar month.

    Barricades:
    • resample('ME').last()  → always the last AVAILABLE date within the
      calendar month; never drifts into the next month.
    • adj_factor outlier clip → ratios outside [0.5, 2.0] on a single day
      are almost certainly a data error; replace with ffill.
    • Columns with < 60 valid months are dropped (same gate as close prices).
    """
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached adj open from {cache_path}")
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = df[df.index <= pd.Timestamp(end) + pd.offsets.MonthEnd(0)]
        print(f"  shape: {df.shape}")
        return df

    print(f"Downloading daily unadjusted OHLCV for {len(tickers)} tickers "
          f"({start} → {end})...")
    raw = yf.download(
        tickers, start=start, end=end, interval='1d',
        auto_adjust=False, progress=True, threads=True,
    )

    if isinstance(raw.columns, pd.MultiIndex):
        opens      = raw['Open']
        closes     = raw['Close']
        adj_closes = raw['Adj Close']
    else:
        # Single-ticker edge case
        opens      = raw[['Open']];      opens.columns      = [tickers[0]]
        closes     = raw[['Close']];     closes.columns     = [tickers[0]]
        adj_closes = raw[['Adj Close']]; adj_closes.columns = [tickers[0]]

    # Adjustment factor: same multiplier yfinance applies to Close
    adj_factor = (adj_closes / closes).replace([np.inf, -np.inf], np.nan)

    # Clip extreme single-day jumps — anything outside [0.5, 2.0] is a bad bar
    adj_factor = adj_factor.clip(lower=0.5, upper=2.0)
    adj_factor = adj_factor.ffill().bfill()

    adj_open = (opens * adj_factor).sort_index()

    # Last trading day of each calendar month
    # resample('ME') groups strictly within the calendar month boundary,
    # so a month-end holiday just means we take the Friday before it.
    monthly_open = adj_open.resample('ME').last()

    # Sanity check: confirm no bleed (every index date is a true month-end)
    bleed = [(d, d + pd.offsets.MonthEnd(0)) for d in monthly_open.index
             if d != d + pd.offsets.MonthEnd(0)]
    if bleed:
        print(f"  WARNING: {len(bleed)} index dates are not month-end "
              f"calendar dates — check yfinance data.")

    min_obs = 60
    valid_cols = monthly_open.columns[monthly_open.notna().sum() >= min_obs]
    monthly_open = monthly_open[valid_cols]

    print(f"  Adj-open matrix: {monthly_open.shape[0]} months "
          f"× {monthly_open.shape[1]} tickers")
    print(f"  Missing: {monthly_open.isna().sum().sum() / monthly_open.size * 100:.1f}%")

    monthly_open.to_csv(cache_path)
    print(f"  Cached → {cache_path}")
    return monthly_open


# ---------------------------------------------------------------------------
# Single run helper (mirrors main.py's run_pit_universe)
# ---------------------------------------------------------------------------

def _run(label, prices, mask, daily_prices, exit_prices=None):
    """Run the full pipeline on a given monthly price matrix."""
    banner = "=" * 80
    print(f"\n{banner}\nRUNNING — {label}\n{banner}")

    from data_fetcher import compute_forward_returns
    fwd_returns   = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, LOOKBACK_WINDOWS)
    stacked       = build_stacked_dataset(
        prices, mask, fwd_returns, momentum_dict, LOOKBACK_WINDOWS,
    )
    res_df = run_expanding_window(stacked, min_train_months=60)
    if res_df is None:
        print("Not enough data — skipping.")
        return None, None

    acc  = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    rebal_dates   = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=24)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END)
    port, counts, _ = simulate_portfolio(
        res_df, regimes, daily_prices, monthly_prices=prices, exit_prices=exit_prices,
    )
    stats = performance_stats(port, periods_per_year=12)
    print_stats(stats, f"NIFTY 100 — {label}", counts, freq_label="mo")
    return stats, counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    csv_paths = [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    mask, tickers = load_historical_composition(csv_paths)

    # Close prices (existing cache or fresh download)
    monthly_close, _ = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)

    # Daily closes for stop-loss monitoring (shared by both runs)
    daily_prices = fetch_daily_prices(tickers, DATA_START, DATA_END)

    # Adjusted open prices (last trading day of each month)
    monthly_open = fetch_adj_open_monthly(tickers, DATA_START, DATA_END)

    stats_close, _ = _run("ADJ CLOSE (baseline)", monthly_close, mask, daily_prices)
    stats_open,  _ = _run("ADJ OPEN  (stress)",   monthly_open,  mask, daily_prices,
                          exit_prices=monthly_open)

    # -------------------------------------------------------------------
    # Side-by-side comparison
    # -------------------------------------------------------------------
    if stats_close and stats_open:
        banner = "=" * 80
        print(f"\n{banner}")
        print("COMPARISON — ADJ CLOSE  vs  ADJ OPEN")
        print(banner)
        rows = [
            ("CAGR (%)",       "ann",    ".2f"),
            ("Volatility (%)", "vol",    ".2f"),
            ("Sharpe",         "sharpe", ".3f"),
            ("Max DD (%)",     "dd",     ".2f"),
            ("Calmar",         "calmar", ".3f"),
            ("Win Rate (%)",   "win",    ".2f"),
        ]
        fmt_hdr = f"{'Metric':<22}  {'ADJ CLOSE':>12}  {'ADJ OPEN':>12}  {'Delta':>10}"
        print(fmt_hdr)
        print("-" * len(fmt_hdr))
        for label, key, fmt in rows:
            vc = stats_close[key]
            vo = stats_open[key]
            delta = vo - vc
            sign  = "+" if delta >= 0 else ""
            print(f"{label:<22}  {vc:>12{fmt}}  {vo:>12{fmt}}  {sign}{delta:>{fmt}}")
        print(banner)


if __name__ == '__main__':
    main()
