"""
Data fetcher for IC diagnostics.

Pulls monthly closing prices for the Nifty 50 universe from yfinance.
Caches to a local parquet file so subsequent runs don't re-download.

Returns a DataFrame: DatetimeIndex (month-end) × tickers, values = adjusted close.
"""

import os
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings('ignore')

CACHE_PATH = os.path.join(os.path.dirname(__file__), 'price_cache.csv')
DAILY_CACHE_PATH = os.path.join(os.path.dirname(__file__), 'daily_cache.csv')
FIRST_OPEN_CACHE = os.path.join(os.path.dirname(__file__), 'monthly_first_open_adj.csv')
LAST_OPEN_CACHE  = os.path.join(os.path.dirname(__file__), 'monthly_last_open_adj.csv')

def load_historical_composition(csv_paths):
    if isinstance(csv_paths, str):
        csv_paths = [csv_paths]
        
    master_df = None
    dates = None
    
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path, header=0)
        df.columns = df.columns.str.strip()
        
        if dates is None:
            dates = pd.to_datetime(df.columns, format='%b-%y') + pd.offsets.MonthEnd(0)
            
        df = df.map(lambda x: str(x).strip() + '.NS' if pd.notna(x) and str(x).strip() != 'nan' else np.nan)
        
        if master_df is None:
            master_df = df
        else:
            master_df = pd.concat([master_df, df], axis=0, ignore_index=True)
    
    unique_tickers = pd.unique(master_df.values.ravel())
    unique_tickers = [t for t in unique_tickers if pd.notna(t)]
    
    mask = pd.DataFrame(False, index=dates, columns=unique_tickers)
    for col, date in zip(master_df.columns, dates):
        tickers_in_month = master_df[col].dropna()
        mask.loc[date, tickers_in_month] = True
        
    return mask, list(unique_tickers)


def fetch_monthly_prices(csv_paths, start, end, cache_path=CACHE_PATH, force_refresh=False):
    """
    Download daily prices, resample to month-end closes.

    Why month-end: the paper forms portfolios monthly and holds for one month.
    Using month-end closes aligns entry/exit timing and avoids intra-month
    lookahead when computing momentum and forward returns.
    """
    mask, tickers = load_historical_composition(csv_paths)

    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached prices from {cache_path}")
        prices = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return prices, mask

    print(f"Downloading daily prices for {len(tickers)} tickers...")
    print(f"Date range: {start} to {end}")

    # yfinance bulk download — faster than per-ticker
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=True,
        progress=True,
        threads=True
    )

    # yfinance returns MultiIndex columns (Price, Ticker) for multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        closes = raw['Close']
    else:
        # Single ticker edge case
        closes = raw[['Close']]
        closes.columns = [tickers[0]]

    # Resample to month-end: take the last available trading day each month
    monthly = closes.resample('ME').last()

    # Drop tickers with too little data (less than 60 months — need this for
    # the longest lookback window)
    min_obs = 60
    valid_cols = monthly.columns[monthly.notna().sum() >= min_obs]
    monthly = monthly[valid_cols]

    # Report coverage
    total_months = len(monthly)
    print(f"\nMonthly price matrix: {total_months} months × {len(monthly.columns)} tickers")
    print(f"Date range: {monthly.index[0].strftime('%Y-%m')} to {monthly.index[-1].strftime('%Y-%m')}")

    missing_pct = monthly.isna().sum().sum() / monthly.size * 100
    print(f"Missing data: {missing_pct:.1f}%")

    # Cache
    monthly.to_csv(cache_path)
    print(f"Cached to {cache_path}")

    return monthly, mask


def fetch_daily_prices(tickers, start, end, cache_path=DAILY_CACHE_PATH, force_refresh=False):
    """
    Daily adjusted closes for the full ticker set. Cached to disk because
    the intra-month stop-loss engine needs to walk every day's close.
    """
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached daily prices from {cache_path}")
        return pd.read_csv(cache_path, index_col=0, parse_dates=True)

    print(f"Downloading daily prices for {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(tickers, start=start, end=end, interval='1d',
                      auto_adjust=True, progress=True, threads=True)
    if isinstance(raw.columns, pd.MultiIndex):
        daily = raw['Close']
    else:
        daily = raw[['Close']]
        daily.columns = [tickers[0]]
    daily = daily.sort_index()
    daily.to_csv(cache_path)
    print(f"Cached daily prices to {cache_path}")
    return daily


def fetch_daily_hlc(tickers, start, end, cache_path=os.path.join(os.path.dirname(__file__), 'daily_hlc_cache.pkl'), force_refresh=False):
    """
    Fetch High, Low, and Close daily data for ATR calculation and stop-loss monitoring.
    Saves to a pickle file to preserve the MultiIndex.
    """
    if os.path.exists(cache_path) and not force_refresh:
        print(f"Loading cached daily HLC data from {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"Downloading daily HLC prices for {len(tickers)} tickers ({start} to {end})...")
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        interval='1d',
        auto_adjust=True,
        progress=True,
        threads=True
    )
    
    if isinstance(raw.columns, pd.MultiIndex):
        hlc = raw[['High', 'Low', 'Close']]
    else:
        hlc = raw[['High', 'Low', 'Close']]
        # Reconstruct into MultiIndex if it's a single ticker
        hlc.columns = pd.MultiIndex.from_product([['High', 'Low', 'Close'], [tickers[0]]])
        
    hlc = hlc.sort_index()
    hlc.to_pickle(cache_path)
    print(f"Cached daily HLC data to {cache_path}")
    return hlc


def fetch_monthly_open_prices(tickers, start, end,
                              first_cache=FIRST_OPEN_CACHE, last_cache=LAST_OPEN_CACHE,
                              force_refresh=False):
    """First-trading-day and last-trading-day Open of each month, auto-adjusted.

    Both matrices are month-end indexed (first_open is shifted from 'MS' to 'ME'
    so it aligns with monthly_close). Used by execution_realism.py.
    """
    yf_tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]

    if os.path.exists(first_cache) and os.path.exists(last_cache) and not force_refresh:
        print(f"Loading cached monthly opens from {first_cache}, {last_cache}")
        fo = pd.read_csv(first_cache, index_col=0, parse_dates=True)
        lo = pd.read_csv(last_cache,  index_col=0, parse_dates=True)
        return fo, lo

    print(f"Downloading daily opens for {len(yf_tickers)} tickers...")
    raw = yf.download(yf_tickers, start=start, end=end, interval='1d',
                      progress=False, auto_adjust=True, group_by='column')
    opens = raw['Open']

    first_open = opens.resample('MS').first()
    last_open  = opens.resample('ME').last()
    first_open.index = first_open.index + pd.offsets.MonthEnd(0)

    first_open.to_csv(first_cache)
    last_open.to_csv(last_cache)
    print(f"Cached monthly opens -> {first_cache}, {last_cache}  (shape {first_open.shape})")
    return first_open, last_open


def compute_forward_returns(prices):
    """
    Compute one-month-ahead simple returns for each stock.

    R_{t+1} = S_{t+1} / S_t - 1

    These are the "actuals" that the IC is computed against.
    The return at index t represents the return realized from
    month-end t to month-end t+1 — so when we compute IC at time t,
    we correlate momentum_t with fwd_return_t (which uses price at t+1).
    """
    fwd_ret = prices.shift(-1) / prices - 1
    # Drop the last row (no forward return available)
    fwd_ret = fwd_ret.iloc[:-1]
    return fwd_ret
