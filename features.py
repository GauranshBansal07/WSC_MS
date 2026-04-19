"""
Feature computation for IC diagnostics.

Two responsibilities:
1. Compute momentum factors at multiple lookback windows (log-return based,
   then rank-transformed cross-sectionally).
2. Compute Rank IC (Spearman correlation) between momentum ranks and
   forward return ranks for each month and lookback.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def compute_momentum(prices, lookback_months):
    """
    Compute log-return momentum for a single lookback window.

    M_{t,m} = ln(S_t / S_{t-m})

    Why log returns: additive across time, symmetric for gains/losses,
    and what the paper uses. For the IC calculation (rank-based), the
    choice between log and simple returns doesn't matter since ranking
    is monotone-invariant. But we use log for consistency.

    Args:
        prices: DataFrame, month-end prices (DatetimeIndex × tickers)
        lookback_months: int, formation period in months

    Returns:
        DataFrame of same shape, NaN where insufficient history.
    """
    # Log return over lookback window
    mom = np.log(prices / prices.shift(lookback_months))
    return mom


def compute_all_momentum(prices, lookback_windows):
    """
    Compute momentum for all lookback windows.

    Returns:
        dict of {lookback: momentum_DataFrame}
    """
    momentum_dict = {}
    for lb in lookback_windows:
        momentum_dict[lb] = compute_momentum(prices, lb)
    return momentum_dict


def rank_cross_section(df):
    """
    Rank-transform each row (cross-section) independently.

    Why rank: Spearman IC is defined on ranks. The paper also feeds
    ranked momentum into the ML models to remove outlier sensitivity.

    Uses average ranking for ties (scipy default for Spearman).
    """
    return df.rank(axis=1, method='average')


def compute_rank_ic_series(momentum_df, forward_returns_df):
    """
    Compute the Rank IC time series for a single lookback's momentum.

    For each month t:
        IC_t = Spearman(rank(momentum_t), rank(forward_return_t))

    where forward_return_t is the return from month t to month t+1.

    Args:
        momentum_df: DataFrame of momentum values (months × tickers)
        forward_returns_df: DataFrame of one-month-ahead returns (months × tickers)

    Returns:
        Series indexed by date, values = Rank IC for each month.
        Only months where both momentum and forward returns have at
        least 15 non-NaN stocks are included (need enough cross-sectional
        observations for a meaningful correlation).
    """
    # Align indices
    common_idx = momentum_df.index.intersection(forward_returns_df.index)
    common_cols = momentum_df.columns.intersection(forward_returns_df.columns)

    mom = momentum_df.loc[common_idx, common_cols]
    fwd = forward_returns_df.loc[common_idx, common_cols]

    ic_values = {}
    min_stocks = 15  # minimum cross-sectional observations for meaningful Spearman

    for date in common_idx:
        mom_row = mom.loc[date].dropna()
        fwd_row = fwd.loc[date].dropna()

        # Intersect: only stocks with both momentum and forward return
        valid_tickers = mom_row.index.intersection(fwd_row.index)

        if len(valid_tickers) < min_stocks:
            continue

        m = mom_row[valid_tickers].values
        f = fwd_row[valid_tickers].values

        corr, pval = spearmanr(m, f)

        if not np.isnan(corr):
            ic_values[date] = {
                'ic': corr,
                'pval': pval,
                'n_stocks': len(valid_tickers)
            }

    result = pd.DataFrame(ic_values).T
    result.index.name = 'date'
    return result


def compute_all_rank_ics(momentum_dict, forward_returns_df):
    """
    Compute Rank IC series for all lookback windows.

    Returns:
        dict of {lookback: DataFrame with columns [ic, pval, n_stocks]}
    """
    ic_dict = {}
    for lb, mom_df in momentum_dict.items():
        ic_dict[lb] = compute_rank_ic_series(mom_df, forward_returns_df)
    return ic_dict
