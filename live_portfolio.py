"""
Live Portfolio Generator
=========================================
Run this script at the start of the month to generate your target portfolio.
It pulls the latest data up to today, trains the CatBoost classifier, 
evaluates the current HMM regime, and outputs exact weights & stop-losses.

Usage:
    python live_portfolio.py --index nifty100
"""

import sys
import argparse
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')

from config import (
    DATA_START, HISTORICAL_COMPOSITION_CSV,
    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS
)
from data_fetcher import load_historical_composition
from features import compute_all_momentum
from regime import get_regimes
import engine as eng

def generate_live_portfolio(index_name):
    print(f"\n=================================================================")
    print(f"  LIVE PORTFOLIO GENERATOR: {index_name.upper()}")
    print(f"=================================================================\n")

    today = pd.Timestamp(datetime.today()).normalize()
    print(f"[*] Execution Date: {today.strftime('%Y-%m-%d')}")

    # ── 1. Fetch Prices & Composition ─────────────────────────────
    print(f"[*] Fetching live prices from Yahoo Finance...")
    
    pit_indices = {
        'nifty50': [HISTORICAL_COMPOSITION_CSV],
        'nifty100': [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    }
    csv_paths = pit_indices[index_name]
    mask, tickers = load_historical_composition(csv_paths)

    # Download all daily prices
    yf.set_tz_cache_location("/tmp/yfinance_tz_cache")
    raw_daily = yf.download(tickers, start=DATA_START, end=today.strftime('%Y-%m-%d'), progress=False)
    
    if isinstance(raw_daily.columns, pd.MultiIndex):
        daily_prices = raw_daily['Close'] if 'Close' in raw_daily.columns.get_level_values(0) else raw_daily.iloc[:, -1]
    else:
        daily_prices = raw_daily

    daily_prices = daily_prices.ffill().dropna(how='all')
    
    # Resample to month-end for our feature generation
    monthly_prices = daily_prices.resample('ME').last()
    target_date = monthly_prices.index[-1]
    print(f"[*] Target Rebalance Date: {target_date.strftime('%Y-%m-%d')}")

    # Extend mask to cover target date (if out-of-date)
    mask = mask.reindex(index=monthly_prices.index)
    mask = mask.ffill().fillna(False) # Forward fill last known composition

    # ── 2. Compute Features ─────────────────────────────────────────
    print(f"[*] Computing cross-sectional momentum features...")
    momentum_dict = compute_all_momentum(monthly_prices, LOOKBACK_WINDOWS)

    fwd_returns = monthly_prices.shift(-1) / monthly_prices - 1.0
    
    # Trick `build_stacked_dataset` into not dropping the current target month 
    # by spoofing the unobservable forward return as 0.0
    fwd_returns.loc[target_date] = 0.0 

    stacked = eng.build_stacked_dataset(
        monthly_prices, mask, fwd_returns, momentum_dict, LOOKBACK_WINDOWS,
        enhanced_features=False
    )

    feature_cols = [c for c in stacked.columns if c.startswith(('mom_', 'zscore_'))]
    
    # Split train vs prediction (test)
    level0 = stacked.index.get_level_values(0)
    train_mask = level0 < target_date
    test_mask = level0 == target_date

    X_train = stacked.loc[train_mask, feature_cols]
    y_train = stacked.loc[train_mask, 'target']
    
    X_test = stacked.loc[test_mask, feature_cols]
    current_universe = X_test.index.get_level_values(1)

    if len(X_test) == 0:
        print("\n[ERROR] No valid features for target date. Are price loads correct?")
        return

    # ── 3. Train Model & Predict ────────────────────────────────────
    print(f"[*] Training Walk-Forward CatBoost on {len(X_train)} historical samples...")
    model = CatBoostClassifier(
        iterations=150, depth=4, learning_rate=0.05,
        l2_leaf_reg=5, random_seed=42, verbose=False, thread_count=-1
    )
    model.fit(X_train, y_train)

    print(f"[*] Predicting probabilities for {len(X_test)} current stocks...")
    proba = model.predict_proba(X_test)[:, 1]

    candidates = pd.DataFrame({
        'ticker': current_universe,
        'pred_prob': proba
    })
    
    # ── 4. Regime Detection ─────────────────────────────────────────
    print(f"[*] Evaluating Current Market Regime...")
    padding_start = (target_date - pd.DateOffset(months=24)).strftime('%Y-%m-%d')
    # Use learned_hmm (3-state) as our optimal default
    regimes = get_regimes([target_date], padding_start, today.strftime('%Y-%m-%d'), method='learned_hmm')
    current_regime = regimes.get(target_date, 'Neutral')
    
    max_size = eng.REGIME_SIZE.get(current_regime, 4)
    regime_stop = eng.REGIME_STOP.get(current_regime, -0.05)

    print(f"\n=================================================================")
    print(f"  MARKET STATE")
    print(f"=================================================================")
    print(f"  Regime Level      : {current_regime.upper()}")
    print(f"  Maximum Positions : {max_size} stocks")
    print(f"  Hard Stop-Loss    : {regime_stop * 100:.1f}%")
    print(f"  Model Threshold   : {eng.PROB_THRESHOLD}")

    # ── 5. Portfolio Selection & Sizing ─────────────────────────────
    # Filter by threshold and take top N
    buys = candidates[candidates['pred_prob'] >= eng.PROB_THRESHOLD].nlargest(max_size, 'pred_prob')
    
    if len(buys) == 0:
        print(f"\n[!] NO STOCKS MET PROBABILITY THRESHOLD. ADVISING 100% CASH.")
        return

    # Compute weights using the centralized engine logic
    weights = eng._compute_weights(buys, daily_prices, target_date, method='prob_invvol')
    
    # Normalize and rescale as per live_portfolio's specific cash-handling logic
    invested_fraction = min(1.0, len(buys) / max_size)
    for t in weights:
        weights[t] = weights[t] * invested_fraction

    cash_final = 1.0 - sum(weights.values())

    print(f"\n=================================================================")
    print(f"  TARGET PORTFOLIO ALLOCATION")
    print(f"=================================================================")
    if cash_final > 0.01:
        print(f"  CASH ALLOCATION   : {cash_final * 100:.1f}% (Yields Risk-Free Rate)")
    print()
    
    # Sort by weight descending
    final_alloc = sorted(weights.items(), key=lambda x: x[1], reverse=True)
    
    print(f"  {'TICKER':<15s} | {'WEIGHT':>8s} | {'SIGNAL PROB':>12s} | {'STOP-LOSS'}")
    print(f"  {'-'*15}-+-{'-'*8}-+-{'-'*12}-+-{'-'*10}")
    
    for t, w in final_alloc:
        p = buys[buys['ticker'] == t]['pred_prob'].values[0]
        print(f"  {t:<15s} | {w*100:>7.1f}% | {p:>12.3f} | {regime_stop*100:>8.1f}%")

    print(f"\n=================================================================")
    print(f"  EXECUTION NOTES")
    print(f"=================================================================")
    print(f"  1. Enter these positions at the open tomorrow.")
    print(f"  2. Monitor closing prices daily. If any ticker closes below")
    print(f"     its entry price by {regime_stop*100:.1f}%, liquidate it the next morning.")
    print(f"  3. Re-run this script at the end of the month.")
    print(f"=================================================================\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Live Portfolio Engine")
    parser.add_argument('--index', choices=['nifty50', 'nifty100'], default='nifty100')
    args = parser.parse_args()
    generate_live_portfolio(args.index)
