"""
CatBoost cross-sectional momentum strategy for Nifty 50 / Nifty 100.

Pipeline:
  1. Load point-in-time constituency and month-end prices.
  2. Build 10 momentum features (raw + cross-sectional z-score at 1/6/12/36/60m).
  3. Label each stock-month: 1 if forward return beats the cohort median, else 0.
  4. Walk-forward CatBoost classifier with expanding training window.
  5. Portfolio construction:
       - Regime filter (Nifty SMA stack -> Bull/Neutral/Bear).
       - Regime-sized book (10 / 4 / 3 names) above a 0.55 probability gate.
       - Equal-weight positions.
       - Monthly stop-loss approximation (floor monthly return at regime stop).
       - Turnover-scaled round-trip transaction cost.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score

sys.path.insert(0, os.path.dirname(__file__))
warnings.filterwarnings('ignore')

from config import (
    HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
    LOOKBACK_WINDOWS, DATA_START, DATA_END, TRANSACTION_COST_BPS,
    SHORT_BORROW_COST_ANNUAL,
)
from data_fetcher import (
    fetch_monthly_prices, fetch_daily_prices, compute_forward_returns,
)
from features import compute_all_momentum

# ---- Portfolio constants --------------------------------------------------
PROB_THRESHOLD = 0.55
SHORT_PROB_THRESHOLD = 0.45
TX_COST_SIDE = TRANSACTION_COST_BPS / 10000.0    # bps per side from config
TX_COST_RT = 2.0 * TX_COST_SIDE                  # round-trip cost (buy + sell)
SHORT_BORROW_MTH = SHORT_BORROW_COST_ANNUAL / 12.0

REGIME_SIZE = {'Bull': 10, 'Neutral': 4, 'Bear': 3}
REGIME_STOP = {'Bull': -0.10, 'Neutral': -0.07, 'Bear': -0.05}
SHORT_REGIME_SIZE = {'Bull': 3, 'Neutral': 4, 'Bear': 10}
SHORT_REGIME_STOP = {'Bull': 0.10, 'Neutral': 0.07, 'Bear': 0.05}


# ---- Feature panel --------------------------------------------------------
def build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks=[1, 6, 12, 36, 60]):
    """Stack panel to (Date, Ticker) rows with cross-sectional label and features."""
    mask_aligned = mask.reindex(index=prices.index, columns=prices.columns).fillna(False)
    cfwd = fwd_returns.where(mask_aligned)

    # Label: 1 if forward return > cross-sectional median that month.
    target = cfwd.apply(lambda x: x > x.median(), axis=1).astype(float)
    target = target.where(mask_aligned & cfwd.notna())

    combined = pd.concat([
        mask_aligned.stack().rename('is_constituent'),
        target.stack().rename('target'),
        cfwd.stack().rename('fwd_return'),
    ], axis=1)

    for lb in lookbacks:
        mom = momentum_dict[lb].where(mask_aligned)
        combined[f'mom_{lb}m'] = mom.stack()
        mom_mean = mom.mean(axis=1).values[:, None]
        mom_std = mom.std(axis=1).values[:, None]
        with np.errstate(divide='ignore', invalid='ignore'):
            z = (mom - mom_mean) / mom_std
        combined[f'zscore_{lb}m'] = z.stack()

    combined = combined[combined['is_constituent']].dropna()
    return combined


# ---- Walk-forward classifier ---------------------------------------------
def run_expanding_window(stacked_data, min_train_months=60):
    dates = sorted(stacked_data.index.get_level_values(0).unique())
    if len(dates) < min_train_months + 10:
        print(f"Not enough dates: {len(dates)}")
        return None

    feature_cols = [c for c in stacked_data.columns if c.startswith(('mom_', 'zscore_'))]
    level0 = stacked_data.index.get_level_values(0)
    results = []

    for t in range(min_train_months, len(dates)):
        train_end = dates[t - 1]
        test_date = dates[t]
        train_mask = level0 <= train_end
        test_mask = level0 == test_date

        X_train = stacked_data.loc[train_mask, feature_cols]
        y_train = stacked_data.loc[train_mask, 'target']
        X_test = stacked_data.loc[test_mask, feature_cols]
        y_test = stacked_data.loc[test_mask, 'target']
        if len(y_test) == 0:
            continue

        model = CatBoostClassifier(
            iterations=150, depth=4, learning_rate=0.05,
            l2_leaf_reg=5, random_seed=42, verbose=False, thread_count=-1,
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)

        results.append(pd.DataFrame({
            'date': test_date,
            'ticker': X_test.index.get_level_values(1),
            'actual': y_test.values,
            'fwd_return': stacked_data.loc[test_mask, 'fwd_return'].values,
            'pred_prob': proba,
            'pred_class': pred,
        }))
    return pd.concat(results, axis=0)


# ---- Regime classifier (daily SMA stack on Nifty 50) ---------------------
def get_macro_regimes(dates, start_date, end_date):
    import logging
    yf.set_tz_cache_location("/tmp/yfinance_tz_cache")
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    nifty = yf.download('^NSEI', start=start_date, end=end_date, interval='1d', progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        prices = nifty['Close'] if 'Close' in nifty.columns.get_level_values(0) else nifty.iloc[:, -1]
    else:
        prices = nifty['Close'] if 'Close' in nifty.columns else nifty.iloc[:, -1]
    prices = prices.squeeze().ffill()

    sma20 = prices.rolling(20).mean()
    sma50 = prices.rolling(50).mean()
    sma100 = prices.rolling(100).mean()
    r1 = (prices / prices.shift(20) - 1) * 100
    r3 = (prices / prices.shift(60) - 1) * 100
    r6 = (prices / prices.shift(126) - 1) * 100
    vol = (prices / prices.shift(1) - 1).rolling(252).std() * np.sqrt(252) * 100

    regimes = {}
    for d in dates:
        window = prices.loc[:d]
        if window.empty or len(window) < 126:
            regimes[d] = 'Neutral'
            continue
        idx = window.index[-1]
        p, s20, s50, s100 = prices.loc[idx], sma20.loc[idx], sma50.loc[idx], sma100.loc[idx]
        ret1, ret3, ret6, v = r1.loc[idx], r3.loc[idx], r6.loc[idx], vol.loc[idx]
        ma = int(s20 > s50) + int(s50 > s100) + int(p > s20)

        bull = (((p > s20 and p > s50 and p > s100) or (ret6 > 8 and ret3 > 5))
                and ret1 > 2 and ret3 > 3 and ret6 > 5 and ma >= 2)
        bear = (((p < s20 and p < s50 and p < s100) or (ret6 < -8 and ret3 < -5))
                and ret1 < -2 and ret3 < -3 and ret6 < -5 and ma <= 1)

        if bull and v < 22:
            regimes[d] = 'Bull'
        elif bear and v < 22:
            regimes[d] = 'Bear'
        else:
            regimes[d] = 'Neutral'
    return regimes


# Monthly approximate return functions removed: Daily trajectory is now used in simulate_portfolio


# ---- Portfolio engine -----------------------------------------------------
def simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True):
    """Monthly rebalance with strictly validated DAILY stop-loss tracing for both Longs and Shorts."""
    returns, holdings_counts, short_counts, rebal_dates = [], [], [], []
    
    from config import RISK_FREE_ANNUAL, SHORT_BORROW_COST_ANNUAL
    prev_weights = {}

    all_dates = sorted(res_df['date'].unique())

    for i, date in enumerate(all_dates):
        group = res_df[res_df['date'] == date]
        rebal_dates.append(pd.Timestamp(date))
        regime = regimes.get(date, 'Neutral')
        
        size = REGIME_SIZE[regime]
        stop = REGIME_STOP[regime]
        
        short_size = SHORT_REGIME_SIZE[regime]
        short_stop = SHORT_REGIME_STOP[regime]
        
        target_legs = size + (short_size if enable_shorts else 0)
        weight_per_leg = 1.0 / target_legs if target_legs > 0 else 0.0

        buys = group[group['pred_prob'] >= PROB_THRESHOLD].nlargest(size, 'pred_prob')
        shorts = group[group['pred_prob'] <= SHORT_PROB_THRESHOLD].nsmallest(short_size, 'pred_prob') if enable_shorts else pd.DataFrame(columns=group.columns)
        
        holdings_counts.append(len(buys))
        short_counts.append(len(shorts))
        
        curr_weights = {}
        for t in buys['ticker']: curr_weights[t] = weight_per_leg
        for t in shorts['ticker']: curr_weights[f"SHORT_{t}"] = weight_per_leg

        next_date = all_dates[i+1] if i + 1 < len(all_dates) else None
        
        # Determine exact duration to accurately compound risk-free rate and short borrow cost
        if next_date is not None:
            days_held = (pd.Timestamp(next_date) - pd.Timestamp(date)).days
            d_window = daily_prices.loc[date:next_date]
        else:
            days_held = 30 # Default approximation for terminal element
            d_window = daily_prices.loc[date:]
            
        rf_period = (1 + RISK_FREE_ANNUAL)**(days_held / 365.25) - 1.0
        short_borrow_period = SHORT_BORROW_COST_ANNUAL * (days_held / 365.25)

        invested_return = 0.0
        
        # Long execution
        for ticker in buys['ticker'].values:
            if ticker in d_window.columns and not d_window[ticker].dropna().empty:
                valid_prices = d_window[ticker].dropna()
                start_price = valid_prices.iloc[0]
                if start_price > 0:
                    path = valid_prices / start_price - 1.0
                    mask_stop = path <= stop
                    if mask_stop.any():
                        invested_return += stop * weight_per_leg
                    else:
                        invested_return += path.iloc[-1] * weight_per_leg
            else:
                ret = group[group['ticker'] == ticker]['fwd_return'].values[0]
                invested_return += max(float(ret), float(stop)) * weight_per_leg

        # Short execution
        for ticker in shorts['ticker'].values:
            if ticker in d_window.columns and not d_window[ticker].dropna().empty:
                valid_prices = d_window[ticker].dropna()
                start_price = valid_prices.iloc[0]
                if start_price > 0:
                    path = valid_prices / start_price - 1.0
                    mask_stop = path >= short_stop
                    if mask_stop.any():
                        invested_return -= short_stop * weight_per_leg
                    else:
                        invested_return -= path.iloc[-1] * weight_per_leg
            else:
                ret = group[group['ticker'] == ticker]['fwd_return'].values[0]
                capped = min(float(ret), float(short_stop))
                invested_return -= capped * weight_per_leg
                
            invested_return -= short_borrow_period * weight_per_leg

        cash_weight = max(0.0, 1.0 - (len(buys) + len(shorts)) * weight_per_leg)
        cash_return = cash_weight * rf_period
        gross = invested_return + cash_return
        
        all_tickers = set(prev_weights.keys()).union(curr_weights.keys())
        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0)) for t in all_tickers)
        tx_cost = turnover * (TRANSACTION_COST_BPS / 10000.0)
        
        net = gross - tx_cost
        returns.append(net)
        prev_weights = curr_weights

    return pd.Series(returns, index=pd.DatetimeIndex(rebal_dates)), holdings_counts, short_counts


def performance_stats(port_ret):
    from config import RISK_FREE_ANNUAL
    rf_monthly = (1 + RISK_FREE_ANNUAL)**(1.0/12.0) - 1.0
    
    total = (np.prod(1 + port_ret) - 1) * 100
    ann = (np.prod(1 + port_ret) ** (12 / len(port_ret)) - 1) * 100
    vol = port_ret.std() * np.sqrt(12) * 100
    
    mean_excess = port_ret.mean() - rf_monthly
    sharpe = (mean_excess / port_ret.std() * np.sqrt(12)) if port_ret.std() > 0 else 0.0
    
    cum = (1 + port_ret).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = ann / abs(dd) if dd != 0 else 0.0
    win = (port_ret > 0).mean() * 100
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, dd=dd, calmar=calmar, win=win)


def trailing_window_stats(port_ret, years=5):
    """Compute trailing window performance stats from dated monthly returns."""
    if len(port_ret) == 0:
        return None
    cutoff = port_ret.index.max() - pd.DateOffset(years=years)
    window = port_ret[port_ret.index > cutoff]
    if len(window) == 0:
        return None
    return performance_stats(window)


# ---- Universe driver ------------------------------------------------------
def evaluate_universe(universe_name, csv_paths):
    banner = "=" * 80
    print(f"\n{banner}\nCATBOOST STRATEGY — {universe_name}\n{banner}")

    print("\n[1] Loading monthly prices + composition mask...")
    prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
    daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, LOOKBACK_WINDOWS)

    print("\n[2] Building feature panel...")
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, LOOKBACK_WINDOWS)
    print(f"  Observations: {len(stacked)}  |  Avg monthly universe: "
          f"{stacked.groupby(level=0).size().mean():.1f}")

    print("\n[3] Walk-forward CatBoost classification...")
    res_df = run_expanding_window(stacked, min_train_months=60)
    if res_df is None:
        return

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")
    print(f"  Transaction cost (bps per side): {TRANSACTION_COST_BPS}")
    print(f"  Transaction cost (round-trip bps): {2 * TRANSACTION_COST_BPS}")

    print("\n[4] Classifying macro regimes...")
    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_macro_regimes(rebal_dates, padding_start, DATA_END)

    print(f"\n[5] Simulating portfolio with DAILY stop approximation + dynamic transaction costs...")
    
    port_ret_long, counts_long, _ = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=False)
    stats_long = performance_stats(port_ret_long)
    
    port_ret_ls, counts_ls, short_counts_ls = simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True)
    stats_ls = performance_stats(port_ret_ls)

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG ONLY\n{banner}")
    print(f"  Avg long holdings / mo: {np.mean(counts_long):.1f}")
    print(f"  Total Return (%):       {stats_long['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_long['ann']:.2f}")
    print(f"  Volatility (%):         {stats_long['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_long['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_long['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_long['calmar']:.3f}")

    print(f"\n{banner}\nPORTFOLIO RESULT — {universe_name} — LONG + SHORT\n{banner}")
    print(f"  Avg long holdings / mo: {np.mean(counts_ls):.1f}")
    print(f"  Avg short holdings / mo:{np.mean(short_counts_ls):.1f}")
    print(f"  Total Return (%):       {stats_ls['total']:.2f}")
    print(f"  Annualized Return (%):  {stats_ls['ann']:.2f}")
    print(f"  Volatility (%):         {stats_ls['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats_ls['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats_ls['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats_ls['calmar']:.3f}")

    stats_5y = trailing_window_stats(port_ret_ls, years=5)


    if stats_5y is not None:
        print("\n  --- Trailing 5-Year Metrics ---")
        print(f"  Total Return (%):       {stats_5y['total']:.2f}")
        print(f"  Annualized Return (%):  {stats_5y['ann']:.2f}")
        print(f"  Volatility (%):         {stats_5y['vol']:.2f}")
        print(f"  Sharpe Ratio:           {stats_5y['sharpe']:.3f}")
        print(f"  Max Drawdown (%):       {stats_5y['dd']:.2f}")
        print(f"  Calmar Ratio:           {stats_5y['calmar']:.3f}")
        print(f"  Win Rate (%):           {stats_5y['win']:.2f}")

    os.makedirs('output', exist_ok=True)
    suffix = universe_name.replace(' ', '_').lower()
    res_df.to_csv(f'output/catboost_preds_{suffix}.csv', index=False)
    print(f"\n  Predictions saved -> output/catboost_preds_{suffix}.csv")


def main():
    evaluate_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV])
    evaluate_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV])


if __name__ == '__main__':
    main()
