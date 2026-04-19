import numpy as np
import pandas as pd
import yfinance as yf
from catboost import CatBoostClassifier
import logging

from config import (
    TRANSACTION_COST_BPS, SHORT_BORROW_COST_ANNUAL, RISK_FREE_ANNUAL
)

# ---- Portfolio constants --------------------------------------------------
PROB_THRESHOLD = 0.55
SHORT_PROB_THRESHOLD = 0.45
TX_COST_SIDE = TRANSACTION_COST_BPS / 10000.0    # bps per side
TX_COST_RT = 2.0 * TX_COST_SIDE                  # round-trip cost
SHORT_BORROW_MTH = SHORT_BORROW_COST_ANNUAL / 12.0

REGIME_SIZE = {'Bull': 10, 'Neutral': 4, 'Bear': 3}
REGIME_STOP = {'Bull': -0.10, 'Neutral': -0.07, 'Bear': -0.05}
SHORT_REGIME_SIZE = {'Bull': 3, 'Neutral': 4, 'Bear': 10}
SHORT_REGIME_STOP = {'Bull': 0.10, 'Neutral': 0.07, 'Bear': 0.05}

# ---- Feature panel --------------------------------------------------------
def build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks):
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
        print(f"Not enough dates to run valid walk-forward: {len(dates)}")
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

# ---- Regime classifier (HMM on Nifty 50) ---------------------
def forward_backward(returns, means, stds, trans, init):
    n = len(returns)
    K = len(means)
    log_emit = np.zeros((n, K))
    for k in range(K):
        log_emit[:, k] = (-0.5 * np.log(2 * np.pi * stds[k] ** 2)
                          - (returns - means[k]) ** 2 / (2 * stds[k] ** 2))
    log_trans = np.log(trans + 1e-15)
    log_init  = np.log(init + 1e-15)
    log_alpha = np.full((n, K), -np.inf)
    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, n):
        for j in range(K):
            log_alpha[t, j] = (np.logaddexp.reduce(log_alpha[t - 1] + log_trans[:, j])
                               + log_emit[t, j])
    log_beta = np.zeros((n, K))
    for t in range(n - 2, -1, -1):
        for i in range(K):
            log_beta[t, i] = np.logaddexp.reduce(
                log_trans[i] + log_emit[t + 1] + log_beta[t + 1])
    log_gamma = log_alpha + log_beta
    log_gamma -= np.logaddexp.reduce(log_gamma, axis=1, keepdims=True)
    return np.exp(log_gamma)

def get_macro_regimes(dates, start_date, end_date):
    yf.set_tz_cache_location("/tmp/yfinance_tz_cache")
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    nifty = yf.download('^NSEI', start=start_date, end=end_date, interval='1d', progress=False)
    if isinstance(nifty.columns, pd.MultiIndex):
        prices = nifty['Close'] if 'Close' in nifty.columns.get_level_values(0) else nifty.iloc[:, -1]
    else:
        prices = nifty['Close'] if 'Close' in nifty.columns else nifty.iloc[:, -1]
    prices = prices.squeeze().ffill()

    # --- HMM Parameters (from config) ---
    means = np.array([0.0009, -0.0006, -0.0020])
    stds  = np.array([0.0080,  0.0145,  0.0310])
    trans = np.array([
        [0.970, 0.025, 0.005],
        [0.040, 0.945, 0.015],
        [0.020, 0.080, 0.900],
    ])
    init = np.array([0.7, 0.2, 0.1])
    
    # Map HMM state -> our logic
    state_map = {0: 'Bull', 1: 'Neutral', 2: 'Bear'}

    regimes = {}
    for d in dates:
        window = prices.loc[:d]
        if window.empty or len(window) < 20:
            regimes[d] = 'Neutral'
            continue
            
        returns_window = (window / window.shift(1) - 1).dropna().values
        
        if len(returns_window) == 0:
            regimes[d] = 'Neutral'
            continue
            
        gamma = forward_backward(returns_window, means, stds, trans, init)
        best_state = int(np.argmax(gamma[-1]))
        regimes[d] = state_map[best_state]
        
    return regimes

# ---- Portfolio engine -----------------------------------------------------
def simulate_portfolio(res_df, regimes, daily_prices, enable_shorts=True):
    """Rebalance with strictly validated DAILY stop-loss tracing for both Longs and Shorts."""
    returns, holdings_counts, short_counts, rebal_dates = [], [], [], []
    
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
        
        if next_date is not None:
            days_held = max(1, (pd.Timestamp(next_date) - pd.Timestamp(date)).days)
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
        tx_cost = turnover * TX_COST_SIDE
        
        net = gross - tx_cost
        returns.append(net)
        prev_weights = curr_weights

    return pd.Series(returns, index=pd.DatetimeIndex(rebal_dates)), holdings_counts, short_counts

def performance_stats(port_ret, periods_per_year=12):
    rf_period = (1 + RISK_FREE_ANNUAL)**(1.0/periods_per_year) - 1.0
    
    total = (np.prod(1 + port_ret) - 1) * 100
    ann = (np.prod(1 + port_ret) ** (periods_per_year / max(1, len(port_ret))) - 1) * 100
    vol = port_ret.std() * np.sqrt(periods_per_year) * 100
    
    mean_excess = port_ret.mean() - rf_period
    sharpe = (mean_excess / port_ret.std() * np.sqrt(periods_per_year)) if port_ret.std() > 0 else 0.0
    
    cum = (1 + port_ret).cumprod()
    dd = ((cum - cum.cummax()) / cum.cummax()).min() * 100
    calmar = ann / abs(dd) if dd != 0 else 0.0
    win = (port_ret > 0).mean() * 100
    return dict(total=total, ann=ann, vol=vol, sharpe=sharpe, dd=dd, calmar=calmar, win=win)

def print_stats(stats, label, counts_long, counts_short=None, freq_label="mo"):
    banner = "=" * 80
    print(f"\n{banner}\nPORTFOLIO RESULT — {label}\n{banner}")
    if counts_long:
        print(f"  Avg long holdings / {freq_label}: {np.mean(counts_long):.1f}")
    if counts_short:
        print(f"  Avg short holdings / {freq_label}:{np.mean(counts_short):.1f}")
    
    print(f"  Total Return (%):       {stats['total']:.2f}")
    print(f"  Annualized Return (%):  {stats['ann']:.2f}")
    print(f"  Volatility (%):         {stats['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats['calmar']:.3f}")
    print(f"  Win Rate (%):           {stats['win']:.2f}")
