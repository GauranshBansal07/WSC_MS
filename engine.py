import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
import logging

from config import (
    TRANSACTION_COST_BPS, RISK_FREE_ANNUAL, LEVERAGE_COST_ANNUAL
)

# ---- Portfolio constants --------------------------------------------------
PROB_THRESHOLD = 0.55
TX_COST_SIDE = TRANSACTION_COST_BPS / 10000.0    # bps per side

# Regime-based long sizing and stop-loss thresholds
# States sorted by HMM fitted volatility: LowVol (calm) → HighVol (stressed)
REGIME_SIZE = {'LowVol': 10, 'MedVol': 4, 'HighVol': 3}
REGIME_STOP = {'LowVol': -0.07, 'MedVol': -0.06, 'HighVol': -0.04}


# ---- Feature panel --------------------------------------------------------
def build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks):
    """Stack panel to (Date, Ticker) rows with cross-sectional label and features."""
    mask_aligned = mask.reindex(index=prices.index, columns=prices.columns).fillna(False)
    cfwd = fwd_returns.where(mask_aligned)

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


# ---- Weighting helpers ----------------------------------------------------
def _compute_weights(buys, daily_prices, date, method='prob_invvol', historical_returns=None):
    """
    Compute per-ticker portfolio weights.
    Normalized so sum(weights) = 1.0 (or 0 if no buys).
    """
    tickers = buys['ticker'].values
    n = len(tickers)
    if n == 0:
        return {}

    if method == 'equal':
        w = 1.0 / n
        return {t: w for t in tickers}

    elif method == 'probability':
        probs = buys['pred_prob'].values
        weights = probs / probs.sum()
        return {t: float(w) for t, w in zip(tickers, weights)}

    elif method == 'inverse_vol':
        vol_lookback = daily_prices.loc[:date].tail(60)
        vols = []
        for t in tickers:
            if t in vol_lookback.columns:
                v = vol_lookback[t].pct_change().std()
                vols.append(max(v, 1e-6))
            else:
                vols.append(0.02)
        vols = np.array(vols)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return {t: float(w) for t, w in zip(tickers, weights)}

    elif method == 'prob_invvol':
        probs = buys['pred_prob'].values
        vol_lookback = daily_prices.loc[:date].tail(60)
        vols = []
        for t in tickers:
            if t in vol_lookback.columns:
                v = vol_lookback[t].pct_change().std()
                vols.append(max(v, 1e-6))
            else:
                vols.append(0.02)
        vols = np.array(vols)
        with np.errstate(divide='ignore', invalid='ignore'):
            raw = probs * (1.0 / vols)
        if raw.sum() == 0:
            weights = np.ones(n) / n
        else:
            weights = raw / raw.sum()
        return {t: float(w) for t, w in zip(tickers, weights)}

    elif method == 'kelly':
        kelly_weights = []
        for idx, row in buys.iterrows():
            p = row['pred_prob']
            b = 1.0
            k = (p * b - (1 - p)) / b
            k = max(k, 0.0)
            kelly_weights.append(0.5 * k)
        kelly_weights = np.array(kelly_weights)
        if kelly_weights.sum() > 0:
            weights = kelly_weights / kelly_weights.sum()
        else:
            weights = np.ones(n) / n
        return {t: float(w) for t, w in zip(tickers, weights)}

    else:
        raise ValueError(f"Unknown weighting method: '{method}'")


# ---- Portfolio engine (long-only, directional HMM sizing) ----------------
def simulate_portfolio(res_df, regimes, daily_prices,
                       weighting='prob_invvol', monthly_prices=None):
    """
    Long-only monthly rebalancer with HMM regime sizing and daily stop-loss.

    Regime gates position count {LowVol:10, MedVol:4, HighVol:3} and stop
    {LowVol:−7%, MedVol:−6%, HighVol:−4%}.

    Stop-loss uses a dual-tranche approach:
      w_held — carried over from last month, entry = last month-end close
      w_new  — added this month, entry = this month-end close
    Each tranche is monitored independently against daily closes.

    monthly_prices is used as the canonical entry-price source to avoid the
    date-alignment bug where daily_prices.iloc[0] can be the next month's
    first-day price when month-end falls on a weekend/holiday.

    Returns (port_returns, holdings_counts, extra)
      extra = dict(turnover_track)
    """
    returns, holdings_counts, rebal_dates, turnover_track = [], [], [], []
    prev_weights = {}
    all_dates = sorted(res_df['date'].unique())

    for i, date in enumerate(all_dates):
        group = res_df[res_df['date'] == date]
        rebal_dates.append(pd.Timestamp(date))

        regime = regimes.get(date, 'MedVol')
        size   = REGIME_SIZE.get(regime, 4)
        stop   = REGIME_STOP.get(regime, -0.06)
        weight_per_leg = 1.0 / size if size > 0 else 0.0

        buys = group[group['pred_prob'] >= PROB_THRESHOLD].nlargest(size, 'pred_prob')
        holdings_counts.append(len(buys))

        rel_weights = _compute_weights(buys, daily_prices, date, method=weighting)
        target_total_weight = len(buys) * weight_per_leg
        curr_weights = {t: w * target_total_weight for t, w in rel_weights.items()}

        prev_date = all_dates[i - 1] if i > 0 else None
        next_date = all_dates[i + 1] if i + 1 < len(all_dates) else None
        if next_date is not None:
            days_held = max(1, (pd.Timestamp(next_date) - pd.Timestamp(date)).days)
            d_window = daily_prices.loc[date:next_date]
        else:
            days_held = 30
            d_window = daily_prices.loc[date:]

        rf_period = (1 + RISK_FREE_ANNUAL) ** (days_held / 365.25) - 1.0
        invested_return = 0.0

        for ticker in buys['ticker'].values:
            w = curr_weights.get(ticker, 0.0)

            # Dual-tranche stop-loss:
            #   w_held uses last month-end close as entry price
            #   w_new  uses this month-end close as entry price
            w_prev = prev_weights.get(ticker, 0.0)
            w_held = min(w_prev, w)
            w_new  = max(0.0, w - w_prev)

            def _mp(d):
                if (monthly_prices is not None
                        and d is not None
                        and ticker in monthly_prices.columns
                        and d in monthly_prices.index):
                    ep = monthly_prices.loc[d, ticker]
                    if pd.notna(ep) and ep > 0:
                        return float(ep)
                return None

            curr_entry = _mp(date)
            prev_entry = _mp(prev_date)

            if ticker in d_window.columns and not d_window[ticker].dropna().empty:
                valid_prices = d_window[ticker].dropna()
                exit_price   = float(valid_prices.iloc[-1])

                fallback = float(valid_prices.iloc[0])
                if curr_entry is None:
                    curr_entry = fallback
                if prev_entry is None:
                    prev_entry = curr_entry

                def _tranche(tranche_w, entry):
                    if tranche_w <= 0 or entry <= 0:
                        return 0.0
                    path = valid_prices / entry - 1.0
                    if (path <= stop).any():
                        return stop * tranche_w
                    return (exit_price / entry - 1.0) * tranche_w

                invested_return += _tranche(w_held, prev_entry)
                invested_return += _tranche(w_new,  curr_entry)
            else:
                ret = group[group['ticker'] == ticker]['fwd_return'].values[0]
                invested_return += max(float(ret), float(stop)) * w

        total_invested = sum(curr_weights.values())
        cash_weight = max(0.0, 1.0 - total_invested)
        gross = invested_return + cash_weight * rf_period

        all_tickers = set(prev_weights.keys()).union(curr_weights.keys())
        turnover = sum(abs(curr_weights.get(t, 0.0) - prev_weights.get(t, 0.0))
                       for t in all_tickers)
        turnover_track.append(turnover)
        tx_cost = turnover * TX_COST_SIDE

        net = gross - tx_cost
        returns.append(net)
        prev_weights = curr_weights

    extra = {'turnover_track': turnover_track}
    return pd.Series(returns, index=pd.DatetimeIndex(rebal_dates)), holdings_counts, extra


def performance_stats(port_ret, periods_per_year=12):
    rf_period = (1 + RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0

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


def print_stats(stats, label, counts_long, freq_label="mo"):
    banner = "=" * 80
    print(f"\n{banner}\nPORTFOLIO RESULT — {label}\n{banner}")
    if counts_long:
        print(f"  Avg long holdings / {freq_label}: {np.mean(counts_long):.1f}")
    print(f"  Total Return (%):       {stats['total']:.2f}")
    print(f"  Annualized Return (%):  {stats['ann']:.2f}")
    print(f"  Volatility (%):         {stats['vol']:.2f}")
    print(f"  Sharpe Ratio:           {stats['sharpe']:.3f}")
    print(f"  Max Drawdown (%):       {stats['dd']:.2f}")
    print(f"  Calmar Ratio:           {stats['calmar']:.3f}")
    print(f"  Win Rate (%):           {stats['win']:.2f}")
