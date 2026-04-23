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

# Regime-based long sizing and stop-loss thresholds (directional scheme)
# States sorted by HMM fitted volatility: LowVol (calm) → HighVol (stressed)
REGIME_SIZE = {'LowVol': 10, 'MedVol': 4, 'HighVol': 3}
REGIME_STOP = {'LowVol': -0.07, 'MedVol': -0.06, 'HighVol': -0.04}

# Barroso-Santa-Clara 2015 vol-scale constants
VOLSCALE_N      = 10     # Fixed book size — always top-10 by pred_prob
VOLSCALE_STOP   = -0.10  # Fixed stop (same as Bull regime)
VOLSCALE_WINDOW = 126    # Trailing realized-vol window (≈ 6 months trading days)
VOLSCALE_CAP    = 1.25   # Max gross exposure (25% leverage)
VOLSCALE_FLOOR  = 0.30   # Min gross exposure (70% cash)

# Hybrid HMM + vol-scaling constants (Formulations A, B, C)
# Three pre-committed regime-specific target vols — do NOT tune post-results
HYBRID_TARGET_VOL = {'LowVol': 0.24, 'MedVol': 0.20, 'HighVol': 0.12}
# Formulation A: position ratio per regime (LowVol=full, MedVol=60%, HighVol=30%)
HYBRID_POS_RATIO  = {'LowVol': 1.0,  'MedVol': 0.6,  'HighVol': 0.3}


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


# ---- Barroso-Santa-Clara 2015 helpers ------------------------------------
def compute_daily_strategy_returns(res_df, daily_prices, n_positions=VOLSCALE_N):
    """
    Generate daily equal-weight portfolio returns for the unscaled n-name
    long-only book.  Used as the raw return series for BSC vol estimation.

    Between each monthly rebalance, the top-n tickers by pred_prob are held
    with equal weight.  Returns a daily pd.Series indexed by date.
    """
    all_dates = sorted(res_df['date'].unique())
    all_daily_rets = {}

    for i, date in enumerate(all_dates):
        group = res_df[res_df['date'] == date]
        tickers = group.nlargest(n_positions, 'pred_prob')['ticker'].tolist()
        held = [t for t in tickers if t in daily_prices.columns]
        if not held:
            continue

        next_date = all_dates[i + 1] if i + 1 < len(all_dates) else None
        if next_date is None:
            continue

        window = daily_prices.loc[date:next_date, held]
        daily_ret = window.pct_change().mean(axis=1).dropna()
        for d, r in daily_ret.items():
            all_daily_rets[d] = r

    return pd.Series(all_daily_rets).sort_index()


def compute_target_vol(res_df, daily_prices,
                       n_positions=VOLSCALE_N, vol_window=VOLSCALE_WINDOW):
    """
    Barroso-Santa-Clara 2015 — calibration step.

    Target vol = MEDIAN of the 126-day rolling annualized realized vol of the
    unscaled (equal-weight top-N) momentum strategy.  Using the median rather
    than the mean is robust to crisis spikes and matches the paper's primary
    specification.  The target is computed once across the full OOS window —
    no look-ahead because we're using it as a fixed scaling denominator, not
    as a per-period forecast of future vol.

    Returns:
        target_vol  — float, annualized
        daily_rets  — pd.Series, daily strategy returns (for per-rebalance scaling)
    """
    daily_rets = compute_daily_strategy_returns(res_df, daily_prices, n_positions)
    rolling_vol = daily_rets.rolling(vol_window).std() * np.sqrt(252)
    target_vol = float(rolling_vol.dropna().median())
    return target_vol, daily_rets


# ---- Portfolio engine (long-only) ----------------------------------------
# ---- Weighting helpers ----------------------------------------------------
def _compute_weights(buys, daily_prices, date, method='equal', historical_returns=None):
    """
    Compute per-ticker portfolio weights using the specified method.
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
        # 60-day trailing realized volatility
        vol_lookback = daily_prices.loc[:date].tail(60)
        vols = []
        for t in tickers:
            if t in vol_lookback.columns:
                v = vol_lookback[t].pct_change().std()
                vols.append(max(v, 1e-6))  # floor to avoid division by zero
            else:
                vols.append(0.02)  # default 2% daily vol
        vols = np.array(vols)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        return {t: float(w) for t, w in zip(tickers, weights)}

    elif method == 'prob_invvol':
        # Probability x inverse volatility
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
        # Fractional Kelly (half-Kelly)
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

def simulate_portfolio(res_df, regimes, daily_prices,
                       sizing_scheme='directional',
                       volscale_params=None, lev_cost=None, weighting='prob_invvol',
                       regime_sizes=None):
    """
    Long-only rebalancer with daily path stop-loss tracing.

    sizing_scheme options:
      'directional' : HMM regime label controls position count {Bull:10, Neutral:4, Bear:3}
      'volscale'      : Barroso-Santa-Clara 2015 — fixed 10-name book, continuous exposure scaling
      'hybrid_a'      : HMM LowVol/MedVol/HighVol caps gross via pos_ratio, vol-scaling within
      'hybrid_b'      : HMM state selects regime-specific target vol {0.24, 0.20, 0.12}
      'hybrid_c'      : HMM posterior-weighted target vol (soft blend of the three)
      'hmm_vol_size'  : HMM-covariance-derived sizes (1/σ normalized). Requires regime_sizes dict.

    volscale_params dict keys (required for non-directional):
      'daily_rets'   — pd.Series of daily strategy returns for realized vol
      'target_vol'   — float (pure volscale only)
      'regime_info'  — dict[date -> {'label': str, 'probs': array}] (hybrid only)

    Returns:
      (port_returns, holdings_counts, extra)
      extra = None for 'directional',
              dict(scaling_factors, labels, target_vols_c) otherwise
    """
    if sizing_scheme not in ('directional', 'hmm_vol_size') and volscale_params is None:
        raise ValueError(f"sizing_scheme='{sizing_scheme}' requires volscale_params dict.")
    if sizing_scheme == 'hmm_vol_size' and regime_sizes is None:
        raise ValueError("sizing_scheme='hmm_vol_size' requires regime_sizes dict from get_regimes_and_vol_sizes().")

    _lev_cost = lev_cost if lev_cost is not None else LEVERAGE_COST_ANNUAL

    returns, holdings_counts, rebal_dates = [], [], []
    scaling_factors_track, labels_track, target_vols_c_track, turnover_track = [], [], [], []
    prev_weights = {}
    all_dates = sorted(res_df['date'].unique())

    for i, date in enumerate(all_dates):
        group = res_df[res_df['date'] == date]
        rebal_dates.append(pd.Timestamp(date))

        # ---- Realized vol (shared across all volscale-family schemes) ------
        realized_vol = None
        if sizing_scheme not in ('directional', 'hmm_vol_size'):
            daily_rets  = volscale_params['daily_rets']
            past_rets   = daily_rets[daily_rets.index < pd.Timestamp(date)]
            if len(past_rets) < VOLSCALE_WINDOW:
                logging.warning(
                    f"[{sizing_scheme}] Warmup: {len(past_rets)} days before {date} "
                    f"(need {VOLSCALE_WINDOW}). Using scale=1.0."
                )
                realized_vol = None   # triggers scale=1.0 fallback below
            else:
                realized_vol = past_rets.iloc[-VOLSCALE_WINDOW:].std() * np.sqrt(252)

        # ---- Per-scheme gross exposure & stop determination ----------------
        if sizing_scheme in ('directional', 'hmm_vol_size'):
            regime = regimes.get(date, 'Neutral')
            if sizing_scheme == 'hmm_vol_size' and regime_sizes is not None:
                size = regime_sizes.get(date, REGIME_SIZE.get(regime, 4))
            else:
                size = REGIME_SIZE.get(regime, 4)
            stop   = REGIME_STOP.get(regime, -0.07)
            weight_per_leg = 1.0 / size if size > 0 else 0.0
            gross_exposure = None    # not used in directional path

        elif sizing_scheme == 'volscale':
            # Barroso-Santa-Clara 2015 — Eq.6: w_t = σ* / h_t
            target_vol = volscale_params['target_vol']
            if realized_vol is None:
                sf = 1.0
            else:
                sf = min(VOLSCALE_CAP, max(VOLSCALE_FLOOR, target_vol / realized_vol))
            scaling_factors_track.append(sf)
            labels_track.append(None)
            target_vols_c_track.append(None)
            size   = VOLSCALE_N
            stop   = VOLSCALE_STOP
            gross_exposure   = sf
            weight_per_leg   = gross_exposure / size

        elif sizing_scheme == 'hybrid_a':
            # Formulation A: pos_ratio from HMM label × BSC scaling (target_vol_med fixed)
            ri    = volscale_params['regime_info'].get(date, {'label': 'MedVol', 'probs': None})
            label = ri['label']
            pos_ratio = HYBRID_POS_RATIO.get(label, 0.6)
            tv = HYBRID_TARGET_VOL['MedVol']   # fixed at 0.20 for A
            sf = 1.0 if realized_vol is None else \
                 min(VOLSCALE_CAP, max(VOLSCALE_FLOOR, tv / realized_vol))
            gross_exposure = pos_ratio * sf
            scaling_factors_track.append(gross_exposure)
            labels_track.append(label)
            target_vols_c_track.append(None)
            size   = VOLSCALE_N
            stop   = VOLSCALE_STOP
            weight_per_leg = gross_exposure / size

        elif sizing_scheme == 'hybrid_b':
            # Formulation B: HMM state selects regime-specific target vol
            ri    = volscale_params['regime_info'].get(date, {'label': 'MedVol', 'probs': None})
            label = ri['label']
            tv    = HYBRID_TARGET_VOL.get(label, 0.20)
            sf = 1.0 if realized_vol is None else \
                 min(VOLSCALE_CAP, max(VOLSCALE_FLOOR, tv / realized_vol))
            scaling_factors_track.append(sf)
            labels_track.append(label)
            target_vols_c_track.append(None)
            size   = VOLSCALE_N
            stop   = VOLSCALE_STOP
            gross_exposure   = sf
            weight_per_leg   = sf / size

        elif sizing_scheme == 'hybrid_c':
            # Formulation C: posterior-weighted target vol (soft regime blend)
            ri    = volscale_params['regime_info'].get(date, {'label': 'MedVol',
                                                               'probs': np.array([0.,1.,0.])})
            probs = ri['probs']   # [P_LowVol, P_MedVol, P_HighVol]
            tv    = float(probs[0] * 0.24 + probs[1] * 0.20 + probs[2] * 0.12)
            sf = 1.0 if realized_vol is None else \
                 min(VOLSCALE_CAP, max(VOLSCALE_FLOOR, tv / realized_vol))
            scaling_factors_track.append(sf)
            labels_track.append(ri['label'])
            target_vols_c_track.append(tv)
            size   = VOLSCALE_N
            stop   = VOLSCALE_STOP
            gross_exposure   = sf
            weight_per_leg   = sf / size

        else:
            raise ValueError(f"Unknown sizing_scheme: '{sizing_scheme}'")

        buys = group[group['pred_prob'] >= PROB_THRESHOLD].nlargest(size, 'pred_prob')
        holdings_counts.append(len(buys))
        
        # Calculate relative weights summing to 1.0 (if possible)
        rel_weights = _compute_weights(buys, daily_prices, date, method=weighting)
        
        # Scale relative weights to intended gross exposure for this rebalance.
        # Previously: curr_weights = {t: weight_per_leg for t in buys['ticker']}
        # To maintain exact equivalence, sum of weights should be len(buys) * weight_per_leg
        target_total_weight = len(buys) * weight_per_leg
        curr_weights = {t: w * target_total_weight for t, w in rel_weights.items()}

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
            if ticker in d_window.columns and not d_window[ticker].dropna().empty:
                valid_prices = d_window[ticker].dropna()
                start_price = valid_prices.iloc[0]
                if start_price > 0:
                    path = valid_prices / start_price - 1.0
                    if (path <= stop).any():
                        invested_return += stop * w
                    else:
                        invested_return += path.iloc[-1] * w
            else:
                ret = group[group['ticker'] == ticker]['fwd_return'].values[0]
                invested_return += max(float(ret), float(stop)) * w

        # Cash / leverage handling
        total_invested = sum(curr_weights.values())
        if sizing_scheme != 'directional':
            if total_invested > 1.0:
                lev_portion = total_invested - 1.0
                lev_drag = lev_portion * _lev_cost * (days_held / 365.25)
                gross = invested_return - lev_drag
            else:
                cash_weight = 1.0 - total_invested
                gross = invested_return + cash_weight * rf_period
        else:
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

    extra = {
        'turnover_track': turnover_track,
        'scaling_factors': scaling_factors_track if sizing_scheme != 'directional' else None,
        'labels': labels_track if sizing_scheme != 'directional' else None,
        'target_vols_c': target_vols_c_track if sizing_scheme != 'directional' else None,
    }
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
