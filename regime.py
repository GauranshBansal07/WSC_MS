import numpy as np
import pandas as pd
import yfinance as yf
import logging

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


def _get_fixed_hmm(rebal_dates, prices):
    means = np.array([0.0009, -0.0006, -0.0020])
    stds  = np.array([0.0080,  0.0145,  0.0310])
    trans = np.array([
        [0.970, 0.025, 0.005],
        [0.040, 0.945, 0.015],
        [0.020, 0.080, 0.900],
    ])
    init = np.array([0.7, 0.2, 0.1])
    state_map = {0: 'Bull', 1: 'Neutral', 2: 'Bear'}
    
    regimes = {}
    for d in rebal_dates:
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


def _get_learned_hmm(rebal_dates, prices):
    try:
        from hmmlearn.hmm import GaussianHMM
    except ImportError:
        logging.warning("hmmlearn not installed! Please run 'pip install hmmlearn'. Returning 'Neutral'.")
        return {d: 'Neutral' for d in rebal_dates}
        
    df = pd.DataFrame(prices)
    df.columns = ['Close']
    df['log_ret'] = np.log(df['Close'] / df['Close'].shift(1))
    df['vol'] = df['log_ret'].rolling(20).std()
    df = df.dropna()
    
    regimes = {}
    last_fit_date = None
    best_model = None
    label_map = None
    
    warned_data = False
    
    for d in rebal_dates:
        # Strictly use data BEFORE rebalance date
        data_up_to_t = df.loc[df.index < d]
        
        if len(data_up_to_t) < 252:
            if not warned_data:
                logging.warning(f"Not enough data to fit HMM before {d} (< 252 days). Using 'Neutral'.")
                warned_data = True
            regimes[d] = 'Neutral'
            continue
            
        # Refit every 12 months
        if last_fit_date is None or (d - last_fit_date).days >= 365:
            X_train = data_up_to_t[['log_ret', 'vol']].values
            
            best_score = -np.inf
            current_best_model = None
            
            # 5 random restarts (seeds 0..4)
            for seed in range(5):
                model = GaussianHMM(n_components=3, covariance_type='full', n_iter=100, random_state=seed)
                try:
                    model.fit(X_train)
                    score = model.score(X_train)
                    if score > best_score:
                        best_score = score
                        current_best_model = model
                except Exception:
                    continue
                    
            if current_best_model is not None:
                best_model = current_best_model
                last_fit_date = d
                
                # Permute labels: sort by means_[:, 0] (log_ret mean) ascending
                mean_returns = best_model.means_[:, 0]
                sorted_idx = np.argsort(mean_returns)
                
                # lowest-mean -> Bear, middle-mean -> Neutral, highest-mean -> Bull
                label_map = {
                    sorted_idx[0]: 'Bear',
                    sorted_idx[1]: 'Neutral',
                    sorted_idx[2]: 'Bull'
                }
            else:
                logging.warning(f"All 5 seeds failed to fit at {d}. Keeping previous model or Neutral.")
                
        if best_model is not None:
            X_infer = data_up_to_t[['log_ret', 'vol']].values
            try:
                state_seq = best_model.predict(X_infer)
                regimes[d] = label_map[state_seq[-1]]
            except Exception:
                regimes[d] = 'Neutral'
        else:
            regimes[d] = 'Neutral'
            
    return regimes


def get_regimes(rebal_dates, start_date, end_date, method='fixed_hmm', **kwargs):
    """
    Main entry point for macro regime detection.
    Supports: 'fixed_hmm', 'learned_hmm', or 'none'
    Returns: dict mapping rebalance date -> 'Bull'|'Neutral'|'Bear'
    """
    if method == 'none':
        return {d: 'Neutral' for d in rebal_dates}
        
    yf.set_tz_cache_location("/tmp/yfinance_tz_cache")
    logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    nifty = yf.download('^NSEI', start=start_date, end=end_date, interval='1d', progress=False)
    
    # Extract close prices cleanly
    if isinstance(nifty.columns, pd.MultiIndex):
        prices = nifty['Close'] if 'Close' in nifty.columns.get_level_values(0) else nifty.iloc[:, -1]
    else:
        prices = nifty['Close'] if 'Close' in nifty.columns else nifty.iloc[:, -1]
        
    prices = prices.squeeze().ffill()

    if method == 'fixed_hmm':
        return _get_fixed_hmm(rebal_dates, prices)
    elif method == 'learned_hmm':
        return _get_learned_hmm(rebal_dates, prices)
    else:
        raise ValueError(f"Unknown regime method: {method}")
