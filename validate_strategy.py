"""
WSC_MS Strategy Validation Suite
=================================
Comprehensive validation & visualization for the CatBoost + HMM
cross-sectional momentum strategy.

Tests:
  1. Monte Carlo Bootstrap — metric confidence intervals
  2. Monte Carlo Path Simulation — equity curve fanout
  3. Parameter Sensitivity — probability threshold sweep
  4. Parameter Sensitivity — stop-loss level sweep
  5. Parameter Sensitivity — regime sizing sweep
  6. Walk-Forward Stability — rolling Sharpe & Calmar
  7. Regime Performance Decomposition
  8. Turnover Analysis

Outputs 14 publication-quality PNGs to output/.

Usage:
  python validate_strategy.py --index nifty50
  python validate_strategy.py --index nifty100 --regime learned_hmm
"""

import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from matplotlib.gridspec import GridSpec
from sklearn.metrics import accuracy_score, precision_score

warnings.filterwarnings('ignore')

# Fix Windows console encoding (cp1252 can't handle Unicode box-drawing chars)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')

# ── Project imports ──────────────────────────────────────────────────────────
from config import (
    DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS,
    TRANSACTION_COST_BPS, RISK_FREE_ANNUAL
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats, print_stats,
    PROB_THRESHOLD, TX_COST_SIDE, REGIME_SIZE, REGIME_STOP
)
from regime import get_regimes

# ── Aesthetics ───────────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0d1117',
    'axes.facecolor': '#161b22',
    'axes.edgecolor': '#30363d',
    'axes.labelcolor': '#c9d1d9',
    'text.color': '#c9d1d9',
    'xtick.color': '#8b949e',
    'ytick.color': '#8b949e',
    'grid.color': '#21262d',
    'grid.alpha': 0.6,
    'font.family': 'sans-serif',
    'font.size': 11,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'savefig.facecolor': '#0d1117',
    'savefig.edgecolor': 'none',
})

# Color palette — GitHub-dark inspired
C_ACCENT = '#58a6ff'
C_GREEN  = '#3fb950'
C_RED    = '#f85149'
C_ORANGE = '#d29922'
C_PURPLE = '#bc8cff'
C_CYAN   = '#39d353'
C_PINK   = '#f778ba'
C_GRID   = '#21262d'

OUTPUT_DIR = 'output'


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1: RUN THE BACKTEST
# ═════════════════════════════════════════════════════════════════════════════

def run_backtest(index_name, is_weekly=False, regime_method='learned_hmm'):
    """
    Run the full CatBoost + HMM strategy and return all intermediate data
    needed for validation analysis.
    """
    freq_label = "WEEKLY" if is_weekly else "MONTHLY"
    print(f"\n{'='*80}")
    print(f"  RUNNING BACKTEST: {index_name} ({freq_label}) — regime={regime_method}")
    print(f"{'='*80}")

    # ── Determine universe type ──────────────────────────────────────────
    pit_indices = {'nifty50': [HISTORICAL_COMPOSITION_CSV],
                   'nifty100': [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]}

    if index_name in pit_indices:
        csv_paths = pit_indices[index_name]
        monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
        daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)

        if is_weekly:
            prices = daily_prices.resample('W-FRI').last().dropna(how='all')
            mask = mask.resample('W-FRI').ffill().reindex(prices.index).ffill().fillna(False)
            periods_per_year = 52
            lookbacks = [1, 4, 12, 24, 52]
            min_train = 156
        else:
            prices = monthly_prices
            periods_per_year = 12
            lookbacks = LOOKBACK_WINDOWS
            min_train = 60
    else:
        # Static universe (nifty250 / nifty500)
        if not os.path.exists("daily_cache_nifty500.csv"):
            print("ERROR: Missing daily_cache_nifty500.csv — run prepare_nifty500.py first.")
            sys.exit(1)

        daily_full = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
        if index_name == 'nifty250':
            top250 = daily_full.notna().sum().nlargest(250).index.tolist()
            daily_prices = daily_full[top250]
        else:
            daily_prices = daily_full

        prices = daily_prices.resample('W-FRI' if is_weekly else 'ME').last().dropna(how='all')
        periods_per_year = 52 if is_weekly else 12
        lookbacks = [1, 4, 12, 24, 52] if is_weekly else [1, 3, 6, 12]
        min_train = 156 if is_weekly else 36
        mask = prices.notna()

    # ── Features + Walk-Forward ──────────────────────────────────────────
    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)

    if res_df is None:
        print("ERROR: Not enough data for walk-forward.")
        sys.exit(1)

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    # ── Regime Detection ─────────────────────────────────────────────────
    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)

    # ── Portfolio Simulation ─────────────────────────────────────────────
    port_returns, holdings_counts, _ = simulate_portfolio(res_df, regimes, daily_prices)
    stats = performance_stats(port_returns, periods_per_year)

    freq_l = "wk" if is_weekly else "mo"
    print_stats(stats, f"{index_name.upper()} {freq_label} — LONG ONLY", holdings_counts, freq_label=freq_l)

    return {
        'port_returns': port_returns,
        'holdings_counts': holdings_counts,
        'stats': stats,
        'res_df': res_df,
        'regimes': regimes,
        'daily_prices': daily_prices,
        'periods_per_year': periods_per_year,
        'index_name': index_name,
        'is_weekly': is_weekly,
        'freq_label': freq_label,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2: BACKTEST VISUALIZATIONS
# ═════════════════════════════════════════════════════════════════════════════

def plot_equity_curve(port_returns, stats, label):
    """Equity curve with drawdown overlay."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.05})

    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    drawdown = (cum - running_max) / running_max * 100

    # Equity curve
    ax1.fill_between(cum.index, 1, cum.values, alpha=0.15, color=C_ACCENT)
    ax1.plot(cum.index, cum.values, color=C_ACCENT, linewidth=2, label='Strategy')
    ax1.axhline(y=1, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.set_ylabel('Cumulative Return (×)', fontsize=12)
    ax1.set_title(f'{label} — Equity Curve\n'
                  f'CAGR: {stats["ann"]:.1f}%  |  Sharpe: {stats["sharpe"]:.2f}  |  '
                  f'Max DD: {stats["dd"]:.1f}%  |  Calmar: {stats["calmar"]:.2f}',
                  fontsize=13, fontweight='bold', pad=12)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Drawdown
    ax2.fill_between(drawdown.index, 0, drawdown.values, alpha=0.4, color=C_RED)
    ax2.plot(drawdown.index, drawdown.values, color=C_RED, linewidth=1.2)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    fig.savefig(os.path.join(OUTPUT_DIR, 'equity_curve.png'))
    plt.close(fig)
    print("  [OK] equity_curve.png")


def plot_monthly_heatmap(port_returns, label):
    """Year × Month returns heatmap."""
    df = port_returns.copy()
    df.index = pd.to_datetime(df.index)
    monthly = df.groupby([df.index.year, df.index.month]).sum() * 100
    monthly.index.names = ['Year', 'Month']
    pivot = monthly.unstack(level='Month')
    pivot.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'][:len(pivot.columns)]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8)))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()))
    sns.heatmap(pivot, annot=True, fmt='.1f', center=0, cmap='RdYlGn',
                vmin=-vmax, vmax=vmax, linewidths=0.5, linecolor='#30363d',
                ax=ax, cbar_kws={'label': 'Return (%)'})

    ax.set_title(f'{label} — Monthly Returns Heatmap (%)', fontsize=13, fontweight='bold', pad=12)
    ax.set_ylabel('Year', fontsize=12)
    ax.set_xlabel('')

    fig.savefig(os.path.join(OUTPUT_DIR, 'monthly_returns_heatmap.png'))
    plt.close(fig)
    print("  [OK] monthly_returns_heatmap.png")


def plot_rolling_sharpe(port_returns, periods_per_year, label):
    """Rolling 6- and 12-period Sharpe ratio."""
    rf_period = (1 + RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0

    fig, ax = plt.subplots(figsize=(14, 5))

    for window, color, lbl in [(6, C_ACCENT, '6-period'), (12, C_PURPLE, '12-period')]:
        if len(port_returns) < window:
            continue
        excess = port_returns - rf_period
        rolling_mean = excess.rolling(window).mean()
        rolling_std = port_returns.rolling(window).std()
        rolling_sr = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)
        ax.plot(rolling_sr.index, rolling_sr.values, color=color, linewidth=1.8, label=f'{lbl} rolling')

    ax.axhline(y=0, color=C_RED, linewidth=0.8, linestyle='--', alpha=0.6)
    ax.axhline(y=1, color=C_GREEN, linewidth=0.8, linestyle='--', alpha=0.4, label='Sharpe = 1')
    ax.set_ylabel('Sharpe Ratio (annualized)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'{label} — Rolling Sharpe Ratio', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'rolling_sharpe.png'))
    plt.close(fig)
    print("  [OK] rolling_sharpe.png")


def plot_drawdown_underwater(port_returns, label):
    """Underwater drawdown chart."""
    cum = (1 + port_returns).cumprod()
    running_max = cum.cummax()
    dd = (cum - running_max) / running_max * 100

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(dd.index, 0, dd.values, alpha=0.5, color=C_RED)
    ax.plot(dd.index, dd.values, color=C_RED, linewidth=1)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_title(f'{label} — Underwater Drawdown Chart', fontsize=13, fontweight='bold', pad=12)
    ax.grid(True, alpha=0.3)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f%%'))

    fig.savefig(os.path.join(OUTPUT_DIR, 'drawdown_underwater.png'))
    plt.close(fig)
    print("  [OK] drawdown_underwater.png")


def plot_holdings_distribution(holdings_counts, label):
    """Histogram of portfolio sizes over time."""
    fig, ax = plt.subplots(figsize=(10, 5))

    counts, bins, patches = ax.hist(holdings_counts, bins=range(0, max(holdings_counts) + 2),
                                     align='left', color=C_ACCENT, alpha=0.75, edgecolor='#30363d')
    ax.axvline(np.mean(holdings_counts), color=C_ORANGE, linewidth=2, linestyle='--',
               label=f'Mean: {np.mean(holdings_counts):.1f}')
    ax.set_xlabel('Number of Holdings', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title(f'{label} — Holdings Distribution', fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    fig.savefig(os.path.join(OUTPUT_DIR, 'holdings_distribution.png'))
    plt.close(fig)
    print("  [OK] holdings_distribution.png")


def plot_return_distribution(port_returns, label):
    """Histogram + KDE of period returns with normal overlay."""
    fig, ax = plt.subplots(figsize=(10, 5))

    sns.histplot(port_returns * 100, kde=True, color=C_ACCENT, alpha=0.6, ax=ax,
                 edgecolor='#30363d', stat='density', label='Actual')

    # Normal overlay
    x = np.linspace(port_returns.min() * 100, port_returns.max() * 100, 200)
    mu, sigma = port_returns.mean() * 100, port_returns.std() * 100
    normal = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal, color=C_ORANGE, linewidth=2, linestyle='--', label='Normal fit')

    ax.axvline(0, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Return (%)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title(f'{label} — Return Distribution\n'
                 f'Mean: {mu:.2f}%  |  Std: {sigma:.2f}%  |  '
                 f'Skew: {port_returns.skew():.2f}  |  Kurt: {port_returns.kurtosis():.2f}',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'return_distribution.png'))
    plt.close(fig)
    print("  [OK] return_distribution.png")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3: VALIDATION TESTS
# ═════════════════════════════════════════════════════════════════════════════

def monte_carlo_bootstrap(port_returns, periods_per_year, n_simulations=10000):
    """
    Bootstrap resample period returns to build confidence intervals
    for Sharpe, CAGR, and Max Drawdown.
    """
    print("\n  ── Monte Carlo Bootstrap (n=10,000) ──")
    returns = port_returns.values
    n = len(returns)

    sharpes, cagrs, max_dds = [], [], []
    rng = np.random.RandomState(42)

    for _ in range(n_simulations):
        sample = rng.choice(returns, size=n, replace=True)
        cum = np.cumprod(1 + sample)
        total_ret = cum[-1] - 1
        years = n / periods_per_year
        cagr = (1 + total_ret) ** (1 / years) - 1

        rf_period = (1 + RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0
        excess = sample.mean() - rf_period
        std = sample.std()
        sharpe = (excess / std * np.sqrt(periods_per_year)) if std > 0 else 0

        running_max = np.maximum.accumulate(cum)
        dd = ((cum - running_max) / running_max).min()

        sharpes.append(sharpe)
        cagrs.append(cagr * 100)
        max_dds.append(dd * 100)

    sharpes = np.array(sharpes)
    cagrs = np.array(cagrs)
    max_dds = np.array(max_dds)

    # Print results
    for name, arr in [('Sharpe', sharpes), ('CAGR (%)', cagrs), ('Max DD (%)', max_dds)]:
        p5, p50, p95 = np.percentile(arr, [5, 50, 95])
        print(f"    {name:12s}:  median={p50:+.2f}  |  90% CI=[{p5:+.2f}, {p95:+.2f}]")

    return sharpes, cagrs, max_dds


def plot_mc_bootstrap(sharpes, cagrs, max_dds, actual_stats, label):
    """Plot Monte Carlo bootstrap distributions."""

    for data, name, actual_val, fname, color in [
        (sharpes, 'Sharpe Ratio', actual_stats['sharpe'], 'mc_bootstrap_sharpe.png', C_ACCENT),
        (cagrs, 'CAGR (%)', actual_stats['ann'], 'mc_bootstrap_cagr.png', C_GREEN),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))

        p5, p95 = np.percentile(data, [5, 95])
        ax.hist(data, bins=80, color=color, alpha=0.6, edgecolor='#30363d', density=True)
        ax.axvline(actual_val, color=C_ORANGE, linewidth=2.5, linestyle='-', label=f'Actual: {actual_val:.2f}')
        ax.axvline(p5, color=C_RED, linewidth=1.5, linestyle='--', label=f'5th pct: {p5:.2f}')
        ax.axvline(p95, color=C_GREEN if color != C_GREEN else C_CYAN, linewidth=1.5,
                   linestyle='--', label=f'95th pct: {p95:.2f}')

        ax.set_xlabel(name, fontsize=12)
        ax.set_ylabel('Density', fontsize=12)
        ax.set_title(f'{label} — Monte Carlo Bootstrap: {name}\n'
                     f'10,000 resamples  |  90% CI: [{p5:.2f}, {p95:.2f}]',
                     fontsize=13, fontweight='bold', pad=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        fig.savefig(os.path.join(OUTPUT_DIR, fname))
        plt.close(fig)
        print(f"  [OK] {fname}")


def monte_carlo_path_simulation(port_returns, n_paths=1000):
    """
    Randomly shuffle the order of returns to generate alternative equity paths.
    Tests whether the strategy's performance is path-dependent.
    """
    print("\n  ── Monte Carlo Path Simulation (n=1,000) ──")
    returns = port_returns.values
    n = len(returns)
    rng = np.random.RandomState(42)

    paths = np.zeros((n_paths, n))
    for i in range(n_paths):
        shuffled = rng.permutation(returns)
        paths[i] = np.cumprod(1 + shuffled)

    actual_path = np.cumprod(1 + returns)

    # Stats
    final_vals = paths[:, -1]
    p5, p50, p95 = np.percentile(final_vals, [5, 50, 95])
    print(f"    Final value — median: {p50:.2f}x  |  90% CI: [{p5:.2f}x, {p95:.2f}x]")
    print(f"    Actual final value:   {actual_path[-1]:.2f}x")

    return paths, actual_path


def plot_mc_paths(paths, actual_path, port_returns, label):
    """Plot Monte Carlo path simulation fanout."""
    fig, ax = plt.subplots(figsize=(14, 7))

    n = paths.shape[1]
    x = np.arange(n)

    # Plot random sample of paths
    sample_idx = np.random.RandomState(0).choice(len(paths), size=min(200, len(paths)), replace=False)
    for i in sample_idx:
        ax.plot(x, paths[i], color=C_ACCENT, alpha=0.04, linewidth=0.5)

    # Percentile bands
    p5 = np.percentile(paths, 5, axis=0)
    p25 = np.percentile(paths, 25, axis=0)
    p75 = np.percentile(paths, 75, axis=0)
    p95 = np.percentile(paths, 95, axis=0)
    median = np.median(paths, axis=0)

    ax.fill_between(x, p5, p95, alpha=0.15, color=C_ACCENT, label='5th–95th pct')
    ax.fill_between(x, p25, p75, alpha=0.25, color=C_ACCENT, label='25th–75th pct')
    ax.plot(x, median, color=C_PURPLE, linewidth=1.5, linestyle='--', label='Median path')
    ax.plot(x, actual_path, color=C_ORANGE, linewidth=2.5, label='Actual path')

    ax.axhline(y=1, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_xlabel('Period', fontsize=12)
    ax.set_ylabel('Cumulative Return (×)', fontsize=12)
    ax.set_title(f'{label} — Monte Carlo Path Simulation (1,000 shuffled paths)\n'
                 f'Actual: {actual_path[-1]:.2f}×  |  Median: {median[-1]:.2f}×  |  '
                 f'90% CI: [{p5[-1]:.2f}×, {p95[-1]:.2f}×]',
                 fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc='upper left', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'mc_path_simulation.png'))
    plt.close(fig)
    print("  [OK] mc_path_simulation.png")


def sensitivity_threshold(res_df, regimes, daily_prices, periods_per_year, label):
    """
    Sweep probability threshold from 0.50 to 0.70 and measure Sharpe & CAGR.
    """
    print("\n  ── Parameter Sensitivity: Probability Threshold ──")
    import engine as eng

    thresholds = np.arange(0.50, 0.725, 0.025)
    results = []

    original_thresh = eng.PROB_THRESHOLD
    for thresh in thresholds:
        eng.PROB_THRESHOLD = thresh
        port, counts, _ = simulate_portfolio(res_df, regimes, daily_prices)
        s = performance_stats(port, periods_per_year)
        results.append({'threshold': thresh, 'sharpe': s['sharpe'], 'cagr': s['ann'],
                        'max_dd': s['dd'], 'win_rate': s['win']})
        print(f"    threshold={thresh:.3f}  ->  Sharpe={s['sharpe']:.3f}  CAGR={s['ann']:.1f}%")
    eng.PROB_THRESHOLD = original_thresh

    results_df = pd.DataFrame(results)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(results_df['threshold'], results_df['sharpe'], color=C_ACCENT, linewidth=2.5,
             marker='o', markersize=8, label='Sharpe', zorder=3)
    ax2.plot(results_df['threshold'], results_df['cagr'], color=C_GREEN, linewidth=2.5,
             marker='s', markersize=8, label='CAGR (%)', zorder=3)

    ax1.axvline(original_thresh, color=C_ORANGE, linewidth=1.5, linestyle='--',
                label=f'Default ({original_thresh})', alpha=0.8)

    ax1.set_xlabel('Probability Threshold', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, color=C_ACCENT)
    ax2.set_ylabel('CAGR (%)', fontsize=12, color=C_GREEN)
    ax1.set_title(f'{label} — Sensitivity: Probability Threshold',
                  fontsize=13, fontweight='bold', pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_threshold.png'))
    plt.close(fig)
    print("  [OK] sensitivity_threshold.png")

    return results_df


def sensitivity_stoploss(res_df, regimes, daily_prices, periods_per_year, label):
    """
    Sweep stop-loss levels from -3% to -15% and measure CAGR + Max DD.
    """
    print("\n  ── Parameter Sensitivity: Stop-Loss Level ──")
    import engine as eng

    stop_levels = np.arange(-0.03, -0.16, -0.01)
    results = []

    original_stops = dict(eng.REGIME_STOP)

    for stop in stop_levels:
        # Apply uniform stop across all regimes
        for regime in eng.REGIME_STOP:
            eng.REGIME_STOP[regime] = stop
        port, counts, _ = simulate_portfolio(res_df, regimes, daily_prices)
        s = performance_stats(port, periods_per_year)
        results.append({'stop_loss': stop * 100, 'sharpe': s['sharpe'], 'cagr': s['ann'],
                        'max_dd': s['dd'], 'calmar': s['calmar']})
        print(f"    stop={stop*100:.0f}%  ->  Sharpe={s['sharpe']:.3f}  CAGR={s['ann']:.1f}%  MaxDD={s['dd']:.1f}%")

    # Restore
    for regime in original_stops:
        eng.REGIME_STOP[regime] = original_stops[regime]

    results_df = pd.DataFrame(results)

    # Plot
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    ax1.plot(results_df['stop_loss'], results_df['cagr'], color=C_GREEN, linewidth=2.5,
             marker='o', markersize=8, label='CAGR (%)')
    ax2.plot(results_df['stop_loss'], results_df['max_dd'], color=C_RED, linewidth=2.5,
             marker='s', markersize=8, label='Max DD (%)')

    # Mark default stops
    for regime, stop_val in original_stops.items():
        ax1.axvline(stop_val * 100, color=C_ORANGE, linewidth=1, linestyle='--', alpha=0.5)

    ax1.set_xlabel('Stop-Loss Level (%)', fontsize=12)
    ax1.set_ylabel('CAGR (%)', fontsize=12, color=C_GREEN)
    ax2.set_ylabel('Max Drawdown (%)', fontsize=12, color=C_RED)
    ax1.set_title(f'{label} — Sensitivity: Stop-Loss Level',
                  fontsize=13, fontweight='bold', pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_stoploss.png'))
    plt.close(fig)
    print("  [OK] sensitivity_stoploss.png")

    return results_df


def sensitivity_regime_sizing(res_df, regimes, daily_prices, periods_per_year, label):
    """
    Sweep LowVol/MedVol sizing combos and measure Sharpe in a heatmap.
    """
    print("\n  ── Parameter Sensitivity: Regime Sizing ──")
    import engine as eng

    bull_sizes = [5, 8, 10, 12, 15]
    neutral_sizes = [2, 3, 4, 5, 6]

    original_sizes = dict(eng.REGIME_SIZE)
    heatmap_data = np.zeros((len(neutral_sizes), len(bull_sizes)))

    for i, n_size in enumerate(neutral_sizes):
        for j, b_size in enumerate(bull_sizes):
            eng.REGIME_SIZE['LowVol'] = b_size
            eng.REGIME_SIZE['MedVol'] = n_size
            eng.REGIME_SIZE['HighVol'] = max(2, n_size - 1)

            port, counts, _ = simulate_portfolio(res_df, regimes, daily_prices)
            s = performance_stats(port, periods_per_year)
            heatmap_data[i, j] = s['sharpe']
            print(f"    LowVol={b_size:2d}  MedVol={n_size}  HighVol={max(2, n_size-1)}  ->  Sharpe={s['sharpe']:.3f}")

    # Restore
    for regime in original_sizes:
        eng.REGIME_SIZE[regime] = original_sizes[regime]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 7))

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[str(x) for x in bull_sizes],
                yticklabels=[str(x) for x in neutral_sizes],
                linewidths=1, linecolor='#30363d', ax=ax,
                cbar_kws={'label': 'Sharpe Ratio'})

    ax.set_xlabel('LowVol Regime — # Holdings', fontsize=12)
    ax.set_ylabel('MedVol Regime — # Holdings', fontsize=12)
    ax.set_title(f'{label} — Sensitivity: Regime-Based Position Sizing\n'
                 f'(HighVol = MedVol - 1, min 2)',
                 fontsize=13, fontweight='bold', pad=12)

    # Mark default
    default_b = bull_sizes.index(original_sizes['LowVol']) if original_sizes.get('LowVol', -1) in bull_sizes else -1
    default_n = neutral_sizes.index(original_sizes['MedVol']) if original_sizes.get('MedVol', -1) in neutral_sizes else -1
    if default_b >= 0 and default_n >= 0:
        ax.add_patch(plt.Rectangle((default_b, default_n), 1, 1,
                                    fill=False, edgecolor=C_ORANGE, linewidth=3))

    fig.savefig(os.path.join(OUTPUT_DIR, 'sensitivity_regime_heatmap.png'))
    plt.close(fig)
    print("  [OK] sensitivity_regime_heatmap.png")


def plot_regime_performance(port_returns, regimes, label):
    """Performance decomposition by regime."""
    print("\n  ── Regime Performance Decomposition ──")

    regime_series = pd.Series(regimes)
    regime_aligned = regime_series.reindex(port_returns.index).ffill().fillna('MedVol')

    regime_stats = {}
    for regime in ['LowVol', 'MedVol', 'HighVol']:
        mask = regime_aligned == regime
        if mask.sum() == 0:
            continue
        ret = port_returns[mask]
        regime_stats[regime] = {
            'count': int(mask.sum()),
            'mean_return': ret.mean() * 100,
            'std_return': ret.std() * 100,
            'win_rate': (ret > 0).mean() * 100,
            'total_return': ((1 + ret).prod() - 1) * 100,
        }
        print(f"    {regime:8s}:  n={mask.sum():3d}  mean={ret.mean()*100:+.2f}%  "
              f"std={ret.std()*100:.2f}%  win={((ret>0).mean()*100):.0f}%  "
              f"total={((1+ret).prod()-1)*100:.1f}%")

    if not regime_stats:
        print("    No regime data to plot.")
        return

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    regime_colors = {'LowVol': C_GREEN, 'MedVol': C_ACCENT, 'HighVol': C_RED}
    regimes_present = list(regime_stats.keys())

    # Mean return
    ax = axes[0]
    vals = [regime_stats[r]['mean_return'] for r in regimes_present]
    colors = [regime_colors.get(r, C_ACCENT) for r in regimes_present]
    bars = ax.bar(regimes_present, vals, color=colors, alpha=0.8, edgecolor='#30363d')
    ax.axhline(0, color='#8b949e', linewidth=0.8, linestyle='--')
    ax.set_ylabel('Mean Return (%)', fontsize=11)
    ax.set_title('Mean Period Return', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:+.2f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Win rate
    ax = axes[1]
    vals = [regime_stats[r]['win_rate'] for r in regimes_present]
    bars = ax.bar(regimes_present, vals, color=colors, alpha=0.8, edgecolor='#30363d')
    ax.axhline(50, color=C_ORANGE, linewidth=1, linestyle='--', alpha=0.6, label='50%')
    ax.set_ylabel('Win Rate (%)', fontsize=11)
    ax.set_title('Win Rate', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=9)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val:.0f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Period count
    ax = axes[2]
    vals = [regime_stats[r]['count'] for r in regimes_present]
    bars = ax.bar(regimes_present, vals, color=colors, alpha=0.8, edgecolor='#30363d')
    ax.set_ylabel('# Periods', fontsize=11)
    ax.set_title('Observation Count', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    fig.suptitle(f'{label} — Performance by Market Regime',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'regime_performance.png'))
    plt.close(fig)
    print("  [OK] regime_performance.png")


def plot_walkforward_stability(port_returns, periods_per_year, label):
    """Rolling Sharpe + Calmar over the OOS window."""
    print("\n  ── Walk-Forward Stability ──")

    rf_period = (1 + RISK_FREE_ANNUAL) ** (1.0 / periods_per_year) - 1.0
    window = max(6, min(12, len(port_returns) // 3))

    fig, ax1 = plt.subplots(figsize=(14, 6))
    ax2 = ax1.twinx()

    # Rolling Sharpe
    excess = port_returns - rf_period
    rolling_sharpe = (excess.rolling(window).mean() / port_returns.rolling(window).std()) * np.sqrt(periods_per_year)

    # Rolling Calmar
    cum = (1 + port_returns).cumprod()
    rolling_cagr = cum.pct_change(window).rolling(1).mean() * periods_per_year / window
    rolling_dd = pd.Series(index=port_returns.index, dtype=float)
    for i in range(window, len(port_returns)):
        segment = cum.iloc[i-window:i+1]
        dd = ((segment - segment.cummax()) / segment.cummax()).min()
        rolling_dd.iloc[i] = abs(dd) if dd != 0 else 0.001

    rolling_calmar = rolling_cagr / rolling_dd

    ax1.plot(rolling_sharpe.index, rolling_sharpe.values, color=C_ACCENT, linewidth=2, label='Rolling Sharpe')
    ax1.axhline(0, color=C_RED, linewidth=0.8, linestyle='--', alpha=0.5)
    ax1.axhline(1, color=C_GREEN, linewidth=0.8, linestyle='--', alpha=0.4)

    ax2.plot(rolling_calmar.dropna().index, rolling_calmar.dropna().values,
             color=C_PURPLE, linewidth=2, alpha=0.7, label='Rolling Calmar')

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Sharpe Ratio', fontsize=12, color=C_ACCENT)
    ax2.set_ylabel('Calmar Ratio', fontsize=12, color=C_PURPLE)
    ax1.set_title(f'{label} — Walk-Forward Stability ({window}-period rolling)',
                  fontsize=13, fontweight='bold', pad=12)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'walkforward_stability.png'))
    plt.close(fig)
    print("  [OK] walkforward_stability.png")


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4: MAIN ORCHESTRATOR
# ═════════════════════════════════════════════════════════════════════════════

def run_full_validation(index_name='nifty50', is_weekly=False, regime_method='learned_hmm'):
    """Run backtest + all validation tests + generate all visuals."""
    ensure_output_dir()

    # ── 1. Run backtest ──────────────────────────────────────────────────
    data = run_backtest(index_name, is_weekly=is_weekly, regime_method=regime_method)
    port_returns = data['port_returns']
    stats = data['stats']
    freq = data['freq_label']
    label = f"{index_name.upper()} {freq}"

    print(f"\n{'='*80}")
    print(f"  GENERATING BACKTEST VISUALS")
    print(f"{'='*80}")

    # ── 2. Backtest visuals ──────────────────────────────────────────────
    plot_equity_curve(port_returns, stats, label)
    plot_monthly_heatmap(port_returns, label)
    plot_rolling_sharpe(port_returns, data['periods_per_year'], label)
    plot_drawdown_underwater(port_returns, label)
    plot_holdings_distribution(data['holdings_counts'], label)
    plot_return_distribution(port_returns, label)

    print(f"\n{'='*80}")
    print(f"  RUNNING VALIDATION TESTS")
    print(f"{'='*80}")

    # ── 3. Monte Carlo Bootstrap ─────────────────────────────────────────
    sharpes, cagrs, max_dds = monte_carlo_bootstrap(
        port_returns, data['periods_per_year'])
    plot_mc_bootstrap(sharpes, cagrs, max_dds, stats, label)

    # ── 4. Monte Carlo Path Simulation ───────────────────────────────────
    paths, actual_path = monte_carlo_path_simulation(port_returns)
    plot_mc_paths(paths, actual_path, port_returns, label)

    # ── 5. Parameter Sensitivity ─────────────────────────────────────────
    sensitivity_threshold(data['res_df'], data['regimes'], data['daily_prices'],
                          data['periods_per_year'], label)
    sensitivity_stoploss(data['res_df'], data['regimes'], data['daily_prices'],
                         data['periods_per_year'], label)
    sensitivity_regime_sizing(data['res_df'], data['regimes'], data['daily_prices'],
                              data['periods_per_year'], label)

    # ── 6. Regime Performance ────────────────────────────────────────────
    plot_regime_performance(port_returns, data['regimes'], label)

    # ── 7. Walk-Forward Stability ────────────────────────────────────────
    plot_walkforward_stability(port_returns, data['periods_per_year'], label)

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  VALIDATION COMPLETE -- {len(os.listdir(OUTPUT_DIR))} charts saved to {OUTPUT_DIR}/")
    print(f"{'='*80}")

    # Print summary table
    print(f"\n  {'Metric':<25s} {'Value':>12s}  {'MC 90% CI':>20s}")
    print(f"  {'-'*25} {'-'*12}  {'-'*20}")

    p5_s, p95_s = np.percentile(sharpes, [5, 95])
    p5_c, p95_c = np.percentile(cagrs, [5, 95])
    p5_d, p95_d = np.percentile(max_dds, [5, 95])

    print(f"  {'Sharpe Ratio':<25s} {stats['sharpe']:>12.3f}  [{p5_s:>+8.3f}, {p95_s:>+8.3f}]")
    print(f"  {'CAGR (%)':<25s} {stats['ann']:>12.2f}  [{p5_c:>+8.2f}, {p95_c:>+8.2f}]")
    print(f"  {'Max Drawdown (%)':<25s} {stats['dd']:>12.2f}  [{p5_d:>+8.2f}, {p95_d:>+8.2f}]")
    print(f"  {'Volatility (%)':<25s} {stats['vol']:>12.2f}")
    print(f"  {'Calmar Ratio':<25s} {stats['calmar']:>12.3f}")
    print(f"  {'Win Rate (%)':<25s} {stats['win']:>12.2f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='WSC_MS Strategy Validation Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python validate_strategy.py --index nifty50
  python validate_strategy.py --index nifty100 --weekly
  python validate_strategy.py --index nifty500 --regime learned_hmm
        """
    )
    parser.add_argument('--index', type=str, default='nifty50',
                        choices=['nifty50', 'nifty100', 'nifty250', 'nifty500'],
                        help='Index universe (default: nifty50)')
    parser.add_argument('--weekly', action='store_true',
                        help='Use weekly frequency (default: monthly)')
    parser.add_argument('--regime', type=str, default='learned_hmm',
                        choices=['fixed_hmm', 'learned_hmm', 'none'],
                        help='Regime method (default: learned_hmm)')

    args = parser.parse_args()
    run_full_validation(args.index, is_weekly=args.weekly, regime_method=args.regime)
