"""
Weighting Method Comparison
============================
Runs the strategy with all 4 weighting methods on the same backtest data,
then generates a head-to-head comparison chart + summary table.

Usage:
  python compare_weights.py --index nifty100
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

warnings.filterwarnings('ignore')

# Fix Windows console encoding
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(errors='replace')
    sys.stderr.reconfigure(errors='replace')

from config import (
    DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV,
    NIFTY_NEXT_50_COMPOSITION_CSV, LOOKBACK_WINDOWS, RISK_FREE_ANNUAL
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, performance_stats
)
from regime import get_regimes

# Aesthetics (GitHub-dark)
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

METHODS = {
    'equal':       {'label': 'Equal Weight',           'color': '#8b949e'},
    'probability': {'label': 'Probability-Weighted',   'color': '#58a6ff'},
    'inverse_vol': {'label': 'Inverse Volatility',     'color': '#3fb950'},
    'prob_invvol':  {'label': 'Prob x Inv-Vol (Hybrid)','color': '#bc8cff'},
    'kelly':       {'label': 'Half-Kelly',             'color': '#f78166'},
}

OUTPUT_DIR = 'output'


def prepare_backtest_data(index_name, regime_method='learned_hmm'):
    """Run the walk-forward classifier and regime detection once, return shared data."""
    print(f"\n{'='*80}")
    print(f"  PREPARING BACKTEST DATA: {index_name.upper()} MONTHLY")
    print(f"{'='*80}")

    pit_indices = {
        'nifty50': [HISTORICAL_COMPOSITION_CSV],
        'nifty100': [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV]
    }

    if index_name in pit_indices:
        csv_paths = pit_indices[index_name]
        monthly_prices, mask = fetch_monthly_prices(csv_paths, DATA_START, DATA_END)
        daily_prices = fetch_daily_prices(mask.columns.tolist(), DATA_START, DATA_END)
        prices = monthly_prices
        periods_per_year = 12
        lookbacks = LOOKBACK_WINDOWS
        min_train = 60
    else:
        if not os.path.exists("daily_cache_nifty500.csv"):
            print("ERROR: Missing daily_cache_nifty500.csv")
            sys.exit(1)
        daily_full = pd.read_csv("daily_cache_nifty500.csv", index_col=0, parse_dates=True)
        if index_name == 'nifty250':
            top250 = daily_full.notna().sum().nlargest(250).index.tolist()
            daily_prices = daily_full[top250]
        else:
            daily_prices = daily_full
        prices = daily_prices.resample('ME').last().dropna(how='all')
        periods_per_year = 12
        lookbacks = [1, 3, 6, 12]
        min_train = 36
        mask = prices.notna()

    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)

    if res_df is None:
        print("ERROR: Not enough data for walk-forward.")
        sys.exit(1)

    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)

    return res_df, regimes, daily_prices, periods_per_year


def run_comparison(index_name='nifty100', regime_method='learned_hmm'):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Prepare data once (expensive part)
    res_df, regimes, daily_prices, periods_per_year = prepare_backtest_data(
        index_name, regime_method)

    # Run each weighting method
    results = {}
    print(f"\n{'='*80}")
    print(f"  RUNNING 5 WEIGHTING METHODS")
    print(f"{'='*80}")

    for method_key, meta in METHODS.items():
        print(f"\n  [{meta['label']}] ...", end='', flush=True)
        port_returns, counts, _ = simulate_portfolio(
            res_df, regimes, daily_prices, weighting=method_key)
        stats = performance_stats(port_returns, periods_per_year)
        results[method_key] = {
            'returns': port_returns,
            'counts': counts,
            'stats': stats,
            'label': meta['label'],
            'color': meta['color'],
        }
        print(f"  Sharpe={stats['sharpe']:.3f}  CAGR={stats['ann']:.1f}%  "
              f"MaxDD={stats['dd']:.1f}%  Calmar={stats['calmar']:.2f}")

    # ── Print comparison table ────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  WEIGHTING METHOD COMPARISON -- {index_name.upper()} MONTHLY")
    print(f"{'='*80}")

    header = f"  {'Method':<25s} {'CAGR':>7s} {'Sharpe':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'Vol':>7s} {'WinRate':>8s}"
    print(header)
    print(f"  {'-'*25} {'-'*7} {'-'*8} {'-'*8} {'-'*8} {'-'*7} {'-'*8}")

    for method_key in METHODS:
        s = results[method_key]['stats']
        lbl = results[method_key]['label']
        print(f"  {lbl:<25s} {s['ann']:>6.1f}% {s['sharpe']:>8.3f} {s['dd']:>7.1f}% "
              f"{s['calmar']:>8.2f} {s['vol']:>6.1f}% {s['win']:>7.1f}%")

    # ── Find best method per metric ───────────────────────────────────────
    print(f"\n  Best by metric:")
    metrics_best = {
        'Sharpe': max(results, key=lambda k: results[k]['stats']['sharpe']),
        'CAGR': max(results, key=lambda k: results[k]['stats']['ann']),
        'Max DD': max(results, key=lambda k: results[k]['stats']['dd']),  # least negative
        'Calmar': max(results, key=lambda k: results[k]['stats']['calmar']),
        'Win Rate': max(results, key=lambda k: results[k]['stats']['win']),
    }
    for metric, best_key in metrics_best.items():
        val = results[best_key]['stats']
        metric_map = {'Sharpe': 'sharpe', 'CAGR': 'ann', 'Max DD': 'dd', 'Calmar': 'calmar', 'Win Rate': 'win'}
        v = val[metric_map[metric]]
        print(f"    {metric:<12s} -> {results[best_key]['label']} ({v:.2f})")

    # ═══════════════════════════════════════════════════════════════════════
    #  CHART 1: Equity curves overlay
    # ═══════════════════════════════════════════════════════════════════════
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                    sharex=True, gridspec_kw={'hspace': 0.05})

    for method_key in METHODS:
        r = results[method_key]
        cum = (1 + r['returns']).cumprod()
        ax1.plot(cum.index, cum.values, color=r['color'], linewidth=2.2,
                 label=f"{r['label']} (Sharpe={r['stats']['sharpe']:.2f})")

        dd = (cum - cum.cummax()) / cum.cummax() * 100
        ax2.plot(dd.index, dd.values, color=r['color'], linewidth=1.2, alpha=0.8)

    ax1.axhline(y=1, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.3)
    ax1.set_ylabel('Cumulative Return (x)', fontsize=13)
    ax1.set_title(f'{index_name.upper()} Monthly -- Weighting Method Comparison',
                  fontsize=15, fontweight='bold', pad=14)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.8)
    ax1.grid(True, alpha=0.3)

    ax2.axhline(y=0, color='#8b949e', linewidth=0.8, linestyle='--', alpha=0.3)
    ax2.set_ylabel('Drawdown (%)', fontsize=13)
    ax2.set_xlabel('Date', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f%%'))

    fig.savefig(os.path.join(OUTPUT_DIR, 'weight_comparison_equity.png'))
    plt.close(fig)
    print(f"\n  [OK] weight_comparison_equity.png")

    # ═══════════════════════════════════════════════════════════════════════
    #  CHART 2: Bar chart comparison of key metrics
    # ═══════════════════════════════════════════════════════════════════════
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    metrics_to_plot = [
        ('sharpe', 'Sharpe Ratio', axes[0, 0]),
        ('ann', 'CAGR (%)', axes[0, 1]),
        ('dd', 'Max Drawdown (%)', axes[0, 2]),
        ('calmar', 'Calmar Ratio', axes[1, 0]),
        ('vol', 'Volatility (%)', axes[1, 1]),
        ('win', 'Win Rate (%)', axes[1, 2]),
    ]

    labels = [METHODS[k]['label'] for k in METHODS]
    colors = [METHODS[k]['color'] for k in METHODS]
    x_pos = np.arange(len(METHODS))

    for metric_key, title, ax in metrics_to_plot:
        vals = [results[k]['stats'][metric_key] for k in METHODS]
        bars = ax.bar(x_pos, vals, color=colors, alpha=0.85, edgecolor='#30363d', width=0.6)

        # Annotate bars
        for bar, val in zip(bars, vals):
            fmt = f'{val:.2f}' if abs(val) < 10 else f'{val:.1f}'
            y_pos = bar.get_height()
            va = 'bottom' if y_pos >= 0 else 'top'
            ax.text(bar.get_x() + bar.get_width() / 2., y_pos,
                    fmt, ha='center', va=va, fontsize=9, fontweight='bold')

        ax.set_xticks(x_pos)
        ax.set_xticklabels([l.replace(' ', '\n') for l in labels], fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        # Highlight best
        if metric_key == 'dd':
            best_idx = np.argmax(vals)  # least negative
        elif metric_key == 'vol':
            best_idx = np.argmin(vals)  # lowest vol
        else:
            best_idx = np.argmax(vals)

        bars[best_idx].set_edgecolor('#d29922')
        bars[best_idx].set_linewidth(3)

    fig.suptitle(f'{index_name.upper()} Monthly -- Metric Comparison by Weighting Method',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, 'weight_comparison_metrics.png'))
    plt.close(fig)
    print(f"  [OK] weight_comparison_metrics.png")

    # ═══════════════════════════════════════════════════════════════════════
    #  CHART 3: Risk-return scatter plot
    # ═══════════════════════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(10, 8))

    for method_key in METHODS:
        r = results[method_key]
        s = r['stats']
        ax.scatter(s['vol'], s['ann'], color=r['color'], s=200, zorder=5,
                   edgecolors='white', linewidth=1.5)
        ax.annotate(r['label'], (s['vol'], s['ann']),
                    xytext=(8, 8), textcoords='offset points', fontsize=10,
                    color=r['color'], fontweight='bold')

    # Add Sharpe reference lines
    vol_range = np.linspace(0, max(results[k]['stats']['vol'] for k in METHODS) * 1.3, 100)
    rf = RISK_FREE_ANNUAL * 100
    for sr, alpha in [(0.5, 0.2), (1.0, 0.3), (1.5, 0.2)]:
        ax.plot(vol_range, rf + sr * vol_range, '--', color='#8b949e', alpha=alpha,
                linewidth=1, label=f'Sharpe={sr}' if sr == 1.0 else None)
        ax.text(vol_range[-1], rf + sr * vol_range[-1], f'SR={sr}',
                fontsize=8, color='#8b949e', alpha=0.6)

    ax.set_xlabel('Annualized Volatility (%)', fontsize=13)
    ax.set_ylabel('CAGR (%)', fontsize=13)
    ax.set_title(f'{index_name.upper()} Monthly -- Risk-Return Profile by Weighting Method',
                 fontsize=14, fontweight='bold', pad=14)
    ax.grid(True, alpha=0.3)

    fig.savefig(os.path.join(OUTPUT_DIR, 'weight_comparison_riskreturn.png'))
    plt.close(fig)
    print(f"  [OK] weight_comparison_riskreturn.png")

    print(f"\n{'='*80}")
    print(f"  COMPARISON COMPLETE -- 3 charts saved to {OUTPUT_DIR}/")
    print(f"{'='*80}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare Weighting Methods')
    parser.add_argument('--index', type=str, default='nifty100',
                        choices=['nifty50', 'nifty100', 'nifty250', 'nifty500'])
    parser.add_argument('--regime', type=str, default='learned_hmm',
                        choices=['fixed_hmm', 'learned_hmm', 'none'])
    args = parser.parse_args()
    run_comparison(args.index, args.regime)
