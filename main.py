import os
import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score
import logging

warnings.filterwarnings('ignore')

from config import (
    DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV,
    LOOKBACK_WINDOWS
)
from data_fetcher import fetch_monthly_prices, fetch_daily_prices, compute_forward_returns
from features import compute_all_momentum
from engine import (
    build_stacked_dataset, run_expanding_window,
    simulate_portfolio, compute_target_vol, performance_stats, print_stats,
    VOLSCALE_CAP, VOLSCALE_FLOOR
)
from regime import get_regimes


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _volscale_report(label, volscale_params, res_df, daily_prices,
                     periods_per_year, freq_label, lev_cost, lev_cost_label):
    """Run one volscale pass and print results including scaling-factor stats."""
    port, counts, extra = simulate_portfolio(
        res_df, {}, daily_prices,
        sizing_scheme='volscale',
        volscale_params=volscale_params,
        lev_cost=lev_cost,
    )
    stats = performance_stats(port, periods_per_year)
    print_stats(stats, label, counts, freq_label=freq_label)

    sf_arr = np.array(extra['scaling_factors'])
    pcts = np.percentile(sf_arr, [25, 50, 75])
    pct_cap   = (sf_arr >= VOLSCALE_CAP).mean() * 100
    pct_floor = (sf_arr <= VOLSCALE_FLOOR).mean() * 100
    print(f"  Scaling factor — min:{sf_arr.min():.2f}  p25:{pcts[0]:.2f}  "
          f"p50:{pcts[1]:.2f}  p75:{pcts[2]:.2f}  max:{sf_arr.max():.2f}")
    print(f"  Cap  (≥{VOLSCALE_CAP:.2f}) binding: {pct_cap:.1f}% of rebalances")
    print(f"  Floor(≤{VOLSCALE_FLOOR:.2f}) binding: {pct_floor:.1f}% of rebalances")
    print(f"  Leverage cost used: {lev_cost_label}")
    return stats


# ---------------------------------------------------------------------------
# Universe runners
# ---------------------------------------------------------------------------

def run_pit_universe(universe_name, csv_paths, is_weekly=False,
                     regime_method='learned_hmm', sizing_scheme='directional'):
    """Run point-in-time universe evaluation (Nifty 50 / Nifty 100)."""
    banner = "=" * 80
    freq_label = "WEEKLY" if is_weekly else "MONTHLY"
    print(f"\n{banner}\nCATBOOST MOMENTUM — {universe_name} (PiT) {freq_label}\n{banner}")

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

    fwd_returns = compute_forward_returns(prices)
    momentum_dict = compute_all_momentum(prices, lookbacks)
    stacked = build_stacked_dataset(prices, mask, fwd_returns, momentum_dict, lookbacks)
    res_df = run_expanding_window(stacked, min_train_months=min_train)
    if res_df is None:
        return

    acc = accuracy_score(res_df['actual'], res_df['pred_class'])
    prec = precision_score(res_df['actual'], res_df['pred_class'])
    print(f"  Classifier accuracy: {acc:.3f}  |  precision: {prec:.3f}")

    rebal_dates = sorted(res_df['date'].unique())
    padding_start = (pd.to_datetime(rebal_dates[0]) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
    freq_label_short = "wk" if is_weekly else "mo"

    if sizing_scheme == 'volscale':
        # ---- Calibration pass ------------------------------------------------
        print("  [VolScale] Calibrating target vol on unscaled top-10 book...")
        target_vol, daily_rets = compute_target_vol(res_df, daily_prices)
        print(f"  [VolScale] Target vol (median 126d rolling): {target_vol:.1%}")
        if not (0.15 <= target_vol <= 0.25):
            print(f"  ERROR: target_vol {target_vol:.1%} outside [15%, 25%]. Stopping.")
            return

        volscale_params = {'target_vol': target_vol, 'daily_rets': daily_rets}

        # ---- Primary run (5% leverage cost) ----------------------------------
        stats_primary = _volscale_report(
            f"{universe_name} {freq_label} — VOLSCALE (lev_cost=5%)",
            volscale_params, res_df, daily_prices,
            periods_per_year, freq_label_short, lev_cost=0.05, lev_cost_label="5% (primary)"
        )

        # ---- Leverage cost sensitivity ---------------------------------------
        print(f"\n  {'-'*60}\n  ROBUSTNESS: Leverage cost sensitivity\n  {'-'*60}")
        stats_8 = _volscale_report(
            f"{universe_name} {freq_label} — VOLSCALE (lev_cost=8%)",
            volscale_params, res_df, daily_prices,
            periods_per_year, freq_label_short, lev_cost=0.08, lev_cost_label="8% (MTF rate)"
        )
        stats_3 = _volscale_report(
            f"{universe_name} {freq_label} — VOLSCALE (lev_cost=3%)",
            volscale_params, res_df, daily_prices,
            periods_per_year, freq_label_short, lev_cost=0.03, lev_cost_label="3% (index futures)"
        )

        # ---- Delta table ----------------------------------------------------
        print(f"\n  {'='*60}")
        print(f"  DELTA vs. primary (lev_cost=5%)")
        print(f"  {'='*60}")
        print(f"  {'Metric':<15} {'3% delta':>10} {'8% delta':>10}")
        print(f"  {'-'*38}")
        for metric in ('sharpe', 'calmar', 'dd', 'ann'):
            d3 = stats_3[metric] - stats_primary[metric]
            d8 = stats_8[metric] - stats_primary[metric]
            print(f"  {metric:<15} {'+' if d3>=0 else ''}{d3:>9.3f} {'+' if d8>=0 else ''}{d8:>9.3f}")

    else:
        # ---- Standard directional HMM sizing ---------------------------------
        regimes = get_regimes(rebal_dates, padding_start, DATA_END, method=regime_method)
        port, counts, _ = simulate_portfolio(res_df, regimes, daily_prices,
                                             sizing_scheme='directional')
        stats = performance_stats(port, periods_per_year)
        print_stats(stats, f"{universe_name} {freq_label} — LONG ONLY", counts,
                    freq_label=freq_label_short)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="CatBoost Momentum Strategy")
    parser.add_argument(
        '--index', type=str, default='all',
        choices=['nifty50', 'nifty100', 'all'],
        help="Universe to backtest (default: all)"
    )
    parser.add_argument(
        '--regime', type=str, default='learned_hmm',
        choices=['fixed_hmm', 'learned_hmm', 'none'],
        help="HMM regime method for directional sizing (default: learned_hmm)"
    )
    parser.add_argument(
        '--sizing', type=str, default='directional',
        choices=['directional', 'volscale'],
        help="Position sizing scheme — directional: HMM {10,4,3} | "
             "volscale: Barroso-Santa-Clara 2015 (default: directional)"
    )
    args = parser.parse_args()

    run_50 = run_100 = False
    if args.index == 'all':
        run_50 = run_100 = True
    elif args.index == 'nifty50':   run_50  = True
    elif args.index == 'nifty100':  run_100 = True

    if run_50:
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV],
                         is_weekly=False, regime_method=args.regime, sizing_scheme=args.sizing)
        run_pit_universe("NIFTY 50", [HISTORICAL_COMPOSITION_CSV],
                         is_weekly=True,  regime_method=args.regime, sizing_scheme=args.sizing)
    if run_100:
        run_pit_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV],
                         is_weekly=False, regime_method=args.regime, sizing_scheme=args.sizing)
        run_pit_universe("NIFTY 100", [HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV],
                         is_weekly=True,  regime_method=args.regime, sizing_scheme=args.sizing)


if __name__ == '__main__':
    main()
