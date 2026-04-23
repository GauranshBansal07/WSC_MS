import pandas as pd
import numpy as np

def main():
    print("Loading Raw Unadjusted Tick Data...")
    raw_df = pd.read_csv("monthly_last_10am_nifty500.csv", index_col=0, parse_dates=True)
    
    print("Loading Adjusted YFinance Data...")
    adj_df = pd.read_csv("price_cache.csv", index_col=0, parse_dates=True)
    
    # Standardize indices to naive month ends for intersection
    raw_df.index = raw_df.index.tz_localize(None) + pd.offsets.MonthEnd(0)
    adj_df.index = adj_df.index.tz_localize(None) + pd.offsets.MonthEnd(0)
    
    common_dates = raw_df.index.intersection(adj_df.index)
    common_tickers = raw_df.columns.intersection(adj_df.columns)
    
    raw = raw_df.loc[common_dates, common_tickers]
    adj = adj_df.loc[common_dates, common_tickers]
    
    # Compute MoM Returns
    ret_raw = (raw / raw.shift(1)) - 1.0
    ret_adj = (adj / adj.shift(1)) - 1.0
    
    # Calculate the absolute difference in return reported
    diff = np.abs(ret_raw - ret_adj)
    
    # Find events
    # > 40% difference = Massive Stock Splits / Bonuses
    # > 10% difference = Moderate Splits / Spin-offs
    # > 2% difference = Standard Dividends / Minor corporate acts
    
    splits = (diff > 0.30).sum().sum()
    major_divs = ((diff > 0.05) & (diff <= 0.30)).sum().sum()
    all_corps = (diff > 0.02).sum().sum()
    
    print("\n--- CORPORATE ACTION DISCREPANCIES (2018-2026) ---")
    print(f"Total Tickers Analyzed: {len(common_tickers)}")
    print(f"Total Months Analyzed: {len(common_dates)}")
    print(f"Total Data Points: {len(common_tickers) * len(common_dates)}")
    print("-" * 50)
    print(f"Massive Stock Splits/Bonuses (>30% fake crash): {splits}")
    print(f"Major Dividends/Spin-offs (5% to 30% fake crash): {major_divs}")
    print(f"Total Noticeable Corporate Actions (>2% difference): {all_corps}")
    print("-" * 50)
    
    # Let's print a few famous examples of massive splits
    print("\nExamples of Massive Splits (Raw vs Adjusted):")
    split_mask = diff > 0.40
    for date in split_mask.index:
        for ticker in common_tickers:
            if split_mask.loc[date, ticker]:
                r_r = ret_raw.loc[date, ticker]
                a_r = ret_adj.loc[date, ticker]
                print(f"[{date.strftime('%Y-%m')}] {ticker}: Raw Return = {r_r*100:6.1f}% | True Adjusted Return = {a_r*100:6.1f}%")

if __name__ == "__main__":
    main()
