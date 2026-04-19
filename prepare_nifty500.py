import os
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

INPUT_DIR = "nifty_500_5min"
DAILY_OUT = "daily_cache_nifty500.csv"
MONTHLY_OUT = "monthly_cache_nifty500.csv"

def process_file(filepath):
    ticker = os.path.basename(filepath).replace(".csv", "")
    try:
        # We only need date and close
        df = pd.read_csv(filepath, usecols=["date", "close"])
        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
        df.set_index('date', inplace=True)
        
        # Convert to Asia/Kolkata timezone to avoid date boundary issues
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Resample to Daily
        daily = df['close'].resample('D').last().dropna()
        daily.name = ticker + ".NS"  # Append .NS to match YFinance format
        
        # Resample to Monthly End
        monthly = df['close'].resample('ME').last().dropna()
        monthly.name = ticker + ".NS"
        
        return daily, monthly
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None

def main():
    print("Finding Nifty 500 CSVs...")
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files.")
    
    start_time = time.time()
    
    # Process files in parallel
    print(f"Processing across {cpu_count()} CPU cores...")
    with Pool(cpu_count()) as p:
        results = p.map(process_file, files)
        
    daily_series = []
    monthly_series = []
    
    for d, m in results:
        if d is not None and not d.empty:
            daily_series.append(d)
        if m is not None and not m.empty:
            monthly_series.append(m)
            
    print("Combining Daily Data...")
    daily_df = pd.concat(daily_series, axis=1)
    # Convert index back to naive timezone (for consistency with yfinance output in python)
    daily_df.index = daily_df.index.tz_localize(None)
    daily_df.sort_index(inplace=True)
    daily_df.to_csv(DAILY_OUT)
    print(f"Saved {DAILY_OUT} -> {daily_df.shape}")

    print("Combining Monthly Data...")
    monthly_df = pd.concat(monthly_series, axis=1)
    monthly_df.index = monthly_df.index.tz_localize(None)
    monthly_df.sort_index(inplace=True)
    
    # Fill tiny gaps but leave missing data NaN
    # Not strongly necessary to fill for monthly, but keeps dataframe tidy
    monthly_df.to_csv(MONTHLY_OUT)
    print(f"Saved {MONTHLY_OUT} -> {monthly_df.shape}")
    
    print(f"Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()
