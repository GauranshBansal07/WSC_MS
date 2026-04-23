import os
import glob
import pandas as pd
from multiprocessing import Pool, cpu_count
import time

INPUT_DIR = "nifty_500_5min"
FIRST_10AM_OUT = "monthly_first_10am_nifty500.csv"
LAST_10AM_OUT = "monthly_last_10am_nifty500.csv"

def process_file(filepath):
    ticker = os.path.basename(filepath).replace(".csv", "")
    try:
        df = pd.read_csv(filepath, usecols=["date", "close"])
        df['date'] = pd.to_datetime(df['date'], format='mixed', utc=True)
        df.set_index('date', inplace=True)
        
        # Convert to Asia/Kolkata
        df.index = df.index.tz_convert('Asia/Kolkata')
        
        # Extract 09:55 to 10:05 window. Take the last tick in this window per day (usually 10:00 or 10:05)
        mornings = df.between_time('09:55', '10:05')
        if mornings.empty:
            return None, None
            
        daily_10am = mornings['close'].resample('D').last().dropna()
        daily_10am.name = ticker + ".NS"
        
        # To get the First day of the month: resample by Month Start 'MS'
        first_10am = daily_10am.resample('MS').first().dropna()
        
        # To get the Last day of the month: resample by Month End 'ME'
        last_10am = daily_10am.resample('ME').last().dropna()
        
        return first_10am, last_10am
        
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        return None, None

def main():
    print("Finding Nifty 500 CSVs...")
    files = glob.glob(os.path.join(INPUT_DIR, "*.csv"))
    print(f"Found {len(files)} files.")
    
    start_time = time.time()
    
    print(f"Extracting Intra-month bounds across {cpu_count()} CPU cores...")
    with Pool(cpu_count()) as p:
        results = p.map(process_file, files)
        
    first_series = []
    last_series = []
    
    for f, l in results:
        if f is not None and not f.empty:
            first_series.append(f)
        if l is not None and not l.empty:
            last_series.append(l)
            
    print("Combining First 10AMs...")
    first_df = pd.concat(first_series, axis=1)
    # Reindex to force Month-End format for consistent merging/lookup later
    first_df.index = first_df.index + pd.offsets.MonthEnd(0)
    first_df.index = first_df.index.tz_localize(None)
    first_df.sort_index(inplace=True)
    first_df.to_csv(FIRST_10AM_OUT)
    print(f"Saved {FIRST_10AM_OUT} -> {first_df.shape}")

    print("Combining Last 10AMs...")
    last_df = pd.concat(last_series, axis=1)
    last_df.index = last_df.index.tz_localize(None)
    last_df.sort_index(inplace=True)
    last_df.to_csv(LAST_10AM_OUT)
    print(f"Saved {LAST_10AM_OUT} -> {last_df.shape}")
    
    print(f"Done in {time.time() - start_time:.1f}s")

if __name__ == '__main__':
    main()
