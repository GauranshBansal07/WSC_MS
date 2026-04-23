import pandas as pd
import yfinance as yf
from config import DATA_START, DATA_END, HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV
from data_fetcher import load_historical_composition

def main():
    print("Loading compositions...")
    mask_c, tickers = load_historical_composition([HISTORICAL_COMPOSITION_CSV, NIFTY_NEXT_50_COMPOSITION_CSV])
    
    # Pre-process appending .NS for Yahoo format if not already present
    yf_tickers = [t if t.endswith('.NS') else f"{t}.NS" for t in tickers]
    
    print(f"Downloading Open prices for {len(yf_tickers)} tickers via YFinance (Adjusted)...")
    
    # We download daily data with auto_adjust=True which applies Corporate Action multipliers equally to Open/High/Low/Close
    raw = yf.download(yf_tickers, start=DATA_START, end=DATA_END, interval='1d',
                      progress=False, auto_adjust=True, group_by='column')
    
    opens = raw['Open']
    
    print("Extracting First Open and Last Open bounds...")
    
    # To get First valid open price of the month
    first_open = opens.resample('MS').first()
    
    # To get Last valid open price of the month
    last_open = opens.resample('ME').last()
    
    # Align indexing precisely to End of Month so that it matches exactly how the old matrix worked
    first_open.index = first_open.index + pd.offsets.MonthEnd(0)
    
    first_open.to_csv("monthly_first_open_adj.csv")
    last_open.to_csv("monthly_last_open_adj.csv")
    
    print(f"Extraction Successful! Matrix Shape: {first_open.shape}")

if __name__ == '__main__':
    main()
