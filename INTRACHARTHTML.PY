import yfinance as yf
import pandas as pd
import json
import time
from datetime import datetime, timedelta
import os

SYMBOLS = ["^NSEI", "AAPL", "MSFT", "GOOG", "TSLA"]
INTERVAL = ["5m","15m"]
FILENAME_TEMPLATE = "intraday15_{}.json"

def fetch_and_save(symbol):
    try:
        print(f"Fetching data for {symbol}...")

        ticker = yf.Ticker(symbol)
        for interva in INTERVAL:
            df = ticker.history(interval=interva, period="1d")

            if df.empty:
                print(f"No data for {symbol}")
                return

            df = df.reset_index()
            df['time'] = df['Datetime'].apply(lambda x: int(x.timestamp()))
            df = df[['time', 'Open', 'High', 'Low', 'Close']].rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close'
            })

            json_data = df.to_dict(orient='records')

            file_path = FILENAME_TEMPLATE.format(symbol)
            with open(file_path, 'w') as f:
                json.dump(json_data, f, indent=2)

            print(f"Saved {len(df)} candles to {file_path}")
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

def is_market_open_ist():
    now = datetime.utcnow() + timedelta(hours=5.5)
    market_start = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_end = now.replace(hour=15, minute=30, second=0, microsecond=0)
    return market_start <= now <= market_end

if __name__ == "__main__":
    '''if not is_market_open_ist():
        print("Market is closed. No fetch.")
    else:'''
    for sym in SYMBOLS:
        fetch_and_save(sym)
