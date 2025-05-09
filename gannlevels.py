import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import os

# 1. Your list of symbols
symbols = ["^NSEI"]

for sym in symbols:
    json_file = f"candles_{sym}.json"
    lvl_file  = f"gann_levels_{sym}.json"

    # 2. Load existing candle JSON (if any)
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            existing = json.load(f)
        # parse out the last date we have
        last_date = max(item['time'] for item in existing)
        start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=1)
    else:
        existing = []
        start_date = datetime.now() - timedelta(days=90)  # default look-back

    # 3. Fetch only from start_date until today
    df = yf.Ticker(sym).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    ).reset_index()

    if df.empty:
        print(f"No new data for {sym} since {start_date.date()}")
        new_candles = []
    else:
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        new_candles = [
            {
                "time": row['Date'].strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
            }
            for _, row in df.iterrows()
        ]

    # 4. Merge old + new, drop any duplicates just in case
    combined = {item['time']: item for item in existing + new_candles}
    candles = [combined[dt] for dt in sorted(combined)]

    # 5. Re-write the JSON file
    with open(json_file, "w") as f:
        json.dump(candles, f, indent=2)
    print(f"Updated {json_file}: {len(new_candles)} new points")

    # 6. Re-calculate Gann levels off the **lowest low** of the full set
    lows = [pt['low'] for pt in candles]
    #base = min(lows)
    closes =[pt['close'] for pt in candles]
    base=closes[-1]
    base_sqrt = base ** 0.5
    fracs = [0.125 * i for i in range(1, 9)]
    gann_levels = [(base_sqrt + af)**2 for af in fracs] + [(base_sqrt - af)**2 for af in fracs]

    # 7. Write levels JSON
    with open(lvl_file, "w") as f:
        json.dump(gann_levels, f, indent=2)
        print(gann_levels)
    print(f"Re-wrote {lvl_file} with base={base:.2f}\n")
