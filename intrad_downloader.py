import yfinance as yf
import pandas as pd
import os
import json
from datetime import datetime, timedelta

# Configuration
SYMBOLS = ["AAPL", "TSLA", "TCS.NS"]
DATA_DIR = "data"
OUTPUT_FILE = os.path.join(DATA_DIR, "final_data.csv")
LAST_UPDATE_FILE = "last_update.json"
IST_OFFSET = timedelta(hours=5, minutes=30)  # Convert UTC to IST

def load_last_update():
    """Load last update date from JSON file."""
    if os.path.exists(LAST_UPDATE_FILE):
        with open(LAST_UPDATE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_last_update(last_date):
    """Save the last available date from the CSV file to the update file."""
    with open(LAST_UPDATE_FILE, "w") as f:
        json.dump({"last_update": last_date}, f)

def fetch_5min_data(symbol, start_date, end_date):
    """Fetch 5-minute historical data from Yahoo Finance and convert to IST."""
    print(f"Fetching {symbol} data from {start_date} to {end_date}...")

    start_dt = datetime.combine(start_date, datetime.min.time())
    end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(hours=24)

    try:
        data = yf.download(symbol, start=start_dt, end=end_dt, interval="5m")
        if not data.empty:
            data = data[["Open", "High", "Low", "Close", "Volume"]].reset_index()

            # Convert UTC to IST
            data["Datetime"] = data["Datetime"] + IST_OFFSET

            # Extract IST Date & Time
            data["Date"] = data["Datetime"].dt.strftime("%Y-%m-%d")
            data["Time"] = data["Datetime"].dt.strftime("%H:%M")

            data["Symbol"] = str(symbol).strip('.NS')

            # Convert numeric values to 2 decimal places
            for col in ["Open", "High", "Low", "Close"]:
                data[col] = data[col].astype(float).round(2)

            return data[["Date", "Time", "Symbol", "Open", "High", "Low", "Close", "Volume"]]
        else:
            print(f"⚠ No 5m data for {symbol}. Possible restriction.")
    except Exception as e:
        print(f"❌ Error fetching data for {symbol}: {e}")

    return None

def save_last_update(last_date):
    """Save the last fetched date to the JSON file."""
    with open(LAST_UPDATE_FILE, "w") as f:
        json.dump({"last_update": last_date}, f)
    print(f"💾 Last update date saved: {last_date}")


def save_data(dataframes):
    """Write data to a single file with one header and append new data correctly."""
    if not dataframes:
        print("⚠ No valid data to save.")
        return
    
    file_exists = os.path.exists(OUTPUT_FILE)

    # Open file once and write data correctly
    with open(OUTPUT_FILE, "a", newline="") as f:
        for df in dataframes:
            df.to_csv(f, index=False, header=False, mode="a")

    print(f"✅ Data saved in {OUTPUT_FILE}")

def load_last_update():
    """Load the last update date from the JSON file."""
    if os.path.exists(LAST_UPDATE_FILE):
        try:
            with open(LAST_UPDATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("last_update")  # Returns the last update date as a string
        except json.JSONDecodeError:
            print("⚠ JSON file corrupted. Resetting last update date.")
    
    return None  # Default if file does not exist or is corrupted


def get_required_dates():
    """Determine the correct start date using the last fetched date from JSON."""
    last_date = load_last_update()

    if last_date:
        try:
            start_date = datetime.strptime(last_date, "%Y-%m-%d").date() + timedelta(days=1)
            if start_date >= datetime.today().date():
                print("✅ Data is already up-to-date. No need to fetch.")
                return None, None
            print(f"📅 Fetching data from {start_date} to {datetime.today().date()}")
        except ValueError:
            print("⚠ Invalid date format in JSON. Resetting to last 30 days.")
            start_date = datetime.today().date() - timedelta(days=30)
    else:
        print("📂 No previous update found. Fetching last 30 days of data.")
        start_date = datetime.today().date() - timedelta(days=30)  # First-time run

    return start_date, datetime.today().date()

def main():
    start_date, end_date = get_required_dates()
    if start_date is None or end_date is None:
        print("✅ Data is already up-to-date. Exiting script.")
        return  # Exit early if there's nothing to fetch
    all_data = []
    failed_symbols = []
    df = pd.read_csv('fnostocks.csv')
    for symbol in df['Symbol']:
        df = fetch_5min_data(symbol+".NS", start_date, end_date)
        if df is not None:
            all_data.append(df)

    save_data(all_data)

    # Update last download date to match last available date in the final CSV
    save_last_update(str(end_date))

    if failed_symbols:
        print(f"❌ Failed to download: {', '.join(failed_symbols)}")

if __name__ == "__main__":
    main()
