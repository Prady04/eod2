#yahoo finance nse data downloader for daily,15 min , 5 mintimeframe
#There is a time period restriction for 5 min, 15 min data from YF so it's recommended to run this weekly/Fortnightly

import os
import yfinance as yf
import pandas as pd
import requests
from datetime import datetime, timedelta


def cleanup(files):
    """
    function to cleanup empty df returned by Yahoo Finance
    Can be refactored in the downloader function itself
    """
    for data_file in files:
        data = pd.read_csv(data_file)
        if len(data) == 0:
            print(f"removing empty file: {data_file}")
            os.remove(data_file)


def downloader(stock, filename, interval, start, end):
    """
    Downloads data for the specified interval
    """
    if not os.path.isfile(filename):
        stock.history(start=start, end=end, interval=interval).to_csv(filename)
    else:
        data = pd.read_csv(filename)

        # to rename the Date column which only comes in initial data export
        if "Date" in data.columns:
            data.rename(columns={"Date": "Datetime"}, inplace=True)

        start = data.iloc[-1]["Datetime"].split()[0]
        if start == end:
            print("Data is up-to-date")
        else:
            start = datetime.strptime(start, "%Y-%m-%d") + timedelta(days=1)
            start = start.strftime("%Y-%m-%d")
            new_data = stock.history(start=start, end=end, interval=interval)
            new_data["Datetime"] = new_data.index.astype(str)

            # appending the new data if you have used the script before
            df = pd.concat([data, new_data], ignore_index=True)
            df.dropna(inplace=True)
            df.to_csv(filename, index=False)


def get_ticks():
    """
    Function for fetching tick data
    """
    if not os.path.isdir("yf_data"):
        os.mkdir("yf_data")

    symbols_file = r"nse_equities.csv"
    url = "https://archives.nseindia.com/content/equities/EQUITY_L.csv"

    # Downloads scripts csv if it's not there you can delete the nse_equities.csv
    # file after new IPOs to update the file
    if not os.path.isfile(symbols_file):
        response = requests.get(url)
        with open(symbols_file, "wb") as f:
            f.write(response.content)
    equities = pd.read_csv(symbols_file)
    symbols = equities["SYMBOL"].tolist()

    # YF adds a .NS suffix to NSE scripts
    symbols = [symbol + ".NS" for symbol in symbols]
    extras = [
        # indices
        "^NSEI",
        "^NSEBANK",
        "^NSMIDCP",
        "^NSEMDCP50",
        "NIFTYSMLCAP50.NS",
        # sector
        "^CNXIT",
        "^CNXAUTO",
        "^CNXFIN",
        "^CNXPHARMA",
        "^CNXFMCG",
        "^CNXMETAL",
        "^CNXREALTY",
        "^CNXENERGY",
        "^CNXINFRA",
        # etfs
        "GOLDBEES.NS",
        "MINDSPACE-RR.NS",
        "N100.NS",
        "NIFTYBEES.NS",
        "BANKBEES.NS",
    ]
    symbols.extend(extras)
    delta = timedelta(59)
    start = (datetime.now() - delta).strftime("%Y-%m-%d")
    daily_start = "2025-01-01"
    end = (datetime.now() + timedelta(1)).strftime("%Y-%m-%d")

    for symbol in symbols:
        try:
            stock = yf.Ticker(symbol)
            symbol = symbol.replace("^", "")

            tick_5 = f"yf_data/{symbol}-5ticks.csv"
            tick_15 = f"yf_data/{symbol}-XV ticks.csv"
            daily_file = f"yf_data/{symbol}-daily.csv"

            print(f"Fetching data for {symbol}")
            downloader(stock, daily_file, "1d", daily_start, end)
            downloader(stock, tick_5, "5m", start, end)
            downloader(stock, tick_15, "15m", start, end)
            cleanup([tick_5, tick_15, daily_file])
        except Exception as ex:
            print(ex)


if __name__ == "__main__":
    get_ticks()
