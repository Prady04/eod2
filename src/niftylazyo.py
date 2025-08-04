# niftylazy_optimized.py
import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import json
import os
import logging
import io
import requests

# Logging setup
handlers = [
    logging.FileHandler('./src/trading.log'),
    logging.StreamHandler()
]
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=handlers)

PORTFOLIO_FILE = "portfolio.json"
DB_FILE = "market_data.db"
NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"
HEADERS = {'User-Agent': 'Mozilla/5.0'}
NSE_HOLIDAYS = ["2025-10-29", "2025-12-25"]
portfolio = {}

# === DB SETUP ===
def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS price_history (
            symbol TEXT,
            date TEXT,
            close REAL,
            volume REAL,
            PRIMARY KEY (symbol, date)
        )""")

# === NIFTY 50 STOCKS ===
def fetch_nifty50_stocks():
    try:
        response = requests.get(NIFTY50_URL, headers=HEADERS)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        return [symbol + ".NS" for symbol in df['Symbol'] if 'Symbol' in df.columns]
    except Exception as e:
        logging.error(f"Error fetching Nifty 50 list: {e}")
        return []

# === PORTFOLIO MGMT ===
def load_portfolio():
    global portfolio
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE, 'r') as f:
            portfolio = json.load(f)
    else:
        portfolio = {}
    return portfolio

def save_portfolio():
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

# === PRICE DATA ===
def get_last_date(symbol):
    with sqlite3.connect(DB_FILE) as conn:
        result = conn.execute("SELECT MAX(date) FROM price_history WHERE symbol = ?", (symbol,)).fetchone()
        return result[0] if result else None

def fetch_and_store_price_data(symbol, start_date, period):
    try:
        stock = yf.Ticker(symbol)

        if period == '20d':
            price = stock.history(period=period)
            price.index = pd.to_datetime(price.index)

            with sqlite3.connect(DB_FILE) as conn:
                for index, row in price.iterrows():
                    date_obj = index.to_pydatetime()
                    conn.execute("""
                        INSERT OR IGNORE INTO price_history (symbol, date, close, volume)
                        VALUES (?, ?, ?, ?)""",
                        (symbol, date_obj.strftime('%Y-%m-%d'), row['Close'], row['Volume'])
                    )

        else:
            price = stock.info.get('regularMarketPrice')
            if not price:
                price = stock.history(period='1d')['Close'][-1]

            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO price_history (symbol, date, close, volume)
                    VALUES (?, ?, ?, ?)""",
                    (symbol, datetime.now().strftime('%Y-%m-%d'), price, 0)
                )
    except Exception as e:
        logging.error(f"Error fetching data for {symbol}: {e}")


def get_current_price(symbol):
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute("SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
        if row:
            return row[0]

    today = datetime.now().strftime('%Y-%m-%d')
    fetch_and_store_price_data(symbol, today, '1d')
    return get_current_price(symbol)

def get_20_day_ma(symbol):
    last_date = get_last_date(symbol)
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = '2022-01-01' if last_date is None else last_date
    fetch_and_store_price_data(symbol, start_date, '20d')

    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 20", (symbol,)).fetchall()
        closes = [r[0] for r in rows]
        return sum(closes) / len(closes) if len(closes) == 20 else None

# === TRADING ===
def place_sell_order(symbol, quantity):
    logging.info(f"SELL ORDER: {quantity} shares of {symbol}")
    return True

def place_buy_order(symbol, quantity):
    logging.info(f"BUY ORDER: {quantity} shares of {symbol}")
    return True

def check_and_trade():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_date = now.strftime('%Y-%m-%d')
    if now.weekday() >= 5 or current_date in NSE_HOLIDAYS:
        return

    nifty50_stocks = fetch_nifty50_stocks()
    if not nifty50_stocks:
        return

    load_portfolio()

    for symbol in portfolio:
        if symbol not in nifty50_stocks:
            logging.warning(f"{symbol} in portfolio but not in Nifty 50.")

    # SELL LOGIC
    max_gain_stock, max_gain_percent, max_entry_index = None, 0.0, None
    for symbol in portfolio:
        for i, entry in enumerate(portfolio[symbol]):
            price = get_current_price(symbol)
            if price:
                gain = ((price - entry['purchase_price']) / entry['purchase_price']) * 100
                if gain >= 5.0 and gain > max_gain_percent:
                    max_gain_stock = symbol
                    max_gain_percent = gain
                    max_entry_index = i

    if max_gain_stock:
        qty = sum(e['quantity'] for e in portfolio[max_gain_stock])
        if place_sell_order(max_gain_stock, qty):
            del portfolio[max_gain_stock]
            save_portfolio()

    # BUY LOGIC
    deviation_list = []
    for symbol in nifty50_stocks:
        if symbol in portfolio:
            continue
        cp = get_current_price(symbol)
        ma20 = get_20_day_ma(symbol)
        if cp and ma20:
            dev = ((cp - ma20) / ma20) * 100
            deviation_list.append((symbol, dev, cp))

    if deviation_list:
        deviation_list.sort(key=lambda x: x[1])
        symbol, deviation, price = deviation_list[0]
        qty = int(15000 / price)
        if qty > 0 and place_buy_order(symbol, qty):
            if symbol not in portfolio:
                portfolio[symbol] = []
            portfolio[symbol].append({
                'purchase_price': price,
                'quantity': qty,
                'purchase_date': current_date
            })
            save_portfolio()

# === ENTRY POINT ===
if __name__ == "__main__":
    init_db()
    check_and_trade()
