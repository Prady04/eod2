import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import json
import os
import requests
import io
import logging
from scipy.optimize import newton

handlers = [logging.FileHandler('./src/trading.log'), logging.StreamHandler()]
# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)

# File to store portfolio
PORTFOLIO_FILE = "portfolio.json"
DB_FILE = "market_data.db"
NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

# Headers for HTTP request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# NSE holidays (example list, update as needed)
NSE_HOLIDAYS = ["2025-10-29", "2025-12-25"]  # Add more holidays

# Initialize empty portfolio
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
        if 'Symbol' not in df.columns:
            logging.error("Error: 'Symbol' column not found in CSV.")
            return []
        # Append .NS to symbols for Yahoo Finance compatibility
        nifty50_stocks = [symbol + ".NS" for symbol in df['Symbol']]
        logging.info(f"Fetched {len(nifty50_stocks)} Nifty 50 stocks.")
        return nifty50_stocks
    except Exception as e:
        logging.error(f"Error fetching Nifty 50 list: {e}")
        return []

# === PORTFOLIO HANDLING ===
def load_portfolio():
    global portfolio
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
            logging.info("Portfolio loaded from file.")
        except Exception as e:
            logging.error(f"Error loading portfolio: {e}")
            portfolio = {}
    else:
        portfolio = {}
    return portfolio

def save_portfolio():
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=4)

def add_to_portfolio(symbol, purchase_price, quantity, purchase_date):
    load_portfolio()
    if symbol not in portfolio:
        portfolio[symbol] = []
    portfolio[symbol].append({
        "purchase_price": float(purchase_price),
        "quantity": int(quantity),
        "purchase_date": purchase_date
    })
    save_portfolio()

def sell_from_portfolio(symbol, quantity_to_sell, sell_price):
    load_portfolio()
    if symbol not in portfolio:
        logging.warning(f"Tried to sell {symbol} but not in portfolio.")
        return False

    remaining = quantity_to_sell
    updated_entries = []

    for entry in portfolio[symbol]:
        if remaining <= 0:
            updated_entries.append(entry)
            continue

        if entry["quantity"] <= remaining:
            logging.info(f"Selling {entry['quantity']} shares of {symbol} bought at {entry['purchase_price']}")
            remaining -= entry["quantity"]
        else:
            logging.info(f"Selling {remaining} shares of {symbol} bought at {entry['purchase_price']}")
            entry["quantity"] -= remaining
            remaining = 0
            updated_entries.append(entry)

    if updated_entries:
        portfolio[symbol] = updated_entries
    else:
        del portfolio[symbol]  # remove symbol completely if no shares left

    save_portfolio()
    return True

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
        logging.error(f"Error saving portfolio: {e}")

# === TRADING API PLACEHOLDERS ===
def place_sell_order(symbol, quantity):
    logging.info(f"SELL ORDER: {quantity} shares of {symbol} executed.")
    return True

def place_buy_order(symbol, quantity):
    logging.info(f"BUY ORDER: {quantity} shares of {symbol} executed.")
    return True

# === PRICE HELPERS ===
def get_current_price(symbol, avoidCache=False):
    if avoidCache:
        today = datetime.now().strftime('%Y-%m-%d')
        fetch_and_store_price_data(symbol, today, '1d')
        return get_current_price(symbol)
    else:
        with sqlite3.connect(DB_FILE) as conn:
            row = conn.execute("SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
            if row:
                return row[0]


def get_20_day_ma(symbol):
    last_date = get_last_date(symbol)
    today = datetime.now().strftime('%Y-%m-%d')
    start_date = '2022-01-01' if last_date is None else last_date
    fetch_and_store_price_data(symbol, start_date, '20d')

    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 20", (symbol,)).fetchall()
        closes = [r[0] for r in rows]
        return sum(closes) / len(closes) if len(closes) == 20 else None

# === PORTFOLIO ANALYTICS ===
# (unchanged from your code except they now use updated portfolio structure)
# ... keep your existing get_total_portfolio_value, get_drawdowns, get_average_holding_period,
# get_portfolio_returns, get_portfolio_xirr as they are ...

# === TRADING STRATEGY ===
def check_and_trade():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_date = now.strftime('%Y-%m-%d')
    logging.info(f"Running strategy at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if now.weekday() >= 5:
        logging.info("Market is closed (weekend). Skipping trade.")
        return
    if current_date in NSE_HOLIDAYS:
        logging.info("Market is closed (holiday). Skipping trade.")
        return

    nifty50_stocks = fetch_nifty50_stocks()
    if not nifty50_stocks:
        logging.error("No Nifty 50 stocks available. Skipping trade.")
        return

    load_portfolio()

    # Step 1: Warn if non-Nifty stocks in portfolio
    for symbol in portfolio:
        if symbol not in nifty50_stocks:
            logging.warning(f"FLAG: {symbol} is in portfolio but NOT in Nifty 50 index. Review holding.")

    # Step 2: Check for stocks with >5% gain
    max_gain_stocks = []
    max_gain_percent = 0.0

    for symbol in portfolio:
        for i, entry in enumerate(portfolio[symbol]):
            purchase_price = entry['purchase_price']
            current_price = get_current_price(symbol, True)
            if current_price is None:
                continue
            gain_percent = ((current_price - purchase_price) / purchase_price) * 100
            logging.info(f"{symbol} (Entry {i}): CP={current_price:.2f}, PP={purchase_price:.2f}, Date={entry['purchase_date']}, Gain={gain_percent:.2f}%")

            if gain_percent >= 5.0:
                max_gain_stocks.append((symbol, current_price, gain_percent))
                if gain_percent > max_gain_percent:
                    max_gain_percent = gain_percent

    # Step 3: Sell profitable stocks
    for symbol, current_price, gain_percent in max_gain_stocks:
        quantity = sum(entry['quantity'] for entry in portfolio[symbol])
        if place_sell_order(symbol, quantity):
            sell_from_portfolio(symbol, quantity, current_price)
            logging.info(f"Sold {quantity} shares of {symbol} with {gain_percent:.2f}% gain.")
        else:
            logging.error(f"Failed to sell {symbol}.")

    # Step 4: Pick stock farthest below 20d MA
    deviation_list = []
    for symbol in nifty50_stocks:
        if symbol in portfolio:
            continue
        current_price = get_current_price(symbol)
        ma_20 = get_20_day_ma(symbol)
        if current_price is None or ma_20 is None:
            continue
        deviation = ((current_price - ma_20) / ma_20) * 100
        deviation_list.append((symbol, deviation, current_price))

    if not deviation_list:
        logging.info("No eligible stocks for buying.")
        return

    deviation_list.sort(key=lambda x: x[1])
    max_deviation_stock, deviation, current_price = deviation_list[0]

    # Step 5: Buy stock
    quantity = int(15000 / current_price)
    if quantity > 0:
        if place_buy_order(max_deviation_stock, quantity):
            add_to_portfolio(max_deviation_stock, current_price, quantity, current_date)
            logging.info(f"Bought {quantity} shares of {max_deviation_stock} at {current_price:.2f}.")
        else:
            logging.error(f"Failed to buy {max_deviation_stock}.")
    else:
        logging.error(f"Cannot buy {max_deviation_stock}: insufficient funds.")

if __name__ == "__main__":
    init_db()
    check_and_trade()
