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

from collections import defaultdict
import numpy_financial as npf  # requires `pip install numpy-financial`

# === PORTFOLIO ANALYTICS ===

def get_total_portfolio_value():
    """Current total market value of portfolio."""
    load_portfolio()
    total_value = 0.0
    for symbol, entries in portfolio.items():
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in entries:
            total_value += entry['quantity'] * current_price
    return total_value


def get_drawdowns(symbol=None):
    """
    Compute drawdowns based on daily mark-to-market portfolio value.
    If symbol is passed, restrict to that stock only.
    """
    load_portfolio()
    with sqlite3.connect(DB_FILE) as conn:
        if symbol:
            rows = conn.execute(
                "SELECT date, close FROM price_history WHERE symbol = ? ORDER BY date",
                (symbol,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT date, SUM(close) FROM price_history GROUP BY date ORDER BY date"
            ).fetchall()

    if not rows:
        return []

    values = [r[1] for r in rows]
    peak = values[0]
    drawdowns = []
    for v in values:
        peak = max(peak, v)
        dd = (v - peak) / peak
        drawdowns.append(dd)
    return drawdowns


def get_average_holding_period():
    """Average days held for exited positions."""
    if not os.path.exists("exits.json"):
        return 0
    with open("exits.json", "r") as f:
        exits = json.load(f)

    days = []
    for e in exits:
        buy_date = datetime.strptime(e["purchase_date"], "%Y-%m-%d")
        sell_date = datetime.strptime(e["sell_date"], "%Y-%m-%d")
        days.append((sell_date - buy_date).days)
    return sum(days) / len(days) if days else 0


def get_portfolio_returns():
    """Simple cumulative return and annualized return."""
    load_portfolio()
    invested = 0.0
    current_val = 0.0
    for symbol, entries in portfolio.items():
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in entries:
            invested += entry["quantity"] * entry["purchase_price"]
            current_val += entry["quantity"] * current_price

    if invested == 0:
        return 0, 0
    cum_return = (current_val - invested) / invested
    ann_return = (1 + cum_return) ** (365/ max(1, len(portfolio))) - 1
    return cum_return, ann_return

def xnpv(rate, cashflows):
    """
    Compute NPV for irregular cashflows.
    cashflows: list of (date, value)
    """
    t0 = min(cf[0] for cf in cashflows)
    return sum(
        cf[1] / ((1 + rate) ** ((cf[0] - t0).days / 365.0))
        for cf in cashflows
    )

def get_portfolio_xirr():
    """Compute XIRR using Newton's method."""
    cashflows = []

    # buys
    load_portfolio()
    if os.path.exists("exits.json"):
        with open("exits.json", "r") as f:
            exits = json.load(f)
    else:
        exits = []

    for symbol, entries in portfolio.items():
        for e in entries:
            cashflows.append((datetime.strptime(e["purchase_date"], "%Y-%m-%d"),
                              -e["quantity"] * e["purchase_price"]))

    for e in exits:
        cashflows.append((datetime.strptime(e["sell_date"], "%Y-%m-%d"),
                          e["quantity"] * e["sell_price"]))

    if not cashflows:
        return 0.0

    # try Newton’s method
    try:
        result = newton(
            lambda r: xnpv(r, cashflows),
            0.1  # initial guess 10%
        )
        return result
    except Exception as e:
        logging.error(f"XIRR calculation failed: {e}")
        return 0.0


# === BACKDATED EXIT HANDLING ===
def sell_from_portfolio_backdated(symbol, quantity_to_sell, sell_price, sell_date):
    """
    Sell with a specific backdated sell_date.
    Records in exits.json for analytics.
    """
    load_portfolio()
    if symbol not in portfolio:
        logging.warning(f"Tried to sell {symbol} but not in portfolio.")
        return False

    exits = []
    if os.path.exists("exits.json"):
        with open("exits.json", "r") as f:
            exits = json.load(f)

    remaining = quantity_to_sell
    updated_entries = []

    for entry in portfolio[symbol]:
        if remaining <= 0:
            updated_entries.append(entry)
            continue

        if entry["quantity"] <= remaining:
            exits.append({
                "symbol": symbol,
                "quantity": entry["quantity"],
                "purchase_price": entry["purchase_price"],
                "purchase_date": entry["purchase_date"],
                "sell_price": sell_price,
                "sell_date": sell_date
            })
            remaining -= entry["quantity"]
        else:
            exits.append({
                "symbol": symbol,
                "quantity": remaining,
                "purchase_price": entry["purchase_price"],
                "purchase_date": entry["purchase_date"],
                "sell_price": sell_price,
                "sell_date": sell_date
            })
            entry["quantity"] -= remaining
            remaining = 0
            updated_entries.append(entry)

    if updated_entries:
        portfolio[symbol] = updated_entries
    else:
        del portfolio[symbol]

    save_portfolio()
    with open("exits.json", "w") as f:
        json.dump(exits, f, indent=4)

    return True
def get_booked_profit():
    """Return total realized (booked) profit from exited trades."""
    if not os.path.exists("exits.json"):
        return 0.0

    with open("exits.json", "r") as f:
        exits = json.load(f)

    profit = 0.0
    for e in exits:
        buy_val = e["purchase_price"] * e["quantity"]
        sell_val = e["sell_price"] * e["quantity"]
        profit += (sell_val - buy_val)
    return profit


def get_mtm_unrealized():
    """Return current unrealized mark-to-market P&L for open positions."""
    load_portfolio()
    mtm = 0.0
    for symbol, entries in portfolio.items():
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for e in entries:
            buy_val = e["purchase_price"] * e["quantity"]
            current_val = current_price * e["quantity"]
            mtm += (current_val - buy_val)
    return mtm

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
            # Backdated sell with current date
            sell_from_portfolio_backdated(symbol, quantity, current_price, current_date)
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
    else:
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

    # === PORTFOLIO ANALYTICS LOGGING ===
    try:
        total_value = get_total_portfolio_value()
        drawdowns = get_drawdowns()
        avg_holding = get_average_holding_period()
        cum_ret, ann_ret = get_portfolio_returns()
        xirr_val = get_portfolio_xirr()
        booked_pnl = get_booked_profit()
        mtm_pnl = get_mtm_unrealized()

        logging.info("=== PORTFOLIO STATS ===")
        logging.info(f"Total Portfolio Value: INR {total_value:,.2f}")
        logging.info(f"Booked Profit (Realized): {booked_pnl:,.2f}")
        logging.info(f"Unrealized P&L (MTM): {mtm_pnl:,.2f}")
        if drawdowns:
            logging.info(f"Max Drawdown: {min(drawdowns)*100:.2f}%")
        logging.info(f"Average Holding Period (days): {avg_holding:.1f}")
        logging.info(f"Cumulative Return: {cum_ret*100:.2f}% | Annualized: {ann_ret*100:.2f}%")
        logging.info(f"XIRR: {xirr_val*100:.2f}%")
    except Exception as e:
        logging.error(f"Error computing portfolio analytics: {e}")

if __name__ == "__main__":
    init_db()
    check_and_trade()
