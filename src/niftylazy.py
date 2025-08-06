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

# Load portfolio from file if it exists
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

# Save portfolio to file
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
        logging.error(f"Error saving portfolio: {e}")

# Placeholder functions for trading API
def place_sell_order(symbol, quantity):
    logging.info(f"SELL ORDER: {quantity} shares of {symbol} executed.")
    return True

def place_buy_order(symbol, quantity):
    logging.info('bought')
    #logging.info(f"BUY ORDER: {quantity} shares of {symbol} executed for Rs.15000.")
    return True

# Fetch current price of a stock
price_cache = {}
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
from datetime import datetime

# Calculate total portfolio size in INR
def get_total_portfolio_value():
    total_value = 0
    for symbol in portfolio:
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in portfolio[symbol]:
            total_value += entry['quantity'] * current_price
    logging.info(f"Total Portfolio Value: Rs.{total_value:.2f}")
    return total_value

# Calculate drawdown per stock and total drawdown
def get_drawdowns():
    total_drawdown = 0
    drawdown_details = {}
    for symbol in portfolio:
        symbol_drawdown = 0
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in portfolio[symbol]:
            purchase_price = entry['purchase_price']
            quantity = entry['quantity']
            drawdown =  (current_price - purchase_price) * quantity
            symbol_drawdown += drawdown
        drawdown_details[symbol] = round(symbol_drawdown,2)
        total_drawdown += symbol_drawdown
    import pprint
    pp = pprint.PrettyPrinter()
    pp.pprint(drawdown_details)

    logging.info(f"Total Drawdown: Rs.{total_drawdown:.2f}")
    logging.info(f"Drawdown per Stock: {drawdown_details}")
    return total_drawdown, drawdown_details

# Calculate average holding period in days
def get_average_holding_period():
    total_days = 0
    entry_count = 0
    today = datetime.now().date()
    for symbol in portfolio:
        for entry in portfolio[symbol]:
            try:
                purchase_date = datetime.strptime(entry['purchase_date'], '%Y-%m-%d').date()
                holding_days = (today - purchase_date).days
                total_days += holding_days
                entry_count += 1
            except Exception as e:
                logging.warning(f"Date parse error: {e}")
    avg_days = total_days / entry_count if entry_count > 0 else 0
    logging.info(f"Average Holding Period: {avg_days:.2f} days")
    return avg_days

import numpy as np


# Calculate percentage return and CAGR
def get_portfolio_returns():
    total_invested = 0
    current_value = 0
    earliest_date = datetime.now().date()

    for symbol in portfolio:
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in portfolio[symbol]:
            quantity = entry['quantity']
            purchase_price = entry['purchase_price']
            purchase_date = datetime.strptime(entry['purchase_date'], '%Y-%m-%d').date()
            invested = purchase_price * quantity
            value = current_price * quantity

            total_invested += invested
            current_value += value

            if purchase_date < earliest_date:
                earliest_date = purchase_date

    if total_invested == 0:
        logging.warning("No invested capital to calculate returns.")
        return

    percent_return = ((current_value - total_invested) / total_invested) * 100

    # CAGR
    days_held = (datetime.now().date() - earliest_date).days
    years_held = days_held / 365.0
    if years_held > 0:
        cagr = ((current_value / total_invested) ** (1 / years_held) - 1) * 100
    else:
        cagr = 0

    logging.info(f"Portfolio % Return: {percent_return:.2f}%")
    logging.info(f"Portfolio CAGR: {cagr:.2f}%")
    return percent_return, cagr


def xirr(cash_flows):
    from scipy.optimize import newton
    from math import isnan

    def npv(rate):
        if rate <= -1.0:
            return float('inf')  # invalid: cannot compound at ≤ -100%
        try:
            return sum(
                cf / ((1 + rate) ** ((date - first_date).days / 365))
                for date, cf in cash_flows
            )
        except Exception as e:
            logging.error(f"Error in NPV calculation: {e}")
            return float('inf')

    if not cash_flows or len(cash_flows) < 2:
        logging.warning("Not enough data points to calculate XIRR.")
        return None

    # Sort by date and extract first date
    cash_flows = sorted(cash_flows, key=lambda x: x[0])
    first_date = cash_flows[0][0]

    try:
        result = newton(npv, 0.1, maxiter=100, tol=1e-6)
        if isnan(result):
            raise RuntimeError("XIRR result is NaN")
        return result
    except RuntimeError as e:
        logging.error(f"XIRR calculation did not converge: {e}")
        return None

def get_portfolio_xirr():
    cash_flows = []

    for symbol in portfolio:
        for entry in portfolio[symbol]:
            date = datetime.strptime(entry['purchase_date'], '%Y-%m-%d').date()
            amount = -entry['purchase_price'] * entry['quantity']
            cash_flows.append((date, amount))

    today = datetime.now().date()
    total_current_value = 0

    for symbol in portfolio:
        current_price = get_current_price(symbol)
        if current_price is None:
            continue
        for entry in portfolio[symbol]:
            total_current_value += current_price * entry['quantity']

    if total_current_value > 0:
        cash_flows.append((today, total_current_value))

    logging.info("XIRR Cash Flows:")
    for dt, amt in cash_flows:
        logging.info(f"  {dt} : {amt:.2f}")

    xirr_result = xirr(cash_flows)
    if xirr_result is not None:
        logging.info(f"Portfolio XIRR: {xirr_result * 100:.2f}%")
        return xirr_result * 100
    else:
        logging.warning("XIRR could not be calculated.")
        return None


# Trading strategy
def check_and_trade():
    # Get current time in IST
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_date = now.strftime('%Y-%m-%d')
    logging.info(f"Running strategy at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    # Check if market is open (NSE: 9:15 AM to 3:30 PM IST, Monday to Friday)
    if now.weekday() >= 5:  # Saturday (5) or Sunday (6)
        logging.info("Market is closed (weekend). Skipping trade.")
        return
    '''if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour > 15 or (now.hour == 15 and now.minute > 30):
        logging.info("Market is closed (outside trading hours). Skipping trade.")
        return'''
    if current_date in NSE_HOLIDAYS:
        logging.info("Market is closed (holiday). Skipping trade.")
        return

    # Fetch Nifty 50 stocks
    nifty50_stocks = fetch_nifty50_stocks()
    if not nifty50_stocks:
        logging.error("No Nifty 50 stocks available. Skipping trade.")
        return

    # Load portfolio
    load_portfolio()

    # Step 1: Check portfolio for stocks not in Nifty 50
    for symbol in portfolio:
        if symbol not in nifty50_stocks:
            logging.warning(f"FLAG: {symbol} is in portfolio but NOT in Nifty 50 index. Review holding.")

    # Step 2: Check portfolio for stocks up 5% or more
    max_gain_stock = None
    max_gain_percent = 0.0
    max_gain_entry_index = None

    for symbol in portfolio:
        for i, entry in enumerate(portfolio[symbol]):
            purchase_price = entry['purchase_price']
            current_price = get_current_price(symbol)
            if current_price is None:
                continue
            gain_percent = ((current_price - purchase_price) / purchase_price) * 100
            logging.info(f"{symbol} (Entry {i}): Current Price={current_price:.2f}, Purchase Price={purchase_price:.2f}, Purchase Date={entry['purchase_date']}, Gain={gain_percent:.2f}%")
            
            if gain_percent >= 5.0:
                if gain_percent > max_gain_percent:
                    max_gain_stock = symbol
                    max_gain_percent = gain_percent
                    max_gain_entry_index = i

    # Step 3: Sell the stock with maximum gain if any
    if max_gain_stock:
        quantity = sum(entry['quantity'] for entry in portfolio[max_gain_stock])
        if place_sell_order(max_gain_stock, quantity):
            logging.info(f"Sold {quantity} shares of {max_gain_stock} with {max_gain_percent:.2f}% gain.")
            print(f"Sold {quantity} shares of {max_gain_stock} with {max_gain_percent:.2f}% gain.")
            del portfolio[max_gain_stock]
            save_portfolio()
        else:
            logging.error(f"Failed to sell {max_gain_stock}.")
    else:
        logging.info("No stocks in portfolio are up 5% or more.")

    # Step 4: Check Nifty 50 stocks for farthest from 20-day MA, excluding portfolio stocks
    deviation_list = []

    for symbol in nifty50_stocks:
        if symbol in portfolio:
            logging.info(f"Skipping {symbol}: Already in portfolio.")
            continue
        current_price = get_current_price(symbol)
        ma_20 = get_20_day_ma(symbol)
        if current_price is None or ma_20 is None:
            continue
        deviation = ((current_price - ma_20) / ma_20) * 100
        logging.info(f"{symbol}: Current Price={current_price:.2f}, 20-day MA={ma_20:.2f}, Deviation={deviation:.2f}%")
        deviation_list.append((symbol, deviation, current_price))

    # Step 5: Buy the stock farthest from 20-day MA (most negative deviation)
    if not deviation_list:
        logging.info("No eligible stocks for buying (all Nifty 50 stocks are in portfolio or no valid data).")
        return

    # Sort by deviation (ascending, most negative first)
    deviation_list.sort(key=lambda x: x[1])
   
    max_deviation_stock, deviation, current_price = deviation_list[0]
    logging.info(f"Selected {max_deviation_stock} for purchase (deviation={deviation:.2f}%).")

    # Step 6: Execute buy order
    quantity = int(15000 / current_price)
    if quantity > 0:
        if place_buy_order(max_deviation_stock, quantity):
            logging.info(f"Bought {quantity} shares of {max_deviation_stock} at {current_price:.2f}.")
            print(f"Bought {quantity} shares of {max_deviation_stock} at {current_price:.2f}.")
            if max_deviation_stock not in portfolio:
                portfolio[max_deviation_stock] = []
            portfolio[max_deviation_stock].append({
                'purchase_price': current_price,
                'quantity': quantity,
                'purchase_date': current_date
            })
            save_portfolio()
        else:
            logging.error(f"Failed to buy {max_deviation_stock}.")
    else:
        logging.error(f"Cannot buy {max_deviation_stock}: Insufficient funds for even 1 share.")

    get_total_portfolio_value()
    get_drawdowns()
    get_average_holding_period()
    get_portfolio_returns()
    get_portfolio_xirr()

# Run the trading strategy
if __name__ == "__main__":
    init_db()
    check_and_trade()