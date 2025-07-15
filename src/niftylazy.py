import yfinance as yf
import pandas as pd
from datetime import datetime
import time
import pytz
import json
import os
import requests
import io
import logging


handlers = [logging.FileHandler('./src/trading.log'), logging.StreamHandler()]
# Set up logging
logging.basicConfig(
    
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers

)

# File to store portfolio
PORTFOLIO_FILE = "portfolio.json"

# URL for Nifty 50 constituents
NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

# Headers for HTTP request
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# NSE holidays (example list, update as needed)
NSE_HOLIDAYS = ["2025-10-29", "2025-12-25"]  # Add more holidays

# Initialize empty portfolio
portfolio = {}

# Fetch Nifty 50 stock list from URL
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
    try:
        with open(PORTFOLIO_FILE, 'w') as f:
            json.dump(portfolio, f, indent=4)
        logging.info("Portfolio saved to file.")
    except Exception as e:
        logging.error(f"Error saving portfolio: {e}")

# Placeholder functions for trading API
def place_sell_order(symbol, quantity):
    logging.info(f"SELL ORDER: {quantity} shares of {symbol} executed.")
    return True

def place_buy_order(symbol, quantity):
    logging.info(f"BUY ORDER: {quantity} shares of {symbol} executed for ₹15000.")
    return True

# Fetch current price of a stock
def get_current_price(symbol):
    try:
        stock = yf.Ticker(symbol)
        price = stock.info.get('regularMarketPrice') or stock.history(period='1d')['Close'][-1]
        time.sleep(1)  # Avoid yfinance rate limits
        return price
    except Exception as e:
        logging.error(f"Error fetching price for {symbol}: {e}")
        return None

# Calculate 20-day moving average
def get_20_day_ma(symbol):
    try:
        stock = yf.Ticker(symbol)
        history = stock.history(period='1mo')  # Fetch 1 month of data
        if len(history) >= 20:
            ma_20 = history['Close'].tail(20).mean()
            time.sleep(1)  # Avoid yfinance rate limits
            return ma_20
        else:
            logging.error(f"Insufficient data for {symbol} to calculate 20-day MA.")
            return None
    except Exception as e:
        logging.error(f"Error calculating 20-day MA for {symbol}: {e}")
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
    if now.hour < 9 or (now.hour == 9 and now.minute < 15) or now.hour > 15 or (now.hour == 15 and now.minute > 30):
        logging.info("Market is closed (outside trading hours). Skipping trade.")
        return
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

# Run the trading strategy
if __name__ == "__main__":
    check_and_trade()