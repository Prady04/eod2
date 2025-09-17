import pandas as pd
import sqlite3
from datetime import datetime, timedelta
import pytz
import json, os
from tabulate import tabulate
import argparse

import requests
import io
import logging
from scipy.optimize import newton
import statistics
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Logging handlers (file + console)
handlers = []
try:
    handlers = [logging.FileHandler('./src/niftylazalgotrade.log'), logging.StreamHandler()]
except Exception:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)

PORTFOLIO_FILE = "kplportfolio.json"
DB_FILE = "kplmarket_data.db"

# example holidays
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

# === STOCK LIST ===

def fetch_nifty_fno_stocks():
    from nsepython import fnolist
    try:
        fno_stocks = fnolist()
        clean = [s.strip().lstrip('$') for s in fno_stocks]
        return [s + ".NS" for s in clean]
       
    except Exception as e:
       logging.error(f"Failed to fetch F&O list from nsepython: {e}")

# === PORTFOLIO HANDLING ===

def load_portfolio():
    global portfolio
    if os.path.exists(PORTFOLIO_FILE):
        try:
            with open(PORTFOLIO_FILE, 'r') as f:
                portfolio = json.load(f)
        except Exception as e:
            logging.error(f"Error reading portfolio file: {e}")
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


def sell_from_portfolio_backdated(symbol, quantity_to_sell, sell_price, sell_date):
    load_portfolio()
    if symbol not in portfolio:
        logging.warning(f"Tried to sell {symbol} but not in portfolio.")
        return False

    exits = []
    if os.path.exists('exits.json'):
        with open('exits.json', 'r') as f:
            try:
                exits = json.load(f)
            except Exception:
                exits = []

    remaining = quantity_to_sell
    updated_entries = []
    for entry in portfolio[symbol]:
        if remaining <= 0:
            updated_entries.append(entry)
            continue
        if entry['quantity'] <= remaining:
            exits.append({
                'symbol': symbol,
                'quantity': entry['quantity'],
                'purchase_price': entry['purchase_price'],
                'purchase_date': entry['purchase_date'],
                'sell_price': sell_price,
                'sell_date': sell_date
            })
            remaining -= entry['quantity']
        else:
            exits.append({
                'symbol': symbol,
                'quantity': remaining,
                'purchase_price': entry['purchase_price'],
                'purchase_date': entry['purchase_date'],
                'sell_price': sell_price,
                'sell_date': sell_date
            })
            entry['quantity'] -= remaining
            remaining = 0
            updated_entries.append(entry)

    if updated_entries:
        portfolio[symbol] = updated_entries
    else:
        del portfolio[symbol]
    save_portfolio()

    with open('exits.json', 'w') as f:
        json.dump(exits, f, indent=4)
    return True

# === PRICE DATA HELPERS ===

def get_last_date(symbol):
    with sqlite3.connect(DB_FILE) as conn:
        r = conn.execute("SELECT MAX(date) FROM price_history WHERE symbol = ?", (symbol,)).fetchone()
        return r[0] if r else None
def rows_in_db(symbol):
    with sqlite3.connect(DB_FILE) as conn:
        r = conn.execute("SELECT COUNT(1) FROM price_history WHERE symbol = ?", (symbol,)).fetchone()
        return r[0] if r else 0
    
import time
def ensure_price_data(symbol, min_rows=20, backoff=1.0, max_retries=3):
    """
    Ensure at least min_rows of historical closes exist for symbol.
    If not, fetch '20d' history from yfinance and store.
    Returns True if min_rows available after attempts, False otherwise.
    """
    symbol = symbol.strip()
    # quick check
    if rows_in_db(symbol) >= min_rows:
        return True

    for attempt in range(1, max_retries+1):
        try:
            fetch_and_store_price_data(symbol, period='20d')
            # small pause between calls to avoid throttling
            time.sleep(backoff * attempt)
        except Exception as e:
            logging.debug(f"ensure_price_data: attempt {attempt} failed for {symbol}: {e}")
        if rows_in_db(symbol) >= min_rows:
            return True
    logging.warning(f"Not enough price rows for {symbol} after {max_retries} attempts (have {rows_in_db(symbol)})")
    return False


# Adjusted fetch_and_store_price_data to return boolean for success
def fetch_and_store_price_data(symbol, start_date=None, period='1d'):
    """
    Fetch price data using yfinance and store into DB. Returns True on success (non-empty),
    False if no data returned.
    """
    if not symbol.find('NIFTY')== -1:
        logging.debug(f"Skipping fetch for index symbol {symbol}")
        return False
    try:
        import yfinance as yf
    except Exception:
        logging.debug("yfinance not installed — fetch_and_store_price_data is a no-op")
        return False

    try:
        if period == '20d':
            df = yf.Ticker(symbol).history(period=period)
            if df is None or df.empty:
                logging.debug(f"yfinance returned empty for {symbol} period={period}")
                return False
            df.index = pd.to_datetime(df.index)
            with sqlite3.connect(DB_FILE) as conn:
                for idx, row in df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    conn.execute("INSERT OR IGNORE INTO price_history (symbol,date,close,volume) VALUES (?,?,?,?)",
                                 (symbol, date_str, round(float(row['Close']),2), float(row.get('Volume', 0))))
            return True
        else:
            hist = yf.Ticker(symbol).history(period='1d')
            if hist is None or hist.empty:
                logging.debug(f"yfinance 1d empty for {symbol}")
                return False
            last = hist['Close'].iat[-1]
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("INSERT OR IGNORE INTO price_history (symbol,date,close,volume) VALUES (?,?,?,?)",
                             (symbol, datetime.now().strftime('%Y-%m-%d'), float(last), 0))
            return True
    except Exception as e:
        logging.error(f"Failed to fetch/store price for {symbol}: {e}")
        return False
def get_current_price(symbol, avoidCache=False):
    try:
        if avoidCache:
            fetch_and_store_price_data(symbol, period='1d')
        with sqlite3.connect(DB_FILE) as conn:
            r = conn.execute(
                "SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        return r[0] if r else None
    except Exception as e:
        logging.error(f"Price fetch failed for {symbol}: {e}")
        return None


# === STRATEGY HELPERS ===
def get_close(symbol, offset=0):
    """Get closing price with offset (0 = today, 1 = yesterday, etc.)"""
    with sqlite3.connect(DB_FILE) as conn:
        row = conn.execute(
            "SELECT close FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT 1 OFFSET ?",
            (symbol, offset)
        ).fetchone()
    return row[0] if row else None

def get_n_day_high(symbol, n=20):
    """Get the last n closes and return the highest."""
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(
            "SELECT close FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT ?",
            (symbol, n)
        ).fetchall()
    closes = [r[0] for r in rows]
    return max(closes) if len(closes) == n else None

def should_buy(symbol):
    # make sure we have at least 20 rows for this symbol
    ok = ensure_price_data(symbol, min_rows=20)
    if not ok:
        return False

    cp = get_close(symbol, 0)   # today’s close
    prev_close = get_close(symbol, 1)  # yesterday’s close
    if cp is None or prev_close is None:
        return False

    high20 = get_n_day_high(symbol, 20)
    if high20 is None:
        return False
   
    ret = (round(cp,2) >= round(high20,2)) and (prev_close < high20)
    print(symbol, "should_buy?", ret)
    return ret



def get_n_day_low(symbol, n=10):
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT close FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT ?", (symbol, n)).fetchall()
    closes = [r[0] for r in rows]
    return min(closes) if len(closes) == n else None

# === ORDER SIMS ===

def place_sell_order(symbol, quantity):
    logging.info(f"(sim) SELL {quantity} {symbol}")
    return True

def place_buy_order(symbol, quantity):
    logging.info(f"(sim) BUY {quantity} {symbol}")
    return True

# === STRATEGY ===

def check_and_trade():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_date = now.strftime('%Y-%m-%d')
    logging.info(f"Running strategy at {now.strftime('%Y-%m-%d %H:%M:%S')}")

    if now.weekday() >= 5 or current_date in NSE_HOLIDAYS:
        logging.info("Market closed. Skipping.")
        return

    fno_stocks = fetch_nifty_fno_stocks()
    if not fno_stocks:
        logging.info("No F&O list — skipping.")
        return

    load_portfolio()

    # --- EXIT CONDITIONS ---
    to_sell = []
    print(portfolio.items(), " portfolio items")
    for s, entries in portfolio.items():
        cp = get_current_price(s, avoidCache=True)
        if cp is None:
            continue
        low10 = get_n_day_low(s, 10)
        for e in entries:
            gain = (cp - e['purchase_price'])/e['purchase_price']*100
            if (low10 is not None and cp < low10) or gain >= 10:
                to_sell.append((s, cp, gain))
    for s, cp, gain in to_sell:
        qty = sum(x['quantity'] for x in portfolio.get(s,[]))
        if place_sell_order(s, qty):
            sell_from_portfolio_backdated(s, qty, cp, current_date)
            logging.info(f"Sold {qty} {s} at {cp:.2f} (gain {gain:.2f}%)")

    # --- ENTRY CONDITIONS ---

    for s in fno_stocks:
        cp = 0
        qty = 0
        if s not in portfolio and should_buy(s):
            cp = get_current_price(s, avoidCache=True)
            
            if cp is not None and isinstance(cp, (int, float)) and cp > 0:
                qty = int(15000 / cp)
                if qty > 0 and place_buy_order(s, qty):
                    add_to_portfolio(s, cp, qty, current_date)
                    logging.info(f"Bought {qty} of {s} at {cp:.2f}")
def backfill_all_fno(min_rows=20, sleep_between=0.5):
    fno = fetch_nifty_fno_stocks()
    if not fno:
        logging.error("No F&O list for backfill")
        return
    for sym in fno:
        sym = sym.strip()
        if not sym.find('NIFTY') == -1:
           continue
        # sanitize (remove $ if present)
        if sym.startswith('$'):
            sym = sym.lstrip('$')
        if not sym.endswith('.NS'):
            sym = sym + '.NS'
        success = ensure_price_data(sym, min_rows=min_rows)
        logging.info(f"Backfill {sym} -> rows={rows_in_db(sym)} success={success}")
        time.sleep(sleep_between)

# === MAIN ===
if __name__ == '__main__':
    
    init_db()
    #backfill_all_fno(min_rows=20, sleep_between=0.4)
    parser = argparse.ArgumentParser()
    parser.add_argument('--viewpf', action='store_true')
    parser.add_argument('--exits', action='store_true')
    parser.add_argument('--pnl', action='store_true')
    parser.add_argument('--stats', nargs='?', const='1')
    args = parser.parse_args()

    if args.viewpf:
        pass  # reuse existing show_portfolio if needed
    elif args.exits:
        pass  # reuse existing show_exits if needed
    elif args.pnl:
        pass  # reuse existing show_pnl if needed
    elif args.stats is not None:
        pass  # reuse existing show_trading_stats if needed

    try:
        check_and_trade()
    except Exception as e:
        logging.error(f"Strategy run failed: {e}")
