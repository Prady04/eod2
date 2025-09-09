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

# Logging handlers (file + console)
handlers = []
try:
    handlers = [logging.FileHandler('./src/trading.log'), logging.StreamHandler()]
except Exception:
    handlers = [logging.StreamHandler()]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=handlers
)

PORTFOLIO_FILE = "portfolio.json"
DB_FILE = "market_data.db"
NIFTY50_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty50list.csv"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
}

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

# === NIFTY 50 STOCKS ===

def fetch_nifty50_stocks():
    try:
        response = requests.get(NIFTY50_URL, headers=HEADERS, timeout=10)
        response.raise_for_status()
        csv_data = io.StringIO(response.text)
        df = pd.read_csv(csv_data)
        if 'Symbol' not in df.columns:
            logging.error("Nifty CSV did not contain 'Symbol' column")
            return []
        return [s.strip() + ".NS" for s in df['Symbol'].astype(str)]
    except Exception as e:
        logging.warning(f"Could not fetch Nifty50 list: {e}")
        return []

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


def sell_from_portfolio(symbol, quantity_to_sell, sell_price):
    load_portfolio()
    if symbol not in portfolio:
        logging.warning(f"Tried to sell {symbol} but it's not in portfolio")
        return False

    remaining = quantity_to_sell
    updated = []
    for entry in portfolio[symbol]:
        if remaining <= 0:
            updated.append(entry)
            continue
        if entry['quantity'] <= remaining:
            remaining -= entry['quantity']
            # drop this entry (sold fully)
        else:
            entry['quantity'] -= remaining
            remaining = 0
            updated.append(entry)

    if updated:
        portfolio[symbol] = updated
    else:
        del portfolio[symbol]
    save_portfolio()
    return True

# Backdated sell records into exits.json
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


def fetch_and_store_price_data(symbol, start_date=None, period='1d'):
    """
    Tries to fetch price data using yfinance if available. If yfinance is not installed
    this function logs a warning and returns without inserting.
    """
    try:
        import yfinance as yf
    except Exception:
        logging.debug("yfinance not installed — fetch_and_store_price_data is a no-op")
        return

    try:
        if period == '20d':
            df = yf.Ticker(symbol).history(period=period)
            if df is None or df.empty:
                return
            df.index = pd.to_datetime(df.index)
            with sqlite3.connect(DB_FILE) as conn:
                for idx, row in df.iterrows():
                    date_str = idx.strftime('%Y-%m-%d')
                    conn.execute("INSERT OR IGNORE INTO price_history (symbol,date,close,volume) VALUES (?,?,?,?)",
                                 (symbol, date_str, float(row['Close']), float(row.get('Volume', 0))))
        else:
            # single day
            hist = yf.Ticker(symbol).history(period='1d')
            if hist is None or hist.empty:
                return
            last = hist['Close'][-1]
            with sqlite3.connect(DB_FILE) as conn:
                conn.execute("INSERT OR IGNORE INTO price_history (symbol,date,close,volume) VALUES (?,?,?,?)",
                             (symbol, datetime.now().strftime('%Y-%m-%d'), float(last), 0))
    except Exception as e:
        logging.warning(f"Failed to fetch/store price for {symbol}: {e}")


def get_current_price(symbol, avoidCache=False):
    if avoidCache:
        fetch_and_store_price_data(symbol, period='1d')
    with sqlite3.connect(DB_FILE) as conn:
        r = conn.execute("SELECT close FROM price_history WHERE symbol = ? ORDER BY date DESC LIMIT 1", (symbol,)).fetchone()
        return r[0] if r else None

# === PORTFOLIO METRICS ===

def get_total_portfolio_value():
    load_portfolio()
    total = 0.0
    for symbol, entries in portfolio.items():
        price = get_current_price(symbol)
        if price is None:
            continue
        for e in entries:
            total += e['quantity'] * price
    return total


def get_average_holding_period():
    if not os.path.exists('exits.json'):
        return 0
    with open('exits.json','r') as f:
        exits = json.load(f)
    days = []
    for e in exits:
        try:
            d1 = datetime.strptime(e['purchase_date'],'%Y-%m-%d')
            d2 = datetime.strptime(e['sell_date'],'%Y-%m-%d')
            days.append((d2-d1).days)
        except Exception:
            pass
    return sum(days)/len(days) if days else 0


def get_total_invested():
    load_portfolio()
    invested = 0.0
    for symbol, entries in portfolio.items():
        for e in entries:
            invested += e['quantity'] * e['purchase_price']
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)
        for e in exits:
            invested += e['quantity'] * e['purchase_price']
    return invested


def get_mtm_unrealized():
    load_portfolio()
    mtm = 0.0
    for symbol, entries in portfolio.items():
        price = get_current_price(symbol)
        if price is None:
            continue
        for e in entries:
            mtm += (price - e['purchase_price']) * e['quantity']
    return mtm


def get_portfolio_returns():
    load_portfolio()
    invested = 0.0
    current_val = 0.0
    earliest = None
    for symbol, entries in portfolio.items():
        price = get_current_price(symbol)
        if price is None:
            continue
        for e in entries:
            invested += e['quantity'] * e['purchase_price']
            current_val += e['quantity'] * price
            d = datetime.strptime(e['purchase_date'],'%Y-%m-%d')
            if earliest is None or d < earliest:
                earliest = d
    realized = 0.0
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)
        for e in exits:
            invested += e['quantity'] * e['purchase_price']
            realized += e['quantity'] * e['sell_price']
            d = datetime.strptime(e['purchase_date'],'%Y-%m-%d')
            if earliest is None or d < earliest:
                earliest = d
    if invested == 0:
        return 0.0,0.0
    total_val = current_val + realized
    cum = (total_val - invested)/invested
    today = datetime.now()
    days = max((today - earliest).days,1) if earliest else 1
    ann = (1+cum)**(365/days)-1
    return cum, ann

# XNPV / XIRR helpers
def xnpv(rate, cashflows):
    t0 = min(cf[0] for cf in cashflows)
    return sum(cf[1]/((1+rate)**((cf[0]-t0).days/365.0)) for cf in cashflows)


def get_portfolio_xirr():
    cashflows = []
    load_portfolio()
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)
    else:
        exits = []
    for symbol, entries in portfolio.items():
        for e in entries:
            cashflows.append((datetime.strptime(e['purchase_date'],'%Y-%m-%d'), -e['quantity']*e['purchase_price']))
    for e in exits:
        cashflows.append((datetime.strptime(e['sell_date'],'%Y-%m-%d'), e['quantity']*e['sell_price']))
    if not cashflows:
        return 0.0
    try:
        r = newton(lambda rr: xnpv(rr, cashflows), 0.1)
        return r
    except Exception as e:
        logging.error(f"XIRR failed: {e}")
        return 0.0

# === SIMPLE STRATEGY PLACEHOLDERS ===

def place_sell_order(symbol, quantity):
    logging.info(f"(sim) SELL {quantity} {symbol}")
    return True

def place_buy_order(symbol, quantity):
    logging.info(f"(sim) BUY {quantity} {symbol}")
    return True

# 20-day MA helper using DB (calls fetch_and_store_price_data for 20d if needed)

def get_20_day_ma(symbol):
    last = get_last_date(symbol)
    start = '2022-01-01' if last is None else last
    fetch_and_store_price_data(symbol, start_date=start, period='20d')
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute("SELECT close FROM price_history WHERE symbol=? ORDER BY date DESC LIMIT 20", (symbol,)).fetchall()
        closes = [r[0] for r in rows]
        return sum(closes)/len(closes) if len(closes)==20 else None

# === STRATEGY (keeps previous behavior but relies on DB price history) ===

def check_and_trade():
    ist = pytz.timezone('Asia/Kolkata')
    now = datetime.now(ist)
    current_date = now.strftime('%Y-%m-%d')
    logging.info(f"Running strategy at {now.strftime('%Y-%m-%d %H:%M:%S')}")
    if now.weekday() >= 5 or current_date in NSE_HOLIDAYS:
        logging.info("Market closed. Skipping.")
        return
    nifty50 = fetch_nifty50_stocks()
    if not nifty50:
        logging.info("No Nifty50 list — skipping buy/sell checks.")
        return
    load_portfolio()
    # warn about non-nifty holdings
    for s in list(portfolio.keys()):
        if s not in nifty50:
            logging.warning(f"Holding {s} not in Nifty50")
    # sell winners >5%
    to_sell = []
    for s, entries in portfolio.items():
        for e in entries:
            cp = get_current_price(s, avoidCache=True)
            if cp is None:
                continue
            gain = (cp - e['purchase_price'])/e['purchase_price']*100
            if gain >= 5:
                to_sell.append((s, cp, gain))
    for s, cp, gain in to_sell:
        qty = sum(x['quantity'] for x in portfolio.get(s,[]))
        if place_sell_order(s, qty):
            sell_from_portfolio_backdated(s, qty, cp, current_date)
            logging.info(f"Sold {qty} {s} with gain {gain:.2f}%")
    # pick farthest below 20d MA
    candidates = []
    for s in nifty50:
        if s in portfolio:
            continue
        cp = get_current_price(s)
        ma = get_20_day_ma(s)
        if cp is None or ma is None:
            continue
        deviation = (cp - ma)/ma*100
        candidates.append((deviation, s, cp))
    if candidates:
        candidates.sort()
        deviation, sym, price = candidates[0]
        qty = int(15000/price) if price>0 else 0
        if qty>0 and place_buy_order(sym, qty):
            add_to_portfolio(sym, price, qty, current_date)
            logging.info(f"Bought {qty} of {sym} at {price:.2f}")

# === DISPLAY / REPORT FUNCTIONS ===

def show_portfolio():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE,'r') as f:
            pf = json.load(f)
        rows = []
        for s,entries in pf.items():
            for e in entries:
                rows.append([s, e['quantity'], f"{e['purchase_price']:,.2f}", e['purchase_date']])
        print('📊 Current Portfolio')
        print(tabulate(rows, headers=["Symbol","Qty","Buy Price","Date"], tablefmt='fancy_grid'))
    else:
        print('No portfolio file')


def show_exits():
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)
        rows = []
        for t in exits:
            pnl = (t['sell_price']-t['purchase_price'])*t['quantity']
            rows.append([t['symbol'], t['quantity'], f"{t['purchase_price']:,.2f}", t['purchase_date'], f"{t['sell_price']:,.2f}", t['sell_date'], f"{pnl:,.2f}"])
        print('✅ Exited Trades')
        print(tabulate(rows, headers=["Symbol","Qty","Buy Price","Buy Date","Sell Price","Sell Date","PnL"], tablefmt='fancy_grid'))
    else:
        print('No exited trades')


def show_pnl():
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE,'r') as f:
            pf = json.load(f)
        rows = []
        invested=0.0; current=0.0; total_pnl=0.0
        for s, entries in pf.items():
            for e in entries:
                qty = e['quantity']; bp = e['purchase_price']
                price = get_current_price(s)
                if price is None:
                    price=0.0
                inv = qty*bp; val=qty*price; pnl=val-inv
                rows.append([s, qty, f"{bp:,.2f}", f"{price:,.2f}", f"{inv:,.2f}", f"{val:,.2f}", f"{pnl:,.2f}"])
                invested += inv; current += val; total_pnl += pnl
        print('💹 Stock-wise P&L')
        print(tabulate(rows, headers=["Symbol","Qty","Buy Price","Curr Price","Invested","Curr Value","PnL"], tablefmt='fancy_grid'))
        booked = get_booked_profit(); mtm = total_pnl
        print('📊 Portfolio Summary')
        print(tabulate([[f"{current:,.2f}", f"{invested:,.2f}", f"{booked:,.2f}", f"{mtm:,.2f}"]], headers=["Total Value (Open)","Invested","Booked Profit","Unrealized (MTM)"], tablefmt='fancy_grid'))
    else:
        print('No portfolio file')

# === EQUITY CURVE & STATS ===

def _get_price_on_or_before(symbol, date_str):
    with sqlite3.connect(DB_FILE) as conn:
        r = conn.execute("SELECT close FROM price_history WHERE symbol = ? AND date <= ? ORDER BY date DESC LIMIT 1", (symbol, date_str)).fetchone()
        return r[0] if r else None


def build_portfolio_equity_curve():
    events = []
    earliest = None
    if os.path.exists(PORTFOLIO_FILE):
        with open(PORTFOLIO_FILE,'r') as f:
            pf = json.load(f)
        for sym, entries in pf.items():
            for e in entries:
                events.append((e['purchase_date'], 'buy', sym, int(e['quantity']), float(e['purchase_price'])))
                d = datetime.strptime(e['purchase_date'],'%Y-%m-%d')
                if earliest is None or d < earliest:
                    earliest = d
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)
        for e in exits:
            events.append((e['purchase_date'],'buy',e['symbol'],int(e['quantity']),float(e['purchase_price'])))
            events.append((e['sell_date'],'sell',e['symbol'],int(e['quantity']),float(e['sell_price'])))
            for ds in (e['purchase_date'], e['sell_date']):
                d = datetime.strptime(ds,'%Y-%m-%d')
                if earliest is None or d < earliest:
                    earliest = d
    if earliest is None:
        return [], []
    start = earliest.strftime('%Y-%m-%d')
    end = datetime.now().strftime('%Y-%m-%d')
    symbols = list({ev[2] for ev in events})
    dates = _all_price_dates_for_symbols(symbols, start, end)
    event_dates = sorted({ev[0] for ev in events})
    all_dates = sorted(set(dates) | set(event_dates))
    if not all_dates:
        d = earliest; all_dates=[]
        while d <= datetime.now():
            all_dates.append(d.strftime('%Y-%m-%d'))
            d += timedelta(days=1)
    events_by_date = {}
    for d, typ, sym, qty, price in events:
        events_by_date.setdefault(d, []).append((typ, sym, qty, price))
    holdings = {}
    cash = 0.0
    equity_curve = []
    for d in all_dates:
        for ev in events_by_date.get(d, []):
            typ, sym, qty, price = ev
            if typ == 'buy':
                holdings[sym] = holdings.get(sym,0) + qty
                cash -= qty*price
            else:
                holdings[sym] = holdings.get(sym,0) - qty
                if holdings[sym] == 0:
                    del holdings[sym]
                cash += qty*price
        mv = 0.0
        for sym, qty in list(holdings.items()):
            if qty == 0: continue
            p = _get_price_on_or_before(sym, d)
            if p is None: continue
            mv += qty * p
        equity = cash + mv
        equity_curve.append((d, equity))
    filtered = [(d,v) for d,v in equity_curve if v is not None]
    if not filtered:
        return [d for d,_ in equity_curve], [v for _,v in equity_curve]
    return [d for d,_ in filtered], [v for _,v in filtered]


def _all_price_dates_for_symbols(symbols, start_date=None, end_date=None):
    if not symbols:
        return []
    placeholders = ','.join('?' for _ in symbols)
    sql = f"SELECT DISTINCT date FROM price_history WHERE symbol IN ({placeholders})"
    params = list(symbols)
    if start_date:
        sql += ' AND date >= ?'
        params.append(start_date)
    if end_date:
        sql += ' AND date <= ?'
        params.append(end_date)
    sql += ' ORDER BY date'
    with sqlite3.connect(DB_FILE) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return [r[0] for r in rows]


def compute_max_drawdown_from_equity(equity_list):
    """Calculate max drawdown from an equity curve (cumulative PnL series)."""
    '''if equity_curve.empty:
        return "N/A"
    running_max = equity_curve.cummax()
    drawdown = (equity_curve - running_max) / running_max
    max_dd = drawdown.min() * 100  # in %
    return f"{max_dd:.2f}%"'''


import yfinance as yf
import logging

def get_benchmark_returns(start, end):
    """Fetch CAGR for NIFTY50 and NIFTY500 between start and end dates."""
    try:
        nifty50 = yf.download("^NSEI", start=start, end=end, auto_adjust=False)["Adj Close"]
    except Exception as e:
        logging.error(f"Error fetching NIFTY50 data: {e}")
        nifty50 = None
    try:
        nifty500 = yf.download("^CRSLDX", start=start, end=end, auto_adjust=False)["Adj Close"]
    except Exception as e:
        logging.error(f"Error fetching NIFTY500 data: {e}")
        nifty500 = None

    def cagr(series):
        if series is None or series.empty:
            return "N/A"
        start_val = float(series.iloc[0])
        end_val = float(series.iloc[-1])
        total_return = end_val / start_val - 1
        years = (series.index[-1] - series.index[0]).days / 365.0
        if years <= 0:
            return "N/A"
        cagr_val = ((1 + total_return) ** (1/years)) - 1
        return cagr_val * 100

    nifty50_cagr = cagr(nifty50)
    nifty500_cagr = cagr(nifty500)

    return (
        f"{nifty50_cagr:.2f}%" if isinstance(nifty50_cagr, float) else nifty50_cagr,
        f"{nifty500_cagr:.2f}%" if isinstance(nifty500_cagr, float) else nifty500_cagr,
    )

def get_portfolio_cagr(trades: pd.DataFrame):
    """Compute portfolio CAGR from closed trades DataFrame."""
    if trades.empty:
        return "N/A"
    start_date = pd.to_datetime(trades["Buy Date"].min())
    end_date = pd.to_datetime(trades["Sell Date"].max())
    invested = (trades["Qty"] * trades["Buy Price"]).sum()
    final_value = invested + trades["PnL"].sum()
    total_return = final_value / invested - 1
    years = (end_date - start_date).days / 365.0
    if years <= 0:
        return "N/A"
    cagr_val = ((1 + total_return) ** (1 / years)) - 1
    return f"{cagr_val * 100:.2f}%"




# === STATS / REPORT ===
def get_booked_profit():
    if not os.path.exists('exits.json'):
        return 0.0
    with open('exits.json', 'r') as f:
        exits = json.load(f)
    booked = 0.0
    for e in exits:
        booked += (e['sell_price'] - e['purchase_price']) * e['quantity']
    return booked

def show_trading_stats():
    load_portfolio()
    invested_total = get_total_invested()
    current_value = get_total_portfolio_value()
    booked_pnl = get_booked_profit()
    mtm_pnl = get_mtm_unrealized()

    pnl_list = []
    exits = []
    if os.path.exists('exits.json'):
        with open('exits.json','r') as f:
            exits = json.load(f)

    trade_count = winning_trades = losing_trades = 0
    largest_win = float('-inf')
    largest_loss = 0.0
    holding_periods = []
    win_streak = loss_streak = max_win_streak = max_loss_streak = 0

    exit_rows = []
    for t in exits:
        pnl = (t['sell_price'] - t['purchase_price']) * t['quantity']
        pnl_list.append(pnl)
        trade_count += 1
        if pnl > 0:
            winning_trades += 1
            win_streak += 1
            loss_streak = 0
            max_win_streak = max(max_win_streak, win_streak)
        elif pnl < 0:
            losing_trades += 1
            loss_streak += 1
            win_streak = 0
            max_loss_streak = max(max_loss_streak, loss_streak)
            largest_loss = min(largest_loss, pnl)
        largest_win = max(largest_win, pnl)
        try:
            d1 = datetime.strptime(t['purchase_date'],'%Y-%m-%d')
            d2 = datetime.strptime(t['sell_date'],'%Y-%m-%d')
            holding_periods.append((d2-d1).days)
        except Exception:
            pass
        exit_rows.append([t['symbol'], t['quantity'], f"{t['purchase_price']:,.2f}", t['purchase_date'], f"{t['sell_price']:,.2f}", t['sell_date'], f"{pnl:,.2f}"])

    print('✅ Closed Trades (Used for Stats)')
    if exit_rows:
        print(tabulate(exit_rows, headers=["Symbol","Qty","Buy Price","Buy Date","Sell Price","Sell Date","PnL"], tablefmt='fancy_grid'))
    else:
        print('No closed trades found.')

    win_rate = (winning_trades/trade_count*100.0) if trade_count else 0.0
    avg_win = statistics.mean([p for p in pnl_list if p>0]) if any(p>0 for p in pnl_list) else 0.0
    avg_loss = statistics.mean([p for p in pnl_list if p<0]) if any(p<0 for p in pnl_list) else 0.0
    profit_factor = (sum(p for p in pnl_list if p>0)/abs(sum(p for p in pnl_list if p<0))) if any(p<0 for p in pnl_list) else float('inf')
    expectancy = (sum(pnl_list)/trade_count) if trade_count else 0.0
    sharpe = 0.0
    if len(pnl_list)>1 and statistics.pstdev(pnl_list)!=0:
        sharpe = statistics.mean(pnl_list)/statistics.pstdev(pnl_list)

    # closed-trades equity curve (cumulative realized PnL)
    closed_equity = []
    cum = 0.0
    for p in pnl_list:
        cum += p
        closed_equity.append(cum)
    max_dd_closed = 0.0
    if closed_equity:
        peak = closed_equity[0]
        for v in closed_equity:
            peak = max(peak, v)
            dd = (peak - v)/peak if peak>0 else 0.0
            max_dd_closed = max(max_dd_closed, dd)

    print('📈 Trading Performance Metrics (Closed Trades Only)')
    print(tabulate([[trade_count, f"{win_rate:.2f}%", f"{profit_factor:.2f}", f"{expectancy:,.2f}", f"{sharpe:.2f}", f"{max_dd_closed*100:.2f}%"]], headers=["Trades","Win Rate","Profit Factor","Expectancy","Sharpe","Max DD"], tablefmt='fancy_grid'))

    avg_holding = statistics.mean(holding_periods) if holding_periods else 0.0
    largest_win_display = f"{largest_win:,.2f}" if largest_win != float('-inf') else 'N/A'
    largest_loss_display = f"{largest_loss:,.2f}" if largest_loss < 0 else 'N/A'
    print('📊 Trade Behavior Metrics')
    print(tabulate([[f"{avg_holding:.1f} days", max_win_streak, max_loss_streak, largest_win_display, largest_loss_display]], headers=["Avg Holding","Longest Win Streak","Longest Loss Streak","Largest Win","Largest Loss"], tablefmt='fancy_grid'))

    # Open portfolio table
    rows = []
    load_portfolio()
    invested = 0.0; current = 0.0; unreal = 0.0
    for s, entries in portfolio.items():
        for e in entries:
            qty = e['quantity']; bp = e['purchase_price']
            price = get_current_price(s)
            if price is None: price = 0.0
            inv = qty*bp; val = qty*price; pnl = val - inv
            rows.append([s, qty, f"{bp:,.2f}", f"{price:,.2f}", f"{inv:,.2f}", f"{val:,.2f}", f"{pnl:,.2f}", f"{(pnl/inv*100) if inv else 0:.2f}%"])
            invested += inv; current += val; unreal += pnl
    print('📊 Open Portfolio (Not included in stats)')
    if rows:
        print(tabulate(rows, headers=["Symbol","Qty","Buy Price","Curr Price","Invested","Curr Value","PnL","Return %"], tablefmt='fancy_grid'))
    else:
        print('No open positions')

    print('💹 Portfolio Overview')
    print(tabulate([[f"{current:,.2f}", f"{invested:,.2f}", f"{booked_pnl:,.2f}", f"{unreal:,.2f}"]], headers=["Total Value (Open)","Invested","Booked Profit","Unrealized (MTM)"], tablefmt='fancy_grid'))

    # portfolio equity and portfolio drawdown
    dates, eq = build_portfolio_equity_curve()
    if eq:
        port_dd = compute_max_drawdown_from_equity(eq)
        print(f"📉 Portfolio Max Drawdown: {port_dd:.2f}%")
        start, end = dates[0], dates[-1]
        nifty50_cagr, nifty500_cagr = get_benchmark_returns(start, end)
        #= get_benchmark_returns(start, end)
        print('📊 Benchmark Comparison (CAGR over same period)')
        print(tabulate([["Portfolio", "TBD", nifty50_cagr,nifty500_cagr]], headers=["Series","Portfolio CAGR","NIFTY50 CAGR","NIFTY 500 CAGR"], tablefmt='fancy_grid'))
    else:
        logging.info('Not enough data for portfolio equity curve')

# === MAIN ===
if __name__ == '__main__':
    init_db()
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', nargs=2, metavar=('SYMBOL','RATIO'))
    parser.add_argument('--bonus', nargs=2, metavar=('SYMBOL','RATIO'))
    parser.add_argument('--viewpf', action='store_true')
    parser.add_argument('--exits', action='store_true')
    parser.add_argument('--pnl', action='store_true')
    parser.add_argument('--stats', nargs='?', const='1')
    args = parser.parse_args()

    if args.split:
        sym, r = args.split
        try:
            apply_split(sym, int(r))
        except Exception:
            logging.error('split failed')
    elif args.bonus:
        sym, r = args.bonus
        try:
            apply_bonus(sym, r)
        except Exception:
            logging.error('bonus failed')
    elif args.viewpf:
        show_portfolio()
    elif args.exits:
        show_exits()
    elif args.pnl:
        show_pnl()
    elif args.stats is not None:
        show_trading_stats()

    # Run strategy once per invocation (non-blocking)
    try:
        check_and_trade()
    except Exception as e:
        logging.error(f"Strategy run failed: {e}")
