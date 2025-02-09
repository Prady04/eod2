import yfinance as yf
import pandas as pd
import sqlite3
from datetime import datetime
from flask import Flask, render_template, request, jsonify

DB_FILE = 'engulfing_patterns.db'
app = Flask(__name__)

# ---------------------- DATABASE FUNCTIONS ----------------------

def create_database():
    """Creates the database and tables if they do not exist."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS engulfing_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        stock TEXT,
                        date TEXT,
                        pattern TEXT,
                        return_pct REAL,
                        last_updated TEXT
                      )''')
    
    cursor.execute('''CREATE TABLE IF NOT EXISTS stock_updates (
                        stock TEXT PRIMARY KEY,
                        last_updated TEXT
                      )''')
    
    conn.commit()
    conn.close()
def get_last_updated_date(ticker):
    """Retrieves the last updated date for a given stock from the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("SELECT last_updated FROM stock_updates WHERE stock = ?", (ticker,))
    result = cursor.fetchone()
    
    conn.close()
    return result[0] if result else None
def store_results(results):
    """Stores detected engulfing patterns in the database."""
    if not results:
        return
    
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.executemany("INSERT INTO engulfing_results (stock, date, pattern, return_pct, last_updated) VALUES (?, ?, ?, ?, ?)",
                       [(stock, date, pattern, return_pct, datetime.now().strftime('%Y-%m-%d %H:%M:%S')) for stock, date, pattern, return_pct in results])
    
    conn.commit()
    conn.close()

# ---------------------- STOCK ANALYSIS FUNCTIONS ----------------------
def update_last_updated_date(ticker, last_date):
    """Updates the last checked date for a stock in the database."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    cursor.execute("INSERT INTO stock_updates (stock, last_updated) VALUES (?, ?) ON CONFLICT(stock) DO UPDATE SET last_updated = ?",
                   (ticker, last_date, last_date))
    
    conn.commit()
    conn.close()

def fetch_stock_data(ticker):
    """Fetches historical stock data for a given ticker."""
    
    last_date = get_last_updated_date(ticker)
    period = '20d' if last_date is None else 'max'
    
    df = yf.download(ticker+".NS", period=period, interval='1d')

    if df.empty:
        return None
    if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0) 
    
    required_columns = ['Open', 'Close']
    if any(col not in df.columns for col in required_columns):
        return None

    df.dropna(subset=required_columns, inplace=True)
    df.sort_index(inplace=True)
    
    df.insert(0, "Date", df.index)  # Explicitly create 'Date' column from index
    df.reset_index(drop=True, inplace=True)  # Now reset index without losing 'Date'
    
    # Flatten column names if multi-index is present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]

    df['Date'] = df['Date'].astype(str)  # Convert dates to string format
    if last_date:
        df = df[df.index > pd.to_datetime(last_date)]
    return df
    



    
    # Ensure 'Date' column exists before reset_index
    df.insert(0, "Date", df.index)  # Explicitly create 'Date' column from index
    df.reset_index(drop=True, inplace=True)  # Now reset index without losing 'Date'
    
    # Flatten column names if multi-index is present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col) if isinstance(col, tuple) else col for col in df.columns]

    df['Date'] = df['Date'].astype(str)  # Convert dates to string format
    return df

def identify_engulfing(curr_row, prev_row):
    """Detects Bullish or Bearish Engulfing pattern."""
    curr_close = curr_row['Close']
    curr_open = curr_row['Open']
    prev_close = prev_row['Close']
    prev_open = prev_row['Open']
    
    curr_body_size = abs(curr_close - curr_open)
    prev_body_size = abs(prev_close - prev_open)
    
    # Bullish Engulfing
    if (curr_close > curr_open and prev_close < prev_open and
        curr_close >= prev_open and curr_open <= prev_close and
        curr_body_size > prev_body_size):
        return 'Bullish Engulfing'
    
    # Bearish Engulfing
    if (curr_close < curr_open and prev_close > prev_open and
        curr_close <= prev_open and curr_open >= prev_close and
        curr_body_size > prev_body_size):
        return 'Bearish Engulfing'
    
    return None

def analyze_engulfing(df, ticker):
    """Analyzes stock data for Engulfing patterns."""
    results = []
    
    if len(df) < 2:
        return results  

    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i - 1]
        curr_date = curr_row['Date']

        pattern = identify_engulfing(curr_row, prev_row)
        if pattern:
            latest_close = df['Close'].iloc[-1]
            pattern_close = curr_row['Close']
            return_pct = ((latest_close - pattern_close) / pattern_close) * 100
            
            results.append((ticker, curr_date, pattern, round(return_pct, 2)))
    
    return results

# ---------------------- FLASK WEB DASHBOARD ----------------------

@app.route('/')
def home():
    """Renders the web dashboard."""
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT stock, date, pattern, return_pct FROM engulfing_results ORDER BY date DESC", conn)
    conn.close()
    return render_template('dashboard.html', data=df.to_dict(orient='records'))

@app.route('/chart/<stock>')
def stock_chart(stock):
    """Renders the stock chart page."""
    df = fetch_stock_data(stock)
    if df is None:
        return f"No data available for {stock}"

    return render_template('chart.html', stock=stock, data=df.to_dict(orient='records'))

# ---------------------- MAIN EXECUTION ----------------------

def main():
    create_database()
    
    stock_list = pd.read_csv('n500.csv')['Symbol'][:5]

    for ticker in stock_list:
        print(f"Analyzing {ticker}...")
        try:
            df = fetch_stock_data(ticker)
            if df is None:
                continue

            results = analyze_engulfing(df, ticker)
            if results:
                store_results(results)
        except Exception as e:
            print(f"Error processing {ticker}: {str(e)}")

if __name__ == "__main__":
    main()
    app.run(debug=True)
