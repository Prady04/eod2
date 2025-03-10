from flask import Flask, request, jsonify,render_template
import pandas as pd
import yfinance as yf
import sqlite3
import datetime
import json

app = Flask(__name__)

EOD_DB = "eod_data.db"
INTRADAY_DB = "intraday_data.db"
HOLIDAY_FILE = "holidays.json"
SYMBOLS = ["AAPL", "MSFT", "GOOGL"]  # Modify as needed

# Load holidays
try:
    with open(HOLIDAY_FILE, "r") as f:
        HOLIDAYS = json.load(f)
except FileNotFoundError:
    HOLIDAYS = []

# Function to fetch and store data


def fetch_store_data(symbol, interval, db_name, table_name):
    conn = sqlite3.connect(db_name)

    # Fetch data
    data = yf.download(symbol, interval=interval, progress=False, auto_adjust=False)

    if not data.empty:
        data.reset_index(inplace=True)

        # Print columns before processing
        print("\n🔍 Columns BEFORE flattening MultiIndex:")
        print(data.columns)

        # Flatten MultiIndex columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

        # Print columns after flattening
        print("\n🔍 Columns AFTER flattening MultiIndex:")
        print(data.columns)

        # Rename Datetime column if it exists
        rename_map = {"Datetime_": "datetime", "Date_": "datetime"}
        for old_col, new_col in rename_map.items():
            if old_col in data.columns:
                data.rename(columns={old_col: new_col}, inplace=True)

        # Print final columns before accessing datetime
        print("\n✅ Final DataFrame Columns:", data.columns)

        # Ensure the datetime column exists before accessing it
        if "datetime" not in data.columns:
            raise ValueError("⚠️ 'datetime' column is missing after renaming!")

        data["datetime"] = data["datetime"].astype(str)
        data["symbol"] = symbol  

        # Standardize column names
        expected_columns = ["datetime", "symbol", "Open_AAPL", "High_AAPL", "Low_AAPL", "Close_AAPL", "Volume_AAPL"]
        renamed_columns = ["datetime", "symbol", "open", "high", "low", "close", "volume"]

        # Ensure expected columns exist dynamically
        data = data[[col for col in expected_columns if col in data.columns]]
        data.columns = renamed_columns[:len(data.columns)]

        # Debugging print
        print("\n✅ Final DataFrame Ready for Database:", data.head())

        # Insert into database
        data.to_sql(table_name, conn, if_exists="append", index=False)

    conn.commit()
    conn.close()







# Fetch data from SQLite
def get_stock_data(symbol):
    conn = sqlite3.connect("eod_data.db")  # Update with your actual database path
    df = pd.read_sql("SELECT * FROM eod WHERE symbol = ?", conn, params=(symbol,))
    conn.close()
    return df

# Route to display HTML
@app.route("/")
def index():
    return render_template("index.html")

# API route to fetch stock data
@app.route("/data/<symbol>")
def fetch_stock_data(symbol):
    df = get_stock_data(symbol)
    return jsonify(df.to_dict(orient="records"))  # Returns JSON data


@app.route("/fetch", methods=["GET"])
def fetch_data():
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    if today in HOLIDAYS:
        return jsonify({"message": "Today is a holiday. No data fetched."})
    
    for symbol in SYMBOLS:
        fetch_store_data(symbol, "1d", EOD_DB, "eod")
        fetch_store_data(symbol, "15m", INTRADAY_DB, "intraday")
    
    return jsonify({"message": "Data fetched successfully"})

@app.route("/add_holiday", methods=["POST"])
def add_holiday():
    data = request.json
    date = data.get("date")
    
    if date and date not in HOLIDAYS:
        HOLIDAYS.append(date)
        with open(HOLIDAY_FILE, "w") as f:
            json.dump(HOLIDAYS, f)
        return jsonify({"message": "Holiday added."})
    return jsonify({"message": "Invalid or duplicate holiday."})

if __name__ == "__main__":
    import sqlite3
  
    conn = sqlite3.connect("eod_data.db")
    cursor = conn.cursor()
  
    cursor.execute("PRAGMA table_info(eod)")
    columns = cursor.fetchall()
    conn.close()

    print(columns)
    app.run(debug=True)
