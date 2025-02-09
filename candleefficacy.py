import pandas as pd
import numpy as np
from tabulate import tabulate

# ---------- Helper Functions ----------
def calculate_body_size(row):
    return abs(row['Close'] - row['Open'])

def calculate_upper_shadow(row):
    return row['High'] - max(row['Open'], row['Close'])

def calculate_lower_shadow(row):
    return min(row['Open'], row['Close']) - row['Low']

def calculate_atr(df, period=14):
    high_low = df['High'] - df['Low']
    high_close = abs(df['High'] - df['Close'].shift())
    low_close = abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    return true_range.rolling(period).mean()

def identify_hammer(row, body_avg):
    body_size = calculate_body_size(row)
    upper_shadow = calculate_upper_shadow(row)
    lower_shadow = calculate_lower_shadow(row)
    
    return lower_shadow > 2 * body_size and upper_shadow < body_size * 0.5

def identify_shooting_star(row, body_avg):
    body_size = calculate_body_size(row)
    upper_shadow = calculate_upper_shadow(row)
    lower_shadow = calculate_lower_shadow(row)
    
    return upper_shadow > 2 * body_size and lower_shadow < body_size * 0.5

def identify_engulfing(curr_row, prev_row):
    curr_body_size = calculate_body_size(curr_row)
    prev_body_size = calculate_body_size(prev_row)
    
    if curr_row['Close'] > curr_row['Open'] and prev_row['Close'] < prev_row['Open']:
        if curr_row['Close'] > prev_row['Open'] and curr_row['Open'] < prev_row['Close'] and curr_body_size > prev_body_size:
            return 'Bullish'
    
    if curr_row['Close'] < curr_row['Open'] and prev_row['Close'] > prev_row['Open']:
        if curr_row['Close'] < prev_row['Open'] and curr_row['Open'] > prev_row['Close'] and curr_body_size > prev_body_size:
            return 'Bearish'
    
    return None

def identify_piercing_dark_cloud(curr_row, prev_row):
    prev_mid = (prev_row['Open'] + prev_row['Close']) / 2
    
    if prev_row['Close'] < prev_row['Open'] and curr_row['Close'] > curr_row['Open']:
        if curr_row['Open'] < prev_row['Low'] and curr_row['Close'] > prev_mid:
            return 'Piercing'
    
    if prev_row['Close'] > prev_row['Open'] and curr_row['Close'] < curr_row['Open']:
        if curr_row['Open'] > prev_row['High'] and curr_row['Close'] < prev_mid:
            return 'Dark Cloud'
    
    return None

def calculate_gain_loss(df, idx, pattern_type, atr):
    entry = df.iloc[idx]['Close']
    stop_loss = None
    
    if pattern_type in ['Hammer', 'Bullish Engulfing', 'Piercing']:
        stop_loss = df.iloc[idx]['Low'] - atr[idx]
        for i in range(idx + 1, len(df)):
            if df.iloc[i]['Low'] <= stop_loss:
                return ((stop_loss - entry) / entry) * 100
            if df.iloc[i]['High'] >= entry * 1.02:
                return 2.0
    else:
        stop_loss = df.iloc[idx]['High'] + atr[idx]
        for i in range(idx + 1, len(df)):
            if df.iloc[i]['High'] >= stop_loss:
                return ((entry - stop_loss) / entry) * 100
            if df.iloc[i]['Low'] <= entry * 0.98:
                return 2.0
    
    last_close = df.iloc[-1]['Close']
    return ((last_close - entry) / entry) * 100 if pattern_type in ['Hammer', 'Bullish Engulfing', 'Piercing'] else ((entry - last_close) / entry) * 100

# ---------- Pattern Analysis ----------
def analyze_patterns(df):
    df['Body_Size'] = df.apply(calculate_body_size, axis=1)
    body_avg = df['Body_Size'].mean()
    atr = calculate_atr(df)
    
    patterns = []
    
    for i in range(1, len(df)):
        curr_row = df.iloc[i]
        prev_row = df.iloc[i-1]
        date = curr_row.name
        
        if identify_hammer(curr_row, body_avg):
            gain_loss = calculate_gain_loss(df, i, 'Hammer', atr)
            patterns.append({'Date': date, 'Pattern': 'Hammer', 'Gain/Loss %': float(gain_loss)})
        
        if identify_shooting_star(curr_row, body_avg):
            gain_loss = calculate_gain_loss(df, i, 'Shooting Star', atr)
            patterns.append({'Date': date, 'Pattern': 'Shooting Star', 'Gain/Loss %': float(gain_loss)})
        
        engulfing = identify_engulfing(curr_row, prev_row)
        if engulfing:
            gain_loss = calculate_gain_loss(df, i, f'{engulfing} Engulfing', atr)
            patterns.append({'Date': date, 'Pattern': f'{engulfing} Engulfing', 'Gain/Loss %': float(gain_loss)})
        
        pierce_dark = identify_piercing_dark_cloud(curr_row, prev_row)
        if pierce_dark:
            gain_loss = calculate_gain_loss(df, i, pierce_dark, atr)
            patterns.append({'Date': date, 'Pattern': pierce_dark, 'Gain/Loss %': float(gain_loss)})
    
    result_df = pd.DataFrame(patterns)

    # Debug: Check Data Types
    print("Data Types Before Sorting:")
    print(result_df.dtypes)

    # Ensure sorting
    return result_df.sort_values(by="Gain/Loss %", ascending=False, ignore_index=True)

# ---------- Formatting Results ----------
def format_results(results):
    pattern_stats = []
    
    for pattern in results['Pattern'].unique():
        pattern_data = results[results['Pattern'] == pattern]
        total_signals = len(pattern_data)
        profitable_trades = len(pattern_data[pattern_data['Gain/Loss %'] > 0])
        success_rate = (profitable_trades / total_signals * 100) if total_signals > 0 else 0
        avg_gain_loss = pattern_data['Gain/Loss %'].mean()
        
        pattern_stats.append({
            'Pattern': pattern,
            'Total Signals': total_signals,
            'Profitable Trades': profitable_trades,
            'Success Rate %': round(success_rate, 2),
            'Avg Gain/Loss %': round(avg_gain_loss, 2)
        })
    
    signals_table = results.copy()
    signals_table['Date'] = signals_table['Date'].dt.strftime('%Y-%m-%d')
    signals_table['Gain/Loss %'] = signals_table['Gain/Loss %'].astype(float).round(2)

    # Sorting explicitly
    return pd.DataFrame(pattern_stats), signals_table.sort_values(by="Gain/Loss %", ascending=False, ignore_index=True)

# ---------- Load & Process Data ----------
df = pd.read_csv('nifty.csv', parse_dates=['Date'])
df.rename(columns=str.strip, inplace=True)
df.set_index('Date', inplace=True)

# Analyze patterns
results = analyze_patterns(df)

if not results.empty:
    pattern_stats, signals_table = format_results(results)

    print("\nPattern-wise Statistics:")
    print(tabulate(pattern_stats, headers='keys', tablefmt='grid', showindex=False))

    print("\nDetailed Signals:")
    print(tabulate(signals_table, headers='keys', tablefmt='grid', showindex=False))
else:
    print("No patterns found in the given date range.")
