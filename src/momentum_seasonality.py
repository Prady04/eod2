import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import warnings
import sqlite3
from pathlib import Path
import time
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle


# --- ADD THESE TWO LINES FOR ROBUSTNESS ---
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent memory errors
# -----------------------------------------

warnings.filterwarnings('ignore')

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

warnings.filterwarnings('ignore')

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Database configuration
DB_PATH = "C:\\temp\\stock_data.db"

def init_database():
    """Initialize SQLite database with required tables"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create table for storing stock prices
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS stock_prices (
            symbol TEXT,
            date DATE,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            PRIMARY KEY (symbol, date)
        )
    ''')
    
    # Create table for storing analysis metadata
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analysis_metadata (
            symbol TEXT PRIMARY KEY,
            last_update DATE,
            first_date DATE,
            last_date DATE,
            total_records INTEGER
        )
    ''')
    
    # Create table for seasonality analysis
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS seasonality_stats (
            symbol TEXT,
            period_type TEXT,
            period_value TEXT,
            avg_return REAL,
            median_return REAL,
            positive_count INTEGER,
            total_count INTEGER,
            win_rate REAL,
            std_dev REAL,
            PRIMARY KEY (symbol, period_type, period_value)
        )
    ''')
    
    # Create table for monthly momentum scores
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS monthly_momentum (
            symbol TEXT,
            year_month TEXT,
            positive_days INTEGER,
            total_days INTEGER,
            percentage REAL,
            monthly_return REAL,
            momentum_score REAL,
            momentum_rating TEXT,
            PRIMARY KEY (symbol, year_month)
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def get_last_date_from_db(symbol):
    """Get the last available date for a symbol from database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        SELECT last_date FROM analysis_metadata WHERE symbol = ?
    ''', (symbol,))
    
    result = cursor.fetchone()
    conn.close()
    
    if result:
        return pd.to_datetime(result[0])
    return None


def save_insights_to_file(symbol, analyzed_data, monthly_stats, monthly_seasonality, dow_seasonality, quarter_seasonality):
    """Save all insights to a text file"""
    filename = f"C:\\temp\\momentum\\{symbol}_insights.txt"
    
    with open(filename, 'w') as f:
        f.write(f"{'='*65}\n")
        f.write(f"STOCK ANALYSIS INSIGHTS FOR {symbol}\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*65}\n\n")
        
        # Overall Statistics
        f.write(f"OVERALL STATISTICS:\n")
        f.write(f"{'-'*65}\n")
        total_occurrences = analyzed_data['Close_Greater_Than_Prev'].sum()
        total_trading_days = len(analyzed_data)
        f.write(f"Total trading days: {total_trading_days}\n")
        f.write(f"Days where Close > Previous Close: {total_occurrences}\n")
        f.write(f"Percentage: {(total_occurrences/total_trading_days)*100:.2f}%\n")
        f.write(f"Average Daily Return: {analyzed_data['Daily_Return'].mean():.2f}%\n")
        f.write(f"Daily Return Volatility: {analyzed_data['Daily_Return'].std():.2f}%\n")
        f.write(f"Maximum Daily Return: {analyzed_data['Daily_Return'].max():.2f}%\n")
        f.write(f"Minimum Daily Return: {analyzed_data['Daily_Return'].min():.2f}%\n\n")
        
        # Monthly Breakdown
        f.write(f"MONTHLY BREAKDOWN WITH MOMENTUM ANALYSIS:\n")
        f.write(f"{'-'*95}\n")
        f.write(monthly_stats[['Positive_Days', 'Total_Days', 'Percentage', 
                              'Monthly_Return', 'Momentum_Score', 'Momentum_Rating']].to_string())
        f.write("\n\n")
        
        # Best and Worst Months
        best_month = monthly_stats['Momentum_Score'].idxmax()
        worst_month = monthly_stats['Momentum_Score'].idxmin()
        f.write(f"KEY INSIGHTS:\n")
        f.write(f"{'-'*65}\n")
        f.write(f"Best performing month: {best_month} (Momentum Score: {monthly_stats.loc[best_month, 'Momentum_Score']})\n")
        f.write(f"Worst performing month: {worst_month} (Momentum Score: {monthly_stats.loc[worst_month, 'Momentum_Score']})\n")
        f.write(f"Highest positive days percentage: {monthly_stats['Percentage'].idxmax()} ({monthly_stats['Percentage'].max():.2f}%)\n")
        f.write(f"Lowest positive days percentage: {monthly_stats['Percentage'].idxmin()} ({monthly_stats['Percentage'].min():.2f}%)\n\n")
        
        # Seasonality Analysis
        f.write(f"SEASONALITY ANALYSIS:\n")
        f.write(f"{'-'*65}\n")
        
        # Monthly Seasonality
        f.write(f"MONTHLY SEASONALITY:\n")
        f.write(f"{'-'*85}\n")
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        for month, value in monthly_seasonality.items():
            f.write(f"{month_names[month-1]:4} | Avg Return: {value:6.2f}%\n")
        
        # Day of Week Seasonality
        f.write(f"\nDAY OF WEEK SEASONALITY:\n")
        f.write(f"{'-'*85}\n")
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        for dow, value in dow_seasonality.items():
            f.write(f"{day_names[dow]:9} | Avg Return: {value:6.2f}%\n")
        
        # Trading Recommendations
        f.write(f"\nTRADING RECOMMENDATIONS:\n")
        f.write(f"{'-'*65}\n")
        
        # Best months to trade
        best_months = monthly_seasonality.nlargest(3).index.tolist()
        f.write(f"Best months to trade (based on historical returns): {', '.join([month_names[m-1] for m in best_months])}\n")
        
        # Best days to trade
        best_days = dow_seasonality.nlargest(2).index.tolist()
        f.write(f"Best days to trade (based on historical returns): {', '.join([day_names[d] for d in best_days])}\n")
        
        # Risk Assessment
        f.write(f"\nRISK ASSESSMENT:\n")
        f.write(f"{'-'*65}\n")
        volatility = analyzed_data['Daily_Return'].std()
        if volatility < 1:
            f.write("Low volatility stock (suitable for conservative investors)\n")
        elif volatility < 2:
            f.write("Moderate volatility stock (suitable for balanced investors)\n")
        else:
            f.write("High volatility stock (suitable for aggressive investors)\n")
        
        # Calculate maximum drawdown
        cumulative_returns = (1 + analyzed_data['Daily_Return']/100).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max * 100
        max_drawdown = drawdown.min()
        f.write(f"Maximum drawdown: {max_drawdown:.2f}%\n")
        
        # Risk warnings
        f.write(f"\nRISK WARNINGS:\n")
        f.write(f"{'-'*65}\n")
        f.write(f"Past performance does not guarantee future results.\n")
        f.write(f"This analysis is based on historical data only.\n")
        f.write(f"Always consider other factors like market conditions, news, and fundamentals.\n")
        f.write(f"Consult with a financial advisor before making investment decisions.\n")
    
    print(f"All insights saved to {filename}")
    
    
def save_to_database(symbol, stock_data):
    """Save stock data to database"""
    conn = sqlite3.connect(DB_PATH)
    
    # Prepare data for insertion
    data_to_insert = stock_data.reset_index()
    data_to_insert.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    data_to_insert['symbol'] = symbol
    data_to_insert['date'] = data_to_insert['date'].dt.strftime('%Y-%m-%d')
    
    # Insert or replace stock prices
    data_to_insert.to_sql('stock_prices', conn, if_exists='append', index=False)
    
    # Update metadata
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO analysis_metadata 
        (symbol, last_update, first_date, last_date, total_records)
        VALUES (?, ?, ?, ?, ?)
    ''', (
        symbol,
        datetime.now().strftime('%Y-%m-%d'),
        data_to_insert['date'].min(),
        data_to_insert['date'].max(),
        len(data_to_insert)
    ))
    
    conn.commit()
    conn.close()
    print(f"Saved {len(data_to_insert)} records for {symbol}")

def load_from_database(symbol):
    """Load stock data from database"""
    conn = sqlite3.connect(DB_PATH)
    
    query = '''
        SELECT date, open, high, low, close, volume
        FROM stock_prices
        WHERE symbol = ?
        ORDER BY date
    '''
    
    df = pd.read_sql_query(query, conn, parse_dates=['date'], index_col='date')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    conn.close()
    return df

def fetch_and_analyze_stock(ticker, days=365):
    """
    Fetch stock data - get only delta if data exists in DB
    """
    #symbol = ticker.replace('.NS', '')
    last_date = get_last_date_from_db(ticker)
    
    end_date = datetime.now()
    
    if last_date:
        # Fetch only new data (delta)
        start_date = last_date + timedelta(days=1)
        print(f"Found existing data for {ticker}. Fetching delta from {start_date.strftime('%Y-%m-%d')}")
        
        # Load existing data
        existing_data = load_from_database(ticker)
        
        if start_date >= end_date:
            print(f"Data is up to date for {ticker}")
            return existing_data
        
        # Fetch new data
        new_data = yf.download(ticker+'.NS', start=start_date, end=end_date, progress=False)
        
        if not new_data.empty:
            if isinstance(new_data.columns, pd.MultiIndex):
                new_data.columns = new_data.columns.droplevel(1)
            
            # Save new data to database
            save_to_database(ticker, new_data)
            
            # Combine with existing data
            combined_data = pd.concat([existing_data, new_data])
            combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
            print(f"Added {len(new_data)} new records. Total: {len(combined_data)}")
            return combined_data
        else:
            print(f"No new data available for {ticker}")
            return existing_data
    else:
        # Fetch full data
        start_date = end_date - timedelta(days=days)
        print(f"Fetching full data for {ticker} from {start_date.strftime('%Y-%m-%d')}")
        
        stock_data = yf.download(ticker+'.NS', start=start_date, end=end_date, progress=False)
        
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        # Save to database
        save_to_database(ticker, stock_data)
        
        print(f"Fetched {len(stock_data)} records")
        return stock_data

def analyze_positive_days(stock_data):
    """Analyze days where close > previous close"""
    data = stock_data.copy()
    data['Prev_Close'] = data['Close'].shift(1)
    data['Close_Greater_Than_Prev'] = data['Close'] > data['Prev_Close']
    data['Daily_Return'] = (data['Close'] - data['Prev_Close']) / data['Prev_Close'] * 100
    data = data.dropna()
    
    total_occurrences = data['Close_Greater_Than_Prev'].sum()
    total_trading_days = len(data)
    
    print(f"\nAnalysis Results:")
    print(f"Total trading days: {total_trading_days}")
    print(f"Days where Close > Previous Close: {total_occurrences}")
    print(f"Percentage: {(total_occurrences/total_trading_days)*100:.2f}%")
    
    return data

def calculate_momentum_score(data, monthly_stats):
    """
    Calculate momentum score for each month based on multiple factors
    """
    momentum_scores = []
    
    for period in monthly_stats.index:
        month_data = data[data['YearMonth'] == period]
        
        if len(month_data) == 0:
            momentum_scores.append(0)
            continue
        
        # Factor 1: Percentage of positive days (0-40 points)
        positive_pct = monthly_stats.loc[period, 'Percentage']
        positive_score = (positive_pct / 100) * 40
        
        # Factor 2: Average daily return (0-25 points, capped at ±10% daily return)
        avg_return = monthly_stats.loc[period, 'Avg_Return']
        return_score = max(-25, min(25, (avg_return / 10) * 25))
        
        # Factor 3: Price momentum - month end vs month start (0-25 points)
        month_start_price = month_data['Close'].iloc[0]
        month_end_price = month_data['Close'].iloc[-1]
        price_momentum = ((month_end_price - month_start_price) / month_start_price) * 100
        price_score = max(-25, min(25, (price_momentum / 20) * 25))
        
        # Factor 4: Consistency - lower volatility gets higher score (0-10 points)
        volatility = monthly_stats.loc[period, 'Return_Std']
        consistency_score = max(0, 10 - (volatility / 5) * 10)
        
        # Total momentum score (0-100)
        total_score = positive_score + return_score + price_score + consistency_score
        momentum_scores.append(round(total_score, 1))
    
    return momentum_scores

def monthly_breakdown(data):
    """
    Create monthly breakdown of positive days with momentum scores
    """
    # Create monthly aggregation
    data['YearMonth'] = data.index.to_period('M')
    
    monthly_stats = data.groupby('YearMonth').agg({
        'Close_Greater_Than_Prev': ['sum', 'count'],
        'Close': ['first', 'last'],
        'Daily_Return': ['mean', 'std', 'min', 'max']
    }).round(2)
    
    # Flatten column names
    monthly_stats.columns = [
        'Positive_Days', 'Total_Days', 'Month_Start_Close', 'Month_End_Close',
        'Avg_Return', 'Return_Std', 'Min_Return', 'Max_Return'
    ]
    monthly_stats['Percentage'] = (monthly_stats['Positive_Days'] / monthly_stats['Total_Days'] * 100).round(2)
    
    # Calculate monthly price change
    monthly_stats['Monthly_Return'] = (
        (monthly_stats['Month_End_Close'] - monthly_stats['Month_Start_Close']) / 
        monthly_stats['Month_Start_Close'] * 100
    ).round(2)
    
    # Calculate momentum scores
    momentum_scores = calculate_momentum_score(data, monthly_stats)
    monthly_stats['Momentum_Score'] = momentum_scores
    
    # Create momentum rating
    def get_momentum_rating(score):
        if score >= 80:
            return "Very Strong"
        elif score >= 60:
            return "Strong"
        elif score >= 40:
            return "Moderate"
        elif score >= 20:
            return "Weak"
        else:
            return "Very Weak"
    
    monthly_stats['Momentum_Rating'] = monthly_stats['Momentum_Score'].apply(get_momentum_rating)
    
    print("\nMonthly Breakdown with Momentum Analysis:")
    print("-" * 95)
    display_cols = ['Positive_Days', 'Total_Days', 'Percentage', 'Monthly_Return', 'Momentum_Score', 'Momentum_Rating']
    print(monthly_stats[display_cols].to_string())
    
    return monthly_stats

def save_momentum_to_db(symbol, monthly_stats):
    """Save monthly momentum data to database"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Delete existing momentum data for this symbol
    cursor.execute('DELETE FROM monthly_momentum WHERE symbol = ?', (symbol,))
    
    # Insert new data
    for period, row in monthly_stats.iterrows():
        cursor.execute('''
            INSERT INTO monthly_momentum 
            (symbol, year_month, positive_days, total_days, percentage, 
             monthly_return, momentum_score, momentum_rating)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            symbol,
            str(period),
            int(row['Positive_Days']),
            int(row['Total_Days']),
            row['Percentage'],
            row['Monthly_Return'],
            row['Momentum_Score'],
            row['Momentum_Rating']
        ))
    
    conn.commit()
    conn.close()

def analyze_seasonality(data, symbol):
    """
    Perform comprehensive seasonality analysis
    """
    print(f"\n{'='*65}")
    print(f"SEASONALITY ANALYSIS FOR {symbol}")
    print(f"{'='*65}")
    
    seasonality_data = []
    
    # Month-wise analysis
    data['Month'] = data.index.month
    data['MonthName'] = data.index.strftime('%B')
    monthly_returns = data.groupby('Month').agg({
        'Daily_Return': ['mean', 'median', 'std', 'count'],
        'Close_Greater_Than_Prev': 'sum'
    })
    monthly_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
    monthly_returns['win_rate'] = (monthly_returns['positive_count'] / monthly_returns['total_count'] * 100).round(2)
    
    print("\nMONTHLY SEASONALITY:")
    print("-" * 85)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, row in monthly_returns.iterrows():
        print(f"{month_names[month-1]:4} | Avg Return: {row['avg_return']:6.2f}% | "
              f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
              f"Trades: {int(row['total_count']):3}")
        
        seasonality_data.append({
            'symbol': symbol,
            'period_type': 'month',
            'period_value': str(month),
            'avg_return': row['avg_return'],
            'median_return': row['median_return'],
            'positive_count': int(row['positive_count']),
            'total_count': int(row['total_count']),
            'win_rate': row['win_rate'],
            'std_dev': row['std_dev']
        })
    
    # Day of week analysis - FIXED: Include all 7 days
    data['DayOfWeek'] = data.index.dayofweek
    data['DayName'] = data.index.strftime('%A')
    dow_returns = data.groupby('DayOfWeek').agg({
        'Daily_Return': ['mean', 'median', 'std', 'count'],
        'Close_Greater_Than_Prev': 'sum'
    })
    dow_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
    dow_returns['win_rate'] = (dow_returns['positive_count'] / dow_returns['total_count'] * 100).round(2)
    
    print("\nDAY OF WEEK SEASONALITY:")
    print("-" * 85)
    # FIXED: Expanded to include all 7 days
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    for dow, row in dow_returns.iterrows():
        print(f"{day_names[dow]:9} | Avg Return: {row['avg_return']:6.2f}% | "
              f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
              f"Trades: {int(row['total_count']):3}")
        
        seasonality_data.append({
            'symbol': symbol,
            'period_type': 'day_of_week',
            'period_value': str(dow),
            'avg_return': row['avg_return'],
            'median_return': row['median_return'],
            'positive_count': int(row['positive_count']),
            'total_count': int(row['total_count']),
            'win_rate': row['win_rate'],
            'std_dev': row['std_dev']
        })
    
    # Quarter analysis
    data['Quarter'] = data.index.quarter
    quarter_returns = data.groupby('Quarter').agg({
        'Daily_Return': ['mean', 'median', 'std', 'count'],
        'Close_Greater_Than_Prev': 'sum'
    })
    quarter_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
    quarter_returns['win_rate'] = (quarter_returns['positive_count'] / quarter_returns['total_count'] * 100).round(2)
    
    print("\nQUARTERLY SEASONALITY:")
    print("-" * 85)
    for quarter, row in quarter_returns.iterrows():
        print(f"Q{quarter}      | Avg Return: {row['avg_return']:6.2f}% | "
              f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
              f"Trades: {int(row['total_count']):3}")
        
        seasonality_data.append({
            'symbol': symbol,
            'period_type': 'quarter',
            'period_value': str(quarter),
            'avg_return': row['avg_return'],
            'median_return': row['median_return'],
            'positive_count': int(row['positive_count']),
            'total_count': int(row['total_count']),
            'win_rate': row['win_rate'],
            'std_dev': row['std_dev']
        })
    
    # Save seasonality data to database
    save_seasonality_to_db(seasonality_data)
    
    return monthly_returns, dow_returns, quarter_returns
def save_seasonality_to_db(seasonality_data):
    """Save seasonality statistics to database"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.DataFrame(seasonality_data)
    
    # Delete existing seasonality data for this symbol
    cursor = conn.cursor()
    if len(seasonality_data) > 0:
        cursor.execute('DELETE FROM seasonality_stats WHERE symbol = ?', (seasonality_data[0]['symbol'],))
    
    df.to_sql('seasonality_stats', conn, if_exists='append', index=False)
    conn.commit()
    conn.close()

def create_visualizations(data, monthly_stats, symbol):
    """
    Create comprehensive visualizations including momentum scores AND seasonality
    """
    fig =plt.figure(figsize=(16, 12))
    
    # Create subplot layout - 5 rows, 3 columns with more spacing
    gs = fig.add_gridspec(5, 3, hspace=0.5, wspace=0.4)
    
    # Plot 1: Monthly positive days count
    ax1 = fig.add_subplot(gs[0, 0])
    months = [str(period) for period in monthly_stats.index]
    bars = ax1.bar(months, monthly_stats['Positive_Days'], color='green', alpha=0.7)
    ax1.set_title('Monthly Positive Days Count', fontsize=12, fontweight='bold', pad=15)
    ax1.set_xlabel('Month', fontsize=10)
    ax1.set_ylabel('Number of Days', fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=9)
    ax1.tick_params(axis='y', labelsize=9)
    ax1.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, monthly_stats['Positive_Days']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # Plot 2: Monthly percentage trend
    ax2 = fig.add_subplot(gs[0, 1])
    line = ax2.plot(months, monthly_stats['Percentage'], marker='o', 
                   linewidth=2, markersize=6, color='blue')
    ax2.set_title('Monthly Positive Days %', fontsize=12, fontweight='bold', pad=15)
    ax2.set_xlabel('Month', fontsize=10)
    ax2.set_ylabel('Percentage (%)', fontsize=10)
    ax2.tick_params(axis='x', rotation=45, labelsize=9)
    ax2.tick_params(axis='y', labelsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    for i, pct in enumerate(monthly_stats['Percentage']):
        ax2.annotate(f'{pct}%', (i, pct), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=9)
    
    # Plot 3: Momentum Scores
    ax3 = fig.add_subplot(gs[0, 2])
    colors = []
    for score in monthly_stats['Momentum_Score']:
        if score >= 80:
            colors.append('darkgreen')
        elif score >= 60:
            colors.append('green')
        elif score >= 40:
            colors.append('orange')
        elif score >= 20:
            colors.append('red')
        else:
            colors.append('darkred')
    
    bars = ax3.bar(months, monthly_stats['Momentum_Score'], color=colors, alpha=0.8)
    ax3.set_title('Monthly Momentum Scores', fontsize=12, fontweight='bold', pad=15)
    ax3.set_xlabel('Month', fontsize=10)
    ax3.set_ylabel('Momentum Score (0-100)', fontsize=10)
    ax3.tick_params(axis='x', rotation=45, labelsize=9)
    ax3.tick_params(axis='y', labelsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    for bar, score in zip(bars, monthly_stats['Momentum_Score']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='darkgreen', label='Very Strong (80+)'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='Strong (60-79)'),
        plt.Rectangle((0,0),1,1, facecolor='orange', label='Moderate (40-59)'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='Weak (20-39)'),
        plt.Rectangle((0,0),1,1, facecolor='darkred', label='Very Weak (<20)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Plot 4: Monthly Returns vs Momentum Score (Scatter)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(monthly_stats['Monthly_Return'], monthly_stats['Momentum_Score'], 
                         c=monthly_stats['Momentum_Score'], cmap='RdYlGn', s=100, alpha=0.7)
    ax4.set_title('Monthly Return vs Momentum Score', fontsize=12, fontweight='bold', pad=15)
    ax4.set_xlabel('Monthly Return (%)', fontsize=10)
    ax4.set_ylabel('Momentum Score', fontsize=10)
    ax4.tick_params(axis='x', labelsize=9)
    ax4.tick_params(axis='y', labelsize=9)
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Momentum Score')
    
    for i, month in enumerate(months):
        ax4.annotate(month.split('-')[1], 
                    (monthly_stats['Monthly_Return'].iloc[i], monthly_stats['Momentum_Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=9)

    # Plot 5: Stock price trend with volume
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5_vol = ax5.twinx()

    try:
        subset = data.iloc[-90:]
        ohlc = subset[['Open', 'High', 'Low', 'Close']].copy()
        ohlc['Date'] = mdates.date2num(ohlc.index)
        ohlc = ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values

        candle_width = 0.8
        candle_col_up = "green"
        candle_col_down = "red"

        for date, open_, high, low, close in ohlc:
            color = candle_col_up if close >= open_ else candle_col_down
            ax5.plot([date, date], [low, high], color=color, linewidth=1)
            rect = Rectangle((date - candle_width/2, min(open_, close)),
                           candle_width, abs(close - open_),
                           facecolor=color, edgecolor=color, alpha=0.9)
            ax5.add_patch(rect)

        ax5.xaxis_date()
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax5.set_xlim(ohlc[:,0].min(), ohlc[:,0].max())
        ax5.set_ylabel("Price (₹)", color="purple", fontsize=10)
        ax5.tick_params(axis="y", labelcolor="purple", labelsize=9)

        ax5_vol.bar(subset.index, subset["Volume"], alpha=0.3, color="gray", width=1)
        ax5_vol.set_ylabel("Volume", color="gray", fontsize=10)
        ax5_vol.tick_params(axis="y", labelcolor="gray", labelsize=9)

        ax5.set_title("Stock Price Trend with Volume", fontsize=12, fontweight="bold", pad=15)
        ax5.set_xlabel("Date", fontsize=10)
        ax5.tick_params(axis='x', labelsize=9)
        ax5.grid(True, alpha=0.3)
    except Exception as e:
        print(f"Error plotting candlestick: {e}")
        pass
    
    # Plot 6: Daily returns distribution
    ax6 = fig.add_subplot(gs[2, 0])
    daily_returns = data['Daily_Return'].dropna()
    n, bins, patches = ax6.hist(daily_returns, bins=30, alpha=0.7, color='orange', edgecolor='black')
    
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val >= 0:
            patch.set_facecolor('green')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
    
    ax6.set_title('Daily Returns Distribution', fontsize=12, fontweight='bold', pad=15)
    ax6.set_xlabel('Daily Return (%)', fontsize=10)
    ax6.set_ylabel('Frequency', fontsize=10)
    ax6.tick_params(axis='x', labelsize=9)
    ax6.tick_params(axis='y', labelsize=9)
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='No Change')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative returns
    ax7 = fig.add_subplot(gs[2, 1])
    cumulative_returns = (1 + data['Daily_Return']/100).cumprod() - 1
    ax7.plot(data.index, cumulative_returns * 100, linewidth=2, color='darkgreen')
    ax7.set_title('Cumulative Returns', fontsize=12, fontweight='bold', pad=15)
    ax7.set_xlabel('Date', fontsize=10)
    ax7.set_ylabel('Cumulative Return (%)', fontsize=10)
    ax7.tick_params(axis='x', labelsize=9)
    ax7.tick_params(axis='y', labelsize=9)
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 8: Momentum Score Trend
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(range(len(months)), monthly_stats['Momentum_Score'], 
             marker='s', linewidth=2, markersize=8, color='darkblue')
    ax8.set_title('Momentum Score Trend', fontsize=12, fontweight='bold', pad=15)
    ax8.set_xlabel('Month', fontsize=10)
    ax8.set_ylabel('Momentum Score', fontsize=10)
    ax8.set_xticks(range(len(months)))
    ax8.set_xticklabels(months, rotation=45, fontsize=9)
    ax8.tick_params(axis='y', labelsize=9)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 100)
    
    ax8.axhline(y=80, color='darkgreen', linestyle='--', alpha=0.7, label='Very Strong')
    ax8.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Strong')
    ax8.axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Moderate')
    ax8.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Weak')
    ax8.legend(fontsize=8)
    
    # Plot 9: Momentum Components Breakdown
    ax9 = fig.add_subplot(gs[3, :])
    
    components_data = []
    for period in monthly_stats.index:
        month_data = data[data['YearMonth'] == period]
        
        positive_pct = monthly_stats.loc[period, 'Percentage']
        positive_score = (positive_pct / 100) * 40
        
        avg_return = monthly_stats.loc[period, 'Avg_Return']
        return_score = max(-25, min(25, (avg_return / 10) * 25))
        
        monthly_return = monthly_stats.loc[period, 'Monthly_Return']
        price_score = max(-25, min(25, (monthly_return / 20) * 25))
        
        volatility = monthly_stats.loc[period, 'Return_Std']
        consistency_score = max(0, 10 - (volatility / 5) * 10)
        
        components_data.append({
            'Month': str(period),
            'Positive_Days': positive_score,
            'Avg_Return': return_score,
            'Price_Momentum': price_score,
            'Consistency': consistency_score
        })
    
    components_df = pd.DataFrame(components_data)
    
    bottom1 = components_df['Positive_Days']
    bottom2 = bottom1 + components_df['Avg_Return']
    bottom3 = bottom2 + components_df['Price_Momentum']
    
    ax9.bar(components_df['Month'], components_df['Positive_Days'], 
            label='Positive Days (40pts)', color='green', alpha=0.7)
    ax9.bar(components_df['Month'], components_df['Avg_Return'], 
            bottom=bottom1, label='Avg Return (25pts)', color='blue', alpha=0.7)
    ax9.bar(components_df['Month'], components_df['Price_Momentum'], 
            bottom=bottom2, label='Price Momentum (25pts)', color='purple', alpha=0.7)
    ax9.bar(components_df['Month'], components_df['Consistency'], 
            bottom=bottom3, label='Consistency (10pts)', color='orange', alpha=0.7)
    
    ax9.set_title('Momentum Score Components Breakdown', fontsize=12, fontweight='bold', pad=15)
    ax9.set_xlabel('Month', fontsize=10)
    ax9.set_ylabel('Score Contribution', fontsize=10)
    ax9.legend(loc='upper left', fontsize=9)
    ax9.tick_params(axis='x', rotation=45, labelsize=9)
    ax9.tick_params(axis='y', labelsize=9)
    ax9.grid(True, alpha=0.3)
    
    # Plot 10: Monthly Seasonality Heatmap
    ax10 = fig.add_subplot(gs[4, 0])
    data['Month'] = data.index.month
    data['Year'] = data.index.year
    monthly_returns_pivot = data.pivot_table(values='Daily_Return', index='Year', columns='Month', aggfunc='mean')
    
    sns.heatmap(monthly_returns_pivot, cmap='RdYlGn', center=0, annot=True, fmt=".2f", 
                linewidths=.5, ax=ax10, cbar_kws={'label': 'Avg Daily Return (%)'})
    ax10.set_title('Monthly Returns Heatmap by Year', fontsize=12, fontweight='bold', pad=15)
    ax10.set_xlabel('Month', fontsize=10)
    ax10.set_ylabel('Year', fontsize=10)
    ax10.tick_params(axis='x', labelsize=9)
    ax10.tick_params(axis='y', labelsize=9)
    
    # Plot 11: Day of Week Seasonality
    ax11 = fig.add_subplot(gs[4, 1])
    data['DayOfWeek'] = data.index.dayofweek
    dow_returns = data.groupby('DayOfWeek')['Daily_Return'].mean()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors = ['green' if x >= 0 else 'red' for x in dow_returns]
    bars = ax11.bar(range(len(dow_returns)), dow_returns, color=colors, alpha=0.7)
    ax11.set_title('Day of Week Returns', fontsize=12, fontweight='bold', pad=15)
    ax11.set_xlabel('Day of Week', fontsize=10)
    ax11.set_ylabel('Average Daily Return (%)', fontsize=10)
    ax11.set_xticks(range(len(dow_returns)))
    ax11.set_xticklabels([day_names[i] for i in dow_returns.index], rotation=45, fontsize=9)
    ax11.tick_params(axis='y', labelsize=9)
    ax11.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax11.grid(True, alpha=0.3)
    
    for i, (bar, value) in enumerate(zip(bars, dow_returns)):
        height = bar.get_height()
        ax11.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.15),
                f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # Plot 12: Quarterly Returns
    ax12 = fig.add_subplot(gs[4, 2])
    data['Quarter'] = data.index.quarter
    quarterly_returns = data.groupby('Quarter')['Daily_Return'].mean()
    quarter_names = ['Q1', 'Q2', 'Q3', 'Q4']
    colors = ['green' if x >= 0 else 'red' for x in quarterly_returns]
    bars = ax12.bar(quarter_names, quarterly_returns, color=colors, alpha=0.7)
    ax12.set_title('Quarterly Returns', fontsize=12, fontweight='bold', pad=15)
    ax12.set_xlabel('Quarter', fontsize=10)
    ax12.set_ylabel('Average Daily Return (%)', fontsize=10)
    ax12.tick_params(axis='x', labelsize=9)
    ax12.tick_params(axis='y', labelsize=9)
    ax12.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax12.grid(True, alpha=0.3)
    
    for bar, value in zip(bars, quarterly_returns):
        height = bar.get_height()
        ax12.text(bar.get_x() + bar.get_width()/2., height + (0.05 if height >= 0 else -0.15),
                f'{value:.2f}%', ha='center', va='bottom' if height >= 0 else 'top', 
                fontweight='bold', fontsize=9)
    
    # Add main title with more padding
    plt.suptitle(f'{symbol} Stock Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust subplot spacing manually instead of using tight_layout
    plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.98, 
                       hspace=0.4, wspace=0.3)
    
    # Save the figure
    plt.savefig("C:\\temp\\momentum\\"+symbol+"_analysis.png", dpi=150, bbox_inches='tight')
    #plt.show()
    
    return fig

def generate_insights(analyzed_data, monthly_stats, symbol):
    """
    Generate comprehensive insights from the analysis
    """
    print(f"\n{'='*65}")
    print(f"COMPREHENSIVE INSIGHTS WITH MOMENTUM ANALYSIS")
    print(f"{'='*65}")
    
    # Basic Statistics
    print("\nBasic Statistics:")
    print("-" * 65)
    total_occurrences = analyzed_data['Close_Greater_Than_Prev'].sum()
    total_trading_days = len(analyzed_data)
    print(f"Total trading days: {total_trading_days}")
    print(f"Days where Close > Previous Close: {total_occurrences}")
    print(f"Percentage: {(total_occurrences/total_trading_days)*100:.2f}%")
    print(f"Average daily return: {analyzed_data['Daily_Return'].mean():.2f}%")
    print(f"Daily return volatility: {analyzed_data['Daily_Return'].std():.2f}%")
    print(f"Maximum daily return: {analyzed_data['Daily_Return'].max():.2f}%")
    print(f"Minimum daily return: {analyzed_data['Daily_Return'].min():.2f}%")
    
    # Momentum Analysis
    print("\nMomentum Analysis:")
    print("-" * 65)
    best_month = monthly_stats['Momentum_Score'].idxmax()
    worst_month = monthly_stats['Momentum_Score'].idxmin()
    print(f"Best performing month: {best_month} (Momentum Score: {monthly_stats.loc[best_month, 'Momentum_Score']})")
    print(f"Worst performing month: {worst_month} (Momentum Score: {monthly_stats.loc[worst_month, 'Momentum_Score']})")
    
    # Find months with highest and lowest positive day percentages
    highest_positive_month = monthly_stats['Percentage'].idxmax()
    lowest_positive_month = monthly_stats['Percentage'].idxmin()
    print(f"Highest positive days percentage: {highest_positive_month} ({monthly_stats.loc[highest_positive_month, 'Percentage']:.2f}%)")
    print(f"Lowest positive days percentage: {lowest_positive_month} ({monthly_stats.loc[lowest_positive_month, 'Percentage']:.2f}%)")
    
    # Seasonality Insights
    print("\nSeasonality Insights:")
    print("-" * 65)
    
    # Monthly seasonality
    analyzed_data['Month'] = analyzed_data.index.month
    monthly_seasonality = analyzed_data.groupby('Month')['Daily_Return'].mean()
    best_month_season = monthly_seasonality.idxmax()
    worst_month_season = monthly_seasonality.idxmin()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"Best month historically: {month_names[best_month_season-1]} (Avg return: {monthly_seasonality.max():.2f}%)")
    print(f"Worst month historically: {month_names[worst_month_season-1]} (Avg return: {monthly_seasonality.min():.2f}%)")
    
    # Day of week seasonality
    analyzed_data['DayOfWeek'] = analyzed_data.index.dayofweek
    dow_seasonality = analyzed_data.groupby('DayOfWeek')['Daily_Return'].mean()
    best_dow = dow_seasonality.idxmax()
    worst_dow = dow_seasonality.idxmin()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    print(f"Best day historically: {day_names[best_dow]} (Avg return: {dow_seasonality.max():.2f}%)")
    print(f"Worst day historically: {day_names[worst_dow]} (Avg return: {dow_seasonality.min():.2f}%)")
    
    # Trading Recommendations
    print("\nTrading Recommendations:")
    print("-" * 65)
    
    # Based on momentum scores
    strong_months = monthly_stats[monthly_stats['Momentum_Score'] >= 60].index.tolist()
    if strong_months:
        print(f"Strong momentum months (score >= 60): {', '.join(str(m) for m in strong_months)}")
    
    weak_months = monthly_stats[monthly_stats['Momentum_Score'] < 40].index.tolist()
    if weak_months:
        print(f"Weak momentum months (score < 40): {', '.join(str(m) for m in weak_months)}")
    
    # Based on positive day percentage
    high_positive_months = monthly_stats[monthly_stats['Percentage'] >= 60].index.tolist()
    if high_positive_months:
        print(f"High positive day months (>= 60%): {', '.join(str(m) for m in high_positive_months)}")
    
    # Risk Assessment
    print("\nRisk Assessment:")
    print("-" * 65)
    volatility = analyzed_data['Daily_Return'].std()
    if volatility < 1:
        print("Low volatility stock (suitable for conservative investors)")
    elif volatility < 2:
        print("Moderate volatility stock (suitable for balanced investors)")
    else:
        print("High volatility stock (suitable for aggressive investors)")
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + analyzed_data['Daily_Return']/100).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    print(f"Maximum drawdown: {max_drawdown:.2f}%")
    
    # Additional insights
    print(f"\nAdditional Insights:")
    print("-" * 65)
    
    # Basic statistics using analyzed_data['Daily_Return']
    print(f"\nBasic Statistics:")
    print(f"Average daily return: {analyzed_data['Daily_Return'].mean():.2f}%")
    print(f"Standard deviation of returns: {analyzed_data['Daily_Return'].std():.2f}%")
    print(f"Maximum daily gain: {analyzed_data['Daily_Return'].max():.2f}%")
    print(f"Maximum daily loss: {analyzed_data['Daily_Return'].min():.2f}%")
    print(f"Sharpe ratio (assuming 0% risk-free rate): {analyzed_data['Daily_Return'].mean()/analyzed_data['Daily_Return'].std():.3f}")
    
    # Volatility analysis
    print(f"\nVolatility Analysis:")
    print(f"Daily volatility: {analyzed_data['Daily_Return'].std():.2f}%")
    print(f"Annualized volatility: {analyzed_data['Daily_Return'].std() * np.sqrt(252):.2f}%")
    
    # Momentum Analysis
    print(f"\nMomentum Analysis:")
    avg_momentum = monthly_stats['Momentum_Score'].mean()
    highest_momentum_month = monthly_stats.loc[monthly_stats['Momentum_Score'].idxmax()]
    lowest_momentum_month = monthly_stats.loc[monthly_stats['Momentum_Score'].idxmin()]
    
    print(f"Average monthly momentum score: {avg_momentum:.1f}")
    print(f"Highest momentum month: {monthly_stats['Momentum_Score'].idxmax()} (Score: {highest_momentum_month['Momentum_Score']:.1f} - {highest_momentum_month['Momentum_Rating']})")
    print(f"Lowest momentum month: {monthly_stats['Momentum_Score'].idxmin()} (Score: {lowest_momentum_month['Momentum_Score']:.1f} - {lowest_momentum_month['Momentum_Rating']})")
    
    # Momentum distribution
    momentum_distribution = monthly_stats['Momentum_Rating'].value_counts()
    print(f"\nMomentum Rating Distribution:")
    for rating, count in momentum_distribution.items():
        print(f"  {rating}: {count} months ({count/len(monthly_stats)*100:.1f}%)")
    
    # Recent momentum trend
    recent_3_months = monthly_stats.tail(3)
    recent_momentum_trend = recent_3_months['Momentum_Score'].mean()
    momentum_change = recent_3_months['Momentum_Score'].iloc[-1] - recent_3_months['Momentum_Score'].iloc[0]
    
    print(f"\nRecent Momentum Trend (Last 3 months):")
    print(f"Average momentum score: {recent_momentum_trend:.1f}")
    print(f"Momentum change: {momentum_change:+.1f} points")
    if momentum_change > 10:
        print("  → Momentum is improving strongly")
    elif momentum_change > 0:
        print("  → Momentum is improving")
    elif momentum_change > -10:
        print("  → Momentum is relatively stable")
    else:
        print("  → Momentum is weakening")
    
    # Correlation analysis
    correlation_return_momentum = monthly_stats['Monthly_Return'].corr(monthly_stats['Momentum_Score'])
    correlation_volatility_momentum = monthly_stats['Return_Std'].corr(monthly_stats['Momentum_Score'])
    
    print(f"\nCorrelation Analysis:")
    print(f"Monthly return vs momentum score: {correlation_return_momentum:.3f}")
    print(f"Volatility vs momentum score: {correlation_volatility_momentum:.3f}")
    
    # Best and worst performing months
    best_month = monthly_stats.loc[monthly_stats['Percentage'].idxmax()]
    worst_month = monthly_stats.loc[monthly_stats['Percentage'].idxmin()]
    
    print(f"\nMonthly Performance:")
    print(f"Best month: {monthly_stats['Percentage'].idxmax()} ({best_month['Percentage']:.2f}% positive days, Momentum: {best_month['Momentum_Score']:.1f})")
    print(f"Worst month: {monthly_stats['Percentage'].idxmin()} ({worst_month['Percentage']:.2f}% positive days, Momentum: {worst_month['Momentum_Score']:.1f})")
    
    # Recent trend analysis
    recent_data = analyzed_data.tail(30)
    recent_positive_days = recent_data['Close_Greater_Than_Prev'].sum()
    recent_total_days = len(recent_data)
    recent_avg_return = recent_data['Daily_Return'].mean()
    
    print(f"\nRecent Trend (Last 30 trading days):")
    print(f"Positive days: {recent_positive_days} out of {recent_total_days} ({(recent_positive_days/recent_total_days)*100:.2f}%)")
    print(f"Average daily return: {recent_avg_return:.2f}%")
    
    # Estimate current month momentum
    if len(recent_data) > 0:
        current_month_positive_pct = (recent_positive_days / recent_total_days) * 100
        current_month_avg_return = recent_avg_return
        current_month_volatility = recent_data['Daily_Return'].std()
        
        positive_score = (current_month_positive_pct / 100) * 40
        return_score = max(-25, min(25, (current_month_avg_return / 10) * 25))
        consistency_score = max(0, 10 - (current_month_volatility / 5) * 10)
        
        estimated_momentum = positive_score + return_score + consistency_score
        
        print(f"\nEstimated Current Month Momentum (based on last 30 days):")
        print(f"Partial momentum score: ~{estimated_momentum:.1f} (excluding price momentum component)")
    
    # Price momentum
    current_price = analyzed_data['Close'].iloc[-1]
    price_30d_ago = analyzed_data['Close'].iloc[-30] if len(analyzed_data) >= 30 else analyzed_data['Close'].iloc[0]
    price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
    
    print(f"\nPrice Momentum:")
    print(f"Current price: ₹{current_price:.2f}")
    print(f"30-day price change: {price_change_30d:.2f}%")
    
    # Risk metrics
    negative_returns = analyzed_data['Daily_Return'][analyzed_data['Daily_Return'] < 0]
    if len(negative_returns) > 0:
        var_95 = np.percentile(analyzed_data['Daily_Return'], 5)
        print(f"\nRisk Metrics:")
        print(f"Value at Risk (95%): {var_95:.2f}%")
        print(f"Average negative return: {negative_returns.mean():.2f}%")
        print(f"Probability of loss: {len(negative_returns)/len(analyzed_data['Daily_Return'])*100:.1f}%")
    
    # Save insights to file
    save_insights_to_file(symbol, analyzed_data, monthly_stats, monthly_seasonality, dow_seasonality, pd.Series([1,2,3,4]))  # dummy quarter data
    
    
def main():
    """Main function to run the stock analysis"""
    # Initialize database
    init_database()
    from tqdm import tqdm
    df= pd.read_csv('N500.csv')
    symbols = df['Symbol']
    for i in tqdm(range(len(symbols)-1)):        
        for ticker in symbols:
        
                try:
            
                    stock_data = fetch_and_analyze_stock(ticker)
                    
                    # Analyze positive days
                    analyzed_data = analyze_positive_days(stock_data)
                    
                    # Monthly breakdown with momentum
                    monthly_stats = monthly_breakdown(analyzed_data)
                    
                    # Save momentum data to database
                    symbol = ticker.replace('.NS', '')
                    save_momentum_to_db(symbol, monthly_stats)
                    
                    # Seasonality analysis
                    monthly_returns, dow_returns, quarter_returns = analyze_seasonality(analyzed_data, symbol)
                    
                    # Create visualizations
                    create_visualizations(analyzed_data, monthly_stats, symbol)
                    
                    # Generate insights (FIXED: passing correct parameters)
                    generate_insights(analyzed_data, monthly_stats, symbol)
                
        
                
                except Exception as e:
                    print(f"Error processing {ticker}: {e}")
                    import traceback
                    traceback.print_exc()
                    pass
        print("\nAnalysis completed successfully!")
if __name__ == "__main__":
    main()
