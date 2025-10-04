import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import warnings
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
import os
import pickle
import time
from functools import lru_cache

# --- ADD THESE TWO LINES FOR ROBUSTNESS ---
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to prevent memory errors
# -----------------------------------------

warnings.filterwarnings('ignore')
logging.basicConfig(filename='momentum_seasonality.log', 
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()
logger.handlers.clear()
file_handler = logging.FileHandler('momentum_seasonality', mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Thread lock for plotting operations
plot_lock = threading.Lock()

# Cache directory
CACHE_DIR = "C:\\temp\\stock_cache\\"
os.makedirs(CACHE_DIR, exist_ok=True)

def get_cache_path(ticker, days):
    """Get the cache file path for a ticker"""
    return os.path.join(CACHE_DIR, f"{ticker}_{days}.pkl")

def is_cache_valid(cache_path, max_age_hours=24):
    """Check if cache file exists and is not too old"""
    if not os.path.exists(cache_path):
        return False
    
    file_time = os.path.getmtime(cache_path)
    current_time = time.time()
    age_hours = (current_time - file_time) / 3600
    
    return age_hours < max_age_hours

def fetch_stock_data(ticker, days=365):
    """Fetch stock data with caching"""
    cache_path = get_cache_path(ticker, days)
    
    # Check if we have valid cached data
    if is_cache_valid(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                stock_data = pickle.load(f)
            logger.info(f"Using cached data for {ticker}")
            return stock_data
        except Exception as e:
            logger.info(f"Error loading cache for {ticker}: {e}")
    
    # Fetch new data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    logger.info(f"Fetching data for {ticker} from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        stock_data = yf.download(ticker+'.NS', start=start_date, end=end_date, progress=False,multi_level_index=False)
        
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Handle MultiIndex columns
        if isinstance(stock_data.columns, pd.MultiIndex):
            stock_data.columns = stock_data.columns.droplevel(1)
        
        # Ensure we have single-level columns
        if any(isinstance(col, tuple) for col in stock_data.columns):
            stock_data.columns = [col[0] if isinstance(col, tuple) else col for col in stock_data.columns]
        
        # Cache the data
        with open(cache_path, 'wb') as f:
            pickle.dump(stock_data, f)
        
        logger.info(f"Fetched and cached {len(stock_data)} records for {ticker}")
        return stock_data
    except Exception as e:
        logger.info(f"Error fetching data for {ticker}: {e}")
        raise

def analyze_positive_days(stock_data):
    """Analyze days where close > previous close"""
    data = stock_data.copy()
    
    # Ensure we have the correct columns
    if 'Close' not in data.columns:
        logger.error("Close column not found in data")
        return pd.DataFrame()
    
    # Handle case where Close might be a DataFrame
    if isinstance(data['Close'], pd.DataFrame):
        if len(data['Close'].columns) > 0:
            data['Close'] = data['Close'].iloc[:, 0]
        else:
            logger.error("Close DataFrame is empty")
            return pd.DataFrame()
    
    # Create Prev_Close column
    data['Prev_Close'] = data['Close'].shift(1)
    data['Close_Greater_Than_Prev'] = data['Close'] > data['Prev_Close']
    data['Daily_Return'] = (data['Close'] - data['Prev_Close']) / data['Prev_Close'] * 100
    data = data.dropna()
    
    total_occurrences = data['Close_Greater_Than_Prev'].sum()
    total_trading_days = len(data)
    
    logger.info(f"\nAnalysis Results:")
    logger.info(f"Total trading days: {total_trading_days}")
    logger.info(f"Days where Close > Previous Close: {total_occurrences}")
    logger.info(f"Percentage: {(total_occurrences/total_trading_days)*100:.2f}%")
    
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
    # Check if data is valid
    if len(data) < 30:  # Need at least 30 days of data
        logger.warning("Insufficient data for monthly breakdown (less than 30 days)")
        return pd.DataFrame()
    
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
    
    logger.info("\nMonthly Breakdown with Momentum Analysis:")
    logger.info("-" * 95)
    display_cols = ['Positive_Days', 'Total_Days', 'Percentage', 'Monthly_Return', 'Momentum_Score', 'Momentum_Rating']
    logger.info(monthly_stats[display_cols].to_string())
    
    return monthly_stats

def analyze_seasonality(data, symbol):
    """
    Perform comprehensive seasonality analysis
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"SEASONALITY ANALYSIS FOR {symbol}")
    logger.info(f"{'='*65}")
    
    # Check if data is valid
    if len(data) < 30:  # Need at least 30 days of data
        logger.warning("Insufficient data for seasonality analysis (less than 30 days)")
        return pd.Series(), pd.Series(), pd.Series()
    
    # Month-wise analysis
    data['Month'] = data.index.month
    data['MonthName'] = data.index.strftime('%B')
    monthly_returns = data.groupby('Month').agg({
        'Daily_Return': ['mean', 'median', 'std', 'count'],
        'Close_Greater_Than_Prev': 'sum'
    })
    monthly_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
    monthly_returns['win_rate'] = (monthly_returns['positive_count'] / monthly_returns['total_count'] * 100).round(2)
    
    logger.info("\nMONTHLY SEASONALITY:")
    logger.info("-" * 85)
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for month, row in monthly_returns.iterrows():
        logger.info(f"{month_names[month-1]:4} | Avg Return: {row['avg_return']:6.2f}% | "
              f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
              f"Trades: {int(row['total_count']):3}")
    
    # Day of week analysis - EXCLUDE WEEKENDS (Saturday=5, Sunday=6)
    data['DayOfWeek'] = data.index.dayofweek
    # Filter out weekends (only keep Monday=0 to Friday=4)
    weekday_data = data[data['DayOfWeek'] <= 4]
    
    if len(weekday_data) > 0:
        dow_returns = weekday_data.groupby('DayOfWeek').agg({
            'Daily_Return': ['mean', 'median', 'std', 'count'],
            'Close_Greater_Than_Prev': 'sum'
        })
        dow_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
        dow_returns['win_rate'] = (dow_returns['positive_count'] / dow_returns['total_count'] * 100).round(2)
        
        logger.info("\nDAY OF WEEK SEASONALITY (Trading Days Only):")
        logger.info("-" * 85)
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        for dow, row in dow_returns.iterrows():
            logger.info(f"{day_names[dow]:9} | Avg Return: {row['avg_return']:6.2f}% | "
                  f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
                  f"Trades: {int(row['total_count']):3}")
    else:
        logger.warning("No weekday data available for day-of-week analysis")
        dow_returns = pd.DataFrame()
    
    # Quarter analysis
    data['Quarter'] = data.index.quarter
    quarter_returns = data.groupby('Quarter').agg({
        'Daily_Return': ['mean', 'median', 'std', 'count'],
        'Close_Greater_Than_Prev': 'sum'
    })
    quarter_returns.columns = ['avg_return', 'median_return', 'std_dev', 'total_count', 'positive_count']
    quarter_returns['win_rate'] = (quarter_returns['positive_count'] / quarter_returns['total_count'] * 100).round(2)
    
    logger.info("\nQUARTERLY SEASONALITY:")
    logger.info("-" * 85)
    for quarter, row in quarter_returns.iterrows():
        logger.info(f"Q{quarter}      | Avg Return: {row['avg_return']:6.2f}% | "
              f"Win Rate: {row['win_rate']:5.1f}% | Volatility: {row['std_dev']:5.2f}% | "
              f"Trades: {int(row['total_count']):3}")
    
    return monthly_returns['avg_return'], dow_returns['avg_return'] if len(dow_returns) > 0 else pd.Series(), quarter_returns['avg_return']


def create_visualizations(data, monthly_stats, symbol):
    """
    Create comprehensive visualizations including momentum scores AND seasonality
    """
    # Check if data is valid
    if len(data) < 30 or len(monthly_stats) == 0:
        logger.warning("Insufficient data for visualization")
        return None
    
    with plot_lock:
        fig = plt.figure(figsize=(18, 24))
        fig.suptitle(f"Comprehensive Stock Analysis for {symbol} (Research Purpose only. NOT financial advise)", fontsize=11, fontweight='bold', y=0.92)
      
        # Create subplot layout - 5 rows, 3 columns with more spacing
        gs = fig.add_gridspec(5, 3, hspace=0.5, wspace=0.4)
        
        # Convert period index to categorical strings for plotting to avoid warnings
        months = pd.Categorical([str(period) for period in monthly_stats.index])
        
        # Plot 1: Monthly positive days count
        ax1 = fig.add_subplot(gs[0, 0])
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
            ax4.annotate(str(month).split('-')[1], 
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
       # Plot 11: Day of Week Seasonality - EXCLUDE WEEKENDS
        ax11 = fig.add_subplot(gs[4, 1])
        data['DayOfWeek'] = data.index.dayofweek
        # Filter out weekends (only keep Monday=0 to Friday=4)
        weekday_data = data[data['DayOfWeek'] <= 4]
    
        if len(weekday_data) > 0:
            dow_returns = weekday_data.groupby('DayOfWeek')['Daily_Return'].mean()
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
            colors = ['green' if x >= 0 else 'red' for x in dow_returns]
            bars = ax11.bar(range(len(dow_returns)), dow_returns, color=colors, alpha=0.7)
            ax11.set_title('Day of Week Returns (Trading Days Only)', fontsize=12, fontweight='bold', pad=15)
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
        else:
            ax11.text(0.5, 0.5, 'No weekday data available', ha='center', va='center', transform=ax11.transAxes)
            ax11.set_title('Day of Week Returns (Trading Days Only)', fontsize=12, fontweight='bold', pad=15)
        
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
        plt.suptitle(f'{symbol} Stock Analysis Dashboard', fontsize=12, fontweight='bold', y=0.98)
        
        # Adjust subplot spacing manually instead of using tight_layout
        plt.subplots_adjust(top=0.94, bottom=0.05, left=0.05, right=0.98, 
                           hspace=0.4, wspace=0.3)
        
        
        
        # Ensure the output directory exists
        output_dir = "C:\\temp\\momentum\\"
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the figure
        output_path = os.path.join(output_dir, f"{symbol}_analysis.png")
        plt.savefig(output_path, dpi=150)
        
        # CRITICAL: Close the figure to free memory and prevent state corruption
        plt.close(fig)
        
        logger.info(f"Visualization saved to {output_path}")
        
    return fig

def save_insights_to_file(symbol, analyzed_data, monthly_stats, monthly_seasonality, dow_seasonality, quarter_seasonality):
    """Save all insights to a text file"""
    output_dir = "C:\\temp\\momentum\\"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"{symbol}_insights.txt")
    
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
        if len(monthly_stats) > 0:
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
        if len(monthly_seasonality) > 0:
            f.write(f"MONTHLY SEASONALITY:\n")
            f.write(f"{'-'*85}\n")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            for month, value in monthly_seasonality.items():
                f.write(f"{month_names[month-1]:4} | Avg Return: {value:6.2f}%\n")
        
        # Day of Week Seasonality
        if len(dow_seasonality) > 0:
            f.write(f"\nDAY OF WEEK SEASONALITY:\n")
            f.write(f"{'-'*85}\n")
            day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            for dow, value in dow_seasonality.items():
                f.write(f"{day_names[dow]:9} | Avg Return: {value:6.2f}%\n")
        
        # Trading Recommendations
        f.write(f"\nTRADING RECOMMENDATIONS:\n")
        f.write(f"{'-'*65}\n")
        
        # Best months to trade
        if len(monthly_seasonality) > 0:
            best_months = monthly_seasonality.nlargest(3).index.tolist()
            f.write(f"Best months to trade (based on historical returns): {', '.join([month_names[m-1] for m in best_months])}\n")
        
        # Best days to trade
        if len(dow_seasonality) > 0:
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
    
    logger.info(f"All insights saved to {filename}")

def generate_insights(analyzed_data, monthly_stats, symbol):
    """
    Generate comprehensive insights from the analysis
    """
    logger.info(f"\n{'='*65}")
    logger.info(f"COMPREHENSIVE INSIGHTS WITH MOMENTUM ANALYSIS")
    logger.info(f"{'='*65}")
    
    # Basic Statistics
    logger.info("\nBasic Statistics:")
    logger.info("-" * 65)
    total_occurrences = analyzed_data['Close_Greater_Than_Prev'].sum()
    total_trading_days = len(analyzed_data)
    logger.info(f"Total trading days: {total_trading_days}")
    logger.info(f"Days where Close > Previous Close: {total_occurrences}")
    logger.info(f"Percentage: {(total_occurrences/total_trading_days)*100:.2f}%")
    logger.info(f"Average daily return: {analyzed_data['Daily_Return'].mean():.2f}%")
    logger.info(f"Daily return volatility: {analyzed_data['Daily_Return'].std():.2f}%")
    logger.info(f"Maximum daily return: {analyzed_data['Daily_Return'].max():.2f}%")
    logger.info(f"Minimum daily return: {analyzed_data['Daily_Return'].min():.2f}%")
    
    # Momentum Analysis
    if len(monthly_stats) > 0:
        logger.info("\nMomentum Analysis:")
        logger.info("-" * 65)
        best_month = monthly_stats['Momentum_Score'].idxmax()
        worst_month = monthly_stats['Momentum_Score'].idxmin()
        logger.info(f"Best performing month: {best_month} (Momentum Score: {monthly_stats.loc[best_month, 'Momentum_Score']})")
        logger.info(f"Worst performing month: {worst_month} (Momentum Score: {monthly_stats.loc[worst_month, 'Momentum_Score']})")
        
        # Find months with highest and lowest positive day percentages
        highest_positive_month = monthly_stats['Percentage'].idxmax()
        lowest_positive_month = monthly_stats['Percentage'].idxmin()
        logger.info(f"Highest positive days percentage: {highest_positive_month} ({monthly_stats.loc[highest_positive_month, 'Percentage']:.2f}%)")
        logger.info(f"Lowest positive days percentage: {lowest_positive_month} ({monthly_stats.loc[lowest_positive_month, 'Percentage']:.2f}%)")
    
    # Seasonality Insights
    logger.info("\nSeasonality Insights:")
    logger.info("-" * 65)
    
    # Monthly seasonality
    analyzed_data['Month'] = analyzed_data.index.month
    monthly_seasonality = analyzed_data.groupby('Month')['Daily_Return'].mean()
    best_month_season = monthly_seasonality.idxmax()
    worst_month_season = monthly_seasonality.idxmin()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    logger.info(f"Best month historically: {month_names[best_month_season-1]} (Avg return: {monthly_seasonality.max():.2f}%)")
    logger.info(f"Worst month historically: {month_names[worst_month_season-1]} (Avg return: {monthly_seasonality.min():.2f}%)")
    
    # Day of week seasonality
    analyzed_data['DayOfWeek'] = analyzed_data.index.dayofweek
    dow_seasonality = analyzed_data.groupby('DayOfWeek')['Daily_Return'].mean()
    best_dow = dow_seasonality.idxmax()
    worst_dow = dow_seasonality.idxmin()
    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    logger.info(f"Best day historically: {day_names[best_dow]} (Avg return: {dow_seasonality.max():.2f}%)")
    logger.info(f"Worst day historically: {day_names[worst_dow]} (Avg return: {dow_seasonality.min():.2f}%)")
    
    # Trading Recommendations
    logger.info("\nTrading Recommendations:")
    logger.info("-" * 65)
    
    # Based on momentum scores
    if len(monthly_stats) > 0:
        strong_months = monthly_stats[monthly_stats['Momentum_Score'] >= 60].index.tolist()
        if strong_months:
            logger.info(f"Strong momentum months (score >= 60): {', '.join(str(m) for m in strong_months)}")
        
        weak_months = monthly_stats[monthly_stats['Momentum_Score'] < 40].index.tolist()
        if weak_months:
            logger.info(f"Weak momentum months (score < 40): {', '.join(str(m) for m in weak_months)}")
    
    # Based on positive day percentage
    if len(monthly_stats) > 0:
        high_positive_months = monthly_stats[monthly_stats['Percentage'] >= 60].index.tolist()
        if high_positive_months:
            logger.info(f"High positive day months (>= 60%): {', '.join(str(m) for m in high_positive_months)}")
    
    # Risk Assessment
    logger.info("\nRisk Assessment:")
    logger.info("-" * 65)
    volatility = analyzed_data['Daily_Return'].std()
    if volatility < 1:
        logger.info("Low volatility stock (suitable for conservative investors)")
    elif volatility < 2:
        logger.info("Moderate volatility stock (suitable for balanced investors)")
    else:
        logger.info("High volatility stock (suitable for aggressive investors)")
    
    # Calculate maximum drawdown
    cumulative_returns = (1 + analyzed_data['Daily_Return']/100).cumprod()
    running_max = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - running_max) / running_max * 100
    max_drawdown = drawdown.min()
    logger.info(f"Maximum drawdown: {max_drawdown:.2f}%")
    
    # Save insights to file
    monthly_seasonality = analyzed_data.groupby('Month')['Daily_Return'].mean()
    dow_seasonality = analyzed_data.groupby('DayOfWeek')['Daily_Return'].mean()
    quarter_seasonality = analyzed_data.groupby('Quarter')['Daily_Return'].mean()
    save_insights_to_file(symbol, analyzed_data, monthly_stats, monthly_seasonality, dow_seasonality, quarter_seasonality)

def process_stock(ticker):
    """Process a single stock symbol"""
    try:
        # Ensure ticker is a string
        if isinstance(ticker, tuple):
            ticker = ticker[0] if ticker else ""
        
        if not ticker:
            logger.warning("Empty ticker symbol")
            return False
        
        # Fetch data with caching
        stock_data = fetch_stock_data(ticker)
        
        # Analyze positive days
        analyzed_data = analyze_positive_days(stock_data)
        
        # Check if we have valid data
        if len(analyzed_data) == 0:
            logger.warning(f"No valid data for {ticker}")
            return False
        
        # Monthly breakdown with momentum
        monthly_stats = monthly_breakdown(analyzed_data)
        
        # Seasonality analysis
        symbol = ticker.replace('.NS', '')
        monthly_returns, dow_returns, quarter_returns = analyze_seasonality(analyzed_data, symbol)
        
        # Create simplified visualizations
        create_visualizations(analyzed_data, monthly_stats, symbol)
        
        # Generate insights
        generate_insights(analyzed_data, monthly_stats, symbol)
        
        return True
    except Exception as e:
        logger.info(f"Error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_fetch_stocks(tickers, days=365):
    """Fetch data for multiple stocks in a single API call"""
    logger.info(f"Batch fetching data for {len(tickers)} stocks...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    try:
        # Ensure all tickers are strings
        string_tickers = []
        for ticker in tickers:
            if isinstance(ticker, tuple):
                string_tickers.append(ticker[0] if ticker else "")
            else:
                string_tickers.append(ticker)
        
        # Filter out empty tickers
        string_tickers = [t for t in string_tickers if t]
        
        if not string_tickers:
            logger.warning("No valid tickers to fetch")
            return False
        
        # Download all stocks at oncecls
        
        all_data = yf.download([t+'.NS' for t in string_tickers], start=start_date, end=end_date, progress=False)
        
        # Handle MultiIndex columns properly
        if isinstance(all_data.columns, pd.MultiIndex):
            # Get the price level (first level)
            price_levels = all_data.columns.get_level_values(0).unique()
            
            # Cache individual stocks
            for ticker in string_tickers:
                ticker_key = ticker + '.NS'
                
                # Create a new DataFrame for this ticker
                stock_data = pd.DataFrame(index=all_data.index)
                
                # Extract data for each price level
                for price in price_levels:
                    if (price, ticker_key) in all_data.columns:
                        stock_data[price] = all_data[(price, ticker_key)]
                
                # Cache the data
                cache_path = get_cache_path(ticker, days)
                with open(cache_path, 'wb') as f:
                    pickle.dump(stock_data, f)
        else:
            # Handle non-MultiIndex columns (fallback)
            for ticker in string_tickers:
                ticker_key = ticker + '.NS'
                if ticker_key in all_data['Close'].columns:
                    # Extract data for this ticker
                    stock_data = pd.DataFrame({
                        'Open': all_data['Open'][ticker_key],
                        'High': all_data['High'][ticker_key],
                        'Low': all_data['Low'][ticker_key],
                        'Close': all_data['Close'][ticker_key],
                        'Volume': all_data['Volume'][ticker_key]
                    })
                    
                    cache_path = get_cache_path(ticker, days)
                    with open(cache_path, 'wb') as f:
                        pickle.dump(stock_data, f)
        
        logger.info(f"Successfully batch fetched and cached data for {len(string_tickers)} stocks")
        return True
    except Exception as e:
        logger.info(f"Error in batch fetching: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main function to run the stock analysis"""
    # Ensure the output directory exists
    output_dir = "C:\\temp\\momentum\\"
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Output will be saved to: {output_dir}")
    
    # Read symbols from CSV
    df = pd.read_csv('N500.csv')
    
    # Debug: Print the first few symbols to check their type
    logger.info(f"First 5 symbols: {df['Symbol'].head()}")
    logger.info(f"Type of first symbol: {type(df['Symbol'].iloc[0])}")
    
    # Convert all symbols to strings
    symbols = []
    for symbol in df['Symbol']:
        if isinstance(symbol, tuple):
            symbols.append(symbol[0] if symbol else "")
        else:
            symbols.append(str(symbol))
    
    # Filter out empty symbols
    symbols = [s for s in symbols if s]
    #symbols= symbols[:10]
    
    # Option 1: Batch fetch data first (faster for large number of stocks)
    batch_size = 50  # Adjust based on API limits
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        batch_fetch_stocks(batch)
    
    # Option 2: Process stocks in parallel using ProcessPoolExecutor
    # Set the number of processes (adjust based on your system)
    max_workers = min(8, os.cpu_count() or 1)  # Use fewer processes than CPU cores for memory efficiency
    
    logger.info(f"Processing {len(symbols)} symbols using {max_workers} processes...")
    
    # Process symbols in parallel
    success_count = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(process_stock, symbol): symbol for symbol in symbols}
        
        # Process completed tasks
        for future in as_completed(future_to_symbol):
            symbol = future_to_symbol[future]
            try:
                if future.result():
                    success_count += 1
                    logger.info(f"Successfully processed {symbol} ({success_count}/{len(symbols)})")
            except Exception as e:
                logger.info(f"Error processing {symbol}: {e}")
    
    logger.info(f"\nAnalysis completed! Successfully processed {success_count}/{len(symbols)} symbols.")

if __name__ == "__main__":
    main()