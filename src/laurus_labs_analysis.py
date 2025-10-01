def generate_insights(data, monthly_stats, daily_returns):
    """
    Generate additional insights from the analysis including momentum analysis
    """
    print(f"\n{'='*65}")
    print(f"COMPREHENSIVE INSIGHTS WITH MOMENTUM ANALYSIS")
    print(f"{'='*65}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Average daily return: {daily_returns.mean():.2f}%")
    print(f"Standard deviation of returns: {daily_returns.std():.2f}%")
    print(f"Maximum daily gain: {daily_returns.max():.2f}%")
    print(f"Maximum daily loss: {daily_returns.min():.2f}%")
    print(f"Sharpe ratio (assuming 0% risk-free rate): {daily_returns.mean()/daily_returns.std():.3f}")
    
    # Volatility analysis
    print(f"\nVolatility Analysis:")
    print(f"Daily volatility: {daily_returns.std():.2f}%")
    print(f"Annualized volatility: {daily_returns.std() * np.sqrt(252):.2f}%")
    
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
    recent_data = data.tail(30)
    recent_positive_days = recent_data['Close_Greater_Than_Prev'].sum()
    recent_total_days = len(recent_data)
    recent_avg_return = recent_data['Daily_Return'].mean()
    
    print(f"\nRecent Trend (Last 30 trading days):")
    print(f"Positive days: {recent_positive_days} out of {recent_total_days} ({(recent_positive_days/recent_total_days)*100:.2f}%)")
    print(f"Average daily return: {recent_avg_return:.2f}%")
    
    # Estimate current month momentum (if we have recent data)
    if len(recent_data) > 0:
        current_month_positive_pct = (recent_positive_days / recent_total_days) * 100
        current_month_avg_return = recent_avg_return
        current_month_volatility = recent_data['Daily_Return'].std()
        
        # Calculate estimated current momentum components
        positive_score = (current_month_positive_pct / 100) * 40
        return_score = max(-25, min(25, (current_month_avg_return / 10) * 25))
        consistency_score = max(0, 10 - (current_month_volatility / 5) * 10)
        
        # We can't calculate price momentum without full month, so use partial estimate
        estimated_momentum = positive_score + return_score + consistency_score
        
        print(f"\nEstimated Current Month Momentum (based on last 30 days):")
        print(f"Partial momentum score: ~{estimated_momentum:.1f} (excluding price momentum component)")
    
    # Price momentum
    current_price = data['Close'].iloc[-1]
    price_30d_ago = data['Close'].iloc[-30] if len(data) >= 30 else data['Close'].iloc[0]
    price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
    
    print(f"\nPrice Momentum:")
    print(f"Current price: ₹{current_price:.2f}")
    print(f"30-day price change: {price_change_30d:.2f}%")
    
    # Risk metrics
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        var_95 = np.percentile(daily_returns, 5)
        print(f"\nRisk Metrics:")
        print(f"Value at Risk (95%): {var_95:.2f}%")
        print(f"Average negative return: {negative_returns.mean():.2f}%")
        print(f"Probability of loss: {len(negative_returns)/len(daily_returns)*100:.1f}%")
    
    # Momentum Score Interpretation
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Set the style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def fetch_and_analyze_stock(ticker="LAURUSLABS.NS", days=365):
    """
    Fetch and analyze stock data with proper MultiIndex handling
    """
    # Define the date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    print(f"Fetching {ticker} data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    try:
        # Fetch the stock data - this returns MultiIndex columns
        stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Check if data was successfully fetched
        if stock_data.empty:
            raise ValueError(f"No data found for {ticker}")
        
        # Handle MultiIndex columns - flatten if needed
        if isinstance(stock_data.columns, pd.MultiIndex):
            # For single ticker, we can flatten the MultiIndex
            stock_data.columns = stock_data.columns.droplevel(1)
        
        print(f"Data fetched successfully! Total records: {len(stock_data)}")
        print("\nFirst few rows:")
        print(stock_data.head())
        print(f"\nColumn structure: {list(stock_data.columns)}")
        
        return stock_data
        
    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()

def analyze_positive_days(stock_data):
    """
    Analyze days where close > previous close
    """
    # Make a copy to avoid modifying original data
    data = stock_data.copy()
    
    # Calculate previous close
    data['Prev_Close'] = data['Close'].shift(1)
    
    # Find where close > previous close
    data['Close_Greater_Than_Prev'] = data['Close'] > data['Prev_Close']
    
    # Calculate daily returns
    data['Daily_Return'] = (data['Close'] - data['Prev_Close']) / data['Prev_Close'] * 100
    
    # Remove the first row as it doesn't have a previous close
    data = data.dropna()
    
    # Count total occurrences where close > prev close
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
        # Normalize volatility (typical daily volatility ranges 1-5%)
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

def create_visualizations(data, monthly_stats,symbol):
    """
    Create comprehensive visualizations including momentum scores
    """
    fig = plt.figure(figsize=(22, 18))
    
    # Create subplot layout
    gs = fig.add_gridspec(4, 3, hspace=0.35, wspace=0.3)
    
    # Plot 1: Monthly positive days count
    ax1 = fig.add_subplot(gs[0, 0])
    months = [str(period) for period in monthly_stats.index]
    bars = ax1.bar(months, monthly_stats['Positive_Days'], color='green', alpha=0.7)
    ax1.set_title('Monthly Positive Days Count', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Days')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, monthly_stats['Positive_Days']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(value)}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Plot 2: Monthly percentage trend
    ax2 = fig.add_subplot(gs[0, 1])
    line = ax2.plot(months, monthly_stats['Percentage'], marker='o', 
                   linewidth=2, markersize=6, color='blue')
    ax2.set_title('Monthly Positive Days %', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Month')
    ax2.set_ylabel('Percentage (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 100)
    
    # Add percentage labels
    for i, pct in enumerate(monthly_stats['Percentage']):
        ax2.annotate(f'{pct}%', (i, pct), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=8)
    
    # Plot 3: Momentum Scores
    ax3 = fig.add_subplot(gs[0, 2])
    
    # Color bars based on momentum score
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
    ax3.set_title('Monthly Momentum Scores', fontsize=9, fontweight='bold')
    ax3.set_xlabel('Month')
    ax3.set_ylabel('Momentum Score (0-100)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 100)
    
    # Add score labels
    for bar, score in zip(bars, monthly_stats['Momentum_Score']):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{score}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Add legend for momentum scores
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='darkgreen', label='Very Strong (80+)'),
        plt.Rectangle((0,0),1,1, facecolor='green', label='Strong (60-79)'),
        plt.Rectangle((0,0),1,1, facecolor='orange', label='Moderate (40-59)'),
        plt.Rectangle((0,0),1,1, facecolor='red', label='Weak (20-39)'),
        plt.Rectangle((0,0),1,1, facecolor='darkred', label='Very Weak (<20)')
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=7)
    
    # Plot 4: Monthly Returns vs Momentum Score (Scatter)
    ax4 = fig.add_subplot(gs[1, 0])
    scatter = ax4.scatter(monthly_stats['Monthly_Return'], monthly_stats['Momentum_Score'], 
                         c=monthly_stats['Momentum_Score'], cmap='RdYlGn', s=100, alpha=0.7)
    ax4.set_title('Monthly Return vs Momentum Score', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Monthly Return (%)')
    ax4.set_ylabel('Momentum Score')
    ax4.grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=ax4, label='Momentum Score')
    
    # Add month labels to points
    for i, month in enumerate(months):
        ax4.annotate(month.split('-')[1], 
                    (monthly_stats['Monthly_Return'].iloc[i], monthly_stats['Momentum_Score'].iloc[i]),
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    import matplotlib.dates as mdates
    from mplfinance.original_flavor import candlestick_ohlc

    # Prepare OHLC data for plotting
    ohlc = data [['Open', 'High', 'Low', 'Close']].copy()
    ohlc['Date'] = mdates.date2num(ohlc.index)   # convert datetime index to matplotlib float
    ohlc = ohlc[['Date', 'Open', 'High', 'Low', 'Close']]
    print(ohlc)

    import matplotlib.dates as mdates
    from matplotlib.patches import Rectangle

    # Plot 5: Stock price trend with volume
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5_vol = ax5.twinx()

    # Prepare OHLC
    try:
        subset = data.iloc[-90:]
        ohlc = subset[['Open', 'High', 'Low', 'Close']].copy()
        ohlc['Date'] = mdates.date2num(ohlc.index)
        
        ohlc = ohlc[['Date', 'Open', 'High', 'Low', 'Close']].values

        candle_width = 0.8  # width in days for daily data
        candle_col_up = "green"
        candle_col_down = "red"

        for date, open_, high, low, close in ohlc:
            color = candle_col_up if close >= open_ else candle_col_down
            
            # Wick (high-low line)
            ax5.plot([date, date], [low, high], color=color, linewidth=1)
            
            # Candle body (rectangle)
            rect = Rectangle((date - candle_width/2, min(open_, close)),     candle_width,        abs(close - open_),        facecolor=color,        edgecolor=color,        alpha=0.9    )
            ax5.add_patch(rect)

        # Format x-axis
        ax5.xaxis_date()
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax5.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        # After plotting candles
        ax5.set_xlim(ohlc[:,0].min(), ohlc[:,0].max())


        ax5.set_ylabel("Price (₹)", color="purple")
        ax5.tick_params(axis="y", labelcolor="purple")

        # Volume bars
        ax5_vol.bar(data.index, data["Volume"], alpha=0.3, color="gray", width=1)
        ax5_vol.set_ylabel("Volume", color="gray")
        ax5_vol.tick_params(axis="y", labelcolor="gray")

        ax5.set_title("Stock Price Trend with Volume", fontsize=11, fontweight="bold")
        ax5.set_xlabel("Date")
        ax5.grid(True, alpha=0.3)
    except Exception as e:
        print(e)
        pass


    
    # Plot 6: Daily returns distribution
    ax6 = fig.add_subplot(gs[2, 0])
    daily_returns = data['Daily_Return'].dropna()
    n, bins, patches = ax6.hist(daily_returns, bins=30, alpha=0.7, color='orange', edgecolor='black')
    
    # Color bars based on positive/negative returns
    for i, (patch, bin_val) in enumerate(zip(patches, bins[:-1])):
        if bin_val >= 0:
            patch.set_facecolor('green')
            patch.set_alpha(0.7)
        else:
            patch.set_facecolor('red')
            patch.set_alpha(0.7)
    
    ax6.set_title('Daily Returns Distribution', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Daily Return (%)')
    ax6.set_ylabel('Frequency')
    ax6.axvline(x=0, color='black', linestyle='--', alpha=0.8, label='No Change')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Cumulative returns
    ax7 = fig.add_subplot(gs[2, 1])
    cumulative_returns = (1 + data['Daily_Return']/100).cumprod() - 1
    ax7.plot(data.index, cumulative_returns * 100, linewidth=2, color='darkgreen')
    ax7.set_title('Cumulative Returns', fontsize=11, fontweight='bold')
    ax7.set_xlabel('Date')
    ax7.set_ylabel('Cumulative Return (%)')
    ax7.grid(True, alpha=0.3)
    ax7.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    
    # Plot 8: Momentum Score Trend
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.plot(range(len(months)), monthly_stats['Momentum_Score'], 
             marker='s', linewidth=2, markersize=8, color='darkblue')
    ax8.set_title('Momentum Score Trend', fontsize=11, fontweight='bold')
    ax8.set_xlabel('Month')
    ax8.set_ylabel('Momentum Score')
    ax8.set_xticks(range(len(months)))
    ax8.set_xticklabels(months, rotation=45)
    ax8.grid(True, alpha=0.3)
    ax8.set_ylim(0, 100)
    
    # Add horizontal lines for momentum thresholds
    ax8.axhline(y=80, color='darkgreen', linestyle='--', alpha=0.7, label='Very Strong')
    ax8.axhline(y=60, color='green', linestyle='--', alpha=0.7, label='Strong')
    ax8.axhline(y=40, color='orange', linestyle='--', alpha=0.7, label='Moderate')
    ax8.axhline(y=20, color='red', linestyle='--', alpha=0.7, label='Weak')
    ax8.legend(fontsize=7)
    
    # Plot 9: Momentum Components Breakdown (for latest month)
    ax9 = fig.add_subplot(gs[3, :])
    
    # Calculate components for each month
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
    
    # Stacked bar chart
    bottom1 = components_df['Positive_Days']
    bottom2 = bottom1 + components_df['Avg_Return']
    bottom3 = bottom2 + components_df['Price_Momentum']
    
    ax9.bar(components_df['Month'], components_df['Positive_Days'], 
           label='Positive Days (40 pts)', color='lightblue', alpha=0.8)
    ax9.bar(components_df['Month'], components_df['Avg_Return'], bottom=bottom1,
           label='Avg Return (±25 pts)', color='lightgreen', alpha=0.8)
    ax9.bar(components_df['Month'], components_df['Price_Momentum'], bottom=bottom2,
           label='Price Momentum (±25 pts)', color='gold', alpha=0.8)
    ax9.bar(components_df['Month'], components_df['Consistency'], bottom=bottom3,
           label='Consistency (10 pts)', color='lightcoral', alpha=0.8)
    
    ax9.set_title('Monthly Momentum Score Components Breakdown', fontsize=11, fontweight='bold')
    ax9.set_xlabel('Month')
    ax9.set_ylabel('Score Points')
    ax9.tick_params(axis='x', rotation=45)
    ax9.legend(loc='upper right', fontsize=9)
    ax9.grid(True, alpha=0.3)
   
    
    plt.suptitle(symbol+'- Comprehensive Analysis with Momentum Scoring', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.savefig("C:\\temp\\"+symbol+".png")
    print('wrote', symbol)
    
    return daily_returns

def generate_insights(data, monthly_stats, daily_returns):
    """
    Generate additional insights from the analysis
    """
    print(f"\n{'='*50}")
    print(f"COMPREHENSIVE INSIGHTS")
    print(f"{'='*50}")
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(f"Average daily return: {daily_returns.mean():.2f}%")
    print(f"Standard deviation of returns: {daily_returns.std():.2f}%")
    print(f"Maximum daily gain: {daily_returns.max():.2f}%")
    print(f"Maximum daily loss: {daily_returns.min():.2f}%")
    print(f"Sharpe ratio (assuming 0% risk-free rate): {daily_returns.mean()/daily_returns.std():.3f}")
    
    # Volatility analysis
    print(f"\nVolatility Analysis:")
    print(f"Daily volatility: {daily_returns.std():.2f}%")
    print(f"Annualized volatility: {daily_returns.std() * np.sqrt(252):.2f}%")
    
    # Best and worst performing months
    best_month = monthly_stats.loc[monthly_stats['Percentage'].idxmax()]
    worst_month = monthly_stats.loc[monthly_stats['Percentage'].idxmin()]
    
    print(f"\nMonthly Performance:")
    print(f"Best month: {monthly_stats['Percentage'].idxmax()} ({best_month['Percentage']:.2f}% positive days)")
    print(f"Worst month: {monthly_stats['Percentage'].idxmin()} ({worst_month['Percentage']:.2f}% positive days)")
    
    # Recent trend analysis
    recent_data = data.tail(30)
    recent_positive_days = recent_data['Close_Greater_Than_Prev'].sum()
    recent_total_days = len(recent_data)
    recent_avg_return = recent_data['Daily_Return'].mean()
    
    print(f"\nRecent Trend (Last 30 trading days):")
    print(f"Positive days: {recent_positive_days} out of {recent_total_days} ({(recent_positive_days/recent_total_days)*100:.2f}%)")
    print(f"Average daily return: {recent_avg_return:.2f}%")
    
    # Price momentum
    current_price = data['Close'].iloc[-1]
    price_30d_ago = data['Close'].iloc[-30] if len(data) >= 30 else data['Close'].iloc[0]
    price_change_30d = ((current_price - price_30d_ago) / price_30d_ago) * 100
    
    print(f"\nPrice Momentum:")
    print(f"Current price: ₹{current_price:.2f}")
    print(f"30-day price change: {price_change_30d:.2f}%")
    
    # Risk metrics
    negative_returns = daily_returns[daily_returns < 0]
    if len(negative_returns) > 0:
        var_95 = np.percentile(daily_returns, 5)
        print(f"\nRisk Metrics:")
        print(f"Value at Risk (95%): {var_95:.2f}%")
        print(f"Average negative return: {negative_returns.mean():.2f}%")
        print(f"Probability of loss: {len(negative_returns)/len(daily_returns)*100:.1f}%")

def main():
    """
    Main function to run the complete analysis
    """
    print("Laurus Labs Stock Analysis with MultiIndex Handling")
    print("="*55)
    
    # Fetch stock data
    df = pd.read_csv('N500.csv')
    lst = df['Symbol']
    import time
    for symbol in lst[495:500]:
        time.sleep(1)
        stock_data = fetch_and_analyze_stock(symbol+".NS", 365)
    
        if stock_data.empty:
            print("Failed to fetch data. Exiting...")
            continue
        
        # Analyze positive days
        analyzed_data = analyze_positive_days(stock_data)
    
        # Monthly breakdown
        monthly_stats = monthly_breakdown(analyzed_data)
    
        # Create visualizations
        try:
            daily_returns = create_visualizations(analyzed_data, monthly_stats,symbol)
        except Exception as e:
            print(symbol, e)
            pass
    
        # Generate insights
        generate_insights(analyzed_data, monthly_stats, daily_returns)

if __name__ == "__main__":
    main()