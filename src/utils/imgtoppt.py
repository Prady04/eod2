import os
import re
import base64
from datetime import datetime
from collections import defaultdict
import calendar
import json
import statistics
import yfinance as yf
import pandas as pd
import numpy as np
import concurrent.futures
import time
from functools import partial

def parse_insights_file(insights_path):
    """Parse insights file with comprehensive pattern matching."""
    try:
        with open(insights_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        print(f"    📄 Parsing: {os.path.basename(insights_path)}")
        insights = {}
        lines = content.split('\n')
        
        # Extract stock name from header
        header_match = re.search(r'STOCK ANALYSIS INSIGHTS FOR (\w+)', content)
        if header_match:
            insights['symbol'] = header_match.group(1)
        
        # Extract generation date
        date_match = re.search(r'Generated on: (\d{4}-\d{2}-\d{2})', content)
        if date_match:
            insights['date'] = date_match.group(1)
        
        # Parse overall statistics
        total_days_match = re.search(r'Total trading days: (\d+)', content)
        if total_days_match:
            insights['total_days'] = int(total_days_match.group(1))
        
        positive_days_match = re.search(r'Days where Close > Previous Close: (\d+)', content)
        if positive_days_match:
            insights['positive_days'] = int(positive_days_match.group(1))
        
        percentage_match = re.search(r'Percentage: ([\d.]+)%', content)
        if percentage_match:
            insights['positive_pct'] = float(percentage_match.group(1))
        
        avg_return_match = re.search(r'Average Daily Return: ([+-]?[\d.]+)%', content)
        if avg_return_match:
            insights['avg_daily_return'] = float(avg_return_match.group(1))
        
        volatility_match = re.search(r'Daily Return Volatility: ([\d.]+)%', content)
        if volatility_match:
            insights['volatility'] = float(volatility_match.group(1))
        
        max_return_match = re.search(r'Maximum Daily Return: ([+-]?[\d.]+)%', content)
        if max_return_match:
            insights['max_daily_return'] = float(max_return_match.group(1))
        
        min_return_match = re.search(r'Minimum Daily Return: ([+-]?[\d.]+)%', content)
        if min_return_match:
            insights['min_daily_return'] = float(min_return_match.group(1))
        
        # Extract monthly breakdown
        monthly_breakdown = []
        monthly_pattern = r'(\d{4}-\d{2})\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([+-]?[\d.]+)\s+([\d.]+)\s+(\w+)'
        for match in re.finditer(monthly_pattern, content):
            monthly_breakdown.append({
                'month': match.group(1),
                'positive_days': int(match.group(2)),
                'total_days': int(match.group(3)),
                'percentage': float(match.group(4)),
                'monthly_return': float(match.group(5)),
                'momentum_score': float(match.group(6)),
                'momentum_rating': match.group(7)
            })
        
        if monthly_breakdown:
            insights['monthly_breakdown'] = monthly_breakdown
        
        # Extract best and worst performing months
        best_month_match = re.search(r'Best performing month: (\d{4}-\d{2}) \(Momentum Score: ([\d.]+)\)', content)
        if best_month_match:
            insights['best_month'] = {
                'month': best_month_match.group(1),
                'momentum_score': float(best_month_match.group(2))
            }
        
        worst_month_match = re.search(r'Worst performing month: (\d{4}-\d{2}) \(Momentum Score: ([\d.]+)\)', content)
        if worst_month_match:
            insights['worst_month'] = {
                'month': worst_month_match.group(1),
                'momentum_score': float(worst_month_match.group(2))
            }
        
        # Extract monthly seasonality
        monthly_seasonality = {}
        month_pattern = r'(\w{3})\s+\|\s+Avg Return:\s+([+-]?[\d.]+)%'
        for match in re.finditer(month_pattern, content):
            month_name = match.group(1)
            avg_return = float(match.group(2))
            monthly_seasonality[month_name] = avg_return
        
        if monthly_seasonality:
            insights['monthly_seasonality'] = monthly_seasonality
        
        # Extract day of week seasonality (excluding weekends)
        dow_seasonality = {}
        dow_pattern = r'(Monday|Tuesday|Wednesday|Thursday|Friday)\s+\|\s+Avg Return:\s+([+-]?[\d.]+)%'
        for match in re.finditer(dow_pattern, content):
            day_name = match.group(1)
            avg_return = float(match.group(2))
            dow_seasonality[day_name] = avg_return
        
        if dow_seasonality:
            insights['dow_seasonality'] = dow_seasonality
        
        # Extract trading recommendations with better pattern matching
        best_months_match = re.search(r'Best months to trade[^:]*:\s*([^\n]+)', content)
        if best_months_match:
            best_months_text = best_months_match.group(1).strip()
            # Clean up the text and split by commas
            best_months = [m.strip() for m in re.split(r',\s*', best_months_text) if m.strip()]
            insights['best_months'] = best_months
        else:
            insights['best_months'] = []  # Default empty list
        
        best_days_match = re.search(r'Best days to trade[^:]*:\s*([^\n]+)', content)
        if best_days_match:
            best_days_text = best_days_match.group(1).strip()
            # Clean up the text and split by commas
            best_days = [d.strip() for d in re.split(r',\s*', best_days_text) if d.strip()]
            # Filter out weekends
            best_days = [d for d in best_days if d not in ['Saturday', 'Sunday']]
            insights['best_days'] = best_days
        else:
            insights['best_days'] = []  # Default empty list
        
        # Extract risk assessment
        risk_assessment_match = re.search(r'High volatility stock \(([^)]+)\)', content)
        if risk_assessment_match:
            insights['risk_profile'] = risk_assessment_match.group(1)
        
        max_drawdown_match = re.search(r'Maximum drawdown: ([+-]?[\d.]+)%', content)
        if max_drawdown_match:
            insights['max_drawdown'] = float(max_drawdown_match.group(1))
        
        return insights if len(insights) > 1 else {'raw': content[:400]}
        
    except Exception as e:
        print(f"    ❌ Error: {e}")
        return {'error': str(e)}

def calculate_relative_strength(symbol, benchmark_symbol="^CRSLDX", period="3mo", timeout=5):
    """Calculate relative strength of a symbol compared to benchmark with timeout."""
    try:
        print(f"    📊 Calculating RS for {symbol} vs {benchmark_symbol}")
        
        # Try to get data with multiple fallback methods
        data = None
        benchmark_data = None
        
        # Method 1: Direct yfinance download
        try:
            ticker = yf.Ticker(symbol+".NS")
            benchmark = yf.Ticker(benchmark_symbol)
            
            data = ticker.history(period=period, timeout=timeout)
            benchmark_data = benchmark.history(period=period, timeout=timeout)
        except Exception as e:
            print(f"    ⚠️  Method 1 failed for {symbol}: {str(e)[:50]}")
        
        # Method 2: Using session with shorter timeout
        if data is None or data.empty or benchmark_data is None or benchmark_data.empty:
            try:
                session = yf.utils.get_ticker_session(timeout=timeout)
                ticker = yf.Ticker(symbol, session=session)
                benchmark = yf.Ticker(benchmark_symbol, session=session)
                
                data = ticker.history(period=period, timeout=timeout)
                benchmark_data = benchmark.history(period=period, timeout=timeout)
            except Exception as e:
                print(f"    ⚠️  Method 2 failed for {symbol}: {str(e)[:50]}")
        
        # Method 3: Use cached data if available
        if data is None or data.empty or benchmark_data is None or benchmark_data.empty:
            print(f"    ⚠️  No data available for {symbol}")
            return None
        
        # Calculate returns
        data['Returns'] = data['Close'].pct_change()
        benchmark_data['Returns'] = benchmark_data['Close'].pct_change()
        
        # Align data
        aligned_data = pd.concat([data['Returns'], benchmark_data['Returns']], axis=1, join='inner')
        aligned_data.columns = ['Stock_Returns', 'Benchmark_Returns']
        
        if aligned_data.empty:
            print(f"    ⚠️  No aligned data for {symbol}")
            return None
        
        # Calculate relative strength (stock return - benchmark return)
        aligned_data['Relative_Return'] = aligned_data['Stock_Returns'] - aligned_data['Benchmark_Returns']
        
        # Calculate cumulative returns
        aligned_data['Stock_Cumulative'] = (1 + aligned_data['Stock_Returns']).cumprod() - 1
        aligned_data['Benchmark_Cumulative'] = (1 + aligned_data['Benchmark_Returns']).cumprod() - 1
        aligned_data['Relative_Cumulative'] = aligned_data['Stock_Cumulative'] - aligned_data['Benchmark_Cumulative']
        
        # Calculate RS metrics
        total_stock_return = aligned_data['Stock_Cumulative'].iloc[-1]
        total_benchmark_return = aligned_data['Benchmark_Cumulative'].iloc[-1]
        total_relative_return = aligned_data['Relative_Cumulative'].iloc[-1]
        
        # Calculate RS ratio (stock performance / benchmark performance)
        rs_ratio = (1 + total_stock_return) / (1 + total_benchmark_return) if (1 + total_benchmark_return) != 0 else 1
        
        # Calculate RS momentum (average of last 3 months relative returns)
        last_90_days = aligned_data.tail(90)
        rs_momentum = last_90_days['Relative_Return'].mean() * 252  # Annualized
        
        # Calculate RS volatility
        rs_volatility = aligned_data['Relative_Return'].std() * np.sqrt(252)
        
        # Calculate RS score (combination of return and momentum)
        rs_score = (total_relative_return * 0.4) + (rs_momentum * 0.3) + (rs_ratio * 0.3)
        
        # Determine RS rating
        if rs_score > 0.1:
            rs_rating = "Strong Outperform"
        elif rs_score > 0.05:
            rs_rating = "Moderate Outperform"
        elif rs_score > -0.05:
            rs_rating = "Neutral"
        elif rs_score > -0.1:
            rs_rating = "Moderate Underperform"
        else:
            rs_rating = "Strong Underperform"
        
        return {
            'symbol': symbol,
            'benchmark': benchmark_symbol,
            'total_stock_return': total_stock_return,
            'total_benchmark_return': total_benchmark_return,
            'total_relative_return': total_relative_return,
            'rs_ratio': rs_ratio,
            'rs_momentum': rs_momentum,
            'rs_volatility': rs_volatility,
            'rs_score': rs_score,
            'rs_rating': rs_rating,
            'period': period
        }
        
    except Exception as e:
        print(f"    ❌ Error calculating RS for {symbol}: {str(e)[:100]}")
        return None

def calculate_relative_strength_batch(symbols, benchmark_symbol="^CRSLDX", period="3mo", max_workers=3):
    """Calculate relative strength for multiple symbols in parallel."""
    print(f"\n📊 Processing {len(symbols)} symbols for RS analysis (using {max_workers} workers)...")
    
    # Create partial function with fixed parameters
    rs_func = partial(calculate_relative_strength, 
                     benchmark_symbol=benchmark_symbol, 
                     period=period, 
                     timeout=5)
    
    results = []
    
    # Use ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {executor.submit(rs_func, symbol): symbol for symbol in symbols}
        
        # Process completed tasks
        for i, future in enumerate(concurrent.futures.as_completed(future_to_symbol, timeout=30)):
            symbol = future_to_symbol[future]
            try:
                result = future.result(timeout=10)
                if result:
                    results.append(result)
                    print(f"    ✅ [{i+1}/{len(symbols)}] {symbol}: RS Score {result['rs_score']:.3f} ({result['rs_rating']})")
                else:
                    print(f"    ⚠️  [{i+1}/{len(symbols)}] {symbol}: No data")
            except Exception as e:
                print(f"    ❌ [{i+1}/{len(symbols)}] {symbol}: {str(e)[:50]}")
    
    return results

def encode_image_to_base64(image_path):
    """Convert image to base64 for embedding in HTML."""
    try:
        with open(image_path, 'rb') as img_file:
            return base64.b64encode(img_file.read()).decode('utf-8')
    except Exception as e:
        print(f"    ❌ Image encoding error: {e}")
        return None

def find_insights_file(folder_path, png_file):
    """Find corresponding insights file."""
    base_name = os.path.splitext(png_file)[0]
    stock_name = base_name.replace('_analysis', '').replace('_chart', '').replace('_dashboard', '')
    
    candidates = [
        f"{stock_name}_insights.txt",
        f"{stock_name}.txt",
        f"{base_name}_insights.txt",
    ]
    
    for candidate in candidates:
        full_path = os.path.join(folder_path, candidate)
        if os.path.exists(full_path):
            return full_path
    
    # Case-insensitive search
    try:
        all_files = os.listdir(folder_path)
        stock_lower = stock_name.lower()
        
        for file in all_files:
            if file.lower().endswith('.txt'):
                file_base = file.lower().replace('_insights.txt', '').replace('.txt', '')
                if file_base == stock_lower or stock_lower in file_base:
                    return os.path.join(folder_path, file)
    except:
        pass
    
    return None

def analyze_seasonality(insights_data):
    """Analyze monthly and day-of-week seasonality across all stocks."""
    print("\n" + "="*60)
    print("📈 SEASONALITY ANALYSIS")
    print("="*60)
    
    # Monthly seasonality
    monthly_data = defaultdict(list)
    for file_name, insights in insights_data.items():
        if 'monthly_seasonality' in insights:
            for month, return_val in insights['monthly_seasonality'].items():
                monthly_data[month].append(return_val)
    
    monthly_stats = {}
    for month, returns in monthly_data.items():
        if returns:
            monthly_stats[month] = {
                'avg_return': statistics.mean(returns),
                'max_positive': max(returns),
                'max_negative': min(returns),
                'count': len(returns),
                'std_return': statistics.stdev(returns) if len(returns) > 1 else 0
            }
    
    # Day of week seasonality (excluding weekends)
    dow_data = defaultdict(list)
    for file_name, insights in insights_data.items():
        if 'dow_seasonality' in insights:
            for day, return_val in insights['dow_seasonality'].items():
                dow_data[day].append(return_val)
    
    dow_stats = {}
    for day, returns in dow_data.items():
        if returns:
            dow_stats[day] = {
                'avg_return': statistics.mean(returns),
                'max_positive': max(returns),
                'max_negative': min(returns),
                'count': len(returns),
                'std_return': statistics.stdev(returns) if len(returns) > 1 else 0
            }
    
    # Find best and worst months/days
    best_month = max(monthly_stats.items(), key=lambda x: x[1]['avg_return']) if monthly_stats else None
    worst_month = min(monthly_stats.items(), key=lambda x: x[1]['avg_return']) if monthly_stats else None
    
    best_day = max(dow_stats.items(), key=lambda x: x[1]['avg_return']) if dow_stats else None
    worst_day = min(dow_stats.items(), key=lambda x: x[1]['avg_return']) if dow_stats else None
    
    print(f"  📊 Monthly seasonality: {len(monthly_stats)} months")
    print(f"  📊 Day-of-week seasonality: {len(dow_stats)} days")
    
    if best_month:
        print(f"  🏆 Best month: {best_month[0]} ({best_month[1]['avg_return']:.2f}%)")
    if worst_month:
        print(f"  📉 Worst month: {worst_month[0]} ({worst_month[1]['avg_return']:.2f}%)")
    if best_day:
        print(f"  🏆 Best day: {best_day[0]} ({best_day[1]['avg_return']:.2f}%)")
    if worst_day:
        print(f"  📉 Worst day: {worst_day[0]} ({worst_day[1]['avg_return']:.2f}%)")
    
    print(f"\n✅ Analysis complete")
    print("="*60 + "\n")
    
    return {
        'monthly': monthly_stats,
        'dow': dow_stats,
        'best_month': best_month,
        'worst_month': worst_month,
        'best_day': best_day,
        'worst_day': worst_day
    }

def analyze_risk_profile(insights_data):
    """Analyze risk profiles across all stocks."""
    print("\n" + "="*60)
    print("🔍 RISK PROFILE ANALYSIS")
    print("="*60)
    
    risk_data = {
        'high_volatility': [],
        'medium_volatility': [],
        'low_volatility': [],
        'max_drawdown': [],
        'avg_daily_return': [],
        'volatility': [],
        'consistent_sd': [],
        'movement_range': []
    }
    
    for file_name, insights in insights_data.items():
        if 'volatility' in insights:
            volatility = insights['volatility']
            risk_data['volatility'].append(volatility)
            
            if volatility > 3.0:
                risk_data['high_volatility'].append(file_name)
            elif volatility > 1.5:
                risk_data['medium_volatility'].append(file_name)
            else:
                risk_data['low_volatility'].append(file_name)
        
        if 'max_drawdown' in insights:
            risk_data['max_drawdown'].append(insights['max_drawdown'])
        
        if 'avg_daily_return' in insights:
            risk_data['avg_daily_return'].append(insights['avg_daily_return'])
        
        # Check for consistent 1+ standard deviation
        if 'volatility' in insights and 'avg_daily_return' in insights:
            # If volatility is 1% or more, it's at least 1 standard deviation
            if insights['volatility'] >= 1.0:
                risk_data['consistent_sd'].append({
                    'file': file_name,
                    'symbol': insights.get('symbol', 'Unknown'),
                    'volatility': insights['volatility']
                })
        
        # Calculate movement range (max - min daily return)
        if 'max_daily_return' in insights and 'min_daily_return' in insights:
            movement_range = insights['max_daily_return'] - insights['min_daily_return']
            risk_data['movement_range'].append({
                'file': file_name,
                'symbol': insights.get('symbol', 'Unknown'),
                'range': movement_range,
                'max': insights['max_daily_return'],
                'min': insights['min_daily_return']
            })
    
    # Calculate statistics
    risk_stats = {}
    for key, values in risk_data.items():
        if key in ['consistent_sd', 'movement_range']:
            # These are already structured data
            risk_stats[key] = values
        elif values and isinstance(values[0], (int, float)):
            risk_stats[key] = {
                'avg': statistics.mean(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        else:
            risk_stats[key] = {
                'count': len(values),
                'items': values
            }
    
    # Find stock with highest movement
    if risk_stats['movement_range']:
        highest_movement = max(risk_stats['movement_range'], key=lambda x: x['range'])
        risk_stats['highest_movement'] = highest_movement
    
    print(f"  📊 High volatility stocks: {risk_stats['high_volatility']['count']}")
    print(f"  📊 Medium volatility stocks: {risk_stats['medium_volatility']['count']}")
    print(f"  📊 Low volatility stocks: {risk_stats['low_volatility']['count']}")
    print(f"  📊 Stocks with consistent 1+ SD movement: {len(risk_stats['consistent_sd'])}")
    
    if 'volatility' in risk_stats:
        print(f"  📊 Average volatility: {risk_stats['volatility']['avg']:.2f}%")
    
    if 'max_drawdown' in risk_stats:
        print(f"  📊 Average max drawdown: {risk_stats['max_drawdown']['avg']:.2f}%")
    
    if 'highest_movement' in risk_stats:
        print(f"  📊 Highest movement stock: {risk_stats['highest_movement']['symbol']} ({risk_stats['highest_movement']['range']:.2f}%)")
    
    print(f"\n✅ Risk analysis complete")
    print("="*60 + "\n")
    
    return risk_stats

def analyze_relative_strength(all_insights):
    """Analyze relative strength across all stocks with optimized processing."""
    print("\n" + "="*60)
    print("📊 RELATIVE STRENGTH ANALYSIS")
    print("="*60)
    
    # Extract unique symbols
    symbols = []
    for file_name, insights in all_insights.items():
        if 'symbol' in insights:
            symbol = insights['symbol']
            if symbol not in symbols:
                symbols.append(symbol)
    
    print(f"  📊 Found {len(symbols)} unique symbols")
    
    # Process in smaller batches to avoid overwhelming the API
    batch_size = 5  # Reduced batch size
    all_rs_data = []
    
    for i in range(0, len(symbols), batch_size):
        batch = symbols[i:i+batch_size]
        print(f"\n  📊 Processing batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1} ({len(batch)} symbols)")
        
        # Calculate RS for batch
        batch_results = calculate_relative_strength_batch(batch, max_workers=2)  # Reduced workers
        all_rs_data.extend(batch_results)
        
        # Longer delay between batches
        if i + batch_size < len(symbols):
            time.sleep(2)
    
    # Sort by RS score
    all_rs_data.sort(key=lambda x: x['rs_score'], reverse=True)
    
    # Calculate statistics
    if all_rs_data:
        avg_rs_score = statistics.mean([rs['rs_score'] for rs in all_rs_data])
        best_rs = all_rs_data[0]
        worst_rs = all_rs_data[-1]
        
        print(f"\n  📊 Processed {len(all_rs_data)} symbols successfully")
        print(f"  📊 Average RS Score: {avg_rs_score:.3f}")
        print(f"  🏆 Best RS: {best_rs['symbol']} ({best_rs['rs_score']:.3f})")
        print(f"  📉 Worst RS: {worst_rs['symbol']} ({worst_rs['rs_score']:.3f})")
    else:
        print(f"\n  ⚠️  No RS data calculated")
    
    print(f"\n✅ RS analysis complete")
    print("="*60 + "\n")
    
    return all_rs_data

def generate_html_dashboard(folder_path, output_name="dashboard.html"):
    """Generate interactive HTML dashboard."""
    
    if not os.path.exists(folder_path):
        print(f"❌ Folder not found: {folder_path}")
        return
    
    png_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith('.png')])
    
    if not png_files:
        print(f"❌ No PNG files found")
        return
    
    print("\n" + "="*60)
    print("🚀 HTML DASHBOARD GENERATOR")
    print("="*60)
    print(f"📁 Folder: {folder_path}")
    print(f"🖼️  Images: {len(png_files)}")
    print("="*60 + "\n")
    
    all_insights = {}
    
    # Process all images and insights
    for idx, png_file in enumerate(png_files, 1):
        print(f"[{idx}/{len(png_files)}] Processing: {png_file}")
        
        insights_file = find_insights_file(folder_path, png_file)
        
        if insights_file:
            insights = parse_insights_file(insights_file)
            all_insights[png_file] = insights
        else:
            print(f"  ⚠️  No insights file")
            insights = {
                'symbol': png_file[:10].upper(),
                'date': datetime.now().strftime('%Y-%m-%d'),
                'best_months': [],  # Ensure default
                'best_days': []     # Ensure default
            }
            all_insights[png_file] = insights
            #from itertools import islice
            #all_insights = dict(islice( all_insights.items(), 10))
    
    # Analyze seasonality
    seasonality_stats = analyze_seasonality(all_insights)
    
    # Analyze risk profiles
    risk_stats = analyze_risk_profile(all_insights)
    
    # Analyze relative strength
    rs_data = analyze_relative_strength(all_insights)
    
    # Add RS data to insights
    for rs in rs_data:
        for file_name, insights in all_insights.items():
            if insights.get('symbol') == rs['symbol']:
                insights['relative_strength'] = rs
                break
    
    # Generate HTML
    html_content = generate_html_content(folder_path, png_files, all_insights, seasonality_stats, risk_stats, rs_data)
    
    # Save HTML
    output_path = os.path.join(folder_path, output_name)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("\n" + "="*60)
    print("✅ SUCCESS!")
    print(f"📄 File: {output_path}")
    print(f"🌐 Open in browser to view dashboard")
    print("="*60 + "\n")

def generate_html_content(folder_path, png_files, all_insights, seasonality_stats, risk_stats, rs_data):
    """Generate the complete HTML content."""
    
    today = datetime.now().strftime("%B %d, %Y")
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Market Analysis Dashboard</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: #fff;
            line-height: 1.6;
        }}
        
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 20px;
        }}
        
        header {{
            text-align: center;
            padding: 40px 20px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        }}
        
        h1 {{
            font-size: 3em;
            color: #ffd700;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        }}
        
        .subtitle {{
            font-size: 1.5em;
            color: #96c8ff;
            margin-bottom: 5px;
        }}
        
        .date {{
            color: #888;
            font-size: 1.1em;
        }}
        
        .nav-buttons {{
            display: flex;
            justify-content: center;
            gap: 15px;
            margin: 30px 0;
            flex-wrap: wrap;
        }}
        
        .nav-btn {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 12px 30px;
            border-radius: 25px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }}
        
        .nav-btn:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }}
        
        .nav-btn.active {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }}
        
        .section {{
            display: none;
            animation: fadeIn 0.5s ease-in;
        }}
        
        .section.active {{
            display: block;
        }}
        
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(20px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        
        .chart-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(800px, 1fr));
            gap: 30px;
            margin-top: 20px;
        }}
        
        .chart-card {{
            background: linear-gradient(135deg, #1e1e30 0%, #2a2a40 100%);
            border-radius: 15px;
            padding: 20px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            border: 2px solid #3a3a50;
            transition: all 0.3s ease;
        }}
        
        .chart-card:hover {{
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(100, 180, 255, 0.3);
            border-color: #64b4ff;
        }}
        
        .chart-image {{
            width: 100%;
            border-radius: 10px;
            margin-bottom: 15px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        }}
        
        .insights-box {{
            background: linear-gradient(135deg, #141428 0%, #1a1a35 100%);
            border: 2px solid #64b4ff;
            border-radius: 12px;
            padding: 20px;
            margin-top: 15px;
        }}
        
        .insights-title {{
            font-size: 1.3em;
            color: #ffd700;
            margin-bottom: 12px;
            font-weight: bold;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        
        .insight-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            margin: 8px 0;
            padding: 10px;
            background: rgba(100, 180, 255, 0.05);
            border-radius: 8px;
        }}
        
        .insight-item {{
            flex: 1;
            min-width: 200px;
        }}
        
        .insight-label {{
            color: #96c8ff;
            font-size: 0.9em;
            margin-bottom: 3px;
        }}
        
        .insight-value {{
            font-size: 1.2em;
            font-weight: bold;
            color: #fff;
        }}
        
        .positive {{
            color: #4ade80 !important;
        }}
        
        .negative {{
            color: #ff6b6b !important;
        }}
        
        .neutral {{
            color: #ffd700 !important;
        }}
        
        .seasonality-table {{
            width: 100%;
            background: linear-gradient(135deg, #1e1e30 0%, #2a2a40 100%);
            border-radius: 15px;
            padding: 30px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.3);
            margin-top: 20px;
            overflow-x: auto;
        }}
        
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }}
        
        th {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: #ffd700;
            padding: 15px;
            text-align: left;
            font-size: 1.1em;
            border-bottom: 3px solid #3a3a50;
        }}
        
        td {{
            padding: 12px 15px;
            border-bottom: 1px solid #3a3a50;
            font-size: 1em;
        }}
        
        tr:hover {{
            background: rgba(100, 180, 255, 0.1);
        }}
        
        .stat-card {{
            background: linear-gradient(135deg, #2a2a40 0%, #1e1e30 100%);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 2px solid #3a3a50;
        }}
        
        .stat-value {{
            font-size: 2.5em;
            font-weight: bold;
            color: #64b4ff;
            margin: 10px 0;
        }}
        
        .stat-label {{
            color: #96c8ff;
            font-size: 1.1em;
        }}
        
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }}
        
        .watermark {{
            position: fixed;
            bottom: 20px;
            right: 20px;
            color: rgba(255,255,255,0.1);
            font-size: 1.5em;
            font-weight: bold;
            pointer-events: none;
        }}
        
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 20px 0;
        }}
        
        .risk-list {{
            margin-top: 20px;
            background: linear-gradient(135deg, #141428 0%, #1a1a35 100%);
            border-radius: 12px;
            padding: 20px;
        }}
        
        .risk-list h3 {{
            color: #ffd700;
            margin-bottom: 15px;
        }}
        
        .risk-item {{
            display: flex;
            justify-content: space-between;
            padding: 10px;
            margin: 5px 0;
            background: rgba(100, 180, 255, 0.05);
            border-radius: 8px;
            transition: all 0.3s ease;
        }}
        
        .risk-item:hover {{
            background: rgba(100, 180, 255, 0.1);
        }}
        
        .risk-symbol {{
            font-weight: bold;
            color: #64b4ff;
        }}
        
        .risk-value {{
            color: #fff;
        }}
        
        .highlight-box {{
            margin-top: 30px;
            padding: 20px;
            background: linear-gradient(135deg, #2a2a40 0%, #1e1e30 100%);
            border-radius: 12px;
            border: 2px solid #ffd700;
        }}
        
        .highlight-box h3 {{
            color: #ffd700;
            margin-bottom: 15px;
        }}
        
        .highlight-content {{
            font-size: 1.1em;
            color: #fff;
            margin: 10px 0;
        }}
        
        .rs-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.85em;
            font-weight: bold;
            margin-left: 10px;
        }}
        
        .rs-strong {{
            background: linear-gradient(135deg, #4ade80, #22c55e);
            color: #fff;
        }}
        
        .rs-moderate {{
            background: linear-gradient(135deg, #60a5fa, #3b82f6);
            color: #fff;
        }}
        
        .rs-neutral {{
            background: linear-gradient(135deg, #fbbf24, #f59e0b);
            color: #fff;
        }}
        
        .rs-weak {{
            background: linear-gradient(135deg, #f87171, #ef4444);
            color: #fff;
        }}
        
        @media (max-width: 900px) {{
            .chart-grid {{
                grid-template-columns: 1fr;
            }}
            
            h1 {{
                font-size: 2em;
            }}
            
            .subtitle {{
                font-size: 1.2em;
            }}
        }}
    </style>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Market Analysis Dashboard</h1>
            <div class="subtitle">{len(png_files)} Charts with Insights + RS Analysis</div>
            <div class="date">{today}</div>
        </header>
        
        <div class="nav-buttons">
            <button class="nav-btn active" onclick="showSection('summary')">📈 Summary</button>
            <button class="nav-btn" onclick="showSection('charts')">📊 Charts</button>
            <button class="nav-btn" onclick="showSection('seasonality')">📅 Seasonality</button>
            <button class="nav-btn" onclick="showSection('risk')">⚠️ Risk Analysis</button>
            <button class="nav-btn" onclick="showSection('rs')">📊 Relative Strength</button>
        </div>
        
        <!-- Summary Section -->
        <div id="summary" class="section active">
            <div class="summary-grid">
                <div class="stat-card">
                    <div class="stat-label">Total Charts</div>
                    <div class="stat-value">{len(png_files)}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Insights Files</div>
                    <div class="stat-value">{len([i for i in all_insights.values() if 'raw' not in i and 'error' not in i])}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">High Volatility Stocks</div>
                    <div class="stat-value">{risk_stats['high_volatility']['count']}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Low Volatility Stocks</div>
                    <div class="stat-value">{risk_stats['low_volatility']['count']}</div>
                </div>
"""
    
    # Add RS summary stats
    if rs_data:
        best_rs = rs_data[0] if rs_data else None
        worst_rs = rs_data[-1] if rs_data else None
        avg_rs = statistics.mean([rs['rs_score'] for rs in rs_data]) if rs_data else 0
        
        html += f"""
                <div class="stat-card">
                    <div class="stat-label">Best RS Stock</div>
                    <div class="stat-value positive">{best_rs['symbol'] if best_rs else 'N/A'}</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">Avg RS Score</div>
                    <div class="stat-value neutral">{avg_rs:.3f}</div>
                </div>
"""
    
    # Add best and worst months if available
    if seasonality_stats['best_month']:
        best_month = seasonality_stats['best_month']
        html += f"""
                <div class="stat-card">
                    <div class="stat-label">Best Month</div>
                    <div class="stat-value positive">{best_month[0]}</div>
                </div>
"""
    
    if seasonality_stats['worst_month']:
        worst_month = seasonality_stats['worst_month']
        html += f"""
                <div class="stat-card">
                    <div class="stat-label">Worst Month</div>
                    <div class="stat-value negative">{worst_month[0]}</div>
                </div>
"""
    
    html += """
            </div>
            
            <!-- Monthly Returns Chart -->
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">📅 Average Monthly Returns</h2>
                <div class="chart-container">
                    <canvas id="monthlyReturnsChart"></canvas>
                </div>
            </div>
            
            <!-- Day of Week Returns Chart -->
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">📅 Average Day of Week Returns</h2>
                <div class="chart-container">
                    <canvas id="dowReturnsChart"></canvas>
                </div>
            </div>
        </div>
        
        <!-- Charts Section -->
        <div id="charts" class="section">
            <div class="chart-grid">
"""
    
    # Add each chart with insights
 
    for idx, png_file in enumerate(png_files, 1):
       
        img_path = os.path.join(folder_path, png_file)
        img_base64 = encode_image_to_base64(img_path)
        
        if not img_base64:
            continue
        try:
            insights = all_insights.get(png_file, {})
        except Exception as e:
            insights = {'error': str(e)}
        
        
        html += f"""
                <div class="chart-card">
                    <img src="data:image/png;base64,{img_base64}" alt="{png_file}" class="chart-image">
                    
                    <div class="insights-box">
                        <div class="insights-title">📊 Key Insights & Metrics</div>
"""
        
        # Add insights data
        if 'raw' in insights:
            html += f"""
                        <div class="insight-row">
                            <div class="insight-item">
                                <div class="insight-value">{insights['raw'][:200]}</div>
                            </div>
                        </div>
"""
        else:
            # First row: Symbol, Date, Volatility, Risk Profile
            html += """
                        <div class="insight-row">
"""
            if 'symbol' in insights:
                html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Symbol</div>
                                <div class="insight-value">📈 {insights['symbol']}</div>
                            </div>
"""
            if 'date' in insights:
                html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Date</div>
                                <div class="insight-value">📅 {insights['date']}</div>
                            </div>
"""
            if 'volatility' in insights:
                vol = insights['volatility']
                vol_class = 'positive' if vol < 1.5 else ('negative' if vol > 3.0 else 'neutral')
                html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Volatility</div>
                                <div class="insight-value {vol_class}">{'🟢' if vol < 1.5 else '🟡' if vol < 3.0 else '🔴'} {vol:.2f}%</div>
                            </div>
"""
            if 'risk_profile' in insights:
                html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Risk Profile</div>
                                <div class="insight-value">⚠️ {insights['risk_profile']}</div>
                            </div>
"""
            html += """
                        </div>
"""
            
            # Second row: Win rate and returns
            if 'positive_pct' in insights or 'avg_daily_return' in insights:
                html += """
                        <div class="insight-row">
"""
                if 'positive_pct' in insights:
                    pct = insights['positive_pct']
                    pct_class = 'positive' if pct > 60 else ('negative' if pct < 40 else 'neutral')
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Win Rate</div>
                                <div class="insight-value {pct_class}">{'✅' if pct > 60 else '⚠️' if pct > 40 else '❌'} {pct:.1f}%</div>
                            </div>
"""
                if 'avg_daily_return' in insights:
                    ret = insights['avg_daily_return']
                    ret_class = 'positive' if ret > 0 else 'negative'
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Avg Daily Return</div>
                                <div class="insight-value {ret_class}">{'📈' if ret > 0 else '📉'} {ret:+.2f}%</div>
                            </div>
"""
                if 'max_drawdown' in insights:
                    dd = insights['max_drawdown']
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Max Drawdown</div>
                                <div class="insight-value negative">📉 {dd:.2f}%</div>
                            </div>
"""
                html += """
                        </div>
"""
            
            # Third row: Relative Strength
            if 'relative_strength' in insights:
                rs = insights['relative_strength']
                rs_class = 'positive' if rs['rs_score'] > 0.05 else ('negative' if rs['rs_score'] < -0.05 else 'neutral')
                rs_badge_class = 'rs-strong' if rs['rs_score'] > 0.1 else ('rs-moderate' if rs['rs_score'] > 0.05 else ('rs-neutral' if rs['rs_score'] > -0.05 else 'rs-weak'))
                
                html += f"""
                        <div class="insight-row">
                            <div class="insight-item">
                                <div class="insight-label">Relative Strength</div>
                                <div class="insight-value {rs_class}">
                                    📊 {rs['rs_score']:.3f}
                                    <span class="rs-badge {rs_badge_class}">{rs['rs_rating']}</span>
                                </div>
                            </div>
                            <div class="insight-item">
                                <div class="insight-label">vs NIFTY500</div>
                                <div class="insight-value {rs_class}">{rs['total_relative_return']:+.2f}%</div>
                            </div>
                        </div>
"""
            
            # Fourth row: Best and worst months
            if 'best_month' in insights or 'worst_month' in insights:
                html += """
                        <div class="insight-row">
"""
                if 'best_month' in insights:
                    best = insights['best_month']
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Best Month</div>
                                <div class="insight-value positive">🏆 {best['month']}</div>
                            </div>
"""
                if 'worst_month' in insights:
                    worst = insights['worst_month']
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Worst Month</div>
                                <div class="insight-value negative">📉 {worst['month']}</div>
                            </div>
"""
                html += """
                        </div>
"""
            
            # Fifth row: Trading recommendations (with safe access)
            if insights.get('best_months') or insights.get('best_days'):
               
                print('in the problematic part')
                html += """
                        <div class="insight-row">
"""
                if insights.get('best_months'):
                    best_months = ', '.join(insights['best_months'])
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Best Months to Trade</div>
                                <div class="insight-value">📅 {best_months}</div>
                            </div>
"""
                if insights.get('best_days'):
                    best_days = ', '.join(insights['best_days'])
                    html += f"""
                            <div class="insight-item">
                                <div class="insight-label">Best Days to Trade</div>
                                <div class="insight-value">📅 {best_days}</div>
                            </div>
"""
                html += """
                        </div>
"""
        
        html += """
                    </div>
                </div>
"""
    
    html += """
            </div>
        </div>
        
        <!-- Seasonality Section -->
        <div id="seasonality" class="section">
"""
    
    if seasonality_stats['monthly']:
        print('Adding seasonality stats to HTML')
        html += """
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">📅 Monthly Seasonality Analysis</h2>
                <p style="color: #96c8ff; margin-bottom: 20px;">Statistical analysis across all months</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Month</th>
                            <th>Avg Return</th>
                            <th>Highest Positive</th>
                            <th>Highest Negative</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        months_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for month in months_order:
            if month in seasonality_stats['monthly']:
                stats = seasonality_stats['monthly'][month]
                avg_return = stats['avg_return']
                max_positive = stats['max_positive']
                max_negative = stats['max_negative']
                
                avg_class = 'positive' if avg_return > 0 else 'negative'
                pos_class = 'positive'
                neg_class = 'negative'
                
                highlight = 'style="background: rgba(100, 180, 255, 0.2);"' if month in [seasonality_stats['best_month'][0], seasonality_stats['worst_month'][0]] else ''
                
                html += f"""
                        <tr {highlight}>
                            <td><strong>{month}</strong></td>
                            <td class="{avg_class}"><strong>{avg_return:.2f}%</strong></td>
                            <td class="{pos_class}"><strong>{max_positive:.2f}%</strong></td>
                            <td class="{neg_class}"><strong>{max_negative:.2f}%</strong></td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
                
                <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #2a2a40 0%, #1e1e30 100%); border-radius: 12px; border: 2px solid #ffd700;">
                    <h3 style="color: #ffd700; margin-bottom: 15px;">🏆 Key Findings</h3>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #4ade80;">BEST MONTH:</strong> {seasonality_stats['best_month'][0]} 
                        ({seasonality_stats['best_month'][1]['avg_return']:.2f}% avg return)
                    </p>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #ff6b6b;">WORST MONTH:</strong> {seasonality_stats['worst_month'][0]} 
                        ({seasonality_stats['worst_month'][1]['avg_return']:.2f}% avg return)
                    </p>
                </div>
            </div>
"""
    
    if seasonality_stats['dow']:
        print('Adding day of week stats to HTML')
        html += """
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">📅 Day of Week Seasonality Analysis</h2>
                <p style="color: #96c8ff; margin-bottom: 20px;">Statistical analysis across all days of the week (excluding weekends)</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Day</th>
                            <th>Avg Return</th>
                            <th>Highest Positive</th>
                            <th>Highest Negative</th>
                        </tr>
                    </thead>
                    <tbody>
"""
        
        days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        
        for day in days_order:
            if day in seasonality_stats['dow']:
                stats = seasonality_stats['dow'][day]
                avg_return = stats['avg_return']
                max_positive = stats['max_positive']
                max_negative = stats['max_negative']
                
                avg_class = 'positive' if avg_return > 0 else 'negative'
                pos_class = 'positive'
                neg_class = 'negative'
                
                highlight = 'style="background: rgba(100, 180, 255, 0.2);"' if day in [seasonality_stats['best_day'][0], seasonality_stats['worst_day'][0]] else ''
                
                html += f"""
                        <tr {highlight}>
                            <td><strong>{day}</strong></td>
                            <td class="{avg_class}"><strong>{avg_return:.2f}%</strong></td>
                            <td class="{pos_class}"><strong>{max_positive:.2f}%</strong></td>
                            <td class="{neg_class}"><strong>{max_negative:.2f}%</strong></td>
                        </tr>
"""
        
        html += f"""
                    </tbody>
                </table>
                
                <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #2a2a40 0%, #1e1e30 100%); border-radius: 12px; border: 2px solid #ffd700;">
                    <h3 style="color: #ffd700; margin-bottom: 15px;">🏆 Key Findings</h3>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #4ade80;">BEST DAY:</strong> {seasonality_stats['best_day'][0]} 
                        ({seasonality_stats['best_day'][1]['avg_return']:.2f}% avg return)
                    </p>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #ff6b6b;">WORST DAY:</strong> {seasonality_stats['worst_day'][0]} 
                        ({seasonality_stats['worst_day'][1]['avg_return']:.2f}% avg return)
                    </p>
                </div>
            </div>
"""
    
    if not seasonality_stats['monthly'] and not seasonality_stats['dow']:
        html += """
            <div class="seasonality-table">
                <h2 style="color: #ffd700;">No seasonality data available</h2>
                <p style="color: #96c8ff;">Insights files need seasonality information for analysis.</p>
            </div>
"""
    
    html += """
        </div>
        
        <!-- Risk Analysis Section -->
        <div id="risk" class="section">
"""
    
    if risk_stats:
        html += """
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">⚠️ Risk Profile Analysis</h2>
                <p style="color: #96c8ff; margin-bottom: 20px;">Risk assessment across all stocks</p>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-top: 20px;">
"""
        
        # Volatility distribution
        html += f"""
                    <div class="stat-card">
                        <div class="stat-label">Volatility Distribution</div>
                        <div style="margin: 20px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span>High (>3%)</span>
                                <span class="negative">{risk_stats['high_volatility']['count']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span>Medium (1.5-3%)</span>
                                <span class="neutral">{risk_stats['medium_volatility']['count']}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                                <span>Low (<1.5%)</span>
                                <span class="positive">{risk_stats['low_volatility']['count']}</span>
                            </div>
                        </div>
                    </div>
"""
        
        # Average metrics
        if 'volatility' in risk_stats:
            html += f"""
                    <div class="stat-card">
                        <div class="stat-label">Average Volatility</div>
                        <div class="stat-value neutral">{risk_stats['volatility']['avg']:.2f}%</div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #96c8ff;">
                            Min: {risk_stats['volatility']['min']:.2f}% | Max: {risk_stats['volatility']['max']:.2f}%
                        </div>
                    </div>
"""
        
        if 'max_drawdown' in risk_stats:
            html += f"""
                    <div class="stat-card">
                        <div class="stat-label">Average Max Drawdown</div>
                        <div class="stat-value negative">{risk_stats['max_drawdown']['avg']:.2f}%</div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #96c8ff;">
                            Min: {risk_stats['max_drawdown']['min']:.2f}% | Max: {risk_stats['max_drawdown']['max']:.2f}%
                        </div>
                    </div>
"""
        
        if 'avg_daily_return' in risk_stats:
            html += f"""
                    <div class="stat-card">
                        <div class="stat-label">Average Daily Return</div>
                        <div class="stat-value {'positive' if risk_stats['avg_daily_return']['avg'] > 0 else 'negative'}">{risk_stats['avg_daily_return']['avg']:+.2f}%</div>
                        <div style="margin-top: 10px; font-size: 0.9em; color: #96c8ff;">
                            Min: {risk_stats['avg_daily_return']['min']:+.2f}% | Max: {risk_stats['avg_daily_return']['max']:+.2f}%
                        </div>
                    </div>
"""
        
        html += """
                </div>
                
                <!-- Stocks with consistent 1+ SD movement -->
                <div class="risk-list">
                    <h3>Stocks with Consistent 1+ Standard Deviation Movement</h3>
"""
        
        if risk_stats['consistent_sd']:
            # Sort by volatility (highest first)
            sorted_sd_stocks = sorted(risk_stats['consistent_sd'], key=lambda x: x['volatility'], reverse=True)
            
            for stock in sorted_sd_stocks[:10]:  # Show top 10
                html += f"""
                    <div class="risk-item">
                        <span class="risk-symbol">{stock['symbol']}</span>
                        <span class="risk-value">Volatility: {stock['volatility']:.2f}%</span>
                    </div>
"""
        else:
            html += """
                    <p style="color: #96c8ff;">No stocks with consistent 1+ SD movement found.</p>
"""
        
        html += """
                </div>
                
                <!-- Stock with highest movement -->
"""
        
        if 'highest_movement' in risk_stats:
            highest = risk_stats['highest_movement']
            html += f"""
                <div class="highlight-box">
                    <h3>Stock with Highest Movement</h3>
                    <div class="highlight-content">
                        <strong>{highest['symbol']}</strong> shows the highest price movement with a range of 
                        <span class="positive">{highest['range']:.2f}%</span> 
                        (Max: <span class="positive">{highest['max']:.2f}%</span>, 
                        Min: <span class="negative">{highest['min']:.2f}%</span>)
                    </div>
                </div>
"""
        
        html += """
                
                <!-- Risk vs Return Scatter Plot -->
                <div style="margin-top: 30px;">
                    <h3 style="color: #ffd700; margin-bottom: 15px;">Risk vs Return Analysis</h3>
                    <div class="chart-container">
                        <canvas id="riskReturnChart"></canvas>
                    </div>
                </div>
            </div>
"""
    else:
        html += """
            <div class="seasonality-table">
                <h2 style="color: #ffd700;">No risk data available</h2>
                <p style="color: #96c8ff;">Insights files need risk information for analysis.</p>
            </div>
"""
    
    html += """
        </div>
        
        <!-- Relative Strength Section -->
        <div id="rs" class="section">
            <div class="seasonality-table">
                <h2 style="color: #ffd700; margin-bottom: 20px;">📊 Relative Strength Analysis (vs ^CRSLDX)</h2>
                <p style="color: #96c8ff; margin-bottom: 20px;">Comparison of each stock's performance against Nifty 500 Index</p>
                
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Symbol</th>
                            <th>RS Score</th>
                            <th>RS Rating</th>
                            <th>Stock Return</th>
                            <th>Benchmark Return</th>
                            <th>Relative Return</th>
                            <th>RS Ratio</th>
                        </tr>
                    </thead>
                    <tbody>
"""
    
    if rs_data:
        for idx, rs in enumerate(rs_data, 1):
            rs_class = 'positive' if rs['rs_score'] > 0.05 else ('negative' if rs['rs_score'] < -0.05 else 'neutral')
            rs_badge_class = 'rs-strong' if rs['rs_score'] > 0.1 else ('rs-moderate' if rs['rs_score'] > 0.05 else ('rs-neutral' if rs['rs_score'] > -0.05 else 'rs-weak'))
            
            html += f"""
                        <tr>
                            <td><strong>{idx}</strong></td>
                            <td><strong>{rs['symbol']}</strong></td>
                            <td class="{rs_class}"><strong>{rs['rs_score']:.3f}</strong></td>
                            <td><span class="rs-badge {rs_badge_class}">{rs['rs_rating']}</span></td>
                            <td class="{'positive' if rs['total_stock_return'] > 0 else 'negative'}">{rs['total_stock_return']:+.2f}%</td>
                            <td class="{'positive' if rs['total_benchmark_return'] > 0 else 'negative'}">{rs['total_benchmark_return']:+.2f}%</td>
                            <td class="{rs_class}"><strong>{rs['total_relative_return']:+.2f}%</strong></td>
                            <td class="{rs_class}">{rs['rs_ratio']:.3f}</td>
                        </tr>
"""
    else:
        html += """
                        <tr>
                            <td colspan="8" style="text-align: center; color: #96c8ff;">No relative strength data available</td>
                        </tr>
"""
    
    html += f"""
                    </tbody>
                </table>
                
                <div style="margin-top: 30px; padding: 20px; background: linear-gradient(135deg, #2a2a40 0%, #1e1e30 100%); border-radius: 12px; border: 2px solid #ffd700;">
                    <h3 style="color: #ffd700; margin-bottom: 15px;">🏆 RS Analysis Summary</h3>
"""
    
    if rs_data:
        best_rs = rs_data[0]
        worst_rs = rs_data[-1]
        avg_rs = statistics.mean([rs['rs_score'] for rs in rs_data])
        
        html += f"""
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #4ade80;">STRONGEST STOCK:</strong> {best_rs['symbol']} 
                        (RS Score: {best_rs['rs_score']:.3f}, Outperformance: {best_rs['total_relative_return']:+.2f}%)
                    </p>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #ff6b6b;">WEAKEST STOCK:</strong> {worst_rs['symbol']} 
                        (RS Score: {worst_rs['rs_score']:.3f}, Underperformance: {worst_rs['total_relative_return']:+.2f}%)
                    </p>
                    <p style="font-size: 1.1em; color: #fff; margin: 10px 0;">
                        <strong style="color: #96c8ff;">AVERAGE RS SCORE:</strong> {avg_rs:.3f}
                    </p>
"""
    
    html += """
                </div>
                
                <!-- RS Distribution Chart -->
                <div style="margin-top: 30px;">
                    <h3 style="color: #ffd700; margin-bottom: 15px;">RS Score Distribution</h3>
                    <div class="chart-container">
                        <canvas id="rsDistributionChart"></canvas>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="watermark">Prady ©</div>
    
    <script>
        function showSection(sectionId) {
            // Hide all sections
            document.querySelectorAll('.section').forEach(section => {
                section.classList.remove('active');
            });
            
            // Remove active class from all buttons
            document.querySelectorAll('.nav-btn').forEach(btn => {
                btn.classList.remove('active');
            });
            
            // Show selected section
            document.getElementById(sectionId).classList.add('active');
            
            // Add active class to clicked button
            event.target.classList.add('active');
            
            // Initialize charts when section is shown
            if (sectionId === 'summary') {
                initSummaryCharts();
            } else if (sectionId === 'risk') {
                initRiskChart();
            } else if (sectionId === 'rs') {
                initRSChart();
            }
        }
        
        // Initialize summary charts
        function initSummaryCharts() {
            // Monthly Returns Chart
            const monthlyCtx = document.getElementById('monthlyReturnsChart').getContext('2d');
            new Chart(monthlyCtx, {
                type: 'bar',
                data: {
                    labels: ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                    datasets: [{
                        label: 'Average Monthly Return (%)',
                        data: [""" + ', '.join([f"{seasonality_stats['monthly'].get(month, {}).get('avg_return', 0):.2f}" for month in ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']]) + """],
                        backgroundColor: function(context) {
                            const value = context.raw;
                            return value > 0 ? 'rgba(74, 222, 128, 0.7)' : 'rgba(255, 107, 107, 0.7)';
                        },
                        borderColor: function(context) {
                            const value = context.raw;
                            return value > 0 ? 'rgba(74, 222, 128, 1)' : 'rgba(255, 107, 107, 1)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    }
                }
            });
            
            // Day of Week Returns Chart
            const dowCtx = document.getElementById('dowReturnsChart').getContext('2d');
            new Chart(dowCtx, {
                type: 'bar',
                data: {
                    labels: ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
                    datasets: [{
                        label: 'Average Day Return (%)',
                        data: [""" + ', '.join([f"{seasonality_stats['dow'].get(day, {}).get('avg_return', 0):.2f}" for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']]) + """],
                        backgroundColor: function(context) {
                            const value = context.raw;
                            return value > 0 ? 'rgba(74, 222, 128, 0.7)' : 'rgba(255, 107, 107, 0.7)';
                        },
                        borderColor: function(context) {
                            const value = context.raw;
                            return value > 0 ? 'rgba(74, 222, 128, 1)' : 'rgba(255, 107, 107, 1)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize risk chart
        function initRiskChart() {
            const riskCtx = document.getElementById('riskReturnChart').getContext('2d');
            new Chart(riskCtx, {
                type: 'scatter',
                data: {
                    datasets: [{
                        label: 'Risk vs Return',
                        data: [""" + ', '.join([f"{{x: {insights.get('volatility', 0)}, y: {insights.get('avg_daily_return', 0) * 100}}}" for insights in all_insights.values() if 'volatility' in insights and 'avg_daily_return' in insights]) + """],
                        backgroundColor: 'rgba(100, 180, 255, 0.7)',
                        borderColor: 'rgba(100, 180, 255, 1)',
                        pointRadius: 8,
                        pointHoverRadius: 10
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        x: {
                            title: {
                                display: true,
                                text: 'Volatility (%)',
                                color: '#96c8ff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Average Daily Return (%)',
                                color: '#96c8ff'
                            },
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        },
                        tooltip: {
                            callbacks: {
                                label: function(context) {
                                    return `Volatility: ${context.parsed.x.toFixed(2)}%, Return: ${context.parsed.y.toFixed(2)}%`;
                                }
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize RS chart
        function initRSChart() {
            const rsCtx = document.getElementById('rsDistributionChart').getContext('2d');
            new Chart(rsCtx, {
                type: 'bar',
                data: {
                    labels: [""" + ', '.join([f"'{rs['symbol']}'" for rs in rs_data[:20]]) + """],
                    datasets: [{
                        label: 'RS Score',
                        data: [""" + ', '.join([f"{rs['rs_score']:.3f}" for rs in rs_data[:20]]) + """],
                        backgroundColor: function(context) {
                            const value = context.raw;
                            if (value > 0.1) return 'rgba(74, 222, 128, 0.7)';
                            if (value > 0.05) return 'rgba(96, 165, 250, 0.7)';
                            if (value > -0.05) return 'rgba(251, 191, 36, 0.7)';
                            return 'rgba(248, 113, 113, 0.7)';
                        },
                        borderColor: function(context) {
                            const value = context.raw;
                            if (value > 0.1) return 'rgba(74, 222, 128, 1)';
                            if (value > 0.05) return 'rgba(96, 165, 250, 1)';
                            if (value > -0.05) return 'rgba(251, 191, 36, 1)';
                            return 'rgba(248, 113, 113, 1)';
                        },
                        borderWidth: 1
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    scales: {
                        y: {
                            beginAtZero: true,
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff'
                            }
                        },
                        x: {
                            grid: {
                                color: 'rgba(255, 255, 255, 0.1)'
                            },
                            ticks: {
                                color: '#96c8ff',
                                maxRotation: 45,
                                minRotation: 45
                            }
                        }
                    },
                    plugins: {
                        legend: {
                            labels: {
                                color: '#fff'
                            }
                        }
                    }
                }
            });
        }
        
        // Initialize charts on page load
        document.addEventListener('DOMContentLoaded', function() {
            initSummaryCharts();
        });
        
        // Add smooth scroll behavior
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', function (e) {
                e.preventDefault();
                document.querySelector(this.getAttribute('href')).scrollIntoView({
                    behavior: 'smooth'
                });
            });
        });
        
        // Add keyboard navigation
        document.addEventListener('keydown', function(e) {
            const sections = ['summary', 'charts', 'seasonality', 'risk', 'rs'];
            const currentSection = document.querySelector('.section.active').id;
            const currentIndex = sections.indexOf(currentSection);
            
            if (e.key === 'ArrowRight' && currentIndex < sections.length - 1) {
                const buttons = document.querySelectorAll('.nav-btn');
                buttons[currentIndex + 1].click();
            } else if (e.key === 'ArrowLeft' && currentIndex > 0) {
                const buttons = document.querySelectorAll('.nav-btn');
                buttons[currentIndex - 1].click();
            }
        });
        
        console.log('📊 Market Analysis Dashboard loaded successfully!');
        console.log('💡 Tip: Use arrow keys to navigate between sections');
    </script>
</body>
</html>
"""
    
    return html

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        folder = "C:\\temp\\momentum"
        output = "market_dashboard.html"
        print(f"ℹ️  Using default: {folder}")
        print(f"ℹ️  Usage: python script.py <folder_path> [output.html]\n")
    else:
        folder = sys.argv[1]
        output = sys.argv[2] if len(sys.argv) > 2 else "market_dashboard.html"
    
    generate_html_dashboard(folder, output)