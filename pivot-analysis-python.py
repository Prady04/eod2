import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import numpy as np
import mplfinance as mpf
class PivotPatternAnalyzer:
    def __init__(self, csv_file):
        """Initialize the analyzer with CSV file path."""
        self.df = pd.read_csv(csv_file)
        self.prepare_data()
        self.patterns = {}
        self.mpf = mpf
        
    def prepare_data(self):
        """Prepare and clean the data."""
        # Convert date string to datetime
        self.df.columns = self.df.columns.str.replace(' ', '') 
        print(self.df.keys())
        self.df['Date'] = pd.to_datetime(self.df['Date'], format='%d-%b-%Y')
        
        # Ensure numeric columns
        numeric_columns = ['Open', 'High', 'Low', 'Close']
        for col in numeric_columns:
            self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
            
        # Sort by date
        self.df = self.df.sort_values('Date')
        
        # Calculate basic indicators
        self.df['Range'] = self.df['High'] - self.df['Low']
        self.df['Body'] = self.df['Close'] - self.df['Open']
        
    def find_inside_bars(self):
        """Identify inside bar patterns."""
        inside_bars = []
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            if (current['High'] <= previous['High'] and 
                current['Low'] >= previous['Low']):
                inside_bars.append({
                    'Date': current['Date'],
                    'Pattern': 'Inside Bar',
                    'High': current['High'],
                    'Low': current['Low']
                })
        
        self.patterns['Inside Bars'] = inside_bars
        
    def find_outside_bars(self):
        """Identify outside bar patterns."""
        outside_bars = []
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            if (current['High'] > previous['High'] and 
                current['Low'] < previous['Low']):
                outside_bars.append({
                    'Date': current['Date'],
                    'Pattern': 'Outside Bar',
                    'High': current['High'],
                    'Low': current['Low']
                })
                
        self.patterns['Outside Bars'] = outside_bars
        
    def find_engulfing_patterns(self):
        """Identify bullish and bearish engulfing patterns."""
        bullish_engulfing = []
        bearish_engulfing = []
        
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            # Bullish engulfing
            if (current['Open'] < previous['Close'] and 
                current['Close'] > previous['Open']):
                bullish_engulfing.append({
                    'Date': current['Date'],
                    'Pattern': 'Bullish Engulfing',
                    'Open': current['Open'],
                    'Close': current['Close']
                })
                
            # Bearish engulfing
            if (current['Open'] > previous['Close'] and 
                current['Close'] < previous['Open']):
                bearish_engulfing.append({
                    'Date': current['Date'],
                    'Pattern': 'Bearish Engulfing',
                    'Open': current['Open'],
                    'Close': current['Close']
                })
                
        self.patterns['Bullish Engulfing'] = bullish_engulfing
        self.patterns['Bearish Engulfing'] = bearish_engulfing
        
    def find_key_reversals(self):
        """Identify key reversal patterns."""
        key_reversals = []
        
        for i in range(1, len(self.df)):
            current = self.df.iloc[i]
            previous = self.df.iloc[i-1]
            
            # Bullish key reversal
            if (current['Low'] < previous['Low'] and 
                current['Close'] > previous['High']):
                key_reversals.append({
                    'Date': current['Date'],
                    'Pattern': 'Bullish Key Reversal',
                    'Low': current['Low'],
                    'Close': current['Close']
                })
                
            # Bearish key reversal
            if (current['High'] > previous['High'] and 
                current['Close'] < previous['Low']):
                key_reversals.append({
                    'Date': current['Date'],
                    'Pattern': 'Bearish Key Reversal',
                    'High': current['High'],
                    'Close': current['Close']
                })
                
        self.patterns['Key Reversals'] = key_reversals
        
    def analyze_all_patterns(self):
        """Run all pattern analysis methods."""
        self.find_inside_bars()
        self.find_outside_bars()
        self.find_engulfing_patterns()
        self.find_key_reversals()
        
    def plot_price_chart(self):
        """Create a price chart with pattern markers."""
        plt.figure(figsize=(15, 8))
        
        # Plot price
        plt.plot(self.df['Date'], self.df['Close'], label='Close Price', color='blue', alpha=0.7)
        
        # Plot patterns
        colors = {'Inside Bars': 'green', 'Outside Bars': 'red', 
                 'Bullish Engulfing': 'lime', 'Bearish Engulfing': 'orange',
                 'Key Reversals': 'purple'}
        
        for pattern_name, pattern_list in self.patterns.items():
            if pattern_list:  # If pattern instances exist
                dates = [p['Date'] for p in pattern_list]
                prices = [self.df.loc[self.df['Date'] == d, 'Close'].iloc[0] for d in dates]
                plt.scatter(dates, prices, label=pattern_name, 
                          marker='^', s=100, color=colors.get(pattern_name, 'gray'))
             
        
        plt.title('Price Chart with Pattern Markers')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
    def plot_candlestick_chart(self):
        
        """Create a candlestick chart."""
        # Convert DataFrame to use datetime as index for mplfinance
        self.df.set_index('Date', inplace=True)  # Set 'Date' as index
        self.df.index = pd.to_datetime(self.df.index)  # Ensure the index is of type DatetimeIndex

        # Define custom markers for patterns
        apds = []
        colors = {'Inside Bars': 'green', 'Outside Bars': 'red', 
                'Bullish Engulfing': 'lime', 'Bearish Engulfing': 'orange',
                'Key Reversals': 'purple'}

        for pattern_name, pattern_list in self.patterns.items():
            if pattern_list:  # If pattern instances exist
                # Filter the dates to only the ones present in the pattern list
                dates = [p['Date'] for p in pattern_list]
                # Ensure that prices align with these dates and check for non-empty values
                prices = []
                for d in dates:
                    if d in self.df.index:
                        price = self.df.loc[d, 'Close']
                        prices.append(price)

                # Ensure the number of dates and prices match
                if len(dates) == len(prices):  # Ensure dates and prices are of the same size
                    color = colors.get(pattern_name, 'gray')
                    apds.append(mpf.make_addplot(
                        prices, type='scatter', markersize=100, marker='^', color=color, secondary_y=False
                    ))
                else:
                    print(f"Pattern {pattern_name}: Mismatch in dates and prices lengths. Skipping.")

        # Plot candlestick chart with additional markers
        mpf.plot(self.df, type='candle', volume=False, style='yahoo', 
                title='Candlestick Chart with Patterns', ylabel='Price',
                datetime_format='%Y-%m-%d', figsize=(15, 8), addplot=apds)

    def generate_report(self):
        """Generate a summary report of all patterns found."""
        print("\n=== Pattern Analysis Report ===\n")
        
        for pattern_name, pattern_list in self.patterns.items():
            print(f"\n{pattern_name} ({len(pattern_list)} instances found):")
            if pattern_list:
                for p in pattern_list:  # Show first 5 instances
                    print(f"  {p['Date'].strftime('%Y-%m-%d')}: ", end='')
                    details = {k: v for k, v in p.items() if k not in ['Date', 'Pattern']}
                    print(f"{', '.join(f'{k}: {v:.2f}' for k, v in details.items())}")
            if len(pattern_list) > 5:
                print("  ...")
           
                
        print("\nPattern Distribution:")
        

        import pickle
        for pattern_name, pattern_list in self.patterns.items():
           
            print(f"  {pattern_name}: {len(pattern_list)} instances")
            with open('outfile', 'wb') as fp:
                pickle.dump(pattern_list, fp)
            for p in pattern_list:
                print(f"  {p['Date'].strftime('%Y-%m-%d')}: ", end='')
                details = {k: v for k, v in p.items() if k not in ['Date', 'Pattern']}
                print(f"{', '.join(f'{k}: {v:.2f}' for k, v in details.items())}")
                
            
            


# Example usage
if __name__ == "__main__":
    # Initialize analyzer with CSV file
    analyzer = PivotPatternAnalyzer('NIFTY 50-01-02-2024-to-10-01-2025.csv')
    
    # Run analysis
    analyzer.analyze_all_patterns()
    
    # Generate and display report
    analyzer.generate_report()
    
    # Create visualization
    analyzer.plot_price_chart()
    plt.show()
    # Create candlestick visualization
    analyzer.plot_candlestick_chart()