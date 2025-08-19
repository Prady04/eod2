import yfinance as yf
import pandas as pd
import json
from datetime import datetime, timedelta
import os

# 1. Your list of symbols
symbols = ["^NSEI"]
json_dir = ".\\json_files"
for sym in symbols:
    json_file = f"{json_dir}\\candles_{sym}.json"
    lvl_file  = f"{json_dir}\\gann_levels_{sym}.json"

    # 2. Load existing candle JSON (if any)
    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            existing = json.load(f)
        # parse out the last date we have
        last_date = max(item['time'] for item in existing)
        start_date = datetime.strptime(last_date, "%Y-%m-%d") + timedelta(days=5)
    else:
        existing = []
        start_date = datetime.now() - timedelta(days=90)  # default look-back

    # 3. Fetch only from start_date until today
    df = yf.Ticker(sym).history(
        start=start_date.strftime("%Y-%m-%d"),
        end=(datetime.now() + timedelta(days=3)).strftime("%Y-%m-%d")
    ).reset_index()

    if df.empty:
        print(f"No new data for {sym} since {start_date.date()}")
        new_candles = []
    else:
        df = df[['Date', 'Open', 'High', 'Low', 'Close']]
        df['Date'] = pd.to_datetime(df['Date'])
        new_candles = [
            {
                "time": row['Date'].strftime("%Y-%m-%d"),
                "open": float(row['Open']),
                "high": float(row['High']),
                "low": float(row['Low']),
                "close": float(row['Close']),
            }
            for _, row in df.iterrows()
        ]

    # 4. Merge old + new, drop any duplicates just in case
    combined = {item['time']: item for item in existing + new_candles}
    candles = [combined[dt] for dt in sorted(combined)]
    import os
    current_directory = os.getcwd()
    print(current_directory)
    # 5. Re-write the JSON file
    with open(json_file, "w") as f:
        json.dump(candles, f, indent=2)
    print(f"Updated {json_file}: {len(new_candles)} new points")

    # 6. Re-calculate Gann levels off the **lowest low** of the full set
    lows = [pt['low'] for pt in candles]
    #base = min(lows)
    closes =[pt['close'] for pt in candles]
    base=closes[-1]
    #base =24942.35
    base_sqrt = base ** 0.5
    fracs = [0.125 * i for i in range(1, 9)]
    gann_levels = [(base_sqrt + af)**2 for af in fracs] + [(base_sqrt - af)**2 for af in fracs]

    # 7. Write levels JSON
    with open(lvl_file, "w") as f:
        json.dump(gann_levels, f, indent=2)
        for x in gann_levels:
            print(round(x,2))

    print(f"Re-wrote {lvl_file} with base={base:.2f}\n")
import yfinance as yf
import pandas as pd
import mplfinance as mpf
import json
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pytz
import os

symbol = "^NSEI"
lvl_file = f"{json_dir}\\gann_levels_{symbol}.json"

# ---- Step 1: Get 1 day of 15-min data ----
start_date = datetime.now() - timedelta(days=5)
df = yf.download(symbol,
                 start=start_date.strftime("%Y-%m-%d"),
                 end=(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d"),
                 interval="15m")

if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

# ensure datetime index
df.index = pd.to_datetime(df.index)

# Convert index to IST if it's UTC (safe-guard)
try:
    # If tz-naive, assume it's UTC and localize first, else convert
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC').tz_convert('Asia/Kolkata')
    else:
        df.index = df.index.tz_convert('Asia/Kolkata')
except Exception:
    # fallback: ignore tz conversion (keeps labels as-is)
    pass

# ---- Step 2: Load Gann levels ----
with open(f'{lvl_file}', "r") as f:
    gann_levels = sorted(json.load(f))

# ---- Step 3: Find nearest Gann level to current price ----
last_close = df['Close'].iloc[-1]
closest_level = min(gann_levels, key=lambda x: abs(x - last_close))

# ---- Step 4: Determine realistic Y-axis limits from data + 2 extra levels ----
min_price = df['Low'].min()
max_price = df['High'].max()

# Find levels currently in view (based on candle lows/highs)
levels_in_view = [level for level in gann_levels if min_price <= level <= max_price]
if levels_in_view:
    first_idx = gann_levels.index(levels_in_view[0])
    last_idx  = gann_levels.index(levels_in_view[-1])
    # Extend by 2 up/down
    first_idx = max(0, first_idx - 2)
    last_idx  = min(len(gann_levels) - 1, last_idx + 2)
    extended_levels = gann_levels[first_idx:last_idx + 3]
else:
    # If nothing falls inside candle range, center around closest_level +/- 2 indices
    closest_idx = gann_levels.index(closest_level)
    first_idx = max(0, closest_idx - 2)
    last_idx  = min(len(gann_levels) - 1, closest_idx + 2)
    extended_levels = gann_levels[first_idx:last_idx + 3]

# Expand ymin/ymax to include these extended levels
ymin = min(min_price, min(extended_levels))
ymax = max(max_price, max(extended_levels))
padding = (ymax - ymin) * 0.10
ymin -= padding
ymax += padding

# ---- Step 5: Prepare style ----
mc = mpf.make_marketcolors(up='g', down='r', wick='black', edge='black')
s  = mpf.make_mpf_style(marketcolors=mc)

# ---- Step 6: Plot candlestick chart ----
fig, ax = plt.subplots(figsize=(14, 7))
mpf.plot(df, type='candle', style=s, ax=ax, ylim=(ymin, ymax), show_nontrading=False)

# Ensure x-axis uses date converter (mplfinance usually does this)
ax.xaxis_date()
from matplotlib import font_manager as fm

# Load custom font
prop = fm.FontProperties(fname=".\\font\\poppins.ttf", size=40)

# Watermark with custom font
fig.text(0.5, 0.5, "prady",
         fontproperties=prop,
         color='gray', alpha=0.1,
         ha='center', va='center', rotation=30)
# get x-axis numeric limits (matplotlib date numbers)
x0, x1 = ax.get_xlim()
# pick a position on the x-axis for labels (fraction from left to right)
label_x_frac = 0.88
label_x = x0 + (x1 - x0) * label_x_frac  # numeric Matplotlib date coordinate

# ---- Step 7: Draw extended Gann levels AND put label ON the line ----
for level in extended_levels:
    color = 'red' if level == closest_level else 'gray'
    # draw the horizontal line
    ax.axhline(y=level, color=color, linestyle='--', alpha=0.8, linewidth=1, zorder=1)
    # place label ON the line (near right side), ensure it's above other artists
    ax.text(label_x, level,
            f'{round(level)}',
            va='center', ha='center',
            fontsize=7,
            color=color,
            weight='bold' if color == 'red' else 'normal',
            bbox=dict(facecolor='white', edgecolor='none', pad=0.6, alpha=0.75),
            zorder=10,
            clip_on=False)

# ---- Footer date (IST) ----
current_date_str = datetime.now(pytz.timezone("Asia/Kolkata")).strftime("%d-%b-%Y %H:%M (IST)")
fig.text(0.5, 0.01, f"Date: {current_date_str}", ha='center', fontsize=9, color='gray')

plt.title(f"NIFTY - 15min", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # leave small bottom margin for footer

# -- Save exported image --
out_file = ".\\img\\nifty_gann_labels_on_lines.png"
plt.savefig(out_file, dpi=150, bbox_inches='tight')
print(f"Saved chart to {out_file}")

plt.show()
