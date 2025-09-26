import yfinance as yf
import pandas as pd
import pandas_ta as ta
import plotly.graph_objects as go
import numpy as np

# Fetch data from Yahoo Finance
ticker = "^NSEI"  # Example ticker
data = yf.download(ticker, start="2023-01-01", end="2025-09-26", multi_level_index=False)


# Calculate RSI using pandas_ta
data['RSI'] = ta.rsi(data['Close'], length=14)

# Drop rows with None in RSI column and get the latest RSI value
data.dropna(subset=['RSI'], inplace=True)
print(data["RSI"].tail())
latest_rsi = data['RSI'].iloc[-1]

# Gauge chart setup


plot_bgcolor = "#def"
quadrant_colors = [plot_bgcolor, "#f25829", "#f2a529", "#eff229", "#85e043", "#2bad4e"]
quadrant_text = ["", "<b>Extreme Overbought</b>", "<b>Momentum Zone</b>", "<b>Neutral</b>", "<b>Over Sold</b>", "<b>Extremely Oversold</b>"]
n_quadrants = len(quadrant_colors) - 1

min_value = 0
max_value = 100  # RSI ranges from 0 to 100
hand_length = np.sqrt(2) / 4
hand_angle =  360 * (-latest_rsi/2 - min_value) / (max_value - min_value) - 180

# Create gauge chart
fig = go.Figure(
    data=[
        go.Pie(
            values=[0.5] + (np.ones(n_quadrants) / 2 / n_quadrants).tolist(),
            rotation=90,
            hole=0.5,
            marker_colors=quadrant_colors,
            text=quadrant_text,
            textinfo="text",
            hoverinfo="skip",
            sort=False
        ),
    ],
    layout=go.Layout(
        showlegend=False,
        margin=dict(b=0,t=10,l=10,r=10),
        width=800,
        height=800,
        paper_bgcolor=plot_bgcolor,
        annotations=[
            go.layout.Annotation(
                text=f"<b>Nifty RSI Level:</b><br>{latest_rsi:.2f}",
                x=0.5, xanchor="center", xref="paper",
                y=0.25, yanchor="bottom", yref="paper",
                showarrow=False,
                font=dict(size=14)
            )
        ],
        shapes=[
            go.layout.Shape(
                type="circle",
                x0=0.48, x1=0.52,
                y0=0.48, y1=0.52,
                fillcolor="#333",
                line_color="#333",
            ),
            go.layout.Shape(
                type="line",
                x0=0.5, x1=0.5 + hand_length * np.cos(np.radians(hand_angle)),
                y0=0.5, y1=0.5 + hand_length * np.sin(np.radians(hand_angle)),
                line=dict(color="#333", width=4)
            )
        ]
    )
)
fig.write_html("nifty_rsi_gauge.html")
fig.write_image("nifty_rsi_gauge.png")
