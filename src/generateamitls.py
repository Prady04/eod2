import pandas as pd
import yfinance as yf
from tqdm import tqdm
from datetime import datetime

def to_date(val):
    try:
        if val is None or pd.isna(val):
            return ""
        return datetime.fromtimestamp(val).strftime("%Y-%m-%d")
    except Exception:
        return ""

# Load symbols
stocks_df = pd.read_csv("C:\\python\\eod2\\src\\Book1.csv")
stocks_df['Symbol'] = stocks_df.iloc[:, 0].str.replace(".NS", "", regex=False)
tickers = stocks_df['Symbol'].tolist()[20:25]

results, failed_symbols = [], []

for symbol in tqdm(tickers, desc="Fetching fundamentals"):
    try:
        t = yf.Ticker(symbol + ".NS")
        info = t.info

        data = {
            "TICKER": symbol,
            "FullName": info.get("longName", ""),
            "Alias": info.get("symbol", ""),
            "Country": info.get("country", ""),
            "Address": info.get("address1", ""),
            "EPS": info.get("trailingEps", ""),
            "EPSEstCurrentYear": info.get("forwardEps", ""),
            "EPSEstNextYear": info.get("forwardEps", ""),
            "EPSEstNextQuarter": "",
            "PEGRatio": info.get("pegRatio", ""),
            "SharesFloat": info.get("floatShares", ""),
            "SharesOut": info.get("sharesOutstanding", ""),
            "DividendPayDate": to_date(info.get("dividendDate")),
            "ExDividendDate": to_date(info.get("exDividendDate")),
            "BookValuePerShare": info.get("bookValue", ""),
            "DividendPerShare": info.get("lastDividendValue", ""),
            "ProfitMargin": info.get("profitMargins", ""),
            "OperatingMargin": info.get("operatingMargins", ""),
            "OneYearTargetPrice": info.get("targetMeanPrice", ""),
            "ReturnOnAssets": info.get("returnOnAssets", ""),
            "ReturnOnEquity": info.get("returnOnEquity", ""),
            "QtrlyRevenueGrowth": info.get("revenueQuarterlyGrowth", ""),
            "GrossProfitPerShare": "",
            "SalesPerShare": info.get("revenuePerShare", ""),
            "EBITDAPerShare": "",
            "QtrlyEarningsGrowth": info.get("earningsQuarterlyGrowth", ""),
            "InsiderHoldPercent": info.get("heldPercentInsiders", ""),
            "InstitutionHoldPercent": info.get("heldPercentInstitutions", ""),
            "SharesShort": info.get("sharesShort", ""),
            "SharesShortPrevMonth": info.get("sharesShortPriorMonth", ""),
            "ForwardDividendPerShare": info.get("dividendRate", ""),
            "ForwardEPS": info.get("forwardEps", ""),
            "OperatingCashFlow": info.get("operatingCashflow", ""),
            "LeveredFreeCashFlow": info.get("freeCashflow", ""),
            "Beta": info.get("beta", ""),
            "LastSplitRatio": info.get("lastSplitFactor", ""),
            "LastSplitDate": to_date(info.get("lastSplitDate")),
            "DelistingDate": "",
            "PointValue": "1"
        }
        results.append(data)

    except Exception as e:
        print(f"❌ Failed to fetch {symbol}: {e}")
        failed_symbols.append(symbol)
        continue

fundamentals_df = pd.DataFrame(results)

# Save ASCII data file
fundamentals_txt = "fundamentals.txt"
fundamentals_df.to_csv(fundamentals_txt, index=False)
print(f"✅ AmiBroker fundamentals data saved to {fundamentals_txt}")

# Save .format file using your exact field names
with open("fundamentals.format", "w") as f:
    f.write("$NOQUOTES 1\n")
    f.write("$SKIPLINES 1\n")
    f.write("$FORMAT TICKER,FullName,Alias,Country,Address,EPS,EPSEstCurrentYear,EPSEstNextYear,EPSEstNextQuarter,PEGRatio,SharesFloat,SharesOut,DividendPayDate,ExDividendDate,BookValuePerShare,DividendPerShare,ProfitMargin,OperatingMargin,OneYearTargetPrice,ReturnOnAssets,ReturnOnEquity,QtrlyRevenueGrowth,GrossProfitPerShare,SalesPerShare,EBITDAPerShare,QtrlyEarningsGrowth,InsiderHoldPercent,InstitutionHoldPercent,SharesShort,SharesShortPrevMonth,ForwardDividendPerShare,ForwardEPS,OperatingCashFlow,LeveredFreeCashFlow,Beta,LastSplitRatio,LastSplitDate,DelistingDate,PointValue\n")

print("✅ AmiBroker format definition saved to fundamentals.format")

# Save failed symbols
if failed_symbols:
    pd.DataFrame(failed_symbols, columns=["Failed_Symbol"]).to_csv("failed_symbols.csv", index=False)
    print("⚠️ Failed symbols saved to failed_symbols.csv")
