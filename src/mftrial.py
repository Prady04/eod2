import pandas as pd
import requests
from io import StringIO
from datetime import datetime, timedelta

# ---------- CONFIG ----------
NAV_URL = "https://www.amfiindia.com/spages/NAVAll.txt"
SCHEME_MASTER_URL = "https://portal.amfiindia.com/DownloadSchemeMasterReport_Po.aspx?mf=0"
BENCHMARK_MAPPING_FILE = "benchmark_mapping.csv"  # From your excel extracted mapping
OUTPUT_FILE = "mf_outperformance.xlsx"

# ---------- STEP 1 – FETCH NAV DATA ----------
def fetch_nav_data():
    r = requests.get(NAV_URL)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text), sep=';')
   
    df.reset_index(inplace=True)
    df.columns = df.columns.str.strip().str.replace("\n", " ")
    for col in df.columns:
        print(col)
    exit()
    df = df[df['Plan'] == 'Growth']  # Keep only growth plans
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    return df

# ---------- STEP 2 – LOAD SCHEME MASTER ----------
def fetch_scheme_master():
    r = requests.get(SCHEME_MASTER_URL)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    return df

# ---------- STEP 3 – LOAD BENCHMARK MAPPING ----------
def load_benchmark_mapping():
    return pd.read_csv(BENCHMARK_MAPPING_FILE)  # columns: SchemeCode, BenchmarkIndex

# ---------- STEP 4 – FETCH BENCHMARK DATA ----------
def fetch_benchmark_data(index_name, start_date):
    # Implement per index — here’s an example for NSE indices
    url = f"https://www.niftyindices.com/IndexConstituent/IndexData?indexName={index_name}&fromDate={start_date.strftime('%d-%m-%Y')}&toDate={datetime.today().strftime('%d-%m-%Y')}"
    r = requests.get(url)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df[['Date', 'Close']]
    return df

# ---------- STEP 5 – CALCULATE RETURNS ----------
def calculate_cagr(start_value, end_value, days):
    return (end_value / start_value) ** (365 / days) - 1

def get_period_returns(price_df, date, period_days):
    end_date = date
    start_date = date - timedelta(days=period_days)
    start_row = price_df.loc[price_df['Date'] <= start_date].tail(1)
    end_row = price_df.loc[price_df['Date'] <= end_date].tail(1)
    if start_row.empty or end_row.empty:
        return None
    return calculate_cagr(start_row['Close'].values[0], end_row['Close'].values[0], period_days)

# ---------- STEP 6 – PROCESS ALL FUNDS ----------
def generate_outperformance_table():
    nav_df = fetch_nav_data()
    mapping_df = load_benchmark_mapping()

    results = []
    for _, row in mapping_df.iterrows():
        scheme_code = row['SchemeCode']
        benchmark_name = row['BenchmarkIndex']

        scheme_prices = nav_df[nav_df['Scheme Code'] == scheme_code].sort_values('Date')
        bench_prices = fetch_benchmark_data(benchmark_name, scheme_prices['Date'].min())

        latest_date = scheme_prices['Date'].max()
        periods = {'1M':30, '3M':90, '6M':180, '1Y':365, '3Y':1095, '5Y':1825}

        data = {'SchemeCode': scheme_code, 'SchemeName': row['SchemeName'], 'Benchmark': benchmark_name}
        for label, days in periods.items():
            scheme_ret = get_period_returns(scheme_prices.rename(columns={'Net Asset Value':'Close'}), latest_date, days)
            bench_ret = get_period_returns(bench_prices, latest_date, days)
            data[f'{label}_Scheme'] = scheme_ret
            data[f'{label}_Bench'] = bench_ret
            data[f'{label}_Outperformance'] = None if (scheme_ret is None or bench_ret is None) else scheme_ret - bench_ret

        results.append(data)

    df_out = pd.DataFrame(results)
    df_out.to_excel(OUTPUT_FILE, index=False)
    print(f"Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    generate_outperformance_table()
