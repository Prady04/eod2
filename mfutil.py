import os
import requests

# Discord Webhook Configuration
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL")

# Mutual Funds & Their IDs to Track
MUTUAL_FUNDS = {
    "Nippon India Small Cap Fund": "118778",
    "Quant Small Cap Fund": "120828",
    "Tata Small Cap Fund": "145206",
    "UTI Nifty Index Fund": "120716",
    "Motilal Oswal SmallCap Fund": "147623",
    "PGIM India Small Cap Fund": "149019",
    "Aditya Birla Sun Life Multi-Index Fund of Funds": "150690",
    "HDFC Nifty Index Fund": "149288",
    "SBI Nifty Index Fund": "148945"
}

THRESHOLDS = {
    "Nippon India Small Cap Fund": {"low": 150.0, "high": 200.0},
    "Quant Small Cap Fund": {"low": 120.0, "high": 180.0},
    "Tata Small Cap Fund": {"low": 90.0, "high": 140.0},
    "UTI Nifty Index Fund": {"low": 80.0, "high": 130.0},
    "Motilal Oswal SmallCap Fund": {"low": 110.0, "high": 160.0},
    "PGIM India Small Cap Fund": {"low": 100.0, "high": 140.0},
    "Aditya Birla Sun Life Multi-Index Fund of Funds": {"low": 70.0, "high": 120.0},
    "HDFC Nifty Index Fund": {"low": 85.0, "high": 135.0},
    "SBI Nifty Index Fund": {"low": 90.0, "high": 140.0}
}

# Fetch Mutual Fund NAVs (Today's NAV)
def fetch_mutual_fund_nav():
    url = "https://api.mfapi.in/mf/"
    results = {}

    for fund, fund_id in MUTUAL_FUNDS.items():
        try:
            response = requests.get(url + fund_id, timeout=5)
            response.raise_for_status()
            nav = float(response.json()["data"][0]["nav"])  # Fetch today's NAV
            results[fund] = nav
        except Exception as e:
            print(f"⚠️ Error fetching {fund}: {e}")
    
    return results
fetch_mutual_fund_nav()
# Send Discord Alert
def send_discord_alert(message):
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=5)
        response.raise_for_status()
    except Exception as e:
        print(f"⚠️ Error sending Discord alert: {e}")
        
        