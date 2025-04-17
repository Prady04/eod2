import mftool

def get_fund_metrics(fund_id):
    mf = mftool.Mftool()
    fund_details = mf.get_scheme_details(fund_id)
    print(mf.calculate_returns(fund_id,1000,500,36))
    print(fund_details)
    y = mf.get_scheme_info(fund_id)
  
    print(y)
    exit()
    if fund_details:
        print(f"Fund Name: {fund_details['scheme_name']}")
        print(f"Net Asset Value (NAV): {fund_details['nav']}")
        print(f"Last Updated: {fund_details['last_updated']}")
        print(f"1 Month Return: {fund_details['1month']}")
        print(f"3 Month Return: {fund_details['3month']}")
        print(f"6 Month Return: {fund_details['6month']}")
        print(f"1 Year Return: {fund_details['1year']}")
        print(f"3 Year Return: {fund_details['3year']}")
        print(f"5 Year Return: {fund_details['5year']}")
        print("-" * 50)
    else:
        print(f"Details not found for Fund ID: {fund_id}")

if __name__ == "__main__":
    portfolio = ['118834', '120594', '119598']  # Replace with your mutual fund IDs
    for fund_id in portfolio:
        get_fund_metrics(fund_id)