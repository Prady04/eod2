# Libraries
import requests
import pandas as pd
from io import StringIO
from scipy import stats
from datetime import datetime


def fetch_data_from_api(url: str, headers: dict, params: dict) -> pd.DataFrame:
    """
    Fetch data from the REST API and return as a DataFrame.

    :param url: URL of the REST API.
    :param headers: HTTP headers for the request.
    :param params: Query parameters for the request.
    :return: DataFrame containing the API response.
    """
    response = requests.get(url, headers=headers, params=params)

    if response.status_code == 200:
        csv_data = response.text
        return pd.read_csv(StringIO(csv_data))
    else:
        raise Exception(
            f"Failed to fetch data. Status code: {response.status_code}, Response: {response.text}"
        )


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the DataFrame by renaming columns, dropping unnecessary columns, and filtering data.

    :param df: DataFrame containing raw data.
    :return: Cleaned DataFrame.
    """
    # Renaming Columns
    df.columns = [
        "legal_name",
        "close_price",
        "close_price_date",
        "category_name",
        "medalist_rating_number",
        "star_rating_till_date",
        "one_year_return",
        "three_years_return_annual",
        "five_years_return_annual",
        "ten_years_return_annual",
        "expense_ratio",
        "initial_purchase",
        "fund_total_nav_ruppes_millions",
        "risk_rating_till_date",
        "one_year_alpha",
        "three_years_alpha",
        "five_years_alpha",
        "one_year_beta",
        "three_years_beta",
        "five_years_beta",
        "one_year_standard_deviation",
        "three_years_standard_deviation",
        "five_years_standard_deviation",
        "one_year_sharpe",
        "three_years_sharpe",
        "five_years_sharpe",
        "tenfore_id",
    ]

    # Drop Column
    df.drop(columns=["tenfore_id"], inplace=True)

    # Filter Data
    df = df.loc[
        (df["fund_total_nav_ruppes_millions"] >= 1000)
        & ~(df["five_years_return_annual"].isna())
        & ~(df["three_years_sharpe"].isna())
        & ~(df["category_name"].isna())
    ]

    # Mapping for required category name
    category_values = {
        "Balanced Allocation": "balanced_allocation",
        "Dynamic Bond": "dynamic_bond",
        "ELSS (Tax Savings)": "elss",
        "Flexi Cap": "flexi_cap",
        "Focused Fund": "focused_fund",
        "Liquid": "liquid",
        "Equity - Other": "equity_other",
        "Fund of Funds": "fund_of_fund",
        "Arbitrage Fund": "arbitrage_fund",
        "Children": "children",
        "Dynamic Asset Allocation": "dynamic_asset_allocation",
        "Sector - Financial Services": "sector_financial_services",
        "Banking & PSU": "banking_psu",
        "Corporate Bond": "corporate_bond",
        "Credit Risk": "credit_risk",
        "Other Bond": "other_bond",
        "Index Funds - Fixed Income": "index_funds_fixed_income",
        "Sector - Technology": "sector_technology",
        "Dividend Yield": "dividend_yield",
        "Large & Mid-Cap": "large_mid_cap",
        "Aggressive Allocation": "aggressive_allocation",
        "Equity Savings": "equity_savings",
        "Equity - ESG": "equity_esg",
        "Fixed Maturity Ultrashort Bond": "fixed_maturity_ultrashort_bond",
        "Fixed Maturity Intermediate-Term Bond": "fixed_maturity_intermediate_term_bond",
        "Fixed Maturity Short-Term Bond": "fixed_maturity_short_term_bond",
        "Floating Rate": "floating_rate",
        "Large-Cap": "large_cap",
        "Mid-Cap": "midcap",
        "Small-Cap": "smallcap",
        "Global - Other": "global_other",
        "Sector - Precious Metals": "sector_precious_metals",
        "Government Bond": "government_bond",
        "Medium to Long Duration": "medium_long_duration",
        "Equity - Consumption": "equity_consumption",
        "Equity - Infrastructure": "equity_infrastructure",
        "Long Duration": "long_duration",
        "Low Duration": "low_duration",
        "Medium Duration": "medium_duration",
        "Mid-Cap": "mid_cap",
        "Money Market": "money_market",
        "Multi Asset Allocation": "multi_asset_allocation",
        "Multi-Cap": "multi_cap",
        "Index Funds": "index_funds",
        "Overnight": "overnight",
        "Sector - Healthcare": "sector_healthcare",
        "Value": "value",
        "Conservative Allocation": "conservative_allocation",
        "Retirement": "retirement",
        "Ultra Short Duration": "ultra_short_duration",
        "Short Duration": "short_duration",
        "Small-Cap": "small_cap",
        "10 yr Government Bond": "ten_yr_government_bond",
        "US Large-Cap Blend Equity": "us_large_cap_blend_equity",
        "Sector - FMCG": "sector_fmcg",
        "Contra": "contra",
        "Sector - Energy": "sector_energy",
        "Canadian Focused Equity": "canadian_focused_equity",
    }
    # Renaming cateogries
    df.loc[:, "category_name"] = df.loc[:, "category_name"].replace(category_values)

    return df


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate return and risk scores, then compute the total score for each row in the DataFrame.

    :param df: DataFrame containing financial data.
    :return: DataFrame with scores added.
    """

    def return_score_by_category(category_df: pd.DataFrame) -> pd.DataFrame:
        category_df.loc[:, "return_score_max_eighty"] = (
            category_df["one_year_return"].apply(
                lambda x: stats.percentileofscore(category_df["one_year_return"], x)
            )
            * 0.30
        )
        category_df.loc[:, "return_score_max_eighty"] += (
            category_df["three_years_return_annual"].apply(
                lambda x: stats.percentileofscore(
                    category_df["three_years_return_annual"], x
                )
            )
            * 0.20
        )
        category_df.loc[:, "return_score_max_eighty"] += (
            category_df["five_years_return_annual"].apply(
                lambda x: stats.percentileofscore(
                    category_df["five_years_return_annual"], x
                )
            )
            * 0.30
        )
        return category_df

    def risk_score_by_category(category_df: pd.DataFrame) -> pd.DataFrame:
        category_df.loc[:, "risk_score_max_twenty"] = (
            100
            - category_df["three_years_standard_deviation"].apply(
                lambda x: stats.percentileofscore(
                    category_df["three_years_standard_deviation"], x
                )
            )
        ) * 0.08
        category_df.loc[:, "risk_score_max_twenty"] += (
            100
            - category_df["five_years_standard_deviation"].apply(
                lambda x: stats.percentileofscore(
                    category_df["five_years_standard_deviation"], x
                )
            )
        ) * 0.12
        return category_df

    def total_score(category_df: pd.DataFrame) -> pd.DataFrame:
        category_df = return_score_by_category(category_df)
        category_df = risk_score_by_category(category_df)
        category_df.loc[:, "total_score"] = (
            category_df["risk_score_max_twenty"]
            + category_df["return_score_max_eighty"]
        )
        return category_df

    return total_score(df)


def save_to_excel(df: pd.DataFrame, file_name: str):
    """
    Save the DataFrame to an Excel file with the given file name.

    :param df: DataFrame to be saved.
    :param file_name: Name of the Excel file.
    """
    df.to_excel(file_name, sheet_name="Sheet1", index=False)
    print(f"DataFrame has been written to {file_name}")


def main():
    # Define the URL and parameters for the API request
    url = "https://lt.morningstar.com/api/rest.svc/g9vi2nsqjb/security/screener"
    params = {
        "page": "1",
        "pageSize": "-1",
        "outputType": "csv",
        "version": "1",
        "languageId": "en",
        "currencyId": "INR",
        "universeIds": "FOIND$$ALL|FCIND$$ALL",
        "securityDataPoints": "legalName,closePrice,closePriceDate,categoryName,Medalist_RatingNumber,starRatingM255,returnM12,returnM36,returnM60,returnM120,expenseRatio,initialPurchase,fundTnav,morningstarRiskM255,alphaM12,alphaM36,alphaM60,betaM12,betaM36,betaM60,standardDeviationM12,standardDeviationM36,standardDeviationM60,sharpeM12,sharpeM36,sharpeM60",
        "filters": "",
        "term": "",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br, zstd",
        "Referer": "https://www.morningstar.in/",
        "Origin": "https://www.morningstar.in",
        "DNT": "1",
        "Connection": "keep-alive",
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "cross-site",
        "Priority": "u=4",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "TE": "trailers",
    }

    try:
        # Fetch and preprocess data
        df = fetch_data_from_api(url, headers, params)
        df = preprocess_data(df)

        # Calculate scores
        category_dataframes = df.groupby("category_name")
        category_dataframes_with_total_scores = {
            category: calculate_scores(category_df)
            for category, category_df in category_dataframes
        }

        # Combine DataFrames and save to Excel
        excel_df = pd.concat(category_dataframes_with_total_scores.values())
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        file_name = f"./src/mf/mutual_funds_ranking_{now}.xlsx"
        save_to_excel(excel_df, file_name)

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    import os 
    print(os.path.exists( f"./src/mf/"))
    
    main()