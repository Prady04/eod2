from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import datetime as dt
import os
import time
import logging
import logging.config
import sqlite3
from colorama import Fore, Style
import Configs
import shutil
from tabulate import tabulate


class MarketSmithScraper:
    def __init__(self):
        logging.getLogger('matplotlib.font_manager').disabled = True
        logging.config.fileConfig('src/msmithlog.conf')
        self.log = logging.getLogger('MSMITH')
        self.driver = self.setup_driver()
        self.now = dt.datetime.now()
        self.db_path = 'src/nhnltrend.db'

    def setup_driver(self):
        options = webdriver.ChromeOptions()
        options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        options.add_experimental_option('excludeSwitches', ['enable-logging'])
        options.add_argument('--log-level=3')
        return webdriver.Chrome(options)

    def login(self):
        try:
            self.driver.get("https://marketsmithindia.com/mstool/landing.jsp#/signIn")
            WebDriverWait(self.driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//button[text()='Accept']"))
            ).click()
        except Exception:
            pass

        try:
            self.driver.get("https://marketsmithindia.com/mstool/landing.jsp#/signIn")
            host = WebDriverWait(self.driver, 30).until(EC.presence_of_element_located((By.ID, 'userlogin')))
            root = self.driver.execute_script("return arguments[0].shadowRoot", host)
            root.find_element(By.NAME, "Email").send_keys(Configs.user_name)
            root.find_element(By.NAME, "Password").send_keys(Configs.pwd)
            root.find_element(By.CLASS_NAME, 'loginsubmitbtn').click()
            time.sleep(5)
            self.log.debug("Login successful")
        except Exception as e:
            self.log.error(f"Login failed: {e}")
            self.driver.quit()
            raise

    def fetch_and_process(self, url, alert_xpath, label, keep_file=False):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 60).until(EC.element_to_be_clickable((By.XPATH, alert_xpath))).click()

            before = set(os.listdir(Configs.dl_path))
            WebDriverWait(self.driver, 60).until(EC.element_to_be_clickable((By.ID, "ideaDownloadList"))).click()
            time.sleep(5)
            after = set(os.listdir(Configs.dl_path))

            new_file = list(after - before)
            if new_file:
                path = os.path.join(Configs.dl_path, new_file[0])
                return self.process_csv(path, label, keep_file)
            else:
                self.log.warning(f"{label}: File download failed")
        except Exception as e:
            self.log.error(f"Error in {label}: {e}")
        return pd.DataFrame()

    def process_csv(self, csv_file, qualifier, keep_file):
        df = pd.read_csv(csv_file)
        columns = df.columns.tolist()
        columns = columns[1:] + [columns[0]]
        df.columns = columns
        df = df[df.Symbol.str.startswith('5') == False]
        df = df[df['RS_Rating'] >= 80].sort_values('RS_Rating', ascending=False)
        if not keep_file:
            os.remove(csv_file)
        df.to_html(f"{qualifier}.html")
        return df

    def write_to_db(self, df, idea):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        skipped = []
        dt_now = dt.datetime.now()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS msmith (
                id INTEGER PRIMARY KEY,
                symbol TEXT,
                name TEXT,
                price NUMERIC,
                pChange NUMERIC,
                rs_rating INTEGER,
                grp_rank INTEGER,
                dat DATETIME,
                idea TEXT
            )
        ''')

        for _, row in df.iterrows():
            try:
                cursor.execute("SELECT 1 FROM msmith WHERE symbol = ?", (row.Symbol,))
                if cursor.fetchone():
                    skipped.append(row.Symbol)
                else:
                    cursor.execute('''
                        INSERT INTO msmith (symbol, name, price, pChange, rs_rating, grp_rank, dat, idea)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        row.Symbol, row.CompanyName, row.Cur_Price, row.Price_Percentage_chg,
                        row.RS_Rating, row.Group_Rank, dt_now, idea
                    ))
            except Exception as e:
                self.log.error(f"DB insert error: {e}")
        conn.commit()
        conn.close()
        return skipped

    def print_on_screen(self, df):
        now = dt.datetime.now()
        filename = f"prady{now.strftime('%Y%m%d%S')}.csv"
        df.to_csv(filename)

        print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))
        symbols = ",".join(df['Symbol'].dropna().astype(str).tolist())
        print(f"\n{symbols}\n")

    def run(self):
        self.login()

        df_mm5 = self.fetch_and_process(
            "https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/minervini-trend-template-5-months/idealists.jsp#/",
            '//*[@id="52_82_subIconAlertBox"]/div/div/div/div[3]/button', "mm5"
        )

        df_mm1 = self.fetch_and_process(
            "https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/minervini-trend-template-1-month/idealists.jsp#/",
            '//*[@id="52_81_subIconAlertBox"]/div/div/div/div[3]/button', "mm1"
        )

        df_nps = self.fetch_and_process(
            "https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/top-rated-ipos/idealists.jsp#/",
            '//*[@id="52_92_subIconAlertBox"]/div/div/div/div[3]/button', "nps", keep_file=True
        )

        df_combined = pd.concat([df_mm5, df_mm1])
        self.write_to_db(df_combined, "mm")
        self.print_on_screen(df_combined)

        df_combined.reset_index(inplace=True)
        df_combined.rename(columns={"Symbol": "nsecode"}, inplace=True)
        df_combined.to_csv("C:\\python\\flexstart\\cache\\mm.csv", index=False)
        shutil.copy("C:\\python\\flexstart\\cache\\mm.csv", "C:\\Nshare\\mm.csv")

        if not df_nps.empty:
            df_total = pd.concat([df_combined, df_nps])
            self.print_on_screen(df_nps)
            self.write_to_db(df_nps, "nps")
        else:
            df_total = df_combined

        df_total["Symbol"] = df_total["nsecode"]
        df_cleaned = df_total.drop(columns=["index", "Sno", "nsecode"], errors="ignore").set_index("Symbol")
        json_output = df_cleaned.reset_index().to_json("others.json", orient="records", indent=2)
        self.driver.quit()


if __name__ == "__main__":
    scraper = MarketSmithScraper()
    scraper.run()
