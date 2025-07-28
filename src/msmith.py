from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.alert import Alert
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import Configs
import pandas as pd
import datetime as dt
from colorama import Fore,Back,Style
import logging
import logging.config


from tabulate import tabulate




import os
import time
import sqlite3
import datetime

logging.getLogger('matplotlib.font_manager').disabled = True

logging.config.fileConfig('src/msmithlog.conf')

# create logger
log = logging.getLogger('MSMITH')
log.setLevel(level=logging.DEBUG)

#logger = HTMLLogger("")

        
def process_csv(csv_file,qualifier, keep_file):    
    stocklist = []
    print(csv_file)
    df = pd.read_csv(csv_file)

    current_columns = df.columns.tolist()

    # Remove the first column name ('Sno')
    current_columns.pop(0)

        # Append the first column name to the end
    current_columns.append(df.columns[0])

    # Rename the DataFrame columns with the shifted column names
    df.columns = current_columns
    #print(df)
    import random
    df = df[df.Symbol.str.startswith('5')==False]
    #print(df1.head())
    df = df.sort_values('RS_Rating', ascending=False)
    #print(df)
    df =df.query('RS_Rating >= 80')   
    keep_file = False
    if not keep_file:  
        os.remove(csv_file)
   
    df.to_csv(f'C:\\Nshare\\NPS'+str(random.random())+qualifier+'.csv')
    df.to_html(qualifier+".html")
    print('wrote to html')
        #print(tabulate([list(row) for row in df.values], headers=list(df.columns),tablefmt='html'))

    return df


def getShadowRoot(host,driver):
    shadow_root = driver.execute_script("return arguments[0].shadowRoot", host)
    return shadow_root  


def do_login(driver):
   
    driver.get("https://marketsmithindia.com/mstool/landing.jsp#/signIn")
    print(driver.title)
    try:
        accept_button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[text()='Accept']")))
      
        accept_button.click()
        driver.maximize_window()
    except Exception as e:
        pass
    try:
        driver.get("https://marketsmithindia.com/mstool/landing.jsp#/signIn")
        host1 = WebDriverWait(driver,30).until(EC.presence_of_element_located((By.ID,'userlogin')))
        root1 = getShadowRoot(host1,driver)
        txt_user = root1.find_element(By.NAME,"Email").send_keys(Configs.user_name)
        txt_pwd = root1.find_element(By.NAME,"Password").send_keys(Configs.pwd)
        btn_pedircita = root1.find_element(By.CLASS_NAME,'loginsubmitbtn')
        btn_pedircita.click()
        time.sleep(5)
        log.debug('Login success')
      
        print(Style.RESET_ALL)
        time.sleep(5)
        
    except Exception as e:
        driver.close()
        log.error(e)
        print('failure in login')
        exit()
    return

def fetch_mm5(driver):
    try:
        desired_url = "https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/minervini-trend-template-5-months/idealists.jsp#/"
        driver.get(desired_url)
        log.info('loaded minervini url 1')
        log.debug('processing mm5')
        
        accept_button = WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="52_82_subIconAlertBox"]/div/div/div/div[3]/button')))
        accept_button.click()
        
        log.debug('clicked close')
        #driver.switch_to.frame('landingIframe')
      
        before = os.listdir(Configs.dl_path)

        idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')))
        try:
            idealist.click()
        except Exception as e:
            accept_button.click()
            idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')))
            idealist.click()
        log.debug('downloading..')
        time.sleep(5)
        after = os.listdir(Configs.dl_path)
        change = set(after) - set(before)
        if len(change) == 1:
            file_name = change.pop()
            log.debug(f'completed downloading {file_name}')
           
            return process_csv(Configs.dl_path+'\\'+file_name, "mm5",False)         
        else:
            print(Fore.RED,"Something went wrong cant get file") 
            
            print(Style.RESET_ALL)    
    except Exception as e:
        print(Fore.RED,"Something went wrong ")   
        print(Style.RESET_ALL)   
        print(e, "mm5")
        
        
        return(pd.DataFrame())
        
   
 
        
def fetch_mm1(driver):
    try:
        desired_url = "https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/minervini-trend-template-1-month/idealists.jsp#/"
        driver.get(desired_url)    
        log.debug('loaded minervini url 2')    
        #/html/body/div[52]/div/div/div/div[3]/button   
        #accept_button = WebDriverWait(driver, 360).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="52_82_subIconAlertBox"]/div/div/div/div[3]/button')))  
        accept_button = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="52_81_subIconAlertBox"]/div/div/div/div[3]/button')))
        accept_button.click()

        
        log.debug('clicked close')
        before = os.listdir(Configs.dl_path)

        idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')))
        try:
            idealist.click()
        except Exception as e:
            accept_button.click()
            idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')))
            idealist.click()
        log.debug('downloading..')
        time.sleep(5)
        after = os.listdir(Configs.dl_path)
        change = set(after) - set(before)
        if len(change) == 1:
            file_name = change.pop()
            
            log.debug(f'completed downloading {file_name} ' )
            print(Style.RESET_ALL)            
            return process_csv(Configs.dl_path+'\\'+file_name, 'mm1',False)      
        else:
            print(Fore.RED,"Something went wrong cant get file")           
            print(Style.RESET_ALL)        
    except Exception as e: 
        print(e)
       
        return pd.DataFrame()
        
        
        
                
def fetch_nps(driver):
    url = 'https://marketsmithindia.com/mstool/list/marketsmith-stock-screens/top-rated-ipos/idealists.jsp#/'
    try:      
        driver.get(url)    
        log.debug('loaded top IPO Stocks')  
        #/html/body/div[63]/div/div/div/div[3]/button
        #accept_button = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '/html/body/div[63]/div/div/div/div[3]/button')))
        accept_button = WebDriverWait(driver, 60).until(EC.visibility_of_element_located((By.XPATH, '//*[@id="52_92_subIconAlertBox"]/div/div/div/div[3]/button')))
      
        accept_button.click()
        log.debug('clicked close')
       
        before = os.listdir(Configs.dl_path)
        idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')),"Timedoout in NPS")
        try:
            idealist.click()
        except Exception as e:
            accept_button.click()
            idealist = WebDriverWait(driver, 60).until(EC.element_to_be_clickable((By.XPATH, '//*[@id="ideaDownloadList"]')))
            idealist.click()
        log.info('downloading..')
        time.sleep(5)
        after = os.listdir(Configs.dl_path)
        change = set(after) - set(before)
        if len(change) == 1:
            file_name = change.pop()
            
            log.info('completed downloading ' + file_name)
            print(Style.RESET_ALL)
            
            
            return process_csv(Configs.dl_path+'\\'+file_name,'ipo', True)      
        else:
            print(Fore.RED,"Something went wrong cant get file")
            print(Style.RESET_ALL) 
            
            return(pd.DataFrame())
            
    except Exception as e: 
        print('Error Occured')
        
        return(pd.DataFrame())
        #driver.close()
        #engine.stop()
        #exit()
 

def adapt_date_iso(val):
    """Adapt datetime.date to ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_iso(val):
    """Adapt datetime.datetime to timezone-naive ISO 8601 date."""
    return val.isoformat()

def adapt_datetime_epoch(val):
    """Adapt datetime.datetime to Unix timestamp."""
    return int(val.timestamp())

sqlite3.register_adapter(datetime.date, adapt_date_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_iso)
sqlite3.register_adapter(datetime.datetime, adapt_datetime_epoch)

def convert_date(val):
    """Convert ISO 8601 date to datetime.date object."""
    return datetime.date.fromisoformat(val.decode())

def convert_datetime(val):
    """Convert ISO 8601 datetime to datetime.datetime object."""
    return datetime.datetime.fromisoformat(val.decode())

def convert_timestamp(val):
    """Convert Unix epoch timestamp to datetime.datetime object."""
    return datetime.datetime.fromtimestamp(int(val))

sqlite3.register_converter("date", convert_date)
sqlite3.register_converter("datetime", convert_datetime)
sqlite3.register_converter("timestamp", convert_timestamp)

def write_to_db(df,idea):
    conn=None
    skipped_symbols = []
    dt = datetime.datetime.now()
    for index, row in df.iterrows():
        try:
            # Create a table to store the data if it doesn't exist
            conn = sqlite3.connect('src/nhnltrend.db')
            #print('q5')
            cursor = conn.cursor()
            #print('q0') 
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS msmith (
                    id INTEGER PRIMARY KEY,
                    symbol TEXT,
                    name TEXT,
                    price NUMERIC,
                    pChange NUMERIC,
                    rs_rating INTEGER,
                    grp_rank INTEGER,
                    dat DATETIME
                
                )
            ''')  
            #print('q1')  
            cursor.execute("SELECT 1 FROM msmith WHERE symbol = ?", (row.Symbol,))
            existing_data = cursor.fetchone()
            if not existing_data:
                #print('q2')  
                cursor.execute('''
                    INSERT INTO msmith (symbol, name, price,pChange,rs_rating,grp_rank,dat, idea)
                    VALUES (?, ?, ?,?,?,?,?,?)''',
                    (row.Symbol,row.CompanyName,row.Cur_Price,row.Price_Percentage_chg, row.RS_Rating, row.Group_Rank, dt, idea)) 
                conn.commit()
                #print('q3')  
            else:
                skipped_symbols.append(row.Symbol) 
                
            
        except Exception as e:
            log.error(e)
            print('Error in writetodb line 265')         
            pass
        finally:
            
            conn.close()
        return skipped_symbols 

def print_on_screen(df):
    from tabulate import tabulate
    from datetime import datetime
    try:
        now = datetime.now()
        f = now.strftime("%Y%m%d%S")
        #df.to_csv('prady'+f+'.csv')
    except Exception as e:
        print(e)
    
    #print(Fore.GREEN,Style.BRIGHT,tabulate(df.query('Price_Percentage_chg>-1.5 and Price_Percentage_chg < 1.5'),headers=df.keys(),tablefmt='psql', showindex=False))
    log.log(25,tabulate(df, headers=df.keys(),tablefmt='simple',showindex=False))
    #print(df)
    #df.to_csv('mm.csv')
    #print((df['Symbol']).to_string(index=False))
    #fname = 'MS'+dt.datetime.now().strftime('%d%m%Y%H%M')+'.csv'
    #df.to_csv(fname)
    #df = df.query('Price_Percentage_chg>-1.5 and Price_Percentage_chg < 1.5')

    st = ""
    strs = ",".join(df['Symbol'])
    strs=strs.replace("'", '')
    strs = strs.strip('[]')
    print('\n',strs)
    
       
   
    
    '''print(Fore.GREEN,Style.BRIGHT,tabulate(nps.query('Price_Percentage_chg>-1.5 and Price_Percentage_chg < 1.5'),headers=nps.keys(),tablefmt='psql', showindex=False))
    nps = nps.query('Price_Percentage_chg>-1.5 and Price_Percentage_chg < 1.5')
    nps.reset_index(inplace = True)
    nps=nps.rename({'Symbol': 'nsecode'}, axis='columns')
    strs = ",".join(nps['Symbol'])
    strs=strs.replace("'", '')
    strs = strs.strip('[]')
    print(strs)'''


import shutil
def msmith():

    sf = []

    chrome_options = webdriver.ChromeOptions()
    chrome_options.binary_location = "C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
    chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
    chrome_options.add_argument('--log-level=3')
   #chrome_options.add_argument("--user-data-dir=chrome-data")
    # Initialize the WebDriver
    ''' 
    # Save the existing cookies from a previous session
    existing_cookies = driver.get_cookies()

    # Close the WebDriver for the current session
    driver.quit()

    # Initialize a new WebDriver instance
    driver = webdriver.Chrome(options=chrome_options)

    # Load the existing cookies into the new WebDriver session
    for cookie in existing_cookies:
        driver.add_cookie(cookie)
        print(cookie)'''
    
    driver = webdriver.Chrome(chrome_options)
    df1=pd.DataFrame()
    df2 = pd.DataFrame()
    nps = pd.DataFrame()
    df = pd.DataFrame()
    do_login(driver)
    log.debug("Done with login")
    print('Login Success')
    df1 = fetch_mm5(driver)
    log.debug('mm5 fetched: '+ str(len(df1)))
    print('done with mm5')
    df2 = fetch_mm1(driver)
    log.debug('mm1 fetched : '+ str(len(df2)))
    print('done with mm1')
    nps = fetch_nps(driver) 
    log.debug('nps fetched: '+ str(len(nps)))
    print('done with ipo')
    


    
    
   
    if not df1.empty and (len(df1) > 0):
        df = df1        
        if len(df2)>0:
            df = pd.concat([df,df2]) 
    else:
        if len(df2) >1:
            df =df2
            
    
    
    
  
    sf.append(write_to_db(df,"mm"))
    print_on_screen(df)

    df.reset_index(inplace = True)
    df=df.rename({'Symbol': 'nsecode'}, axis='columns')
    df.to_csv(f'C:\\python\\flexstart\\cache\\mm.csv') 
    #shutil.copyfile('C:\\python\\flexstart\\cache\\mm.csv', 'C:\\Nshare\\mm.csv')
   
    if len(nps)>0:
        df = pd.concat([df,nps])
        print_on_screen(nps)
        log.info('writing to db...') 
        sf.append(write_to_db(nps, "nps"))
        #log.debug('Done')
    print(Fore.GREEN,'all done ')
    driver.close()
    df.to_csv('onlytemp.csv')
 
  
    df['Symbol'] = df['nsecode']  # Copy nsecode values to Symbol column
    df_cleaned = df.drop([ 'index', 'Sno', 'nsecode'], axis=1)
    df_cleaned = df_cleaned.set_index('Symbol')

    print("Cleaned DataFrame:")
    print(df_cleaned.head())
    print("\nDataFrame info:")
    print(df_cleaned.info()) 
    df = df[['Symbol','CompanyName','RS_Rating']]
    df_clean = df.dropna(subset=['Symbol'])  # Remove null symbols
    df_clean = df_clean.drop(['level_0', 'index', 'Sno'], axis=1, errors='ignore')
    df_clean = df_clean.set_index('Symbol')  # Symbol as row names
    df_renamed_single = df_clean.rename(columns={'CompanyName': 'Company Name'})
    shutil.copyfile("C:\\npm\\dotchart-main\\public\\others.json", "C:\\npm\\dotchart-main\\public\\others_b.json")
    json_output = df_renamed_single.reset_index().to_json("C:\\npm\\dotchart-main\\public\\others.json",orient='records', indent=2)
    print(json_output)


if __name__=="__main__":   
    msmith()

    
    
    
    