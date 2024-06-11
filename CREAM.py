# -*- coding: utf-8 -*-

"""
Created on Thu Sep  9 11:52:27 2021

@author: DanielKorpon
"""
repo = "C:/stellar-grove/"
import sys;sys.path.append(repo)
import bitstaemr.stuffs as stuffs
import requests
import pandas as pd
import pyodbc as db
import pandas.io.json as json
import sqlalchemy
import numpy as np
import os
import bitstaemr.stuffs as stuffs
from bitstaemr import tools
import yfinance as yf
import datetime as dt
import scipy.stats

computerName = os.environ['COMPUTERNAME']
DB = {'servername': f'{computerName}\SQLEXPRESS' ,
      'database': 'fudge',
      'driver': 'driver=SQL Server Native Client 11.0'
      ,'tgtSchema':'alphav'
      ,'tgtTbl':'BalanceSheet'}

#server = 'STARFIGHER533\SQLEXPRESS' 
#database = 'fudge' 
#cnxn = db.connect('DRIVER={SQL Server};SERVER='+DB['servername']+';DATABASE=' + DB['database'])
dataPath = f'C:/Users/DanielKorpon/Stellar Grove/bitstaemr - Documents/data/mkt/fudge/BalanceSheet.csv'

database = DB['database']
server = DB['servername']
driver = DB['driver']
tgtTbl = DB['tgtTbl']
tgtSchema = DB['tgtSchema']

dataPath = f'C:/Users/DanielKorpon/Stellar Grove/bitstaemr - Documents/data/mkt/{database}/{tgtSchema}/{tgtTbl}.csv'
#sqlcon = create_engine('mssql://' + servername + '/' + dbname + '?trusted_connection=yes')
engine = sqlalchemy.create_engine(f'mssql+pyodbc://{server}/{database}?{driver},echo=True')

class AlphaVantage(object):

    def __init__(self, llave) -> None:
        self.llave = llave
        self.ticker = []
        self.data = {}

# - Constants -------------------------------------------------------
    llave = tools.get_stones('AlphaVantage')

# - Financials ------------------------------------------------------
    def BALANCE_SHEET(self,ticker):
        url = f'https://www.alphavantage.co/query?function=BALANCE_SHEET&symbol={ticker}&apikey={self.llave}'
        r = requests.get(url).json()
        df_qtrly = pd.DataFrame().from_dict(r['quarterlyReports'])
        df_annual = pd.DataFrame().from_dict(r['annualReports'])
        return (df_qtrly, df_annual)
    
    def INCOME_STATEMENT(self,ticker):
        url = f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={self.llave}'
        r = requests.get(url).json()
        df_qtrly = pd.DataFrame().from_dict(r['quarterlyReports'])
        df_annual = pd.DataFrame().from_dict(r['annualReports'])
        return (df_qtrly, df_annual)

    def CASH_FLOW(self,ticker):
        url = f'https://www.alphavantage.co/query?function=CASH_FLOW&symbol={ticker}&apikey={self.llave}'
        r = requests.get(url).json()
        df_qtrly = pd.DataFrame().from_dict(r['quarterlyReports'])
        df_annual = pd.DataFrame().from_dict(r['annualReports'])
        return (df_qtrly, df_annual)

    def EARNINGS(self,ticker):
        url = f'https://www.alphavantage.co/query?function=EARNINGS&symbol={ticker}&apikey={self.llave}'
        r = requests.get(url).json()
        df_qtrly = pd.DataFrame().from_dict(r['quarterlyEarnings'])
        df_annual = pd.DataFrame().from_dict(r['annualEarnings'])
        return (df_qtrly, df_annual)

    def OVERVIEW(self,ticker):
        url = f'https://www.alphavantage.co/query?function=OVERVIEW&symbol={ticker}&apikey={self.llave}'
        r = requests.get(url).json()
        df_qtrly = pd.DataFrame().from_dict(r,orient='index')
        return (df_qtrly)
    
    def DIVIDENDS(self,ticker):
        url = f'https://www.alphavantage.co/query?function=DIVIDENDS&symbol={ticker}&apikey={self.llave}'
        df = pd.DataFrame(requests.get(url).json()['data'])
        return (df)

# - Prices ------------------------------------------------------
    def INTRADAY(self,ticker, time_period):
        if time_period in ['1min', '5min', '15min', '30min', '60min']:
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={ticker}&interval={time_period}&outputsize=full&apikey={self.llave}'
            r = requests.get(url).json()
            key = f'Time Series ({time_period})'
            df = pd.DataFrame.from_dict(r[key],orient="index").reset_index()
            df.columns = ['datetime','open','high','low','close','volume']
            df.loc[:,'datetime'] = pd.to_datetime(df.loc[:,'datetime'])
            return df

    def DAILY(self,ticker,datatype='csv',outputsize:str='compact'):
        url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={ticker}&outputsize=full&apikey={self.llave}'
        r = requests.get(url).json()
        return r

    def set_ticker(self, ticker):
        if type(ticker) == list:
            lstTkrs = ticker
        if type(ticker) == str:
            lstTkrs = [ticker]
        self.ticker = lstTkrs
        return lstTkrs

    def get_data(self, ticker, function):
        for tkr in ticker:
            symbol = tkr
            url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={self.llave}'
            #print(url)
            r = requests.get(url)
            data = r.json()
            self.data = data

    def transform_data(self, data:dict):
        if len(data) > 0:
            ts = data['quarterlyReports']
            dfSrc = pd.DataFrame().from_dict(ts)
            dfSrc['fiscalDateEnding'] = pd.to_datetime(dfSrc['fiscalDateEnding'])
            dfSrc['symbol'] = symbol
            dfSrc['frequency'] = 'quarterly'
            dfSrc['RowId'] = dfSrc['fiscalDateEnding'].dt.strftime('%Y%m%d') + symbol + dfSrc['frequency']
            dfSrc = dfSrc.replace('None',0)
        
    spyder_text = '''
                        for tkr in lstTkrs:
                            symbol = tkr
                            url = f'https://www.alphavantage.co/query?function={function}&symbol={symbol}&apikey={llave}'
                            #print(url)
                            r = requests.get(url)
                            data = r.json()
                            if len(data) > 0:
                                ts = data['quarterlyReports']
                                dfSrc = pd.DataFrame().from_dict(ts)
                                dfSrc['fiscalDateEnding'] = pd.to_datetime(dfSrc['fiscalDateEnding'])
                                dfSrc['symbol'] = symbol
                                dfSrc['frequency'] = 'quarterly'
                                dfSrc['RowId'] = dfSrc['fiscalDateEnding'].dt.strftime('%Y%m%d') + symbol + dfSrc['frequency']
                                dfSrc = dfSrc.replace('None',0)
                                #dfSrc.to_csv(dataPath,header=True)
                                qry = f""" select * from {tgtSchema}.{tgtTbl} """
                                #dfDatabase = pd.read_sql(qry,cnxn)
                                #dfCsvColumns = dfDatabase.columns
                                dfCsv = pd.read_csv(dataPath)
                                dfCsv = (pd.read_csv(dataPath)).replace('None',0)
                                #dfDBCompare = dfSrc[~dfSrc['RowId'].isin(dfDatabase['RowId'])]
                                dfCsvCompare = dfSrc[~dfSrc['RowId'].isin(dfCsv['RowId'])]
                                chunk = np.floor(2100/dfSrc.shape[1]).astype(int)
                                #dfDBCompare.to_sql(tgtTbl,schema=tgtSchema,con=engine,if_exists='append',index=False, chunksize=chunk)
                                dfCsvCompare.to_csv(dataPath,mode='a',header=None)
                                time.sleep(13)

    '''

class stocks(object):

    def __init__(self, ticker) -> None:
        self.config = {}
        self.ticker = yf.Ticker(ticker)
        self.data = {}

    def setTicker(self, ticker):
        self.ticker = yf.Ticker(ticker)
        return yf.Ticker(ticker)

    def getHistory(self,time_period):
        history = self.ticker.history(period=time_period)
        self.data['history'] = history
        self.data['history_meta'] = self.ticker.history_metadata
        return history

    def getOptions(self,ticker):
        tkr = self.setTicker(ticker)
        chain = tkr.option_chain(tkr)
        return chain

    def getIncomeStatement(self):
        inc = self.ticker.income_stmt.T.reset_index()
        qtInc = self.ticker.quarterly_income_stmt.T.reset_index()
        return (inc, qtInc)

    def getBalanceSheet(self):
        bal = self.ticker.balance_sheet
        qtBal = self.ticker.quarterly_balance_sheet
        return (bal, qtBal)

    def getCashFlow(self):
        cf = self.ticker.cashflow
        qtCf = self.ticker.quarterly_cashflow
        return (cf, qtCf)
    
    def getCorporateActions(self):
        act = self.ticker.actions
        div = self.ticker.dividends
        splits = self.ticker.splits
        return (act, div, splits)

    spydertext = '''

        repo = 'C:/stellar-grove/'
import sys;sys.path.append(repo)
import bitstaemr.CREAM as cream
import yfinance as yf

s = cream.stocks()


tkr = yf.Ticker("XOM")


# get all stock info
tkr.info

# get historical market data
hist = tkr.history(period="max")

# show meta information about the history (requires history() to be called first)
tkr.history_metadata

# show actions (dividends, splits, capital gains)
act = tkr.actions
div = tkr.dividends
splt = tkr.splits
cg = tkr.capital_gains  # only for mutual funds & etfs

# show share count
full_shares = tkr.get_shares_full(start="2022-01-01", end=None)

# show financials:
# - income statement
inc = tkr.income_stmt
qtInc = tkr.quarterly_income_stmt
# - balance sheet
bal = tkr.balance_sheet
qtBal = tkr.quarterly_balance_sheet
# - cash flow statement
cf = tkr.cashflow
qtCf = tkr.quarterly_cashflow
# see `Ticker.get_income_stmt()` for more options

# show holders
mh = tkr.major_holders
ih = tkr.institutional_holders
mfh = tkr.mutualfund_holders
intx = tkr.insider_transactions
inpur = tkr.insider_purchases
inroshol = tkr.insider_roster_holders

# show recommendations
rec = tkr.recommendations
recsum = tkr.recommendations_summary
ugdg = tkr.upgrades_downgrades

# Show future and historic earnings dates, returns at most next 4 quarters and last 8 quarters by default. 
# Note: If more are needed use tkr.get_earnings_dates(limit=XX) with increased limit argument.
erndt = tkr.earnings_dates

# show ISIN code - *experimental*
# ISIN = International Securities Identification Number
isin = tkr.isin

# show options expirations
opt = tkr.options

# show news
news = tkr.news

# get option chain for specific expiration
optch = tkr.option_chain('2024-05-24')
# data available via: opt.calls, opt.puts

'''
    
class StockMVP(object):

    def __init__(self,ticker) -> None:
        self.ticker = ticker
        self.config = {'dl':stuffs.folders.download}
        self.data = {}
    
    def getIncomeStatement(self):
        folder = self.config['dl']
        file = f'{folder}{self.ticker}_income_statement_quarter.csv'
        df = pd.read_csv(file,keep_default_na=False)
        df = df.ffill(axis=0)
        mask = df.astype(bool).any(axis=1)
        df = df[mask]
        df = df.rename(columns={"date":"Section","Unnamed: 1":"Category"})

        return df.T


    spyder_text="""

period = 'quarter'
financial = 'income_statement'
tkr = 'NVDA'
file_name = f'{tkr.lower()}_{financial}_{period}.csv'
f = f'{stuffs.folders().download}{tkr.lower()}_{financial}_{period}.csv'
dfInit = pd.read_csv(f)

dfClean = dfInit.dropna(how='all')
dfClean.loc[:,"date"] = dfClean["date"].ffill()
dfClean.iloc[:4,0] = 'REVENUE'
dfClean = dfClean.dropna(thresh=len(dfClean.columns)-1)

"""

class MCMC(object):
    
    def __init__(self,
                 holdings:list=["WMT","TSLA","PLTR","NVDA","SPY"],
                 risk_free:float=0) -> None:
        self.data = {"holdings":holdings}
        self.ratios = {}
        self.stats = {'risk_free':risk_free,
                      }

    def gatherData(self, time_pd:str="1y",source:str="yFinance",file_path:str=None):
        # Using yFinance
        if source.lower() in ["yf", "yfin", "yfinance", "yahoo", "yhoo"]:
            lst_df = []
            for ticker in self.data["holdings"]:
                lst_df.append(stocks(ticker).getHistory(time_pd))
            columns = ['Close'] * len(lst_df)
            data = pd.concat([df[column] for df, column in zip(lst_df, columns)], axis=1)
            data.columns = self.data['holdings']
        # Pulling from a file
        if source.lower() in ['file', 'csv', 'txt']:
            if file_path == None:
                file_path = "C:/stellar-grove/"
                data = file_path


        return data

    def calc_sharpe_ratio(self, data, risk_free_rate = 0):
        mean_return = data["Daily Return"].mean()
        std = data["Daily Return"].std()
        sharpe_ratio = (mean_return-risk_free_rate) / std
        self.ratios = {'sharpe':sharpe_ratio}
        return sharpe_ratio
    
    def calc_sortino_ratio(self, data, target, risk_free_rate=0):
        mean_return = data["Daily Return"].mean()
        downside = data[data["Daily Return"] < target]["Daily Return"]
        std = downside.std()
        sortino_ratio = (mean_return-risk_free_rate) / std
        self.ratios = {'sortino':sortino_ratio}
        return sortino_ratio
    
    def compute_prob_sharpe_ratio(self, data, benchmark=0):
        sr = self.calc_sharpe_ratio(data, 0)
        skew = scipy.stats.skew(data["Daily Return"])
        # Use fisher kurtosis
        kurtosis = scipy.stats.kurtosis(data["Daily Return"], fisher=True)  
        n = len(data)
        std = ( (1 / (n-1)) * (1 + 0.5 * sr**2 - skew * sr + (kurtosis / 4) * sr**2))**0.5
        ratio = (sr - benchmark) / std
        prob_sharpe_ratio = scipy.stats.norm.cdf(ratio)
        self.ratios["psr"] = prob_sharpe_ratio
        return prob_sharpe_ratio
    
    def generate_random_weights(self, size):
        wts = np.random.random(size)
        return wts / np.sum(wts)
    
    def calc_returns(self, weights ,log_rets):
        return np.sum(log_rets.mean()*weights) * 252 #Annualized Returns

    def calc_log_returns(self, data):
        return np.log(data/data.shift(1))
        
    def calc_log_returns_cov_matrix(self,log_rets):
        return log_rets.cov()

    def calc_volatility(self, weights,log_rets_cov_mat):
        annualized_cov = np.dot(log_rets_cov_mat*252,weights)
        vol = np.dot(weights.transpose(),annualized_cov)
        self.stats["volatility"] = np.sqrt(vol)
        return np.sqrt(vol)

    def run_weight_opt(self, size, log_returns):
        # data: this should be the log returns.

        mc_portfolio_returns = []
        mc_portfolio_vol = []
        mc_weights = []
        N = len(self.data['holdings'])
        for sim in range(size):
            # This may take awhile!
            weights = self.generate_random_weights(size = N)
            mc_weights.append(weights)
            mc_portfolio_returns.append(self.calc_returns(weights,log_returns))
            mc_portfolio_vol.append(self.calc_volatility(weights,log_returns.cov()))
        self.data["mc_weights"] = mc_weights
        self.data["mc_vol"] = mc_portfolio_vol
        self.data['mc_returns'] = mc_portfolio_returns

spyder_text = """

dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
#sys.path.append(dk_repo)
import os
import numpy as np
from tara.tests import test_ticondagrova as tara_test

tara_test
tara_test.TestDistributions().test_generateTSP()


import bitstaemr.tools as tools
import bitstaemr.tests.tests_bitstaemr as bits_test
import bitstaemr.tests.data.bitstaemr as test_data

test_data.TEST_STONES
tools.get_stones('tester_bester')
bits_test.TestTools().test_get_stones()



os.getenv('StellarGrove')


from bitstaemr.CREAM import MCMC

mcmc = MCMC()
data = mcmc.gatherData(source='yf')
data2 = mcmc.gatherData(source="file")

len(mcmc.data['holdings'])
mcmc.generate_random_weights()
sims = mcmc.run_sim(100, data)

np.random.random(len(mcmc.data['holdings']))


"""