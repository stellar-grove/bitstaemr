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
from sqlalchemy import types
import numpy as np
import time
import os
import bitstaemr.stuffs as stuffs
from bitstaemr import tools
import yfinance as yf



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





dataPath = f'C:/Users/DanielKorpon/Stellar Grove/bitstaemr - Documents/data/mkt/' + DB['database'] + '/' + DB['tgtSchema'] + '/' + DB['tgtTbl'] + '.csv'
#sqlcon = create_engine('mssql://' + servername + '/' + dbname + '?trusted_connection=yes')
engine = sqlalchemy.create_engine('mssql+pyodbc://' + DB['servername'] + '/' + DB['database'] + "?" + DB['driver'],echo=True)
tgtTbl = DB['tgtTbl']
tgtSchema = DB['tgtSchema']


class AlphaVantage(object):

    def __init__(self) -> None:
        self.config = {}
        self.ticker = []
        self.data = {}


    llave = tools.get_stones('AlphaVantage')
    function = 'BALANCE_SHEET'

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

    def __init__(self) -> None:
        self.config = {}
        self.ticker = []
        self.data = {}

    def setTicker(self, ticker):
        self.ticker = yf.Ticker(ticker)
        return yf.Ticker(ticker)


    def getHistory(self,ticker,time_period):
        print(ticker, time_period, len(ticker))
        tkr = yf.Ticker(ticker)
        history = tkr.history(period=time_period)
        self.data['history'] = history
        self.data['history_meta'] = tkr.history_metadata
        return history
    
    

    def getOptions(self,ticker):
        tkr = self.setTicker(ticker)
        chain = tkr.option_chain(tkr)
        return chain
    

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




