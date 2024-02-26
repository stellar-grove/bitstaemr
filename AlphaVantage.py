# -*- coding: utf-8 -*-

"""
Created on Thu Sep  9 11:52:27 2021

@author: DanielKorpon
"""

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

llave = 'A43QXKR43WZJ10VV'
function = 'BALANCE_SHEET'
lstTkrs = ['TSLA','PATH','MSFT','AMZN','FB','V','MNMD','PINS','PLTR','TROW','BLK','NVD','BA','INTC','SOFI','TDOC']

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