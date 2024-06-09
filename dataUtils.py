import os
import sys; sys.path.append("C:/stellar-grove/")
import pandas as pd
from bitstaemr import (stuffs, tools)
import statsapi as mlb




class LowCostDataInfrastructure(object):

    dbLocation = stuffs.folders.download

    def __init__(self,config={}) -> None:
        self.config = {"dbLocation":self.dbLocation}
        self.stats = {"error_details": []}
        self.data = {}
        self.log = {}

    def getDatabaseLocation(self):
        return self.dbLocation
    
    def getDB(self):
        db = []
        
        return db

    def setDatabaseLocation(self, dbLocation: str):
        self.config["dbLocation"] = dbLocation

    def createNewDatabase(self, dbName: str):
        dbName = f'{self.dbLocation}/{dbName}'
        os.mkdir(dbName)
        return
    
    def createNewSchema(self, dbPath: str, schemaName: str):
        schemaName = f'{dbPath}/{schemaName}'
        os.mkdir(schemaName)
    
    def createNewTable(self, schemaPath: str, tableName: str, columns: list):
        table = pd.DataFrame(columns=columns)
        table.to_csv(schemaPath,index=False)

class MLB(object):

    def __init__(self) -> None:
        
        self.log = {}

    def getMeta(metaType:str):
        mlb.meta

    def getSchedule(self, start_date:str='2024-01-01', end_date:str='2024-06-07'):
        schedule = mlb.schedule(start_date=start_date, end_date=end_date)
        schedule = tools.listDictionaryToDataFrame(schedule)
        return schedule
    
    def getBoxScore(self,game_id=int):
        box_score = mlb.boxscore_data(501205)
        return box_score
    
