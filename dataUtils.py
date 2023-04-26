import os
import sys
import pandas as pd


class LowCostDataInfrastructure(object):

    dbLocation = "C:/Users/DanielKorpon/Stellar Grove/dkorpon - Documents/data"

    def __init__(self,config={}) -> None:
        self.config = {"dbLocation":self.dbLocation}
        self.stats = {"error_details": []}
        self.data = {}

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
    
    def createNewSchema(self, dbPath: str, schemaName: str):
        schemaName = f'{dbPath}/{schemaName}'
        os.mkdir(schemaName)
    
    def createNewTable(self, schemaPath: str, tableName: str, columns: list):
        table = pd.DataFrame(columns=columns)
        table.to_csv(schemaPath,index=False)
