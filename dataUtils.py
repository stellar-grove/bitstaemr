import os
import sys; sys.path.append("../")
import pandas as pd
import stuffs




class LowCostDataInfrastructure(object):

    dbLocation = constants.dbLocation

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
