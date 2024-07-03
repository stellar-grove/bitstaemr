import os
import sys; sys.path.append("C:/stellar-grove/")
import pandas as pd
from bitstaemr import (stuffs as stuffs, tools as tools)
import statsapi as mlb
from datetime import datetime
import requests


def remove_characters(text):
    cleaned_text = text.replace(r'[^a-zA-Z0-9\s]', '')
    cleaned_text = cleaned_text.replace(' ','')
    cleaned_text = cleaned_text.replace('(','')
    cleaned_text = cleaned_text.replace(')','')
    cleaned_text = cleaned_text.replace(':','')    
    cleaned_text = cleaned_text.replace(';','')    
    cleaned_text = cleaned_text.replace(',','')
    cleaned_text = cleaned_text.replace('/','')
    cleaned_text = cleaned_text.replace('[','')
    cleaned_text = cleaned_text.replace(']','')
    cleaned_text = cleaned_text.replace('|','')    
    cleaned_text = cleaned_text.replace('-','')        
    cleaned_text = cleaned_text.replace('!','')        
    cleaned_text = cleaned_text.replace('&','')

    return cleaned_text

def determine_deltas(dfSource: pd.DataFrame, dfTarget: pd.DataFrame, lstIdCompare: list):
    """
    dfSource is 
    """
    sourceId = lstIdCompare[0]
    targetId = lstIdCompare[1]
    dfCompare = dfSource[~dfSource[sourceId].isin(dfTarget[targetId])]
    return dfCompare

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
        self.data_dir = f'{tools.taraWD}Baseball/data/'
        self.start_date = '2024-01-01'
        self.end_date = '2024-12-31'
        self.log = {}    

    def getSchedule(self, start_date:str='2024-01-01', end_date:str='2024-12-31'):
        schedule = mlb.schedule(start_date=start_date, end_date=end_date)
        schedule = tools.listDictionaryToDataFrame(schedule)
        return schedule
    
    def getBoxScore(self,game_id=int):
        box_score = mlb.boxscore_data(game_id)
        return box_score

    def getRoster(self,teamId,roster_type=None,return_df:bool=True):

        roster_lines = mlb.roster(teamId, rosterType=None, season=datetime.now().year, date=None).split('\n')
        roster = []
        for line in roster_lines:
            roster.append(line.split('  '))
        roster = pd.DataFrame(roster).dropna()
        roster.columns = ['NUMBER','POSITION','PLAYER_NAME']
        roster.loc[:,'TEAM_NUMBER'] = teamId
        return roster

    class infrastructure(object):
        def __init__(self)->None:
            self.data_dir = f'{tools.taraWD}Baseball/data/'
            self.database_path = f'{self.data_dir}schedule.csv'

        def get_meta(self,meta_type:str="baseballStats",return_list=True):
            if meta_type in self.meta_types:
                meta = mlb.meta(meta_type)
                for meta_type in meta:
                    print(meta_type)
                if return_list:
                    print(f"--------{meta_type}-------")
                    print(meta)
                    return meta

        def infrastructure_Teams(self,schedule:pd.DataFrame):
            list_teams = []
            away_teams = schedule[['away_id','away_name']]
            away_teams.columns = ['team_id','team_name']
            home_teams = schedule[['home_id','home_name']]
            home_teams.columns = ['team_id','team_name']
            teams = pd.concat([away_teams,home_teams],axis=0).to_dict()
            teams = pd.DataFrame().from_dict(teams,orient='index').T
            ids = teams.value_counts(['team_id','team_name']).reset_index().drop('count',axis=1)
            return ids

        def find_new_games(self, df:pd.DataFrame):
            current_schedule = df
            schedule_database =  pd.read_csv(self.database_path)
            current_schedule, schedule_database = current_schedule.align(schedule_database, join='inner', axis=1)
            deltas = determine_deltas(current_schedule, schedule_database,['game_id','game_id'])
            return deltas
        
        def get_database(self):
            return pd.read_csv(self.database_path)
        
        def write_to_csv(self,df:pd.DataFrame,file_name):
            df.to_csv(file_name,header=True,index=False,mode='a')
            text = f'''There were {df.shape[0]} records written to the database.
            The database is found at {self.database_path}'''
            return print(text)



        meta_types = [
            'awards',
            'baseballStats',
            'eventTypes',
            'gameStatus',
            'gameTypes',
            'hitTrajectories',
            'jobTypes',
            'languages',
            'leagueLeaderTypes',
            'logicalEvents',
            'metrics',
            'pitchCodes',
            'pitchTypes',
            'platforms',
            'positions',
            'reviewReasons',
            'rosterTypes',
            'scheduleEventTypes',
            'situationCodes',
            'sky',
            'standingsTypes',
            'statGroups',
            'statTypes',
            'windDirection',

        ]
        #def infrastructure_WriteRoster


        spyder_text = """

mlb = du.MLB()

start_date = '2024-01-01'
end_date = '2024-12-31'
schedule = mlb.getSchedule(mlb.start_date,mlb.end_date)
schedule.columns
columns = ['game_id','game_date','game_num','game_type','status','home_id','home_name','away_id','away_name',
           'winning_team','losing_team', 'home_score', 'away_score',
           'current_inning','series_status']

clean_schedule = schedule[columns]

game_id = 747023
box_score = mlb.getBoxScore(game_id)
box_score.keys()
box_score['awayPitchers']

box_score['teamInfo']
box_score['gameId']

clean_schedule[clean_schedule['game_id']==game_id]   

teams = mlb.infrastructure_Teams(schedule) 

ids = teams.groupby(by=['team_id','team_name']).count()

print( statsapi.player_stats(next(x['id'] for x in statsapi.get('sports_players',{'season':2008,'gameType':'W'})['people'] if x['fullName']=='Chase Utley'), 'pitching', 'career') )



lst = statsapi.player_stats(554430,group='pitching').split('\n')


roster_lines = mlb.getRoster(103).split('\n')
roster = []
for line in roster_lines:
    roster.append(line.split('  '))
    
roster = mlb.getRoster(103)


infra = mlb.infrastructure()
for meta_type in infra.meta_types:
    infra.get_meta(meta_type)


"""

class dosia(object):
    
    class MD(object):
        def __init__(self)->None:
            self.data = {}
            self.stats = {}

    # -----------------------------------------------------------------------------------------------------
    # Constants
    # -----------------------------------------------------------------------------------------------------  
        DOWNLOAD_FOLDER = f'{stuffs.folders.cannabis_data}'
        DOWNLOAD_PATH = f'{stuffs.folders.cannabis_data}dosia_data.xlsx'
        MKT_SALES_DB = f'{DOWNLOAD_FOLDER}MD/mkt_sales.csv'
        
        def DOWNLOAD(self):
            url = 'https://cannabis.maryland.gov/Documents/2024%20Data%20Dashboard/MCA%20Dashboard%20Data%20Download.xlsx'
            r = requests.get(url, stream = True)
            with open(self.DOWNLOAD_PATH,"wb") as excel: 
                for chunk in r.iter_content(chunk_size=1024):         
                    # writing one chunk at a time to pdf file 
                    if chunk: 
                        excel.write(chunk)
                        
        def READ_DATA(self):
            try:
                df = pd.read_excel(self.DOWNLOAD_PATH, sheet_name='Market Sales Data')
                df['us_state'] = 'MD'
            except FileNotFoundError:
                print(f"Error: File {self.DOWNLOAD_PATH} not found")
            except KeyError:
                print(f"Error: Table {self.DOWNLOAD_PATH} not found")
            self.data['mkt_sales'] = df
            return df
        
        def TRANSFORM_DATA(self):
            df = self.data['mkt_sales']
            for col in df.columns:
                df = df.rename(columns = {col:remove_characters(col)})
            df['date'] = df['Month'] + ' ' + df['Year'].astype(str)
            df['date'] = pd.to_datetime(df['date'],format="%B %Y").dt.strftime("%Y-%m-%d")
            df['id'] = (df['us_state'] + '.' +
                        df['SalesCustomerType'].apply(remove_characters) + '.' + 
                        df['date'].apply(remove_characters) + '.' + 
                        df['ProductType'].apply(remove_characters) + '.' + 
                        df['ProductCategoryName'].apply(remove_characters)
            )
            df['id'] = df['id'].apply(str.upper)
            self.data['mkt_sales'] = df
            return df
        
        def PROCESS_SALES(self,print_df:bool=False):
            df = self.data['mkt_sales']
            db = pd.read_csv(self.MKT_SALES_DB)
            deltas = determine_deltas(df,db,['id','id'])
            self.data['deltas'] = deltas
            if print_df:
                return deltas

        def WRITE_NEW_RECORDS(self):
            deltas = self.data['deltas']
            path = self.MKT_SALES_DB
            deltas.to_csv(path, mode='a', header=False, index=False)
            self.stats['records_written'] = deltas.shape[0]
            print(self.stats)
            
        def RUN(self):
            self.DOWNLOAD()
            self.READ_DATA()
            self.TRANSFORM_DATA()
            self.PROCESS_SALES()
            self.WRITE_NEW_RECORDS()

        
    

        
        
