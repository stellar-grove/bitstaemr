import os
import sys; sys.path.append("C:/stellar-grove/")
import pandas as pd
from bitstaemr import (stuffs as stuffs, tools as tools)
import statsapi as mlb
from datetime import datetime, timedelta, date
import requests
import numpy as np
import pybaseball as pyball
import sqlalchemy

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

    def getPlayers(self,return_df:bool=False):
        player_stats = mlb.lookup_player('')
        df_players = pd.DataFrame()
        for i in range(len(player_stats)):
            player = player_stats[i]
            new = pd.json_normalize(player)
            if new.shape[1] == 19:
                new['nickName'] = np.nan
            df_players = pd.concat([df_players,new],axis=0)
            if return_df:
                return df_players
    
    def getAttendance(self, game_id=int, return_df:bool=False):
        game_info = self.getBoxScore(game_id)['gameBoxInfo']
        for i in range(len(game_info)):
            if game_info[i]['label'] == 'Att':
                attendance = int(game_info[i]['value'].replace(",","").replace(".",""))
                return attendance
            #else:
            #    return None
        
    class statcast(object):

        def __init__(self) -> None:
            self.data_dir = f'{tools.taraWD}Baseball/data/'
            self.database_loc = f'{tools.taraWD}Baseball/data/statcast.csv'
            self.start_date = '2024-01-01'
            self.end_date = '2024-12-31'
            self.log = {}   

        def get_data(self,start_date=None,end_date=None,team=None,return_df:bool=False):
            df = pyball.statcast(start_dt=start_date, end_dt=end_date, team=team)
            df['id'] = df['game_pk'].astype(str) + '.' + df['at_bat_number'].astype(str) + '.' + df['pitch_number'].astype(str)
            sort_columns = ['game_pk','at_bat_number','pitch_number']
            df = df.sort_values(by=sort_columns)
            if return_df:
                return df
        
        def determine_deltas(self,data,return_df:bool=False):
            df_source = data
            df_target = MLB.infrastructure().get_database(database_name='statcast')
            deltas = determine_deltas(df_source, df_target, ['id','id'])
            if return_df:
                return deltas
        
        def write_to_csv(self,data:pd.DataFrame):
            file_location = self.database_loc
            data.to_csv(file_location,header=False,index=False,mode='a')
            records = data.shape[0]
            db_records_final = pd.read_csv(file_location).shape[0]
            text = f'There were {records} written to the database.  The current number of records in the database is {db_records_final}.'
            print(text)

        def write_to_sql(self,data:pd.DataFrame,cnxn:str):
            engine = sqlalchemy.create_engine(f'mssql+pyodbc://{server}/{database}?{driver},echo=True')
            

            return 'oppo'

        def DRAFT(self, year:int, round, keep_stats:bool=True):
            df = pyball.amateur_draft(year, draft_round=1, keep_stats=keep_stats)
            return df

        def DRAFT_TEAM(self, team:str="PHI", year:int=None, keep_stats:bool=True):
            df = pyball.amateur_draft_by_team(team, year, keep_stats=False)
            return df

        def BWAR_BAT(self):
            df = pyball.bwar_pitch(return_all=True)
            return df

        def BWAR_PITCH(self):
            df = pyball.bwar_pitch(return_all=True)
            return df

        def BATTING_STATS(self,start:int=None,end:int=None):
            df = pyball.batting_stats(start,end)
            return df

        def BATTING_STATS_BREF(self,season:int=None):
            df = pyball.batting_stats_bref(season)
            return df
        
        def CATCHER_FRAMING(self, season:int=None):
            df = pyball.statcast_catcher_framing(season)
            return df

        def CATCHER_POPTIME(self, season:int=None):
            df = pyball.statcast_catcher_poptime(season)
            return df
        
        def CATCHER_POPTIME(self, season:int=None):
            df = pyball.statcast_outfielder_jump(season)
            return df
        
        def PITCHER_EXPECTED_STATS(self, season:int=None):
            df = pyball.statcast_pitcher_expected_stats(season)
            return df

        def PITCHER_PITCH_ARSENAL(self, season:int=None):
            df = pyball.statcast_pitcher_pitch_arsenal(season)
            return df
        
        def PITCHER_PITCH_ARSENAL_SPEED(self, season:int=None):
            df = pyball.statcast_pitcher_pitch_arsenal(season,arsenal_type="avg_speed")
            return df
        
        def PITCHER_PITCH_ARSENAL_SPIN(self, season:int=None):
            df = pyball.statcast_pitcher_pitch_arsenal(season,arsenal_type="avg_spin")
            return df
        
        def PITCHER_PITCH_ARSENAL_PCT(self, season:int=None):
            df = pyball.statcast_pitcher_pitch_arsenal(season,arsenal_type="n_")
            return df
        
        def PITCHER_PITCH_ARSENAL_STATS(self, season:int=None):
            df = pyball.statcast_pitcher_arsenal_stats(season)
            return df
        
        def PITCHER_PERCENTILE_RANKS(self, season:int=None):
            df = pyball.statcast_pitcher_percentile_ranks(season)
            return df
        
        def SPRINT_SPEED(self, season:int=None):
            df = pyball.statcast_sprint_speed(season)
            return df
        
        def RUNNING_SPLITS(self, season:int=None):
            df = pyball.statcast_running_splits(season)
            return df
        
        def SPRINT_SPEED(self, season:int=None):
            df = pyball.statcast_single_game(season)
            return df
        
        def SINGLE_GAME(self, game_id:int=529429):
            df = pyball.statcast_single_game(game_id)
            return df
   
        def run(self,game_date=None,team=None):
            
            today = date.today()
            yesterday = today - timedelta(days=1)
            yesterday = yesterday.strftime("%Y-%m-%d")
            data = self.get_data(start_date=yesterday,
                                 end_date=yesterday,
                                 team=None,
                                 return_df=True)            
            deltas = self.determine_deltas(data,return_df=True)
            if deltas.shape[0] == 0:
                print("No records to be written")
                return None
            else:
                self.write_to_csv(deltas)
            
    class infrastructure(object):

        def __init__(self)->None:
            self.data_dir = f'{tools.taraWD}Baseball/data/'
            self.db_schedule = f'{self.data_dir}schedule.csv'

        def get_meta(self,meta_type:str="baseballStats",return_list=True):
            if meta_type in self.meta_types:
                meta = mlb.meta(meta_type)
                for meta_type in meta:
                    print(meta_type)
                if return_list:
                    print(f"--------{meta_type}-------")
                    print(meta)
                    return meta

        def get_teams(self,schedule:pd.DataFrame):
            list_teams = []
            away_teams = schedule[['away_id','away_name']]
            away_teams.columns = ['team_id','team_name']
            home_teams = schedule[['home_id','home_name']]
            home_teams.columns = ['team_id','team_name']
            teams = pd.concat([away_teams,home_teams],axis=0).to_dict()
            teams = pd.DataFrame().from_dict(teams,orient='index').T
            ids = teams.value_counts(['team_id','team_name']).reset_index().drop('count',axis=1)
            return ids

        def find_new_games(self,df_schedule:pd.DataFrame):
            
            current_schedule = df_schedule
            schedule_database =  pd.read_csv(self.db_schedule)
            current_schedule, schedule_database = current_schedule.align(schedule_database, join='inner', axis=1)
            deltas = determine_deltas(current_schedule, schedule_database,['game_id','game_id'])
            return deltas
        
        def get_database(self, database_name):
            database_path = f'{self.data_dir}{database_name}.csv'
            return pd.read_csv(database_path)

        def find_new_players(self):
            players = MLB.getPlayers()
            db_players = self.get_database(players)
            
        def write_to_csv(self,df:pd.DataFrame,file_name):
            df.to_csv(file_name,header=False,index=False,mode='a')
            text = f'''There were {df.shape[0]} records written to the database.
            The database is found at {self.db_schedule}'''
            return print(text)

        def write_to_sql(self,df:pd.DataFrame,tgtTbl:str):
            computerName = os.environ['COMPUTERNAME']
            database = 'tara'
            server = f'{computerName}\SQLEXPRESS'
            driver = 'driver=ODBC Driver 17 for SQL Server'
            tgtSchema = 'mlb'
            #engine = sqlalchemy.create_engine(f'mssql+pyodbc://{server}/{database}?{driver},echo=True')
            engine = sqlalchemy.create_engine('mssql+pyodbc://STARFIGHTER533\\SQLEXPRESS/tara?driver=ODBC+Driver+17+for+SQL+Server', echo=True)
            print(tgtTbl,engine)
            df.to_sql(tgtTbl,con=engine,if_exists='append',schema=tgtSchema,index=False)

        def update_schedule(self):
            schedule = MLB.getSchedule(MLB)
            schedule = schedule.loc[schedule["status"] in stuffs.lists.MLB.list_game_completion_types]
            new_games = self.find_new_games(schedule)

            if new_games.shape[0] != 0:
                self.write_to_csv(new_games,self.db_schedule)
                print(f'{new_games.shape[0]} records were written.')
            return new_games
        # -- Lists and constants

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

        
    

        
        
