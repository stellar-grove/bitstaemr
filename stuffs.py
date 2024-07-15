import os
import pyodbc as db
import sqlalchemy
from dotenv import load_dotenv


class folders(object):
    # ------ Begin Folders ------ #
    DATA_FOLDER_BITS = "C:/stellar-grove/bitstaemr/data/"
    DATA_FOLDER_TARA = "C:/stellar-grove/tara/data/"
    DATA_FOLDER_SONGROVE = f"C:/stellar-grove/tara/SongroveBotanicals/data/"
       # ------ Begin Constants ----- #
    peep = os.environ["USERNAME"]
    robot = os.environ["COMPUTERNAME"]
    homeD = f'{os.environ["HOMEDRIVE"]}{os.environ["HOMEPATH"]}'.replace('\\','/')
    SGWD = f'{homeD}/Stellar Grove/'
    bitsWD = f'{SGWD}bitstaemr - Documents/'
    taraWD = f'{SGWD}ticondagrova - Documents/'
    download = f'C:/Users/{peep}/Downloads/'
    kaggleWD = f'{taraWD}Kaggle/'
    financeWD = f'{taraWD}data/Financial/'
    server = f'{robot}\SQLEXPRESS'
    sniffnet = 'sniffnet.database.windows.net'
    cannabis_data = f'{taraWD}data/Cannabis/'
    udemyWD = f'{bitsWD}Development/Udemy/'
    DB_tara = {'servername': server,
            'database': 'tara',
            'driver': 'driver=SQL Server Native Client 11.0'
            ,'tgtSchema':'pll'
            ,'tgtTbl':'player_stats'
            ,'fileRoot':'C:/stellar-grove/tara/data/pll/'}

    dbAzureTARA = {'servername': sniffnet,
                'database': 'tara',
                'driver': 'driver=SQL Server Native Client 11.0'
                    }

class constants(object):
    chemistry_constants = {
    "avagadro_constant":602214076000000000000000,
    "gas_constant":8.31446261815324,
    "boltzman_constant":1.389649e-23
}

class lists(object):
    class MLB(object):
        list_meta = ['awards',
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
                      'statTypes'
                      ,'windDirection'
                      ]
        list_game_completion_types = [
                        'Final',
                        'Completed Early',
                        'Final: Tied',
                        'Completed Early: Rain',
                        'Cancelled',
                        'Postponed',
                        'Completed Early: Wet Grounds',
                        'Suspended: Rain',
                        'Completed Early: Power',
                        'Game Over',
                        'Completed Early: Wind',
                        

        ]
    
class urls(object):
    
    SIM_MKT_SEGMENTATION = "https://archive.ics.uci.edu/ml/machine-learning-databases/00352/Online%20Retail.xlsx"
    DOSIA_MD_DATA_DASHBOARD = "https://cannabis.maryland.gov/Pages/Data-Dashboard.aspx"
    PLL_PLAYER_STATS = "https://stats.premierlacrosseleague.com/player-table"
    
class CREAM(object):
    EQUITY = 'EQUITY'
    ETF = 'ETF'

class connections(object):
    computerName = os.environ['COMPUTERNAME']
    MLB_SQL_SERVER_EXPRESS = {
        'server': f'{computerName}\SQLEXPRESS',
        'database':'tara',
        'driver':'driver=SQL Server Native Client 11.0',
        'schema':'mlb',
        'trusted':'trusted_connection=yes'
        }
    
    MLB_CONNECTION = (sqlalchemy.create_engine(
        'mssql+pyodbc://' + MLB_SQL_SERVER_EXPRESS['server'] +
        '/' + MLB_SQL_SERVER_EXPRESS['database'] + 
        "?" + MLB_SQL_SERVER_EXPRESS['driver'] , echo=True)
    )

class coding(object):
    webscraping = """
import requests  
from bs4 import BeautifulSoup  
  
'''  
URL of the archive web-page which provides link to  
all video lectures. It would have been tiring to  
download each video manually.  
In this example, we first crawl the webpage to extract  
all the links and then download videos.  
'''
  
# specify the URL of the archive here  
archive_url = "http://www-personal.umich.edu/~csev/books/py4inf/media/"
  
def get_video_links():  
      
    # create response object  
    r = requests.get(archive_url)  
      
    # create beautiful-soup object  
    soup = BeautifulSoup(r.content,'html5lib')  
      
    # find all links on web-page  
    links = soup.findAll('a')  
  
    # filter the link sending with .mp4  
    video_links = [archive_url + link['href'] for link in links if link['href'].endswith('mp4')]  
  
    return video_links  
  
  
def download_video_series(video_links):  
  
    for link in video_links:  
  
        '''iterate through all links in video_links  
        and download them one by one'''
          
        # obtain filename by splitting url and getting  
        # last string  
        file_name = link.split('/')[-1]  
  
        print( "Downloading file:%s"%file_name)  
          
        # create response object  
        r = requests.get(link, stream = True)  
          
        # download started  
        with open(file_name, 'wb') as f:  
            for chunk in r.iter_content(chunk_size = 1024*1024):  
                if chunk:  
                    f.write(chunk)  
          
        print( '%s downloaded!\n'%file_name ) 
  
    print ("All videos downloaded!") 
    return
  
  
if __name__ == "__main__":  
  
    # getting all video links  
    video_links = get_video_links()  
  
    # download all videos  
    download_video_series(video_links) 



"""


    FineTuningEmbeddingModels = """

Creating a Pipeline for Generating Synthetic Data for Fine-Tuning Custom Embedding Models: 👀

Step 1: Create a Knowledge Base: Start with preparing your domain specific knowledge base, such as PDFs or other documents containing information. Convert the content of these documents into a plain text format.

Step 2: Chunk the Data: Divide your text data into manageable chunks of approximately 256 tokens each (chunk size used in RAG later).

Step 3: Generate Questions Using LLM: Use a Language Model (LLM) to generate K questions for each chunk of text. The questions should be answerable based on the content within the chunk. Example prompt: "Generate five questions that can be answered using the following text: [insert chunk here]."

Step 4: Optionally Generate Hard Negative Examples: Create hard negative examples by generating questions that are similar to the correct questions but have answers that are incorrect or misleading. Alternatively, use random other samples from the batch as negative examples during training (in-batch negatives).

Step 5: Deduplicate and Filter Pairs: Remove “duplicate” question-context pairs to ensure uniqueness. Use the LLM to judge and filter out lower-quality pairs by defining custom rubrics for quality assessment.

Step 6: Fine-Tune Embedding Models: Use the prepared data to fine-tune your embedding models with Sentence Transformers 3.0

"""

    baseball = """


# --------------------------------------------------------------

dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
#sys.path.append(dk_repo)
import statsapi as mlb
from bitstaemr.dataUtils import MLB
import pandas as pd
MLB = MLB()
start_date = '2024-06-28'
end_date = '2024-06-28'
games = MLB.getSchedule(start_date, end_date)
game = games.iloc[0,0]


bs = mlb.boxscore_data(game)



for i in range(0,10):
    game = games.iloc[i,0]
#    box_score = MLB.getBoxScore(game)
    print(mlb.linescore(game))

pd.DataFrame().from_dict(bs['homePitchers'],orient='index')
df_home_pitchers = pd.DataFrame().from_dict(bs['homePitchers'])
df_away_pitchers = pd.DataFrame().from_dict(bs['awayPitchers'])
df_away_batters = pd.DataFrame().from_dict(bs['awayBatters'])
df_home_batters = pd.DataFrame().from_dict(bs['homeBatters'])

df_away_batting_totals = pd.DataFrame().from_dict(bs['awayBattingTotals'],orient='index')
df_home_batting_totals = pd.DataFrame().from_dict(bs['homeBattingTotals'],orient='index')

df_home = pd.DataFrame().from_dict(bs['home']['batters'])



df_players = pd.DataFrame().from_dict(bs['playerInfo']).T
df_teams = pd.DataFrame().from_dict(bs['teamInfo']).T.reset_index()

df_game_box_info = pd.DataFrame().from_dict(bs['gameBoxInfo'])




gh = mlb.game_highlight_data(game)


for g in gh:
    print(g.keys())


# Make the dataframe to have the description be blank if it's not included
# in the dictionaruy



dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
#sys.path.append(dk_repo)
from bitstaemr import (dataUtils as dutils,
                       stuffs as stuffs,
                       tools as tools,
                       simulators as sims)
import pandas as pd
import statsapi as mlb

MLB = dutils.MLB()
infra = MLB.infrastructure()

schedule =  MLB.getSchedule()
teams = infra.infrastructure_Teams(schedule)

teams['year'] = 2024


file_location = f'{infra.data_dir}teams.csv'
teams.to_csv(file_location,header=True,index=False)


standings = mlb.standings_data()
LEAGUES = ['American', 'National']
DIVISIONS = ['East', 'West', 'Central']

div_name = f'{LEAGUES[0]} League {DIVISIONS[0]}'

division = [200,201,202,203,204,205]

d = 203
division = standings[d]['div_name']
teams = standings[d]['teams']

df_teams = pd.DataFrame()
team_num = len(teams)
for i in range(team_num):
    df_team = pd.DataFrame.from_dict(teams[i],orient='index').T
    df_teams = pd.concat([df_teams,df_team],axis=0)
    df_teams['league'] = 'American'
    df_teams['division'] = division
    
df = MLB.getRoster(103)
pitcherschers = df[df['POSITION']=='P']


import json
import numpy

player_stats = mlb.lookup_player('')
df_players = pd.DataFrame()
for i in range(len(player_stats)):
    player = player_stats[i]
    new = pd.json_normalize(player)
    if new.shape[1] == 19:
        new['nickName'] = numpy.nan
    df_players = pd.concat([df_players,new],axis=0)
    print(new.shape)
    #ls = player.pop('primaryPosition')
    #df_player = pd.DataFrame().from_dict(player['id']).T
    #print(player)
    

file_location = f'{infra.data_dir}players.csv'
df_players.to_csv(file_location,header=True,index=False)



"""

    baseball_scraping = """


# --------------------------------------------------------------

dk_repo = "C:/repo/bitstaemr";sg_repo = "C:/stellar-grove"
import sys;sys.path.append(sg_repo)
#sys.path.append(dk_repo)


from bs4 import BeautifulSoup as Soup

import requests as req



url = 'https://www.baseball-reference.com/players/a/aaronha01.shtml#batting_advanced'
pl = req.get(url)
content = pl.content
soup = Soup(content, 'html.parser')
table = soup.find('table')
table_data = []
for row in table.find_all('tr'):
    cells = row.find_all(['th', 'td'])
    row_data = [cell.text.strip() for cell in cells]
    table_data.append(row_data)
print (table_data)

import pandas as pd

pd.DataFrame(table_data)


import requests

def get_game_events(game_pk):
    url = "https://baseballsavant.mlb.com/statcast_search/csv"
    params = {
        "type": "game",
        "game_pk": game_pk
    }

    response = requests.get(url, params=params)

    if response.status_code == 200:
        data = response.content
        print(data)
    else:
        print("Request failed with status code:", response.status_code)

# Example usage:
game_pk = 634367  # Replace with the game_pk you want to retrieve data for
get_game_events(game)
"""

class web_scraping(object):
    # Get the path to the directory this file is in
    BASEDIR = os.path.abspath(os.path.dirname(__file__))
    # Load environment variables
    load_dotenv(os.path.join(BASEDIR, 'config.env'))
    CHROMEDRIVER_PATH = "C:/repo/bitstaemr/chromedriver/chromedriver.exe"
    ARGUMENTS = "--headless"
    # URLs
    BOOKSTOSCRAPE_URL = 'https://books.toscrape.com/'
    GOLDBUGS_URL = 'https://www.thegoldbugs.com/'
    IMGUR_UPLOAD_URL = 'https://imgur.com/upload'
    JQUERYUI_URL = 'https://jqueryui.com/'
    PYTHON_URL = 'https://www.python.org/'
    PYTHON_DOWNLOAD_URL = 'https://www.python.org/downloads/'
    QUOTESTOSCRAPE_URL = 'https://quotes.toscrape.com/'
    SELENIUM_DOCS_SEARCH_URL = 'https://selenium-python.readthedocs.io/search.html'
    SELENIUM_URL = 'https://www.selenium.dev'
    WIKIPEDIA_URL = 'https://en.wikipedia.org/wiki/Main_Page'


    # Constants
    WAIT_TIME = 10  # seconds