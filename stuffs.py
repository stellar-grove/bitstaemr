import os


class folders(object):
    # ------ Begin Folders ------ #
    DATA_FOLDER = "C:/stellar-grove/tara/data/"
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
    server = f'{robot}\SQLEXPRESS'
    sniffnet = 'sniffnet.database.windows.net'
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






