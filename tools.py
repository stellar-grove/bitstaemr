# -*- coding: utf-8 -*-
"""
Pyton file for tools that we use in all of our programs.  The idea is to call this file with from dkUtils import tools
then run the functions we need run.
"""     

# Import packages needed to run the functions.
import pandas
import os
import numpy as np
from collections.abc import MutableMapping
import shutil
import math
from sqlalchemy import create_engine, text
import openai
import sys; sys.path.append("../")

# ------ Begin Constants ----- #
peep = os.environ["USERNAME"]
homeD = f'{os.environ["HOMEDRIVE"]}{os.environ["HOMEPATH"]}'.replace('\\','/')

SGWD = f'{homeD}/Stellar Grove/'
bitsWD = f'{SGWD}/bitstaemr - Documents/'
korpWD = f'{SGWD}/dkorpon - Documents/'
taraWD = f'{SGWD}/ticondagrova - Documents/'



# ------  Begin Functions -------------------


def concatList(lst, separator):
    lst = [str(x) for x in lst]
    result = separator.join(lst)
    return result

def calculate_trendline_angle(x:list, y:list):
    slope = (y[1] - y[0]) / (x[1] - x[0])
    angle_radians = math.atan(slope)
    angle_degrees = angle_radians * 180 / math.pi
    payload = [angle_degrees, angle_radians]
    return  payload

def flatten_dictionary(d: MutableMapping, parent_key: str = '', sep: str ='_') -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def listDictionaryToDataFrame(listDictionary:list):
    df_final = pandas.DataFrame()
    for dictionary in listDictionary:
        dictionary = flatten_dictionary(dictionary)
        df_interim = pandas.DataFrame().from_dict(dictionary,orient="index").T
        df_final = pandas.concat([df_final,df_interim],ignore_index = True)
    return df_final

def get_stones(key_set:str=None):
    stone_dict = {}
    stones = os.getenv('StellarGrove').split(';')
    for entry in os.getenv('StellarGrove').split(';'):

        if ':' in entry:
            if 'stellar' not in entry:
                key, value = entry.split(':')
                stone_dict[key] = value
        
    if key_set != None:
        stone_dict = stone_dict[key_set]    
    return stone_dict

def get_llaves(key_set=None):
    llaves = {}
    computerName = os.environ["COMPUTERNAME"]
    userName = os.environ["USERNAME"]
    server={"name":f'{computerName}\\SQLEXPRESS'}
    workingDirectory={
                    "bitstaemr":"C:/Users/DanielKorpon/Stellar Grove/bitstaemr - Documents/"
                    ,"dkorpon":"C:/Users/DanielKorpon/Stellar Grove/dkorpon - Documents/"
                    ,"ticondagrova":"C:/Users/DanielKorpon/Stellar Grove/ticondagrova - Documents/"
                    }

    dataBases = {'DB':{"serverName":server,
                       "driver":{"sql":'driver=SQL Server Native Client 11.0'}},
                }
    udemy = "Development/Udemy/"
    dataSources={
                    "Udemy":{
                            "computerVision":f"{udemy}Computer-Vision-with-Python/DATA/",
                            "NLP":{ "root":f"{udemy}NLP/",
                                    "TextFiles":f"{udemy}NLP/TextFiles/",
                                    "movieReviews":f"{udemy}NLP/moviereviews/",
                                }
                    }
                }

    quandl = {"stuff":{"llave":'XWQXAFtixVtS7MX2yai6',}
            }

    
    columns = ['TxId','TxSource','TxDate','PostDate','TxDescription','TxCategory','TxSubCategory','TxFriendlyName'
                ,'TxAmount','DebitAmount','CreditAmount','isDebit','isCredit','SourceType','TxNotes']
    
    cream = {
            "tgtSchema":"cream",
            "tgtTbl":"AllTransactions",
            "dlFolder":f"C:/Users/{userName}/Downloads/",
            "dbFolder":f"C:/Users/{userName}/Stellar Grove/dkorpon - Documents/data/mkt/cream/",
            "archiveFolder":f"C:/Users/{userName}/Stellar Grove/dkorpon - Documents/data/archive/",
            "Amazon":{
                "dbFileName":"Amazon",
                "dfFileSearchRoot":"Report",
                "fillColumns":{
                        "TxSource":"Amazon",
                        'PostDate':np.nan,
                        "TxSubCategory":np.nan,
                        "TxFriendlyName":np.nan,
                        "SourceType":"OnlineRetailer",
                        "TxNotes":np.nan,
                        }
            },
            "Barclays":{
                "dbFileName":"Barclays",
                "dlFileSearchRoot":"trans",
                "fillColumns":{ 
                        "TxSource":"Barclays",
                        'PostDate':np.nan,
                        "TxSubCategory":np.nan,
                        "TxFriendlyName":np.nan,
                        "SourceType":"OnlineSavings",
                        "TxNotes":np.nan,
                        }

            },
            "BankOfAmerica":{
                "dlFileSearchRoot":"Checking",
                "dbFileName":"BankOfAmerica",
                "fillColumns":{"TxSource":"BankOfAmerica",
                               "PostDate":np.nan,
                               "TxSubCategory":np.nan,
                               "TxFriendlyName":np.nan,
                               "SourceType":"Checking",
                               "TxNotes":np.nan,}
            },
            "ChaseSW":{
                "dlFileSearchRoot":"Chase",
                "dbFileName":"ChaseSW",
                "TxDateFormat":r'%m/%dd/%YYYY',
                "fillColumns":{"TxSource":"ChaseSWVisa",
                               "PostDate":np.nan,
                               "TxSubCategory":np.nan,
                               "TxFriendlyName":np.nan,
                               "SourceType":"Credit",
                               "TxNotes":np.nan,}
            },
            "CapitalOneRewards":{
                "dlFileSearchRoot":"Rewards",
                "dbFileName":"CapitalOneRewards",
                "fillColumns":{"TxSource":"CapitalOne",
                               "TxSubCategory":np.nan,
                               "TxFriendlyName":np.nan,
                               "SourceType":"Credit",
                               "TxNotes":np.nan,}},
            "WellsFargo":{
                "dlFolder":"C:/Users/DanielKorpon/Downloads/",
                "dbFileName":"WellsFargo",
                "dlFileSearchRoot":"Checking1",
                "fillColumns":{"TxSource":"WellsFargo",
                               "PostDate":np.nan,
                               "TxSubCategory":np.nan,
                               "TxFriendlyName":np.nan,
                               "SourceType":"Checking",
                               "TxNotes":np.nan,}
            },
            "Mint":{
                "dbFileName":"Mint",
                "dlFileSearchRoot":"transfers",
                "fillColumns":{}
            },
            "AllTransactions":{
                "tgtColumns":columns,
                "fillColumns":{},
                "dbFileName":"AllTransactions.csv"
                }
                
            }

# ------ Write dictionaries to llaves.                
    llaves["computerName"] = computerName
    llaves["server"] = server
    llaves["stones"] = 'stones'
    llaves["workingDirectory"] = workingDirectory
    llaves["cream"] = cream
    llaves["dataSources"] = dataSources
    llaves["quandl"] = quandl

# ------ Take only keys that were given.
    if key_set==None: llaves = llaves
    if key_set!=None: llaves = llaves[key_set]
    
    return llaves

def get_file(file_path,header=0):
    data_frame = pandas.read_csv(file_path,header)
    return data_frame

def move_file(source_path, destination_path):
    try:
        shutil.move(source_path, destination_path)
        print(f"File moved successfully from {source_path} to {destination_path}.")
    except Exception as e:
        print(e)

def print_keys(dict:dict):
    return dict.keys()



# ------ NLP FUNCTIONS ---------

class NaturalLangaugeProcessing(object):

    def __init__(self)-> None:
        self.config = {}

    def dataframe_to_database(df, table_name):
        """Convert a pandas dataframe to a database.
            Args:
                df (dataframe): pandas.DataFrame which is to be converted to a database
                table_name (string): Name of the table within the database
            Returns:
                engine: SQLAlchemy engine object
        """
        engine = create_engine(f'sqlite:///:memory:', echo=False)
        df.to_sql(name=table_name, con=engine, index=False)
        return engine

    def handle_OpenAI_response(response):
        """Handles the response from OpenAI.

        Args:
            response (openAi response): Response json from OpenAI

        Returns:
            string: Proposed SQL query
        """
        query = response["choices"][0]["text"]
        if query.startswith(" "):
            query = "Select"+ query
        return query

    def execute_query(engine, query):
        """Execute a query on a database.

        Args:
            engine (SQLAlchemy engine object): database engine
            query (string): SQL query

        Returns:
            list: List of tuples containing the result of the query
        """
        with engine.connect() as conn:
            result = conn.execute(text(query))
            return result.fetchall()
        
    def create_table_definition_prompt(df, table_name):
        """This function creates a prompt for the OpenAI API to generate SQL queries.

        Args:
            df (dataframe): pandas.DataFrame object to automtically extract the table columns
            table_name (string): Name of the table within the database

            Returns: string containing the prompt for OpenAI
        """
        
        prompt = '''### sqlite table, with its properties:
    #
    # {}({})
    #
    '''.format(table_name, ",".join(str(x) for x in df.columns))
        
        return prompt

    def user_query_input():
        """Ask the user what they want to know about the data.

        Returns:
            string: User input
        """
        user_input = input("Tell OpenAi what you want to know about the data: ")
        return user_input

    def combine_prompts(fixed_sql_prompt, user_query):
        """Combine the fixed SQL prompt with the user query.

        Args:
            fixed_sql_prompt (string): Fixed SQL prompt
            user_query (string): User query

        Returns:
            string: Combined prompt
        """
        final_user_input = f"### A query to answer: {user_query}\nSELECT"
        return fixed_sql_prompt + final_user_input

    def send_to_openai(prompt):
        """Send the prompt to OpenAI

        Args:
            prompt (string): Prompt to send to OpenAI

        Returns:
            string: Response from OpenAI
        """
        response = openai.Completion.create(
            engine="code-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=150,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            stop=["#", ";"]
        )
        return response
