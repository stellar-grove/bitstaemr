import pandas as pd
import re
import os
import numpy as np
import datetime

# ----------------------------------------------
# Package Wide Variables
# ----------------------------------------------


llaves = {}
computerName = os.environ["COMPUTERNAME"]
userName = os.environ["USERNAME"]
server={"name":f'{computerName}SQLEXPRESS'}



# ------  Begin Functions -------------------
# Removes special characters from text to give you a string of case
# sensative text that can serve as an ID.  To use on a DataFrame
# use the .apply function on the column and reference the removeCharacters
# function.
def removeCharacters(text):
    text = re.sub(r'\W+',"",text)
    return text

def addNumber(number):
    number = number + number
    return number

# ----------------------------------------------
# Returns working directories and keys
# ----------------------------------------------

def get_llaves(key_set=None):
    llaves = {}
    computerName = os.environ["COMPUTERNAME"]
    userName = os.environ["USERNAME"]
    server={"name":f'{computerName}\SQLEXPRESS'}

    workingDirectory={
                    f"bitstaemr":"C:/Users/{userName}/Stellar Grove/bitstaemr - Documents/",
                    f"ticondagrova":"C:/Users/{userName}/Stellar Grove/ticondagrova - Documents/"
                    }

# ------ Write dictionaries to llaves.                
    llaves["computerName"] = computerName
    llaves["server"] = server
    llaves["workingDirectory"] = workingDirectory
    
# ------ Take only keys that were given.
    if key_set==None: llaves = llaves
    if key_set!=None: llaves = llaves[key_set]
    
    return llaves

def findDateXDaysAgo(days):
    if type(days)==pd.DataFrame:
        days['Birthdate'] = datetime.now().date() - days['Age_Days']
        return days
    if type(days) in [float,int]:
        days = datetime.now().date() - days
        return days
    if type(days) in [str]:
        days = days.astype(float)
        days = datetime.now().date() - days
        return days
