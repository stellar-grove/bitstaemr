import sys; sys.path.append("/..")
import pandas as pd
import re
import os
import numpy as np
import datetime
import string
import secrets
from nltk.corpus import words
import winsound
import time
import numpy as np
from bitstaemr.data.lists import morse_dict
# ----------------------------------------------
# Package Wide Variables
# ----------------------------------------------


llaves = {}
computerName = os.environ["COMPUTERNAME"]
userName = os.environ["USERNAME"]
server={"name":f'{computerName}SQLEXPRESS'}
alphabet = string.ascii_letters + string.digits
word_choices = words.words()
short_words = [word.lower() for word in words.words() if len(word) <= 5]
#morse_dict = lists.morse_dict

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

class stuffs(object):
    #alphabet = string.ascii_letters + string.digits
    morse_dict = morse_dict

    def suggestPassword(self, numberOfChars:int):
        password = "".join(secrets.choice(alphabet) for i in range(numberOfChars))
        return password

    def generateCipher(numberOfChars:int):
        password = "".join(secrets.choice(alphabet) for i in range(0,numberOfChars))
        return password

    def suggestXKCDPassword(numberOfWords:int,wordLength=None):
        terds = word_choices
        if wordLength is not None:
            password = ''.join(secrets.choice(terds) for i in range(numberOfWords))
        else:
            terds = short_words
            password = ''.join(secrets.choice(terds) for i in range(numberOfWords))        
        return password

    def generateOTPWord(self, word):
        OTP = self.generateCipher(len(word)).upper()
        return OTP
        
    def generateOTP(self,sentence):
        newLine = []
        for word in sentence.split('\n'):
            newLine.append(self.generateOTPWord(word))
        return newLine

    def translate_sentence_to_morse(self,sentence):
        translated_sentence = []
        for line in sentence.splitlines():
            list_line = []
            for word in line.split(" "):
                print(f"{word}: ", self.translate_to_morse(word))
                list_line.append(self.translate_to_morse(word))
            translated_sentence.append(list_line)
        #print(morse_dict)
        return translated_sentence

    def translate_to_morse(self,char_list):
        morse_list = []
        for char in char_list:
            if char.lower() in morse_dict:
                morse_list.append(morse_dict[char.lower()])
            else:
                morse_list.append('')
        return morse_list

    def play_morse_code(string):
        for char in string:
            if char.lower() in ("/n"):
                continue
            if char.lower() in morse_dict.keys():
                char = char.lower()
                morse = morse_dict[char]
                for signal in morse:
                    if signal == '.':
                        winsound.Beep(800, 300)  # Play a 800 Hz beep for 100 ms
                    elif signal == '-':
                        winsound.Beep(800, 500)  # Play a 800 Hz beep for 300 ms
                    time.sleep(0.1)
            else:
                time.sleep(0.3)  # Pause between words

    def convertLatLong(degrees, minutes, seconds):
        convertedLatLong = np.sum(degrees, minutes / 60, seconds / 3600)
        return convertedLatLong