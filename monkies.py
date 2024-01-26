# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 22:07:56 2023

@author: DanielKorpon
"""

repo = "C:\\repo\\bitstaemr"
import sys;sys.path.append(repo)
import secrets
import string
from nltk.corpus import words
from tara import monkies
from tara.data import dictionaries
import winsound
import time
import numpy as np

alphabet = string.ascii_letters + string.digits
morse_dict = dictionaries.morse_dict

def suggestPassword(numberOfChars:int):
    password = "".join(secrets.choice(alphabet) for i in range(numberOfChars))
    return password

def generateCipher(numberOfChars:int):
    password = "".join(secrets.choice(alphabet) for i in range(numberOfChars))
    return password

def suggestXKCDPassword(numberOfWords:int,wordLength=None):
    terds = words.words()
    if wordLength is not None:
        password = ''.join(secrets.choice(terds) for i in range(numberOfWords))
    else:
        terds = [word.lower() for word in words.words() if len(word) <= 5]
        password = ''.join(secrets.choice(terds) for i in range(numberOfWords))        
    return password

def generateOTPWord(word):
    OTP = generateCipher(len(word)).upper()
    return OTP
    
def generateOTP(sentence):
    newLine = []
    for word in sentence.split('\n'):
        newLine.append(monkies.generateOTPWord(word))
    return newLine

def translate_sentence_to_morse(sentence):
    translated_sentence = []
    for line in sentence.splitlines():
        list_line = []
        for word in line.split(" "):
            print(f"{word}: ", translate_to_morse(word))

            list_line.append(translate_to_morse(word))
        translated_sentence.append(list_line)
    return translated_sentence

def translate_to_morse(char_list):
    morse_list = []
    for char in char_list:
        if char.upper() in morse_dict:
            morse_list.append(morse_dict[char.upper()])
        else:
            morse_list.append('')
    return morse_list

def play_morse_code(string):
    for char in string:
        if char in ("/n"):
            continue
        if char.upper() in morse_dict:
            morse = morse_dict[char.upper()]
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