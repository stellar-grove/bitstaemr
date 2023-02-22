import pandas as pd
import re

# Removes special characters from text to give you a string of case
# sensative text that can serve as an ID.  To use on a DataFrame
# use the .apply function on the column and reference the removeCharacters
# function.

def removeCharacters(text):
    text = re.sub(r'\W+',"",text)
    return text