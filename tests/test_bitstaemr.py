import sys; sys.path.append("../")
import pytest
import bitstaemr.monkies as monkies
import bitstaemr.tests.data.bitstaemr as test_data

def check_dictionary_keys(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert key in dictionary_two.keys()
    for key in dictionary_two.keys():
        assert key in dictionary_one.keys()
def check_dictionary_values(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert dictionary_one[key] == dictionary_two[key]


class TestMonkies:

    def test_constants(self):
        assert test_data.TEST_MONKIES_CONSTANTS_SEED == monkies.constants.seed
