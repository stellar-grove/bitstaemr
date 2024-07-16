import sys; sys.path.append("../../")
import tools
import bitstaemr.tests.data.bitstaemr as test_data

def check_dictionary_keys(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert key in dictionary_two.keys()
    for key in dictionary_two.keys():
        assert key in dictionary_one.keys()
def check_dictionary_values(dictionary_one:dict, dictionary_two:dict):
    for key in dictionary_one.keys():
        assert dictionary_one[key] == dictionary_two[key]


class TestTools:

    def test_get_stones(self):
        stones = tools.get_stones('tester_bester')
        test_stones = test_data.TEST_STONES['tester_bester']
        assert stones == test_stones