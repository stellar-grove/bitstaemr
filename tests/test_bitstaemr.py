import sys; sys.path.append("../")
import pytest
import utils

# Constants


# Utils Class
# The class declared below houses the functions to test the functions in the utils file.

class TestUtils:
    def test_remove_characters(self):
        string = "X!@#$%^&*()_+=-1234567890[];'./,|}{:?><"
        new_string = utils.removeCharacters(string)
        assert new_string == 'X_1234567890'
        