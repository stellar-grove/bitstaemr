import sys; sys.path.append("../")
import pytest
import simulators
import data.biostats as test_data
import random


class TestBioStats:
    bio = simulators.biostats()
    def test_createCohort(self):
        # This test simply looks to make sure that the
        # dimensions of the data are correct.  Phase two
        # can have things to ensure the data is good.
        
        cohortSize = (10,14)
        df_cohort = self.bio.createCohort(cohortSize[0])
        rows = df_cohort.shape[0]
        columns = df_cohort.shape[1]
        condition_one = (rows == cohortSize[0])
        condition_two = (columns == cohortSize[1])
        assert (condition_one, condition_two) == (True, True)

        