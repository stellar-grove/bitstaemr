dk_repo = "C:/repo/bitstaemr/bitstaemr"
sg_repo = "C:/stellar-grove/bitstaemr"
import sys;sys.path.append(dk_repo);sys.path.append(sg_repo)
import pandas as pd
#import dkUtils.tools as tools
import tara.distributions as dists
from tara.distributions import DaCountDeMonteCarlo as dcmc
import numpy as np
from num2words import num2words
import datetime as dt



class PERT(object):
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
    
    def get_working_directories(self):
        wd = tools.get_llaves("workingDirectory")
        return wd

    def set_distribution(self, distribution_number = 1):
        dict_distribution = {
            1:"TSP",
            2:"PERT",
            3:"Herrerias"
        }
        self.config["distribution"] = dict_distribution[distribution_number]
    
    def generate_data(self):
        df = pd.DataFrame()
        return df

class biostats(object):
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
        self.patient = {}
        
    def clearPatient(self):
        self.patient = {}
    
    def createPatient(self):
        
        self.generateGender()
        self.generateAge()
        self.generateBP()
        self.generateCholesterol()
        self.generateBMI()
        self.generatePhysicalActivity()
        self.generateDiabetes()
        self.generatePersonalityType()
        self.generatePSS()
        self.generateSWLS()        
        self.generateSTAI()
        return self.patient

    def createCohort(self,cohortSize:int):
        dfCohort = pd.DataFrame()
        print("OK")
        for i in range(cohortSize):
            df = pd.DataFrame().from_dict(self.createPatient(),orient="index")
            df = df[:-4].T
            dfCohort = pd.concat([dfCohort,df])
        return dfCohort
    
    def generatePersonalityType(self):
    # 0 Miserable, 1 Debbie Downer, 2 Even Steven, 3 Optimistic Oliver, 4 Richard Fucking Simons
    # Generate a random number between 0 and 4 to designate the personality type.
        pt = int(np.random.randint(5))
        self.patient["PersonalityType"] = pt

    def generateGender(self):
        genderValue = np.round(np.random.rand(),0)
        self.patient["genderValue"] = genderValue
        if genderValue == 1: 
            self.patient["genderName"] = "Female"
        else: self.patient["genderName"] = "Male"
    
    def generateAge(self):
        sample = np.random.rand()
        mean = 42.5
        std = 14.17
        age = dcmc.sampleFromNormal(dcmc,mean = mean, std = std, sample = sample)
        ageDays = age * 365.25
        birthDate = dt.datetime.now() - pd.to_timedelta(ageDays,"D")
        age = float(np.round(age,2))
        self.patient["age"] = age
        self.patient["birthDate"] = birthDate.strftime("%Y-%m-%d")

    def generateBP(self):
        gender = self.patient["genderValue"]
        if gender == 1: bp = dcmc.sampleFromNormal(self,mean=120,std=10,sample=np.random.rand())
        else: bp = dcmc.sampleFromNormal(self,mean=110,std=20,sample=np.random.rand())
        self.patient["systolicBP"] = bp
        if gender == 1: bp = dcmc.sampleFromNormal(self,mean=120,std=10,sample=np.random.rand())
        else: bp = dcmc.sampleFromNormal(self,mean=75,std=20,sample=np.random.rand())
        self.patient["diastolicBP"] = bp

    def generateCholesterol(self):
        gender = self.patient["genderValue"]
        age = self.patient["age"]
        # Case Men Over 40
        if gender == 1 and age > 40:
            sample = np.random.rand()
            mean = 40; std = 10
            LDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 60; std = 20
            HDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 50; std = 10
            TRYG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Men Under 40
        if gender == 1 and age <= 40:
            sample = np.random.rand()
            mean = 30; std = 10
            LDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 70; std = 20
            HDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 35; std = 10
            TRYG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Women Over 40
        if gender == 0 and age > 40:
            sample = np.random.rand()
            mean = 20; std = 5
            LDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 80; std = 20
            HDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 60; std = 10
            TRYG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Women Under 40
        if gender == 0 and age <= 40:
            sample = np.random.rand()
            mean = 20; std = 5
            LDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 70; std = 20
            HDL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 30; std = 10
            TRYG = dcmc.sampleFromNormal(dcmc, mean, std, sample)

        self.patient["HDL"] = HDL
        self.patient["LDL"] = LDL
        self.patient["TRY"] = TRYG

    def generateBMI(self):
        gender = self.patient["genderValue"]
        age = self.patient["age"]
        sample = np.random.rand()
        # Case Men Over 40
        if gender == 1 and age > 40:
            mean = 40; std = 10
            BMI = dcmc.sampleFromNormal(dcmc,mean, std, sample)
        # Case Men Under 40
        if gender == 1 and age <= 40:
            mean = 30; std = 10
            BMI = dcmc.sampleFromNormal(dcmc,mean, std, sample)
        # Case Women Over 40
        if gender == 0 and age > 40:
            mean = 20; std = 5
            BMI = dcmc.sampleFromNormal(dcmc,mean, std, sample)
        # Case Women Under 40
        if gender == 0 and age <= 40:
            mean = 20; std = 5
            BMI = dcmc.sampleFromNormal(dcmc,mean, std, sample)
        
        self.patient["BMI"] = BMI
    
    def generatePhysicalActivity(self):
        activityLevel = np.floor(np.random.rand() * 7)
        self.patient["PhysicalActivity"] = activityLevel
    
    def generateSmokingStatus(self):
        smokingStatus = np.floor(np.random.rand())
        self.patient["PhysicalActivity"] = smokingStatus
    
    def generateFamilyHistory(self):
        familyHistory = np.floor(np.random.rand())
        self.patient["FamilyHistory"] = familyHistory
    
    def generateDiabetes(self):
        gender = self.patient["genderValue"]
        age = self.patient["age"]
        # Case Men Over 40
        if gender == 1 and age > 40:
            sample = np.random.rand()
            mean = 40; std = 10
            HBA1C = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 60; std = 20
            ORAL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 50; std = 10
            FPG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Men Under 40
        if gender == 1 and age <= 40:
            sample = np.random.rand()
            mean = 30; std = 10
            HBA1C = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 70; std = 20
            ORAL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 35; std = 10
            FPG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Women Over 40
        if gender == 0 and age > 40:
            sample = np.random.rand()
            mean = 20; std = 5
            HBA1C = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 80; std = 20
            ORAL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 60; std = 10
            FPG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        # Case Women Under 40
        if gender == 0 and age <= 40:
            sample = np.random.rand()
            mean = 20; std = 5
            HBA1C = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 70; std = 20
            ORAL = dcmc.sampleFromNormal(dcmc,mean, std, sample)
            mean = 30; std = 10
            FPG = dcmc.sampleFromNormal(dcmc, mean, std, sample)
        
        self.patient["HBA1C"] = HBA1C
        self.patient["ORAL"] = ORAL
        self.patient["FPG"] = FPG

    def generateMedicalHistory(self):
        medicationUse = np.floor(np.random.rand())
        self.patient["MedicationUse"] = medicationUse

    def generateSWLS(self,individualReponses = False):
        personalityType = self.patient["PersonalityType"]
        # Create the bounds of how high the personality can score on each of the 5 questions
        listBounds = [
                        [0,3], # Miserable
                        [1,4], # Debbie Downer
                        [2,5], # Even Steven
                        [3,6], # Optimistic Oliver
                        [4,7]  # Richard Fucking Simons
                      ]
        responseBounds = listBounds[personalityType]
        low = responseBounds[0]
        high = responseBounds[1]

        if individualReponses:
            print("fire")
            responseList = dcmc.createUniformData(dcmc,a = low, b = high, sampleSize = 5)
            responseList = [int(response) for response in responseList]
            dictResponse = {}
            for index, response in enumerate(responseList):
                questionNumber = num2words(index + 1)
                dictResponse[f"{questionNumber}"] = response
            dictResponse["total"] = sum(responseList)
            self.patient["SWLS"] = dictResponse
        else:
            SWLS = int(dcmc.createNormalData(dcmc,mean = 20, std = (14/3), size = 1)[0])
            dictResponse = {"total":SWLS}
            self.patient["SWLS"] = dictResponse

    def generatePSS(self, individualRespones = False):
        personalityType = self.patient["PersonalityType"]
        # Create the bounds of how high the personality can score on each of the 5 questions
        listBounds = [
                        [0,1], # Miserable
                        [1,2], # Debbie Downer
                        [2,3], # Even Steven
                        [3,4], # Optimistic Oliver
                        [4,4]  # Richard Fucking Simons
                      ]
        responseBounds = listBounds[personalityType]
        low = responseBounds[0]
        high = responseBounds[1]
        if individualRespones:
            responseList = dcmc.createUniformData(dcmc,a = low, b = high, sampleSize = 10)
            responseList = [int(response) for response in responseList]
            dictResponse = {}
            for index, response in enumerate(responseList):
                questionNumber = num2words(index + 1)
                dictResponse[f"{questionNumber}"] = response
                self.patient["PSS"] = dictResponse
        else:
            PSS = int(dcmc.createNormalData(dcmc,mean = 20, std = (40/3), size = 1)[0])
            dictResponse = {"total":PSS}
            self.patient["PSS"] = dictResponse
        
    def generateSTAI(self, individualResponses = False):
        personalityType = self.patient["PersonalityType"]
        # Create the bounds of how high the personality can score on each of the 5 questions
        listBounds = [
                        [0.2], # Miserable
                        [0.4], # Debbie Downer
                        [0.5], # Even Steven
                        [0.6], # Optimistic Oliver
                        [0.8]  # Richard Fucking Simons
                      ]
        responseBounds = listBounds[personalityType]
        p = responseBounds[0]
        if individualResponses:
            responseList = dcmc.createBernoulliData(dcmc,p,0,20)
            responseList = [int(response) for response in responseList]
            dictResponse = {}
            for index, response in enumerate(responseList):
                questionNumber = num2words(index + 1)
                dictResponse[f"{questionNumber}"] = response
            dictResponse["total"] = sum(responseList)
            self.patient["STAIS"] = dictResponse

            responseList = dcmc.createBernoulliData(dcmc,p,0,20)
            responseList = [int(response) for response in responseList]
            dictResponse = {}
            for index, response in enumerate(responseList):
                questionNumber = num2words(index + 1)
                dictResponse[f"{questionNumber}"] = response
            dictResponse["total"] = sum(responseList)
            self.patient["STAIT"] = dictResponse
        else:
            STAIS = int(dcmc.createNormalData(dcmc,mean = 10, std = (20/3), size = 1)[0])
            STAIS = min(max(STAIS,0),20)
            STAIT = int(dcmc.createNormalData(dcmc,mean = 10, std = (20/3), size = 1)[0])
            self.patient["STAIS"] = {"total":STAIS}
            self.patient["STAIT"] = {"total":STAIT}

    def generatePatientDescription(self):
        description = f"""
        The patient is a {self.patient["age"]} year-old {self.patient["genderName"]}.
        Their vitals are:
            BP: {self.patient["systolicBP"]} / {self.patient["diastolicBP"]}.
            Cholesterol: HDL: {self.patient["HDL"]}, LDL: {self.patient["LDL"]}, Triglycerides: {self.patient["TRY"]}.

        """
        self.patient["PatientDescription"] = description.strip()

import modsim as sim

class MSP(object):

    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}

    
    def setInitialStates(lstInitialStates):
        a = lstInitialStates[0]; b = lstInitialStates[1]
        sm = sim.State(spotA = a,spotB = b)
        return sm

    def ItemToA(sm):
        sm.spotA += 1
        sm.spotB -= 1
        
    
    def ItemToB(sm):
        sm.spotA -= 1
        sm.spotB += 1

class planetaryorbit(object):
    def __init__(self) -> None:
        self.config = {}
        self.stats = {"error_details": []}
        self.data = {}
        self.system = {}

    def defineSystem(self,dictParameters):
        centerObjectName = dictParameters["centerObjectName"]
        NumberOfPlanets = dictParameters["NumberOfPlanets"]
        CenterDistance = dictParameters["CenterDistance"]
        self.system.update(centerObjectName = centerObjectName,
                           NumberOfPlanets = NumberOfPlanets,
                           CenterDistance=CenterDistance)
        return self.system

class MarkovPlaysTetris(object):
    def __init__(self) -> None:
        self.config = {}
        self.stats = {"error_details": []}
        self.data = {}
        self.system = {}

    def target_distribution(x):
        return np.exp(-x**2)

    def proposal_distribution(x):
        return np.random.normal(loc=x, scale=1)

    def metropolis_hastings(self, iterations)->tuple:
        samples = []
        current_x = 0
        for i in range(iterations):
            current_x = self.proposal_distribution(current_x)
            acceptance_ratio = self.target_distribution(current_x) / self.target_distribution(samples[-1])
            if np.random.uniform() < acceptance_ratio:
                samples.append(current_x)
            else:
                samples.append(samples[-1])
        
        return np.array(samples), acceptance_ratio

    