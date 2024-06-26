sg_repo = "C:/stellar-grove/"
import sys;sys.path.append(sg_repo)
import pandas as pd
import bitstaemr.tools as tools
import tara.distributions as dists
import numpy as np
from num2words import num2words
import datetime as dt
from scipy.stats import (expon, 
                         norm, 
                         poisson,
                         ttest_ind, 
                         ks_2samp,
                         poisson_means_test
                         )
from sklearn. preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (mean_squared_error,
                             mean_absolute_error)

dcmc = dists.DaCountDeMonteCarlo()


#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                       PERT                                                                 |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                       Biostatistics                                                        |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
class biostats(object):
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
        self.patient = {}

#--------------------------------------------------------------------------------------------------------
# Clearing
#--------------------------------------------------------------------------------------------------------      
    def clearPatient(self):
        self.patient = {}

    def clearCohort(self):
        if "cohort" in self.data.keys():
            self.data["cohort"] = {}
#--------------------------------------------------------------------------------------------------------
# Creating 
#--------------------------------------------------------------------------------------------------------    
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

    def createCohort(self,cohortSize:int,storeDf:bool=True):
        dfCohort = pd.DataFrame()
        print("OK")
        for i in range(cohortSize):
            df = pd.DataFrame().from_dict(self.createPatient(),orient="index")
            df = df[:-4].T
            dfCohort = pd.concat([dfCohort,df])
        if storeDf:
            self.data["cohort"] = dfCohort
        return dfCohort
 
#--------------------------------------------------------------------------------------------------------
# Generating 
#--------------------------------------------------------------------------------------------------------
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
 
#--------------------------------------------------------------------------------------------------------
 # Display
 #--------------------------------------------------------------------------------------------------------
    def displayFeature(self,feature:str):
        # Check to make sure that the feature is in the data dictionary
        if feature in self.data["cohort"].keys:
            True

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                       Modeling and Simulation Using Python                                 |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|


#import modsim as sim
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


    def make_system(beta, gamma):
        init = sim.State(s = 89, i = 1, r = 0)
        init /= init.sum()
        return sim.System(init=init, t_end=7*14,beta=beta, gamma=gamma)

    def update_func(t, state, system):
        s, i, r = state.s, state.i, state.r

        infected = system.beta * i * s
        recovered = system.gamma * i

        s -= infected
        i += infected - recovered
        r += recovered

        return sim.State(s=s, i=i, r=r)


    def plot_results(S, I, R):
        S.plot(style='--', label='Susceptible')
        I.plot(style='-', label='Infected')
        R.plot(style=':', label='Resistant')
        sim.decorate(xlabel='Time (days)',
                ylabel='Fraction of population')
       
    def run_simulation(system, update_func):
        frame = sim.TimeFrame(columns=system.init.index)
        frame.loc[0] = system.init
        for t in range(0, system.t_end):
            frame.loc[t+1] = update_func(t, frame.loc[t], system)
        
        return frame

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                       Planetary Orbits                                                     |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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

#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
#|                       MCMC Modelling                                                       |
#|~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
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

# Grok Inspired  
#from Bio import SeqIO
#from Bio.SeqFeature import SeqFeature, FeatureLocation
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

class GrokSays(object):

    
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
    spyder_text = """
        # Load the genome sequence
        def load_genome(file_path):
            genome_seq = SeqIO.read(file_path, 'fasta')
            return str(genome_seq.seq)

        # Define a function to identify genes using a simple pattern matching approach
        def find_genes(genome_seq, gene_patterns):
            genes = []
            for gene_pattern in gene_patterns:
                match = gene_pattern.search(genome_seq)
                if match:
                    start, end = match.span()
                    genes.append((gene_pattern.pattern, start, end))
            return genes

        # Define a simple classification model
        def classify_creature(training_data, test_data):
            X_train, X_test, y_train, y_test = train_test_split(training_data[0], training_data[1], test_size=0.2, random_state=42)

            clf = svm.SVC(kernel='linear', C=1)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)

            accuracy = accuracy_score(y_test, y_pred)
            print(f'Accuracy: {accuracy}')

            return clf.predict(test_data)

        # Example usage:
        genome_file_path = 'your_genome_file.fasta'
        genome_seq = load_genome(genome_file_path)

        # Define gene patterns to search for
        gene_patterns = [
            r'ATG\w{10}TGA',  # Example pattern for a start codon (ATG) followed by 10 nucleotides and a stop codon (TGA)
        ]

        # Find genes in the genome sequence
        genes = find_genes(genome_seq, gene_patterns)
        print(f'Found {len(genes)} genes:')
        for gene in genes:
            print(f'Gene: {gene[0]}')
            print(f'Start: {gene[1]}')
            print(f'End: {gene[2]}')

        # Load or create training data and perform classification
        # training_data = [training_features, training_labels]
        # test_data = [test_features]

        # classified_creature = classify_creature(training_data, test_data)
        # print(f'The creature is classified as: {classified_creature}')

    """
    class GenomeSequencing():

        introduction = """
                    Imagine being able to identify an animal's species just by looking at its DNA. 
                    Sounds like something out of a sci-fi movie, right? 
                    Well, thanks to the wonders of modern technology, this is now a reality! 
                    In this study, we'll be using Python to develop a machine learning model 
                    that can classify animals based on their genomic sequences.

                    """
        def __init__(self,config={}) -> None:
            self.config = config
            self.stats = {"error_details": []}
            self.introduction = self.introduction

        def load_sample_data(self,file_location:str=None, return_df:bool=True):
            danimals = [
                        {"name": "Elephant", "sequence": "ATGCATGCATGCATGC"},
                        {"name": "Giraffe", "sequence": "TGCATGCATGCATGC"},
                        {"name": "Lion", "sequence": "GCTGCATGCATGC"},
                        {"name": "Tiger", "sequence": "CATGCATGCATGC"},
                        {"name": "Bear", "sequence": "GCATGCATGCATGC"}
                        ]
            df_danimals = pd.DataFrame(danimals)
            if file_location != None:
                df_danimals.to_csv(file_location)
            return df_danimals

    



    class QuantFinance(object):
        def __init__(self,config={}) -> None:
            self.config = config
            self.stats = {"error_details": []}
            self.data = {}

        def getData(self):
            return "Pumpkin Head needs to finish this."


import random
import math
import matplotlib.pyplot as plt

class FlightPath(object):
    
    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
        self.flight_path = {}
    
    def set_obstacles(self,obstacleList):
        self.data['obstacles'] = obstacleList

    def set_path(self, n_steps=10):
        flightpath = np.random.random_integers(low=-10,high=10,size=n_steps)
        self.data['path'] = flightpath.tolist()
        return flightpath

    def calc_distance(position1:tuple, position2:tuple):
        x1, y1 = position1
        x2, y2 = position2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def showPath(self):
        obs = self.data['obstacles']
        path = self.data['path']
        return plt.scatter(path[0],path[1])

    def simulate_flight_path(self,obstacles, bird_position, flight_distance):

        while True:
            # Randomly generate a direction and a distance
            direction = random.choice([(-1, 0), (1, 0), (0, -1), (0, 1)])
            distance = random.uniform(0, flight_distance)

            # Calculate the new bird position
            new_bird_position = (
                bird_position[0] + direction[0] * distance,
                bird_position[1] + direction[1] * distance
            )

            # Check if the bird collides with any obstacles
            for obstacle in obstacles:
                print(obstacle)
                if self.calc_distance(new_bird_position, obstacle) < 2:
                    
                    return self.flight_path

            # Add the new position to the flight path
            self.flight_path.append(new_bird_position)

            # Update the bird's position
            bird_position = new_bird_position
            
    def run_simulation(self, bird_position, flight_distance):
        flight_distance = flight_distance  # Adjust this value to change the flight distance per step
        flight_path = self.simulate_flight_path(self.data['obstacles'], bird_position, flight_distance)

        print(f"The bird's flight path is: {flight_path}") 

class HIV(object):
    
    def __init__(self,config={}) -> None:
        self.config = config
        self.viral_load = None
        self.data = None
        
    data_loc = 'C:/Users/DanielKorpon/Stellar Grove/ticondagrova - Documents/PMLSdata/'
        
    def generate_time(self, start, stop, size=10):
        time = np.linspace(start,stop,size)
        self.time = time

    def generate_viral_load(self, alpha, beta, A, B, size, return_df=False):
        self.generate_time(size)
        viral_load = A * np.exp(-alpha * self.time) + B * np.exp(-beta * self.time)
        self.viral_load = viral_load
        if return_df:
            return viral_load
        
    def plot_viral_load(self):
        plt.plot(self.time, self.viral_load)
        
    def load_data(self, file_name):
        data_loc = f'{self.data_loc}{file_name}'
        data_set = np.loadtxt(data_loc,delimiter=',')
        self.data = data_set

class Bacteria(object):
    
    def __init__(self,config={}) -> None:
        self.config = config
        self.viral_load = None
        self.data = None
        self.w_load = None

    def generate_time(self, start, stop, size=10):
        time = np.linspace(start,stop,size)
        self.time = time

    def generate_viral_load(self, time, tau):
        self.viral_load = 1 - np.exp(-time / tau)
        return self.viral_load
    
    def W(self, time, tau, A):
        self.w_load = A * (np.exp(-time / tau) - 1 + (time / tau))
        return self.w_load
    
class TaxiCab(object):
    
    def __init__(self,config={}) -> None:
        self.config = config
        self.data = None

    def taxicab(self, pointA:tuple, pointB:tuple):
        interval = abs(pointB[0] - pointA[0]) + abs(pointB[1] - pointA[1])
        return interval
    
    def trip_cost(self, start, stop, fare_rate):
        return np.round(self.taxicab(start, stop) * fare_rate, 2)

class Distance(object):

    def distance(pointA:tuple, pointB:tuple=(0,0), metric='taxi'):

        if metric == 'taxi':
            interval = abs(pointB[0] - pointA[0]) + abs(pointB[1] - pointA[1])

        else:
            interval = np.sqrt((pointB[0] - pointA[0])**2 + (pointB[1] - pointA[1])**2)
        
        return interval
        
class Baseball(object):
        def __init__(self) -> None:
            self.config = {}
            self.stats = {"error_details": []}
            self.data = {}
            self.system = {}
        
        def pitch_ball(pitcherName:str='Yoko Ono'):
            # Look up the pitcher to determine their attributes. This requires a dictionary to be made
            for pitch in np.arange(1,7):
                pitch_location = np.random.randint(100)
                if pitch_location < 60:
                    ball_strike = 'Strike'
                else:
                    ball_strike = "Ball"
                print(f"pitch Number: {pitch} had a location of {pitch_location}, which correlates to a {ball_strike}")
            return pitch_location

class Marketing(object):

    class ABTesting(object):
        def __init__(self, distribution:str, values:list, size:int) -> None:
            self.distribution = distribution
            self.values = values
            self.size = size
            self.data = {}
            self.stats = {}

        def createData(self):

            if self.distribution.lower() in ["exponential", "expon","exp"]:
                
                # Exponential distribution parameters
                lam1 = self.values[0]
                lam2 = self.values[1]

                # Generate exponentially distributed samples
                np.random.seed(0)
                control_data = norm.rvs(scale=1/lam1, size=self.size)
                treatment_data = expon.rvs(scale=1/lam2, size=self.size)
                self.data['control'] = control_data
                self.data['treatment'] = treatment_data

            if self.distribution.lower() in ["normal", "norm", "n"]:
                # Exponential distribution parameters
                mu1 = self.values[0][0]; sigma1 = self.values[0][1]
                mu2 = self.values[1][0]; sigma2 = self.values[1][1]

                # Generate exponentially distributed samples
                np.random.seed(0)
                self.data['control'] = norm.rvs(loc=mu1, scale=sigma1, size=self.size)
                self.data['treatment'] = expon.rvs(loc = mu2, scale=sigma2, size=self.size)

            if self.distribution.lower() in ["poisson", "poison", "pson", "pois"]:
                self.data['control'] = poisson.rvs(self.values[0], size=self.size)
                self.data['treatment'] = poisson.rvs(self.values[1], size=self.size)
                
            return (self.data['control'], self.data['treatment'])

        def performTest(self, control_data, treatment_data):
            
            if self.distribution.lower() in ["exponential", "expon","exp"]: 
                # Perform statistical tests
                self.stats["stat"], self.stats["p_value"] = ks_2samp(control_data, treatment_data)

            if self.distribution.lower() in ["poisson", "poison", "pson", "pois"]:
                # Perform statistical tests
                stat, p_value = poisson_means_test(self.values[0], self.size, self.values[1], self.size)
                self.stats['stat'] = stat
                self.stats['p_value'] = p_value

            if self.distribution.lower() in ["normal", "norm", "n"]:
                # Perform statistical tests
                stat, p_value = ttest_ind(self.data['control'], self.data['treatment'])
                self.stats['stat'] = stat
                self.stats['p_value'] = p_value
            

            return (stat, p_value)

        def run(self, print_stats:bool=False):
            self.performTest(self.createData()[0], self.createData()[1])
            if print_stats:
                return self.stats

        def plotHistograms(self,control_data,treatment_data):
            plt.hist(control_data, bins=30, alpha=0.5, label='Control')
            plt.hist(treatment_data, bins=30, alpha=0.5, label='Treatment')
            plt.xlabel('Time (seconds)')
            plt.ylabel('Frequency')
            plt.title('A/B Testing Results')

        # # Visualize the results

        # plt.xlabel('Time (seconds)')
        # plt.ylabel('Frequency')
        # plt.title('A/B Testing Results')
        # plt.legend()
        # plt.show()

class MSP(object):
    SimpleNamespace = type(sys.implementation)
    class SettableNamespace(SimpleNamespace):
        """Contains a collection of parameters.

        Used to make a System object.

        Takes keyword arguments and stores them as attributes.
        """
        def __init__(self, namespace=None, **kwargs):
            super().__init__()
            if namespace:
                self.__dict__.update(namespace.__dict__)
            self.__dict__.update(kwargs)

        def get(self, name, default=None):
            """Look up a variable.

            name: string varname
            default: value returned if `name` is not present
            """
            try:
                return self.__getattribute__(name, default)
            except AttributeError:
                return default

        def set(self, **variables):
            """Make a copy and update the given variables.

            returns: Params
            """
            new = copy(self)
            new.__dict__.update(variables)
            return new

    class System(SettableNamespace):
        """Contains system parameters and their values.

        Takes keyword arguments and stores them as attributes.
        """
        pass

    class Params(SettableNamespace):
        """Contains system parameters and their values.

        Takes keyword arguments and stores them as attributes.
        """
        pass

    def State(**variables):
        """Contains the values of state variables."""
        return pd.Series(variables, name='state')     

class Wildlife(object):

    def __init__(self, config = {}) -> None:
            self.config = config

    def birth_process(self, lam:int=1):
        for gen in range(10):
            print(lam)
