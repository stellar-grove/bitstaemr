sg_repo = "C:/stellar-grove/bitstaemr"
import sys;sys.path.append(sg_repo)
import pandas as pd
#import dkUtils.tools as tools
import tara.distributions as dists
from tara.distributions import DaCountDeMonteCarlo as dcmc
import numpy as np
from num2words import num2words
import datetime as dt
import scipy.stats as stats

def generatePersonalityType():
    # 0 Miserable, 1 Debbie Downer, 2 Even Steven, 3 Optimistic Oliver, 4 Richard Fucking Simons
    # Generate a random number between 0 and 4 to designate the personality type.
    personalityType = int(np.floor(dcmc.createUniformData(dcmc,0, 5, 1)[0]))
    return personalityType

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
        self.generateSWLS()
        self.generatePSS()
        self.generateSTAI()
        return self.patient

    def createCohort(self,cohortSize):
        dfCohort = pd.DataFrame()
        for i in range(cohortSize):
            df = pd.DataFrame().from_dict(self.createPatient(),orient="index")
            df = df[:-4].T
            dfCohort = pd.concat([dfCohort,df])
        return dfCohort

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
        age = np.round(age,2)
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
        
        personalityType = generatePersonalityType()
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
        personalityType = generatePersonalityType()
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
        personalityType = generatePersonalityType()
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


# Da Count Helpers

def getDefaultParameters():
        dict_config = {
            "sampleSize":100,
            "tsp":{
                    "name":"tsp",
                    "low":13,
                    "mid":18,
                    "hi":25,
                    "n":2
                    },
            "normal":{
                    "name":"normal",
                    "mean":3,
                    "std":1.4
                   },
            "poisson":{
                        "name":"poisson",
                        "mu":4
                    },
            "binomial":{
                        "name":"binomial",
                        "n":5,
                        "p":0.4
                        },
            "bernoulli":{
                        "name":"bernoulli",
                        "p":0.271
                        }
                
        }
        return dict_config

def getDataFrameNames():
    lst = ["dataframe", "df", "data-frame"]
    return lst

class DaCountDeMonteCarlo(object):
    def __init__(self,config={}) -> None:
        self.config = getDefaultParameters()
        self.stats = {"error_details": []}
        self.data = {}

    def setParameters(self,dict_update):
        self.config.update(dict_update)

    def generateSingleSample(self, dict_distribution):
        dfOutput = pd.DataFrame(self.createUniformData(0, 1, self.config["sampleSize"]), columns=["uniSample"])
        if dict_distribution["distributionName"].lower() in ["tsp","twosidedpower","two-sided-power"]:
            listParameters = [dict_distribution["distributionParameters"]["low"],
                                dict_distribution["distributionParameters"]["mid"],
                                dict_distribution["distributionParameters"]["high"],
                                dict_distribution["distributionParameters"]["n"]
                                ]
            print(listParameters)
            dfOutput.loc[:,"TSP"] = dfOutput["uniSample"].apply(lambda x: generateTSP(listParameters, x))
        if dict_distribution["distributionName"].lower() == "normal":
            listParameters = [dict_distribution["distributionParameters"]["mean"],dict_distribution["distributionParameters"]["std"]]
            dfOutput.loc[:,"Normal"] = dfOutput["uniSample"].apply(lambda x: self.sampleFromNormal(listParameters[0], listParameters[1], x))
        if dict_distribution["distributionName"].lower() == "poisson":
            listParameters = [dict_distribution["distributionParameters"]["mu"]]
            dfOutput.loc[:,"Poisson"] = dfOutput["uniSample"].apply(lambda x: self.sampleFromPoisson(listParameters[0], x))
        return dfOutput

#------------------------------------------------------------------------------------------------------------------------------
#   Functions to sample data from a distribution.
#   These functions generate the value of a distribution, given a percentile.  This is different 
#------------------------------------------------------------------------------------------------------------------------------

    def sampleFromNormal(self, mean, std, sample):
        z = stats.norm.ppf(sample)
        sampledValue = mean + (z * std)
        return sampledValue
    
    def sampleFromPoisson(self, mu, sample):
        value = stats.poisson.ppf(sample, mu)
        return value

    def sampleFromBernoulli(self, sample, p, loc):
        sampledValue = stats.bernoulli.ppf(sample, p, loc)
        return sampledValue
    
    def sampleFromBinomial(self, sample, n, p, loc):
        sampledValue = stats.binom.ppf(sample, n, p, loc)
        return sampledValue
    def sampleFromGamma(self, sample, alpha, loc, scale):
        sampledValue = stats.gamma.ppf(sample, alpha, loc, scale)
        return sampledValue
    
    def sampleFromExponential(self, sample, loc, scale):
        sampledValue = stats.expon.ppf(sample, loc, scale)
        return sampledValue
    
    def sampleFromBeta(self, sample, a, b, loc, scale):
        sampledValue = stats.beta.ppf(self, sample, a, b, loc, scale)
        return sampledValue
    

#------------------------------------------------------------------------------------------------------------------------------
#   Functions to create data
#   The functions below are intended to be used as helper functions that generated data needed to run analyses.  The class
#   contains these functions because ultimately the class is devoted to monte carlo simulations, and needing to generate random
#   samples from a distribution are needed.
#------------------------------------------------------------------------------------------------------------------------------

    def createPoissonData(self, mu, sampleSize, output = "list"):
        lst = stats.poisson.rvs(mu,sampleSize)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createUniformData(a, b, sampleSize, output = "list"):
        lst = np.random.uniform(a, b, sampleSize)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createNormalData(self, mean,std,size,output = "list"):
        lst = stats.norm.rvs(mean,std,size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createGammaData(self, alpha, size, output = "list"):
        lst = stats.gamma.rvs(alpha,size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    # This requires that you pass the scale as equal to 1/lambda
    def createExponentialData(self, scale,location,size, output = "list"):
        lst = stats.expon.rvs(scale=(scale),loc=location,size=size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBinomialData(self, n, p, size, output = "list"):
        lst = stats.binom.rvs(n,p,size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBernoulliData(self, p, loc, size, output = "list"):
        lst = stats.bernoulli.rvs(p, loc ,size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst

    def createBetaData(self, a, b, loc, scale, size, output = "list"):
        lst = stats.beta.rvs(a, b, loc, scale, size)
        if output.lower() in ["list"]: lst
        if output.lower() in getDataFrameNames(): lst = pd.DataFrame(lst,columns=["GeneratedData"])
        if output.lower() in ["dict","dictionary"]: lst = dict(zip(range(0,len(lst)),lst))
        return lst
        
        
        
        
        
from Bio import SeqIO
from Bio.SeqFeature import SeqFeature, FeatureLocation
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

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