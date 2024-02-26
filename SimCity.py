import modsim as sim

class bikeshare(object):

    def __init__(self,config={}) -> None:
        self.config = config
        self.stats = {"error_details": []}
        self.data = {}
        
  
    def setStates(self, lstStates):
        a = lstStates[0]
        b = lstStates[1]
        self.data["state"] = sim.State(spotA = a, spotB = b)

    def ItemToA(self, sharemodel):
        sharemodel.spotA +=1
        sharemodel.spotB -=1
        self.data["state"] = sharemodel
    
    def ItemToB(self, sharemodel):
        sharemodel.spotA -=1
        sharemodel.spotB +=1
        self.data["state"] = sharemodel
        
# Chapter 1


# Chapter 2

bikeshare = sim.State(olin = 10, wellesley = 2)
bikeshare.olin
bikeshare.wellesley

def bike_to_wellesley():
    print("Moving Bike to Wellesley")
    bikeshare.olin -= 1
    bikeshare.wellesley += 1
    
    
def bike_to_olin():
    print("Moving Bike to Olin")
    bikeshare.olin += 1
    bikeshare.wellesley -= 1
    
def step(state, p1, p2):
    if sim.flip(p1):
        state.wellesley += 1
        
    if sim.flip(p2):
        state.olin += 1

results = sim.TimeSeries() 
results[0] = bikeshare.olin       
for i in range(3):
    print(i)
    step(0.6, 0.6)
    results[i+1] = bikeshare.olin
results

results.plot()    
sim.decorate(title="Olin-Wellesley bikeshare",
         xlabel="Time step (min)",
         ylabel="Number of bikes")

# Chapter 3
def bike_to_wellesley(state):

    if state.wellesley == 0:
        return
    state.wellesley -= 1
    state.olin += 1
    
def bike_to_olin(state):
    if state.wellesley == 0:
        state.wellesley_empty +=1
        return
    state.wellesley -= 1
    state.olin += 1
    
bikeshare = sim.State(olin=12, wellesley=0,
                  wellesley_empty=0)

bike_to_olin(bikeshare)



# Chapter 4

def run_simulation(p1, p2, num_steps):
    state = sim.State(olin = 10, wellesley = 2,
                      olin_empty = 0, wellesley_empty = 0)
    
    for i in range(num_steps):
        step(state, p1, p2)
        
    return state

final_state = run_simulation(0.3, 0.2, 60)

print(final_state.olin_empty,
      final_state.wellesley_empty)

sweep = sim.SweepSeries()

p1_array = sim.linspace(0, 0.6, 31)
p2 = 0.2
num_steps = 60

for p1 in p1_array:
    final_state = run_simulation(p1, p2, num_steps)
    sweep[p1] = final_state.olin_empty

sweep.plot(label = 'Olin', color = 'C1')


# Chapter 5

import pandas as pd

url = 'https://en.wikipedia.org/wiki/Estimates_of_historical_world_population'

filename = f'{url}World_population_estimates.html'
tables = pd.read_html(url,header=0,index_col=0,decimal='M')
table2 = tables[2]

table2.columns = ['census', 'prb', 'un', 'maddison',
                  'hyde', 'tanton', 'biraben', 'mj',
                  'thomlinson', 'durand','clark']

census = table2.census / 1e9
census.tail()

un = table2.un / 1e9
un.tail()


def plot_estimates():
    census.plot(style=":", label='US Census')
    un.plot(style="--", label=["UN DESA"]) 
    sim.decorate(xlabel='Year', ylabel='World population (billions)')

plot_estimates()    

import numpy as np


abs_error = abs(un - census)
np.mean(abs_error)
np.max(abs_error)

rel_error = 100 * abs_error / census
rel_error.tail()
np.mean(rel_error)

# MOdelling Population Growth
census[1950]

total_growth = census[2016] - census[1950]

t_0 = census.index[0]
t_0

t_end = census.index[-1]
t_end

elapsed_time = t_end - t_0
elapsed_time

p_0 = census[t_0]
p_end = census[t_end]

total_growth = p_end - p_0
total_growth

annual_growth = total_growth / elapsed_time
annual_growth

results = sim.TimeSeries()

results[t_0] = p_0

for t in range(t_0, t_end):
    results[t+1] = results[t] + annual_growth

results.plot(color='gray', label='model')
un.plot(style=":")
plot_estimates()


# Chapter 6

import pandas as pd
import matplotlib.pyplot as plt

url = 'https://en.wikipedia.org/wiki/Estimates_of_historical_world_population'

filename = f'{url}World_population_estimates.html'
tables = pd.read_html(url,header=0,index_col=0,decimal='M')
table2 = tables[2]

table2.columns = ['census', 'prb', 'un', 'maddison',
                  'hyde', 'tanton', 'biraben', 'mj',
                  'thomlinson', 'durand','clark']

census = table2.census / 1e9
un = table2.un / 1e9
t_0 = int(census.index[0])
t_end = int(census.index[-1])
elapsed_time = t_end - t_0

p_0 = census[t_0]
p_end = census[t_end]

total_growth = p_end - p_0
annual_growth = total_growth / elapsed_time

system = sim.System(t_0=t_0,
                    t_end=t_end,
                    p_0=p_0,
                    annual_growth=annual_growth)


def run_simulation1(system):
    results = sim.TimeSeries()
    results[system.t_0] = system.p_0

    for t in range(int(system.t_0), int(system.t_end)):
        results[t+1] = results[t] + system.annual_growth

    return results
    
results1 = run_simulation1(system)

def plot_estimates():
    census.plot(style=":", label='US Census')
    un.plot(style="--", label='UN DESA')
    sim.decorate(xlabel="Year",
                 ylabel="World Population (billions)")
    
results1.plot(label='model', color='gray')
plot_estimates()
sim.decorate(title="Constant growth model")


# A Proporational Growth Model

def run_simulation2(system):
    results = sim.TimeSeries()
    results[system.t_0] = system.p_0
    
    for t in range(int(system.t_0), int(system.t_end)):
        births = system.birth_rate * results[t]
        deaths = system.death_rate * results[t]
        results[t+1] = results[t] + births - deaths
        
    return results

system.death_rate = 7.7 / 1000
system.birth_rate = 25 / 1000
results2 = run_simulation2(system)
results2.plot(label='model', color = 'gray')
plot_estimates()
sim.decorate(title='Proporational growth model')

# Factoring Out the Update Function

def growth_func1(t, pop, system):
    births = system.birth_rate * pop
    deaths = system.death_rate * pop
    return births - deaths


def run_simulation(system, growth_func):
    results = sim.TimeSeries()
    results[system.t_0] = system.p_0
    
    for t in range(int(system.t_0), int(system.t_end)):
        growth = growth_func(t, results[t], system)
        results[t+1] = results[t] + growth
        
    return results

results = run_simulation(system, growth_func1)

system.alpha = system.birth_rate - system.death_rate

def growth_func2(t, pop, system):
    return system.alpha * pop

results = run_simulation(system,growth_func2)



# Chapter 11







































































































































