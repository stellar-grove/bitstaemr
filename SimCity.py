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