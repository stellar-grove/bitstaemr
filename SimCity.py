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
        
