import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from dataclasses import dataclass

#import config file - we can use config functions as if they were defined here
#the only difference is that we prefix them with c. (so "function(x) becomes c.function(x)")
import config as c

#TODO: convert to use numpy arrays (faster)
#TODO: ask if its ok that arrays are fixed length - this makes optimizing them substantially easier
#TODO: get some examples of ground-truth behavior to check models against
#TODO: ask if they want runs saved, if so how

class GillespieModelSwitchTime:
    def __init__(self,steps,param_dict, totalpop, pop_methyl, pop_unmethyl,SwitchDirection, debug = False, FinishAndSave = False):
        #TODO: find a better (shorter) name for the index than "currstep" - maybe step? i?
        self.debug = debug #this is a debug toggle - if its true, we print out graphs and additional info.
        self.population = totalpop
        self.currstep = 1         #index of which step we're on
        self.steps = steps        #total steps
        self.methylated = [0]*steps    
        self.unmethylated = [0]*steps
        self.methylated[0] = pop_methyl
        self.unmethylated[0] = pop_unmethyl
        self.tarr = [0]*steps     #time array
        self.rng = np.random.default_rng() 
        #create an rng object - think of it as buying the dice we'll roll later - 
        #we can seed the rng object if we want reproducible results
        #is this initialization unnesceary? can we just point back to the param dict?
        self.params = param_dict
        #is the model methylated or unmethylated - we'll check this later to see if it switches
        self.startstate = c.find_state(self,0)
        self.FinishAndSave = FinishAndSave
        self.SwitchDirection = SwitchDirection
    
    def main(self):
        for i in range(1, self.steps):
            dynamic_rates = {}
            relative_probabilities = {}
            prob_sum = 0
            sum_so_far = 0

            #calculate dynamic rates - that is, the rates given current state of model
            for key in c.rate_calculation:
                dynamic_rates[key] = c.rate_calculation[key](self)
                prob_sum += dynamic_rates[key]
            #print("sum of probabilities: ", prob_sum)

            #find tau for our current state and update our time array
            tau = self.rng.exponential(scale = 1/prob_sum) 
            #print("resulting tau = ", tau)
            self.tarr[i] = tau + self.tarr[i-1]

            #calculate relative probability of each event happening
            #this is the 'width' of the event in the interval (0,1)
            for key in dynamic_rates:
                relative_probabilities[key] = dynamic_rates[key] / prob_sum


            #Select which event happens by comparing a number from a uniform distribution on (0,1)
            #to the various relative probabilities
            uniform = self.rng.uniform() #generate a uniform R.V.
            for key in relative_probabilities:
                #this is the case that the R.V. fell in the probability range of this event
                if uniform < sum_so_far + relative_probabilities[key]:
                    c.base_events[key](self) #call the function for this event
                    #print("matched with key", key, "new n is ", self.narr[i])
                    break
                #R.V. didn't fall in probability range for this event - 
                else:
                    sum_so_far += relative_probabilities[key]
            self.currstep += 1

            #this block will stop the graph when it switches ONLY IF the finish and save parameter was true
            if self.FinishAndSave == False:
                #check to see if we switched
                curr_state = c.find_state(self,i)
                if curr_state == 0: #can't tell what state we're in
                    continue
                # elif curr_state != self.startstate: #we're in the opposite state - we switched!
                #     #TODO: be very very careful about off by one errors - do we want tarr of i, i-1, or i+1????
                #     #return (self.tarr[i],self.tarr,self.methylated, self.unmethylated)
                #     if self.startstate == 0:
                #         #this is the case that the simulation started with an even mix of methylated/unmethylated
                #         #we set the start state to whichever state it reaches first
                #         self.startstate = curr_state
                #         continue
                #     if (self.debug):
                #         c.debug_graph(self,i, self.debug, self.FinishAndSave)   
                #     return self.tarr[i]
                elif curr_state == self.SwitchDirection: #case for switching to a targeted direction.
                    #we assume that the model was NOT in the self.switchdirection state to begin with
                    #so, as soon as it ends up in the target state, we return
                    if (self.debug):
                        c.debug_graph(self,i, self.debug, self.FinishAndSave)   
                    return self.tarr[i]
        # never switched
        if self.debug:
            print("never switched - ran for max iterations")
            #display the graph - FinishAndSave controls whether its saved or not
            c.debug_graph(self,i, self.debug)
        #save image without showing graph / printing debug info
        elif self.FinishAndSave:
            c.debug_graph(self, i, self.debug, self.FinishAndSave)
             
        return self.tarr[i]

# #TODO - write a wrapper function to call and time this function, to measure optimization impact