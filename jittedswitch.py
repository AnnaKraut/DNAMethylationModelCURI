import numpy as np
from numba import jit

#import config file - we can use config functions as if they were defined here
#the only difference is that we prefix them with c. (so "function(x) becomes c.function(x)")
import config as c


# Jitted Switch
# This file is a refactoring of the gillespie simulation that's designed to allow it to run better with jit compiling.
# to do this, we need to convert the python objects into simpler objects (ideally numpy arrays) that play better with jit/numba.
# Specifically, config files might need to be defined in the file instead of imported, and dictionaries will be converted into lists
# also, there will be much less room for object-oriented code :(
#We will use prange across different simulations (since each simulation is seperate, can be executed in parallel)
#This code does not support debugging toggles, and doesn't print output at the simulation level.
#Might be some weirdness with rng, see here: https://numba.readthedocs.io/en/stable/reference/pysupported.html

#for reference only
default_parameters = {"r_hm": 0.5,          #0
                      "r_hm_m": 20/totalpop, #1
                      "r_hm_h": 10/totalpop, #2
                      "r_uh": 0.35,         #3
                      "r_uh_m": 11/totalpop,#4
                      "r_uh_h": 5.5/totalpop,#5
                      "r_mh": 0.1,           #6
                      "r_mh_u": 10/totalpop, #7
                      "r_mh_h": 5/totalpop,  #8
                      "r_hu": 0.1,            #9
                      "r_hu_u": 10/totalpop, #10
                      "r_hu_h": 5/totalpop,   #11
                      "birth_rate": 1         #12
}

def maintenance_rate_collaborative(methylated, unmethylated, population, param_local):
    hemimethylated = population - (methylated + unmethylated)
    return hemimethylated * (param_local[0] + param_local[2]*hemimethylated + param_local[1]*methylated)#r_hm param
    #rate = hemimethylated * (self.params["r_hm"] + self.params["r_hm_h"]*hemimethylated + self.params["r_hm_m"]*methylated)

def denovo_rate_collaborative(methylated, unmethylated, population, param_local):
    hemimethylated = population - (methylated + unmethylated)
    return unmethylated * (param_local[3] + param_local[5]*hemimethylated + param_local[4]*methylated)
    #rate = unmethylated * (self.params["r_uh"] + self.params["r_uh_h"]*hemimethylated + self.params["r_uh_m"]*methylated)

def demaintenance_rate_collaborative(methylated, unmethylated, population, param_local):
    hemimethylated = population - (methylated + unmethylated)
    return hemimethylated * (param_local[9] + param_local[11]*hemimethylated + param_local[10]*unmethylated)
    #rate = hemimethylated * (self.params["r_hu"] + self.params["r_hu_h"]*hemimethylated + self.params["r_hu_u"]*unmethylated)

def demethylation_rate_collaborative(methylated, unmethylated, population, param_local):
    hemimethylated = population - (methylated + unmethylated)
    return methylated * (param_local[6] + param_local[8]*hemimethylated + param_local[7]*unmethylated)
    #rate = methylated * (self.params["r_mh"] + self.params["r_mh_h"]*hemimethylated + self.params["r_mh_u"]*unmethylated)

def birth_rate(param_local):
      return param_local[12]

#Helper function that finds the state of the model for given population
#1 means >70% methylated, -1 means >70% unmethylated, 0 means somewhere in the middle
def find_state(methylated, unmethylated, population):
      if (methylated/ population) > 0.7:
            return 1
      if (unmethylated/ population) > 0.7:
            return -1
      return 0

#This function defines the events that can happen. It's equivalent to the event list in config.py
def events(methylated, unmethylated, totalpop, i_local, rng_local):
    #maintenance event
    if i_local == 0:
        return methylated+1, unmethylated
    #denovo methylation event
    elif i_local == 1:
        return methylated, unmethylated-1
    #demaintenance event
    elif i_local == 2:
        return methylated, unmethylated+1
    #demethylation event
    elif i_local == 3:
        return methylated-1, unmethylated
    #birth event
    elif i_local == 4:
        hemimethylated = totalpop - (methylated + unmethylated)
        newly_unmethylated = rng_local.binomial(hemimethylated, 0.5)
        return 0, (unmethylated + newly_unmethylated)



def GillespieSwitchFun(steps,param_arr,totalpop,pop_methyl, pop_unmethyl, SwitchDirection):
    currstep = 1
    methylated_arr = np.zeros(steps)
    unmethylated_arr = np.zeros(steps)
    methylated_arr[0] = pop_methyl
    unmethylated_arr[0] = pop_unmethyl
    time_arr = np.zeros(steps) #initialize first value to zero?
    time_arr[0] = 0
    param_count = len(param_arr)
    rng = np.random.default_rng()
    start_state = find_state(pop_methyl,pop_unmethyl, totalpop)

    rates = np.zeros(5)

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl
        dynamic_rates = np.zeros_like(param_arr)
        relative_probabilities = np.zeros_like(param_arr)

        #find the rates of each event for the current parameters
        rates[0] = maintenance_rate_collaborative(methylated_arr[i],unmethylated_arr[i],param_arr)
        rates[1] = denovo_rate_collaborative(methylated_arr[i],unmethylated_arr[i],param_arr)
        rates[2] = demaintenance_rate_collaborative(methylated_arr[i],unmethylated_arr[i],param_arr)
        rates[3] = demethylation_rate_collaborative(methylated_arr[i],unmethylated_arr[i],param_arr)
        rates[4] = birth_rate(param_arr)
        prob_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/prob_sum)
        time_arr[i] = tau + time_arr[i-1]

        #normalize the rates to be within (0,1)
        #this implicitly tries to divide an array by an int - might need to convert something to a float
        rates /= prob_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for i in range(5):
            if uniform < rates[i] + sum_so_far:
                #This multiple assignnment might not be allowed in numba
                methylated_arr[i], unmethylated_arr[i] = events(methylated_arr[i], unmethylated_arr[i], totalpop,i,rng)
                break
            else:
                sum_so_far += rates[i]
        
        #decide which state we are in - if we switched, this block will terminate the program
        curr_state = find_state(methylated_arr[i], unmethylated_arr[i], totalpop)
        if curr_state == 0:
            continue
        elif curr_state == SwitchDirection:
            if start_state == 0:
                start_state = curr_state
                continue
            return time_arr[i]
        
    #we timed out - return a negative value to indicate this was a timeout
    return -1 * time_arr[i]






class GillespieModelSwitchTime:
    def __init__(self,steps,param_dict, totalpop, pop_methyl, pop_unmethyl,SwitchDirection):
        #TODO: find a better (shorter) name for the index than "currstep" - maybe step? i?
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

            #find tau for our current state and update our time array
            tau = self.rng.exponential(scale = 1/prob_sum) 
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

            #This code will always switch when we switch states
            curr_state = c.find_state(self,i)
            if curr_state == 0: #can't tell what state we're in
                continue
            elif curr_state == self.SwitchDirection: #case for switching to a targeted direction.
                #we assume that the model was NOT in the self.switchdirection state to begin with
                #so, as soon as it ends up in the target state, we return
                if self.startstate == 0:
                    #this is the case that the simulation started with an even mix of methylated/unmethylated
                    #we set the start state to whichever state it reaches first
                    self.startstate = curr_state
                    continue
                return self.tarr[i]
             
        return -1 * self.tarr[i] #return a negative to indicate it timed out

# #TODO - write a wrapper function to call and time this function, to measure optimization impact