import numpy as np
from numba import njit
import matplotlib.pyplot as plt

"""
This variant of the optimized gillespie simulation returns a tuple object containing the time at which the switch occured,
and another tuple containing the number of sites that were methylated and unmethylated at this time. 

This file is a refactoring of the gillespie simulation that's designed to allow it to run better with jit compiling.
To do this, we need to convert the python objects into simpler objects (ideally numpy arrays) that play better with jit/numba.
Specifically, config files might need to be defined in the file instead of imported, and dictionaries will be converted into lists
We will use prange across different simulations (since each simulation is seperate, can be executed in parallel)
This code does not support debugging toggles, and doesn't print output at the simulation level.
There might be some weirdness with rng, see here: https://numba.readthedocs.io/en/stable/reference/pysupported.html
"""

@njit
def maintenance_rate_collaborative(methylated, unmethylated, site_count, param_local):
    hemimethylated = site_count - (methylated + unmethylated)
    return hemimethylated * (param_local[0] + param_local[2]*hemimethylated + param_local[1]*methylated)#r_hm param
    #rate = hemimethylated * (self.params["r_hm"] + self.params["r_hm_h"]*hemimethylated + self.params["r_hm_m"]*methylated)
@njit
def denovo_rate_collaborative(methylated, unmethylated, site_count, param_local):
    hemimethylated = site_count - (methylated + unmethylated)
    return unmethylated * (param_local[3] + param_local[5]*hemimethylated + param_local[4]*methylated)
    #rate = unmethylated * (self.params["r_uh"] + self.params["r_uh_h"]*hemimethylated + self.params["r_uh_m"]*methylated)
@njit
def demaintenance_rate_collaborative(methylated, unmethylated, site_count, param_local):
    hemimethylated = site_count - (methylated + unmethylated)
    return hemimethylated * (param_local[9] + param_local[11]*hemimethylated + param_local[10]*unmethylated)
    #rate = hemimethylated * (self.params["r_hu"] + self.params["r_hu_h"]*hemimethylated + self.params["r_hu_u"]*unmethylated)
@njit
def demethylation_rate_collaborative(methylated, unmethylated, site_count, param_local):
    hemimethylated = site_count - (methylated + unmethylated)
    return methylated * (param_local[6] + param_local[8]*hemimethylated + param_local[7]*unmethylated)
    #rate = methylated * (self.params["r_mh"] + self.params["r_mh_h"]*hemimethylated + self.params["r_mh_u"]*unmethylated)
@njit
def birth_rate(param_local):
      return param_local[12]

#Helper function that finds the state of the model for given site_count
#1 means >70% methylated, -1 means >70% unmethylated, 0 means somewhere in the middle
@njit
def find_state(methylated, unmethylated, site_count):
      if (methylated/ site_count) > 0.7:
            return 1
      if (unmethylated/ site_count) > 0.7:
            return -1
      return 0

#This function defines the events that can happen. It's equivalent to the event list in config.py
#i_local indicates which loop called this function - that is, i_local indicates which event we're doing.
@njit
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

@njit
def GillespieSwitchFun(steps, param_arr, totalpop, pop_methyl, pop_unmethyl, SwitchDirection, rng):
    methylated_arr = np.zeros(steps)
    unmethylated_arr = np.zeros(steps)
    methylated_arr[0] = pop_methyl 
    unmethylated_arr[0] = pop_unmethyl
    time_arr = np.zeros(steps) 
    time_arr[0] = 0 #initialize first value to zero to stay in sync with (un)methylated_arr
    start_state = find_state(pop_methyl,pop_unmethyl, totalpop)
    rates = np.zeros(5) #using numpy array may or may not be optimal here - possible refactor point

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl

        #find the rates of each event for the current parameters
        rates[0] = maintenance_rate_collaborative(methylated_arr[i-1],unmethylated_arr[i-1],totalpop,param_arr)
        rates[1] = denovo_rate_collaborative(methylated_arr[i-1],unmethylated_arr[i-1],totalpop,param_arr)
        rates[2] = demaintenance_rate_collaborative(methylated_arr[i-1],unmethylated_arr[i-1],totalpop,param_arr)
        rates[3] = demethylation_rate_collaborative(methylated_arr[i-1],unmethylated_arr[i-1],totalpop,param_arr)
        rates[4] = birth_rate(param_arr)
        rate_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/rate_sum)
        time_arr[i] = tau + time_arr[i-1]

        #normalize the rates to be within (0,1)
        normalized_rates= rates / rate_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for event_number in range(5):
            if uniform < normalized_rates[event_number] + sum_so_far:
                methylated_arr[i], unmethylated_arr[i] = events(methylated_arr[i-1], unmethylated_arr[i-1], totalpop,event_number,rng)
                break
            else:
                sum_so_far += normalized_rates[event_number]

        # decide which state we are in - if we switched, this block will terminate the program
        curr_state = find_state(methylated_arr[i], unmethylated_arr[i], totalpop)
        if curr_state == SwitchDirection:
            #select final coordinates and return them
            final_coords = (methylated_arr[i],unmethylated_arr[i])
            return time_arr[i], final_coords
        
    #we timed out - return a negative value to indicate that this isn't a normal run.
    return (-1 * time_arr[i]), (-1,-1)
