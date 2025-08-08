import numpy as np
from numba import njit
import matplotlib.pyplot as plt

"""
This runs the gillespie algorithm simulation and returns lists with the x_m and x_u populations at each instant in the event timepoints array
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
def GillespieLongRunFun(steps, param_arr, totalpop, pop_methyl, pop_unmethyl, rng):
    methylated_arr = np.zeros(steps)
    unmethylated_arr = np.zeros(steps)
    time_arr = np.zeros(steps) 

    curr_methyl = pop_methyl 
    curr_unmethyl = pop_unmethyl
    total_time = 0
    divisions = 0

    rates = np.zeros(5) #using numpy array may or may not be optimal here - possible refactor point

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl

        #find the rates of each event for the current parameters
        rates[0] = maintenance_rate_collaborative(curr_methyl,curr_unmethyl,totalpop,param_arr)
        rates[1] = denovo_rate_collaborative(curr_methyl,curr_unmethyl,totalpop,param_arr)
        rates[2] = demaintenance_rate_collaborative(curr_methyl,curr_unmethyl,totalpop,param_arr)
        rates[3] = demethylation_rate_collaborative(curr_methyl,curr_unmethyl,totalpop,param_arr)
        rates[4] = birth_rate(param_arr)
        rate_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/rate_sum)
        total_time = tau + total_time

        #normalize the rates to be within (0,1)
        normalized_rates= rates / rate_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for event_number in range(5):
            if uniform < normalized_rates[event_number] + sum_so_far:
                if event_number == 4:
                    methylated_arr[divisions], unmethylated_arr[divisions], time_arr[divisions] = curr_methyl, curr_unmethyl, total_time
                    divisions += 1
                curr_methyl, curr_unmethyl = events(curr_methyl, curr_unmethyl, totalpop,event_number,rng)
                break
            else:
                sum_so_far += normalized_rates[event_number]
            
        
    #we timed out - return a negative value to indicate that this isn't a normal run.
    return (methylated_arr[:divisions], unmethylated_arr[:divisions], time_arr[:divisions])
