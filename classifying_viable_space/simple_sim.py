import numpy as np
from numba import jit
from numba import njit
import matplotlib.pyplot as plt
import sys

"""
This file represents a long-run version of the gillespie algorithm. 

Goal: Get the proportion of time spent in a variety of regions for a specific parameter set

Input: steps, parameter list, target parameter, parameter start, parameter end, bin count
Output: wide variety of descriptive statistics

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
#2 means less than 30% methylated
@njit
def classify_state(methylated, unmethylated, site_count):
      if methylated > 0.7*site_count:
          return 1
      elif unmethylated > 0.7*site_count:
          return -1
      elif unmethylated < 0.3*site_count:
          return 2
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
def GillespieLongRunFun(steps, param_arr, totalpop, pop_methyl, pop_unmethyl, rng):
    #set the first elements of the methylated/unmethylated arrays to the starting values
    curr_methylated = pop_methyl 
    curr_unmethylated = pop_unmethyl
    #set the first element of the time array to zero, so that it stays synced up with the (un)methylated arrays
    total_time = 0
    rates = np.zeros(5)
    #define our four amounts of cumulative time spent in different areas. By the end these will sum to time_arr[-1]
    methyl_cumulative = 0
    unmethyl_cumulative = 0
    middle_cumulative = 0
    sortamethl_cumulative = 0

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl

        #find the rates of each event for the current parameters
        rates[0] = maintenance_rate_collaborative(curr_methylated,curr_unmethylated,totalpop,param_arr)
        rates[1] = denovo_rate_collaborative(curr_methylated,curr_unmethylated,totalpop,param_arr)
        rates[2] = demaintenance_rate_collaborative(curr_methylated,curr_unmethylated,totalpop,param_arr)
        rates[3] = demethylation_rate_collaborative(curr_methylated,curr_unmethylated,totalpop,param_arr)
        rates[4] = birth_rate(param_arr)
        rate_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/rate_sum)
        total_time += tau

        #calculate the time increment after calculating tau but BEFORE calculating the next step
        curr_state = classify_state(curr_methylated, curr_unmethylated, totalpop)
        if curr_state == 1:
            methyl_cumulative += tau
        elif curr_state == -1:
            unmethyl_cumulative += tau
        elif curr_state == 2: 
            sortamethl_cumulative += tau
        else:
            middle_cumulative += tau

        #normalize the rates to be within (0,1)
        normalized_rates= rates / rate_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for event_number in range(5):
            if uniform < normalized_rates[event_number] + sum_so_far:
                curr_methylated, curr_unmethylated = events(curr_methylated, curr_unmethylated, totalpop,event_number,rng)
                break
            else:
                sum_so_far += normalized_rates[event_number]


    #we should reach this return point on every run
    return (methyl_cumulative / total_time,
            unmethyl_cumulative / total_time,
            middle_cumulative / total_time,
            sortamethl_cumulative / total_time)
