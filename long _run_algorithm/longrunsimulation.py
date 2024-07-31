import numpy as np
import longrungillespie
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange


#-----------parameterization-----------
trial_max_length = 10000000
#define starting population
totalpop = 100
methylatedpop = 50
unmethylatedpop = 50
#SwitchDirection - a simulation terminates when it reaches this state
#-----------Rates Dictionary---------
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
                      "birth_rate": 1.6        #12
}

#This dictionary just matches each parameter to its place in the list.
default_indices = {
    "r_hm": 0,          
    "r_hm_m": 1,
    "r_hm_h": 2,
    "r_uh": 3,
    "r_uh_m": 4, 
    "r_uh_h": 5,
    "r_mh": 6,
    "r_mh_u": 7,
    "r_mh_h": 8,
    "r_hu": 9,
    "r_hu_u": 10,
    "r_hu_h": 11,
    "birth_rate": 12

}

parameter_labels = ["r_hm", "r_hm_m","r_hm_h", "r_uh", "r_uh_m", "r_uh_h", "r_mh", "r_mh_u", "r_mh_h", "r_hu", "r_hu_u", "r_hu_h", "birth_rate"]

default_arr = np.array([default_parameters[key] for key in parameter_labels])

#-----------simulation-----------
@numba.jit()
def main(rng):
    #this loop runs in parallel because it uses prange() instead of range() - keep this in mind when debugging it!
        methyl_time, unmethyl_time, middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop = longrungillespie.GillespieLongRunFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng)
        return methyl_time,unmethyl_time,middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop
    
#-----------setup-----------

generator = np.random.default_rng()

#-----------Call simulation-----------
methylated_time, unmethylated_time, time_in_middle, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop = main(generator)

print("Check that everything adds up:")
total_time = time_arr[-1]
print(f'Total time: {total_time}')
total_prop = methyl_cumulative_prop[-1] + unmethyl_cumulative_prop[-1] + sortamethyl_cumulative_prop[-1]
# print(f'Total prop: {total_prop}')
methylated_prop = methylated_time/total_time
unmethylated_prop = unmethylated_time/total_time
time_in_middle_prop = time_in_middle/total_time
proportions = [methylated_prop,unmethylated_prop,time_in_middle_prop]
labels = ['methylated_prop','unmethylated_prop','time_in_middle_prop']
print('Proportions:')
print(f"Methylated : {methylated_prop}, Unmethylated: {unmethylated_prop}, middle: {time_in_middle_prop}, middle (<30% unmethylated) {sortamethyl_cumulative_prop[-1]}")
print(f"Sum of proportions: {methylated_prop + unmethylated_prop + time_in_middle_prop + sortamethyl_cumulative_prop[-1]}")
print('Times:')
print(f"Methylated : {methylated_time}, Unmethylated: {unmethylated_time}, middle: {time_in_middle}, middle (<30% unmethylated) {total_time * sortamethyl_cumulative_prop[-1]}")
# simple_arr = np.zeros_like(methylated_arr)
# for i in range(len(methylated_arr)):
#         if methylated_arr[i] > 0.7*totalpop:
#             simple_arr[i] = 1
#         elif unmethylated_arr[i] > 0.7*totalpop:
#             simple_arr[i] = -1

# print(xes)
# print(simple_arr)
# plt.plot(xes, simple_arr,ls='',marker=',')
# plt.show()
xes = list(range(trial_max_length//100))
methyl_cumulative_prop_thinned = np.zeros(trial_max_length//100)
unmethyl_cumulative_prop_thinned = np.zeros(trial_max_length//100)
sortamethyl_cumulative_prop_thinned = np.zeros(trial_max_length//100)
middle_cumulative_prop_thinned = np.zeros(trial_max_length//100)
for i in range(trial_max_length//100):
      methyl_cumulative_prop_thinned[i] = methyl_cumulative_prop[i*100]
      unmethyl_cumulative_prop_thinned[i] = unmethyl_cumulative_prop[i*100]
      sortamethyl_cumulative_prop_thinned[i] = sortamethyl_cumulative_prop[i*100]
      middle_cumulative_prop_thinned[i] = 1 - (methyl_cumulative_prop_thinned[i] + unmethyl_cumulative_prop_thinned[i] + sortamethyl_cumulative_prop_thinned[i])


plt.plot(xes, methyl_cumulative_prop_thinned,label="Methylated")
plt.plot(xes, unmethyl_cumulative_prop_thinned,label="Unmethylated")
plt.plot(xes, sortamethyl_cumulative_prop_thinned,label="Sort of methylated")
plt.plot(xes, middle_cumulative_prop_thinned,label="Transitionary")
plt.legend(loc='upper right')
plt.show()
      
        