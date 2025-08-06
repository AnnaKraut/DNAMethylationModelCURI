import numpy as np
import gillespie_longrun as gillespie_longrun
import matplotlib.pyplot as plt
import numba

"""
Performs a single, very long, gillespie run to see what proportion of time is spent in each state - 
(methylated, unmethylated, neither, or sort-of methylated).
"sort-of methylated" refers to a state that is less than 30% unmethylated - in other words,
it is a state where 70% of the sites are either methylated or hemimethylated.

There are no alternate output options for this program - just edit the parameters and run it to get a graph.
"""


#-----------parameters - edit here-----------
#number of steps that the gillespie algorithm will take (large values can cause memory issues, starting around 1,000,000,000)
trial_max_length = 500000000
#define starting population
totalpop = 100
methylatedpop = 50
unmethylatedpop = 50
#-----------Rates Dictionary---------

default_parameters = {
    "r_hm": 4.8790e-01,
    "r_hm_m": 1.9257e00 * 2,
    "r_hm_h": 1.9257e00,
    "r_uh": 9.0000e-04,
    "r_uh_m": 2.0000e-04 * 2,
    "r_uh_h": 2.0000e-04,
    "r_mh": 8.5300e-02,
    "r_mh_u": 1.3000e-03 * 2,
    "r_mh_h": 1.3000e-03,
    "r_hu": 2.5620e-01,
    "r_hu_u": 2.8000e-03 * 2,
    "r_hu_h": 2.8000e-03,
    "birth_rate": 1,
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

#these labels allow us to convert the conviently-labelled dictionary into a more efficient array
parameter_labels = ["r_hm", "r_hm_m","r_hm_h", "r_uh", "r_uh_m", "r_uh_h", "r_mh", "r_mh_u", "r_mh_h", "r_hu", "r_hu_u", "r_hu_h", "birth_rate"]
default_arr = np.array([default_parameters[key] for key in parameter_labels])

#-----------simulation-----------
@numba.jit()
def main(rng):
        methyl_time, unmethyl_time, middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop = gillespie_longrun.GillespieLongRunFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng)
        return methyl_time,unmethyl_time,middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop
    
#-----------setup-----------

#create a random number generator - this generator can be seeded if desired
generator = np.random.default_rng()

#-----------Call simulation-----------
#call our gillespie algorithm and save the raw data
methylated_time, unmethylated_time, time_in_middle, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop = main(generator)

#print the amount of time that our simulation lasted
total_time = time_arr[-1]
print(f'Check that everything adds up: \nTotal time: {total_time}')

#calculate the proportion of time that we spent in each state
total_prop = methyl_cumulative_prop[-1] + unmethyl_cumulative_prop[-1] + sortamethyl_cumulative_prop[-1]
methylated_prop = methylated_time/total_time
unmethylated_prop = unmethylated_time/total_time
time_in_middle_prop = time_in_middle/total_time

#print out the proportions of time that we spent in each state
print('Proportions:')
print(f"Methylated : {methylated_prop}, Unmethylated: {unmethylated_prop}, middle: {time_in_middle_prop}, middle (<30% unmethylated) {sortamethyl_cumulative_prop[-1]}")
print(f"Sum of proportions: {methylated_prop + unmethylated_prop + time_in_middle_prop + sortamethyl_cumulative_prop[-1]}")
print('Times:')
print(f"Methylated : {methylated_time}, Unmethylated: {unmethylated_time}, middle: {time_in_middle}, middle (<30% unmethylated) {total_time * sortamethyl_cumulative_prop[-1]}")

#thin out our data by saving only every 1000th observation - this makes it easier to graph
xes = np.zeros(trial_max_length//1000) 
methyl_cumulative_prop_thinned = np.zeros(trial_max_length//1000)
unmethyl_cumulative_prop_thinned = np.zeros(trial_max_length//1000)
sortamethyl_cumulative_prop_thinned = np.zeros(trial_max_length//1000)
middle_cumulative_prop_thinned = np.zeros(trial_max_length//1000)

#populate the arrays by picking every 1000th number
for i in range(trial_max_length//1000):
      methyl_cumulative_prop_thinned[i] = methyl_cumulative_prop[i*1000]
      unmethyl_cumulative_prop_thinned[i] = unmethyl_cumulative_prop[i*1000]
      sortamethyl_cumulative_prop_thinned[i] = sortamethyl_cumulative_prop[i*1000]
      middle_cumulative_prop_thinned[i] = 1 - (methyl_cumulative_prop_thinned[i] + unmethyl_cumulative_prop_thinned[i] + sortamethyl_cumulative_prop_thinned[i])
      xes[i] = i * 1000

#plot our results

plt.rcParams['font.size'] = 20
# plt.title(f'Methylated : {methylated_prop:.3f}, Unmethylated: {unmethylated_prop:.3f},\n middle: {time_in_middle_prop:.3f}, middle (<30% unmethylated) {sortamethyl_cumulative_prop[-1]:.3f} \n simulated with {totalpop} sites over {trial_max_length} iterations')
plt.xlabel('Simulation steps (sampled every 1000 points)')
plt.ylabel('Cumulative proportion of time spent')
plt.plot(xes, methyl_cumulative_prop_thinned,label="Hypermethylated",color="#D55E00",linewidth=2.0)
plt.plot(xes, unmethyl_cumulative_prop_thinned,label="Hypomethylated",color="#0072B2",linewidth=2.0)
plt.plot(xes, middle_cumulative_prop_thinned,label="Transitionary",color="#CC79A7",linewidth=2.0)
plt.legend(loc='upper right')
plt.show()
      
        