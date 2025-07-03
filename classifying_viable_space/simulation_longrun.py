import numpy as np
import gillespie_longrun
import matplotlib.pyplot as plt
import numba

# This is the file I am currently using to run tests with converging by RMSD (Root Mean Squared Deviance)

"""
Performs a single, very long, gillespie run to see what proportion of time is spent in each state - 
(methylated, unmethylated, neither, or sort-of methylated).
"sort-of methylated" refers to a state that is less than 30% unmethylated - in other words,
it is a state where 70% of the sites are either methylated or hemimethylated.

There are no alternate output options for this program - just edit the parameters and run it to get a graph.
"""


#-----------parameters - edit here-----------
#number of steps that the gillespie algorithm will take (large values can cause memory issues, starting around 1,000,000,000)
trial_max_length = 100000000
#define starting population
totalpop = 100
methylatedpop = 50
unmethylatedpop = 50
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
                      
                      #adjust birth rate directly - edit here
                      "birth_rate": 1     #12
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
        methyl_time, unmethyl_time, middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop, rmsd_arr = gillespie_longrun.GillespieLongRunFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng)
        return methyl_time,unmethyl_time,middle_time, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop, rmsd_arr
    
#-----------setup-----------

#create a random number generator - this generator can be seeded if desired
generator = np.random.default_rng()

#-----------Call simulation-----------
#call our gillespie algorithm and save the raw data
methylated_time, unmethylated_time, time_in_middle, time_arr, methyl_cumulative_prop, unmethyl_cumulative_prop, sortamethyl_cumulative_prop, rmsd_arr = main(generator)
total_steps = len(time_arr)

#print the amount of time that our simulation lasted
total_time = time_arr[-1]

#calculate the proportion of time that we spent in each state
total_prop = methyl_cumulative_prop[-1] + unmethyl_cumulative_prop[-1] + sortamethyl_cumulative_prop[-1]
methylated_prop = methylated_time/total_time
unmethylated_prop = unmethylated_time/total_time
time_in_middle_prop = time_in_middle/total_time
proportions = [methylated_prop,unmethylated_prop,time_in_middle_prop]
labels = ['methylated_prop','unmethylated_prop','time_in_middle_prop']

#thin out our data by saving only every 100th observation - this makes it easier to graph
xes = list(range(total_steps//100)) 

#create the arrays that we will use
methyl_cumulative_prop_thinned = np.zeros(total_steps//100)
unmethyl_cumulative_prop_thinned = np.zeros(total_steps//100)
sortamethyl_cumulative_prop_thinned = np.zeros(total_steps//100)
middle_cumulative_prop_thinned = np.zeros(total_steps//100)
#populate the arrays by picking every 100th number
for i in range(total_steps//100):
      methyl_cumulative_prop_thinned[i] = methyl_cumulative_prop[i*100]
      unmethyl_cumulative_prop_thinned[i] = unmethyl_cumulative_prop[i*100]
      sortamethyl_cumulative_prop_thinned[i] = sortamethyl_cumulative_prop[i*100]
      middle_cumulative_prop_thinned[i] = 1 - (methyl_cumulative_prop_thinned[i] + unmethyl_cumulative_prop_thinned[i] + sortamethyl_cumulative_prop_thinned[i])

#plot our results
plt.subplot(2,1,1)
plt.title(f'Methylated : {methylated_prop:.3f}, Unmethylated: {unmethylated_prop:.3f},\n middle: {time_in_middle_prop:.3f}, middle (<30% unmethylated) {sortamethyl_cumulative_prop[-1]:.3f} \n simulated with {totalpop} sites over {total_steps} iterations',fontsize=10)
plt.xlabel('x-axis samples every hundredth point to improve readability')
plt.ylabel('cumulative proportion of time spent')
plt.plot(xes, methyl_cumulative_prop_thinned,label="Methylated")
plt.plot(xes, unmethyl_cumulative_prop_thinned,label="Unmethylated")
plt.plot(xes, sortamethyl_cumulative_prop_thinned,label="Sort of methylated")
plt.plot(xes, middle_cumulative_prop_thinned,label="Transitionary")
plt.legend(loc='upper right')

plt.subplot(2,1,2)
plt.plot(rmsd_arr)

plt.show()
      
        