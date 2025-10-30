import numpy as np
import state_sim as gillespie_longrun
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
trial_max_length = 10000
#define starting population
totalpop = 100
methylatedpop = 6
unmethylatedpop = 73
#-----------Rates Dictionary---------

default_parameters = {"r_hm": 18.6735,          #0
                      "r_hm_m": 0.3747 * 2, #1
                      "r_hm_h": 0.3747, #2
                      "r_uh": 0.0006,         #3
                      "r_uh_m": 0.0004 * 2,#4
                      "r_uh_h": 0.0004,#5
                      "r_mh": 0.2461,           #6
                      "r_mh_u": 0.0014 * 2, #7
                      "r_mh_h": 0.0014,  #8
                      "r_hu": 0.2828,            #9
                      "r_hu_u": 0.0030 * 2, #10
                      "r_hu_h": 0.0030,   #11
                      
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
def main(rng):
    return gillespie_longrun.GillespieLongRunFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng)
    
#-----------setup-----------

#create a random number generator - this generator can be seeded if desired
generator = np.random.default_rng()

#-----------Call simulation-----------
#call our gillespie algorithm and save the raw data
x_m_arr, x_u_arr, time_arr = main(generator)

#print the amount of time that our simulation lasted
total_time = time_arr[-1]
print(f'Check that everything adds up: \nTotal time: {total_time}')

#thin out our data by saving only every 100th observation - this makes it easier to graph
xes = list(range(trial_max_length//100)) 

#create the arrays that we will use
x_m_thinned = np.zeros(trial_max_length//100)
x_u_thinned = np.zeros(trial_max_length//100)
x_h_thinned = np.zeros(trial_max_length//100)
time_thinned = np.zeros(trial_max_length//100)

#populate the arrays by picking every 100th number
for i in range(trial_max_length//100):
      x_m_thinned[i] = x_m_arr[i*100]
      x_u_thinned[i] = x_u_arr[i*100]
      time_thinned[i] = time_arr[i*100]

x_h_thinned = totalpop - x_m_thinned - x_u_thinned

#plot our results
plt.title(f'Populations of CpG dyads over time by methylation level  \n simulated with {totalpop} sites over {trial_max_length} iterations',fontsize=10)
# plt.xlabel('x-axis samples every hundredth point to improve readability')
plt.ylabel('population')
plt.plot(time_thinned, x_m_thinned,label="Methylated")
plt.plot(time_thinned, x_u_thinned,label="Unmethylated")
# plt.plot(time_thinned, x_h_thinned,label="Hemimethylated")
plt.legend(loc='upper right')
plt.show()
      
        