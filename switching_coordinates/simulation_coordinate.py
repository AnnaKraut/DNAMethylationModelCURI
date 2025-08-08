import numpy as np
import gillespie_coordinate
import matplotlib.pyplot as plt
import numba
import statistics
from numba import prange

"""
switching_coordinates: performs many gillespie runs at once to find the average amount of methylation and unmethylation when switches happen. 

Edit the birth rate directly in the dictionary (on line 37) in this file.

If you want histograms showing more detailed distributions, find the code blocks that say `edit here` and uncomment the graphing code.
This will make a histogram for each switching direction in addition to the single, default graph.
"""

# -----------parameters-----------
# user should enter begin, end, step for the parameter they want to change.
param_to_change = "birth_rate"
param_begin_val = 0
param_end_val = 3
step_count = 49                         # number of evenly spaced parameter points to check

batch_size = 1000                       # number of simulations ran per parameter
trial_max_length = 5000000              # maximum length of trials (in simulation steps)

totalpop = 100                          # number of CpG sites to simulate

# initial conditions
methylatedpop = 15
unmethylatedpop = 75

#SwitchDirection - a simulation terminates when it reaches this state
SwitchDirection = 1 #1 -> mostly methylated, -1-> mostly unmethylated

#-----------Rates Dictionary---------
default_parameters = {"r_hm": 8.1734e+00,          #0
                      "r_hm_m": 2.0121e+00 * 2, #1
                      "r_hm_h": 2.0121e+00, #2
                      "r_uh": 3.9000e-03,         #3
                      "r_uh_m": 2.0000e-04 * 2,#4
                      "r_uh_h": 2.0000e-04,#5
                      "r_mh": 1.7010e-01,           #6
                      "r_mh_u": 4.7000e-03 * 2, #7
                      "r_mh_h": 4.7000e-03,  #8
                      "r_hu": 3.4970e-01,            #9
                      "r_hu_u": 6.6000e-03 * 2, #10
                      "r_hu_h": 6.6000e-03,   #11
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

parameter_labels = ["r_hm", "r_hm_m","r_hm_h", "r_uh", "r_uh_m", "r_uh_h", "r_mh", "r_mh_u", "r_mh_h", "r_hu", "r_hu_u", "r_hu_h", "birth_rate"]

#this line creates a numpy array with the same values as the dictionary - it is VITAL that they stay in the same order!!
#changing the order of either the labels or the stuff in this list will create subtle errors in the rate calculations!
default_arr = np.array([default_parameters[key] for key in parameter_labels])

#-----------simulation - unmethylated to methylated-----------
@numba.jit(nopython=True, parallel=True)
def main(rng):
    output_array = np.zeros(batch_size)
    crossing_coordinates = [(-1,-1)] * batch_size

    #run a batch of identical gillespie algorithms, store the results in output_array[step]
    for i in prange(batch_size):
        output_array[i],crossing_coordinates[i]  = gillespie_coordinate.GillespieSwitchFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rng)
    return output_array,crossing_coordinates

generator = np.random.default_rng()

#-----------Call simulation-----------
output,crossing_coordinates = main(generator)

#-----------Process results#----------
#filter out all timed-out runs and their coordinates
methyl_tuple_output = [(output[index],crossing_coordinates[index]) for index in range(batch_size) if output[index] >= 0]
#unpack the valid pairs into times and coordinates
methyl_valid_times, methyl_valid_coordinates = zip(*methyl_tuple_output)
#unpack the coordinates into x and y arrays
methyl_xcoords, methyl_ycoords = zip(*methyl_valid_coordinates)

#if histograms are desired, uncomment the following code
#-------edit here-------
# plt.close()
# plt.title("U->M histogram")
# plt.hist(methyl_ycoords)
# plt.show()

#-----------switch parameters-----------
SwitchDirection = -1
temp = methylatedpop
methylatedpop = unmethylatedpop
unmethylatedpop = temp

#-----------simulation - methylated to unmethylated-----------
@numba.jit(nopython=True, parallel=True)
def main(rng):
    output_array = np.zeros(batch_size)
    crossing_coordinates = [(-1,-1)] * batch_size

    #run a batch of identical gillespie algorithms, store the results in output_array[step]
    for i in range(batch_size):
        output_array[i],crossing_coordinates[i]  = gillespie_coordinate.GillespieSwitchFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rng)
    return output_array,crossing_coordinates

generator = np.random.default_rng()

#-----------Call simulation-----------
output,crossing_coordinates = main(generator)

#-----------Process results-----------
#filter out all timed-out runs and their coordinates
unmethyl_tuple_output = [(output[index],crossing_coordinates[index]) for index in range(batch_size) if output[index] >= 0]
#unpack the valid pairs into times and coordinates
unmethyl_valid_times, unmethyl_valid_coordinates = zip(*unmethyl_tuple_output)
#unpack the coordinates into x and y arrays
unmethyl_xcoords, unmethyl_ycoords = zip(*unmethyl_valid_coordinates)

#if histograms are desired, uncomment the following code
#-------edit here-------
# plt.close()
# plt.title("M->U histogram")
# plt.hist(unmethyl_xcoords)
# plt.show()

plt.rcParams["figure.autolayout"] = True
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
plt.title(f"""Mean of methylated side: ({statistics.mean(methyl_xcoords):.2f},{statistics.mean(methyl_ycoords):.2f}),
        Mean of unmethylated side: ({statistics.mean(unmethyl_xcoords):.2f},{statistics.mean(unmethyl_ycoords):.2f})\n
        Start condition = {methylatedpop}/{unmethylatedpop}, Birth rate = {default_parameters['birth_rate']}""")

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.grid(True)
#plot scatter and mean for the U->M transition
ax.scatter(methyl_xcoords,methyl_ycoords,linestyle='',marker='.')
ax.plot(statistics.mean(methyl_xcoords),statistics.mean(methyl_ycoords),'ro')

#plot scatter and mean for the M->U transition
ax.scatter(unmethyl_xcoords,unmethyl_ycoords,linestyle='',marker='.')
ax.plot(statistics.mean(unmethyl_xcoords),statistics.mean(unmethyl_ycoords),'ro')


ax.plot([100, 0],[0, 100], label='Boundary of Triangle')
plt.show()

count_str = 'Methylated' if SwitchDirection == -1 else 'Unmethylated'
direction_str = ' M-> U' if SwitchDirection == -1 else ' U -> M'
plt.title('Proportion of ' + count_str + ' when switching from' + direction_str)