import numpy as np
import switching_coordinates.gillespie_coordinate as jittedswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange


#-----------parameterization-----------
#edit desired parameters directly in the default dictionary
#define batch size - how many different runs should we average for each step? 
batch_size = 5000
#define length of trials in steps (default 1000) - they will usually stop earlier, this is more for allocating space
trial_max_length = 10000
#define starting population
totalpop = 100
methylatedpop = 15
unmethylatedpop = 75
#SwitchDirection - a simulation terminates when it reaches this state
SwitchDirection = 1 #1 -> mostly methylated, -1-> mostly unmethylated
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
                      #edit birth_rate directly in this dictionary
                      "birth_rate": 1.6         #12
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
    for i in range(batch_size):
        output_array[i],crossing_coordinates[i]  = jittedswitch.GillespieSwitchFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rng)
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
        output_array[i],crossing_coordinates[i]  = jittedswitch.GillespieSwitchFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rng)
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