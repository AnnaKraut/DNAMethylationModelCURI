import numpy as np
import switching_times.gillespie_time as gillespie_time
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange


#-----------parameterization-----------
#user should enter begin, end, step for the parameter they want to change.
param_begin_val = 0.7
param_end_val = 1.6
step_count = 7
# define a parameter to vary - must be in the parameters dictionary
param_to_change = "birth_rate"
#define batch size - how many different runs we average for each step
batch_size = 50000
#define length of trials in steps (default 1000) - they will usually stop earlier, this is more for allocating space
trial_max_length = 100000
#define starting population - the starting counts of methylated/unmethylated are further down in the file
totalpop = 100
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
                      "birth_rate": 1         #12
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
#This line just calculates where the parameter we want to change occurs in the list
index_to_change = default_indices[param_to_change]

#find the size of each step, rounded to 5 decimal places.
step_size = round((param_end_val-param_begin_val)/step_count, 5)
# step_array = step_count

#-----------simulation-----------
@numba.jit(nopython=True, parallel=True)
def main(rngs,SwitchDirection,methylatedpop, unmethylatedpop):
    output_array = np.zeros(shape=(step_count, batch_size))
    #this loop runs in parallel because it uses prange() instead of range() - keep this in mind when debugging it!
    for step in prange(step_count):
        #make a copy of the default parameters, change the parameter we want to study
        temp_arr = default_arr.copy()
        temp_arr[index_to_change] = param_begin_val + (step*step_size)
        print("Testing parameters: ", temp_arr)

        #run a batch of identical gillespie algorithms, store the results in output_array[step]
        for i in range(batch_size):
            output_array[step][i] = gillespie_time.GillespieSwitchFun(trial_max_length, temp_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rngs[step])
    return output_array
#-----------setup - METHYLATED TO UNMETHYLATED-----------
SwitchDirection = -1
#generate the arrays for our output - None (or null value) is the default
exponential_parameters_MtoU = [None] * step_count
gamma_shape_MtoU = [None] * step_count
gamma_location_MtoU = [None] * step_count
gamma_scale_MtoU = [None] * step_count
exponential_KS_MtoU = [None] * step_count
gamma_KS_MtoU = [None] * step_count
timeouts_MtoU = [0] * step_count
empirical_mean_MtoU = [None] * step_count
#list comprehension that creates an array of the values we tested for our chosen parameter
step_array = [step_size * i for i in range(step_count)]

#create an array of random number generators that we will pass into our function
#this makes it easier to reproduce, and also keeps Numba happy.
generators = [None]*step_count
for i in range(step_count):
    generators[i] = np.random.default_rng()

#-----------Call simulation-----------
methylatedpop = 71
unmethylatedpop = 13
output = main(generators,-1,methylatedpop, unmethylatedpop)

#-----------postprocessing-----------

#go through the output row-by-row and find the exponential parameters
for step in range(step_count):
    #this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [output[step][index] for index in range(batch_size) if output[step][index] >= 0]
    #this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts_MtoU[step] = 10*(raw_timeouts/batch_size) #scale the timeouts to fit with the other info on the graph

    #guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size/2:
        #fit distributions to the data
        exponential_parameters_MtoU[step] = stats.expon.fit(valid_array,floc=0)[1]

        empirical_mean_MtoU[step] = statistics.fmean(valid_array)

        #calculate error for parameters with Kolmogorov-Smirnov test
        #TODO: add args for expon
        exponential_KS_MtoU[step] = 10 * (stats.kstest(valid_array, 'expon', args=(0,exponential_parameters_MtoU[step]), N=len(valid_array)).statistic)
        print(exponential_KS_MtoU[step])
    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))
    print('exponential paramater MtoU = ' + str(exponential_parameters_MtoU[step]))


#-----------setup - UNMETHYLATED TO METHYLATED-----------
SwitchDirection = 1
methylatedpop = 4
unmethylatedpop = 72
#generate the arrays for our output - None (or null value) is the default
exponential_parameters_UtoM = [None] * step_count
gamma_shape_UtoM = [None] * step_count
gamma_location_UtoM = [None] * step_count
gamma_scale_UtoM = [None] * step_count
exponential_KS_UtoM = [None] * step_count
gamma_KS_UtoM = [None] * step_count
normal_KS_UtoM = [None] * step_count
normal_sd_UtoM = [None] * step_count
normal_mean_UtoM = [None] * step_count
timeouts_UtoM = [0] * step_count
empirical_mean_UtoM = [None] * step_count
#list comprehension that creates an array of the values we tested for our chosen parameter
#TODO: add offset of initial size
step_array = [step_size * i for i in range(step_count)]

output = main(generators,1,methylatedpop, unmethylatedpop)

for step in range(step_count):
    #this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [output[step][index] for index in range(batch_size) if output[step][index] >= 0]
    #this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts_UtoM[step] = 10*(raw_timeouts/batch_size) #scale the timeouts to fit with the other info on the graph

    #guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size/2:
        #fit distributions to the data
        exponential_parameters_UtoM[step] = stats.expon.fit(valid_array,floc=0)[1]

        empirical_mean_UtoM[step] = statistics.fmean(valid_array)

        #calculate error for parameters with Kolmogorov-Smirnov test
        #TODO: add args for expon
        exponential_KS_UtoM[step] = 10 * (stats.kstest(valid_array, 'expon', N=len(valid_array), args=(0,exponential_parameters_UtoM[step])).statistic)
        print(exponential_KS_MtoU[step])

    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))
    print('exponential paramater UtoM= ' + str(exponential_parameters_UtoM[step]))

#-----------graphing-----------

plt.close()
final_label = "Two-way switching directions with Population = 100"
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"

#MtoU
plt.plot(step_array, exponential_parameters_MtoU,label="exponential parameters")
plt.plot(step_array,timeouts_MtoU, label = "proportion timed out, scaled by 10x")
plt.plot(step_array, exponential_KS_MtoU, label="Exponential KS error, scaled by 10x")
# plt.plot(step_array, empirical_mean_MtoU, label='Empirical Mean',linestyle='none',marker='.')

#UtoM
plt.plot(step_array, exponential_parameters_UtoM,label="exponential parameters", linestyle='dashed')
plt.plot(step_array,timeouts_UtoM, label = "proportion timed out, scaled by 10x", linestyle='dashed')
plt.plot(step_array, exponential_KS_UtoM, label="Exponential KS error, scaled by 10x", linestyle='dashed')
# plt.plot(step_array, empirical_mean_UtoM, label='Empirical Mean', linestyle='none',marker='.')

plt.title(final_label + "\n" + run_stats + "\n" + "Solid: Hyper-to-Hypomethylated, dashed: Hypo-to-Hypermethylated")
plt.xlabel('Value of parameter '+ param_to_change)
plt.ylabel('Exponential parameter of switching time distribution')
plt.legend(loc='upper right')
plt.show()