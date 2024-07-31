import numpy as np
import optimized_algorithm.jittedswitch as jittedswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange


#-----------parameterization-----------
#TODO: add easier ways for users to input data
#user should enter begin, end, step for the parameter they want to change.
param_begin_val = 0.5
param_end_val = 0.5
step_count = 1
# define a parameter to vary - must be in the parameters dictionary - this should probably be selectable on command line
param_to_change = "birth_rate"
#define batch size - how many different runs should we average for each step? 
batch_size = 5000
#define length of trials in steps (default 1000) - they will usually stop earlier, this is more for allocating space
trial_max_length = 10000
#define starting population
totalpop = 100
methylatedpop = 90
unmethylatedpop = 10
#SwitchDirection - a simulation terminates when it reaches this state
SwitchDirection = -1 #1 -> mostly methylated, -1-> mostly unmethylated
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
def main(rngs):
    output_array = np.zeros(shape=(step_count, batch_size))
    #this loop runs in parallel because it uses prange() instead of range() - keep this in mind when debugging it!
    for step in prange(step_count):
        #make a copy of the default parameters, change the parameter we want to study
        temp_arr = default_arr.copy()
        temp_arr[index_to_change] = param_begin_val + (step*step_size)
        print("Testing parameters: ", temp_arr)

        #run a batch of identical gillespie algorithms, store the results in output_array[step]
        for i in range(batch_size):
            output_array[step][i] = jittedswitch.GillespieSwitchFun(trial_max_length, temp_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection,rngs[step])
    return output_array

        # plt.close() #ensure the previous graph is done
        # plt.hist(valid_array, bins=200)
        # #generate strings, we will concatenate these into a single title string for the graphs
        # param_string = "parameter: " + str(param_to_change) +  " = " + str(step_array[step]) + " -> Exponential Parameter = " + str(exponential_parameters[step])
        # step_string = "Step " + str(step+1) + "/" + str(step_count)
        # batch_string = "Batch of " + str(batch_size) + ", running for " + str(trial_max_length) + " steps each " + str(batch_size-valid_size) + " failed to finish"
        # plt.title(param_string + "\n" + step_string + "\n" + batch_string)
        # plt.savefig("histograms/" + str(time.perf_counter()) + "with" + str(batch_size) + "of" + str(trial_max_length) + '.png')
        # plt.show()
        # plt.close()
    
#-----------setup-----------

#generate the arrays for our output - None (or null value) is the default
exponential_parameters = [None] * step_count
exponential_KS = [None] * step_count

gamma_shape = [None] * step_count
gamma_location = [None] * step_count
gamma_scale = [None] * step_count
gamma_KS = [None] * step_count
inverse_gamma_scale = [None] * step_count

normal_KS = [None] * step_count
normal_sd = [None] * step_count
normal_mean = [None] * step_count

timeouts = [0] * step_count
empirical_mean = [None] * step_count

line = [None] * step_count


#list comprehension that creates an array of the values we tested for our chosen parameter
step_array = [step_size * i for i in range(step_count)]

#create an array of random number generators that we will pass into our function
#this makes it easier to reproduce, and also keeps Numba happy.
generators = [None]*step_count
for i in range(step_count):
    generators[i] = np.random.default_rng()

#-----------Call simulation-----------
output = main(generators)

#-----------postprocessing-----------

#go through the output row-by-row and find the exponential parameters
for step in range(step_count):
    #this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [output[step][index] for index in range(batch_size) if output[step][index] >= 0]
    #this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts[step] = 10*(raw_timeouts/batch_size) #scale the timeouts to fit with the other info on the graph

    #create a line representing the parameter we are varying on the y axis
    line[step] = step_array[step]


    #guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size/2:
        #fit distributions to the data
        exponential_parameters[step] = stats.expon.fit(valid_array,floc=0)[1]
        print('exponential paramater = ' + str(exponential_parameters[step]))

        gamma_shape[step],gamma_location[step],gamma_scale[step]=stats.gamma.fit(valid_array,floc=0)
        inverse_gamma_scale[step] = 1/gamma_scale[step]

        normal_mean[step], normal_sd[step] =  stats.norm.fit(valid_array,)
        print(f'Normal mean is {normal_mean[step]} and S.D. is {normal_sd[step]}')

        empirical_mean[step] = statistics.fmean(valid_array)

        #calculate error for parameters with Kolmogorov-Smirnov test
        #note that we lock the first argument, location, to 0 for the exponential distribution
        exponential_KS[step] = 10 * (stats.kstest(valid_array, 'expon', N=len(valid_array), args=(0,exponential_parameters[step])).statistic)
        print(exponential_KS[step])
        # normal_KS[step] = 10 * (stats.kstest(valid_array, 'norm', N=len(valid_array), args=(normal_mean[step], normal_sd[step])).statistic)
        # print(normal_KS[step])
        gamma_KS[step] = 10 * (stats.kstest(valid_array, stats.gamma.cdf, N=len(valid_array), args=(gamma_shape[step],0,gamma_scale[step])).statistic)
        print(gamma_KS[step])

    # print("predicted exponential parameter: ", exponential_parameters[step])
    # print("predicted gamma shape parameter: ", gamma_shape[step])
    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))

#-----------graphing-----------


#convert all these to f strings
plt.close()
final_label = "Switching times from unmethylated to methylated as birth rate changes \n Population = " + str(totalpop)
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"
plt.plot(step_array, exponential_parameters,label="exponential parameters", linestyle='dashed')
plt.plot(step_array,timeouts, label = "proportion timed out, scaled by 10x")
plt.plot(step_array, exponential_KS, label="Exponential KS error, scaled by 10x")

# plt.plot(step_array, gamma_shape,label="Gamma shape")
# plt.plot(step_array, gamma_location,label="Gamma location")
# plt.plot(step_array, gamma_scale,label="Gamma scale")
# plt.plot(step_array, inverse_gamma_scale,label = "1/Gamma scale",linestyle='dashed')
# plt.plot(step_array, gamma_KS, label="Gamma KS error, scaled by 10x")
plt.plot(step_array, line, linestyle='dotted', label = 'Birth Rate')

# plt.plot(step_array, normal_mean, label='Normal mean',marker='.',linestyle='')
# plt.plot(step_array, normal_sd, label='Normal S.D.',marker='.',linestyle='')
# plt.plot(step_array, normal_KS, label="Normal KS error, scaled by 10x",marker='.',linestyle='')
# plt.plot(step_array, empirical_mean, label='Empirical Mean', linestyle='dashed')

plt.title(final_label + "\n" + run_stats)
plt.xlabel('Value of parameter '+ param_to_change)
plt.ylabel('Exponential parameter of switching time distribution')
plt.legend(loc='upper right')
plt.show()