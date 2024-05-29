import numpy as np
import jittedswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
from numba import prange

#-----------parameterization-----------
#TODO: add easier ways for users to input data
#user should enter begin, end, step for the parameter they want to change.
param_begin_val = 0.5
param_end_val = 3
step_count = 50
# define a parameter to vary - must be in the parameters dictionary - this should probably be selectable on command line
param_to_change = "birth_rate"
#define batch size - how many different runs should we average for each step? (default 10 for testing, should increase)
batch_size = 5000
#define length of trials (default 1000) - they will usually stop earlier, this is more for allocating space
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
steps_to_test = step_count

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
        # param_string = "parameter: " + str(param_to_change) +  " = " + str(steps_to_test[step]) + " -> Exponential Parameter = " + str(exponential_parameters[step])
        # step_string = "Step " + str(step+1) + "/" + str(step_count)
        # batch_string = "Batch of " + str(batch_size) + ", running for " + str(trial_max_length) + " steps each " + str(batch_size-valid_size) + " failed to finish"
        # plt.title(param_string + "\n" + step_string + "\n" + batch_string)
        # plt.savefig("histograms/" + str(time.perf_counter()) + "with" + str(batch_size) + "of" + str(trial_max_length) + '.png')
        # plt.show()
        # plt.close()
    
#-----------setup-----------

#generate the arrays for our output
exponential_parameters = [None] * step_count
timeouts = [0] * step_count
#list comprehension that creates an array of the values we tested for our chosen parameter
steps_to_test = [step_size * i for i in range(step_count)]

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
    #only guess parameter if more than half of the runs finished
    if len(valid_array) > batch_size/2:
        exponential_parameters[step] = stats.expon.fit(valid_array,floc=0)[1]
    else: #otherwise, mark the parameter as negative, meaning its a no go
        exponential_parameters[step] = -1

    print("predicted exponential parameter: ", exponential_parameters[step])
    print("timed-out simulations: " + str(batch_size-len(valid_array)) + " out of " + str(batch_size))
    timeouts[step] = batch_size-len(valid_array)

plt.close()
final_label = "Switching times from methylated to unmethylated as birth rate changes \n Population = 100"
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"
plt.plot(steps_to_test, exponential_parameters,label="exponential parameters")
# plt.plot(steps_to_test,timeouts, label = "timed out simulations")
plt.title(final_label + "\n" + run_stats)
plt.xlabel('Value of parameter '+ param_to_change)
plt.ylabel('Exponential parameter of switching time distribution')
plt.legend(loc='upper right')
plt.show()