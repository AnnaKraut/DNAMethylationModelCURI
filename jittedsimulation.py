import numpy as np
import jittedswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
import numba
from numba import jit
from numba import njit
from numba import types
from numba.typed import Dict
#-----------parameterization-----------
#TODO: add easier ways for users to input data
#user should enter begin, end, step for the parameter they want to change.
param_begin_val = 0.5
param_end_val = 2
param_step_size = 0.05
# define a parameter to vary - must be in the parameters dictionary - this should probably be selectable on command line
param_to_change = "birth_rate"
#define batch size - how many different runs should we average for each step? (default 10 for testing, should increase)
batch_size = 5
#define length of trials (default 1000) - they will usually stop earlier, this is more for allocating space
trial_max_length = 10000
#define starting population
totalpop = 100
methylatedpop = 90
unmethylatedpop = 10
#SwitchDirection - a simulation terminates when it reaches this state
SwitchDirection = -1 #1 -> mostly methylated, -1-> mostly unmethylated
#set the debug toggle
debug = False
batch_debug = False
FinishAndSave = False
#-----------Rates Dictionary---------
default_parameters = Dict.empty(
    key_type=types.unicode_type,
    value_type=types.float64
)
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

parameter_labels = ["r_hm", "r_hm_m","r_hm_h", "r_uh", "r_uh_m", "r_uh_h", "r_mh", "r_mh_u", "r_mh_h", "r_hu", "r_hu_u", "r_hu_h", "birth_rate"]

#-----------setup-----------
#generate an array of parameters to test, between the values that the user specified
steps_to_test = np.arange(param_begin_val, param_end_val, param_step_size)
step_count = len(steps_to_test)

#generate the arrays for our output
output_array = []
exponential_parameters = [None] * step_count
timeouts = [0] * step_count

for i in range(step_count):
    output_array.append(np.zeros(batch_size))
#-----------simulation-----------

#break all of the computation into a seperate, jittable function? 
#preprocessing step to identify which index in the dictionary we are changing
#feed in a list, a step size and step amount. Then, for each batch, compute the new step (from step size * index) and change the input list. 
#(DO NOT do a +=, calculate from step size * index) this way we avoid needing dictionaries and can parallelize.

#call batches in parallel with "@numba.jit(parallel=True)"
#call things within a batch in parallel too maybe> just for fun
#actually - if we call batches in parallel, we don't need to call things within a batch in parallel, meaning we can implement anna's idea to kill runs with high timeout rates
#run things within a batch sequentially, and if we compute with a specific confidence that the batch has high timeouts, we kill the batch
#My laptop only has 10 cores for example, and even clusters probably wont have over a hundred, so likely batch-level parallelism is fine.

#each batch outputs into an array, which is indexed into a bigger array (DO NOT APPEND - THIS CAUSES ISSUES WITH PARALLELIZING)
#compute should return one massive array, which is post-processed by another function as needed
#maybe later we can try post-processing inside the loop, which would save space, but potentially cause parallelization issues.

# @numba.jit(parallel=True)
def main():
    for step in range(len(steps_to_test)):
        valid_size = 0
        #make a copy of the default parameters, change the parameter we want to study
        input_dict = default_parameters.copy() # shallow copy
        input_dict[param_to_change] = steps_to_test[step]

        #IN THE THE JITTED VERSION, WE NEED TO CONVERT THE INPUT DICTIONARY TO A LIST BEFORE PASSING IT INTO THE SWITCHING ALGORITHM 
        input_arr = [input_dict[key] for key in parameter_labels]


        #run a batch of identical gillespie algorithms, store the results in output_array[step]
        for i in range(batch_size):
            output_array[step][i] = jittedswitch.GillespieSwitchFun(trial_max_length, input_arr, totalpop, methylatedpop, unmethylatedpop, SwitchDirection)
            if output_array[step][i] >= 0: #this result was valid
                valid_size += 1

        #valid_array contains all the simulations that did NOT time-out
        valid_index = 0
        valid_array = np.zeros(valid_size)
        for j in range(batch_size):
            if output_array[step][j] >= 0:
                valid_array[valid_index] = output_array[step][j]
                valid_index += 1

        #only guess parameter if more than half of the runs finished
        if valid_size > batch_size/2:
            exponential_parameters[step] = stats.expon.fit(valid_array,floc=0)[1]
        else: #otherwise, mark the parameter as negative, meaning its a no go
            exponential_parameters[step] = None

        print("predicted exponential parameter: ", exponential_parameters[step])
        print("timed-out simulations: " + str(batch_size-valid_size) + " out of " + str(batch_size))
        timeouts[step] = batch_size-valid_size

        #save a graph of batch results if toggle is enabled
        if batch_debug:
            plt.close() #ensure the previous graph is done
            plt.hist(valid_array, bins=20)

            #generate strings, we will concatenate these into a single title string for the graphs
            param_string = "parameter: " + str(param_to_change) +  " = " + str(steps_to_test[step]) + " -> Exponential Parameter = " + str(exponential_parameters[step])
            step_string = "Step " + str(step+1) + "/" + str(step_count)
            batch_string = "Batch of " + str(batch_size) + ", running for " + str(trial_max_length) + " steps each " + str(batch_size-valid_size) + " failed to finish"
            plt.title(param_string + "\n" + step_string + "\n" + batch_string)
            plt.savefig("histograms/" + str(time.perf_counter()) + "with" + str(batch_size) + "of" + str(trial_max_length) + '.png')
            plt.close()
#-----------analysis-----------
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
main()