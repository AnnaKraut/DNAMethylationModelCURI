import numpy as np
import gillespieswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
import time
#TODO: just for fun, in the timing layer, output the amount of data processed to an external file
#so that we can track how much data the simulator has processed

#use this to prettify the graphs later (red for DNFs)
#https://stackoverflow.com/questions/26726100/plotting-multiple-segments-with-colors-based-on-some-variable-with-matplotlib?noredirect=1&lq=1

#TODO: use 90/10 splits, play with birth rate
#measure switch at 70/30
#measure methylated to unmethylated and vice versa seperately


#-----------parameterization-----------
#TODO: add easier ways for users to input data
#user should enter begin, end, step for the parameter they want to change.
param_begin_val = 0.6 
param_end_val = 2
param_step_size = 0.05
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
#set the debug toggle
debug = False
batch_debug = False
FinishAndSave = False
#-----------Rates Dictionary---------
default_parameters = {"r_hm": 0.5,#changed this - was 0.5
                      "r_hm_m": 20/totalpop, #changed - was 20
                      "r_hm_h": 10/totalpop,
                      "r_uh": 0.35,#cahgned this - was 0.35
                      "r_uh_m": 11/totalpop,#changed - was 11
                      "r_uh_h": 5.5/totalpop,
                      "r_mh": 0.1,
                      "r_mh_u": 10/totalpop,
                      "r_mh_h": 5/totalpop,
                      "r_hu": 0.1,#changed this - was 0.1
                      "r_hu_u": 10/totalpop, #changed - was 10
                      "r_hu_h": 5/totalpop, #changed - was 5
                      "birth_rate": 1
}

#-----------setup-----------
#generate an array of parameters to test, between the values that the user specified
steps_to_test = np.arange(param_begin_val, param_end_val, param_step_size)
step_count = len(steps_to_test)

#generate the arrays for our output
output_array = []
exponential_parameters = [None] * step_count

for i in range(step_count):
    output_array.append(np.zeros(batch_size))
#-----------simulation-----------

for step in range(len(steps_to_test)):
    valid_size = 0
    #make a copy of the default parameters, change the parameter we want to study
    input_dict = default_parameters.copy() # shallow copy
    input_dict[param_to_change] = steps_to_test[step]
    #run a batch of identical gillespie algorithms, store the results in output_array[step]
    for i in range(batch_size):
        # if i == 0:
        #     FinishAndSave = True
        # else:
        #     FinishAndSave = False
        output_array[step][i] = gillespieswitch.GillespieModelSwitchTime(trial_max_length,input_dict, totalpop, methylatedpop, unmethylatedpop, SwitchDirection, debug, FinishAndSave).main()
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
        exponential_parameters[step] = -1

    print("predicted exponential parameter: ", exponential_parameters[step])
    print("timed-out simulations: " + str(batch_size-valid_size) + " out of " + str(batch_size))

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
plt.plot(steps_to_test, exponential_parameters)
plt.title(final_label + "\n" + run_stats)
plt.xlabel('Value of parameter '+ param_to_change)
plt.ylabel('Exponential parameter of switching time distribution')
plt.show()