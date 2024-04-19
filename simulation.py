import numpy as np
import gillespieswitch
import matplotlib.pyplot as plt
import scipy.stats as stats
#TODO: just for fun, in the timing layer, output the amount of data processed to an external file
#so that we can track how much data the simulator has processed

#define a dictionary with the parameters - these will be passed into the simulations and should be easily changable
#the dictionary might not play well with multithreading - may need to change this
#fixed number of parameters - could just be a list if needed - if so abstract that away from the user
default_parameters = {"r_hm": 0.5,
                      "r_hm_m": 20,
                      "r_hm_h": 10,
                      "r_uh": 0.35,
                      "r_uh_m": 11,
                      "r_uh_h": 5.5,
                      "r_mh": 0.1,#changed this - ORIGINALLY 0.1
                      "r_mh_u": 10,
                      "r_mh_h": 5,
                      "r_hu": 0.1,
                      "r_hu_u": 10,
                      "r_hu_h": 5
}
#-----------parameterization-----------
#TODO: add easier ways for users to input data
#define a parameter to vary - must be in the dictionary above - this should probably be selectable on command line
param_to_change = "r_hu"
#define step size of parameter - ie, how much will each run be different
step_size = 0.05
#define step count - how many different values of the parameter to test
step_count = 5
#define batch size - how many different runs should we average for each step? (default 10 for testing, should increase)
batch_size = 1
#define length of trials (default 1000) - they will usually stop earlier, this is more for allocating space
trial_max_length = 30000
#define starting population
totalpop = 1000
methylatedpop = 800
unmethylatedpop = 100
#-----------setup-----------
output_array = []
exponential_parameters = [None] * step_count
#create arrays to use later
for i in range(step_count):
    output_array.append(np.zeros(batch_size))

#do some math to figure out what values of the param we want to test - store in an array (steps_to_test)
#TODO: figure out how to do generate parameters - do we generate parameter array automatically, or should the user define it?
steps_to_test = [0.05,0.1,0.15,0.2,0.25]
    
#-----------simulation-----------

for step in range(len(steps_to_test)):
    #make a copy of the default parameters, change the parameter we want to study
    input_dict = default_parameters.copy() # shallow copy
    input_dict[param_to_change] = steps_to_test[step]
    #run a batch of identical gillespie algorithms, store the results in output_array[step]
    for i in range(batch_size):
        output_array[step][i] = gillespieswitch.GillespieModelSwitchTime(trial_max_length,input_dict, totalpop, methylatedpop, unmethylatedpop,True).main()
    plt.hist(output_array[step],bins=20)
    plt.show()
    #guess an exponential parameter
    exponential_parameters[step] = stats.expon.fit(output_array[step],floc=0)
    print("predicted exponential parameter: ", exponential_parameters[step])
# print(output_array)
#-----------analysis-----------



#for each batch (batch number is equivalent to number of steps)
#--look at the arrays, process them to find out the distribution of switching times
#--match them to an exponential distribution, infer the parameter of this distribution
#--store the lambda
#do some math to manipulate these lambdas and something?? unsure of this part