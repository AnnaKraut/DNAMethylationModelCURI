import numpy as np
import gillespie_time as gillespie_time
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange
import dill

"""
Performs many gillespie runs at once, in both directions, to get information about the time 
that it takes to switch from methylated to unmethylated and vice versa.

Edit the parameters in the `parameters` block.

Important! Since this program simulates methylated->unmethylated and unmethylated->methylated transitions, 
you must edit the starting populations for BOTH transitions, and these populations are NOT in the parameters block. 
You can find these starting populations by searching for `edit here` in the file.

There are many different graphing options for this simulation! 
You can graph various parameters and goodness-of-fit measures for exponential, normal, and gamma fits,
as well as the empirical mean of the measurements. To enable these graphs, simply uncomment them at the bottom of the file.
"""


def nan_array(n):
    return [np.nan for i in range(n)]


# -----------parameters-----------
# user should enter begin, end, step for the parameter they want to change.
param_to_change = "birth_rate"
param_begin_val = 0
param_end_val = 3
step_count = 49                     # number of evenly spaced parameter points to check

batch_size = 1000                   # number of simulations ran per parameter
trial_max_length = 5000000          # maximum length of trials (in simulation steps)

totalpop = 100                      # number of CpG sites to simulate

# starting conditions for each switching direction
u_to_m_initial = {"methylated":  5, "unmethylated": 74}
m_to_u_initial = {"methylated": 71, "unmethylated": 29}

outfile = "P8/two_way_sim_data.pkl"    # the filename to save data to to graph later



# -----------Rates Dictionary---------
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
                      
                      #adjust birth rate directly - edit here
                      "birth_rate": 1     #12
}


# This dictionary just matches each parameter to its place in the list.
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
    "birth_rate": 12,
}

parameter_labels = [
    "r_hm",
    "r_hm_m",
    "r_hm_h",
    "r_uh",
    "r_uh_m",
    "r_uh_h",
    "r_mh",
    "r_mh_u",
    "r_mh_h",
    "r_hu",
    "r_hu_u",
    "r_hu_h",
    "birth_rate",
]

# this line creates a numpy array with the same values as the dictionary - it is VITAL that they stay in the same order!!
# changing the order of either the labels or the stuff in this list will create subtle errors in the rate calculations!
default_arr = np.array([default_parameters[key] for key in parameter_labels])
# This line just calculates where the parameter we want to change occurs in the list
index_to_change = default_indices[param_to_change]

# find the size of each step, rounded to 5 decimal places.
step_size = round((param_end_val - param_begin_val) / (step_count - 1), 5)
# step_array = step_count


# -----------simulation-----------
@numba.jit(nopython=True, parallel=True)
def main(rngs, SwitchDirection, methylatedpop, unmethylatedpop):
    output_array = np.zeros(shape=(step_count, batch_size))
    # this loop runs in parallel because it uses prange() instead of range() - keep this in mind when debugging it!
    for step in prange(step_count):
        # make a copy of the default parameters, change the parameter we want to study
        temp_arr = default_arr.copy()
        temp_arr[index_to_change] = param_begin_val + (step * step_size)
        print("Testing parameters: ", temp_arr)

        # run a batch of identical gillespie algorithms, store the results in output_array[step]
        for i in range(batch_size):
            output_array[step][i] = gillespie_time.GillespieSwitchFun(
                trial_max_length,
                temp_arr,
                totalpop,
                methylatedpop,
                unmethylatedpop,
                SwitchDirection,
                rngs[step],
            )
    return output_array


# -----------setup - METHYLATED TO UNMETHYLATED-----------
# generate the arrays for our output - None (or null value) is the default
exponential_parameters_MtoU = nan_array(step_count)
gamma_shape_MtoU = nan_array(step_count)
gamma_location_MtoU = nan_array(step_count)
gamma_scale_MtoU = nan_array(step_count)
exponential_KS_MtoU = nan_array(step_count)
gamma_KS_MtoU = nan_array(step_count)
timeouts_MtoU = [0] * step_count
empirical_mean_MtoU = nan_array(step_count)
# list comprehension that creates an array of the values we tested for our chosen parameter
step_array = [param_begin_val + step_size * i for i in range(step_count)]

# create an array of random number generators that we will pass into our function
# this makes it easier to reproduce, and also keeps Numba happy.
generators = nan_array(step_count)
for i in range(step_count):
    generators[i] = np.random.default_rng()

# -----------UNMETHYLATED TO METHYLATED-----------

output = main(
    generators, -1, m_to_u_initial["methylated"], m_to_u_initial["unmethylated"]
)  # M to U switching

# -----------postprocessing-----------

# go through the output row-by-row and find the exponential parameters
for step in range(step_count):
    # this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [
        output[step][index] for index in range(batch_size) if output[step][index] >= 0
    ]
    # this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts_MtoU[step] = (
        raw_timeouts / batch_size
    )  # scale the timeouts to fit with the other info on the graph

    # guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size / 2:
        # fit distributions to the data
        exponential_parameters_MtoU[step] = stats.expon.fit(valid_array, floc=0)[1]

        empirical_mean_MtoU[step] = statistics.fmean(valid_array)

        # calculate error for parameters with Kolmogorov-Smirnov test
        exponential_KS_MtoU[step] = stats.kstest(
            valid_array,
            "expon",
            args=(0, exponential_parameters_MtoU[step]),
            N=len(valid_array),
        ).statistic
        print(exponential_KS_MtoU[step])
    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))
    print("exponential paramater MtoU = " + str(exponential_parameters_MtoU[step]))


# -----------UNMETHYLATED TO METHYLATED-----------

output = main(
    generators, 1, u_to_m_initial["methylated"], u_to_m_initial["unmethylated"]
)  # U to M switching

# generate the arrays for our output - None (or null value) is the default
exponential_parameters_UtoM = nan_array(step_count)
gamma_shape_UtoM = nan_array(step_count)
gamma_location_UtoM = nan_array(step_count)
gamma_scale_UtoM = nan_array(step_count)
exponential_KS_UtoM = nan_array(step_count)
gamma_KS_UtoM = nan_array(step_count)
normal_KS_UtoM = nan_array(step_count)
normal_sd_UtoM = nan_array(step_count)
normal_mean_UtoM = nan_array(step_count)
timeouts_UtoM = [0] * step_count
empirical_mean_UtoM = nan_array(step_count)
# list comprehension that creates an array of the values we tested for our chosen parameter
step_array = [param_begin_val + step_size * i for i in range(step_count)]


for step in range(step_count):
    # this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [
        output[step][index] for index in range(batch_size) if output[step][index] >= 0
    ]
    # this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts_UtoM[step] = (
        raw_timeouts / batch_size
    )  # scale the timeouts to fit with the other info on the graph

    # guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size / 2:
        # fit distributions to the data
        exponential_parameters_UtoM[step] = stats.expon.fit(valid_array, floc=0)[1]

        empirical_mean_UtoM[step] = statistics.fmean(valid_array)

        # calculate error for parameters with Kolmogorov-Smirnov test
        # TODO: add args for expon
        exponential_KS_UtoM[step] = stats.kstest(
            valid_array,
            "expon",
            N=len(valid_array),
            args=(0, exponential_parameters_UtoM[step]),
        ).statistic
        print(exponential_KS_MtoU[step])

    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))
    print("exponential paramater UtoM= " + str(exponential_parameters_UtoM[step]))

# -----------saving data----------

dill.dump_session(outfile)
