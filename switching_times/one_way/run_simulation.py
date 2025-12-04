import numpy as np
import gillespie_time
import scipy.stats as stats
import numba
import statistics
from numba import prange
import dill

"""
Performs many gillespie runs at once to get information about the time 
that it takes to switch from methylated to unmethylated and vice versa.

To use, edit the parameters to test in the parameters block, and set the initial conditions
for each switch ideally using the average entry coordinates from the switching coordinates
simulation. Set the baseline parameters of the simulation in the Rates Dictionary.

The data is saved to file path specified so that you don't have to rerun the simulation to
change the aesthetics of the graph
"""

# -----------parameters-----------
# SwitchDirection - a simulation terminates when it reaches this state (1 -> mostly methylated, -1-> mostly unmethylated)
### Uncomment For U to M Direction:
SwitchDirection = 1
output_file = "P8/u_to_m_switch.pkl"
### Uncomment For M to U Direction:
# SwitchDirection = -1
# output_file = "P8/m_to_u_switch.pkl"

# user should enter begin, end, step for the parameter they want to change.
param_to_change = "birth_rate"
param_begin_val = 0
param_end_val = 3
step_count = 49                         # number of evenly spaced parameter points to check

batch_size = 1000                       # number of simulations ran per parameter
trial_max_length = 5000000              # maximum length of trials (in simulation steps)

totalpop = 100                          # number of CpG sites to simulate

# initial conditions
methylatedpop = 6
unmethylatedpop = 73

# -----------Rates Dictionary---------
default_parameters = {
    "r_hm": 8.1734e00,  # 0
    "r_hm_m": 2.0121e00 * 2,  # 1
    "r_hm_h": 2.0121e00,  # 2
    "r_uh": 3.9000e-03,  # 3
    "r_uh_m": 2.0000e-04 * 2,  # 4
    "r_uh_h": 2.0000e-04,  # 5
    "r_mh": 1.7010e-01,  # 6
    "r_mh_u": 4.7000e-03 * 2,  # 7
    "r_mh_h": 4.7000e-03,  # 8
    "r_hu": 3.4970e-01,  # 9
    "r_hu_u": 6.6000e-03 * 2,  # 10
    "r_hu_h": 6.6000e-03,  # 11
    "birth_rate": 1,  # 12
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


# -----------simulation-----------
@numba.jit(nopython=True, parallel=True)
def main(rngs):
    output_array = np.zeros(shape=(step_count, batch_size))
    # this loop runs in parallel because it uses prange() instead of range() - keep this in mind if debugging it
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


# -----------setup-----------

# generate the arrays for our output - None (or null value) is the default
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


# list comprehension that creates an array of the values we tested for our chosen parameter
step_array = [param_begin_val + step_size * i for i in range(step_count)]

# create an array of random number generators that we will pass into our function
# this makes it easier to reproduce, and also keeps Numba happy.
generators = [None] * step_count
for i in range(step_count):
    generators[i] = np.random.default_rng()

# -----------Call simulation-----------
output = main(generators)

# -----------postprocessing-----------

# go through the output row-by-row and find the exponential parameters
for step in range(step_count):
    # this list comprehension makes an array of all the positive values in a given row of output_array
    valid_array = [
        output[step][index] for index in range(batch_size) if output[step][index] >= 0
    ]
    # this list comprehension counts up all the negative (meaning timed out) values
    raw_timeouts = batch_size - len(valid_array)
    timeouts[step] = (
        raw_timeouts / batch_size
    )  # scale the timeouts to fit with the other info on the graph

    # create a line representing the parameter we are varying on the y axis
    line[step] = step_array[step]

    # guess parameters only if less than half our simulations timed out
    if len(valid_array) > batch_size / 2:
        # fit distributions to the data
        exponential_parameters[step] = stats.expon.fit(valid_array, floc=0)[1]
        print("exponential paramater = " + str(exponential_parameters[step]))

        gamma_shape[step], gamma_location[step], gamma_scale[step] = stats.gamma.fit(
            valid_array, floc=0
        )
        inverse_gamma_scale[step] = 1 / gamma_scale[step]

        normal_mean[step], normal_sd[step] = stats.norm.fit(
            valid_array,
        )
        print(f"Normal mean is {normal_mean[step]} and S.D. is {normal_sd[step]}")

        empirical_mean[step] = statistics.fmean(valid_array)

        # calculate error for parameters with Kolmogorov-Smirnov test
        # note that we lock the first argument, location, to 0 for the exponential distribution
        exponential_KS[step] = stats.kstest(
            valid_array,
            "expon",
            N=len(valid_array),
            args=(0, exponential_parameters[step]),
        ).statistic
        print(exponential_KS[step])
        normal_KS[step] = stats.kstest(
            valid_array,
            "norm",
            N=len(valid_array),
            args=(normal_mean[step], normal_sd[step]),
        ).statistic
        # print(normal_KS[step])
        gamma_KS[step] = stats.kstest(
            valid_array,
            stats.gamma.cdf,
            N=len(valid_array),
            args=(gamma_shape[step], 0, gamma_scale[step]),
        ).statistic
        print(gamma_KS[step])

    # print("predicted exponential parameter: ", exponential_parameters[step])
    # print("predicted gamma shape parameter: ", gamma_shape[step])
    print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))

# saves the workspace variables at the path held in output_file
dill.dump_session(output_file)
