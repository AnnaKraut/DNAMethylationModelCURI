import numpy as np
import gillespie_time
import scipy.stats as stats
import matplotlib.pyplot as plt
import numba


"""
Performs many gillespie runs at once at one value of cell birth rate (b) and generates a 
histogram of the simulated switching times.

To use, edit the parameters to test in the parameters block, and set the initial conditions
for each switch ideally using the average entry coordinates from the switching coordinates
simulation. Set the baseline parameters of the simulation in the Rates Dictionary.
"""

# -----------parameters-----------
# user should enter begin, end, step for the parameter they want to change.
param_to_change = "birth_rate"
param_val = 1.25

batch_size = 1000                       # number of simulations to run
trial_max_length = 5000000              # maximum length of trials (in simulation steps)

totalpop = 100                          # number of CpG sites to simulate

# initial conditions
methylatedpop = 6
unmethylatedpop = 73

# SwitchDirection - a simulation terminates when it reaches this state
SwitchDirection = 1                     # 1 -> mostly methylated, -1-> mostly unmethylated

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



# -----------simulation-----------
@numba.jit(nopython=True, parallel=True)
def main(gen):
    output_array = np.zeros(batch_size)
    # this loop runs in parallel because it uses prange() instead of range() - keep this in mind if debugging it
    for i in range(batch_size):
        # run a batch of identical gillespie algorithms, store the results in output_array[step]
        output_array[i] = gillespie_time.GillespieSwitchFun(
            trial_max_length,
            default_arr,
            totalpop,
            methylatedpop,
            unmethylatedpop,
            SwitchDirection,
            gen,
        )
    return output_array

# -----------Call simulation-----------
gen = np.random.default_rng()
output = main(gen)

# -----------postprocessing-----------

# Fit exponential distribution
valid_array = [
    output[index] for index in range(batch_size) if output[index] >= 0
]

params = stats.expon.fit(valid_array, floc=0) # scale = 1/lambda
scale = params[1]

# Generate fitted PDF of exponential
x = np.linspace(0, max(output), 100)
fitted_exp = stats.expon.pdf(x, loc=0, scale=scale)

factor = max(output) / 100 * batch_size

plt.hist(output, bins= 100, density = True)
plt.scatter(x, fitted_exp, color="red")
plt.title(f'Histogram of switching times for b={param_val} of {"U->M" if SwitchDirection==1 else "M->U"}')

plt.show()