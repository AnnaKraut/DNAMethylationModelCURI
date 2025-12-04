import numpy as np
import state_sim_EoG as gillespie_longrun
import matplotlib.pyplot as plt
import numba

"""
Performs a single, very long, gillespie run to see what proportion of time is spent in each state - 
(methylated, unmethylated, neither, or sort-of methylated).
"sort-of methylated" refers to a state that is less than 30% unmethylated - in other words,
it is a state where 70% of the sites are either methylated or hemimethylated.

There are no alternate output options for this program - just edit the parameters and run it to get a graph.
"""


# -----------parameters - edit here-----------
# number of steps that the gillespie algorithm will take (large values can cause memory issues, starting around 1,000,000,000)
trial_max_length = 10000000
# define starting population
totalpop = 100
methylatedpop = 50
unmethylatedpop = 50
# -----------Rates Dictionary---------

default_parameters = {
    "r_hm": 8.1734e00,
    "r_hm_m": 2.0121e00 * 2,
    "r_hm_h": 2.0121e00,
    "r_uh": 3.9000e-03,
    "r_uh_m": 2.0000e-04 * 2,
    "r_uh_h": 2.0000e-04,
    "r_mh": 1.7010e-01,
    "r_mh_u": 4.7000e-03 * 2,
    "r_mh_h": 4.7000e-03,
    "r_hu": 3.4970e-01,
    "r_hu_u": 6.6000e-03 * 2,
    "r_hu_h": 6.6000e-03,
    "birth_rate": 1,
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

# these labels allow us to convert the conviently-labelled dictionary into a more efficient array
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
default_arr = np.array([default_parameters[key] for key in parameter_labels])


# -----------simulation-----------
def main(rng):
    return gillespie_longrun.GillespieLongRunFun(
        trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng
    )


# -----------setup-----------

# create a random number generator - this generator can be seeded if desired
generator = np.random.default_rng()

# -----------Call simulation-----------
# call our gillespie algorithm and save the raw data
x_m_arr, x_u_arr, time_arr = main(generator)

x_h_arr = totalpop - x_m_arr - x_u_arr

# print the amount of time that our simulation lasted
total_time = time_arr[-1]
print(f"Check that everything adds up: \nTotal time: {total_time}")


plt.rcParams['font.size'] = 20

# plot our results
plt.title(
    f"Populations of CpG dyads pre cell split over time by methylation  \n simulated with {totalpop} sites over {trial_max_length} iterations",
    fontsize=20,
)
plt.xlabel("Time (average cell generations)")
plt.ylabel("Population (dyads)")
plt.plot(time_arr, x_m_arr, label="Methylated", alpha=.7, color="#D55E00")
plt.plot(time_arr, x_u_arr, label="Unmethylated", alpha=.7, color="#0072B2")
# plt.plot(time_arr, x_h_arr,label="Hemimethylated", alpha=.7, color="#CC79A7")
plt.legend(loc="upper right")
plt.show()
