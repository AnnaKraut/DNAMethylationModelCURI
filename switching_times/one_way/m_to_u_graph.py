import numpy as np
import matplotlib.pyplot as plt
import dill

"""
This script takes the one way switching simulation data and outputs graphs for the hyper to hypo methylated switch

The graphs outputted compare the error of fitting normal and gamma distributions to the switching times
"""

input_file = "P8/u_to_m_switch.pkl"         # change this to the output file specified when using run_simulation.py


#---------loads the simulation data---------

dill.load_session(input_file)

# -----------graphing - edit here -----------

plt.close()
final_label = "Switching times from methylated to unmethylated as birth rate changes \n Population = " + str(totalpop)
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"

ax1 = plt.subplot(2, 1, 1)

plt.title(final_label + "\n" + run_stats)   # the axis marks on the second subplot don't behave unless this is after a plt.subplot()...

ax1.plot(step_array, timeouts, label="proportion timed out", color="black")

ax1.plot(step_array, exponential_KS, label="Exponential KS error", color="#F0E442")
ax1.plot(step_array, gamma_KS, label="Gamma KS error", color="#009E73")

ax1.set_ylim(0, 1)
ax1.set_ylabel("Proportion and error", fontsize="small")
ax1.legend(loc="upper left", fontsize="small")

ax2 = ax1.twinx()

ax2.plot(step_array, gamma_shape, label="Gamma shape", color="#009E73", linestyle="dashed")
ax2.set_ylim(0, 3)
ax2.set_ylabel("Gamma Scale Parameter", fontsize="small")
ax2.legend(loc="upper right", fontsize="small")


ax3 = plt.subplot(2, 1, 2)

inverse_exponential = [
    1 / scale if type(scale) == float else np.nan for scale in exponential_parameters
]
ax3.plot(step_array, inverse_exponential, label="Exponential lambda", color="#F0E442")
ax3.plot(step_array, inverse_gamma_scale, label="Gamma lambda", color="#009E73")
ax3.legend(loc="upper left", fontsize="small")

ax3.set_ylabel("Exponential parameter of\nswitching time distribution", fontsize="small")
ax3.set_xlabel("Birth rate", fontsize="small")

plt.show()
