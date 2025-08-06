import numpy as np
import math
import gillespie_time
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange
import dill


def scale_array(arr, scale):
    return [scale * item if not item is None else np.nan for item in arr]

dill.load_session("u_to_m_switch.pkl")


#-----------graphing - edit here -----------

#plotting - much of this can be removed if desired
plt.close()
final_label = "Switching times from unmethylated to methylated as birth rate changes \n Population = " + str(totalpop)
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"

plt.subplot(2,1,1)
plt.plot(step_array, normal_mean, label='Normal mean',marker='.',linestyle='')
plt.plot(step_array, normal_sd, label='Normal S.D.',marker='.',linestyle='')
plt.plot(step_array, empirical_mean, label='Empirical Mean', linestyle='dashed')

plt.ylabel('Exponential parameter of switching time distribution')
plt.plot(step_array, exponential_parameters,label="exponential parameters", linestyle='dashed')

plt.title(final_label + "\n" + run_stats)
plt.legend(loc='upper right')


plt.subplot(2,1,2)
plt.ylim(0,1)
plt.plot(step_array, timeouts, label = "proportion timed out",linestyle=':')
plt.plot(step_array, normal_KS, label="Normal KS error",marker='.',linestyle='')
plt.plot(step_array, exponential_KS, label="Exponential KS error")

plt.legend(loc='upper right')
plt.xlabel('Value of parameter '+ param_to_change)


plt.show()