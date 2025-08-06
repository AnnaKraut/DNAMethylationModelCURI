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

dill.load_session("m_to_u_switch.pkl")


#-----------graphing - edit here -----------

plt.close()
final_label = "Switching times from methylated to unmethylated regions as birth rate changes \n Population = " + str(totalpop)
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"

plt.rcParams['font.size'] = 20

plt.subplot(2,1,1)
plt.plot(step_array, scale_array(timeouts, 3), label = "proportion timed out scaled x 3")
plt.plot(step_array, scale_array(exponential_KS, 3), label="Exponential KS error scaled x 3")

plt.plot(step_array, gamma_shape,label="Gamma shape scaled")
plt.plot(step_array, scale_array(gamma_KS, 3), label="Gamma KS error scaled x 3")
# plt.plot(step_array, line, linestyle='dotted', label = 'Birth Rate')


plt.title(final_label + "\n" + run_stats)
plt.xlabel('Value of parameter '+ param_to_change)
plt.legend(loc='upper right')

plt.subplot(2,1,2)

plt.ylabel('Exponential parameter of switching time distribution')
plt.plot(step_array, [1/scale if type(scale) == float else np.nan for scale in exponential_parameters],label="exponential parameters", linestyle='dashed')
plt.plot(step_array, inverse_gamma_scale,label = "1/Gamma scale",linestyle='dashed')
plt.legend(loc='upper right')


plt.show()