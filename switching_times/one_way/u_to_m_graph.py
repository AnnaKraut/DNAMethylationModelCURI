import numpy as np
import matplotlib.pyplot as plt
import dill

"""
This script takes the one way switching simulation data and outputs graphs for the hypo to hyper methylated switch

The graphs outputted compare the error of fitting normal and exponential distributions to the switching times
"""

input_file = "P8/u_to_m_switch.pkl"         # change this to the output file specified when using run_simulation.py


#---------loads the simulation data---------

dill.load_session(input_file)

#-----------graphing - edit here -----------

plt.close()
final_label = "Switching times from unmethylated to methylated as birth rate changes \n Population = " + str(totalpop)
run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"
plt.title(final_label + "\n" + run_stats)

plt.subplot(2,1,1)
plt.plot(step_array, normal_mean, label='Normal mean',marker='.',linestyle='')
plt.plot(step_array, normal_sd, label='Normal S.D.',marker='.',linestyle='')
plt.plot(step_array, empirical_mean, label='Empirical Mean', linestyle='dashed')

plt.ylabel('Exponential parameter of switching time distribution')
plt.plot(step_array, exponential_parameters,label="exponential parameters", linestyle='dashed')

plt.legend(loc='upper right')


plt.subplot(2,1,2)
plt.ylim(0,1)
plt.plot(step_array, timeouts, label = "proportion timed out",linestyle=':')
plt.plot(step_array, normal_KS, label="Normal KS error",marker='.',linestyle='')
plt.plot(step_array, exponential_KS, label="Exponential KS error")

plt.legend(loc='upper right')
plt.xlabel('Value of parameter '+ param_to_change)


plt.show()