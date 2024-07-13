import numpy as np
import longrungillespie
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import statistics
from numba import prange


#-----------parameterization-----------
trial_max_length = 1000000
#define starting population
totalpop = 100
methylatedpop = 50
unmethylatedpop = 50
#SwitchDirection - a simulation terminates when it reaches this state
#-----------Rates Dictionary---------
default_parameters = {"r_hm": 0.5,          #0
                      "r_hm_m": 20/totalpop, #1
                      "r_hm_h": 10/totalpop, #2
                      "r_uh": 0.35,         #3
                      "r_uh_m": 11/totalpop,#4
                      "r_uh_h": 5.5/totalpop,#5
                      "r_mh": 0.1,           #6
                      "r_mh_u": 10/totalpop, #7
                      "r_mh_h": 5/totalpop,  #8
                      "r_hu": 0.1,            #9
                      "r_hu_u": 10/totalpop, #10
                      "r_hu_h": 5/totalpop,   #11
                      "birth_rate": 1         #12
}

#This dictionary just matches each parameter to its place in the list.
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
    "birth_rate": 12

}

parameter_labels = ["r_hm", "r_hm_m","r_hm_h", "r_uh", "r_uh_m", "r_uh_h", "r_mh", "r_mh_u", "r_mh_h", "r_hu", "r_hu_u", "r_hu_h", "birth_rate"]

default_arr = np.array([default_parameters[key] for key in parameter_labels])

#-----------simulation-----------
@numba.jit()
def main(rng):
    #this loop runs in parallel because it uses prange() instead of range() - keep this in mind when debugging it!
        methyl_time, unmethyl_time, middle_time,methylated_arr, unmethylated_arr, time_arr = longrungillespie.GillespieLongRunFun(trial_max_length, default_arr, totalpop, methylatedpop, unmethylatedpop, rng)

        # plt.close() #ensure the previous graph is done
        # plt.hist(valid_array, bins=200)
        # #generate strings, we will concatenate these into a single title string for the graphs
        # param_string = "parameter: " + str(param_to_change) +  " = " + str(step_array[step]) + " -> Exponential Parameter = " + str(exponential_parameters[step])
        # step_string = "Step " + str(step+1) + "/" + str(step_count)
        # batch_string = "Batch of " + str(batch_size) + ", running for " + str(trial_max_length) + " steps each " + str(batch_size-valid_size) + " failed to finish"
        # plt.title(param_string + "\n" + step_string + "\n" + batch_string)
        # plt.savefig("histograms/" + str(time.perf_counter()) + "with" + str(batch_size) + "of" + str(trial_max_length) + '.png')
        # plt.show()
        # plt.close()
        return methyl_time,unmethyl_time,middle_time,methylated_arr, unmethylated_arr, time_arr
    
#-----------setup-----------




generator = np.random.default_rng()

#-----------Call simulation-----------
methylated_time, unmethylated_time, time_in_middle, methylated_arr, unmethylated_arr, time_arr = main(generator)

total_time = methylated_time + unmethylated_time + time_in_middle
methylated_prop = methylated_time/total_time
unmethylated_prop = unmethylated_time/total_time
time_in_middle_prop = time_in_middle/total_time
proportions = [methylated_prop,unmethylated_prop,time_in_middle_prop]
labels = ['methylated_prop','unmethylated_prop','time_in_middle_prop']
print('proportions')
print(f"Methylated : {methylated_prop}, Unmethylated: {unmethylated_prop}, middle: {time_in_middle_prop}")
print('times')
print(f"Methylated : {methylated_time}, Unmethylated: {unmethylated_time}, middle: {time_in_middle}")

# plt.hist(proportions, 3, density=1, histtype='bar', stacked=True, label=labels)
# plt.legend(loc="upper right")
# plt.show()

plt.rcParams["figure.figsize"] = (18,6)
plt.plot(time_arr, methylated_arr, color='pink',ls='', marker=',')
plt.plot(time_arr, unmethylated_arr, color='black',ls='', marker=',')
# plt.axvline(x = self.tarr[step])
plt.xlabel("Time (s)")
plt.ylabel("Population")
#if the user wants to save the image, we do that here.
plt.show()
plt.close()


#bonus

simple_arr = np.zeros_like(methylated_arr)
xes = list(range(len(methylated_arr)))

for i in range(len(methylated_arr)):
        if methylated_arr[i] > 0.7*totalpop:
            simple_arr[i] = 1
        elif unmethylated_arr[i] > 0.7*totalpop:
            simple_arr[i] = -1

# print(xes)
# print(simple_arr)
plt.plot(xes, simple_arr,ls='',marker=',')
plt.show()
        


#-----------postprocessing-----------

# #go through the output row-by-row and find the exponential parameters
# for step in range(step_count):
#     #this list comprehension makes an array of all the positive values in a given row of output_array
#     valid_array = [output[step][index] for index in range(batch_size) if output[step][index] >= 0]
#     #this list comprehension counts up all the negative (meaning timed out) values
#     raw_timeouts = batch_size - len(valid_array)
#     timeouts[step] = 10*(raw_timeouts/batch_size) #scale the timeouts to fit with the other info on the graph

#     #create a line representing the parameter we are varying on the y axis
#     line[step] = step_array[step]


#     #guess parameters only if less than half our simulations timed out
#     if len(valid_array) > batch_size/2:
#         #fit distributions to the data
#         exponential_parameters[step] = stats.expon.fit(valid_array,floc=0)[1]

#         gamma_shape[step],gamma_location[step],gamma_scale[step]=stats.gamma.fit(valid_array,floc=0)
#         inverse_gamma_scale[step] = 1/gamma_scale[step]

#         normal_mean[step], normal_sd[step] =  stats.norm.fit(valid_array,)
#         print(f'Normal mean is {normal_mean[step]} and S.D. is {normal_sd[step]}')

#         empirical_mean[step] = statistics.fmean(valid_array)

#         #calculate error for parameters with Kolmogorov-Smirnov test
#         #note that we lock the first argument, location, to 0 for the exponential distribution
#         exponential_KS[step] = 10 * (stats.kstest(valid_array, 'expon', N=len(valid_array), args=(0,exponential_parameters[step])).statistic)
#         print(exponential_KS[step])
#         # normal_KS[step] = 10 * (stats.kstest(valid_array, 'norm', N=len(valid_array), args=(normal_mean[step], normal_sd[step])).statistic)
#         # print(normal_KS[step])
#         gamma_KS[step] = 10 * (stats.kstest(valid_array, stats.gamma.cdf, N=len(valid_array), args=(gamma_shape[step],0,gamma_scale[step])).statistic)
#         print(gamma_KS[step])

#     # print("predicted exponential parameter: ", exponential_parameters[step])
#     # print("predicted gamma shape parameter: ", gamma_shape[step])
#     print("timed-out simulations: " + str(raw_timeouts) + " out of " + str(batch_size))

# #-----------graphing-----------


# #convert all these to f strings
# plt.close()
# final_label = "Switching times from unmethylated to methylated as birth rate changes \n Population = " + str(totalpop)
# run_stats = "Batches of " + str(batch_size) + ", running for maximum of " + str(trial_max_length) + " steps each"
# plt.plot(step_array, exponential_parameters,label="exponential parameters", linestyle='dashed')
# plt.plot(step_array,timeouts, label = "proportion timed out, scaled by 10x")
# plt.plot(step_array, exponential_KS, label="Exponential KS error, scaled by 10x")

# plt.plot(step_array, gamma_shape,label="Gamma shape")
# plt.plot(step_array, gamma_location,label="Gamma location")
# plt.plot(step_array, gamma_scale,label="Gamma scale")
# plt.plot(step_array, inverse_gamma_scale,label = "1/Gamma scale",linestyle='dashed')
# plt.plot(step_array, gamma_KS, label="Gamma KS error, scaled by 10x")
# plt.plot(step_array, line, linestyle='dotted', label = 'Birth Rate')

# # plt.plot(step_array, normal_mean, label='Normal mean',marker='.',linestyle='')
# # plt.plot(step_array, normal_sd, label='Normal S.D.',marker='.',linestyle='')
# # plt.plot(step_array, normal_KS, label="Normal KS error, scaled by 10x",marker='.',linestyle='')
# # plt.plot(step_array, empirical_mean, label='Empirical Mean', linestyle='dashed')

# plt.title(final_label + "\n" + run_stats)
# plt.xlabel('Value of parameter '+ param_to_change)
# plt.ylabel('Exponential parameter of switching time distribution')
# plt.legend(loc='upper right')
# plt.show()