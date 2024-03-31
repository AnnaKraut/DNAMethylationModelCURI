import numpy as np
import pandas as pd
import seaborn as sns


#declare global variables and parameters
steps = 50
M = 10000    #total cell population
N_u = 4000   #unmethylated cells
N_m = 4000   #methylated cells
#we will use this in graphing but not in calculations:
N_h = M - N_u - N_m  #hemimethylated cells

#declare reaction rates
b_rate = 0.2
d_rate = 0.1


#declare array of states and times
n = [0] * steps
t = [0] * steps

n[0] = 5000
t[0] = 0
#create an instance of the generator class (Can customize this to change the rng)
rng = np.random.default_rng()

for i in range(1, steps):
    #compute individual rates
    R_b = n[i-1]*b_rate
    R_d = n[i-1]*d_rate

    #compute exponential random variable tau
    prob_sum = R_b + R_d
    print("sum of probabilities: ", prob_sum)
    tau = rng.exponential(scale = prob_sum)
    print("resulting tau = ", tau)
    




