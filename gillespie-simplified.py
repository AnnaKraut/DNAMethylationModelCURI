import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")



#declare global variables and parameters
steps = 500
M = 10000    #total cell population
N_u = 4000   #unmethylated cells
N_m = 4000   #methylated cells
#we will use this in graphing but not in calculations:
N_h = M - N_u - N_m  #hemimethylated cells

#declare reaction rates
b_rate = 0.1
d_rate = 0.1


#declare array of states and times
n = [0] * steps
t = [0] * steps

n[0] = 10000
t[0] = 0
#create an instance of the generator class (Can customize this to change the rng)
rng = np.random.default_rng()

for i in range(1, steps):
    #compute individual rates
    #can we store R_b, R_d, etc in a dictionary?
    R_b = n[i-1]*b_rate
    R_d = n[i-1]*d_rate

    #compute exponential random variable tau, assign it to t
    prob_sum = R_b + R_d
    print("sum of probabilities: ", prob_sum)
    tau = rng.exponential(scale = 1/prob_sum)
    print("resulting tau = ", tau)
    t[i] = tau+t[i-1]

    #determine relative probability of each event happening
    R_b_prob = R_b/prob_sum
    R_d_prob = R_d/prob_sum

    #draw a uniform random variable to assign event at time t
    uniform = rng.uniform()


    if uniform < R_b_prob: 
        n[i] = n[i-1] + 1
    elif uniform < R_b_prob+R_d_prob:
        n[i] = n[i-1] - 1
    print("given that R_b_prob = ", R_b_prob, "and R_d_prob = ", R_d_prob)
    print("we drew a random variable ", uniform, "changing ", n[i-1], "into", n[i])

#visualize results
plt.plot(t,n)
plt.show()



