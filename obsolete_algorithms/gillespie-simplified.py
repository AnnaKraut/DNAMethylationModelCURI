import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")
from dataclasses import dataclass

#we will pass this dataclass to our functions - its just a standardized way to store the model state
@dataclass
class Parameters:
    n: int
    t: float
#initialize model state (this will be updated before its actually used)
current_params = Parameters(0,0)

#define the functions you want for your events
def birth_event(params):
    #unpack parameters from the class
    n = params.n
    t = params.t
    #DEFINE FUNCTION HERE
    n = n+1
    #pack up parameters again
    return Parameters(n,t)

def death_event(params):
    n = params.n
    t = params.t
    n = n-1
    return Parameters(n,t)



steps = 100
#declare dictionary of elements and their reaction rates
b_rate = 0.1
d_rate = 0.1
base_rates = {"birth":0.1,
              "death":0.1}
base_events = {"birth":birth_event(current_params),
               "death":death_event(current_params)
}


#declare array of states and times
n = [0] * steps
t = [0] * steps

n[0] = 10000
t[0] = 0

#create an instance of the generator class (Can customize this to change the rng)
rng = np.random.default_rng()

for i in range(1, steps):
    #compute individual rates
    #this creates a new dictionary, with the same keys as base_rates, 
    #but with the values assosciated with those keys being updated each loop
    dynamic_rates = {}
    prob_sum = 0
    for key in base_rates:
        dynamic_rates[key] = n[i-1]*base_rates[key] #calculate the rate for the given N
        prob_sum += dynamic_rates[key] #add this rate to prob_sum to use later

    #compute exponential random variable tau, assign it to t
    print("sum of probabilities: ", prob_sum)
    tau = rng.exponential(scale = 1/prob_sum)
    print("resulting tau = ", tau)
    t[i] = tau+t[i-1]

    #determine relative probability of each event happening
    relative_probabilities = {}
    for key in dynamic_rates:
        relative_probabilities[key] = dynamic_rates[key] / prob_sum

    print(relative_probabilities)
    #draw a uniform random variable to assign event at time t
    uniform = rng.uniform()
    print(uniform)
    #bundle our current parameters into a variable
    current_params = Parameters(n[i-1],t[i])

    sum_so_far = 0
    for key in relative_probabilities:
        if uniform < sum_so_far + relative_probabilities[key]:
            new_params = base_events[key]
            n[i] = new_params.n
            print("matched with key", key, "new n is ", n[i])
            break
        else:
            sum_so_far += relative_probabilities[key]
            print("didn't match anywhere :(")

#visualize results
plt.plot(t,n)
plt.show()



