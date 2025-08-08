import numpy as np


#-------Parameters-------

BIRTH_RATE = .1
# DEATH_RATE = 0.25
CARRYING_CAPACITY = 100000
U_TO_M_RATE = 6.22e-5
M_TO_U_RATE = 1.214e-4

def u_birth(methylated, unmethylated):
    # return 0
    return unmethylated * BIRTH_RATE

def u_death(methylated, unmethylated):
    # return 0
    return unmethylated * BIRTH_RATE * (unmethylated + methylated) / CARRYING_CAPACITY  # logistic
    # return unmethylated * DEATH_RATE                                                  # exponential

def u_switch(methylated, unmethylated):
    # return 0
    return unmethylated * U_TO_M_RATE

def m_birth(methylated, unmethylated):
    # return 0
    return methylated * BIRTH_RATE

def m_death(methylated, unmethylated):
    # return 0
    return methylated * BIRTH_RATE * (unmethylated + methylated) / CARRYING_CAPACITY    # logistic
    # return methylated * DEATH_RATE                                                    # exponential

def m_switch(methylated, unmethylated):
    # return 0
    return methylated * M_TO_U_RATE

#This function defines the events that can happen. It's equivalent to the event list in config.py
#i_local indicates which loop called this function - that is, i_local indicates which event we're doing.

def events(methylated, unmethylated, i_local):
    # U birth
    if i_local == 0:
        return methylated, unmethylated + 1
    # U death
    elif i_local == 1:
        return methylated, unmethylated - 1
    # U switch (to M)
    elif i_local == 2:
        return methylated + 1, unmethylated - 1
    # M birth
    elif i_local == 3:
        return methylated + 1, unmethylated
    # M death
    elif i_local == 4:
        return methylated - 1, unmethylated
    # M switch (to U)
    elif i_local == 5:
        return methylated - 1, unmethylated + 1


def Gillespie(steps, pop_methyl, pop_unmethyl, rng):
    # output array setup
    methylated_arr = np.zeros(steps)
    unmethylated_arr = np.zeros(steps)
    time_arr = np.zeros(steps)
    
    methylated_arr[0] = pop_methyl 
    unmethylated_arr[0] = pop_unmethyl 
    time_arr[0] = 0
    
    # figure out how often to sample points
    
    
    # actual variables used to track state
    # curr_methyl = pop_methyl
    # curr_unmethyl = pop_unmethyl
    # curr_time = 0
    
    
    rates = np.zeros(6) 

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl

        #find the rates of each event for the current parameters
        rates[0] = u_birth(methylated_arr[i-1],unmethylated_arr[i-1])
        rates[1] = u_death(methylated_arr[i-1],unmethylated_arr[i-1])
        rates[2] = u_switch(methylated_arr[i-1],unmethylated_arr[i-1])
        rates[3] = m_birth(methylated_arr[i-1],unmethylated_arr[i-1])
        rates[4] = m_death(methylated_arr[i-1],unmethylated_arr[i-1])
        rates[5] = m_switch(methylated_arr[i-1],unmethylated_arr[i-1])
        rate_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/rate_sum)
        time_arr[i] = tau + time_arr[i-1]

        #normalize the rates to be within (0,1)
        normalized_rates= rates / rate_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for event_number in range(6):
            if uniform < normalized_rates[event_number] + sum_so_far:
                methylated_arr[i], unmethylated_arr[i] = events(methylated_arr[i-1], unmethylated_arr[i-1],event_number)
                break
            else:
                sum_so_far += normalized_rates[event_number]
        
    return (methylated_arr, unmethylated_arr, time_arr)
