import numpy as np


#-------Parameters-------

BIRTH_RATE = 1
CARRYING_CAPACITY = 1000

TREATMENT_EFFECT = .01
TREATMENT_DECAY = .05

U_TO_M_RATE = 6.22e-5
M_TO_U_RATE = 1.214e-4

def u_birth(methylated, unmethylated, treatment):
    # return 0
    return unmethylated * BIRTH_RATE

def u_death(methylated, unmethylated, treatment):
    # death rate for the u population is the limitting term from logistic growth plus the treatment 
    return (unmethylated * BIRTH_RATE * (unmethylated + methylated) / CARRYING_CAPACITY) + unmethylated * treatment * TREATMENT_EFFECT

def u_switch(methylated, unmethylated, treatment):
    # return 0
    return unmethylated * U_TO_M_RATE

def m_birth(methylated, unmethylated, treatment):
    # return 0
    return methylated * BIRTH_RATE

def m_death(methylated, unmethylated, treatment):
    # return 0
    return methylated * BIRTH_RATE * (unmethylated + methylated) / CARRYING_CAPACITY    # logistic
    # return methylated * DEATH_RATE                                                    # exponential

def m_switch(methylated, unmethylated, treatment):
    # return 0
    return methylated * M_TO_U_RATE

def t_decay(methylated, unmethylated, treatment):
    return treatment * TREATMENT_DECAY

#This function defines the events that can happen. It's equivalent to the event list in config.py
#i_local indicates which loop called this function - that is, i_local indicates which event we're doing.

def events(methylated, unmethylated, treatment, i_local):
    # U birth
    if i_local == 0:
        return methylated, unmethylated + 1, treatment
    # U death
    elif i_local == 1:
        return methylated, unmethylated - 1, treatment
    # U switch (to M)
    elif i_local == 2:
        return methylated + 1, unmethylated - 1, treatment
    # M birth
    elif i_local == 3:
        return methylated + 1, unmethylated, treatment
    # M death
    elif i_local == 4:
        return methylated - 1, unmethylated, treatment
    # M switch (to U)
    elif i_local == 5:
        return methylated - 1, unmethylated + 1, treatment
    elif i_local == 6:
        return methylated, unmethylated, treatment - 1


def Gillespie(steps, pop_methyl, pop_unmethyl, treatment_times, rng):
    # output array setup
    methylated_arr = np.zeros(steps)
    unmethylated_arr = np.zeros(steps)
    treatment_arr = np.zeros(steps)
    time_arr = np.zeros(steps)
    
    methylated_arr[0] = pop_methyl 
    unmethylated_arr[0] = pop_unmethyl
    treatment_arr[0] = 0
    time_arr[0] = 0
    
    max_doses = len(treatment_times)
    doses = 0
    
    # figure out how often to sample points
    
    
    # actual variables used to track state
    # curr_methyl = pop_methyl
    # curr_unmethyl = pop_unmethyl
    # curr_time = 0
    
    
    rates = np.zeros(7) 

    #main loop - each generation or step is one iteration of this loop
    for i in range(1, steps): #start at 1, since the first step is given by pop_methyl/pop_unmethyl

        #find the rates of each event for the current parameters
        rates[0] = u_birth(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[1] = u_death(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[2] = u_switch(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[3] = m_birth(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[4] = m_death(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[5] = m_switch(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rates[6] = t_decay(methylated_arr[i-1],unmethylated_arr[i-1],treatment_arr[i-1])
        rate_sum = np.sum(rates)

        #find the expected wait for an event to happen
        tau = rng.exponential(scale = 1/rate_sum)
        time_arr[i] = tau + time_arr[i-1]
        
        if doses < max_doses:
            if time_arr[i] >= treatment_times[doses,0]:
                time_arr[i] = treatment_times[doses, 0]
                methylated_arr[i] = methylated_arr[i-1]
                unmethylated_arr[i] = unmethylated_arr[i-1]
                treatment_arr[i] = treatment_arr[i-1] + treatment_times[doses, 1]
                doses += 1
                continue

        #normalize the rates to be within (0,1)
        normalized_rates= rates / rate_sum

        #select which event happens by comparing the normalized rates to a random variable
        sum_so_far = 0
        uniform = rng.uniform()
        for event_number in range(7):
            if uniform < normalized_rates[event_number] + sum_so_far:
                methylated_arr[i], unmethylated_arr[i], treatment_arr[i] = events(methylated_arr[i-1], unmethylated_arr[i-1], treatment_arr[i-1],event_number)
                break
            else:
                sum_so_far += normalized_rates[event_number]
        
    return (methylated_arr, unmethylated_arr, treatment_arr, time_arr)
