# methylation-modeling
This repository allows you to generate and analyze large batches of stochastic simulations of methylation behavior. Specifically, it focuses on measuring switching times between methylated and unmethylated states as various parameters are changed.

## Components
- An implementation of the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm) that simulates the stochastic methylation behavior of a single cell (**gillespieswitch.py**)
- A wrapper function that generates large batches of the Gillespie algorithm, varying a single parameter, and saves their results to a graph, so that the user can see trends in switching times as the parameter changes (**simulation.py**)
- A config file that exposes all of the model parameters, allowing researchers to change parameters and be confident that their changes won't cause issues with the underlying logic (**config.py**)
- TODO: yet another wrapper function that abstracts away the details of simulation.py (batch size, run length, debugging toggles, etc).

## Design
The simulation operates at 3 increasingly abstract levels of detail, each of which generates different graphs.
- Individual simulations, modeled by a single run of the **gillespieswitch.py** component
    - Automatically takes the parameters of a single run as input and runs the Gillespie algorithm until a switch occurs
    - Outputs the time taken to switch (that is, from methylated to unmethylated, or vice versa).
    - View graphs for this component by setting `debug = True`
    - ![Methylated and unmethylated sites over time in a single simulation](images/output1.png)
- Batches of simulations with the same parameters, modeled by a single loop within the **simulation.py** program.
    - Automatically takes in parameters, and runs `batch_size` individual simulations 
    - Fits an exponential distribution to the distribution of their switching times
    - Outputs the parameter of that exponential distribution
    - View graphs for this component by setting `batch_debug = True`
    - ![Distribution of switching times across all runs of the simulation](images/histogram1.png)
- Comparison of switching times across multiple batches, modeled by a single run of **simulation.py**
    - **Manually** takes in a parameter and range of values (in the config section), and automatically runs a batch for each value in the range
    - Returns a visualization of how the exponential parameter of switching times changes relative to the chosen parameter
    - A single graph for this component is always displayed
    - ![ExponentialParameters](images/ExponentialParameters.png)


## Example usage
**The only files the user should modify are the `#-----------parameterization-----------` section in simulation.py and the config.py file.**

Let's say the user wants to simulate 5000 runs of a simulation with the default methylation parameters and a maximum simulation length of 10,000 steps.   
Go to the `#-----------parameterization-----------` heading in simulation.py and change batch_size to 5000, and trial_max_length to 10000.  
Then set the initial condition of the simulation - by default, there are 100 sites, of which 90 are methylated and 10 unmethylated. This means the cell's **starting state is methylated -** so, it's probably most interesting to observe the switching times **to unmethylated.**  To do this, set SwitchDirection to -1 (or 1 to switch to methylated).  

This will run the code on a single parameter regime.  

After these parameters are adjusted to your liking, simply **run simulations.py** - it will handle everything else.


There are additional config parameters that allow users to vary a single parameter across a wide range of values (param_to_change, param_begin_val, param_end_val, param_step_size): the user will enter a range and step size, and the program will run a batch of simulations for each step in the range.



## Installation
At this time this code does not have a frontend: to run it you will need to open and run it with a code editor (like VSCode). If you don't have the correct Python libraries installed you may get some error messages - [Here is a guide to importing libraries and fixing those errors.](https://python.land/virtual-environments/installing-packages-with-pip)




## Acknowledgements
This repo represents work funded by a URS grant for the project  
"Stochastic Simulation of Transitions Between Methylation States"   
awarded 3/17/2024 with expected completion by 7/12/2024.  
**This project was supported by the University of Minnesota's Office of Undergraduate
Research.**