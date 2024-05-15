# methylation-modeling
This repository allows you to generate and analyze large batches of stochastic simulations of methylation behavior. Specifically, it focuses on measuring switching times between methylated and unmethylated states as various parameters are changed.

## Components
- An implementation of the [Gillespie algorithm](https://en.wikipedia.org/wiki/Gillespie_algorithm) that simulates the stochastic methylation behavior of a single cell (**gillespieswitch.py**)
- A wrapper function that generates large batches of the Gillespie algorithm, varying a single parameter, and saves their results to a graph, so that the user can see trends in switching times as the parameter changes (**simulation.py**)
- A config file that exposes all of the model parameters, allowing researchers to change parameters and be confident that their changes won't cause issues with the underlying logic (**config.py**)
- TODO: yet another wrapper function that abstracts away the details of simulation.py (batch size, run length, debugging toggles, etc).

## Example usage
Let's say the user wants to simulate 5000 runs of a simulation with the default methylation parameters and a maximum simulation length of 10,000 steps.   
Go to the `#-----------parameterization-----------` heading in simulation.py and change batch_size to 5000, and trial_max_length to 10000.  
Then set the initial condition of the simulation - by default, there are 100 sites, of which 90 are methylated and 10 unmethylated. This means the cell's **starting state is methylated -** so, it's probably most interesting to observe the switching times **to unmethylated.**  To do this, set SwitchDirection to -1 (or 1 to switch to methylated).  
There are additional parameters that allow users to vary a single parameter across a wide range of values (param_to_change, step_size, step_count, steps_to_test), but these will be abstracted away in the (currently unfinished) wrapper function. For now, **just set step_count to 1 and steps_to_test to the default value of param_to_change** - this will run the code on a single parameter regime.  
After these parameters are adjusted to your liking, simply run simulations.py - it will handle everything else. **Make sure you have folders titled "histograms" and "output" in the same folder** - simulations.py will write a graph of the first simulation to "output" for debugging purposes, and a histogram of the switching times across all simulations to the "histograms" folder. These graphs will look something like this:  
![Methylated and unmethylated sites over time in a single simulation](images/output1.png)
![Distribution of switching times across all runs of the simulation](images/histogram1.png)




## Installation
At this time this code does not have a frontend: to run it you will need to open and run it with a code editor (like VSCode). If you don't have the correct Python libraries installed you may get some error messages - [Here is a guide to importing libraries and fixing those errors.](https://python.land/virtual-environments/installing-packages-with-pip)




## Acknowledgements
This repo represents work funded by a URS grant for the project  
"Stochastic Simulation of Transitions Between Methylation States"   
awarded 3/17/2024 with expected completion by 7/12/2024.  
**This project was supported by the University of Minnesota's Office of Undergraduate
Research.**