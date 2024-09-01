# methylation-modeling
This repository allows you to generate and analyze large batches of stochastic simulations of methylation behavior. Specifically, it focuses on measuring switching times between methylated and unmethylated states as various parameters are changed.


## Assumptions and Terminology
- All of these simulations use 100 sites (sometimes referred to as a "population" of 100). This count of sites does not change. Each of these sites is either Methylated, Hemimethylated, or Unmethylated. To simplify calculations, only Methylated and Unmethylated cell counts are tracked. Hemimethylated counts can be calculated as follows: 100 - (# of methylated cells + # of unmethylated cells).
- Familiarity with the Gillespie Algorithm is essential for using this package. This is why I've left the obsolete_algorithm folder - it contains the simple_algorithm folder, which is a very simple, generic version of the Gillespie algorithm that incorporates only births and deaths. Playing around with this simple program was how I familiarized myself with the Gillespie algorithm, and it might be a good start if the python implementation of the algorithm is confusing to you.
- The x-axis of the generated graphs will always start from 0. So if you try to graph a parameter between 0.5 and 1.5, you will get a graph that goes from 0-1 on the x-axis. Keep this in mind when interpreting the graphs.

## Components
Each folder is independant (doesn't use code from other folders) and does different things. Each of them uses an implementation of the Gillespie algorithm with the same **core logic**, but different **output methods**. 
- switching_times: performs many gillespie runs at once to get information about the time that it takes to switch from methylated to unmethylated and vice versa
    - Details: There are options to fit the distribution of the switching times to exponential, normal, and gamma distributions, along with other interesting display options like empirical mean and the rate parameter of the gamma distribution. The below graphs show a few of these options.
    - Output (twoway-simulation.py):  
    <img src="images\2-way-switching-from-0-to-3-birth-rate-90-10-ratio.png" width="400" height="300">
    - Output (simulation-time.py)  
    <img src="images\1-way-switching-from-0-to-3-example.png" width="400" height="300">
- switching_coordinates: performs many gillespie runs at once to find the average amount of methylation and unmethylation where switches happen. 
    - Details: Since a "switch" is recorded whenever 70% or more of the sites are either methylated or unmethylated, in practice this algorithm is measuring whichever category is not at 70%. For example, for a switch from hypo-to-hyper-methylated to actually count as a switch, there will always be 71 hyper-methylated sites; the number of **un**methylated sites will change, however, and the program measures this. The nature of cellular division causing large jumps in unmethylated sites will also be captured by this program.
    - Output  
    <img src="images\coordinates-BR-1-75-15.png" width="400" height="300">
- long_run: performs a single, very long, gillespie run to see what proportion of time is spent in each state (methylated, unmethylated, neither, or sort-of methylated*)
    - Details: *sort-of methylated refers to the condition of being less than 30% unmethylated, and less than 70% methylated. This condition represents the situation after cell division leaves a previously methylated cell with lots of hemimethylation but no fully methylated sites. Cells in this condition often quickly become methylated again, and thus a specific descriptor for them was desirable.
    - Output  
    <img src="images\longtermgraph.png" width="400" height="300">

## Getting Started
- Make sure you have an IDE or code environment that can run the proper version of python, and have installed all needed packages (they are listed at the top of each file)
    - This code was developed in Python 3.12.1 64-bit, and library versions were current as of 8/20/2024.
- Decide which kind of information you are looking for, and open the corresponding folder: long run proportions, switching times, or switching coordinates. I will use the switching time folder for this example.
- Open the simulation file. You will make all your edits in this file, and it will automatically call the gillespie file when needed. You don't need to edit anything in the gillespie file!
- Edit your parameters in the first `Parameters` block.
    - for the parameter you're changing, input the beginning, end, and number of values. So if you want to test [0.8,0.9,1,1.1,1.2], you would input 0.8 for param_begin_val, 1.2 for param_end_val, and 5 for step_count
    - define your batch length and trial length - batch length is the amount of gillespie algorithms that will run for each value of the parameter, and trial length is the amount of steps after which each gillespie algorithm will "time out" and return an error.
    - define the starting methylation state (or initial condition) of your sites. This determines how many sites are methylated or unmethylated at the start of each gillespie run.
        - for the twoway_simulation program, you will need to edit the condition in two places. Use ctrl-f (or command-f on mac) to search for the text `edit here`, and set each initial condition separately.
        - for every other simulation program, you will set the initial condition only once, in the parameters block at the beginning.
- Edit your output metrics and graphs
    - The `graphing` and `postprocessing` sections at the end of each simulation file have a variety of output options. By default, all of the various statistics (fits for various distributions and the goodness of those fits) are calculated. However, it is up to you which ones are shown on the graph! Just comment out the lines starting with `plt.plot` that you don't want on the graph.
    - There are also some debug printouts that display various statistics (number of time-outs, value of parameters for fits, etc.) as the programs run. In general, any lines that involve a `print` statement are for debugging purposes, and can be added or removed without affecting the program's functionality.
- Run the simulation program, and wait for the results!
    - NOTE: running the gillespie algorithm by itself (like `gillespie_time.py`) won't do anything, since this file is just a component of the simulation program and isn't set up to output anything on its own.


## Acknowledgements
This repo represents work funded by a URS grant for the project  
"Stochastic Simulation of Transitions Between Methylation States"
conducted between March 2024 and January 2025.  
**This project was supported by the University of Minnesota's Office of Undergraduate
Research.**

## Contact
Feel free to contact me at mande315@umn.edu with any questions, comments, or concerns.