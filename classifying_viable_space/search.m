% this file actually runs the search with the parameters listed below

addpath("HYPERSPACE-v1.2.1\Source\")

% biologically reasonable bounds:
LOWER_BOUNDS = [0 0 0 0 0 0 0 0];
UPPER_BOUNDS = [300 6 0.04 0.0008 0.25 0.005 0.35 0.007];

%%% EDIT SEARCH PARAMETERS HERE: %%%
MAX_ITERATIONS = 50000;
INITIAL_POINT = [38.6428 3.6870 0.0022 0.0001 0.1392 0.0045 0.0895 0.0066];
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

OutM = MCexp('costfun', .9, INITIAL_POINT, UPPER_BOUNDS, LOWER_BOUNDS, MAX_ITERATIONS)

%%% EDIT OUTPUT FILE HERE: %%
FILENAME = "SearchOutput";

% save(FILENAME + ".mat", OutM)                       % uncomment to save a Matlab matrix
% writematrix([OutM.V OutM.cost], FILENAME + ".csv")  % uncomment to save a csv with the parameter points and their costs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%