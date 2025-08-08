addpath("HYPERSPACE-v1.2.1\Source\")

% this file actually runs the search with the parameters listed below

LOWER_BOUNDS = [0 0 0 0 0 0 0 0];
UPPER_BOUNDS = [300 6 0.04 0.0008 0.25 0.005 0.35 0.007];
MAX_ITERATIONS = 100000;

OutM = MCexp('costfun', 1, [150 3 0.02 0.0004 0.125 0.0025 0.175 .0035], UPPER_BOUNDS, LOWER_BOUNDS, MAX_ITERATIONS)