from early_termination import GillespieLongRunFun
import numpy as np

X = 0 # centering parameter for cost function

MAX = 100000000 # max number of steps
POP = 100       # total number of sites
POP_M = 50      # initial n methylated
POP_H = 50      # initial n unmethylated

RNG = np.random.default_rng()

UPPER_BOUNDS = np.array([1000, 20, 10, .1, 2, .1, 1, .01], dtype='float64')
OUTPUT_FILE = 'output.txt'

def cost_function(params):
    methyl, unmethyl, middle, sorta = GillespieLongRunFun(MAX, params, POP, POP_M, POP_M, RNG)
    cost = 0
    if middle > 0.8:
        cost = float('Inf')
    else:
        denominator = methyl + unmethyl
        if denominator == 0:
            cost = float('Inf')
        else:
            cost = abs(methyl - unmethyl + X) / denominator
    return cost

# making the grid
r_hm   = np.linspace(0,UPPER_BOUNDS[0],10)
r_hm_h = np.linspace(0,UPPER_BOUNDS[1],10)
r_uh   = np.linspace(0,UPPER_BOUNDS[2],10)
r_uh_h = np.linspace(0,UPPER_BOUNDS[3],10)
r_mh   = np.linspace(0,UPPER_BOUNDS[4],10)
r_mh_h = np.linspace(0,UPPER_BOUNDS[5],10)
r_hu   = np.linspace(0,UPPER_BOUNDS[6],10)
r_hu_h = np.linspace(0,UPPER_BOUNDS[7],10)

results = []
# header
results.append('r_hm, r_hm_h, r_uh, r_uh_h, r_mh, r_mh_h, r_hu, r_hu_h, cost\n')

for r1 in r_hm:
    for r2 in r_hm_h:
        for r3 in r_uh:
            for r4 in r_uh_h:
                for r5 in r_mh:
                    for r6 in r_mh_h:
                        for r7 in r_hu:
                            for r8 in r_hu_h:
                                params = np.array([
                                    r1, 2*r2, r2,   # maintenance rates
                                    r3, 2*r4, r4,   # de novo rates
                                    r5, 2*r6, r6,   # demethylation rates (m -> h)
                                    r7, 2*r8, r8,   # demethylation rates (h -> u)
                                    1               # fixed cell div rate
                                ], dtype='float64')
                                cost = cost_function(params)
                                # print(cost)
                                results.append(f'{r1}, {r2}, {r3}, {r4}, {r5}, {r6}, {r7}, {r8}, {cost}\n')

with open(OUTPUT_FILE, 'w') as file:
    file.writelines(results)
