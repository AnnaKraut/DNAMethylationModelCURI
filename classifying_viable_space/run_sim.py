import simple_sim as simple_sim
import numpy as np

# NOTE: The purpose of this file is to interface with the MatLab file costfun.m
#       
#       When this script is ran from MatLab, the values in the params array are initialized by MatLab

MAX = 100000000
POP = 100
POP_M = 50
POP_H = 50

# these are intentionally not defined because of how MatLab interfaces with this script
params = np.array([r_hm, 2*r_hm_h, r_hm_h,
          r_uh, 2*r_uh_h, r_uh_h,
          r_mh, 2*r_mh_h, r_mh_h,
          r_hu, 2*r_hu_h, r_hu_h,
          r_cell_div])

generator = np.random.default_rng()

output = simple_sim.GillespieLongRunFun(MAX, params, POP, POP_M, POP_H, generator)