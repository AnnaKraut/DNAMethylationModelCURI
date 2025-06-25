# number of iterations before check:
import math

import numpy as np


N = 1000

# cost threshold for viable points:
THRESHOLD = .5

# for inverse temperature scaling:
b = 10
f_0 = .3

# for covariance matrix scaling
s = 10
f_l = .3
f_u = .7

# lots of helper functions along the way:
def cost_function(param_point):
  return sum(param_point) # temporary cost function just for testing

def sample_new(param_point, covariance):
  new_point = np.random.multivariate_normal(param_point, covariance)
  # todo: add limits to new params, e.g. floor and ceiling of parameter ranges?
  return new_point

def accept(old_point, new_point, inverse_temp):
  """
  returns true if we should transition to new_point, false otherwise

  We transition if E(theta) < E(theta_old),
  otherwise with probability exp(-inverse_temp * (E(theta) - E(theta_old)))
  """
  new_cost = cost_function(new_point)
  old_cost = cost_function(old_point)

  if new_cost < old_cost:  # E(theta) < E(theta_i)
    accepted = True
  else:
    n = np.random.uniform()
    check = math.exp(-inverse_temp*(new_cost - old_cost))
    if check > n:       # accept
      accepted = True
    else:               # reject
      accepted = False

  return accepted

def update_temp(inverse_temp, f_v):
  if f_v == 0:
    return b * inverse_temp
  elif f_v <= f_0:
    return inverse_temp
  else:
    return inverse_temp / b

def update_covar(covariance, f_a):
  if f_a > f_u:
    return [[s * x for x in row] for row in covariance]
  elif f_a > f_l:
    return covariance
  else:
    return [[x / s for x in row] for row in covariance]

def fit_ellipsoids(viable_points):
  pass

def test_convergence(volumes):
    pass

# phase one: Out-of-Equilibrium Adaptive Monte Carlo sampling
def oeamc(start_point, inverse_temp, covariance):

  # initialization
  i = 0
  viable_points = []
  volumes = []
  last_point = start_point

  condition = True
  while condition:
    i += 1

    num_viable = 0
    num_transitions = 0
    # loop N times then check condition
    for j in range(N):
      # choose new parameter point
      new_point = sample_new(last_point, covariance)

      # VIABILITY check
      cost = cost_function(new_point)
      if cost < THRESHOLD:
        viable_points.append(new_point)
        num_viable += 1

      # TRANSITION check
      accept_point = accept(last_point, new_point, inverse_temp)
      if accept_point:
        last_point = new_point
        num_transitions += 0

    # check ellipsoid stuff
    volumes.append(fit_ellipsoids(viable_points))

    # check for convergence
    if test_convergence(volumes):
      condition = False
    else:
      covariance = update_covar(covariance, num_viable/N)
      inverse_temp = update_temp(inverse_temp, num_transitions/N)


  return viable_points