# number of iterations before check:
import math
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

import numpy as np
from numpy import linalg


# number of steps before reevaluating covariance and temperature
N = 1000

# cost threshold for viable points:
THRESHOLD = 0.5

# for inverse temperature scaling:
b = 10
f_0 = 0.3

# for covariance matrix scaling:
s = 10
f_l = 0.3
f_u = 0.7

# for checking if ellipsoid volumes are converging:
LIMIT = 100


# lots of helper functions along the way:
def cost_function(param_point):
    return sum(param_point)  # temporary cost function just for testing


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
        check = math.exp(-inverse_temp * (new_cost - old_cost))
        if check > n:  # accept
            accepted = True
        else:  # reject
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

# taken from https://github.com/minillinim/ellipsoid/tree/master
# TODO: modify to keep only essential bits (we only need volume?) and possibly add improvements from paper (bootstrap points?)
def fit_mvee(P=None, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
            [x,y,z,...],
            [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)
    
    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
                    np.dot(P.T, np.dot(np.diag(u), P)) - 
                    np.array([[a * b for b in center] for a in center])
                    ) / d
                    
    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    
    return (center, radii, rotation)

# TODO: possibly implement my own kmeans clustering to speed things up and avoid bloat
def fit_ellipsoids(viable_points):
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(viable_points)

    dim = len(viable_points[0])

    kmeans = KMeans(
        init="random",
        n_clusters=dim,
        n_init=10,
        max_iter=300,
    )
    kmeans.fit(scaled_data)
    
    partitions = np.array([[] for i in range(dim)])
    # loop over each cluster and fit an ellipsoid
    for i in reversed(kmeans.labels_):
        partitions[i].append(viable_points.pop())
    
    total_volume = 0
    for i in range(dim):
        # fits an ellipsoid to the partition
        fit = fit_mvee(partitions[i])
        
        # calculates the volume of the ellipsoid
        vol = math.pi ** (dim/2) / math.gamma(dim/2 + 1)
        for radius in fit[1]:
            vol *= radius
        
        # adds the volume to the total
        total_volume += vol
    
    return total_volume

# TODO: figure out what it means for the volumes to converge
def test_convergence(volumes):
    # if we haven't done enough loops yet
    loop_count = len(volumes)
    if loop_count < 4:
        return False

    # then check if we have reached our upper limit
    if loop_count >= LIMIT:
        return True

    # check if the last four volumes are converging


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
            covariance = update_covar(covariance, num_viable / N)
            inverse_temp = update_temp(inverse_temp, num_transitions / N)

    return viable_points
