import numpy as np
from sys import argv
import h5py
import datetime


def nn_sum(x, i, j):
    """
    Args:
        x: Spin configuration
        i, j: Indices describing the position of one spin

    Returns:
        Sum of the spins in x which are nearest neighbors of (i, j)
    """
    result = x[(i+1)%L, j] + x[(i-1)%L, j]
    result += x[i, (j+1)%L] + x[i, (j-1)%L]
    return int(result)


def move(x):
    """
    Args:
        x: Spin configuration

    Returns:
        Updated x after one Monte Carlo move
    """
    # pick one spin at random
    i = int(L*np.random.rand())
    j = int(L*np.random.rand())
    x_old = x[i, j]

    # flip the spin according to the Metropolis algorithm
    nn = nn_sum(x, i, j)
    if x_old == 1:
        R = table_spin_up[int((nn+4)/2)] # Metropolis acceptance probability
    else:
        R = table_spin_down[int((nn+4)/2)]
    eta = np.random.rand()
    if R > eta:
        x[i, j] *= -1

    return x


L = 128 # size of the system
J = 1 # coupling constant
Nthermalization = 100*L**3 # number of thermalization steps
Nconfig = 10000 # number of configurations
configs = np.zeros((Nconfig, L, L), dtype=np.int8) # configuration storage

# read in temperature
if (len(argv) == 1):
    print('python3 ising_data.py T')
    exit()
T = float(argv[1])

# probability look-up tables
table_spin_up = np.exp(-2.0*J*np.array([-4, -2, 0, 2, 4])/T)
table_spin_down = np.exp(+2.0*J*np.array([-4, -2, 0, 2, 4])/T)

for ns in range(Nconfig):
    print('T = %f ns = %d '%(T, ns+1))

    # random initial configuration
    x = np.ones((L, L))
    for i in range(L):
        for j in range(L):
            if np.random.rand() < 0.5:
                x[i, j] = -1

    # thermalization
    for nt in range(Nthermalization):
        x = move(x)
    configs[ns] = x

data_file = h5py.File('ising_data.hdf5', 'a') # load/create data file
data_file.create_dataset('T=%f'%T, data=configs)
data_file.close()
print('Finished ' + str(datetime.datetime.now()))
