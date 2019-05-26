import numpy as np
from scipy.optimize import curve_fit

def calcEnergy(config):
    """
    Calculate the Enegy of a given configuration
    :param config (numpy array):
                                Ising configuration LxL
    :return (float):
                    Energy
    """
    energy = 0
    L = config.shape[0]
    for i in range(L):
        for j in range(L):
            S = config[i,j]
            nb = config[(i+1)%L, j] + config[i,(j+1)%L] + config[(i-1)%L, j] + config[i,(j-1)%L]
            energy += -nb*S
    return energy/(4*L*L)

def correlation_function(arr,radius):
    """
    Calculate the correlation function for a given Ising configuration with the 4 neighbours at a given distance
    G(r_i,r_j) = <s_is_j> - <s_i><s_j>

    :param arr (numpy array):
                Ising configuration LxL

    :param radius (int) :
                        distance of the neighbours with which calculate the correlation function
    :return (float): correlation function
    """
    ij=0
    L = arr.shape[0]
    for i in range(L):
        for j in range(L):
            ij+= arr[i, j]*(
                arr[((i - radius) % L), j] +  # -1 in i-dimension
                arr[((i + radius) % L), j] +  # +1 in i-dimension
                arr[i, ((j - radius) % L)] +  # -1 in j-dimension
                arr[i, ((j + radius) % L)]    # +1 in j-dimension
            )
    magn = np.mean(arr)
    return ij/(L*L*4) -magn**2

def func_G(r, theta, xi):
    """
    Theoretical exponential decay of correlation function away from the critical temperature
    G(r;T) = r^(-theta) *e^(-r/xi)
    :param r (int): radius
    :param theta:
    :param xi:
    :return:
    """
    return r**(-theta) * np.exp(-r / xi)

def correlation_lenght(correlation_r_T,radii,param_variance = False):
    """
    Find the best fit for the theoretical exponential decay of the correlation function
    :param correlation_r_T (numpy array, shape(#T,#radii):
                                                            Numpy array containing correlation for different radii and
                                                            temperatures calculated using correlation_function.

    :param radii (list):
                        list of all radii considered for correlation function

    :param param_variance (bool):
                                if True returns the variance of the estimated parameters.
    :return:
            xi: list of estimated correlation lenght at different temperatures .
            param_var: if param_variance is True return also the variance of xi estimations.
    """
    xi = []
    param_var = []
    for i in range(correlation_r_T.shape[0]):
        popt, pcov = curve_fit(func_G, radii, correlation_r_T[i,:])
        xi.append(popt[1])
        param_var.append(pcov)
    if param_variance:
        return xi, param_var
    else:
        return xi