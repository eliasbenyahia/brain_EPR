# IMPORT PYSWARM
import numpy as np
import pyswarms as ps
from pyswarms.utils.functions import single_obj as fx

# Define default set of options
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}


def dLinear(data,pars):
    """Matrix product between matrix pars and vector data.

    Args:
        data (np.array): shape (n)
        pars (np.array): must be shape (n*n)

    Returns:
        np_array: shape n
    """
    dimension = data.shape[0]
    pars_m = pars.reshape((dimension, dimension))
    f = np.dot(pars_m, data)
    return f


def compute_entropy_n(data, dt, n_iter=50, Np=100, options = options):
    """Compute entropy for n-dimensional time-series

    Args:
        data (array): n-dimensional time-series
        dt (float): time interval of time-series
        n_iter (int, optional): number of iteration in pyswarm optimisation. Defaults to 50.
        Np (int, optional): number of particles in pyswarm optimisation. Defaults to 100.
        options (dict, optional): options for pyswarm optimisation. Defaults to {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}.

    Returns:
        sigma, pos, [data_mid, data_diff]: infered epr, optimal function, [data_mid, data_diff]
    """
    dout = data
    dimension, length=np.shape(dout)
    optimizer = ps.single.LocalBestPSO(n_particles=Np, dimensions=dimension**2, options=options)
    data_mid = (data[:,1:] + data[:,:length-1])/2
    data_diff = data[:,1:] - data[:,:length-1]

    
    def costLinear(pars):  
        JJ=np.sum(dLinear(data_mid,pars)* data_diff, axis=0)
        return 2*(np.mean(JJ))*(np.mean(JJ))/(np.var(JJ)*dt)
    
    
    def costLinear2(pars):  
        return 1/(costLinear(pars)+1)
    
    def fun4(x):
        dime = Np
        cosi=np.zeros(Np)
        b=x
        for im in range(Np):
            cosi[im]=costLinear2(b[im,:])
        return cosi
    
    cost, pos = optimizer.optimize(fun4, iters=n_iter)
    sigma = 1/cost-1
    return sigma, pos, [data_mid, data_diff]