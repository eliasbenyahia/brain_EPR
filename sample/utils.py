import numpy as np

def make_param_dict(Ns,N, T, prob, nu, dt, n_iter, Np, options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}):
    param_sim = {}
    param_sim['Ns'] = Ns
    param_sim['n_trials'] = N
    param_sim['T'] = T
    param_sim['prob'] = prob.tolist()
    param_sim['nu'] = nu
    param_sim['dt'] = dt

    param_opti = {}
    param_opti['n_iter'] = n_iter
    param_opti['Np'] = Np
    param_opti['options'] = options

    param = {}
    param['param_sim'] = param_sim
    param['param_opti'] = param_opti

    return param