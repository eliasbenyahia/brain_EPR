#import

import numpy as np
import scipy.interpolate




def simul_traj(s_in, W, T):
    """Gillespie algorithm to generate a trajectory in a ctMp.

    Args:
        s_in (int in {0,1,...,len(W)}): initial state
        W (np array shape (Ns,Ns)): transition rate matrix
        T (float): total time of simulation

    Returns:
        list of float, list of int, float: jump times, states just after jump times, reversibility of the trajectory
    """
    current_s = s_in
    t=0
    R=0
    ss = [current_s]
    t_jump = [t]

    while t<T:
        r1 = np.random.rand()
        r2 = np.random.rand()

        total_rate = W[:, current_s].sum()
        tau = 1/total_rate

        t = t + tau * np.log(1/r1)
        cumulative_sum = np.cumsum(W[:, current_s])
        reaction_index = np.searchsorted(cumulative_sum, r2 * total_rate)
        R += np.log(W[reaction_index,current_s]/W[current_s,reaction_index])
        current_s = reaction_index
        ss.append(current_s)
        t_jump.append(t)

    return t_jump, ss, R

def discretise(t_jump, ss,T, dt):
    """discretise a trajectory.

    Args:
        t_jump (list): jumping times
        ss (list): states
        dt (float): discretisation time

    Returns:
        array (shape (int(T/dt),)), array(shape (int(T/dt),)): step times, states at step times, array representation of the time series
    """
    Nt = int(T/dt)
    ss_interp = scipy.interpolate.interp1d(t_jump,ss, kind='previous')
    t_dis = np.linspace(0,T-dt, Nt)
    ss_dis = ss_interp(t_dis).astype(int)
    return t_dis,  ss_dis

def represent(ss_dis):
    """build an array representation of the time series.

    Args:
        ss_dis (array(shape (int(T/dt),))): states at step times
        Ns (int): number of states in the system

    Returns:
        array(shape (Ns,int(T/dt))): _description_
    """
    Nt = len(ss_dis)
    visited_states, inverse = np.unique(ss_dis, return_inverse=True)
    Ns_visited = len(visited_states)
    nn = np.zeros((Ns_visited, Nt))
    for i in range(Nt):
        nn[inverse[i],i] = 1
    return  nn

def asym_rw_trans_mat(Ns, p, nu):
    """build a transition matrix for an asymetric 1D random walk process with periodic boundary conditions.

    Args:
        Ns (_type_): number of states
        p (_type_): probability of going up
        nu (float): jump rate
        
    """
    W = np.zeros((Ns,Ns))
    for i in range(Ns):
        W[(i+1)%Ns,i] =1-p
        W[i,(i+1)%Ns] =p
    return nu*W
