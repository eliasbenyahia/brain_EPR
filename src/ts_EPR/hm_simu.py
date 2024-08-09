#import

import numpy as np
import scipy.interpolate
from .ps_entropy import compute_entropy_n


class Ctmc:
    """_summary_
    """
    def __init__(self, Ns, W, compute_epr = True):
        self.Ns = Ns
        self.W = W
        self.trajs = []
        self.epr = None
        if compute_epr:
            self.compute_epr()
    
    def compute_epr(self):
        self.epr = 0
        for i in range(self.Ns):
            for j in range(i+1, self.Ns):
                self.epr += self.W[i,j] * np.log(self.W[i,j] / self.W[j,i] )

    @property
    def n_traj(self):
        return len(self.trajs)
    
    @property
    def trajs_epr(self):
        return np.array([traj.epr for traj in self.trajs])
    
    @property
    def trajs_epr_infered(self):
        return np.array([traj.epr_infered for traj in self.trajs])

    def display(self, show_trajs=False):
        print('\nNumber of states:', self.Ns)
        print('Transition rate matrix:', self.W)

        print('\nNumber of trajectories:', self.n_traj)
        if show_trajs:
            for i, traj in enumerate(self.trajs):
                print('\nTraj number:', i)
                traj.display()

    
    def propagate(self, duration, s_in):
        """Gillespie algorithm to generate a trajectory in a ctMc.

        Args:
            duration (float): total time of simulation
            s_in (int in {0,1,...,len(W)}): initial state
            
        Returns:
            list of float, list of int, float: jump times, states just after jump times, reversibility of the trajectory
        """
        current_s = s_in
        t = 0
        ep = 0
        ss = [current_s]
        t_jumps = [t]

        while t < duration:
            r1 = np.random.rand()
            r2 = np.random.rand()

            rates = np.copy(self.W[:, current_s])
            rates[current_s] = 0
            total_rate = rates.sum()
            tau = 1/total_rate

            t = t + tau * np.log(1/r1)
            
            cumulative_sum = np.cumsum(rates)
            reaction_index = np.searchsorted(cumulative_sum, r2 * total_rate)
            ep += np.log(self.W[reaction_index,current_s]/self.W[current_s,reaction_index])
            current_s = reaction_index

            ss.append(current_s)
            t_jumps.append(t)
        epr = ep / duration
        return t_jumps, ss, epr
    
    def add_trajectory(self, duration, s_in, dt=0.1, compute_epr=True, n_iter = 100, Np=50):
        """propagate and save one traj in self.trajs.

        Args:
            duration (float): total time of simulation.
            s_in (int in {0,1,...,len(W)}): initial state.
            discretize (bool, optional): generate discretized trajectory if set to True. Defaults to False.
            dt (float, optional): time bin for discretisation if discretise is set to True. Default is 0.1.

        Returns:
            _type_: _description_
        """
        t_jumps, ss, epr = self.propagate(duration, s_in)
        new_traj = Traj(duration=duration, t_jumps=t_jumps, ss=ss, epr=epr)
        
        if compute_epr:
            new_traj.compute_entropy(dt, n_iter = n_iter, Np = Np)

        self.trajs.append(new_traj)

        return 0
    
    def add_trajectories(self, N_traj, durations, s_ins, dt=0.1, compute_epr=True, n_iter = 100, Np=50):
        """Propagate and save several traj in self.trajs.

        Args:
            N_traj (int): number of trajectories to generate and save
            durations (int or list of float): durations of trajectories
            s_ins (int or list of int): initial states
            discretise (bool, optional): generate discretized trajectory if set to True. Defaults to False.
            dt (float, optional): time bin for discretisation if discretise is set to True. Default is 0.1.
        """
        if isinstance(durations, int) or isinstance(durations, float):
            for i in range(N_traj):
                self.add_trajectory(durations, s_ins, dt=dt, compute_epr=compute_epr, n_iter = n_iter, Np=Np)       
    
        else:  
            for i in range(N_traj):
                self.add_trajectory(durations[i], s_ins[i], dt=dt, compute_epr=compute_epr, n_iter = n_iter, Np=Np)   

    def recompute_epr(self):
        for traj in self.trajs:
            ep = 0
            for i in range(len(traj._ss) - 1):
                ep += np.log(self.W[traj._ss[i+1],traj._ss[i]]/self.W[traj._ss[i],traj._ss[i+1]])
            epr = ep / traj._duration
            traj._epr = epr

class Traj:
    def __init__(self, **kwargs):
        self._duration = kwargs.get('duration', 0.)
        self._t_jumps = kwargs.get('t_jumps', [])
        self._ss = kwargs.get('ss', [])
        self._epr = kwargs.get('epr', None)
        self._epr_infered = kwargs.get('infered_epr', None)
        self._dt = kwargs.get('dt', None)
        try:
            if len(self._t_jumps) != len(self._ss):
                raise ValueError
        except ValueError:
            print("Number of jump times different from number of successive state. Should be equal numbers.")
    
    @property
    def s_in(self):
        if self._ss == []:
            print("Empty trajectory")
            return None
        else:
            return self._ss[0]
    
    @property
    def epr(self):
        return self._epr
    
    @property
    def epr_infered(self):
        return self._epr_infered
    
    @property
    def dt(self):
        return self._dt
    
    def display(self):
        print('Duration:', self._duration)
        print('Jump times:', self._t_jumps)
        print('Successive states:', self._ss)
        print('Entropy production rate:', self.epr)

    def __set_duration(self, T):
        self._duration = T

    
    def discretise(self, dt):
        """discretise a trajectory.

        Args:
            t_jump (list): jumping times
            ss (list): states
            T (float): total time of simulation
            dt (float): discretisation time

        Returns:
            array (shape (int(T/dt),)), array(shape (int(T/dt),)): step times, states at step times, array representation of the time series
        """
        Nt = int(self._duration/dt)
        ss_interp = scipy.interpolate.interp1d(self._t_jumps,self._ss, kind='previous')
        t_jumps_dis = np.linspace(0,self._duration-dt, Nt)
        ss_dis = ss_interp(t_jumps_dis).astype(int)
        self._dt = dt
        return t_jumps_dis,  ss_dis
        
    def binarise(self,  ss_dis):
        """build an array representation of the time series.

        Args:
            ss_dis (array(shape (int(T/dt),))): states at step times

        Returns:
            numpy.ndarray (shape (Ns,int(T/dt))): _description_
        """
        Nt = len(ss_dis)
        visited_states, inverse = np.unique(ss_dis, return_inverse=True)
        Ns_visited = len(visited_states)
        bins = np.zeros((Ns_visited, Nt))
        for i in range(Nt):
            bins[inverse[i],i] = 1
        return  bins
    
    def compute_entropy(self, dt, n_iter=50, Np=100):
        """compute entropy production rate of the trajectory using pyswarm algo

        Args:
            n_iter (int, optional): number of iteration in pyswarm optimisation. Defaults to 50.
            Np (int, optional): number of particles in pyswarm optimisation. Defaults to 100.
        
        Returns:
            _type_: _description_
        """
        t_jumps_dis,  ss_dis = self.discretise(dt)
        bins = self.binarise(ss_dis)
        sigma, pos, [data_mid, data_diff] = compute_entropy_n(bins, self.dt, n_iter=n_iter, Np=Np)
        self._epr_infered = sigma
        return sigma


# def compute_epr(W):
#     epr = 0
#     Ns = len(W)
#     for i in range(Ns):
#         for j in range(i+1, Ns):
#             epr += W[i,j] * np.log(W[i,j] / W[j,i] )
#     return epr


