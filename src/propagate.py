### import class and functions ###

import ts_EPR.hm_simu as hms
import ts_EPR.utils as utils

### build the transition matrix and create the associated process ###

# Parameters for our ctmc 
Ns = 3
p = .4
nu = 2

W = utils.asym_rw_trans_mat(Ns, p, nu)
process = hms.Ctmc(Ns, W)

### add one trajectory ###

# parameters for trajectories
duration = 10
s_in = 0

process.add_trajectory(duration, s_in)

# display the results
process.display(show_trajs=True)

### add several trajectories ###

n_trajs = 9
process.add_trajectories(n_trajs, duration, s_in, discretise=True, binarise=True)

# display the results
process.display(show_trajs=False)