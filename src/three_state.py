import numpy as np
import os

import ts_EPR.hm_simu as hms
from ts_EPR.ps_entropy import compute_entropy_n
import ts_EPR.utils as utils
import json
import pickle

### Prepare output folder ###

output_dir_path = "output"

if not os.path.isdir(output_dir_path):
    os.mkdir(output_dir_path)

output_three_state_dir_path = os.path.join(output_dir_path, "three_state")
if not os.path.isdir(output_three_state_dir_path):
    os.mkdir(output_three_state_dir_path)


### set parameters ###

prob = np.linspace(.5,.5, 1)
Ns = 3
T = 50
nu = 3
dt = .001

N = 100 # number of repetirion for p in prob

n_iter = 100
Np = 100

# save parameters in a dict
param_dict = utils.make_param_dict(Ns, N, T, prob, nu, dt, n_iter, Np)
param_filename = os.path.join(output_three_state_dir_path, "param.json")
with open(param_filename,'w') as param_file:
    param_file.write(json.dumps(param_dict))


### start simulation ###

for j,p in enumerate(prob):
    W = utils.asym_rw_trans_mat(Ns,p,nu)
    process = hms.Ctmc(Ns, W, compute_epr=True)
    s_ins = np.random.randint(0,Ns, N).astype(int)
    Ts = T * np.ones(N)
    process.add_trajectories(N, Ts, s_ins, dt=dt, compute_epr=True, n_iter = n_iter, Np=Np)
    
    process_filename = os.path.join(output_three_state_dir_path, "ctmc_p%.1f.pkl"%p)
    with open(process_filename, 'wb') as f:
        pickle.dump(process, f)


