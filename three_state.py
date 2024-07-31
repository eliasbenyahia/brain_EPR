import numpy as np
import os
import sample.hm_simu as hms
from sample.ps_entropy import compute_entropy_n
from sample.utils import *
import json
import pickle


prob = np.linspace(.1,.5, 5)
Ns = 3
T = 50
nu = 3
dt = .001

N = 10
sigma_infered = np.zeros((N,5))
sigma_ex = np.zeros((N,5))

n_iter = 100
Np = 100

for j,p in enumerate(prob):
    W = hms.asym_rw_trans_mat(Ns,p,nu)
    for k in range(N):
        s_in = int(np.random.randint(0,Ns))
        t_jump, ss, R = hms.simul_traj(s_in,W, T)
        t_dis,  ss_dis = hms.discretise(t_jump, ss, T, dt)
        nn = hms.represent(ss_dis)

        sigma_p, pos, _= compute_entropy_n(nn, dt, n_iter=n_iter, Np=Np)

        sigma_infered[k,j] = sigma_p
        sigma_ex[k,j] = R/T

output_dir_path = "output"
if not os.path.isdir(output_dir_path):
    os.mkdir(output_dir_path)

output_three_state_dir_path = os.path.join(output_dir_path, "three_state")
if not os.path.isdir(output_three_state_dir_path):
    os.mkdir(output_three_state_dir_path)

param_dict = make_param_dict(Ns,N, T, prob, nu, dt, n_iter, Np)
param_filename = os.path.join(output_three_state_dir_path, "param.json")
with open(param_filename,'w') as param_file:
    param_file.write(json.dumps(param_dict))

sigma_infered_filename = os.path.join(output_three_state_dir_path, "infered_EPR.pkl")
with open(sigma_infered_filename, 'wb') as f:
    pickle.dump(sigma_infered, f)

sigma_exact_filename = os.path.join(output_three_state_dir_path, "exact_EPR.pkl")
with open(sigma_exact_filename, 'wb') as f:
    pickle.dump(sigma_ex, f)