import numpy as np
import sample.hm_simu as hms
from sample.ps_entropy import compute_entropy_n



prob = np.linspace(.1,.5, 5)
Ns = 3
T = 50
nu = 3
dt = .001

N = 5
sigma_infered = np.zeros((N,5))
sigma_ex = np.zeros((N,5))

n_iter = 100
Np = 200

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