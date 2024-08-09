import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pickle

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



def make_anim(filename_rate, filename_gif, rows = 36, tstart=200, tfin = 1200, tstep=12, cmap='OrRd'):

    with open(filename_rate, 'rb') as f:
        ts = pickle.load(f)[0]
    fig, ax = plt.subplots()
    im = ax.imshow(ts[:rows*rows,tstart].reshape(rows,rows), origin='lower', cmap=cmap, vmin=0, vmax=1)
    fig.colorbar(im,  ticks = [0,0.5,1])


    def update(frame):
        # for each frame, update the data stored on each artist.
        data = ts[:rows*rows,tstart + frame].reshape(rows,rows)
        # update the image:
        im.set_data(data)
        # update the title:
        ax.set_title('time = '+str(tstart + frame))
        return (im)


    ani = animation.FuncAnimation(fig=fig, func=update, frames=range(0,tfin - tstart ,tstep), interval=1)
    ani.save(filename=filename_gif, writer="pillow")


def asym_rw_trans_mat(Ns, p, nu):
    """build a transition matrix for an asymetric 1D random walk process with periodic boundary conditions.

    Args:
        Ns (int): number of states
        p (float): probability of going up
        nu (float): jump rate
        
    """
    W = np.zeros((Ns,Ns))
    for i in range(Ns):
        W[(i+1) % Ns, i] = (1-p) * nu
        W[i, (i+1) % Ns] = p * nu
        W[i, i] = -nu
    return W

def compute_epr_rw(p, nu):
    return p*nu * np.log(p/(1-p)) + (1-p)*nu * np.log((1-p)/p)