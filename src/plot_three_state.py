import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pickle 
import ts_EPR.hm_simu as hms
import ts_EPR.utils as utils
import seaborn as sns
import pandas as pd
# path to data files
output_dir_path = "output"
output_three_state_dir_path = os.path.join(output_dir_path, "three_state")

# load data and param files
param_filename = os.path.join(output_three_state_dir_path, "param.json")
with open(param_filename) as json_file:
    param = json.load(json_file)

# useful parameters
prob = np.array(param['param_sim']['prob'])
N = param['param_sim']['n_trials']
nu = param['param_sim']['nu']

epr_infered = np.zeros((len(prob), N))
epr_exact = np.zeros((len(prob), N))

for i,p in enumerate(prob):
    ctmc_filename = os.path.join(output_three_state_dir_path, "ctmc_p%.1f.pkl"%(p))
    with open(ctmc_filename, 'rb') as f:
        ctmc = pickle.load(f)
    epr_infered[i] = ctmc.trajs_epr_infered
    epr_exact[i] = ctmc.trajs_epr

df_exact = pd.DataFrame(epr_exact.T, columns = prob)
df_exact = df_exact.melt(var_name='prob', value_name='epr')
df_exact['type'] = 'exact'

df_infered = pd.DataFrame(epr_infered.T, columns = prob)
df_infered = df_infered.melt(var_name='prob', value_name='epr')
df_infered['type'] = 'infered'

df = pd.concat([df_exact, df_infered], axis=0)

prob_th = np.linspace(.1, .9, 91)
epr_th = utils.compute_epr_rw(prob_th, nu)

fig, ax = plt.subplots(1, figsize=(6,4))

plt.plot(prob_th, epr_th, color='red', label='theory')

palette = ('green', 'orange')
vplot = sns.violinplot(ax=ax, data=df, x='prob', y='epr', hue='type', split=True, inner=None, native_scale=True, gap=.1, density_norm='width', alpha=.5, palette=palette, cut=0)

plt.show()