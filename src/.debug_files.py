import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pickle 
import ts_EPR.hm_simu as hms
import ts_EPR.utils as utils
# path to data files
output_dir_path = "output"
output_three_state_dir_path = os.path.join(output_dir_path, "three_state")

# load data and param files
param_filename = os.path.join(output_three_state_dir_path, "param.json")
with open(param_filename) as json_file:
    param = json.load(json_file)

# useful parameters
prob = np.array(param['param_sim']['prob'])[:4]


for i,p in enumerate(prob):
    ctmc_filename = os.path.join(output_three_state_dir_path, "ctmc_p%.1f.pkl"%(p))
    with open(ctmc_filename, 'rb') as f:
        ctmc = pickle.load(f)

    ctmc.recompute_epr()
    with open(ctmc_filename, 'wb') as f:
        pickle.dump(ctmc, f)