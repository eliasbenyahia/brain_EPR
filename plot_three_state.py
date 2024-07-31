import numpy as np
import os
import matplotlib.pyplot as plt
import json
import pickle 

# path to data files
output_dir_path = "output"
output_three_state_dir_path = os.path.join(output_dir_path, "three_state")
sigma_infered_filename = os.path.join(output_three_state_dir_path, "infered_EPR.pkl")
sigma_exact_filename = os.path.join(output_three_state_dir_path, "exact_EPR.pkl")
param_filename = os.path.join(output_three_state_dir_path, "param.json")

# load data and param files
with open(param_filename) as json_file:
    param = json.load(json_file)
with open(sigma_infered_filename, 'rb') as f:
    sigma_infered = pickle.load(f)
with open(sigma_exact_filename, 'rb') as f:
    sigma_exact = pickle.load(f)

# useful parameters
prob = np.array(param['param_sim']['prob'])
nu = param['param_sim']['nu']

### PLOT ###
fig, ax = plt.subplots(1, figsize=(10,6))

# Plot theory
prob_th = np.linspace(.1,.5, 51)
sigma_th = prob_th*nu * np.log(prob_th/(1-prob_th)) + (1-prob_th)*nu * np.log((1-prob_th)/prob_th)
plt.plot(prob_th, sigma_th, color='red', label = 'Theory for average')

# plot exact values
std_ex = np.std(sigma_exact, axis=0)
plt.scatter(prob, sigma_exact.mean(axis=0), color='green', label = 'exact')
plt.errorbar(prob, sigma_exact.mean(axis=0),std_ex,fmt = 'o', color = 'green')

# plot infered values
sigma_exp = np.mean(sigma_infered, axis=0)
error = np.std(sigma_infered, axis=0)
plt.scatter(prob,sigma_exp, color='orange', label='Infered using TUR')
plt.errorbar(prob, sigma_exp,error,fmt = 'o', color = 'orange')

plt.show()