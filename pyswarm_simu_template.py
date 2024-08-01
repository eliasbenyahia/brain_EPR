import numpy as np
import numpy.linalg as LA
import os
import pickle
import pandas as pd


# Import entropy computations functions
from sample.ps_entropy import compute_entropy_n


# optimization parameters

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
Np=200
n_iter = 500
n_components= 7 # for PCA
shifts = [0.25,0.5,0.75,1.0,1.5,2.0,2.5]
dt = 1
data = []
df = pd.DataFrame()
set = '2.5_35'
# Paths to data

dir_simu = "../simu/perlin"
dir_data = os.path.join(dir_simu, 'data')
dir_simu_output = os.path.join(dir_simu, 'output')
if not os.path.isdir(dir_simu_output):
    os.mkdir(dir_simu_output)
dir_output_allshifts = os.path.join(dir_simu_output, 'simplex_noise_' + set + '_2_0.5_36_std_1.0_1.5_allshifts_0.75_0.5_8.0_drive_10.0_30.0_transfer_50.0_0.25')
if not os.path.isdir(dir_output_allshifts):
    os.mkdir(dir_output_allshifts)

output_filename = os.path.join(dir_output_allshifts, 'EPR_PCA' + str(n_components) + '_pyswarm_Np' + str(Np) + '_iter' + str(n_iter) + '.csv' )

save_singular_values = True

for s,shift in enumerate(shifts):

    data_param = "simplex_noise_" + set + "_2_0.5_36_std_1.0_1.5_"+str(shift)+"_0.75_0.5_8.0_drive_10.0_30.0_transfer_50.0_0.25"
    
    dir_output_data = os.path.join(dir_simu_output, data_param)
    dir_data_param = os.path.join(dir_data,data_param )
           
    for i in range(20):
        print('shift: ', shift)
        print('seed: ', i)
        filename_rate = os.path.join(dir_data_param, "rate_simplex_noise_baseline_"+ str(i) +".bn")
        if not os.path.isdir(dir_output_data):
            os.mkdir(dir_output_data)
        sv_filename = os.path.join(dir_output_data, 'sv_seed' + str(i) +'.pkl')
        lsv_filename = os.path.join(dir_output_data, 'lsv_seed' + str(i) +'.pkl')
        rsv_filename = os.path.join(dir_output_data, 'rsv_seed' + str(i) +'.pkl')
        
        
        with open(filename_rate, 'rb') as f:
            data = pickle.load(f)[0]

        # U, Sigma, VT = randomized_svd(data, 
        #                 n_components=n_components,
        #                 n_iter=5,
        #                 random_state=None)
        U, Sigma, VT = LA.svd(data, full_matrices=False)
        scale = np.linalg.norm(Sigma) / np.sqrt(len(data))
        data_pca=  scale * VT[:n_components,:]
        sigma, pos,_= compute_entropy_n(data_pca, dt, n_iter=n_iter, Np=Np)

        df_simu = pd.DataFrame.from_dict({'sigma' :[sigma], 'shift' : [shift], 'seed': [i]})
        df = pd.concat([df,df_simu], ignore_index=True)
        if save_singular_values:
            with open(sv_filename, 'wb') as f:
                pickle.dump(Sigma, f)
            with open(lsv_filename, 'wb') as f:
                pickle.dump(U, f)
            with open(rsv_filename, 'wb') as f:
                pickle.dump(VT, f)
        
        
# writing to csv file
df.to_csv(output_filename)


            
