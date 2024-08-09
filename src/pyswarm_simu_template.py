import numpy as np
import scipy.linalg as LA
import os
import pickle
import pandas as pd
from sklearn.decomposition import PCA

# Import entropy computations functions
from ts_EPR.ps_entropy import compute_entropy_n


# optimization parameters

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
Np=200
n_iter = 500
n_components_range= [3, 4, 5, 6, 7] # for PCA
shifts = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5]
shifts = [2.0, 2.5]
dt = 1
data = []

set = '35'
corr_width = 3
# Paths to data
dir_simu = os.path.join('..', 'simu', 'perlin')
dir_data = os.path.join(dir_simu, 'data')
dir_simu_output = os.path.join(dir_simu, 'output')
if not os.path.isdir(dir_simu_output):
    os.mkdir(dir_simu_output)
# dir_output_allshifts = os.path.join(dir_simu_output, 'simplex_noise_' + set + '_2_0.5_36_std_1.0_1.5_allshifts_0.75_0.5_8.0_drive_10.0_30.0_transfer_50.0_0.25')
# if not os.path.isdir(dir_output_allshifts):
#     os.mkdir(dir_output_allshifts)

save_singular_values = False
save_singular_vectors = False

for s,shift in enumerate(shifts):
    df = pd.DataFrame()
    data_param = "simplex_noise_" + str(corr_width) + '_' + set + "_2_0.5_36_std_1.0_1.5_" + str(shift) + "_0.75_0.5_8.0_drive_10.0_30.0_transfer_50.0_0.25"
    dir_data_param = os.path.join(dir_data, data_param )
    dir_output_param = os.path.join(dir_simu_output, data_param)
    if not os.path.isdir(dir_output_param):
        os.mkdir(dir_output_param)
    output_filename = os.path.join(dir_output_param, 'EPR_pyswarm_Np' + str(Np) + '_iter' + str(n_iter) + '.csv' )

    for i in range(20):
        print('\nshift:', shift)
        print('seed:', i)
        filename_rate = os.path.join(dir_data_param, "rate_simplex_noise_baseline_"+ str(i) +".bn")
        sv_filename = os.path.join(dir_output_param, 'sv_seed' + str(i) +'.pkl')
        lsv_filename = os.path.join(dir_output_param, 'lsv_seed' + str(i) +'.pkl')
        rsv_filename = os.path.join(dir_output_param, 'rsv_seed' + str(i) +'.pkl')
        
        
        with open(filename_rate, 'rb') as f:
            data = pickle.load(f)[0]

        # U, svs, VT = randomized_svd(data, 
        #                 n_components=n_components,
        #                 n_iter=5,
        #                 random_state=None)

        # U, svs, VT = LA.svd(data, full_matrices=False)
        # scale = np.linalg.norm(svs) / np.sqrt(len(data))

        pca = PCA(n_components=n_components_range[-1])
        data_pca = pca.fit_transform(data.T)
        for n_components in n_components_range:
            
            print('n_components:', n_components)

            # data_pca=  scale * VT[:n_components, :]

            sigma, pos,_= compute_entropy_n(data_pca.T[:n_components], dt, n_iter=n_iter, Np=Np)

            df_simu = pd.DataFrame.from_dict({'sigma' :[sigma], 'corr_width': [corr_width], 'shift' : [shift], 'seed': [i], 'PCA' : [n_components]})
            df = pd.concat([df,df_simu], ignore_index=True)
            if save_singular_values:
                with open(sv_filename, 'wb') as f:
                    pickle.dump(svs, f)

            if save_singular_vectors:
                with open(lsv_filename, 'wb') as f:
                    pickle.dump(U, f)
                with open(rsv_filename, 'wb') as f:
                    pickle.dump(VT, f)
                
            
    # writing to csv file
    df.to_csv(output_filename)


                
