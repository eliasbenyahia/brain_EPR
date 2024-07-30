import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.linalg as LA
import numpy.matlib
import os
import pickle
import mne
from mne.minimum_norm import read_inverse_operator
import csv
import scipy.interpolate
import scipy.stats

# Import entropy computations functions
from entropy.ps_entropy import compute_entropy_n

# Import pca
import sklearn.decomposition

import json 



# optimization parameters

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
Np=100
n_iter = 100
n_components=11 # for PCA

# Paths to data

subjects_dir = "/archive/21098_pd_dimensionality/MEG"
fs_dir = "/home/pashel/Python/24109_pd_network/fs_subjects_dir"



list_subjs=os.listdir(subjects_dir)
list_output=os.listdir('/home/pashel/Python/Elias/output')
list_todo=[]
for subj in list_subjs:
    if subj + '_output_ses2' not in list_output:
        list_todo.append(subj)
        

for subject in list_subjs[:-1]:
    #subject = '0314'
    print(subject)
    try:
        for ses in ['ses1','ses2']:
            # Paths to output
            
            output_dir = os.path.join('/home/pashel/Python/Elias/output', subject + '_output_' + ses)
    
            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)
                
            # Filename for saving eigenvalues and eigenvectors
            eig_filename = os.path.join(output_dir, subject + '_' + ses + '_full_brain_eig.pkl')
            eig_vec_filename = os.path.join(output_dir, subject + '_' + ses + '_full_brain_eig_vec.pkl')
    
            # Source data filename
            stc_filename = os.path.join(subjects_dir, subject, subject + '-' + ses + '-dSPM-lh.stc')
    
            # Output filename
            output_filename = os.path.join(output_dir, subject + '_' + ses + '_full_brain_PCA' + str(n_components) + '_pyswarm_Np' + str(Np) + '_iter' + str(n_iter) + '.csv' )
    
    
            # Read stc and src
            stc = mne.read_source_estimate(stc_filename)
            data = stc.data

    
            # run optimization
    
            dt = stc.tstep
            
            try:
                print('start computing z_score')
                z_patient = scipy.stats.zscore(data, axis=1)
                print('z_score successfully computed')
            except np.exceptions.AxisError:
                sigma = np.NaN


            L,M = z_patient.shape

            #PCA


            if L>n_components:
                print("computing PCA with %d components..."%n_components)
                pca_patient = sklearn.decomposition.PCA(n_components = n_components)
                data_pca_patient = pca_patient.fit_transform(z_patient.T).T
                print("PCA successfully computed.")
                print("Computing all eigenvalues and eigenvectors...")
                full_PCA = sklearn.decomposition.PCA()
                full_PCA.fit(z_patient.T)
                eig = full_PCA.explained_variance_
                eig_vec=full_PCA.components_
                print('Eigenvalues and eigenvectors successfully computed.')

                #compute entropy production rate
                sigma, pos,_ = compute_entropy_n(data_pca_patient, dt, n_iter=n_iter, Np=Np)
            else:
                sigma = np.NaN

            # Create a dictionary for saving entropy production rate values
            mydict = []
            mydict.append({'region' : 'full_brain', 'sigma' : sigma})
            # field names
            fields = ['region', 'sigma']
    
            # writing to csv file
            with open(output_filename, 'w') as csvfile:
                # creating a csv dict writer object
                writer = csv.DictWriter(csvfile, fieldnames=fields)
    
                # writing headers (field names)
                writer.writeheader()
    
                # writing data rows
                writer.writerows(mydict)
            
            #Save file parameters

            param_dict = {}
            param_dict['subject'] = subject
            param_dict['session'] = ses
            param_dict['n_step'] = L
            param_dict['dt'] = dt

            param_filename = 'data_properties.txt'
    
            with open(param_filename,'w') as param_file:
                param_file.write(json.dumps(param_dict))
                
            # Save eigenvalues and eigenvectors
            with open(eig_vec_filename, 'wb') as f:
                pickle.dump(eig_vec, f)
            with open(eig_filename, 'wb') as f:
                pickle.dump(eig, f)

            
    except Exception as e:
        # Handle the error
        print(f"An error occurred: {e}")
        # Optionally, you can continue to the next iteration
        continue
