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
from sample.ps_entropy import compute_entropy_n

# Import pca and svd
import sklearn.decomposition
from sklearn.utils.extmath import randomized_svd
from mne.source_estimate import _prepare_label_extraction


import json 

# optimization parameters

options = {'c1': 0.5, 'c2': 0.3, 'w':0.9, 'k': 2, 'p': 2}
Np=100
n_iter = 200
n_components=4 # for PCA

# Paths to data

subjects_dir = "/archive/21098_pd_dimensionality/MEG"
fs_dir = "/home/pashel/Python/24109_pd_network/fs_subjects_dir"
spacing = 'ico4'
parc = 'HCPMMP1'
mode = 'PCA_flip'

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
                
            # Filename for saving eigenvalues    
            eig_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + 'eig.pkl')
            eig_vec_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + 'eig_vec.pkl')
            
            # Filename for saving singular values
            sv_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + 'sv.pkl')
            lsv_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + 'lsv.pkl')
            rsv_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + 'rsv.pkl')
            
            compute_eig=False
            save_singular_values = True
            #if os.path.isfile(eig_filename):
            #     compute_eig = False
    
            
            inv_filename = os.path.join(subjects_dir, subject, subject + '-' + ses + '-inv.fif' )
            #src_filename = os.path.join(subjects_dir, subject, 'src', subject + '-' + spacing + '-src.fif' )
            stc_filename = os.path.join(subjects_dir, subject, subject + '-' + ses + '-dSPM-lh.stc')
    
            # Output filename
            output_filename = os.path.join(output_dir, subject + '_' + ses + '_' + parc + '_' + mode + str(n_components) + '_pyswarm_Np' + str(Np) + '_iter' + str(n_iter) + '.csv' )
    
    
            # Read stc and src
            inv = read_inverse_operator(inv_filename)
            src = inv["src"]
            #src = mne.read_source_spaces(src_filename)
            stc = mne.read_source_estimate(stc_filename)

            # read labels
    
            labels_parc = mne.read_labels_from_annot(
                subject, parc=parc, subjects_dir=fs_dir, sort=False
            )
    
            # Remove the indices with brain region name ???
            inds = []
            for label,ind_ in zip(labels_parc,range(len(labels_parc))):
                if '???' not in label.name:
                    inds.append(ind_)
    
    
            label_ts=mne.extract_label_time_course(stc, labels_parc, src, mode=None, allow_empty=True, mri_resolution=True)
            label_vertidx, src_flip = _prepare_label_extraction(stc, labels_parc, src, 'pca_flip', allow_empty=True, use_sparse=False)

    
            # run optimization
    
            dt = stc.tstep
            sigma = np.zeros(len(label_ts))
            sigma[:] = np.nan

            eig=[]
            eig_vec=[]
            
            for i in inds:
                print(i,"/",len(inds))
                data_0 = np.array(label_ts[i])
                
                # compute z_score
                try:
                    z_patient = scipy.stats.zscore(data_0, axis=1)
                except np.exceptions.AxisError:
                    sigma[i] = np.NaN
                    continue
    
                L,M = z_patient.shape
    
                #PCA
    
                if L>n_components:
                    if compute_eig:
                        full_PCA = sklearn.decomposition.PCA()
                        full_PCA.fit(z_patient.T)
                        eig_region = full_PCA.explained_variance_
                        eig.append(eig_region)
                        eig_vec_region=full_PCA.components_
                        eig_vec.append(eig_region)

                    if mode == 'PCA':

                        pca_patient = sklearn.decomposition.PCA(n_components = n_components)
                        data_pca_patient = pca_patient.fit_transform(z_patient.T).T
                        
        
                        print(data_pca_patient.shape)

                    elif mode == 'PCA_flip':
                        U, S, VT = randomized_svd(data_0, n_components=n_components,n_iter='auto',random_state=None)
                        data_pca_patient = np.zeros((n_components, M))
                        for k in range(n_components):
                            sign = np.sign(U[:,k]@src_flip[i])
                            scale = np.linalg.norm(S) / np.sqrt(len(data_0))
                            data_pca_patient[k,:]= sign[0] * scale * VT[k]

                        print(data_pca_patient.shape)

                    #compute entropy production rate
                    sigma_0, pos,_ = compute_entropy_n(data_pca_patient, dt, n_iter, Np)
                    sigma[i] = sigma_0
    
            mydict = []
            for i in inds:
                s = sigma[i]
                mydict.append({'region' : labels_parc[i].name, 'sigma' : s})
    
            # field names
            fields = ['region', 'sigma']
    
            # name of csv file
    
    
    
    
            # writing to csv file
            with open(output_filename, 'w') as csvfile:
                # creating a csv dict writer object
                writer = csv.DictWriter(csvfile, fieldnames=fields)
    
                # writing headers (field names)
                writer.writeheader()
    
                # writing data rows
                writer.writerows(mydict)
            
            
            param_dict = {}
            param_dict['subject'] = subject
            param_dict['session'] = ses
            param_dict['n_step'] = L
            param_dict['dt'] = dt
            param_filename = os.path.join(output_dir, subject + '_' + ses + '_data_properties.txt')
    
            with open(param_filename,'w') as param_file:
                param_file.write(json.dumps(param_dict))
                
            if compute_eig:
                with open(eig_vec_filename, 'wb') as f:
                    pickle.dump(eig_vec, f)
                with open(eig_filename, 'wb') as f:
                    pickle.dump(eig, f)

            if save_singular_values:
                with open(sv_filename, 'wb') as f:
                    pickle.dump(S, f)
                with open(lsv_filename, 'wb') as f:
                    pickle.dump(U, f)
                with open(rsv_filename, 'wb') as f:
                    pickle.dump(VT, f)
            
    except Exception as e:
        # Handle the error
        print(f"An error occurred: {e}")
        # Optionally, you can continue to the next iteration
        continue
