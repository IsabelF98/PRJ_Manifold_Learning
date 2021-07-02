# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Manifold 2
#     language: python
#     name: manifold2
# ---

# # Data Intrinsic Dimension
# This notebook is created to compute the intrinsic dimension (ID) of the data. The intrinsic dimension of a manifold infers the minimum number of features needed to represent a manifold with the least amount of data lost. The method works by taking neighborhood sets of each data point (using k-NN) and finding the dimnesion of that neighborhood manifold. The flat tangent space of the manifold is used to estimate the dimension of the neighborhood manifold. An ID value is returned for each data point and the dimsnon of the entire data set is the average (or median) of the ID vlaues. We will look at the dimenions of both the rest and task fMRI data.
#
# Isabel Fernandez 7/1/2021

import skdim
import os
import os.path as osp
import pandas as pd
import numpy as np
from scipy.io import loadmat
import xarray as xr

PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

# ***
# ## Scikit-Dimension Example

# + jupyter={"source_hidden": true}
#generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions
data = np.zeros((1000,10))
data[:,:5] = skdim.datasets.hyperBall(n = 1000, d = 5, radius = 1, random_state = 0)
data

# + jupyter={"source_hidden": true}
#estimate global intrinsic dimension
danco = skdim.id.DANCo().fit(data)
#estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
lpca = skdim.id.lPCA().fit_pw(data,
                              n_neighbors = 100,
                              n_jobs = -1)

# + jupyter={"source_hidden": true}
#get estimated intrinsic dimension
print(danco.dimension_, np.median(lpca.dimension_pw_))
# -

# ***
# ## Rest fMRI Data

# +
# Load rs fMRI subject information
# --------------------------------
rs_fMRI_sub_df = pd.read_pickle(PRJDIR+'/Data/Samika_DSet02/valid_run_df.pkl') # Load valid subject info

rs_fMRI_SubDict = {} # Empty dictionary
for i,idx in enumerate(rs_fMRI_sub_df.index):
    sbj  = rs_fMRI_sub_df.loc[idx]['Sbj']
    run  = rs_fMRI_sub_df.loc[idx]['Run']
    if sbj in rs_fMRI_SubDict.keys():
        rs_fMRI_SubDict[sbj].append(run)
    else:
        rs_fMRI_SubDict[sbj] = ['All',run]

# List of subjects
rs_fMRI_SubjectList = list(rs_fMRI_SubDict.keys())
# -

# Compute and save ID data
# ------------------------
for rest_WL_sec in [30, 46, 60]: # All availalbe window lenghts
    rest_ID = {} # Empty dictionary for ID data to be stored
    for rest_SBJ in rs_fMRI_SubjectList: # For every subject
        rs_fMRI_RuntList = rs_fMRI_SubDict[rest_SBJ] # List of runs for given subject
        for rest_RUN in rs_fMRI_RuntList:
            # Load rest fMRI SWC data
            # -----------------------
            rest_file_name = rest_SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(rest_WL_sec).zfill(3)+'s_ws002s_'+rest_RUN+'_PCA_vk97.5.swcorr.pkl' # Data file name
            rest_data_path = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',rest_SBJ,'D02_Preproc_fMRI',rest_file_name) # Path to data
            rest_data = pd.read_pickle(rest_data_path).T.to_numpy() # Read data into pandas data frame, transpose and turn into np array
            rest_num_samp, rest_num_conn = rest_data.shape # Save number of samples and connections as a varable
            # Compute ID
            # ----------
            rest_lpca = skdim.id.lPCA().fit_pw(rest_data, n_neighbors = 70, n_jobs = -1)
            # Add data to data frame
            # ----------------------
            if rest_SBJ in rest_ID.keys(): # If not a new subject
                rest_ID[rest_SBJ][rest_RUN] = rest_lpca.dimension_pw_ # Add ID data for run as np.array
            else:
                rest_ID[rest_SBJ] = {rest_RUN: rest_lpca.dimension_pw_} # Create new subject key in data dictionary and add ID data for run as np.array
    np.save(PRJDIR+'/Data/Samika_DSet02/Intrinsic_Dimension_WL'+str(rest_WL_sec)+'sec.npy', rest_ID) # Save ID data for given WL as a numpy file
    print('++INFO: Saved ID data for WL', rest_WL_sec)

import hvplot.pandas

pd.DataFrame(rest_lpca.dimension_pw_).hvplot.hist(bins=50)

# ***
# ## Task fMRI Data

task_fMRI_SubjectList  = ['SBJ06', 'SBJ08', 'SBJ10', 'SBJ12', 'SBJ16', 'SBJ18', 'SBJ20', 'SBJ22', 'SBJ24', 'SBJ26', 'SBJ07', 'SBJ09',
                          'SBJ11', 'SBJ13', 'SBJ17', 'SBJ19', 'SBJ21', 'SBJ23', 'SBJ25', 'SBJ27'] # List of subjects

# Compute and save ID data
# ------------------------
for task_WL_sec in [30, 45]: # All availalbe window lenghts
    task_ID = {} # Empty dictionary for ID data to be stored
    for task_SBJ in task_fMRI_SubjectList: # For every subject
        # Load task fMRI SWC data
        # -----------------------
        task_file_name = task_SBJ+'_CTask001_WL0'+str(task_WL_sec)+'_WS01_NROI0200_dF.mat' # Data file name
        task_data_path = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02/PrcsData_PNAS2015',task_SBJ,'D02_CTask001',task_file_name) # Path to data
        task_data      = loadmat(task_data_path)['CB']['snapshots'][0][0] # Read data
        # Compute ID
        # ----------
        task_lpca = skdim.id.lPCA().fit_pw(task_data, n_neighbors = 70, n_jobs = -1)
        # Add data to data frame
        # ----------------------
        task_ID[task_SBJ]= task_lpca.dimension_pw_ # Create new subject key in data dictionary and add ID data as np.array
    np.save(PRJDIR+'/Data/MultiTask/Intrinsic_Dimension_WL'+str(task_WL_sec)+'sec.npy', task_ID) # Save ID data for given WL as a numpy file
    print('++INFO: Saved ID data for WL', task_WL_sec)
