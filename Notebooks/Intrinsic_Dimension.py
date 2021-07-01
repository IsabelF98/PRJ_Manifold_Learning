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

# ***
# ## Scikit-Dimension Example

#generate data : np.array (n_points x n_dim). Here a uniformly sampled 5-ball embedded in 10 dimensions
data = np.zeros((1000,10))
data[:,:5] = skdim.datasets.hyperBall(n = 1000, d = 5, radius = 1, random_state = 0)
data

#estimate global intrinsic dimension
danco = skdim.id.DANCo().fit(data)
#estimate local intrinsic dimension (dimension in k-nearest-neighborhoods around each point):
lpca = skdim.id.lPCA().fit_pw(data,
                              n_neighbors = 100,
                              n_jobs = -1)

#get estimated intrinsic dimension
print(danco.dimension_, np.median(lpca.dimension_pw_))

# ***
# ## Rest fMRI Data

# Subject, run, and WL
# --------------------
rest_SBJ    = 'sub-S26' # Subject
rest_RUN    = 'All' # Run
rest_WL_sec = 30 # Windowlength in seconds

# +
# Load rest fMRI SWC data
# -----------------------
rest_file_name = rest_SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(rest_WL_sec).zfill(3)+'s_ws002s_'+rest_RUN+'_PCA_vk97.5.swcorr.pkl' # Data file name
rest_data_path = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',rest_SBJ,'D02_Preproc_fMRI',rest_file_name) # Path to data
rest_data = pd.read_pickle(rest_data_path).T.to_numpy() # Read data into pandas data frame, transpose and turn into np array
rest_num_samp, rest_num_conn = rest_data.shape # Save number of samples and connections as a varable

print('Number of Samples:    ',rest_num_samp)
print('Number of Connections:',rest_num_conn)
# -

# Compute ID
# ----------
rest_lpca = skdim.id.lPCA().fit_pw(rest_data, n_neighbors = 70, n_jobs = -1)
print('Intrinsic Dimension: ', np.median(rest_lpca.dimension_pw_))

# ***
# ## Task fMRI Data

# Subject, windowlenght in seconds, and pure specificity
# ------------------------------------------------------
task_SBJ    = 'SBJ06'
task_WL_sec = 30
PURE        = ''

# +
# Load task fMRI SWC data
# -----------------------
task_file_name = task_SBJ+'_CTask001_WL0'+str(task_WL_sec)+'_WS01'+PURE+'_NROI0200_dF.mat' # Data file name
task_data_path = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02/PrcsData_PNAS2015',task_SBJ,'D02_CTask001',task_file_name) # Path to data
task_data      = loadmat(task_data_path)['CB']['snapshots'][0][0] # Read data
task_num_samp, task_num_conn = task_data.shape # Save number of samples and connections as a varable

print('Number of Samples:    ',task_num_samp)
print('Number of Connections:',task_num_conn)
# -

# Compute ID
# ----------
task_lpca = skdim.id.lPCA().fit_pw(task_data, n_neighbors = 70, n_jobs = -1)
print('Intrinsic Dimension: ', np.median(task_lpca.dimension_pw_))
