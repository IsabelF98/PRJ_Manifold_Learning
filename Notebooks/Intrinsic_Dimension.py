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
import hvplot.pandas
import panel as pn

PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

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
for n in [10,20,30,40,50,60,70,80,90]:
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
                rest_lpca = skdim.id.lPCA().fit_pw(rest_data, n_neighbors = n, n_jobs = -1)
                # Add data to data frame
                # ----------------------
                values = rest_lpca.dimension_pw_ # ID values as numpy array
                temp = np.empty((2224,)) # Temp numpy array of lenght of longest data set (All data length 2224)
                temp[:] = np.nan # Fill temp array with NaN
                temp[:values.shape[0]] = values # Set first values as ID values in temp array
                rest_ID[(rest_SBJ,rest_RUN)] = temp # Add data to dictionary       
        rest_ID_df = pd.DataFrame(rest_ID).T # Change data to pandas data frame
        rest_ID_df.to_pickle(PRJDIR+'/Data/Samika_DSet02/Intrinsic_Dimension_WL'+str(rest_WL_sec)+'sec_NN'+str(n)+'.pkl') # Save data frame as pickle file
    
        print('++INFO: Saved ID data for WL', rest_WL_sec, 'and n', n)

# ## ID Distributions

# Load pre computed ID data
# -------------------------
rest_WL_sec = 30
n = 90
rest_ID_df = pd.read_pickle(PRJDIR+'/Data/Samika_DSet02/Intrinsic_Dimension_WL'+str(rest_WL_sec)+'sec_NN'+str(n)+'.pkl')
print('++INFO: ID data dimensions:', rest_ID_df.shape)

# +
# Distribution plot for single subject
# ------------------------------------
ssbj='sub-S24' # Selected subject
sbj_ID_df = rest_ID_df.loc[[(ssbj,'All'),(ssbj,'SleepAscending'),(ssbj,'SleepDescending'),(ssbj,'SleepRSER'),(ssbj,'WakeAscending'),
                            (ssbj,'WakeDescending'),(ssbj,'WakeRSER')]].T # ID data frame for selected subject
sbj_ID_df.columns=['All','SleepAscending','SleepDescending','SleepRSER','WakeAscending','WakeDescending','WakeRSER'] # Name columns by run

color_map = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
                                      'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','All':'gray'}

gspec = pn.GridSpec(width=1700,height=400) # Empty plot grip
# Add distribution plots for each run and all runs
gspec[0,0:1]  = (sbj_ID_df.hvplot.hist(y='SleepAscending', normed=True) * sbj_ID_df.hvplot.kde(y='SleepAscending', alpha=0.5)).opts(toolbar=None)
gspec[1,0:1]  = (sbj_ID_df.hvplot.hist(y='WakeAscending', normed=True) * sbj_ID_df.hvplot.kde(y='WakeAscending', alpha=0.5)).opts(toolbar=None)
gspec[0,2:3]  = (sbj_ID_df.hvplot.hist(y='SleepDescending', normed=True) * sbj_ID_df.hvplot.kde(y='SleepDescending', alpha=0.5)).opts(toolbar=None)
gspec[1,2:3]  = (sbj_ID_df.hvplot.hist(y='WakeDescending', normed=True) * sbj_ID_df.hvplot.kde(y='WakeDescending', alpha=0.5)).opts(toolbar=None)
gspec[0,4:5]  = (sbj_ID_df.hvplot.hist(y='SleepRSER', normed=True) * sbj_ID_df.hvplot.kde(y='SleepRSER', alpha=0.5)).opts(toolbar=None)
gspec[1,4:5]  = (sbj_ID_df.hvplot.hist(y='WakeRSER', normed=True) * sbj_ID_df.hvplot.kde(y='WakeRSER', alpha=0.5)).opts(toolbar=None)
gspec[:,6:10] = (sbj_ID_df.hvplot.kde(y=['SleepAscending','SleepDescending','SleepRSER','WakeDescending','WakeAscending','WakeRSER'], 
                 alpha=0.5) * sbj_ID_df.hvplot.hist(y='All', normed=True, alpha=0.7) * sbj_ID_df.hvplot.kde(y='All',alpha=0.5)).opts(toolbar=None)
gspec
# -

# Run ID median data frame
# ------------------------
rest_ID_medians = pd.DataFrame(rest_ID_df.median(axis=1, skipna=True)) # Compute median ID for all runs
rest_ID_medians.columns=['Median Dim']

rest_ID_medians.iloc[157:157+7] # Medians for one subject

run_ID_medians = rest_ID_medians.reset_index()
run_ID_medians.columns=['Sbj','Run','Median Dim']
(run_ID_medians[run_ID_medians['Run']=='All'].hvplot.hist(normed=True) * \
run_ID_medians[run_ID_medians['Run']=='All'].hvplot.kde(alpha=0.5, title='Median Dims for All Subjects (All)')).opts(toolbar=None)

# ***
# ## Task fMRI Data

task_fMRI_SubjectList  = ['SBJ06', 'SBJ08', 'SBJ10', 'SBJ12', 'SBJ16', 'SBJ18', 'SBJ20', 'SBJ22', 'SBJ24', 'SBJ26', 'SBJ07', 'SBJ09',
                          'SBJ11', 'SBJ13', 'SBJ17', 'SBJ19', 'SBJ21', 'SBJ23', 'SBJ25', 'SBJ27'] # List of subjects

# Compute and save ID data
# ------------------------
for n in [10,20,30,40,50,60,70,80,90]:
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
            task_lpca = skdim.id.lPCA().fit_pw(task_data, n_neighbors = n, n_jobs = -1)
            # Add data to data frame
            # ----------------------
            task_ID[task_SBJ]= task_lpca.dimension_pw_ # Create new subject key in data dictionary and add ID data as np.array
        new_ID_df = pd.DataFrame(task_ID).T # Change data to pandas data frame
        new_ID_df.to_pickle(PRJDIR+'/Data/MultiTask/Intrinsic_Dimension_WL'+str(task_WL_sec)+'sec_NN'+str(n)+'.pkl') # Save data frame as pickle file
        
        print('++INFO: Saved ID data for WL', task_WL_sec, 'and n', n)

# ## ID Distributions

task_WL_sec = 30
n = 90
task_ID_df = pd.read_pickle(PRJDIR+'/Data/MultiTask/Intrinsic_Dimension_WL'+str(task_WL_sec)+'sec_NN'+str(n)+'.pkl')
print('++INFO: ID data dimensions:', task_ID_df.shape)

# +
# Distribution plot for single subject
# ------------------------------------
ssbj='SBJ07' # Selected subject
sbj_ID_df = task_ID_df.loc[[ssbj]].T # ID data frame for selected subject

(sbj_ID_df.hvplot.hist(y=ssbj, normed=True) * sbj_ID_df.hvplot.kde(y=ssbj, alpha=0.5)).opts(toolbar=None) # ID distribution plot for selected subject

# +
# ID median data frame
# --------------------
task_ID_medians = pd.DataFrame(task_ID_df.median(axis=1, skipna=True)) # Compute median ID for all subjects
task_ID_medians.columns=['Median Dim']

(task_ID_medians.hvplot.hist(y='Median Dim', normed=True) * task_ID_medians.hvplot.kde(y='Median Dim', alpha=0.5)).opts(toolbar=None)
