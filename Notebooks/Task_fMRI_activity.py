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
#     display_name: Manifold
#     language: python
#     name: manifold
# ---

# # Task fMRI Activity Data
#
# Observe task data by activity rather then connectivity.
#
# Isabel Fernandez 6/7/2021

import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.manifold  import TSNE
from sklearn.manifold  import SpectralEmbedding
import panel as pn
import holoviews as hv
import plotly.express as px
pn.extension('plotly')

# +
DATADIR = '/data/SFIMJGC/PRJ_CognitiveStateDetection01/PrcsData/' # Data directory path
PRJDIR  = '/data/SFIMJGC/PRJ_Manifold_Learning/' # Project directory path

SubjectList  = ['SBJ06', 'SBJ08', 'SBJ10', 'SBJ12', 'SBJ16', 'SBJ18', 'SBJ20', 'SBJ22', 'SBJ24', 'SBJ26', 'SBJ07', 'SBJ09', 'SBJ11', 'SBJ13',
                'SBJ17', 'SBJ19', 'SBJ21', 'SBJ23', 'SBJ25', 'SBJ27'] # List of subjects
# -

# ***
# ## Compute activity data w/ SD
#
# If the files already exist there is no need to recompute them.

# +
WL_sec = 45 # Activity window lengths in seconds
WL_TRs = int(WL_sec/1.5) # Activity window lengths in TRs

# For loop to compute activity data for each subject
for SBJ in SubjectList:
    stream   = os.popen('ls '+DATADIR+SBJ+'/D02_CTask001/DXX_NROIS0200/'+SBJ+'_CTask001.Craddock_T2Level_0200.lowSigma.???.WL180.SF.1D | wc -l')
    num_ROIs = int(stream.read()) # Number of ROI time sereies for a given subject

    ROI_data_df  = pd.DataFrame() # Empty ROI time sereis data frame

    for ROI in range(0,num_ROIs): # For each ROI
        input_file_name = SBJ+'_CTask001.Craddock_T2Level_0200.lowSigma.'+str(ROI).zfill(3)+'.WL180.SF.1D' # ROI time series file name
        input_file_path = DATADIR+SBJ+'/D02_CTask001/DXX_NROIS0200/'+input_file_name # ROI time series file path
        ROI_ts          = pd.read_csv(input_file_path, header=None) # Load ROI time series
        ROI_data_df['ROI_'+str(ROI).zfill(3)] = ROI_ts[0] # Save ROI time seres to 'ROI_data_df' as a column

    num_TRs = ROI_data_df.shape[0] # Number of TR's in data
    
    act_data_df = pd.DataFrame(columns=['SD_'+str(ROI).zfill(3) for ROI in range(0,num_ROIs)]) # Empty activity data frame
    
    # For loop to compute activity for each window using SD
    start = 0 # Start at 0th TR
    for i in range(0,num_TRs-WL_TRs+1):
        end = start + WL_TRs - 1 # Last TR in window
        window    = ROI_data_df.loc[start:end] # Iscolate window data
        window_sd = np.array(window.std(axis=0)) # Compute SD for each ROI (column) of window
        act_data_df.loc[len(act_data_df.index)] = window_sd # Save SD to 'act_data_df' as a row
        start = start + 1 # Next window start TR
        
    output_file_name = SBJ+'_CTask001.Craddock_T2Level_0200.lowSigma.WL180.WL'+str(WL_sec).zfill(3)+'sec.SF.SD_Activity.pkl' # Name of output file
    output_file_path  = PRJDIR+'/Data/MultiTask/'+output_file_name # Output file path
    act_data_df.to_pickle(output_file_path) # Save file as pkl file
    print('++INFO: Completed activity data for '+SBJ)
# -

# ***
# ## Compute and plot activity embedding

# +
# Embedding widgets
# -----------------
SubjSelect   = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0], width=200) # Select subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,45], width=200) # Select window length

k_list  = [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300] # All k values computed
kSelect = pn.widgets.Select(name='Select k Value', options=k_list, value=k_list[7], width=200) # Select k value for nearest neighbor

p_list     = [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300] # List of perplexity values
Perplexity = pn.widgets.Select(name='Select Perplexity', options=p_list, value=p_list[0], width=200) # Select perplexity value

l_list       = [10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000] # List of learning rates
LearningRate = pn.widgets.Select(name='Select Learning Rate', options=l_list, value=l_list[0], width=200) # Select learning rate

# +
# Task data frame
# ---------------
orig_data_task = pd.DataFrame(index=range(0,1016),columns=['Task']) # Empty task data frame
task_list = ['Rest', 'Memory', 'Video', 'Math', 'Memory', 'Rest', 'Math', 'Video'] # List of tasks in order they were performed
tr = 0 # Starting at 0th TR
for task in task_list: # For each task in the list of tasks
    orig_data_task['Task'][tr:tr+120] = task # Append 120 TRs (180s) for the given task
    orig_data_task['Task'][tr+120:tr+128] = 'Inbetween' # Append 8 TRs (12s) for inbetween tasks
    tr = tr+128 # Move to next task start TR
orig_num_TR = orig_data_task.shape[0] # Oginal number of TRs in data (1016)

# Function to create task data frame after activity map is computed
# Function inputs are the window length in seconds and the number of windows
def task_data(WL_sec,num_win):
    task_df = pd.DataFrame(index=range(0,num_win),columns=['Task']) # Empty task data frame
    WL_trs = int(WL_sec/1.5) # Window length in TR's (TR = 1.5s)
    for i in range(0,num_win): # For each window index i
        task_array  = np.array([x for x in orig_data_task.loc[i:i+(WL_trs-1), 'Task']]) # Create an array of all tasks in a given window
        if np.all(task_array == task_array[0]): # If all tasks are the same in the window make window task = task
            task_df.loc[i, 'Task'] = task_array[0]
        else:  # If all tasks are NOT the same in the window make window task = 'Inbetween'
            task_df.loc[i, 'Task'] = 'Inbetween'
    return task_df

WL30_task_df = task_data(30,998) # Task data frames for WL = 30sec
WL45_task_df = task_data(45,988) # Task data frames for WL = 45sec


# -

# Function for plotting embeddings of activity data
# -------------------------------------------------
@pn.depends(SubjSelect.param.value, WindowSelect.param.value, kSelect.param.value, Perplexity.param.value, LearningRate.param.value)
def embedding_plots(SBJ,WL_sec,k,p,l):
    file_name = SBJ+'_CTask001.Craddock_T2Level_0200.lowSigma.WL180.WL'+str(WL_sec).zfill(3)+'sec.SF.SD_Activity.pkl' # Data file name
    file_path  = PRJDIR+'/Data/MultiTask/'+file_name # Data file path
    data_df = pd.read_pickle(file_path) # Load data
    
    # Load task info df based on WL
    if WL_sec == 30:
        task_df = WL30_task_df
    elif WL_sec == 45:
        task_df = WL45_task_df
        
    cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'} # Color key for tasks
    
    # Laplacian Eigenmap
    # ------------------
    LE_embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, n_neighbors=k) # Compute 3D transform
    LE_data_transformed = LE_embedding.fit_transform(data_df) # Transform data using embedding
    
    LE_plot_df = pd.DataFrame(LE_data_transformed, columns=['x','y','z']) # Change data to pandas data frame
    LE_plot_df['Task'] = task_df.astype(str) # Add column of task identifier with elements as type string
    
    # Plot embedding
    LE_plot = px.scatter_3d(LE_plot_df, x='x', y='y', z='z', color='Task', color_discrete_map=cmap, width=700, height=600, opacity=0.7, title='Laplacian Eigenmap')
    LE_plot = LE_plot.update_traces(marker=dict(size=5,line=dict(width=0)))
    
    # t-SNE
    # -----
    tSNE_data_transformed = TSNE(n_components=3, perplexity=p, learning_rate=l, n_jobs=-1).fit_transform(data_df) # Apply TSNE to transform data to 3D
    
    tSNE_plot_df = pd.DataFrame(tSNE_data_transformed, columns=['x','y','z']) # Change data to pandas data frame
    tSNE_plot_df['Task'] = task_df.astype(str) # Add column of task identifier with elements as type string
    
    # Plot embedding
    tSNE_plot = px.scatter_3d(tSNE_plot_df, x='x', y='y', z='z', color='Task', color_discrete_map=cmap, width=700, height=600, opacity=0.7, title='t-SNE')
    tSNE_plot = tSNE_plot.update_traces(marker=dict(size=5,line=dict(width=0)))
    
    return pn.Row(LE_plot, tSNE_plot)


pn.Column(pn.Row(SubjSelect, WindowSelect, kSelect, Perplexity, LearningRate), embedding_plots) # Display widgets and plots
