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

# # Laplacian Eignemaps
#
# This notebook is desighned to observe and understand the plots generated by the sklearn.manifold.SpectralEmbedding function.
#
# Isabel Fernandez 5/26/2021

import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.manifold  import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph
import scipy
from scipy.spatial.distance import correlation as dis_corr
from scipy.io import loadmat
from sklearn.metrics import pairwise_distances
from scipy.sparse import save_npz, load_npz
from sklearn.datasets import load_digits
import holoviews as hv
from holoviews.operation.datashader import rasterize
import panel as pn
import plotly.express as px
from holoviews import dim, opts
hv.extension('bokeh')
pn.extension('plotly')

# +
PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

port_tunnel = int(os.environ['PORT2']) # Get port tunnel for gui display
print('++ INFO: Second Port available: %d' % port_tunnel)
# -

# ***
# ## Digits Data

# +
# Load digits data
dig_img_df, dig_lab_df = load_digits(return_X_y=True,as_frame=True)

print('++ INFO: Digits data frame dimension ',dig_img_df.shape)
# -

# ***
# ## Fashion Data

# +
# Load fashion data (first 1000 of test only)
fashion_path = os.path.join(PRJDIR,'Data','Fashion_Data') # Path to fashion data set

fash_img_full_df = np.load(fashion_path+'/test_images.npy') # Load test images
fash_lab_full_df = np.load(fashion_path+'/test_labels.npy') # Load test labels

fash_img_df = pd.DataFrame(fash_img_full_df.reshape((fash_img_full_df.shape[0], 784)))[0:1000] # Flatten image matricies and convert images array to pandas df
fash_lab_df = pd.DataFrame(fash_lab_full_df)[0:1000] # Convert lables array to pandas df

# Chnage fashion labels by name not number
for i in range(0,fash_lab_df.shape[0]):
    if fash_lab_df.loc[i, 0] == 0:
        fash_lab_df.loc[i, 0] = 'T-shirt/Top'
    elif fash_lab_df.loc[i, 0] == 1:
        fash_lab_df.loc[i, 0] = 'Trouser'
    elif fash_lab_df.loc[i, 0] == 2:
        fash_lab_df.loc[i, 0] = 'Pullover'
    elif fash_lab_df.loc[i, 0] == 3:
        fash_lab_df.loc[i, 0] = 'Dress'
    elif fash_lab_df.loc[i, 0] == 4:
        fash_lab_df.loc[i, 0] = 'Coat'
    elif fash_lab_df.loc[i, 0] == 5:
        fash_lab_df.loc[i, 0] = 'Sandal'
    elif fash_lab_df.loc[i, 0] == 6:
        fash_lab_df.loc[i, 0] = 'Shirt'
    elif fash_lab_df.loc[i, 0] == 7:
        fash_lab_df.loc[i, 0] = 'Sneaker'
    elif fash_lab_df.loc[i, 0] == 8:
        fash_lab_df.loc[i, 0] = 'Bag'
    elif fash_lab_df.loc[i, 0] == 9:
        fash_lab_df.loc[i, 0] = 'Ankle Boot'

print('++ INFO: Digits data frame dimension ',fash_img_df.shape)
# -

# ***
# ## Resting State fMRI

# +
# Load rs fMRI subject information
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

# +
# Create widgets for rs fMRI data
rs_fMRI_SubjSelect   = pn.widgets.Select(name='Select Subject', options=rs_fMRI_SubjectList, value=rs_fMRI_SubjectList[0], width=200) # Select subject
rs_fMRI_RunSelect    = pn.widgets.Select(name='Select Run', options=rs_fMRI_SubDict[rs_fMRI_SubjSelect.value], value=rs_fMRI_SubDict[rs_fMRI_SubjSelect.value][1], width=200) # Select run for chosen subject
rs_fMRI_WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60], width=200) # Select window lenght

# Updates available runs given SubjSelect value
def update_run(event):
    rs_fMRI_RunSelect.options = rs_fMRI_SubDict[event.new]
rs_fMRI_SubjSelect.param.watch(update_run,'value')


# -

def winner_takes_all(my_array):
    """
    This function desighned to output the number that apears most frequently in a nupy array.
    This function will be used to slelect the sleep staging value of each window after sliding window correlation.
    Since the values only range from 0 to 3 any NaN values are changed to a value of 4.
    Then using the function np.bincount each value in the np array is counted.
    The value with the highest count is called the "winner".
    """
    # Changes NaN to value of 4
    if np.isnan(np.sum(my_array)) == True:
        my_array[np.isnan(my_array)] = 4
    my_array = my_array.astype(int) # Change all values in array as integers
    counts = np.bincount(my_array) # Count each element
    winner = np.argmax(counts) # Choose element with highest count
    return winner


# Load rs fMRI data function
@pn.depends(rs_fMRI_SubjSelect.param.value, rs_fMRI_RunSelect.param.value, rs_fMRI_WindowSelect.param.value)
def load_rsfMRI_data(SBJ, RUN, WL_sec):
    WL_trs = int(WL_sec/2) # Window length in TR's
    
    # Load SWC data
    # -------------
    file_name = SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(WL_sec).zfill(3)+'s_ws002s_'+RUN+'_PCA_vk97.5.swcorr.pkl' # Data file name
    data_path = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',SBJ,'D02_Preproc_fMRI',file_name) # Path to data
    data_df = pd.read_pickle(data_path).T # Read data into pandas data frame
    num_TR = data_df.shape[0] # Number of TR's in run
    num_orig_TR = num_TR + WL_trs - 1 # Number of original TR's before SWC
    
    # Load sleep staging data
    # -----------------------
    WL_trs = int(WL_sec/2) # Window length in TR's
    sleep_temp = pd.DataFrame(columns=['Sleep Value','Sleep Stage']) # Temporary data frame to organize data by sleep stage
    # 1. Sleep staging data is saved as an individual data frame for each run which we will call "EEG_sleep_df".
    # If a single run is selected that data frame is simply loaded.
    # If all runs are slected then "EEG_sleep_df" is a single data frame with all runs concatinated in the same order as the concatinated data
    if RUN != 'All': # Single run
        sleep_file_path = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+RUN+'_EEG_sleep.pkl') # Load sleep data
        EEG_sleep_df    = pd.read_pickle(sleep_file_path) # Save as pandas data frame
    else: # All runs
        run_list = rs_fMRI_SubDict[SBJ].copy() # List of runs for that subject
        run_list.remove('All') # Remove "All" from list of runs
        EEG_sleep_df = pd.DataFrame(columns=['dataset','subject','cond','TR','sleep','drowsiness','spectral','seconds','stage']) # Empty sleep staged data frame with coulumn names
        # Append each runs sleep stage data to end of EEG_sleep_df
        for r in run_list:
            sleep_file_path    = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',SBJ,'D02_Preproc_fMRI',SBJ+'_'+r+'_EEG_sleep.pkl')
            run_sleep_df = pd.read_pickle(sleep_file_path)
            EEG_sleep_df = EEG_sleep_df.append(run_sleep_df).reset_index(drop = True)
    # 2. The sleep stage value for each window is chosen using the winner_takes_all() function for that windows window length.
    for i in range(0,num_orig_TR-WL_trs+1): # Iterate through number of windows for given data (Number_of_TRs - Number_of_TRs_per_window + 1)
        sleep_array  = np.array([x for x in EEG_sleep_df.loc[i:i+(WL_trs-1), 'sleep']]) # Numpy array of values pertaining to window
        sleep_val = winner_takes_all(sleep_array) # Choose sleep value using winner_takes_all() function
        sleep_temp.loc[i, 'Sleep Value'] = int(sleep_val) # Ensure sleep value is an integer
    # 3. Asighn sleep stage to each sleep value
    #    0 = wake
    #    1 = stage 1
    #    2 = stage 2
    #    3 = stage 3
    #    4 = undetermined stage
    for i,idx in enumerate(sleep_temp.index):
        if sleep_temp.loc[idx, 'Sleep Value'] == 0:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Wake'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 1:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 1'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 2:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 2'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 3:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Stage 3'
        elif sleep_temp.loc[idx, 'Sleep Value'] == 4:
            sleep_temp.loc[idx, 'Sleep Stage'] = 'Undetermined'
    # Save sleep stage data as label_df
    label_df = sleep_temp['Sleep Stage']
    
    return data_df, label_df


# +
# Player to display points over time in rs fMRI data
# Function gets number of time points for each run
@pn.depends(rs_fMRI_SubjSelect.param.value, rs_fMRI_RunSelect.param.value, rs_fMRI_WindowSelect.param.value)
def get_num_tp(SBJ,RUN,WL_sec):
    data_df, label_df = load_rsfMRI_data(SBJ,RUN,WL_sec)
    value = data_df.shape[0]
    return value

# Window player over time
rs_player = pn.widgets.Player(name='Player', start=0, end=get_num_tp(rs_fMRI_SubjSelect.value, rs_fMRI_RunSelect.value, rs_fMRI_WindowSelect.value),
                           value=get_num_tp(rs_fMRI_SubjSelect.value, rs_fMRI_RunSelect.value ,rs_fMRI_WindowSelect.value), loop_policy='loop', width=800, step=1)

# Function updates player range based on run selected
@pn.depends(rs_fMRI_SubjSelect, rs_fMRI_RunSelect, rs_fMRI_WindowSelect, watch=True)
def update_player(SBJ,RUN,WL_sec):
    end_value = get_num_tp(SBJ,RUN,WL_sec) # Get number of time points from get_num_tp() function
    rs_player.value = end_value # Update player value to last player value or new end value
    rs_player.end = end_value # Update end value


# -

# ***
# ## Task fMRI

# Create widgets for task fMRI data
task_fMRI_SubjectList  = ['SBJ06', 'SBJ08', 'SBJ10', 'SBJ12', 'SBJ16', 'SBJ18', 'SBJ20', 'SBJ22', 'SBJ24', 'SBJ26', 'SBJ07', 'SBJ09',
                          'SBJ11', 'SBJ13', 'SBJ17', 'SBJ19', 'SBJ21', 'SBJ23', 'SBJ25', 'SBJ27'] # List of subjects
task_fMRI_SubjSelect   = pn.widgets.Select(name='Select Subject', options=task_fMRI_SubjectList, value=task_fMRI_SubjectList[0], width=200) # Select subject
task_fMRI_PureSelect   = pn.widgets.Select(name='Select Window Type', options=['pure', 'not pure'], value='not pure', width=200) # Select window purity
task_fMRI_WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,45], width=200) # Select window lenght

# +
# Task data frame
orig_data_task = pd.DataFrame(index=range(0,1016),columns=['Task']) # Empty task data frame
task_list = ['Rest', 'Memory', 'Video', 'Math', 'Memory', 'Rest', 'Math', 'Video'] # List of tasks in order they were performed
tr = 0 # Starting at 0th TR
for task in task_list: # For each task in the list of tasks
    orig_data_task['Task'][tr:tr+120] = task # Append 120 TRs (180s) for the given task
    orig_data_task['Task'][tr+120:tr+128] = 'Inbetween' # Append 8 TRs (12s) for inbetween tasks
    tr = tr+128 # Move to next task start TR
orig_num_TR = orig_data_task.shape[0] # Oginal number of TRs in data (1016)

# Function to create task data frame after SWC is computed
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
    
    # Create a task data frame for pure windows with tasks not windows inbetween tasks
    pure_task_df = task_df.copy()
    pure_task_df = pure_task_df.drop(pure_task_df[pure_task_df['Task'] == 'Inbetween'].index).reset_index(drop = True) # Drop inbetween windows
    pure_task_df = pure_task_df[:-1] # Drop the last data point (idk why we have an extra data point?)
    
    return task_df, pure_task_df

WL30_task_df, WL30pure_task_df = task_data(30,998) # Task data frames for WL = 30s
WL45_task_df, WL45pure_task_df = task_data(45,988) # Task data frames for WL = 45s


# -

# Load task data function
@pn.depends(task_fMRI_SubjSelect.param.value, task_fMRI_PureSelect.param.value, task_fMRI_WindowSelect.param.value)
def load_taskfMRI_data(SBJ, PURE, WL_sec):
    # Define task label data frame
    # ----------------------------
    if WL_sec == 30:
        if PURE == 'pure':
            label_df = WL30pure_task_df
        else:
            label_df = WL30_task_df
    else:
        if PURE == 'pure':
            label_df = WL45pure_task_df
        else:
            label_df = WL45_task_df
    
    # Define PURE varaible based on widget
    # ------------------------------------
    if PURE == 'not pure':  
        PURE = '' # Load data with non pure windows
    
    # Load task fMRI data
    # -------------------
    file_name = SBJ+'_CTask001_WL0'+str(WL_sec)+'_WS01'+PURE+'_NROI0200_dF.mat' # Data file name
    data_path = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02/PrcsData_PNAS2015',SBJ,'D02_CTask001',file_name) # Path to data
    data_df   = loadmat(data_path)['CB']['snapshots'][0][0] # Read data
    
    return data_df, label_df


# ***
# ## Create LE Widgets

# +
d_list   = ['Digits', 'Fashion', 'rs fMRI', 'task fMRI'] # List of data sets
DataType = pn.widgets.Select(name='Select Data', options=d_list, value=d_list[0], width=200) # Select data set

k_list  = [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300] # All n values computed
kSelect = pn.widgets.Select(name='Select k Value', options=k_list, value=k_list[7], width=200) # Select n value for nearest neighbor

EigenSelect = pn.widgets.Select(name='Select Eigen Solver', options=['arpack','lobpcg','amg'], value='arpack', width=200) # Select eigen value solver


# -

# Add data specific widgets
@pn.depends(DataType.param.value)
def data_widg(d):
    if d == 'rs fMRI':
        panel = pn.Column(pn.Row(rs_fMRI_SubjSelect, rs_fMRI_RunSelect, rs_fMRI_WindowSelect), rs_player)
    elif d == 'task fMRI':
        panel = pn.Row(task_fMRI_SubjSelect, task_fMRI_PureSelect, task_fMRI_WindowSelect)
    else:
        panel = pn.Row()
    return panel


# ***
# ## Plotting Function

# Plotting function using t-SNE
@pn.depends(DataType.param.value, kSelect.param.value, EigenSelect.param.value, rs_fMRI_SubjSelect.param.value, rs_fMRI_RunSelect.param.value,
            rs_fMRI_WindowSelect.param.value, task_fMRI_SubjSelect.param.value, task_fMRI_PureSelect.param.value, task_fMRI_WindowSelect.param.value,
            rs_player.param.value)
def TSNE_3D_plot(d, k, eigen_solver, rs_SBJ, rs_RUN, rs_WL_sec, task_SBJ, task_PURE, task_WL_sec, rs_max_win):
    # Load data and data labels based on selected data
    if d == 'Digits':
        data_df  = dig_img_df
        label_df = dig_lab_df
        cmap = {'0': 'rgb(102, 197, 204)',
                '1': 'rgb(246, 207, 113)',
                '2': 'rgb(248, 156, 116)',
                '3': 'rgb(220, 176, 242)',
                '4': 'rgb(135, 197, 95)',
                '5': 'rgb(158, 185, 243)',
                '6': 'rgb(254, 136, 177)',
                '7': 'rgb(201, 219, 116)',
                '8': 'rgb(139, 224, 164)',
                '9': 'rgb(180, 151, 231)'} # Color map for digits
    elif d == 'Fashion':
        data_df  = fash_img_df
        label_df = fash_lab_df
        cmap = {'T-shirt/Top': 'rgb(95, 70, 144)',
                'Trouser': 'rgb(29, 105, 150)',
                'Pullover': 'rgb(56, 166, 165)',
                'Dress': 'rgb(15, 133, 84)',
                'Coat': 'rgb(115, 175, 72)',
                'Sandal': 'rgb(237, 173, 8)',
                'Shirt': 'rgb(225, 124, 5)',
                'Sneaker': 'rgb(204, 80, 62)',
                'Bag': 'rgb(148, 52, 110)',
                'Ankel Boot': 'rgb(111, 64, 112)'} # Color map for fashion
    elif d == 'rs fMRI':
        data_df, label_df = load_rsfMRI_data(rs_SBJ, rs_RUN, rs_WL_sec)
        cmap = {'Wake':'orange', 'Stage 1':'yellow', 'Stage 2':'green', 'Stage 3':'blue', 'Undetermined':'gray'} # Color key for sleep stages
    elif d == 'task fMRI':
        data_df, label_df = load_taskfMRI_data(task_SBJ, task_PURE, task_WL_sec)
        cmap = {'Rest': 'gray', 'Memory': 'blue', 'Video': '#F4D03F', 'Math': 'green', 'Inbetween': 'black'} # Color key for tasks
    
    # 3D embedding transform created using default Euclidean metric
    embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, eigen_solver=eigen_solver, n_neighbors=k)
    data_transformed = embedding.fit_transform(data_df) # Transform data using embedding
    
    plot_input = pd.DataFrame(data_transformed, columns=['x','y','z']) # Change data to pandas data frame
    plot_input['Label'] = label_df.astype(str) # Add column of number identifier with elements as type string
    
    # Created 3D scatter plot of embedded data and color by label
    if d == 'rs fMRI': # For rs fMRI data only display points up to window determined by player
        plot = px.scatter_3d(plot_input[0:rs_max_win], x='x', y='y', z='z', color='Label', color_discrete_map=cmap, width=700, height=600, opacity=0.7)
        plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)))
        sleep_plot = hv.Curve(label_df, vdims=['Sleep Stage']).opts(width=600, height=150) # Line plot of sleep stage
        dash = pn.Row(plot,sleep_plot)
    else: # For all other data sets display all points
        plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Label', color_discrete_map=cmap, width=700, height=600, opacity=0.7)
        plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)))
        dash = plot
    
    return dash


dash = pn.Column(pn.pane.Markdown("## Laplacian Eigenmap"),pn.Row(DataType, kSelect, EigenSelect), data_widg, TSNE_3D_plot) # Create embedding dashboard

dash_server = dash.show(port=port_tunnel, open=False) # Run dashboard and create link

dash_server.stop() # Stop dashboard link
