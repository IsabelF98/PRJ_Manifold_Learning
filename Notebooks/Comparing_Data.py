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

# # Comparing Data Sets
# This notebook is desighned to compare the different data sets used in the evaluation of different manifold learning techniques. The data sets that will be observed are:
# * Digits Data
# * Fashion Data
# * Task fMRI
# * Rest fMRI
# * Another
#
# For each data set we will observe its corelation matrix, distance matrix, and the distribution of the these matricies.
#
# By Isabel Fernandez 5/13/2021

# Needed imports for notebook
import os
import os.path as osp
import gzip
import pandas as pd
import numpy as np
import scipy
from scipy.io import loadmat
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
import holoviews as hv
from holoviews.operation.datashader import rasterize
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')

PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

# ***
# ## Digits Data
#
# The MNIST digits data is loaded from from Scikit-Learn's toy data sets. It is a copy of the test set of the UCI ML hand-written digits datasets https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits. Each sample is an 8x8 pixel handwriten digit from 0-9.
# * Number of samples: 1797
# * Number of features: 64 (8x8)
# * Number of groups: 10 (0-9)
# * Pixle value range: 0-16
#  
# | Digit | Number of Samples |
# | :---: | :---: |
# | 0 | 178 |
# | 1 | 182 |
# | 2 | 177 |
# | 3 | 183 |
# | 4 | 181 |
# | 5 | 182 |
# | 6 | 181 |
# | 7 | 179 |
# | 8 | 174 |
# | 9 | 180 |

# +
# Load digits data
# ----------------
dig_img_df, dig_lab_df = load_digits(return_X_y=True,as_frame=True)

dig_img_df['Digit'] = dig_lab_df # Combime data frames
dig_img_df = dig_img_df.sort_values(by=['Digit']).reset_index(drop=True) # Sort data by digit
dig_lab_df = pd.DataFrame(dig_img_df['Digit']) # Savel labels as seperate data frame
dig_img_df = dig_img_df.drop(['Digit'], axis=1) # Drop label column

dig_num_samp = dig_img_df.shape[0]  # Save number of samples as a varable

print('++ INFO: Digits data frame dimension ',dig_img_df.shape)
# -

# Compute correlation and distance matrix
# ---------------------------------------
dig_corr = np.corrcoef(dig_img_df) # Correlation matrix
dig_dist = pairwise_distances(dig_img_df, metric='euclidean') # Distance matrix

# Compute distribution of correlation and distance matrix
# -------------------------------------------------------
triangle = np.mask_indices(dig_num_samp, np.triu, k=1) # Top triangle mask for matricies
dig_corr_freq, dig_corr_edges = np.histogram(np.array(dig_corr)[triangle], 50) # Compute histogram of top triangle of correlation matrix (50 bars)
dig_dist_freq, dig_dist_edges = np.histogram(np.array(dig_dist)[triangle], 50) # Compute histogram of top triangle of distance matrix (50 bars)

# Create widgets for digits data
# ------------------------------
dig_CorrRange = pn.widgets.RangeSlider(name='Correlation Range', start=-1, end=1, value=(-1, 1), step=0.1) # Correlation range slider


# Function for fashion data plots
# -------------------------------
@pn.depends(dig_CorrRange.param.value) # Function dependent on correlation range slider
def dig_plots(corr_range):
    # Create matrix and histogram plots
    # ---------------------------------
    dig_corr_img = hv.Image(np.rot90(dig_corr), bounds=(-0.5, -0.5, dig_num_samp-1.5, dig_num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                   height=300, width=400, title='Correlation Matrix').redim.range(z=corr_range)
    dig_dist_img = hv.Image(np.rot90(dig_dist), bounds=(-0.5, -0.5, dig_num_samp-1.5, dig_num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                   height=300, width=400, title='Distance Matrix')
    dig_corr_his = hv.Histogram((dig_corr_edges, dig_corr_freq)).opts(xlabel='Correlation', height=300, width=400, title='Correlation Histogram')
    dig_dist_his = hv.Histogram((dig_dist_edges, dig_dist_freq)).opts(xlabel='Distance', height=300, width=400, title='Distance Histogram')
    
    dash = (dig_corr_img+dig_corr_his+dig_dist_img+dig_dist_his).opts(opts.Layout(shared_axes=False)).cols(2) # Dashboard of all plots
    
    return dash


# Display all digits data plots and widgets
# -----------------------------------------
pn.Column(dig_CorrRange, dig_plots)

# ***
# ## Fashion Data
#
# The MNIST fashion data is loaded from https://github.com/zalandoresearch/fashion-mnist. The data set consists of a training set of 60,000 images and a test set of 10,000 images. Each image is a 28x28 grayscale image, associated with a label from 10 classes. 
# * Number of samples: 10,000 (test only)
# * Number of features: 784 (28x28)
# * Number of groups: 10
# * Pixle value range: 0-255
#
# | Label | Description | Number of Samples |
# | :---: | :---: | :---: |
# | 0 | T-shirt/top | 1000 |
# | 1 | Trouser | 1000 |
# | 2 | Pullover | 1000 |
# | 3 | Dress | 1000 |
# | 4 | Coat | 1000 |
# | 5 | Sandal | 1000 |
# | 6 | Shirt | 1000 |
# | 7 | Sneaker | 1000 |
# | 8 | Bag | 1000 |
# | 9 | Ankle boot | 1000 |

# +
# Load fashion data (test only)
# -----------------------------
fashion_path = os.path.join(PRJDIR,'Data','Fashion_Data') # Path to fashion data set

fash_test_img  = np.load(fashion_path+'/test_images.npy') # Load test images
fash_test_lab  = np.load(fashion_path+'/test_labels.npy') # Load test labels

fash_num_samp = fash_test_img.shape[0] # Save number of samples as a varable

fash_img_df = pd.DataFrame(fash_test_img.reshape((fash_num_samp, 784))) # Flatten image matricies and convert images array to pandas df
fash_lab_df = pd.DataFrame(fash_test_lab) # Convert lables array to pandas df

fash_img_df['Label'] = fash_lab_df # Combime data frames
fash_img_df = fash_img_df.sort_values(by=['Label']).reset_index(drop=True) # Sort data by label
fash_lab_df = pd.DataFrame(fash_img_df['Label']) # Savel labels as seperate data frame
fash_img_df = fash_img_df.drop(['Label'], axis=1) # Drop label column

print('++ INFO: Digits data frame dimension ',fash_img_df.shape)
# -

# Compute correlation and distance matrix
# ---------------------------------------
fash_corr = np.corrcoef(fash_img_df) # Correlation matrix
fash_dist = pairwise_distances(fash_img_df, metric='euclidean') # Distance matrix

# Compute distribution of correlation and distance matrix
# -------------------------------------------------------
triangle = np.mask_indices(fash_num_samp, np.triu, k=1) # Top triangle mask for matricies
fash_corr_freq, fash_corr_edges = np.histogram(np.array(fash_corr)[triangle], 50) # Compute histogram of top triangle of correlation matrix (50 bars)
fash_dist_freq, fash_dist_edges = np.histogram(np.array(fash_dist)[triangle], 50) # Compute histogram of top triangle of distance matrix (50 bars)

# Create widgets for fashion data
# -------------------------------
fash_CorrRange = pn.widgets.RangeSlider(name='Correlation Range', start=-1, end=1, value=(-1, 1), step=0.1) # Correlation range slider


# Function for fashion data plots
# -------------------------------
@pn.depends(fash_CorrRange.param.value) # Function dependent on correlation range slider
def fash_plots(corr_range):
    # Create matrix and histogram plots
    # ---------------------------------
    # raterize() fucntion used for big data set
    fash_corr_img = rasterize(hv.Image(np.rot90(fash_corr), bounds=(-0.5, -0.5, fash_num_samp-1.5, fash_num_samp-1.5)).opts(cmap='viridis', colorbar=True, 
                    height=300, width=400, title='Correlation Matrix')).redim.range(z=corr_range)
    fash_dist_img = rasterize(hv.Image(np.rot90(fash_dist), bounds=(-0.5, -0.5, fash_num_samp-1.5, fash_num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                    height=300, width=400, title='Distance Matrix'))
    fash_corr_his = rasterize(hv.Histogram((fash_corr_edges, fash_corr_freq)).opts(xlabel='Correlation', height=300, width=400, title='Correlation Histogram'))
    fash_dist_his = rasterize(hv.Histogram((fash_dist_edges, fash_dist_freq)).opts(xlabel='Distance', height=300, width=400, title='Distance Histogram'))
    
    dash = (fash_corr_img+fash_corr_his+fash_dist_img+fash_dist_his).opts(opts.Layout(shared_axes=False)).cols(2) # Dashboard of all plots
    
    return dash


# Display all fashion data plots and widgets
# ------------------------------------------
pn.Column(fash_CorrRange, fash_plots)

# ***
# ## Resting State fMRI Data

# +
# Load rs fMRI subject information as a dictionary
# ------------------------------------------------
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
# -------------------------------
rs_fMRI_SubjSelect   = pn.widgets.Select(name='Select Subject', options=rs_fMRI_SubjectList, value=rs_fMRI_SubjectList[0], width=200) # Select subject
rs_fMRI_RunSelect    = pn.widgets.Select(name='Select Run', options=rs_fMRI_SubDict[rs_fMRI_SubjSelect.value], value=rs_fMRI_SubDict[rs_fMRI_SubjSelect.value][1], width=200) # Select run for chosen subject
rs_fMRI_WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60], width=200) # Select window lenght
rs_fMRI_CorrRange = pn.widgets.RangeSlider(name='Correlation Range', start=-1, end=1, value=(-1, 1), step=0.1) # Correlation range slider

# Updates available runs given SubjSelect value
def update_run(event):
    rs_fMRI_RunSelect.options = rs_fMRI_SubDict[event.new]
rs_fMRI_SubjSelect.param.watch(update_run,'value')


# -

# Function for rs fMRI data plots
# -------------------------------
# Function dependent on subject, run, and window length widget values
@pn.depends(rs_fMRI_SubjSelect.param.value, rs_fMRI_RunSelect.param.value, rs_fMRI_WindowSelect.param.value, rs_fMRI_CorrRange.param.value)
def rs_fMRI_plots(SBJ, RUN, WL_sec, corr_range):
    # Load rs fMRI data
    # -----------------
    file_name = SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(WL_sec).zfill(3)+'s_ws002s_'+RUN+'_PCA_vk97.5.swcorr.pkl' # Data file name
    data_path = osp.join('/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/PrcsData',SBJ,'D02_Preproc_fMRI',file_name) # Path to data
    data_df = pd.read_pickle(data_path).T # Read data into pandas data frame
    num_samp = data_df.shape[0]  # Save number of samples as a varable
    
    # Load sleep segmenting data
    # --------------------------
    seg_path = osp.join(PRJDIR,'Data','Samika_DSet02','Sleep_Segments',SBJ+'_'+RUN+'_WL_'+str(WL_sec)+'sec_Sleep_Segments.pkl') # Path to segment data
    seg_df   = pd.read_pickle(seg_path) # Load segment data
 
    # Compute correlation and distance matrix
    # ---------------------------------------
    data_corr = np.corrcoef(data_df) # Correlation matrix
    data_dist = pairwise_distances(data_df, metric='euclidean') # Distance matrix
    
    # Compute distribution of correlation and distance matrix
    # -------------------------------------------------------
    triangle = np.mask_indices(num_samp, np.triu, k=1) # Top triangle mask for matricies
    corr_freq, corr_edges = np.histogram(np.array(data_corr)[triangle], 100) # Compute histogram of top triangle of correlation matrix (100 bars)
    dist_freq, dist_edges = np.histogram(np.array(data_dist)[triangle], 100) # Compute histogram of top triangle of distance matrix (100 bars)
    
    # Create sleep segments plots
    # ---------------------------
    sleep_color_map = {'Wake':'orange', 'Stage 1':'yellow', 'Stage 2':'green', 'Stage 3':'blue', 'Undetermined':'gray'} # Color key for sleep staging
    seg_x = hv.Segments(seg_df, [hv.Dimension('start', range=(-10,num_samp-1.5)), hv.Dimension('start_event', range=(-5,num_samp-1.5)),
                              'end', 'end_event'], 'stage').opts(color='stage', cmap=sleep_color_map, line_width=7, show_legend=True) # x axis segments
    seg_y = hv.Segments(seg_df, [hv.Dimension('start_event', range=(-10,num_samp-1.5)), hv.Dimension('start', range=(-5,num_samp-1.5)),
                              'end_event', 'end'], 'stage').opts(color='stage', cmap=sleep_color_map, line_width=7, show_legend=False) # y axis segments
    seg_plot = (seg_x*seg_y).opts(xlabel=' ', ylabel=' ', show_legend=False) # All segments
    
    # Create matrix and histogram plots
    # ---------------------------------
    # raterize() fucntion used for big data set
    corr_img = rasterize(hv.Image(np.rot90(data_corr), bounds=(-0.5, -0.5, num_samp-1.5, num_samp-1.5)).opts(cmap='viridis', colorbar=True, 
                         title='Correlation Matrix')).redim.range(z=corr_range)
    dist_img = rasterize(hv.Image(np.rot90(data_dist), bounds=(-0.5, -0.5, num_samp-1.5, num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                         title='Distance Matrix'))
    corr_his = rasterize(hv.Histogram((corr_edges, corr_freq)).opts(xlabel='Correlation', height=300, width=400, title='Correlation Histogram'))
    dist_his = rasterize(hv.Histogram((dist_edges, dist_freq)).opts(xlabel='Distance', height=300, width=400, title='Distance Histogram'))
    
    corr_img_wseg = (corr_img*seg_plot).opts(width=600, height=300, legend_position='right') # Overlay sleep segemnt plot with correlation matrix
    dist_img_wseg = (dist_img*seg_plot).opts(width=600, height=300, legend_position='right') # Overlay sleep segemnt plot with distance matrix
    
    dash = (corr_img_wseg+corr_his+dist_img_wseg+dist_his).opts(opts.Layout(shared_axes=False)).cols(2) # Dashboard of all plots
    
    return dash


# Display all rs fMRI data plots and widgets
# ------------------------------------------
pn.Column(pn.Row(rs_fMRI_SubjSelect, rs_fMRI_RunSelect, rs_fMRI_WindowSelect), rs_fMRI_CorrRange, rs_fMRI_plots)

# ***
# ## Multi Task fMRI Data

# Create widgets for task fMRI data
# ---------------------------------
task_fMRI_SubjectList  = ['SBJ06', 'SBJ08', 'SBJ10', 'SBJ12', 'SBJ16', 'SBJ18', 'SBJ20', 'SBJ22', 'SBJ24', 'SBJ26', 'SBJ07', 'SBJ09',
                          'SBJ11', 'SBJ13', 'SBJ17', 'SBJ19', 'SBJ21', 'SBJ23', 'SBJ25', 'SBJ27'] # List of subjects
task_fMRI_SubjSelect   = pn.widgets.Select(name='Select Subject', options=task_fMRI_SubjectList, value=task_fMRI_SubjectList[0], width=200) # Select subject
task_fMRI_PureSelect   = pn.widgets.Select(name='Select Window Type', options=['pure', 'not pure'], value='not pure', width=200) # Select window purity
task_fMRI_WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,45], width=200) # Select window lenght
task_fMRI_CorrRange    = pn.widgets.RangeSlider(name='Correlation Range', start=-1, end=1, value=(-1, 1), step=0.1) # Correlation range slider

# +
# Task to window data frame
# -------------------------
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


# +
# Task segments
# -------------
# Function to create taks segments data frame
# Function input is the task data frames created in the cell above
def taskseg_data(data_df):
    segment_df = pd.DataFrame(columns=['task','start','end']) # Empty segment data frame
    start = 0 # Start at 0th window
    segment = [] # Empty list to append all tasks in a segment (should all be the same task)
    for i,idx in enumerate(data_df.index): # Iterate through data_df index
        task = str(data_df.loc[idx]['Task']) # Save task name
        if idx == (data_df.shape[0]-1): # If its the last index of run
            segment.append(task) # Append task to segment list
            end = start + (len(segment) - 1) # Last window of the segment is the lenght of the segment list plus the start number of the segment
            segment_df = segment_df.append({'task':task, 'start':start, 'end':end}, ignore_index=True) # Append values to segment_df
        elif task == str(data_df.loc[idx+1]['Task']): # If the next task is equal to the current task
            segment.append(task) # Append task to segment list and loop again
        elif task != str(data_df.loc[idx+1]['Task']): # If the next task is NOT equal to the task
            segment.append(task) # Append task to segment list
            end = start + (len(segment) - 1) # Last window of the segment is the lenght of the segment list plus the start number of the segment
            segment_df = segment_df.append({'task':task, 'start':start, 'end':end}, ignore_index=True) # Append values to segment_df
            start = end + 1 # Starting windwo of next task is the last window of the last segment + 1
            segment = [] # New empty segment list for next segment
            
    # Add 0.5 to each end of segment to span entire heat map
    segment_df['start'] = segment_df['start'] - 0.5 
    segment_df['end'] = segment_df['end'] + 0.5
    
    # 'start_event' and 'end_event' represent the axis along which the segments will be (-2 so it is not on top of the heat map)
    segment_df['start_event'] = -2
    segment_df['end_event']   = -2

    return segment_df

WL30_taskseg_df     = taskseg_data(WL30_task_df) # Segment data for WL = 30s not pure windows
WL30pure_taskseg_df = taskseg_data(WL30pure_task_df) # Segment data for WL = 30s pure windows
WL45_taskseg_df     = taskseg_data(WL45_task_df) # Segment data for WL = 45s not pure windows
WL45pure_taskseg_df = taskseg_data(WL45pure_task_df) # Segment data for WL = 30s pure windows


# -

# Function for task fMRI data plots
# ---------------------------------
# Function dependent on subject, pure, and window length widget values
@pn.depends(task_fMRI_SubjSelect.param.value, task_fMRI_PureSelect.param.value, task_fMRI_WindowSelect.param.value, task_fMRI_CorrRange.param.value)
def task_fMRI_plots(SBJ, PURE, WL_sec, corr_range):
    # Define segment data
    # -------------------
    if WL_sec == 30:
        if PURE == 'pure':
            seg_df = WL30pure_taskseg_df
        else:
            seg_df = WL30_taskseg_df
    else:
        if PURE == 'pure':
            seg_df = WL45pure_taskseg_df
        else:
            seg_df = WL45_taskseg_df
    
    # Define PURE varaible based on widget
    # ------------------------------------
    if PURE == 'not pure':  
        PURE = '' # Load data with non pure windows
    
    # Load task fMRI data
    # -------------------
    file_name = SBJ+'_CTask001_WL0'+str(WL_sec)+'_WS01'+PURE+'_NROI0200_dF.mat' # Data file name
    data_path = osp.join('/data/SFIMJGC_HCP7T/PRJ_CognitiveStateDetection02/PrcsData_PNAS2015',SBJ,'D02_CTask001',file_name) # Path to data
    data_df   = loadmat(data_path)['CB']['snapshots'][0][0] # Read data
    num_samp  = data_df.shape[0]  # Save number of samples as a varable
    
    # Create sleep segments plots
    # ---------------------------
    task_color_map = {'Rest': 'gray', 'Memory': 'blue', 'Video': 'yellow', 'Math': 'green', 'Inbetween': 'black'} # Color key for task segments
    seg_x = hv.Segments(seg_df, [hv.Dimension('start', range=(-10,num_samp-1.5)), hv.Dimension('start_event', range=(-5,num_samp-1.5)),
                              'end', 'end_event'], 'task').opts(color='task', cmap=task_color_map, line_width=7, show_legend=True) # x axis segments
    seg_y = hv.Segments(seg_df, [hv.Dimension('start_event', range=(-10,num_samp-1.5)), hv.Dimension('start', range=(-5,num_samp-1.5)),
                              'end_event', 'end'], 'task').opts(color='task', cmap=task_color_map, line_width=7, show_legend=False) # y axis segments
    seg_plot = (seg_x*seg_y).opts(xlabel=' ', ylabel=' ', show_legend=False) # All segments
    
    # Compute correlation and distance matrix
    # ---------------------------------------
    data_corr = np.corrcoef(data_df) # Correlation matrix
    data_dist = pairwise_distances(data_df, metric='euclidean') # Distance matrix
    
    # Compute distribution of correlation and distance matrix
    # -------------------------------------------------------
    triangle = np.mask_indices(num_samp, np.triu, k=1) # Top triangle mask for matricies
    corr_freq, corr_edges = np.histogram(np.array(data_corr)[triangle], 100) # Compute histogram of top triangle of correlation matrix (100 bars)
    dist_freq, dist_edges = np.histogram(np.array(data_dist)[triangle], 100) # Compute histogram of top triangle of distance matrix (100 bars)
    
    # Create matrix and histogram plots
    # ---------------------------------
    corr_img = hv.Image(np.rot90(data_corr), bounds=(-0.5, -0.5, num_samp-1.5, num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                        height=300, width=400, title='Correlation Matrix').redim.range(z=corr_range)
    dist_img = hv.Image(np.rot90(data_dist), bounds=(-0.5, -0.5, num_samp-1.5, num_samp-1.5)).opts(cmap='viridis', colorbar=True,
                        height=300, width=400, title='Distance Matrix')
    corr_his = hv.Histogram((corr_edges, corr_freq)).opts(xlabel='Correlation', height=300, width=400, title='Correlation Histogram')
    dist_his = hv.Histogram((dist_edges, dist_freq)).opts(xlabel='Distance', height=300, width=400, title='Distance Histogram')
    
    corr_img_wseg = (corr_img*seg_plot).opts(width=600, height=300, legend_position='right') # Overlay task segemnt plot with correlation matrix
    dist_img_wseg = (dist_img*seg_plot).opts(width=600, height=300, legend_position='right') # Overlay task segemnt plot with distance matrix
    
    dash = (corr_img_wseg+corr_his+dist_img_wseg+dist_his).opts(opts.Layout(shared_axes=False)).cols(2) # Dashboard of all plots
    
    return dash


# Display all task fMRI data plots and widgets
# --------------------------------------------
pn.Column(pn.Row(task_fMRI_SubjSelect, task_fMRI_PureSelect, task_fMRI_WindowSelect), task_fMRI_CorrRange, task_fMRI_plots)
