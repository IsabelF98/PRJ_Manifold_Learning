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

# # Resting State fMRI Data Carpet Plots
#
# This notbook is desighned to look at the resting state data to determine why run specific information remains in the data following pre processing. The three plots that are to be computed are:
# 1. A carpet plot of the raw ROI data before SWC
# 2. A SWC matrix of the data without PCA
# 3. A SWC matrix of the data with PCA
#
# Isabel Fernandez 05/28/2021

# Needed imports for notebook
import os
import os.path as osp
import pandas as pd
import xarray as xr
import numpy as np
import scipy
import sklearn
import hvplot.xarray
import holoviews as hv
from holoviews.operation.datashader import rasterize
import panel as pn
from holoviews import dim, opts
hv.extension('bokeh')

# +
PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

port_tunnel = int(os.environ['PORT2']) # Get port tunnel for gui display
print('++ INFO: Second Port available: %d' % port_tunnel)

# +
DATADIR = '/data/SFIM_Vigilance/PRJ_Vigilance_Smk02/' # Path to project directory
sub_DF = pd.read_pickle(DATADIR+'Notebooks/utils/valid_run_df.pkl') # Data frame of all subjects info for vaild runs

# Dictionary of subject with valid runs
# The dictionary is organized by subject. Keys are the subject and the values are a list of tuples as such:
# (run name, number of TR's in the data, min index of run in the concatinated data, max index of run in the concatinated data)
SubDict = {} # Empty dictionary
for i,idx in enumerate(sub_DF.index): # Iterate through each row of data frame
    sbj  = sub_DF.loc[idx]['Sbj']
    run  = sub_DF.loc[idx]['Run']
    time = sub_DF.loc[idx]['Time']
    tp_min = sub_DF.loc[idx]['Time Point Min']
    tp_max = sub_DF.loc[idx]['Time Point Max']
    if sbj in SubDict.keys():
        SubDict[sbj].append((run,time,tp_min,tp_max)) # Add run tuple (described above)
    else:
        SubDict[sbj] = [(run,time,tp_min,tp_max)] # If subject is not already in the directory a new element is created
SubjectList = list(SubDict.keys()) # list of subjects
# -

SubjSelect = pn.widgets.Select(name='Select Subject', options=SubjectList, value=SubjectList[0], width=200) # Select subject
WindowSelect = pn.widgets.Select(name='Select Window Length (in seconds)', options=[30,46,60], width=200) # Select window lenght


# +
def load_data(SBJ,WL_sec):
    WL_trs = int(WL_sec/2) # Window length in TR's
    WS_trs = 1 # Window spaces (1 TR)
    
    # Load raw ROI time series data
    # -----------------------------
    ts_path = osp.join(DATADIR,'PrcsData',SBJ,'D02_Preproc_fMRI','errts.'+SBJ+'.Craddock_T2Level_0200.wl'+str(WL_sec).zfill(3)+'s.fanaticor_ts.1D') # Path to data
    ts_df   = pd.read_csv(ts_path, sep='\t', header=None) # Read data as pandas dataframe
    Nacq,Nrois = ts_df.shape # Save number of time points and number of ROI's
    roi_names  = ['ROI'+str(r+1).zfill(3) for r in range(Nrois)] # ROI names (should eventually be actual names)
    
#    # Compute SWC without PCA
#    # -----------------------
#    winInfo = {'durInTR':int(WL_trs),'stepInTR':int(WS_trs)} # Create window information
#    winInfo['numWins']   = int(np.ceil((Nacq-(winInfo['durInTR']-1))/winInfo['stepInTR'])) # Computer number of windows
#    winInfo['onsetTRs']  = np.linspace(0,winInfo['numWins'],winInfo['numWins']+1, dtype='int')[0:winInfo['numWins']] # Compute window onsets
#    winInfo['offsetTRs'] = winInfo['onsetTRs'] + winInfo['durInTR'] # Compute window offsets
#    winInfo['winNames']  = ['W'+str(i).zfill(4) for i in range(winInfo['numWins'])] # Create window names
#    window = np.ones((WL_trs,)) # Create boxcar window
#    # Compute SWC Matrix
#    for w in range(winInfo['numWins']):
#        aux_ts          = ts_df[winInfo['onsetTRs'][w]:winInfo['offsetTRs'][w]]
#        aux_ts_windowed = aux_ts.mul(window,axis=0)
#        aux_fc          = aux_ts_windowed.corr()
#        sel             = np.triu(np.ones(aux_fc.shape),1).astype(np.bool)
#        aux_fc_v        = aux_fc.where(sel)
#        if w == 0:
#            swc_r  = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
#        else:
#            new_df = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
#            swc_r  = pd.concat([swc_r,new_df],axis=1)
#    SWC_wo_PCA = swc_r.apply(np.arctanh)
    
    # Load SWC with PCA (already computed)
    # ------------------------------------
    file_name = SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(WL_sec).zfill(3)+'s_ws002s_All_PCA_vk97.5.swcorr.pkl' # Data file name
    swc_path  = osp.join(DATADIR,'PrcsData',SBJ,'D02_Preproc_fMRI',file_name) # Path to data
    swc_df    = pd.read_pickle(swc_path) # Read data into pandas data frame
    
    # Load run segments
    # -----------------
    run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ]))] # List of all runs
    time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ]))] # List of all run lenghts in TR's (in the same order as runs in list above)
    
    run_seg_df = pd.DataFrame(columns=['run','start','end']) # Emptly data frame for segment legths of runs
    # For each run a row is appended into the data frame created above (run_seg_df) with the run name and the start and end TR of the run
    y = 0 # Start at 0th TR
    for i in range(len(run_list)):
        time = time_list[i] # Number of windows in run
        run  = run_list[i] # Name of run
        end  = y + time - 1 # Last TR of run
        run_seg_df = run_seg_df.append({'run':run,'start':y,'end':end}, ignore_index=True) # Append run info
        y = end + 1
    
    win_run_seg_df = pd.DataFrame(columns=['run','start','end']) # Emptly data frame for segment legths of runs    
    # For each run a row is appended into the data frame created above (win_run_seg_df) with the run name and the start and end window of the run
    # For the windows that overlap runs the run will be called 'Inbetween Runs'
    x=0 # Starting at 0th window
    for i in range(len(run_list)):
        time = time_list[i] # Number of windows in run
        run  = run_list[i] # Name of run
        end = x + time - WL_trs # Last window of run
        if i == len(run_list)-1: # If its the last run no need to append inbetween run
            win_run_seg_df = win_run_seg_df.append({'run':run,'start':x,'end':end}, ignore_index=True) # Append run info
        else: 
            win_run_seg_df = win_run_seg_df.append({'run':run,'start':x,'end':end}, ignore_index=True) # Append run info
            x=end+1
            win_run_seg_df = win_run_seg_df.append({'run':'Inbetween Runs','start':x,'end':(x-1)+(WL_trs-1)}, ignore_index=True) # Append inbetween run info
            x=x+(WL_trs-1)

    # Add 0.5 to each end of segment to span entire heat map
    run_seg_df['start'] = run_seg_df['start'] - 0.5 
    run_seg_df['end']   = run_seg_df['end'] + 0.5
    win_run_seg_df['start'] = win_run_seg_df['start'] - 0.5 
    win_run_seg_df['end']   = win_run_seg_df['end'] + 0.5
        
    # 'start_event' and 'end_event' represent the axis along which the segments will be (-5 so it is not on top of the heat map or sleep segments)
    run_seg_df['start_event'] = -5
    run_seg_df['end_event']   = -5
    win_run_seg_df['start_event'] = -5
    win_run_seg_df['end_event']   = -5
    
    return ts_df, swc_df, run_seg_df, win_run_seg_df


# -

@pn.depends(SubjSelect.param.value, WindowSelect.param.value)
def plot(SBJ,WL_sec):
    # Load data
    # ---------
    ts_df, swc_df, run_seg_df, win_run_seg_df = load_data(SBJ,WL_sec)
    
    # Run segments plot
    # -----------------
    # Color key for runs
    run_color_map = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
                         'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'black'}
    # Plot of run segements along the x axis
    run_seg = hv.Segments(run_seg_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event'],
                          'run').opts(color='run', cmap=run_color_map, line_width=10, show_legend=True,)
    win_run_seg = hv.Segments(win_run_seg_df, [hv.Dimension('start'), hv.Dimension('start_event'), 'end', 'end_event'],
                              'run').opts(color='run', cmap=run_color_map, line_width=10, show_legend=True,)
    
    # Time series plot
    # ----------------
    ts_plot = (xr.DataArray(ts_df.values,dims=['Time [TRs]','ROIs']).hvplot.image(cmap='gray') * 
               run_seg).opts(title='ROI Time Series Carpet Plot', height=300, width=1200, legend_position='right')
    
    # SWC plot with PCA
    # -----------------
    swc_plot = (rasterize(xr.DataArray(swc_df.values.T,dims=['Time [Windows]','Connection']).hvplot.image(cmap='jet')) * 
                win_run_seg).opts(title='SWC Matrix with PCA', height=300, width=1200, legend_position='right')
    
    plots = pn.Column(ts_plot,swc_plot)
    
    return plots


pn.Column(pn.Row(SubjSelect, WindowSelect), plot)
