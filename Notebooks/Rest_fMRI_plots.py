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
import hvplot.pandas
import holoviews as hv
from holoviews.operation.datashader import rasterize
import panel as pn
from holoviews import dim, opts
from sklearn.manifold  import SpectralEmbedding
import plotly.express as px
pn.extension('plotly')
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


def load_data(SBJ,WL_sec):
    WL_trs = int(WL_sec/2) # Window length in TR's
    WS_trs = 1 # Window spaces (1 TR)
    
    # Load raw ROI time series data
    # -----------------------------
    ts_path = osp.join(DATADIR,'PrcsData',SBJ,'D02_Preproc_fMRI','errts.'+SBJ+'.Craddock_T2Level_0200.wl'+str(WL_sec).zfill(3)+'s.fanaticor_ts.1D') # Path to data
    ts_df   = pd.read_csv(ts_path, sep='\t', header=None) # Read data as pandas dataframe
    Nacq,Nrois = ts_df.shape # Save number of time points and number of ROI's
    roi_names  = ['ROI'+str(r+1).zfill(3) for r in range(Nrois)] # ROI names (should eventually be actual names)
    
    # Load SWC with PCA (already computed)
    # ------------------------------------
    file_name = SBJ+'_fanaticor_Craddock_T2Level_0200_wl'+str(WL_sec).zfill(3)+'s_ws002s_All_PCA_vk97.5.swcorr.pkl' # Data file name
    swc_path  = osp.join(DATADIR,'PrcsData',SBJ,'D02_Preproc_fMRI',file_name) # Path to data
    swc_df    = pd.read_pickle(swc_path) # Read data into pandas data frame
    num_win , num_con = swc_df.T.shape
    
    # Run key
    # -------
    run_df = pd.DataFrame(index=range(0,num_win), columns=['Run'])
    time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ]))] # List of TR's in each run
    run_list = [SubDict[SBJ][i][0] for i in range(0,len(SubDict[SBJ]))] # List of runs for that subject
    x=0
    for i in range(len(time_list)):
        run_df.loc[x:(x-1)+time_list[i]-(WL_trs-1), 'Run'] = [run_list[i]]
        x=x+time_list[i]-(WL_trs-1)
        if i != len(time_list)-1:
            run_df.loc[x:(x-1)+(WL_trs-1), 'Run'] = ['Inbetween Runs']
            x=x+(WL_trs-1)
    
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
    
    return ts_df, swc_df, run_seg_df, win_run_seg_df, run_df


# ***
# ## Carpet Plots and Matrices

@pn.depends(SubjSelect.param.value, WindowSelect.param.value)
def plot(SBJ,WL_sec):
    # Load data
    # ---------
    ts_df, swc_df, run_seg_df, win_run_seg_df, run_df = load_data(SBJ,WL_sec)
    
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
    swc_plot = (xr.DataArray(swc_df.values.T,dims=['Time [Windows]','Connection']).hvplot.image(cmap='jet') * 
                win_run_seg).opts(title='SWC Matrix with PCA', height=300, width=1200, legend_position='right')
    
    plots = pn.Column(ts_plot,swc_plot)
    
    return plots


pn.Column(pn.Row(SubjSelect, WindowSelect), plot)

# ***
# ## Adding surogate conecctions with knonw temporal structure

# +
k_list  = [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300]
kSelect = pn.widgets.Select(name='Select k Value', options=k_list, value=k_list[7], width=200)

percent_list  = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1,2,3,4,5,6,7,8,9,10]
PercentSelect = pn.widgets.Select(name='Select Percent', options=percent_list, value=percent_list[0], width=200)

color_list = ['fake class', 'run']
ColorSelect = pn.widgets.Select(name='Select Color', options=color_list, value=color_list[0], width=200)


# -

@pn.depends(SubjSelect.param.value, WindowSelect.param.value, PercentSelect.param.value, kSelect.param.value, ColorSelect.param.value)
def embeddings(SBJ,WL_sec,PER,k,COLOR):
    WL_trs = int(WL_sec/2) # Window length in TR's
    WS_trs = 1 # Window spaces (1 TR)
    
    # Load SWC data
    # -------------
    ts_df, swc_df, run_seg_df, win_run_seg_df, run_df = load_data(SBJ,WL_sec)
    data_df = swc_df.T.reset_index(drop=True).copy()
    num_win , num_con = data_df.shape # Number of windows and number of connections
    
    # Add Fake Data
    # -------------
    class_length = int(num_win/8) # Classifier legnths
    num_rows = int(num_con * (PER/100)) # Number of rows of data to add
    add_data = np.concatenate([np.repeat(-1.5,class_length),np.repeat(-0.5,class_length),np.repeat(-1.5,class_length),np.repeat(1.5,class_length),
                               np.repeat(0.5,class_length),np.repeat(1.5,class_length),np.repeat(-0.5,class_length),np.repeat(0.5,class_length)]) # Fake data
    for r in range(0,num_rows):
        add_noise = add_data+0.7*np.random.rand(num_win) # Add noise to fake data
        data_df   = pd.concat([data_df,pd.DataFrame(add_noise)], axis=1, ignore_index=True) # Add fake data
    
    # Classifier key
    # --------------
    class_df = pd.DataFrame(index=range(0,num_win), columns=['Class'])
    class_df.loc[0:class_length, 'Class'] = '-1.5'
    class_df.loc[class_length:class_length*2, 'Class'] = '-0.5'
    class_df.loc[class_length*2:class_length*3, 'Class'] = '-1.5'
    class_df.loc[class_length*3:class_length*4, 'Class'] = '1.5'
    class_df.loc[class_length*4:class_length*5, 'Class'] = '1'
    class_df.loc[class_length*5:class_length*6, 'Class'] = '1.5'
    class_df.loc[class_length*6:class_length*7, 'Class'] = '-0.5'
    class_df.loc[class_length*7:class_length*8, 'Class'] = '1'
    
    # Compute Laplacian Eigenmap
    # --------------------------
    # 3D embedding transform created using default Euclidean metric
    embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, eigen_solver='arpack', n_neighbors=k)
    data_transformed = embedding.fit_transform(data_df) # Transform data using embedding
    
    # Plot Embedding
    # --------------
    LE_plot_input = pd.DataFrame(data_transformed, columns=['x','y','z']) # Change data to pandas data frame
    if COLOR == 'fake class':
        LE_plot_input['Class'] = class_df # Add column of number identifier with elements as type string
        LE_plot = px.scatter_3d(LE_plot_input, x='x', y='y', z='z', color='Class', width=700, height=600, opacity=0.7)
    else:
        LE_plot_input['Run'] = run_df
        color_map = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
                                      'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}
        LE_plot = px.scatter_3d(LE_plot_input, x='x', y='y', z='z', color='Run', color_discrete_map=color_map, width=700, height=600, opacity=0.7)
        
    LE_plot = LE_plot.update_traces(marker=dict(size=5,line=dict(width=0)))
    
    # SWC Matrix Plot
    # ---------------
    #fig, ax = plt.subplots(1,1,figsize=(20,5))
    #a = sns.heatmap(data_df.T, cmap='jet', vmin=-2, vmax=2,ax=ax).figure
    
    return LE_plot


dash = pn.Column(pn.Row(SubjSelect, WindowSelect, kSelect, PercentSelect, ColorSelect), embeddings)

dash_server = dash.show(port=port_tunnel, open=False) # Run dashboard and create link

dash_server.stop() # Stop dashboard link

# ***
# ## Correlation between run and connection

# ### Across run SWC data w/ PCA

# +
SBJ = 'sub-S26'
WL_sec = 30
WL_TRs = int(WL_sec/2)

# Load SWC data
# -------------
ts_df, swc_df, run_seg_df, win_run_seg_df, run_df = load_data(SBJ,WL_sec) # Load data
data_df = swc_df.T.reset_index(drop=True).copy() 
num_win , num_con = data_df.shape # Number of windows and number of connections

# Test data
# ---------
time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ]))] # List of all run lenghts in TR's
run1 = np.concatenate([np.repeat(1,time_list[0]-WL_TRs+1),np.repeat(0,num_win-(time_list[0]-WL_TRs+1))]) # Fake data to test correlaitons
run2 = np.concatenate([np.repeat(0,time_list[0]),np.repeat(1,time_list[1]-WL_TRs+1),np.repeat(0,num_win-(time_list[1]-WL_TRs+1)-time_list[0])]) # Fake data to test correlaitons
run3 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]),np.repeat(1,time_list[2]-WL_TRs+1),np.repeat(0,num_win-(time_list[2]-WL_TRs+1)-time_list[0]-time_list[1])]) # Fake data to test correlaitons
run4 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]+time_list[2]),np.repeat(1,time_list[3]-WL_TRs+1),np.repeat(0,num_win-(time_list[3]-WL_TRs+1)-time_list[0]-time_list[1]-time_list[2])]) # Fake data to test correlaitons
run5 = np.concatenate([np.repeat(0,num_win-(time_list[4]-WL_TRs+1)-time_list[5]),np.repeat(1,time_list[4]-WL_TRs+1),np.repeat(0,time_list[5])]) # Fake data to test correlaitons
run6 = np.concatenate([np.repeat(0,num_win-(time_list[5]-WL_TRs+1)),np.repeat(1,time_list[5]-WL_TRs+1)]) # Fake data to test correlaitons

# Compute correlation
# -------------------
def corr_conn(data_df,test_run):
    corr_df = pd.DataFrame(index= range(0,1), columns=[str(con).zfill(4) for con in range(0,num_con)]) # Empty correlation data frame
    for i in range(0,num_con):
        corrcoef = np.corrcoef(data_df.values[:,i], test_run)[0,1] # Compute correlation between fake test data and connection
        corr_df[str(i).zfill(4)] = corrcoef # Add correalation value to data frame
    return corr_df

run1_corr_df = corr_conn(data_df,run1)
run2_corr_df = corr_conn(data_df,run2)
run3_corr_df = corr_conn(data_df,run3)
run4_corr_df = corr_conn(data_df,run4)
run5_corr_df = corr_conn(data_df,run5)
run6_corr_df = corr_conn(data_df,run6)
# -

# Sort data by correlation value
# ------------------------------
run1_sorted_corr_df = run1_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run2_sorted_corr_df = run2_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run3_sorted_corr_df = run3_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run4_sorted_corr_df = run4_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run5_sorted_corr_df = run5_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run6_sorted_corr_df = run6_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})

# Plot correlations
# -----------------
hv.Scatter(run6_sorted_corr_df, 'Connection', 'Correlation').opts(width=1200, tools=['hover']) *\
hv.VLine(400) * hv.VLine(run3_sorted_corr_df.shape[0]-400) * hv.HLine(0)

# Plot test run and data
# ----------------------
hv.Curve(data_df.values[:,4694]).opts(width=1000)*hv.Curve(run1).opts(width=1000)

# +
# Connections to remove
# ---------------------
to_remove = 400 # Number of connections to remove
remove_conn = (run1_sorted_corr_df[0:to_remove]['Connection'].to_list() + run1_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run2_sorted_corr_df[0:to_remove]['Connection'].to_list() + run2_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run3_sorted_corr_df[0:to_remove]['Connection'].to_list() + run3_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run4_sorted_corr_df[0:to_remove]['Connection'].to_list() + run4_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run5_sorted_corr_df[0:to_remove]['Connection'].to_list() + run5_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run6_sorted_corr_df[0:to_remove]['Connection'].to_list() + run6_sorted_corr_df[-to_remove:]['Connection'].to_list())
print('Number of connections to remove:       ', len(remove_conn))
remove_conn = list(set(remove_conn)) # Only unique conections
remove_conn = list(map(int, remove_conn)) # Elements change to int type
to_drop = []
for conn in remove_conn:
    column  = data_df.columns[conn]
    to_drop.append(column)
print('Number of unique connections to remove:', len(to_drop))
print(' ')

# Remove connection with high and low correlations
# ------------------------------------------------
dropped_data_df = data_df.copy()
print('Original shape of data:', dropped_data_df.shape)
dropped_data_df = dropped_data_df.drop(to_drop, axis=1)
print('New shape of data:     ', dropped_data_df.shape)

# +
# Compute LE with droped data
# ---------------------------
embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, eigen_solver='arpack', n_neighbors=70)
data_transformed = embedding.fit_transform(dropped_data_df) # Transform data using embedding

plot_input = pd.DataFrame(data_transformed, columns=['x','y','z']) # Change data to pandas data frame
plot_input['Run'] = run_df # Add column of number identifier with elements as type string

cmap = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
        'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Run', color_discrete_map=cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)))

plot

# +
# Roemove random connections
# --------------------------
test_df = data_df.T.sample(5502).T

embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, eigen_solver='arpack', n_neighbors=70)
data_transformed = embedding.fit_transform(test_df) # Transform data using embedding

plot_input = pd.DataFrame(data_transformed, columns=['x','y','z']) # Change data to pandas data frame
plot_input['Run'] = run_df # Add column of number identifier with elements as type string

cmap = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
        'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Run', color_discrete_map=cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)))

plot
# -

# ### Across run SWC data w/o PCA

# +
SBJ = 'sub-S26'
WL_sec = 30
WL_TRs = int(WL_sec/2) # Window length in TR's

WL_trs = int(WL_sec/2) # Window length in TR's
WS_trs = 1 # Window spaces (1 TR)

# Load raw ROI time series data
# -----------------------------
ts_df, swc_df, run_seg_df, win_run_seg_df, run_df = load_data(SBJ,WL_sec) # Load data
Nacq,Nrois = ts_df.shape # Save number of time points and number of ROI's
roi_names  = ['ROI'+str(r+1).zfill(3) for r in range(Nrois)] # ROI names (should eventually be actual names)

# Compute SWC without PCA
# -----------------------
winInfo = {'durInTR':int(WL_trs),'stepInTR':int(WS_trs)} # Create window information
winInfo['numWins']   = int(np.ceil((Nacq-(winInfo['durInTR']-1))/winInfo['stepInTR'])) # Computer number of windows
winInfo['onsetTRs']  = np.linspace(0,winInfo['numWins'],winInfo['numWins']+1, dtype='int')[0:winInfo['numWins']] # Compute window onsets
winInfo['offsetTRs'] = winInfo['onsetTRs'] + winInfo['durInTR'] # Compute window offsets
winInfo['winNames']  = ['W'+str(i).zfill(4) for i in range(winInfo['numWins'])] # Create window names
window = np.ones((WL_trs,)) # Create boxcar window
# Compute SWC Matrix
for w in range(winInfo['numWins']):
    aux_ts          = ts_df[winInfo['onsetTRs'][w]:winInfo['offsetTRs'][w]]
    aux_ts_windowed = aux_ts.mul(window,axis=0)
    aux_fc          = aux_ts_windowed.corr()
    sel             = np.triu(np.ones(aux_fc.shape),1).astype(np.bool)
    aux_fc_v        = aux_fc.where(sel)
    if w == 0:
        swc_r  = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
    else:
        new_df = pd.DataFrame(aux_fc_v.T.stack().rename(winInfo['winNames'][w]))
        swc_r  = pd.concat([swc_r,new_df],axis=1)
SWC_df = swc_r.apply(np.arctanh)

# +
data_df = SWC_df.T.reset_index(drop=True).copy() 
num_win , num_con = data_df.shape # Number of windows and number of connections

# Test data
# ---------
time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ]))] # List of all run lenghts in TR's
run1 = np.concatenate([np.repeat(1,time_list[0]-WL_TRs+1),np.repeat(0,num_win-(time_list[0]-WL_TRs+1))]) # Fake data to test correlaitons
run2 = np.concatenate([np.repeat(0,time_list[0]),np.repeat(1,time_list[1]-WL_TRs+1),np.repeat(0,num_win-(time_list[1]-WL_TRs+1)-time_list[0])]) # Fake data to test correlaitons
run3 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]),np.repeat(1,time_list[2]-WL_TRs+1),np.repeat(0,num_win-(time_list[2]-WL_TRs+1)-time_list[0]-time_list[1])]) # Fake data to test correlaitons
run4 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]+time_list[2]),np.repeat(1,time_list[3]-WL_TRs+1),np.repeat(0,num_win-(time_list[3]-WL_TRs+1)-time_list[0]-time_list[1]-time_list[2])]) # Fake data to test correlaitons
run5 = np.concatenate([np.repeat(0,num_win-(time_list[4]-WL_TRs+1)-time_list[5]),np.repeat(1,time_list[4]-WL_TRs+1),np.repeat(0,time_list[5])]) # Fake data to test correlaitons
run6 = np.concatenate([np.repeat(0,num_win-(time_list[5]-WL_TRs+1)),np.repeat(1,time_list[5]-WL_TRs+1)]) # Fake data to test correlaitons

# Compute correlation
# -------------------
def corr_conn(data_df,test_run):
    corr_df = pd.DataFrame(index= range(0,1), columns=[str(con).zfill(4) for con in range(0,num_con)]) # Empty correlation data frame
    for i in range(0,num_con):
        corrcoef = np.corrcoef(data_df.values[:,i], test_run)[0,1] # Compute correlation between fake test data and connection
        corr_df[str(i).zfill(4)] = corrcoef # Add correalation value to data frame
    return corr_df

run1_corr_df = corr_conn(data_df,run1)
run2_corr_df = corr_conn(data_df,run2)
run3_corr_df = corr_conn(data_df,run3)
run4_corr_df = corr_conn(data_df,run4)
run5_corr_df = corr_conn(data_df,run5)
run6_corr_df = corr_conn(data_df,run6)
# -

# Sort data by correlation value
# ------------------------------
run1_sorted_corr_df = run1_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run2_sorted_corr_df = run2_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run3_sorted_corr_df = run3_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run4_sorted_corr_df = run4_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run5_sorted_corr_df = run5_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})
run6_sorted_corr_df = run6_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'Connection', 0:'Correlation'})

# Plot correlations
# -----------------
hv.Scatter(run6_sorted_corr_df, 'Connection', 'Correlation').opts(width=1200, tools=['hover'])

# +
# Connections to remove
# ---------------------
to_remove = 1000 # Number of connections to remove
remove_conn = (run1_sorted_corr_df[0:to_remove]['Connection'].to_list() + run1_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run2_sorted_corr_df[0:to_remove]['Connection'].to_list() + run2_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run3_sorted_corr_df[0:to_remove]['Connection'].to_list() + run3_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run4_sorted_corr_df[0:to_remove]['Connection'].to_list() + run4_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run5_sorted_corr_df[0:to_remove]['Connection'].to_list() + run5_sorted_corr_df[-to_remove:]['Connection'].to_list() +
               run6_sorted_corr_df[0:to_remove]['Connection'].to_list() + run6_sorted_corr_df[-to_remove:]['Connection'].to_list())
print('Number of connections to remove:       ', len(remove_conn))
remove_conn = list(set(remove_conn)) # Only unique conections
remove_conn = list(map(int, remove_conn)) # Elements change to int type
to_drop = []
for conn in remove_conn:
    column  = data_df.columns[conn]
    to_drop.append(column)
print('Number of unique connections to remove:', len(to_drop))
print(' ')

# Remove connection with high and low correlations
# ------------------------------------------------
dropped_data_df = data_df.copy()
print('Original shape of data:', dropped_data_df.shape)
dropped_data_df = dropped_data_df.drop(to_drop, axis=1)
print('New shape of data:     ', dropped_data_df.shape)
# -

# Compute LE
# ----------
embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=-1, eigen_solver='arpack', n_neighbors=70)
data_transformed = embedding.fit_transform(dropped_data_df) # Transform data using embedding

# +
# Plot embeddings
# ---------------
plot_input = pd.DataFrame(data_transformed, columns=['x','y','z']) # Change data to pandas data frame
plot_input['Run'] = run_df # Add column of number identifier with elements as type string

cmap = {'SleepAscending':'#DE3163','SleepDescending':'#FF7F50','SleepRSER':'#FFBF00','WakeAscending':'#6495ED',
        'WakeDescending':'#40E0D0','WakeRSER':'#CCCCFF','Inbetween Runs':'gray'}

plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Run', color_discrete_map=cmap, width=700, height=600, opacity=0.7)
plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)))

plot
# -

# ### Across run TS data

# +
SBJ = 'sub-S26'
WL_sec = 30

# Load SWC data
# -------------
ts_df, swc_df, run_seg_df, win_run_seg_df, run_df = load_data(SBJ,WL_sec) # Load data
num_TRs, num_ROIs = ts_df.shape

# Test data
# ---------
time_list = [SubDict[SBJ][i][1] for i in range(0,len(SubDict[SBJ]))] # List of all run lenghts in TR's
run1 = np.concatenate([np.repeat(1,time_list[0]),np.repeat(0,num_TRs-(time_list[0]))]) # Fake data to test correlaitons
run2 = np.concatenate([np.repeat(0,time_list[0]),np.repeat(1,time_list[1]),np.repeat(0,num_TRs-time_list[1]-time_list[0])]) # Fake data to test correlaitons
run3 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]),np.repeat(1,time_list[2]),np.repeat(0,num_TRs-time_list[2]-time_list[0]-time_list[1])]) # Fake data to test correlaitons
run4 = np.concatenate([np.repeat(0,time_list[0]+time_list[1]+time_list[2]),np.repeat(1,time_list[3]),np.repeat(0,num_TRs-time_list[3]-time_list[0]-time_list[1]-time_list[2])]) # Fake data to test correlaitons
run5 = np.concatenate([np.repeat(0,num_TRs-time_list[4]-time_list[5]),np.repeat(1,time_list[4]),np.repeat(0,time_list[5])]) # Fake data to test correlaitons
run6 = np.concatenate([np.repeat(0,num_TRs-time_list[5]),np.repeat(1,time_list[5])]) # Fake data to test correlaitons

# Compute correlation
# -------------------
def corr_ts(data_df,test_run):
    corr_df = pd.DataFrame(index= range(0,1), columns=[str(ROI).zfill(3) for ROI in range(0,num_ROIs)]) # Empty correlation data frame
    for i in range(0,num_ROIs):
        corrcoef = np.corrcoef(data_df.values[:,i], test_run)[0,1] # Compute correlation between fake test data and connection
        corr_df[str(i).zfill(3)] = corrcoef # Add correalation value to data frame
    return corr_df

run1_corr_df = corr_ts(ts_df,run1)
run2_corr_df = corr_ts(ts_df,run2)
run3_corr_df = corr_ts(ts_df,run3)
run4_corr_df = corr_ts(ts_df,run4)
run5_corr_df = corr_ts(ts_df,run5)
run6_corr_df = corr_ts(ts_df,run6)

# Sort data by correlation value
# ------------------------------
run1_sorted_corr_df = run1_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
run2_sorted_corr_df = run2_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
run3_sorted_corr_df = run3_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
run4_sorted_corr_df = run4_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
run5_sorted_corr_df = run5_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
run6_sorted_corr_df = run6_corr_df.T.sort_values(by=0).reset_index().rename(columns={'index':'ROI', 0:'Correlation'})
# -

# Plot correlations
# -----------------
hv.Scatter(run1_sorted_corr_df, 'ROI', 'Correlation').opts(width=1200, tools=['hover'])
