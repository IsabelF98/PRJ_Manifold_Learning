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
#     display_name: Vigilance
#     language: python
#     name: vigilance
# ---

# # Laplacian Eigenmaps
#
# This notebook is desighned to observe and understand further the plots generated by the sklearn.manifold.SpectralEmbedding function. We do so by displaying two plots, one using the Euclidean metric and one using the distance correlation to compute the affinity matrix. We also provide selector widgets to choose a different eigen solver algorithum and n value for nearest neighbor calculation.
#
# Isabel Fernandez 3/1/2021

import os
import os.path as osp
import pandas as pd
import numpy as np
from sklearn.manifold  import SpectralEmbedding
from sklearn.neighbors import kneighbors_graph
import scipy
from scipy.spatial.distance import correlation as dis_corr
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
n_cpus = 26 # Number of cpus

PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory path

port_tunnel = int(os.environ['PORT2']) # Get port tunnel for gui display
print('++ INFO: Second Port available: %d' % port_tunnel)
# -

# ***
# ## Load Digits Data

# Load number data and print dimensions
data_df, num_df = load_digits(return_X_y=True,as_frame=True)
print('++ INFO: Data frame dimension ',data_df.shape)

# Make copy of original data
data_df_orig = data_df.copy()
num_df_orig  = num_df.copy()

# Sort data by number in ascending order
data_df['Number'] = num_df_orig
data_df = data_df.sort_values(by=['Number']).reset_index(drop=True)
num_df  = pd.DataFrame(data_df['Number'])
data_df = data_df.drop(['Number'], axis=1)

# +
# Segment data by number and only use 1, 3, 7, and 8
data_df['Number'] = num_df_orig

ones   = data_df[data_df['Number'] == 1][0:25]
threes = data_df[data_df['Number'] == 3][0:100]
sevens = data_df[data_df['Number'] == 7][0:160]
eights = data_df[data_df['Number'] == 8][0:55]
data_df = pd.concat([ones, threes, sevens, eights]).reset_index(drop=True)

num_df  = pd.DataFrame(data_df['Number'])
data_df = data_df.drop(['Number'], axis=1)

# +
# Small number of data points
data_df['Number'] = num_df_orig

twos  = data_df[data_df['Number'] == 2][0:15]
fives = data_df[data_df['Number'] == 5][0:22]
nines = data_df[data_df['Number'] == 9][0:7]
data_df = pd.concat([twos, fives, nines]).reset_index(drop=True)

num_df  = pd.DataFrame(data_df['Number'])
data_df = data_df.drop(['Number'], axis=1)
# -

# ***
# ## Save Affinity Matrix with Distance Correlation Metric
#
# These cells compute the affinity matrix using the scipy distance correlation metric for several n values (for computing nearest neighbor) and saves the matricies as npz files in a directory called "Affinity_Matrix" within the directory this notebook is located.
#
# Note: Thses cells do not need to be executed every time you wish to run this notbook. If the files already exist then you can skip these cells.

# Load affinity matrix for each n value using distance correlation
data_affinity_3   = kneighbors_graph(data_df, n_neighbors=3, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=3')
data_affinity_4   = kneighbors_graph(data_df, n_neighbors=4, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=4')
data_affinity_5   = kneighbors_graph(data_df, n_neighbors=5, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=5')
data_affinity_6   = kneighbors_graph(data_df, n_neighbors=6, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=6')
data_affinity_7   = kneighbors_graph(data_df, n_neighbors=7, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=7')
data_affinity_8   = kneighbors_graph(data_df, n_neighbors=8, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=8')
data_affinity_9   = kneighbors_graph(data_df, n_neighbors=9, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=9')
data_affinity_10  = kneighbors_graph(data_df, n_neighbors=10, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=10')
data_affinity_12  = kneighbors_graph(data_df, n_neighbors=12, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=12')
data_affinity_14  = kneighbors_graph(data_df, n_neighbors=14, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=14')
data_affinity_16  = kneighbors_graph(data_df, n_neighbors=16, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=16')
data_affinity_18  = kneighbors_graph(data_df, n_neighbors=18, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=18')
data_affinity_20  = kneighbors_graph(data_df, n_neighbors=20, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=20')
data_affinity_25  = kneighbors_graph(data_df, n_neighbors=25, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=25')
data_affinity_30  = kneighbors_graph(data_df, n_neighbors=30, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=30')
data_affinity_35  = kneighbors_graph(data_df, n_neighbors=35, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=35')
data_affinity_40  = kneighbors_graph(data_df, n_neighbors=40, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=40')
data_affinity_45  = kneighbors_graph(data_df, n_neighbors=45, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=45')
data_affinity_50  = kneighbors_graph(data_df, n_neighbors=50, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=50')
data_affinity_60  = kneighbors_graph(data_df, n_neighbors=60, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=60')
data_affinity_70  = kneighbors_graph(data_df, n_neighbors=70, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=70')
data_affinity_80  = kneighbors_graph(data_df, n_neighbors=80, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=80')
data_affinity_90  = kneighbors_graph(data_df, n_neighbors=90, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=90')
data_affinity_100 = kneighbors_graph(data_df, n_neighbors=100, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=100')
data_affinity_150 = kneighbors_graph(data_df, n_neighbors=150, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=150')
data_affinity_200 = kneighbors_graph(data_df, n_neighbors=200, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=200')
data_affinity_250 = kneighbors_graph(data_df, n_neighbors=250, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=250')
data_affinity_300 = kneighbors_graph(data_df, n_neighbors=300, mode='connectivity', n_jobs=n_cpus, include_self=True, metric=dis_corr)
print('++ INFO: Finished affinity matrix for n=300')

# +
os.system('if [ ! -d '+PRJDIR+'/Data/LE_Affinity_Matrix ]; then mkdir '+PRJDIR+'/Data/LE_Affinity_Matrix; fi') # Creates directory for affninity matrices if doesnt already exist

# Dictionary of affinity matrices created above
affinity_dict = {3:data_affinity_3, 4:data_affinity_4, 5: data_affinity_5, 6:data_affinity_6, 7:data_affinity_7, 8:data_affinity_8, 9:data_affinity_9,
                 10:data_affinity_10, 12:data_affinity_12, 14:data_affinity_14, 16:data_affinity_16, 18:data_affinity_18, 20:data_affinity_20, 25:data_affinity_25,
                 30:data_affinity_30, 35:data_affinity_35, 40:data_affinity_40, 45:data_affinity_45, 50:data_affinity_50, 60:data_affinity_60, 70:data_affinity_70,
                 80:data_affinity_80, 90:data_affinity_90, 100:data_affinity_100, 150:data_affinity_150, 200:data_affinity_200, 250:data_affinity_250,
                 300:data_affinity_300}

# Save affinity matricies to Affinity_Matrix directory
for n in affinity_dict.keys():
    affin = affinity_dict[n]
    path = osp.join(PRJDIR,'Data','LE_Affinity_Matrix','affinity_matrix_'+str(n)+'.dist_corr.ratio_data.npz')
    save_npz(path, affin)
# -

# ***
# ## Create Widgets

n_list = [3,4,5,6,7,8,9,10,12,14,16,18,20,25,30,35,40]#,45,50,60,70,80,90,100,150,200,250,300] # All n values computed
eigen_sel = pn.widgets.Select(name='Select Eigen Solver', options=['arpack','lobpcg','amg'], value='arpack', width=200) # Select eigen value solver
n_sel     = pn.widgets.Select(name='Select k Value', options=n_list, value=n_list[7], width=200) # Select n value for nearest neighbor


# ***
# ## Plotting Functions

# Returns embedding plot for Euclidean metric
@pn.depends(eigen_sel.param.value,n_sel.param.value)
def plot_embedding1(eigen_solver,n):
    n = int(n) # n value for nearest neighbor
    
    # 3D embedding transform created using default Euclidean metric
    embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=n_cpus, eigen_solver=eigen_solver, n_neighbors=n)
    data_transformed = embedding.fit_transform(data_df) # Transform data using embedding
    
    plot_input = pd.DataFrame(data_transformed,columns=['x','y','z']) # Change data to pandas data frame
    plot_input['Number'] = num_df.astype(str) # Add column of number identifier with elements as type string
    
    # Created 3D scatter plot of embedded data and color by number
    plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Number', width=700, height=600, opacity=0.7, title='Euclidean Metric k='+str(n))
    plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)), hovertemplate=["idx: "+str(x) for x in plot_input.index])

    return plot


# Returns embedding plot for distance correlation metric
@pn.depends(eigen_sel.param.value,n_sel.param.value)
def plot_embedding2(eigen_solver,n):
    n = int(n) # n value for nearest neighbor
    
    # 3D embedding transform created using distance correlation metric
    embedding = SpectralEmbedding(n_components=3, affinity='precomputed', n_jobs=n_cpus, eigen_solver=eigen_solver)
    
    path = osp.join(PRJDIR,'Data','LE_Affinity_Matrix','affinity_matrix_'+str(n)+'.dist_corr.ratio_data.npz') # Path to precomputed affinity matrix using distance correlation
    data_affinity = load_npz(path) # Load affinity matrix
    
    #data_affinity = scipy.sparse.csr_matrix.toarray(data_affinity) 
    #data_affinity = np.where(data_affinity==0.5, 0, data_affinity) # TEST 1
    #data_affinity = data_affinity - np.identity(data_affinity.shape[0]) # TEST 2
    
    data_affinity = 0.5 * (data_affinity + data_affinity.T) # Make affinity matrix symetric by finding the average
    data_transformed = embedding.fit_transform(data_affinity) # Transform data using embedding
    
    plot_input = pd.DataFrame(data_transformed,columns=['x','y','z']) # Change data to pandas data frame
    plot_input['Number'] = num_df.astype(str) # Add column of number identifier with elements as type string
    
    # Created 3D scatter plot of embedded data and color by number
    plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Number', width=600, height=550, opacity=0.7, title='Distance Correlation Metric k='+str(n))
    plot = plot.update_traces(marker=dict(size=5,line=dict(width=0)), hovertemplate=["t: {}".format(x) for x in  plot_input.index])
    
    return plot


# Retruns plot of affinity matrix for Euclidean metric
@pn.depends(eigen_sel.param.value,n_sel.param.value)
def affinity_matrix1(eigen_solver,n):
    n = int(n) # n value for nearest neighbor
    
    # 3D embedding transform created using default Euclidean metric
    embedding = SpectralEmbedding(n_components=3, affinity='nearest_neighbors', n_jobs=n_cpus, eigen_solver=eigen_solver, n_neighbors=n)
    data_transformed = embedding.fit_transform(data_df) # Transform data using embedding
    
    data_affinity = scipy.sparse.csr_matrix.toarray(embedding.affinity_matrix_) # Affinity matrix as a numpy array
    
    plot = hv.Image(data_affinity).opts(width=500, height=400, colorbar=True, cmap='Greys', title='Euclidean Affinity k='+str(n)) # Plot aray as image
    
    return plot


# Retruns plot of affinity matrix for distance correlation metric
@pn.depends(eigen_sel.param.value,n_sel.param.value)
def affinity_matrix2(eigen_solver,n):
    n = int(n) # n value for nearest neighbor
    
    # 3D embedding transform created using default Euclidean metric
    embedding = SpectralEmbedding(n_components=3, affinity='precomputed', n_jobs=n_cpus, eigen_solver=eigen_solver)
    
    path = osp.join(PRJDIR,'Data','LE_Affinity_Matrix','affinity_matrix_'+str(n)+'.dist_corr.ratio_data.npz') # Path to precomputed affinity matrix using distance correlation
    data_affinity = load_npz(path) # Load affinity matrix
    data_affinity = 0.5 * (data_affinity + data_affinity.T) # Make affinity matrix symetric by finding the average
    
    data_affinity = scipy.sparse.csr_matrix.toarray(data_affinity) # Affinity matrix as a numpy array
    
    #data_affinity = np.where(data_affinity==0.5, 0, data_affinity) # TEST 1
    #data_affinity = data_affinity - np.identity(data_affinity.shape[0]) # TEST 2
    
    plot = hv.Image(data_affinity).opts(width=500, height=400, colorbar=True, cmap='Greys', title='Distance Correlation Affinity k='+str(n)) # Plot aray as image
    
    return plot


# ***
# ## Dash Board Display

dash = pn.Column(pn.Row(eigen_sel,n_sel),pn.Row(plot_embedding1, plot_embedding2),pn.Row(affinity_matrix1, affinity_matrix2)) # Create embedding dash board

dash_server = dash.show(port=port_tunnel, open=False) # Run dashboard and create link

dash_server.stop() # Stop dashboard link

# ***
# ## Display Number as Image
# Each diget coresponds to 64 pixels, giving you an 8x8 picture. Each pixel value ranges from 0-16.

# +
idx = 8 # Index value for the number you wish to display (index determined by 'data_df')
dig = pd.Series(data_df.loc[idx]).array # Turn pixel values for that diget into an array

dig_df = pd.DataFrame() # Emptly data frame for diget image 

# Append every 8 pixels into 'dig_df' as a column
x=0
for row in [1,2,3,4,5,6,7,8]:
    dig_df[row] = dig[x:x+7]
    x=x+8

# Display dig_df as an image to display diget
hv.Image(dig_df.to_numpy().T).opts(cmap='Greys')
# -

# ***
# ## Affinity Matrix Analysis
# These scripts are desighned to demonstrate how the affinity matrix for the Laplacian Eigenmap is computed. To show this we use a simple example of 6 2D points and find the nearest neighbor affnitnity matrix with an n value of 2. In this example we use Sci-kit Learns function's sklearn.neighbors.NearestNeighbors().

from sklearn.neighbors import NearestNeighbors
import numpy as np
import holoviews as hv
import plotly.express as px
hv.extension('matplotlib')

X = np.array([[-3, -2], [-2, -1], [-1, -1], [1, 1], [2, 1], [3, 2]]) # Example points
hv.Points(X) # Plot the example points

nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(X) # Compute nearest neighbor with n=2 and a distence metric of Euclidean
distances, indices = nbrs.kneighbors(X) # Execute nearest neighbor on our example points 

distances # Display distances between point and its nearest neighbor

indices # Display the points and its nearest neighbor

affinity = nbrs.kneighbors_graph(X).toarray() # Create affinity matrix using nearest neighbor computed above
affinity # Display affinity matrix

affinity_sym = 0.5*(affinity+affinity.T) # Make 'affinity' symetric by averaging each point
affinity_sym # Display symetric affinity matrix

# As you can see, the affinity matrix labeled 'affinity' is not symmetric. With an assigned n of 2 each point must find 2 nearest neighbors including itself. A 1 in the matrix means that that point is a nearest neighbor for the point in which the row pertains to. In the example of the first point (the first row in the matrix), its nearest neighbor is itself and the second point. For point 2 (the second row in the matrix) its nearest neighbor is itself and point 3 not point 1. This happens because point 3 is closer to 2 then point 1 in terms of the Euclidean distance.
#
# The affinity matrix is made symmetric by averaging it by its transpose. As you can see from 'affinity_sym' some elements of the matrix now have a value of 0.5. For example, the first row (pertaining to point 1) has a value of 0.5 where it used to have a value of 1 in the second column. This tells us that point 1's nearest neighbor is point 2 but point 1 is not a nearest neighbor of point 2.
#
# Looking back at the matrix 'affinity', if we sum the matrix by its rows each row will have a value of 2 (the n value we assigned) because each point is required to find its 2 nearest neighbors. If we sum the matrix by its columns, we will get different values for each column. Column 1 will sum to 1, telling us that it is only a nearest neighbor to itself. Column 2, on the other hand, will sum to 3, telling us that it is the nearest neighbor to 2 other points besides itself.

# ***
# ## Distance Correlation Matrix
# These cells are created to test if the sepctral embedding can be computed using a distance matrix rather then an affinity matrix.

dis_corr_matrix = pairwise_distances(data_df[0:500], metric=dis_corr) # A distance correlation matrix is computed using the diget data (this matrix is symmetric)
dis_corr_matrix
hv.Image(dis_corr_matrix).opts(width=400, height=300, colorbar=True, cmap='jet') # Plot distance matrix

D = 0.25 # Distance condition
cond_matrix = np.where(dis_corr_matrix<D, 1, 0) # All distances less then 'D' are assighned a 1 and all distances greater then 'D' are assighned a 0
hv.Image(cond_matrix).opts(width=400, height=300, colorbar=True, cmap='Greys') # Plot conditional matrix 'cond_matrix'

embedding = SpectralEmbedding(n_components=3, affinity='precomputed', n_jobs=n_cpus, eigen_solver='arpack') # Compute 3D spectral embedding
data_transformed = embedding.fit_transform(np.abs(cond_matrix)) # Compute data transform by using 'cond_matrix' created above

plot_input = pd.DataFrame(data_transformed,columns=['x','y','z']) # Save transformed data at data frame
plot_input['Number'] = num_df.astype(str) # Add column of diget lables
plot = px.scatter_3d(plot_input, x='x', y='y', z='z', color='Number', width=500, height=400, opacity=0.7) # Create plot of embedding
#plot # Display plot
