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

# ### Understanding Run Effects on Spectral Embeddings

import holoviews as hv
import pandas as pd
import numpy as np
from sklearn.manifold import SpectralEmbedding
import hvplot.pandas
import xarray as xr
import hvplot.xarray
import panel as pn
import os

port_tunnel  = int(os.environ['PORT2'])
print('++ INFO: Second Port available: %d' % port_tunnel)

# +
r1_nc_slider   = pn.widgets.IntSlider(name='Run 1 (#Conn)', start=0, end=1000, step=1, value=10)
r2_nc_slider   = pn.widgets.IntSlider(name='Run 2 (#Conn)', start=0, end=1000, step=1, value=20)
rx_nc_slider   = pn.widgets.IntSlider(name='#Stable Conn', start=10, end=1000, step=5, value=50)
r1_dur_slider  = pn.widgets.IntSlider(name='Run1 (Duration)', start=1, end=1000, step=10, value=50)
r2_dur_slider  = pn.widgets.IntSlider(name='Run2 (Duration)', start=1, end=1000, step=10, value=200)
knn_slider     = pn.widgets.IntSlider(name='kNN (Spectral Embedding)', start=1, end=1000, step=1, value=100)

desc_panel     = pn.pane.HTML("""<h2>Description:</h2>
                                 <p>This GUI allows us to evaluate how many connections with systematically different connectivity for a given run are sufficient for Spectral Emebddings to capture run effects.</p>
                                 <p>Below we describe the different inputs to this simulation.</p> 
                                 <ol><li><u>Run 1 (#Conn):</u> number of Connections with systematically elevated connectivity during run 1.</li>
                                     <li><u>Run 1 (Duration):</u> number of windows spanning the duration of simmulated run 1.</li>
                                     <li><u>Run 2 (#Conn):</u> number of windows spanning the duration of simmulated run 2.</li>
                                     <li><u>Run 2 (Duration):</u> number of windows spanning the duration of simmulated run 2.</li>
                                     <li><u>#Stable Conn:</u> number of connections with no systematic differences in connectivity across runs.</li>
                                     <li><u>kNN (Spectral Embedding):</u> number of neighbors for the k-neirest neighbor portion of the spectral embedding algorithm.</li>
                                     </ol>
                                     <hr/>""")


# -

def create_sym_data(r1_nc,r2_nc,rx_nc,r1_dur,r2_dur):
    data = pd.DataFrame()
    #Create connections with elevated connectivity during run 1
    for i in np.arange(r1_nc):
        aux = pd.DataFrame(np.concatenate([0.7*np.repeat(1,r1_dur)+0.3*np.random.rand(r1_dur),0.5*np.repeat(1,r2_dur)+0.2*np.random.rand(r2_dur)]))
        data = pd.concat([data,aux],axis=1)
    #Create connections with elevated connectivity during run 2
    for i in np.arange(r2_nc):
        aux = pd.DataFrame(np.concatenate([0.5*np.repeat(1,r1_dur)+0.2*np.random.rand(r1_dur),0.3*np.repeat(1,r2_dur)+0.2*np.random.rand(r2_dur)]))
        data = pd.concat([data,aux],axis=1)
    #Create connections that do not systematically change across runs
    for i in np.arange(rx_nc):
        aux  = pd.DataFrame(0.5*np.repeat(1,r1_dur+r2_dur)+0.3*np.random.rand(r1_dur+r2_dur))
        data = pd.concat([data,aux],axis=1)
    data = data.T
    data.columns =['WIN'+str(i+1).zfill(4) for i in np.arange(data.shape[1])]
    data.index   =['C'+str(i+1).zfill(4) for i in np.arange(data.shape[0])]
    return data


@pn.depends(r1_nc_slider.param.value,r2_nc_slider.param.value,rx_nc_slider.param.value,r1_dur_slider.param.value,r2_dur_slider.param.value,knn_slider.param.value)
def plot_data(r1_nc,r2_nc,rx_nc,r1_dur,r2_dur,knn):
    ncon = r1_nc + r2_nc + rx_nc
    #Create simulated data
    data = create_sym_data(r1_nc,r2_nc,rx_nc,r1_dur,r2_dur)
    #Carpet plot of connectivity
    swc_plot = data.hvplot.heatmap(cmap='jet',clim=(0,1), 
                                   xticks=[0,r1_dur,r1_dur+r2_dur], xlabel='Time (Windows)',
                                   yticks=[0,r1_nc,r1_nc + r2_nc, r1_nc + r2_nc + rx_nc],ylabel='Connections',
                                   title='Simulated SWC Matrix', shared_axes=False).opts(toolbar=None)
    #Correlation matrix for SCC
    cm = data.corr()
    cm.columns = ['WIN'+str(i+1).zfill(4) for i in np.arange(r1_dur+r2_dur)]
    cm.index   = ['WIN'+str(i+1).zfill(4) for i in np.arange(r1_dur+r2_dur)]
    cm_plot    = cm.hvplot.heatmap(cmap='RdBu_r',clim=(-1,1), 
                                   shared_axes=False,
                                   xlabel='Time (Windows)',ylabel='Time (Windows)',
                                   title='Win-2-Win Correlation (Similarity across samples)',
                                   xticks=[0,r1_dur,r1_dur+r2_dur],
                                   yticks=[0,r1_dur,r1_dur+r2_dur])
    # Create Embedding
    se = SpectralEmbedding(n_components=2, n_neighbors=knn)
    se.fit(data.T)
    se_df                       = pd.DataFrame(se.embedding_)
    se_df.columns               = ['x','y']
    se_df['class']              = 'Run 2'
    se_df.loc[0:r1_dur,'class'] = 'Run 1'
    se_plot                     = se_df.hvplot.scatter(x='x',y='y',color='class',
                                                      title='Spectral Embedding',
                                                      xlabel='SE Dim 1',
                                                      ylabel='SE Dim 2')
    return (swc_plot+cm_plot+se_plot).cols(1)


app = pn.Row(pn.Column(desc_panel,pn.Row(r1_nc_slider,r1_dur_slider),
                 pn.Row(r2_nc_slider,r2_dur_slider),
                 pn.Row(rx_nc_slider),
                 pn.Row(knn_slider)),
       plot_data)

app_server = app.show(port=port_tunnel,open=False)

app_server.stop()
