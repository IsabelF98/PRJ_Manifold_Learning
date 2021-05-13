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
import sklearn
from sklearn.datasets import load_digits
from sklearn.metrics import pairwise_distances
import holoviews as hv
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
# * Number of samples in groups: (0, 178), (1, 182), (2, 177), (3, 183), (4, 181), (5, 182), (6, 181), (7, 179), (8, 174), (9, 180)

# Load digits data
# ----------------
dig_data_df, dig_num_df = load_digits(return_X_y=True,as_frame=True)
print('++ INFO: Digits data frame dimension ',dig_data_df.shape)

# Compute correlation and distance matrix
# ---------------------------------------
dig_corr = np.corrcoef(dig_data_df) # Correlation matrix
dig_dist = pairwise_distances(dig_data_df, metric='euclidean') # Distance matrix

# Compute distribution of correlation and distance matrix
# -------------------------------------------------------
triangle = np.mask_indices(dig_data_df.shape[0], np.triu, k=1) # Top triangle mask for matricies
dig_corr_freq, dig_corr_edges = np.histogram(np.array(dig_corr)[triangle], 50) # Compute histogram of top triangle of correlation matrix (50 bars)
dig_dist_freq, dig_dist_edges = np.histogram(np.array(dig_dist)[triangle], 50) # Compute histogram of top triangle of distance matrix (50 bars)

# Create matrix and histogram plots
# ---------------------------------
dig_corr_img = hv.Image(dig_corr, bounds=(-0.5, -0.5, 1797.5, 1797.5)).opts(colorbar=True, height=300, width=400, title='Correlation Matrix')
dig_dist_img = hv.Image(dig_dist, bounds=(-0.5, -0.5, 1797.5, 1797.5)).opts(colorbar=True, height=300, width=400, title='Distance Matrix')
dig_corr_his = hv.Histogram((dig_corr_edges, dig_corr_freq)).opts(xlabel='Correlation', height=300, width=400, title='Correlation Histogram')
dig_dist_his = hv.Histogram((dig_dist_edges, dig_dist_freq)).opts(xlabel='Distance', height=300, width=400, title='Distance Histogram')

# Display all digit data plots
# ----------------------------
(dig_corr_img+dig_corr_his+dig_dist_img+dig_dist_his).opts(opts.Layout(shared_axes=False)).cols(2)

# ***
# ## Fashion Data
#
# The MNIST fashion data is loaded from https://github.com/zalandoresearch/fashion-mnist. The data set consists of a training set of 60,000 images and a test set of 10,000 images. Each image is a 28x28 grayscale image, associated with a label from 10 classes. 
# * Number of samples: 70,000 (60,000 train + 10,000 test)
# * Number of features: 784 (28x28)
# * Number of groups: 10
# * Pixle value range: 
# * Number of samples in groups: 
#
# | Label | Description |
# | ----- | ----------- |
# | 0 | T-shirt/top |
# | 1 | Trouser |
# | 2 | Pullover |
# | 3 | Dress |
# | 4 | Coat |
# | 5 | Sandal |
# | 6 | Shirt |
# | 7 | Sneaker |
# | 8 | Bag |
# | 9 | Ankle boot |

# +
# Load fashion data
# -----------------
fashion_path = os.path.join(PRJDIR,'Data','Fashion_Data')

fash_test_img  = np.load(fashion_path+'/test_images.npy')
fash_test_lab  = np.load(fashion_path+'/test_labels.npy')
fash_train_img = np.load(fashion_path+'/train_images.npy')
fash_train_lab = np.load(fashion_path+'/train_labels.npy')

fash_img = np.concatenate((fash_test_img, fash_train_img), axis=0)

fash_lab_df = pd.DataFrame(np.concatenate((fash_test_lab, fash_train_lab), axis=0))
