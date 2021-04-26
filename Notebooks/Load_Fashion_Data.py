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
#     display_name: Tensorflow
#     language: python
#     name: tf
# ---

# # Load Fashion Data
#
# This notebook is desighned to load the fashion-MNIST data set.
#
# Isabel Fernandez 4/26/2021

# +
# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)
# -

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

PRJDIR = '/data/SFIMJGC/PRJ_Manifold_Learning' # Project directory
os.system('if [ ! -d '+PRJDIR+'/Data/Fashion_Data ]; then mkdir '+PRJDIR+'/Data/Fashion_Data; fi') # Creates directory for Fashion Data if doesnt already exist

# Save fashion data to Data directory
np.save(PRJDIR+'/Data/Fashion_Data/test_images.npy',test_images)
np.save(PRJDIR+'/Data/Fashion_Data/train_images.npy',train_images)
np.save(PRJDIR+'/Data/Fashion_Data/test_labels.npy',test_labels)
np.save(PRJDIR+'/Data/Fashion_Data/train_labels.npy',train_labels)
