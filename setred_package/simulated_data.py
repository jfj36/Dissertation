# Create simulated data

# Common libraries
import numpy as np 
import pandas as pd
import os 

import matplotlib.colors as mcolors
from sklearn.datasets import make_classification, make_blobs
from sslearn.model_selection import artificial_ssl_dataset



def create_data(n, kcenters, K, p, std = 2.5, label_rate=0.01, export_path=None):
    """
    Create a dataset with n samples, K classes and p features.

    Parameters
    ----------
    n : int number of samples
    kcenters : int number of centers (classes)
    K : int number of classes
    p : int number of features
    std : float standard deviation of the clusters
    """
    X_ori, y_ori =  make_blobs(n_samples = n, n_features = p,
                       centers = kcenters, cluster_std =std,
                       random_state = 42)
    y_ori = y_ori % K

    X, y, X_unlabel, y_unlabel = artificial_ssl_dataset(X_ori, y_ori, label_rate=0.01, random_state=42)

    if export_path is not None:
        np.save(os.path.join(export_path, 'X_ori.npy'), X_ori)
        np.save(os.path.join(export_path, 'y_ori.npy'), y_ori)
        np.save(os.path.join(export_path, 'X.npy'), X)
        np.save(os.path.join(export_path, 'y.npy'), y)
        np.save(os.path.join(export_path, 'X_unlabel.npy'), X_unlabel)
        np.save(os.path.join(export_path, 'y_unlabel.npy'), y_unlabel)

    return  X_ori, y_ori, X, y, X_unlabel, y_unlabel



