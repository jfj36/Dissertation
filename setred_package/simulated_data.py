# Create simulated data

# Common libraries
import numpy as np 
import pandas as pd
import os 

import matplotlib.colors as mcolors
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
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
    X_ori, y_ori =  make_blobs(
                            n_samples = n, 
                            n_features = p,
                            centers = kcenters,
                            cluster_std =std,
                            random_state = 42,
                            center_box=(-10.0, 10.0)
                            )
    y_ori = y_ori % K

    X,X_test,y,y_test = train_test_split(X_ori, y_ori, test_size=0.2, random_state=42, stratify=y_ori)

    X, y, X_unlabel, y_unlabel = artificial_ssl_dataset(X, y, label_rate=label_rate, random_state=42)

    # Standardize the explanatory variables (features)
    scaler = StandardScaler()

    # Fit scaler on the full original dataset (or on labeled only if preferred)

    # Apply standardization
    X_ori = scaler.fit_transform(X_ori)
    X = scaler.transform(X)
    X_unlabel = scaler.transform(X_unlabel)
    X_test = scaler.transform(X_test)

    if export_path is not None:
        np.save(os.path.join(export_path, 'X_ori.npy'), X_ori)
        np.save(os.path.join(export_path, 'y_ori.npy'), y_ori)
        np.save(os.path.join(export_path, 'X.npy'), X)
        np.save(os.path.join(export_path, 'y.npy'), y)
        np.save(os.path.join(export_path, 'X_unlabel.npy'), X_unlabel)
        np.save(os.path.join(export_path, 'y_unlabel.npy'), y_unlabel)
        np.save(os.path.join(export_path, 'X_test.npy'), X_test)
        np.save(os.path.join(export_path, 'y_test.npy'), y_test)
        
        # Concatenate each matrix X with the corresponding y and export data frames
        df_X_ori = pd.DataFrame(X_ori)
        df_y_ori = pd.DataFrame(y_ori, columns=['target'])
        df_X = pd.DataFrame(X)
        df_y = pd.DataFrame(y, columns=['target'])
        df_X_unlabel = pd.DataFrame(X_unlabel)
        df_y_unlabel = pd.DataFrame(y_unlabel, columns=['target'])
        df_X_test = pd.DataFrame(X_test)
        df_y_test = pd.DataFrame(y_test, columns=['target'])
        df_X_ori = pd.concat([df_X_ori, df_y_ori], axis=1)
        df_X = pd.concat([df_X, df_y], axis=1)
        df_X_unlabel = pd.concat([df_X_unlabel, df_y_unlabel], axis=1)
        df_X_test = pd.concat([df_X_test, df_y_test], axis=1)
        
        # Export data frames to CSV files
        df_X_ori.to_csv(os.path.join(export_path, 'df_ori.csv'), index=False)
        df_X.to_csv(os.path.join(export_path, 'df_X.csv'), index=False)
        df_X_unlabel.to_csv(os.path.join(export_path, 'df_unlabel.csv'), index=False)
        df_X_test.to_csv(os.path.join(export_path, 'df_test.csv'), index=False)

    return  X_ori, y_ori, X, y, X_unlabel, y_unlabel, X_test, y_test