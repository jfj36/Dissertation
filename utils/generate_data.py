import os,sys
sys.path.append(os.path.abspath(".."))
from setred_package import simulated_data

print(f"Running in {os.getcwd()}")
# Parameters of the simulation
n = 17000  # Number of samples
K = 5     # Number of classes
p = 5      # Number of features
label_rate=0.01 # Label rate for semi-supervised learning
std = 4 # Standard deviation of the clusters
kcenters = 5 # Number of centers (classes) for the blobs
# Generate data
X_ori, y_ori, X, y, X_unlabel, y_unlabel, _,_ = simulated_data.create_data(
        n=n,
        kcenters=kcenters,
        K=K,
        p=p,
        std=std,
        label_rate=label_rate,
        export_path="../data/"
)